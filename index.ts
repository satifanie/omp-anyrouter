import { mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs"
import { randomBytes, randomUUID } from "node:crypto"
import { ProxyAgent, fetch as undiciFetch } from "undici"
import { homedir } from "node:os"
import { join } from "node:path"
import type { ExtensionAPI } from "@oh-my-pi/pi-coding-agent"
import {
  calculateCost,
  createAssistantMessageEventStream,
  type Api,
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Context,
  type Message,
  type Model,
  type SimpleStreamOptions,
  type StopReason,
  type TextContent,
  type ThinkingContent,
  type Tool,
  type ToolResultMessage,
  type ImageContent
} from "@oh-my-pi/pi-ai"

type Json = Record<string, any>
type StreamMode = "off" | "auto" | "force"
type FetchInit = Parameters<typeof fetch>[1]

/** 模拟的原生客户端类型。未配置时按模型 id 推断。 */
type ClientKind = "claude" | "codex"

/**
 * 模拟客户端的指纹。配置文件只允许覆盖主版本号，其余字段（OS、架构、终端、Stainless 元数据）固定，
 * 避免拼出真实客户端不会发的组合。UA 在配置加载时算好，请求路径上不再拼串。
 */
type ClientFingerprint = {
  /** 主版本号，如 2.1.226 / 0.146.0 */
  version: string
  /** 完整 user-agent */
  userAgent: string
}

/** 上游访问参数。模型级配置可覆盖全局值。 */
type Endpoint = {
  baseUrl: string
  apiKey: string
  /** 是否在 system 中注入 Claude Code 归属标识块 */
  attribution: boolean
  /** 模拟哪种原生客户端 */
  client: ClientKind
  /** client 对应的指纹 */
  fingerprint: ClientFingerprint
}

type ProviderModelConfig = {
  id: string
  name?: string
  api?: string
  /** 显式指定模拟客户端；缺省时由 resolveClientKind 按模型 id 推断 */
  client?: ClientKind
  /** 该模型专用的中转站地址，覆盖全局 baseUrl */
  baseUrl?: string
  /** 该模型专用的密钥，覆盖全局 apiKey */
  apiKey?: string
  /** 该模型是否注入归属标识块，覆盖全局 attribution */
  attribution?: boolean
  reasoning?: boolean
  input?: ("text" | "image")[]
  cost?: { input?: number; output?: number; cacheRead?: number; cacheWrite?: number }
  contextWindow?: number
  maxTokens?: number
}

type ProviderConfigFile = {
  baseUrl?: string
  apiKey?: string
  /** 所有模型的默认客户端，单个模型的 client 优先 */
  client?: ClientKind
  /** 是否注入 Claude Code 归属标识块，默认 false */
  attribution?: boolean
  /** 覆盖 Claude Code 模拟的主版本号，如 "2.1.226" */
  claudeVersion?: string
  /** 覆盖 Codex 模拟的主版本号，如 "0.146.0" */
  codexVersion?: string
  models?: ProviderModelConfig[]
}

/** 配置加载后每个模型的最终形态：凭据、客户端与指纹都已解析完毕。 */
type ResolvedModelConfig = ProviderModelConfig & Endpoint

const DEFAULT_CONFIG_PATH = join(homedir(), ".omp", "agent", "anyrouter.json")
const CONFIG_PATH = process.env.PI_ANYROUTER_CONFIG || DEFAULT_CONFIG_PATH
const PROVIDER_NAME = "anyrouter"
const LOG_PREFIX = "[anyrouter]"
// 使用自定义 API ID，避免触发内置 anthropic-messages 实现
const API_ID = "anyrouter-messages" as Api
const DEBUG_ENABLED = process.env.PI_ANYROUTER_DEBUG === "1"
const DEBUG_DIR =
  process.env.PI_ANYROUTER_DEBUG_DIR || join(process.cwd(), ".omp", "anyrouter-debug")
/** 调试目录只在首次落盘时创建；序号避免同毫秒内的两次异步写撞同名。 */
let debugDirReady = false
let debugSeq = 0
// Claude Code CLI 指纹。官方 UA 形如 claude-cli/2.1.226 (external, cli)。
const DEFAULT_CLAUDE_CODE_VERSION = "2.1.226"
// UA 里的入口标识，必须与归属标识块的 cc_entrypoint 一致
const CLAUDE_CODE_ENTRYPOINT = "cli"
// 归属标识块的 cc_version 是 <版本>.<构建号>
const CLAUDE_CODE_BUILD_TAG = "b94"
const STAINLESS_PACKAGE_VERSION = "0.94.0"
const STAINLESS_OS = "MacOS"
const STAINLESS_ARCH = "arm64"
const STAINLESS_RUNTIME = "node"
const STAINLESS_RUNTIME_VERSION = "v26.3.0"
const ANTHROPIC_BETA =
  "claude-code-20250219,context-1m-2025-08-07,interleaved-thinking-2025-05-14,thinking-token-count-2026-05-13,context-management-2025-06-27,prompt-caching-scope-2026-01-05,mid-conversation-system-2026-04-07,advisor-tool-2026-03-01,effort-2025-11-24"
const CLAUDE_CONFIG_PATH = join(homedir(), ".claude.json")

// Codex CLI 指纹。installation_id 每进程一个，与真实 codex-tui 行为一致。
const DEFAULT_CODEX_VERSION = "0.146.0"
const CODEX_ORIGINATOR = "codex-tui"
const CODEX_OS = "Mac OS 26.6"
const CODEX_ARCH = "arm64"
const CODEX_TERMINAL = "ghostty/1.3.1"
const CODEX_INSTALLATION_ID = randomUUID()

function buildClaudeUserAgent(version: string) {
  return `claude-cli/${version} (external, ${CLAUDE_CODE_ENTRYPOINT})`
}

function buildCodexUserAgent(version: string) {
  return `${CODEX_ORIGINATOR}/${version} (${CODEX_OS}; ${CODEX_ARCH}) ${CODEX_TERMINAL} (${CODEX_ORIGINATOR}; ${version})`
}

/**
 * 会话 id 每进程一个（一个 omp 进程即一次会话），不能每请求重新生成：
 * Codex 用它做 prompt_cache_key，逐请求变化会让上游 prompt 缓存全部落空；
 * Claude 侧它同时是 metadata.user_id 与 x-claude-code-session-id，逐请求变化也不符合原生客户端行为。
 */
const SESSION_ID = randomUUID()

/** reader.read() 结束时 value 为 undefined，复用同一个空块避免每次分配。 */
const EMPTY_CHUNK = new Uint8Array()

// ============================================================================
// 日志：复用 omp 的 file logger，输出到 ~/.omp/logs/omp.<date>.<pid>.log
// ============================================================================

type OmpLogger = {
  error(message: string, context?: Record<string, unknown>): void
  warn(message: string, context?: Record<string, unknown>): void
  info(message: string, context?: Record<string, unknown>): void
  debug(message: string, context?: Record<string, unknown>): void
}

/** 由高到低排列，索引即严重程度阈值。 */
const LOG_LEVELS = ["error", "warn", "info", "debug"] as const
type LogLevel = (typeof LOG_LEVELS)[number]

/** 默认只输出异常（error + warn），正常流水日志需显式开启。 */
const DEFAULT_LOG_THRESHOLD = LOG_LEVELS.indexOf("warn")

/**
 * 解析 PI_ANYROUTER_LOG：
 * 缺省或非法值只输出异常（error/warn）；on/1/true 打开全量；
 * off/0/false/none/silent 完全静音；error/warn/info/debug 直接作为最低输出级别。
 */
function resolveLogThreshold() {
  const raw = process.env.PI_ANYROUTER_LOG?.trim().toLowerCase()
  if (!raw) return DEFAULT_LOG_THRESHOLD
  if (["on", "1", "true"].includes(raw)) return LOG_LEVELS.length - 1
  if (["off", "0", "false", "none", "silent"].includes(raw)) return -1
  const index = (LOG_LEVELS as readonly string[]).indexOf(raw)
  return index === -1 ? DEFAULT_LOG_THRESHOLD : index
}

const LOG_THRESHOLD = resolveLogThreshold()

/** 扩展加载阶段注入；未注入前回退到 stderr，保证日志不丢。 */
let ompLogger: OmpLogger | undefined

function setLogger(logger: OmpLogger) {
  ompLogger = logger
}

function logAt(level: LogLevel, message: string, context?: Record<string, unknown>) {
  if (LOG_LEVELS.indexOf(level) > LOG_THRESHOLD) return
  if (ompLogger) {
    ompLogger[level](`${LOG_PREFIX} ${message}`, context)
    return
  }
  // 尚未注入 logger（或运行在 probe 脚本里）时的兜底输出
  const detail = context ? ` ${JSON.stringify(context)}` : ""
  console.error(`${LOG_PREFIX} ${level}: ${message}${detail}`)
}

const log = {
  error: (message: string, context?: Record<string, unknown>) => logAt("error", message, context),
  warn: (message: string, context?: Record<string, unknown>) => logAt("warn", message, context),
  info: (message: string, context?: Record<string, unknown>) => logAt("info", message, context),
  debug: (message: string, context?: Record<string, unknown>) => logAt("debug", message, context)
}

/** 统一提取错误信息，避免各处重复 instanceof 判断。 */
function errorText(error: unknown) {
  return error instanceof Error ? error.message : String(error)
}

/**
 * ~/.claude.json 存了全部项目历史，常见几 MB 起步，整份 JSON.parse 纯属浪费。
 * 先正则扫 64 位十六进制的 userID：命中唯一值直接用，只有多处取值冲突时才回退解析取顶层字段。
 */
function loadClaudeDeviceId() {
  const configured = process.env.PI_ANYROUTER_DEVICE_ID?.trim()
  if (configured && /^[0-9a-f]{64}$/i.test(configured)) return configured

  try {
    const content = readFileSync(CLAUDE_CONFIG_PATH, "utf8")
    const pattern = /"userID"\s*:\s*"([0-9a-f]{64})"/gi
    let found: string | undefined
    let ambiguous = false
    for (let match = pattern.exec(content); match; match = pattern.exec(content)) {
      if (found && found !== match[1]) {
        ambiguous = true
        break
      }
      found = match[1]
    }
    if (found && !ambiguous) return found
    if (found) {
      const parsed = JSON.parse(content)
      if (typeof parsed.userID === "string" && /^[0-9a-f]{64}$/i.test(parsed.userID))
        return parsed.userID
    }
  } catch {
    // Claude Code may not be installed on the machine running pi.
  }

  return randomBytes(32).toString("hex")
}

/** 惰性解析：codex-only 的配置永远不会去读 ~/.claude.json。 */
let claudeDeviceId: string | undefined

const NAME_MAP: Record<string, string> = {
  read: "Read",
  write: "Write",
  edit: "Edit",
  bash: "Bash",
  grep: "Grep",
  glob: "Glob",
  ls: "LS",
  todowrite: "TodoWrite",
  webfetch: "WebFetch",
  websearch: "WebSearch",
  google_search: "Google_Search"
}

/** NAME_MAP 的反向表，避免每个 tool_use 块都线性扫描一遍 entries。 */
const REVERSE_NAME_MAP: Record<string, string> = Object.fromEntries(
  Object.entries(NAME_MAP).map(([from, to]) => [to.toLowerCase(), from])
)

function toClaudeCodeName(name?: string | null) {
  if (!name || typeof name !== "string") return name
  return NAME_MAP[name.toLowerCase()] ?? name.charAt(0).toUpperCase() + name.slice(1)
}

/** 去掉 omp 的 `[1m]` 等上下文后缀，上游只认裸模型名。 */
function stripModelSuffix(modelId: string) {
  return modelId.replace(/\[[^\]]*\]$/, "")
}

function toClaudeCodeRequestModel(modelId: string) {
  return stripModelSuffix(modelId)
}

function toCodexRequestModel(modelId: string) {
  return stripModelSuffix(modelId)
}

function fromClaudeCodeName(name?: string | null): string {
  if (!name || typeof name !== "string") return ""
  return REVERSE_NAME_MAP[name.toLowerCase()] ?? name.charAt(0).toLowerCase() + name.slice(1)
}

/**
 * 只把未配对的代理码元替换成 U+FFFD。
 * 不能用 /[\uD800-\uDFFF]/g —— 它会逐 code unit 匹配，把合法代理对（emoji、星平面字符）也拆成两个替换符。
 */
function sanitizeText(text: string) {
  if (typeof text !== "string") return ""
  return text.toWellFormed()
}

/** omp 的 systemPrompt 是分段数组，上游只接受单个字符串。 */
function joinSystemPrompt(systemPrompt: string[] | undefined, fallback: string) {
  const joined = (systemPrompt ?? []).filter(Boolean).join("\n\n").trim()
  return joined || fallback
}

/** 解析布尔型环境变量；未设置或无法识别时返回 undefined，交由下一级默认值决定。 */
function parseBooleanEnv(value?: string) {
  const raw = value?.trim().toLowerCase()
  if (!raw) return undefined
  if (["1", "true", "on", "yes"].includes(raw)) return true
  if (["0", "false", "off", "no"].includes(raw)) return false
  return undefined
}

/** 只接受纯数字版本号（如 2.1.226）。非法值告警并回退到内置默认，避免拼出上游不认的 UA。 */
const VERSION_PATTERN = /^\d+(?:\.\d+){0,3}$/

function normalizeVersion(value: string | undefined, fallback: string, field: string) {
  const raw = value?.trim()
  if (!raw) return fallback
  if (VERSION_PATTERN.test(raw)) return raw
  log.warn(`❓ invalid ${field} "${raw}", falling back to ${fallback}`, {
    value: raw,
    expected: "digits separated by dots, e.g. 2.1.226"
  })
  return fallback
}

/**
 * 指纹只暴露主版本号一个旋钮：配置文件（或同名环境变量）给什么版本就发什么版本，
 * OS / 架构 / 终端 / Stainless 元数据保持内置值，避免拼出真实客户端不存在的组合。
 */
function resolveFingerprints(parsed: ProviderConfigFile): Record<ClientKind, ClientFingerprint> {
  const claudeVersion = normalizeVersion(
    process.env.PI_ANYROUTER_CLAUDE_VERSION || parsed.claudeVersion,
    DEFAULT_CLAUDE_CODE_VERSION,
    "claudeVersion"
  )
  const codexVersion = normalizeVersion(
    process.env.PI_ANYROUTER_CODEX_VERSION || parsed.codexVersion,
    DEFAULT_CODEX_VERSION,
    "codexVersion"
  )
  return {
    claude: { version: claudeVersion, userAgent: buildClaudeUserAgent(claudeVersion) },
    codex: { version: codexVersion, userAgent: buildCodexUserAgent(codexVersion) }
  }
}

function loadSourceProvider() {
  let content = ""
  try {
    content = readFileSync(CONFIG_PATH, "utf8")
  } catch {
    throw new Error(
      `Config file not found: ${CONFIG_PATH}. Create it from anyrouter.example.json or set PI_ANYROUTER_CONFIG.`
    )
  }

  let parsed: ProviderConfigFile
  try {
    parsed = JSON.parse(content) as ProviderConfigFile
  } catch (error) {
    throw new Error(
      `Invalid JSON in ${CONFIG_PATH}: ${error instanceof Error ? error.message : String(error)}`
    )
  }

  // 环境变量优先级最高，覆盖配置文件中的全局值（但不覆盖模型级显式配置）
  const baseUrl = process.env.PI_ANYROUTER_BASE_URL || parsed.baseUrl || ""
  const apiKey = process.env.PI_ANYROUTER_API_KEY || parsed.apiKey || ""
  const defaultClient = normalizeClientKind(process.env.PI_ANYROUTER_CLIENT || parsed.client)
  // 归属标识块默认关闭：中转站不会像 api.anthropic.com 那样剥离它，留着会污染 prompt 和缓存键
  const defaultAttribution =
    parseBooleanEnv(process.env.PI_ANYROUTER_ATTRIBUTION) ?? parsed.attribution ?? false
  const fingerprints = resolveFingerprints(parsed)
  const rawModels = parsed.models || []

  if (!rawModels.length)
    throw new Error(`No models configured in ${CONFIG_PATH}. Add at least one model entry.`)

  const models: ResolvedModelConfig[] = rawModels.map((model) => {
    const modelBaseUrl = model.baseUrl || baseUrl
    const modelApiKey = model.apiKey || apiKey
    if (!modelBaseUrl)
      throw new Error(
        `Missing baseUrl for model "${model.id}" in ${CONFIG_PATH}. Set it globally, per model, or via PI_ANYROUTER_BASE_URL.`
      )
    if (!modelApiKey)
      throw new Error(
        `Missing apiKey for model "${model.id}" in ${CONFIG_PATH}. Set it globally, per model, or via PI_ANYROUTER_API_KEY.`
      )
    const client = resolveClientKind(model.id, normalizeClientKind(model.client), defaultClient)
    return {
      ...model,
      baseUrl: modelBaseUrl,
      apiKey: modelApiKey,
      attribution: model.attribution ?? defaultAttribution,
      client,
      fingerprint: fingerprints[client]
    }
  })

  return {
    baseUrl,
    apiKey,
    models,
    modelIndex: new Map(models.map((model) => [model.id, model])),
    defaultClient,
    defaultAttribution,
    fingerprints
  }
}

/**
 * 配置最短复查间隔。窗口内直接吃缓存，请求路径上连 stat 都不做；
 * 设为 0 恢复"每次请求都 stat"的旧行为。
 */
function resolveConfigTtlMs() {
  const raw = process.env.PI_ANYROUTER_CONFIG_TTL_MS?.trim()
  if (!raw) return 5000
  const parsed = Number(raw)
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : 5000
}

const CONFIG_TTL_MS = resolveConfigTtlMs()

let cachedSource: { source: SourceProvider; mtimeMs: number; checkedAt: number } | undefined

/**
 * 两级缓存：TTL 窗口内零系统调用；窗口过期后只 stat 一次，mtime 未变仍复用已解析的模型表，
 * 只有文件真的改了才重新读盘 + 全量重建。文件临时不可读时沿用已注册的配置，避免打断进行中的会话。
 */
function getSourceProvider(): SourceProvider {
  const now = Date.now()
  if (cachedSource && now - cachedSource.checkedAt < CONFIG_TTL_MS) return cachedSource.source

  let mtimeMs: number
  try {
    mtimeMs = statSync(CONFIG_PATH).mtimeMs
  } catch {
    if (!cachedSource) return loadSourceProvider()
    cachedSource.checkedAt = now
    return cachedSource.source
  }

  if (cachedSource?.mtimeMs === mtimeMs) {
    cachedSource.checkedAt = now
    return cachedSource.source
  }

  const source = loadSourceProvider()
  cachedSource = { source, mtimeMs, checkedAt: now }
  return source
}

/** 取模型的实际访问端点；未在配置中登记的模型回退到全局凭据。 */
function resolveEndpoint(source: SourceProvider, modelId: string): Endpoint {
  const configured = source.modelIndex.get(modelId)
  if (configured) return configured
  if (!source.baseUrl || !source.apiKey) {
    throw new Error(
      `Model "${modelId}" is not configured in ${CONFIG_PATH} and no global baseUrl/apiKey is available.`
    )
  }
  const client = resolveClientKind(modelId, undefined, source.defaultClient)
  return {
    baseUrl: source.baseUrl,
    apiKey: source.apiKey,
    attribution: source.defaultAttribution,
    client,
    fingerprint: source.fingerprints[client]
  }
}

/** 校验并归一化 client 值，非法值降级为 undefined（改走自动推断）并告警。 */
function normalizeClientKind(value?: string): ClientKind | undefined {
  if (!value) return undefined
  const normalized = value.trim().toLowerCase()
  if (normalized === "claude" || normalized === "codex") return normalized
  log.warn(`❓ unknown client "${value}", falling back to model-id detection`, {
    value,
    expected: ["claude", "codex"]
  })
  return undefined
}

/**
 * 决定某个模型走哪种客户端模拟：
 * 显式 client > 配置文件默认 client > 按模型 id 推断（gpt/codex/o系列 → codex，其余 → claude）。
 */
function resolveClientKind(
  modelId: string,
  configured?: ClientKind,
  fallback?: ClientKind
): ClientKind {
  if (configured) return configured
  if (fallback) return fallback
  if (/(?:^|[-_.])(gpt|codex)(?:[-_.]|$)/i.test(modelId) || /^o\d(?:[-_.]|$)/i.test(modelId))
    return "codex"
  return "claude"
}

function convertContentBlocks(content: (TextContent | ImageContent)[]) {
  const hasImages = content.some((c) => c.type === "image")
  if (!hasImages) return sanitizeText(content.map((c) => (c as TextContent).text).join("\n"))

  const blocks = content.map((block) => {
    if (block.type === "text") return { type: "text", text: sanitizeText(block.text) }
    return {
      type: "image",
      source: { type: "base64", media_type: block.mimeType, data: block.data }
    }
  })
  if (!blocks.some((b) => b.type === "text"))
    blocks.unshift({ type: "text", text: "(see attached image)" })
  return blocks
}

function convertMessages(messages: Message[]) {
  const params: any[] = []
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i]
    if (msg.role === "user") {
      if (typeof msg.content === "string") {
        const text = sanitizeText(msg.content)
        if (text.trim()) params.push({ role: "user", content: [{ type: "text", text }] })
      } else {
        const blocks = msg.content.map((item) =>
          item.type === "text"
            ? { type: "text", text: sanitizeText(item.text) }
            : {
                type: "image",
                source: { type: "base64", media_type: item.mimeType, data: item.data }
              }
        )
        if (blocks.length > 0) params.push({ role: "user", content: blocks })
      }
      continue
    }

    if (msg.role === "assistant") {
      const blocks: any[] = []
      for (const block of msg.content) {
        if (block.type === "text" && block.text.trim())
          blocks.push({ type: "text", text: sanitizeText(block.text) })
        else if (block.type === "thinking" && block.thinking.trim()) {
          if ((block as ThinkingContent).thinkingSignature) {
            blocks.push({
              type: "thinking",
              thinking: sanitizeText(block.thinking),
              signature: (block as ThinkingContent).thinkingSignature
            })
          } else {
            blocks.push({ type: "text", text: sanitizeText(block.thinking) })
          }
        } else if (block.type === "toolCall") {
          blocks.push({
            type: "tool_use",
            id: block.id,
            name: toClaudeCodeName(block.name),
            input: block.arguments
          })
        }
      }
      if (blocks.length > 0) params.push({ role: "assistant", content: blocks })
      continue
    }

    if (msg.role === "toolResult") {
      const toolResults: any[] = []
      const pushToolResult = (toolMsg: ToolResultMessage) => {
        toolResults.push({
          type: "tool_result",
          tool_use_id: toolMsg.toolCallId,
          content: convertContentBlocks(toolMsg.content),
          is_error: toolMsg.isError
        })
      }
      pushToolResult(msg as ToolResultMessage)
      let j = i + 1
      while (j < messages.length && messages[j].role === "toolResult") {
        pushToolResult(messages[j] as ToolResultMessage)
        j++
      }
      i = j - 1
      params.push({ role: "user", content: toolResults })
    }
  }

  if (params.length > 0) {
    const last = params[params.length - 1]
    if (last.role === "user" && Array.isArray(last.content) && last.content.length > 0) {
      last.content[last.content.length - 1].cache_control = { type: "ephemeral" }
    }
  }
  return params
}

function convertTools(tools: Tool[]) {
  return tools.map((tool) => {
    // 整份 schema 透传：$defs / $ref / additionalProperties / items 丢失会让嵌套参数在上游失效
    const schema = (tool.parameters as Json | undefined) ?? {}
    return {
      name: toClaudeCodeName(tool.name),
      description: tool.description,
      input_schema: {
        ...schema,
        type: "object",
        properties: schema.properties ?? {},
        required: schema.required ?? []
      }
    }
  })
}

function mapReasoningEffort(level?: SimpleStreamOptions["reasoning"]) {
  switch (level) {
    case "minimal":
    case "low":
      return "low"
    case "medium":
      return "medium"
    case "high":
      return "high"
    case "xhigh":
      return "xhigh"
    default:
      return "medium"
  }
}

function mapStopReason(reason: string): StopReason {
  switch (reason) {
    case "end_turn":
    case "pause_turn":
    case "stop_sequence":
      return "stop"
    case "max_tokens":
      return "length"
    case "tool_use":
      return "toolUse"
    default:
      return "error"
  }
}

function getClaudeCodeHeaders(endpoint: Endpoint, retryCount = 0, sessionId: string) {
  return {
    "content-type": "application/json",
    accept: "application/json",
    authorization: `Bearer ${endpoint.apiKey}`,
    "x-api-key": endpoint.apiKey,
    "anthropic-version": "2023-06-01",
    "anthropic-dangerous-direct-browser-access": "true",
    "anthropic-beta": ANTHROPIC_BETA,
    "user-agent": endpoint.fingerprint.userAgent,
    "x-app": "cli",
    "x-claude-code-session-id": sessionId,
    "x-stainless-retry-count": String(retryCount),
    "x-stainless-timeout": "600",
    "x-stainless-lang": "js",
    "x-stainless-package-version": STAINLESS_PACKAGE_VERSION,
    "x-stainless-os": STAINLESS_OS,
    "x-stainless-arch": STAINLESS_ARCH,
    "x-stainless-runtime": STAINLESS_RUNTIME,
    "x-stainless-runtime-version": STAINLESS_RUNTIME_VERSION
  }
}

function createClaudeCodeMetadata(sessionId: string) {
  return {
    user_id: JSON.stringify({
      device_id: (claudeDeviceId ??= loadClaudeDeviceId()),
      account_uuid: "",
      session_id: sessionId
    })
  }
}

/**
 * 归属标识块必须是 system 数组的第一个独立条目，api.anthropic.com 会按位置剥离它。
 * 中转站不做这个剥离，块会进入 prompt 与缓存键，因此默认不发送。
 */
function createClaudeCodeSystem(systemPrompt: string, endpoint: Endpoint) {
  return [
    ...(endpoint.attribution
      ? [
          {
            type: "text",
            text: `x-anthropic-billing-header: cc_version=${endpoint.fingerprint.version}.${CLAUDE_CODE_BUILD_TAG}; cc_entrypoint=${CLAUDE_CODE_ENTRYPOINT};`
          }
        ]
      : []),
    {
      type: "text",
      text: "You are a Claude agent, built on Anthropic's Claude Agent SDK.",
      cache_control: { type: "ephemeral" }
    },
    { type: "text", text: sanitizeText(systemPrompt), cache_control: { type: "ephemeral" } }
  ]
}

function redactHeaders(headers: Record<string, string>) {
  const redacted = { ...headers }
  if (redacted.authorization) redacted.authorization = "Bearer ***"
  if (redacted["x-api-key"]) redacted["x-api-key"] = "***"
  return redacted
}

const PROXY_AGENTS = new Map<string, ProxyAgent>()

function hostMatchesNoProxy(hostname: string, pattern: string) {
  const item = pattern.trim().toLowerCase()
  if (!item) return false
  if (item === "*") return true
  const host = hostname.toLowerCase()
  if (item.startsWith(".")) return host === item.slice(1) || host.endsWith(item)
  return host === item || host.endsWith(`.${item}`)
}

function getProxyUrl(url: string) {
  const parsed = new URL(url)
  const noProxy = process.env.NO_PROXY || process.env.no_proxy || ""
  if (noProxy.split(",").some((item) => hostMatchesNoProxy(parsed.hostname, item))) return undefined
  if (parsed.protocol === "https:")
    return (
      process.env.HTTPS_PROXY ||
      process.env.https_proxy ||
      process.env.HTTP_PROXY ||
      process.env.http_proxy
    )
  return process.env.HTTP_PROXY || process.env.http_proxy
}

function getProxyAgent(proxyUrl: string) {
  let agent = PROXY_AGENTS.get(proxyUrl)
  if (!agent) {
    agent = new ProxyAgent(proxyUrl)
    PROXY_AGENTS.set(proxyUrl, agent)
  }
  return agent
}

function fetchWithProxy(url: string, init: FetchInit) {
  const proxyUrl = getProxyUrl(url)
  if (!proxyUrl) return fetch(url, init)
  return undiciFetch(url, {
    ...init,
    dispatcher: getProxyAgent(proxyUrl)
  } as any) as unknown as Promise<Response>
}

/** 快照体可传 thunk：debug 关闭时不会被求值，避免在热路径上白构造 headers/body 对象。 */
type DebugPayload = Json | (() => Json)

/**
 * 落盘详细请求/响应快照（需 PI_ANYROUTER_DEBUG=1）。
 * 与之独立：error 类别始终写入 omp 日志，便于不开 debug 时也能排查。
 */
function writeDebugFile(
  kind: "request" | "response" | "error",
  modelId: string,
  requestId: string | undefined,
  payload: DebugPayload
) {
  if (kind !== "error") {
    writeDebugFileRaw(kind, modelId, requestId, payload)
    return
  }
  const resolved = typeof payload === "function" ? payload() : payload
  log.error(`💥 upstream request failed`, {
    model: modelId,
    requestId,
    status: resolved.status,
    transport: resolved.transport,
    retryAttempt: resolved.retryAttempt,
    detail: summarizeForLog(resolved)
  })
  writeDebugFileRaw(kind, modelId, requestId, resolved)
}

/** 仅落盘，不写 omp 日志。调用方已自行记录时使用。 */
function writeDebugFileRaw(
  kind: "request" | "response" | "error",
  modelId: string,
  requestId: string | undefined,
  payload: DebugPayload
) {
  if (!DEBUG_ENABLED) return
  try {
    if (!debugDirReady) {
      mkdirSync(DEBUG_DIR, { recursive: true })
      debugDirReady = true
    }
    const safeModel = modelId.replace(/[^a-zA-Z0-9._-]+/g, "_")
    const safeRequestId = (requestId || "no-request-id").replace(/[^a-zA-Z0-9._-]+/g, "_")
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-")
    const seq = (debugSeq++).toString(36).padStart(3, "0")
    const path = join(DEBUG_DIR, `${timestamp}-${seq}-${safeModel}-${safeRequestId}-${kind}.json`)
    const resolved = typeof payload === "function" ? payload() : payload
    // 同步写：异步写实测每份 5.7 MB 快照只省 1.3 ms，却会在进程被直接结束时丢掉最后一份快照，
    // 而那恰好是排查现场最需要的一份。
    writeFileSync(path, JSON.stringify(resolved, null, 2), "utf8")
  } catch (error) {
    // 调试落盘失败不应影响正常请求
    log.warn(`📄 failed to write debug file`, { dir: DEBUG_DIR, error: errorText(error) })
  }
}

/** 从错误 payload 中摘出一行可读信息，避免把整个响应体灌进日志。 */
function summarizeForLog(payload: Json) {
  const message = payload?.body?.error?.message ?? payload?.body?.message ?? payload?.errorMessage
  if (typeof message === "string") return message.slice(0, 500)
  const raw = typeof payload?.raw === "string" ? payload.raw : undefined
  if (raw) return raw.slice(0, 500)
  return undefined
}

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

const RETRYABLE_STATUS = new Set([408, 409, 429, 500, 502, 503, 504, 520, 522, 524])

function isRetryableStatus(status: number) {
  return RETRYABLE_STATUS.has(status)
}

function parseRetryAfterMs(value: string | null) {
  if (!value) return undefined
  const seconds = Number(value)
  if (Number.isFinite(seconds) && seconds >= 0) return seconds * 1000
  const at = Date.parse(value)
  if (Number.isFinite(at)) {
    const delta = at - Date.now()
    return delta > 0 ? delta : 0
  }
  return undefined
}

function getMaxRetries() {
  return Math.max(0, Number(process.env.PI_ANYROUTER_MAX_RETRIES || "10") || 0)
}

function getRetryDelayMs(attempt: number, retryAfterMs?: number) {
  if (typeof retryAfterMs === "number") return Math.max(0, Math.min(retryAfterMs, 30_000))
  const base = Math.min(1000 * 2 ** attempt, 15_000)
  const jitter = Math.floor(Math.random() * 250)
  return base + jitter
}

function getStreamMode(): StreamMode {
  const value = String(process.env.PI_ANYROUTER_STREAM_MODE || "auto")
    .trim()
    .toLowerCase()
  if (["1", "true", "on", "auto"].includes(value)) return "auto"
  if (["force", "only"].includes(value)) return "force"
  return "off"
}

function createEmptyUsage() {
  return {
    input: 0,
    output: 0,
    cacheRead: 0,
    cacheWrite: 0,
    totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 }
  }
}

function tryParseJson(text: string) {
  try {
    return text ? JSON.parse(text) : {}
  } catch {
    return undefined
  }
}

function extractRequestId(parsed: any, headers: Headers) {
  return (
    parsed?.error?.message?.match(/request id:\s*([^\)]+)/i)?.[1] ||
    headers.get("x-oneapi-request-id") ||
    undefined
  )
}

function updateUsageFromAnthropic(output: AssistantMessage, usage: any, model: Model<Api>) {
  if (usage?.input_tokens != null) output.usage.input = usage.input_tokens
  if (usage?.output_tokens != null) output.usage.output = usage.output_tokens
  if (usage?.cache_read_input_tokens != null) output.usage.cacheRead = usage.cache_read_input_tokens
  if (usage?.cache_creation_input_tokens != null)
    output.usage.cacheWrite = usage.cache_creation_input_tokens
  output.usage.totalTokens =
    output.usage.input + output.usage.output + output.usage.cacheRead + output.usage.cacheWrite
  calculateCost(model, output.usage)
}

function resetOutputState(output: AssistantMessage) {
  output.content = []
  output.usage = createEmptyUsage()
  output.stopReason = "stop"
  output.errorMessage = undefined
  output.responseId = undefined
}

// ============================================================================
// Codex 客户端模拟（OpenAI Responses API）
// ============================================================================

/** 是否为 codex 专用模型（如 gpt-5.1-codex），区别于通用 GPT 模型。 */
function isCodexNativeModel(modelId: string) {
  return /codex/i.test(modelId)
}

function getCodexResponsesUrl(baseUrl: string) {
  const normalized = baseUrl.replace(/\/+$/, "")
  if (normalized.endsWith("/responses")) return normalized
  if (normalized.endsWith("/v1")) return `${normalized}/responses`
  return `${normalized}/v1/responses`
}

function convertCodexMessages(context: Context) {
  const input: any[] = []
  const systemPrompt = joinSystemPrompt(context.systemPrompt, "")
  if (systemPrompt) {
    input.push({
      type: "message",
      role: "developer",
      content: [{ type: "input_text", text: sanitizeText(systemPrompt) }]
    })
  }

  for (const msg of context.messages) {
    if (msg.role === "user") {
      if (typeof msg.content === "string") {
        if (msg.content.trim()) {
          input.push({
            type: "message",
            role: "user",
            content: [{ type: "input_text", text: sanitizeText(msg.content) }]
          })
        }
      } else {
        const content = msg.content.map((item) =>
          item.type === "text"
            ? { type: "input_text", text: sanitizeText(item.text) }
            : {
                type: "input_image",
                detail: "auto",
                image_url: `data:${item.mimeType};base64,${item.data}`
              }
        )
        if (content.length) input.push({ type: "message", role: "user", content })
      }
      continue
    }

    if (msg.role === "assistant") {
      for (const block of msg.content) {
        if (block.type === "thinking" && block.thinkingSignature) {
          // 回放上一轮的加密 reasoning item，Codex 依赖它保持思考连续性
          const reasoning = tryParseJson(block.thinkingSignature)
          if (reasoning) input.push(reasoning)
        } else if (block.type === "text" && block.text.trim()) {
          input.push({
            type: "message",
            role: "assistant",
            status: "completed",
            content: [{ type: "output_text", text: sanitizeText(block.text), annotations: [] }]
          })
        } else if (block.type === "toolCall") {
          const [callId, itemId] = block.id.split("|")
          input.push({
            type: "function_call",
            ...(itemId ? { id: itemId } : {}),
            call_id: callId,
            name: block.name,
            arguments: JSON.stringify(block.arguments)
          })
        }
      }
      continue
    }

    if (msg.role === "toolResult") {
      const toolMsg = msg as ToolResultMessage
      const text = toolMsg.content
        .filter((item) => item.type === "text")
        .map((item) => (item as TextContent).text)
        .join("\n")
      const images = toolMsg.content.filter((item) => item.type === "image") as ImageContent[]
      const output = images.length
        ? [
            ...(text ? [{ type: "input_text", text: sanitizeText(text) }] : []),
            ...images.map((image) => ({
              type: "input_image",
              detail: "auto",
              image_url: `data:${image.mimeType};base64,${image.data}`
            }))
          ]
        : sanitizeText(text || "(no tool output)")
      input.push({
        type: "function_call_output",
        call_id: toolMsg.toolCallId.split("|")[0],
        output
      })
    }
  }
  return input
}

function convertCodexTools(tools: Tool[]) {
  return tools.map((tool) => ({
    type: "function",
    name: tool.name,
    description: tool.description,
    parameters: tool.parameters,
    strict: false
  }))
}

function createCodexMetadata(sessionId: string, turnId: string) {
  const windowId = `${sessionId}:0`
  const turnMetadata = JSON.stringify({
    installation_id: CODEX_INSTALLATION_ID,
    session_id: sessionId,
    thread_id: sessionId,
    turn_id: turnId,
    window_id: windowId,
    request_kind: "turn",
    thread_source: "user",
    turn_started_at_unix_ms: Date.now()
  })
  return {
    windowId,
    turnMetadata,
    clientMetadata: {
      session_id: sessionId,
      thread_id: sessionId,
      turn_id: turnId,
      "x-codex-installation-id": CODEX_INSTALLATION_ID,
      "x-codex-window-id": windowId,
      "x-codex-turn-metadata": turnMetadata
    }
  }
}

type CodexMetadata = ReturnType<typeof createCodexMetadata>

function createCodexHeaders(
  endpoint: Endpoint,
  sessionId: string,
  metadata: CodexMetadata,
  modelId: string
) {
  return {
    authorization: `Bearer ${endpoint.apiKey}`,
    accept: "text/event-stream",
    "content-type": "application/json",
    originator: CODEX_ORIGINATOR,
    "user-agent": endpoint.fingerprint.userAgent,
    // responses-lite 只被 *-codex 模型接受，通用 GPT 模型带上会被上游 400 拒绝
    ...(isCodexNativeModel(modelId) ? { "x-openai-internal-codex-responses-lite": "true" } : {}),
    "x-codex-beta-features": "remote_compaction_v2",
    "x-codex-window-id": metadata.windowId,
    "x-codex-turn-metadata": metadata.turnMetadata,
    "x-client-request-id": sessionId,
    "session-id": sessionId,
    "thread-id": sessionId
  }
}

function buildCodexRequestBody(
  model: Model<Api>,
  context: Context,
  options: SimpleStreamOptions | undefined,
  sessionId: string,
  metadata: CodexMetadata
) {
  const body: Json = {
    model: toCodexRequestModel(model.id),
    input: convertCodexMessages(context),
    tool_choice: "auto",
    parallel_tool_calls: false,
    reasoning: {
      effort: mapReasoningEffort(options?.reasoning),
      context: "all_turns"
    },
    store: false,
    stream: true,
    text: { verbosity: "low" },
    max_output_tokens: options?.maxTokens || model.maxTokens,
    include: ["reasoning.encrypted_content"],
    prompt_cache_key: sessionId,
    client_metadata: metadata.clientMetadata
  }
  if (context.tools?.length) body.tools = convertCodexTools(context.tools)
  return body
}

function applyCodexUsage(output: AssistantMessage, response: any, model: Model<Api>) {
  const usage = response?.usage
  if (!usage) return
  const cached = usage.input_tokens_details?.cached_tokens || 0
  const cacheWrite = usage.input_tokens_details?.cache_write_tokens || 0
  output.usage.input = Math.max(0, (usage.input_tokens || 0) - cached - cacheWrite)
  output.usage.output = usage.output_tokens || 0
  output.usage.cacheRead = cached
  output.usage.cacheWrite = cacheWrite
  output.usage.totalTokens =
    usage.total_tokens || output.usage.input + output.usage.output + cached + cacheWrite
  calculateCost(model, output.usage)
}

type CodexSlot = { type: "text" | "thinking" | "toolCall"; block: any; contentIndex: number }

function applyCodexSsePayload(
  payload: any,
  output: AssistantMessage,
  stream: AssistantMessageEventStream,
  model: Model<Api>,
  slots: Map<number, CodexSlot>
) {
  const type = payload?.type
  if (!type || type === "response.in_progress" || type === "response.metadata") return
  if (type === "error") throw new Error(payload.message || JSON.stringify(payload))
  if (type === "response.failed")
    throw new Error(payload.response?.error?.message || "Codex response failed")

  if (type === "response.created") {
    output.responseId = payload.response?.id || output.responseId
    return
  }

  if (type === "response.output_item.added") {
    const item = payload.item
    if (item?.type === "message") {
      const block = { type: "text", text: "" }
      output.content.push(block as any)
      const contentIndex = output.content.length - 1
      slots.set(payload.output_index, { type: "text", block, contentIndex })
      stream.push({ type: "text_start", contentIndex, partial: output })
    } else if (item?.type === "reasoning") {
      const block = { type: "thinking", thinking: "", thinkingSignature: "" }
      output.content.push(block as any)
      const contentIndex = output.content.length - 1
      slots.set(payload.output_index, { type: "thinking", block, contentIndex })
      stream.push({ type: "thinking_start", contentIndex, partial: output })
    } else if (item?.type === "function_call") {
      // id 编码为 `call_id|item_id`，回放时需要还原两者
      const block = {
        type: "toolCall",
        id: `${item.call_id}|${item.id}`,
        name: item.name,
        arguments: {},
        partialJson: item.arguments || ""
      }
      output.content.push(block as any)
      const contentIndex = output.content.length - 1
      slots.set(payload.output_index, { type: "toolCall", block, contentIndex })
      stream.push({ type: "toolcall_start", contentIndex, partial: output })
    }
    return
  }

  const slot = slots.get(payload.output_index)
  if (type === "response.output_text.delta" && slot?.type === "text") {
    const delta = String(payload.delta || "")
    slot.block.text += delta
    stream.push({ type: "text_delta", contentIndex: slot.contentIndex, delta, partial: output })
  } else if (
    (type === "response.reasoning_summary_text.delta" ||
      type === "response.reasoning_text.delta") &&
    slot?.type === "thinking"
  ) {
    const delta = String(payload.delta || "")
    slot.block.thinking += delta
    stream.push({ type: "thinking_delta", contentIndex: slot.contentIndex, delta, partial: output })
  } else if (type === "response.function_call_arguments.delta" && slot?.type === "toolCall") {
    const delta = String(payload.delta || "")
    slot.block.partialJson += delta
    const parsed = tryParseJson(slot.block.partialJson)
    if (parsed !== undefined) slot.block.arguments = parsed
    stream.push({ type: "toolcall_delta", contentIndex: slot.contentIndex, delta, partial: output })
  } else if (type === "response.function_call_arguments.done" && slot?.type === "toolCall") {
    slot.block.partialJson = String(payload.arguments || slot.block.partialJson)
    slot.block.arguments = tryParseJson(slot.block.partialJson) || {}
  } else if (type === "response.output_item.done") {
    const item = payload.item
    if (slot?.type === "text" && item?.type === "message") {
      slot.block.text =
        item.content?.map((part: any) => part.text || part.refusal || "").join("") ||
        slot.block.text
      stream.push({
        type: "text_end",
        contentIndex: slot.contentIndex,
        content: slot.block.text,
        partial: output
      })
    } else if (slot?.type === "thinking" && item?.type === "reasoning") {
      slot.block.thinking =
        item.summary?.map((part: any) => part.text).join("\n\n") ||
        item.content?.map((part: any) => part.text).join("\n\n") ||
        slot.block.thinking
      // 完整 reasoning item 存进 signature，下一轮原样回放
      slot.block.thinkingSignature = JSON.stringify(item)
      stream.push({
        type: "thinking_end",
        contentIndex: slot.contentIndex,
        content: slot.block.thinking,
        partial: output
      })
    } else if (slot?.type === "toolCall" && item?.type === "function_call") {
      slot.block.arguments = tryParseJson(item.arguments || slot.block.partialJson) || {}
      delete slot.block.partialJson
      stream.push({
        type: "toolcall_end",
        contentIndex: slot.contentIndex,
        toolCall: slot.block,
        partial: output
      })
    }
    slots.delete(payload.output_index)
  } else if (type === "response.completed" || type === "response.incomplete") {
    output.responseId = payload.response?.id || output.responseId
    applyCodexUsage(output, payload.response, model)
    output.stopReason =
      type === "response.incomplete"
        ? "length"
        : output.content.some((block) => block.type === "toolCall")
          ? "toolUse"
          : "stop"
  }
}

async function tryStreamCodex(
  url: string,
  body: Json,
  endpoint: Endpoint,
  model: Model<Api>,
  output: AssistantMessage,
  stream: AssistantMessageEventStream,
  sessionId: string,
  metadata: CodexMetadata,
  signal?: AbortSignal
) {
  const bodyText = JSON.stringify(body)
  const maxRetries = getMaxRetries()
  let response: Response | undefined
  // 每次尝试重建：放弃这条流时靠 abort 断连（见下方流式段落的注释）
  let abortStream = new AbortController()

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const headers = createCodexHeaders(endpoint, sessionId, metadata, model.id)
    if (attempt === 0)
      writeDebugFile("request", model.id, undefined, () => ({
        url,
        headers: redactHeaders(headers),
        body,
        transport: "codex-sse"
      }))
    abortStream = new AbortController()
    const attemptSignal = signal
      ? AbortSignal.any([signal, abortStream.signal])
      : abortStream.signal
    try {
      response = await fetchWithProxy(url, {
        method: "POST",
        signal: attemptSignal,
        headers,
        body: bodyText
      })
    } catch (error) {
      if (attempt < maxRetries && !signal?.aborted) {
        log.warn(`🔄 codex request transport error, retrying`, {
          model: model.id,
          attempt,
          error: errorText(error)
        })
        await delay(getRetryDelayMs(attempt))
        continue
      }
      throw error
    }

    if (response.ok && (response.headers.get("content-type") || "").includes("text/event-stream"))
      break

    const raw = await response.text()
    const parsed = tryParseJson(raw) || { raw }
    const requestId = extractRequestId(parsed, response.headers)
    writeDebugFile("error", model.id, requestId, {
      status: response.status,
      requestId,
      body: parsed,
      raw,
      transport: "codex-sse",
      retryAttempt: attempt
    })
    if (!response.ok && attempt < maxRetries && isRetryableStatus(response.status)) {
      await delay(getRetryDelayMs(attempt, parseRetryAfterMs(response.headers.get("retry-after"))))
      response = undefined
      continue
    }
    throw new Error(raw || `HTTP ${response.status}`)
  }

  const sseResponse = response
  if (!sseResponse?.body) throw new Error("Codex stream response body missing")

  const slots = new Map<number, CodexSlot>()
  const reader = sseResponse.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ""
  let terminal = false

  const consume = (chunk: string) => {
    const data = parseSseData(chunk)
    if (!data || data === "[DONE]") return
    const payload = tryParseJson(data)
    if (!payload) throw new Error(`invalid Codex SSE payload: ${data.slice(0, 200)}`)
    applyCodexSsePayload(payload, output, stream, model, slots)
    if (payload.type === "response.completed" || payload.type === "response.incomplete")
      terminal = true
  }

  /**
   * 中途放弃（事件解析抛错、上游报错、被 abort）必须 abort 掉这次请求，否则上游会继续朝
   * 无人读取的连接推流。实测 omp 所用的 Bun 运行时里 reader.cancel() 不会关闭 socket
   * （1.5 秒内服务端还写进 1.2 万条事件），只有 abort 立刻断开；Node/undici 两者都有效。
   */
  try {
    while (true) {
      const { value, done } = await reader.read()
      buffer += decoder.decode(value ?? EMPTY_CHUNK, { stream: !done })
      buffer = drainSseChunks(buffer, consume)
      if (done) break
    }

    const tail = buffer.trim()
    if (tail) consume(tail)
  } catch (error) {
    abortStream.abort()
    throw error
  }

  if (!terminal) throw new Error("Codex stream ended before a terminal response event")

  writeDebugFile(
    "response",
    model.id,
    sseResponse.headers.get("x-oneapi-request-id") || undefined,
    () => ({
      status: sseResponse.status,
      responseId: output.responseId,
      stopReason: output.stopReason,
      usage: output.usage,
      transport: "codex-sse"
    })
  )
}

function applyJsonResponseToOutput(
  response: any,
  output: AssistantMessage,
  stream: AssistantMessageEventStream,
  model: Model<Api>
) {
  updateUsageFromAnthropic(output, response?.usage || {}, model)
  output.stopReason = mapStopReason(response?.stop_reason || "end_turn")

  const content = Array.isArray(response?.content) ? response.content : []
  for (const block of content) {
    if (block?.type === "text") {
      output.content.push({ type: "text", text: "" })
      const contentIndex = output.content.length - 1
      stream.push({ type: "text_start", contentIndex, partial: output })
      const text = String(block.text || "")
      ;(output.content[contentIndex] as any).text = text
      if (text) stream.push({ type: "text_delta", contentIndex, delta: text, partial: output })
      stream.push({ type: "text_end", contentIndex, content: text, partial: output })
    } else if (block?.type === "thinking") {
      output.content.push({
        type: "thinking",
        thinking: String(block.thinking || ""),
        thinkingSignature: block.signature || ""
      } as any)
      const contentIndex = output.content.length - 1
      stream.push({ type: "thinking_start", contentIndex, partial: output })
      if (block.thinking)
        stream.push({
          type: "thinking_delta",
          contentIndex,
          delta: String(block.thinking),
          partial: output
        })
      stream.push({
        type: "thinking_end",
        contentIndex,
        content: String(block.thinking || ""),
        partial: output
      })
    } else if (block?.type === "tool_use") {
      const toolCall = {
        type: "toolCall" as const,
        id: block.id,
        name: fromClaudeCodeName(block.name),
        arguments: block.input || {}
      }
      output.content.push(toolCall as any)
      const contentIndex = output.content.length - 1
      stream.push({ type: "toolcall_start", contentIndex, partial: output })
      stream.push({
        type: "toolcall_delta",
        contentIndex,
        delta: JSON.stringify(toolCall.arguments),
        partial: output
      })
      stream.push({ type: "toolcall_end", contentIndex, toolCall, partial: output })
    }
  }
}

/**
 * 取出一个 SSE 事件里的 data 负载。两个调用方都只看 data（分发一律按 payload.type），
 * 因此不解析 event 名，也不按行 split —— 手写扫描省掉每事件一个数组加若干行字符串。
 */
function parseSseData(chunk: string) {
  let data: string | undefined
  let extra: string[] | undefined
  let pos = 0
  while (pos <= chunk.length) {
    let lineEnd = chunk.indexOf("\n", pos)
    if (lineEnd === -1) lineEnd = chunk.length
    let end = lineEnd
    if (end > pos && chunk.charCodeAt(end - 1) === 13) end--
    // 跳过空行与注释行（以 : 开头）
    if (end > pos && chunk.charCodeAt(pos) !== 58 && chunk.startsWith("data:", pos)) {
      let start = pos + 5
      while (start < end && (chunk.charCodeAt(start) === 32 || chunk.charCodeAt(start) === 9))
        start++
      const value = chunk.slice(start, end)
      if (data === undefined) data = value
      else (extra ??= [data]).push(value)
    }
    pos = lineEnd + 1
  }
  return extra ? extra.join("\n") : (data ?? "")
}

/**
 * 消费 buffer 中所有完整的 SSE 事件，返回剩余的不完整尾部。
 *
 * 边界位置跨事件复用：一旦某种分隔符搜出 -1，说明剩余缓冲里都没有它，
 * 而游标只增，因此不必再搜。逐事件重扫两种分隔符会在中转站缓冲整段响应时
 * 退化成 O(n²)——实测 3.5 MB / 2 万事件 1140 ms，复用后 0.5 ms。
 */
function drainSseChunks(buffer: string, consume: (chunk: string) => void) {
  let cursor = 0
  let unix = buffer.indexOf("\n\n")
  let dos = buffer.indexOf("\r\n\r\n")
  while (unix !== -1 || dos !== -1) {
    const useDos = dos !== -1 && (unix === -1 || dos < unix)
    const end = useDos ? dos : unix
    consume(buffer.slice(cursor, end))
    cursor = end + (useDos ? 4 : 2)
    if (unix !== -1 && unix < cursor) unix = buffer.indexOf("\n\n", cursor)
    if (dos !== -1 && dos < cursor) dos = buffer.indexOf("\r\n\r\n", cursor)
  }
  return cursor > 0 ? buffer.slice(cursor) : buffer
}

function applySsePayloadEvent(
  payload: any,
  output: AssistantMessage,
  stream: AssistantMessageEventStream,
  model: Model<Api>,
  blockIndexByEventIndex: Map<number, number>
) {
  if (!payload?.type || payload.type === "ping" || payload.type === "message_stop") return

  if (payload.type === "error") {
    const errorText =
      payload?.error?.message || payload?.error || payload?.message || JSON.stringify(payload)
    throw new Error(String(errorText))
  }

  if (payload.type === "message_start") {
    output.responseId = payload.message?.id || output.responseId
    updateUsageFromAnthropic(output, payload.message?.usage || {}, model)
    return
  }

  if (payload.type === "content_block_start") {
    const block = payload.content_block
    if (block?.type === "text") {
      output.content.push({ type: "text", text: "" })
      const contentIndex = output.content.length - 1
      blockIndexByEventIndex.set(payload.index, contentIndex)
      stream.push({ type: "text_start", contentIndex, partial: output })
      return
    }
    if (block?.type === "thinking" || block?.type === "redacted_thinking") {
      output.content.push({
        type: "thinking",
        thinking: block.type === "redacted_thinking" ? "[Reasoning redacted]" : "",
        thinkingSignature: block.type === "redacted_thinking" ? String(block.data || "") : "",
        redacted: block.type === "redacted_thinking" ? true : undefined
      } as ThinkingContent)
      const contentIndex = output.content.length - 1
      blockIndexByEventIndex.set(payload.index, contentIndex)
      stream.push({ type: "thinking_start", contentIndex, partial: output })
      return
    }
    if (block?.type === "tool_use") {
      const toolCall = {
        type: "toolCall" as const,
        id: block.id,
        name: fromClaudeCodeName(block.name),
        arguments: (block.input as Json) || {},
        partialJson: ""
      }
      output.content.push(toolCall as any)
      const contentIndex = output.content.length - 1
      blockIndexByEventIndex.set(payload.index, contentIndex)
      stream.push({ type: "toolcall_start", contentIndex, partial: output })
    }
    return
  }

  if (payload.type === "content_block_delta") {
    const contentIndex = blockIndexByEventIndex.get(payload.index)
    if (contentIndex == null) return
    const block = output.content[contentIndex] as any
    if (!block) return

    if (payload.delta?.type === "text_delta" && block.type === "text") {
      block.text += String(payload.delta.text || "")
      stream.push({
        type: "text_delta",
        contentIndex,
        delta: String(payload.delta.text || ""),
        partial: output
      })
      return
    }
    if (payload.delta?.type === "thinking_delta" && block.type === "thinking") {
      block.thinking += String(payload.delta.thinking || "")
      stream.push({
        type: "thinking_delta",
        contentIndex,
        delta: String(payload.delta.thinking || ""),
        partial: output
      })
      return
    }
    if (payload.delta?.type === "input_json_delta" && block.type === "toolCall") {
      block.partialJson += String(payload.delta.partial_json || "")
      try {
        block.arguments = JSON.parse(block.partialJson)
      } catch {
        // partial json is expected during streaming
      }
      stream.push({
        type: "toolcall_delta",
        contentIndex,
        delta: String(payload.delta.partial_json || ""),
        partial: output
      })
      return
    }
    if (payload.delta?.type === "signature_delta" && block.type === "thinking") {
      block.thinkingSignature = `${block.thinkingSignature || ""}${String(payload.delta.signature || "")}`
    }
    return
  }

  if (payload.type === "content_block_stop") {
    const contentIndex = blockIndexByEventIndex.get(payload.index)
    if (contentIndex == null) return
    const block = output.content[contentIndex] as any
    if (!block) return

    // eventIndex → contentIndex 的映射只存在于 blockIndexByEventIndex，块上不再挂冗余字段
    blockIndexByEventIndex.delete(payload.index)

    if (block.type === "text") {
      stream.push({ type: "text_end", contentIndex, content: block.text, partial: output })
      return
    }
    if (block.type === "thinking") {
      stream.push({ type: "thinking_end", contentIndex, content: block.thinking, partial: output })
      return
    }
    if (block.type === "toolCall") {
      if (block.partialJson) {
        try {
          block.arguments = JSON.parse(block.partialJson)
        } catch {
          block.arguments = block.arguments || {}
        }
      }
      delete block.partialJson
      stream.push({ type: "toolcall_end", contentIndex, toolCall: block, partial: output })
    }
    return
  }

  if (payload.type === "message_delta") {
    if (payload.delta?.stop_reason) output.stopReason = mapStopReason(payload.delta.stop_reason)
    updateUsageFromAnthropic(output, payload.usage || {}, model)
  }
}

async function tryStreamClaude(
  url: string,
  body: Json,
  endpoint: Endpoint,
  model: Model<Api>,
  output: AssistantMessage,
  stream: AssistantMessageEventStream,
  sessionId: string,
  signal?: AbortSignal
) {
  const maxRetries = getMaxRetries()
  // body 含完整对话历史，序列化只做一次；重试之间变化的只有 header 里的 retry-count
  const requestBody = { ...body, stream: true }
  const bodyText = JSON.stringify(requestBody)

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const headers = getClaudeCodeHeaders(endpoint, attempt, sessionId)
    writeDebugFile("request", model.id, undefined, () => ({
      url,
      headers: redactHeaders(headers),
      body: requestBody,
      transport: "sse",
      retryAttempt: attempt,
      maxRetries
    }))

    // 别名让 debug thunk 捕获到已收窄的 Response（闭包不保留对 let 的收窄）
    let pending: Response
    // 每次尝试单独一个 controller：放弃这条流时靠 abort 断连（见下方流式段落的注释）
    const abortStream = new AbortController()
    const attemptSignal = signal
      ? AbortSignal.any([signal, abortStream.signal])
      : abortStream.signal
    try {
      pending = await fetchWithProxy(url, {
        method: "POST",
        signal: attemptSignal,
        headers,
        body: bodyText
      })
    } catch (error) {
      if (attempt < maxRetries && output.content.length === 0 && !signal?.aborted) {
        await delay(getRetryDelayMs(attempt))
        continue
      }
      throw error
    }

    const settled = pending
    const contentType = settled.headers.get("content-type") || ""
    if (!settled.ok || !contentType.includes("text/event-stream")) {
      const raw = await settled.text()
      const parsed = tryParseJson(raw) || { raw }
      const requestId = extractRequestId(parsed, settled.headers)
      writeDebugFile(settled.ok ? "response" : "error", model.id, requestId, () => ({
        status: settled.status,
        statusText: settled.statusText,
        requestId,
        headers: Object.fromEntries(settled.headers.entries()),
        body: parsed,
        raw,
        transport: "sse",
        retryAttempt: attempt,
        maxRetries
      }))
      if (
        attempt < maxRetries &&
        output.content.length === 0 &&
        !signal?.aborted &&
        isRetryableStatus(settled.status)
      ) {
        await delay(getRetryDelayMs(attempt, parseRetryAfterMs(settled.headers.get("retry-after"))))
        continue
      }
      if (settled.ok)
        throw new Error(`stream response was not SSE (content-type=${contentType || "<missing>"})`)
      throw new Error(raw || `HTTP ${settled.status}`)
    }

    if (!settled.body) throw new Error("stream response body missing")

    const blockIndexByEventIndex = new Map<number, number>()
    const reader = settled.body.getReader()
    const decoder = new TextDecoder()
    let buffer = ""

    const consume = (chunk: string) => {
      const data = parseSseData(chunk)
      if (!data || data === "[DONE]") return
      const payload = tryParseJson(data)
      if (!payload) throw new Error(`invalid SSE payload: ${data.slice(0, 200)}`)
      applySsePayloadEvent(payload, output, stream, model, blockIndexByEventIndex)
    }

    /**
     * 中途放弃（事件解析抛错、上游报错、被 abort）必须 abort 掉这次请求，否则上游会继续朝
     * 无人读取的连接推流，重试还会叠加。实测 omp 所用的 Bun 运行时里 reader.cancel() 不会
     * 关闭 socket（1.5 秒内服务端还写进 1.2 万条事件），只有 abort 立刻断开。
     */
    try {
      while (true) {
        const { value, done } = await reader.read()
        buffer += decoder.decode(value ?? EMPTY_CHUNK, { stream: !done })
        buffer = drainSseChunks(buffer, consume)
        if (done) break
      }

      const tail = buffer.trim()
      if (tail) consume(tail)
    } catch (error) {
      abortStream.abort()
      if (attempt < maxRetries && output.content.length === 0 && !signal?.aborted) {
        resetOutputState(output)
        await delay(getRetryDelayMs(attempt))
        continue
      }
      throw error
    }

    writeDebugFile(
      "response",
      model.id,
      settled.headers.get("x-oneapi-request-id") || undefined,
      () => ({
        status: settled.status,
        statusText: settled.statusText,
        headers: Object.fromEntries(settled.headers.entries()),
        body: {
          responseId: output.responseId,
          stopReason: output.stopReason,
          usage: output.usage,
          contentBlocks: output.content.length
        },
        transport: "sse",
        retryAttempt: attempt,
        maxRetries
      })
    )
    return
  }

  throw new Error("SSE request failed after retries")
}

async function postJson(
  url: string,
  body: Json,
  endpoint: Endpoint,
  modelId: string,
  sessionId: string,
  signal?: AbortSignal
) {
  const maxRetries = getMaxRetries()
  const bodyText = JSON.stringify(body)
  let lastErrorText = ""

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    const headers = getClaudeCodeHeaders(endpoint, attempt, sessionId)
    if (attempt === 0) {
      writeDebugFile("request", modelId, undefined, () => ({
        url,
        headers: redactHeaders(headers),
        body
      }))
    }

    let response: Response
    try {
      response = await fetchWithProxy(url, { method: "POST", signal, headers, body: bodyText })
    } catch (error) {
      if (attempt < maxRetries && !signal?.aborted) {
        await delay(getRetryDelayMs(attempt))
        continue
      }
      throw error
    }

    const settled = response
    const text = await settled.text()
    lastErrorText = text
    const parsed = tryParseJson(text) ?? { raw: text }
    const requestId = extractRequestId(parsed, settled.headers)

    writeDebugFile(settled.ok ? "response" : "error", modelId, requestId, () => ({
      status: settled.status,
      statusText: settled.statusText,
      requestId,
      headers: Object.fromEntries(settled.headers.entries()),
      body: parsed,
      raw: text,
      retryAttempt: attempt,
      maxRetries
    }))

    if (settled.ok) return parsed
    if (attempt < maxRetries && !signal?.aborted && isRetryableStatus(settled.status)) {
      await delay(getRetryDelayMs(attempt, parseRetryAfterMs(settled.headers.get("retry-after"))))
      continue
    }
    throw new Error(text || `HTTP ${response.status}`)
  }

  throw new Error(lastErrorText || "HTTP request failed after retries")
}

type SourceProvider = ReturnType<typeof loadSourceProvider>

/** Claude Code 模拟：Anthropic Messages API，优先 SSE，PI_ANYROUTER_STREAM_MODE=off 时退回一次性 JSON。 */
async function streamClaude(
  endpoint: Endpoint,
  model: Model<Api>,
  context: Context,
  options: SimpleStreamOptions | undefined,
  output: AssistantMessage,
  stream: AssistantMessageEventStream,
  sessionId: string
) {
  const url = `${endpoint.baseUrl.replace(/\/$/, "")}/v1/messages?beta=true`
  const requestBody: Json = {
    model: toClaudeCodeRequestModel(model.id),
    messages: convertMessages(context.messages),
    max_tokens: options?.maxTokens || model.maxTokens || 32000,
    stream: false,
    metadata: createClaudeCodeMetadata(sessionId),
    system: createClaudeCodeSystem(
      joinSystemPrompt(
        context.systemPrompt,
        "You are an expert coding assistant operating inside omp."
      ),
      endpoint
    ),
    context_management: {
      edits: [{ type: "clear_thinking_20251015", keep: "all" }]
    },
    tools: convertTools(context.tools || [])
  }
  if (options?.reasoning && model.reasoning) {
    requestBody.thinking = { type: "adaptive", display: "omitted" }
    requestBody.output_config = { effort: mapReasoningEffort(options.reasoning) }
  }

  if (getStreamMode() !== "off") {
    await tryStreamClaude(
      url,
      requestBody,
      endpoint,
      model,
      output,
      stream,
      sessionId,
      options?.signal
    )
    return
  }

  const response = await postJson(url, requestBody, endpoint, model.id, sessionId, options?.signal)
  applyJsonResponseToOutput(response, output, stream, model)
}

/** Codex CLI 模拟：OpenAI Responses API，仅支持 SSE。 */
async function streamCodex(
  endpoint: Endpoint,
  model: Model<Api>,
  context: Context,
  options: SimpleStreamOptions | undefined,
  output: AssistantMessage,
  stream: AssistantMessageEventStream,
  sessionId: string
) {
  const metadata = createCodexMetadata(sessionId, randomUUID())
  const requestBody = buildCodexRequestBody(model, context, options, sessionId, metadata)
  await tryStreamCodex(
    getCodexResponsesUrl(endpoint.baseUrl),
    requestBody,
    endpoint,
    model,
    output,
    stream,
    sessionId,
    metadata,
    options?.signal
  )
}

function streamAnyRouter(
  model: Model<Api>,
  context: Context,
  options?: SimpleStreamOptions
): AssistantMessageEventStream {
  const stream = createAssistantMessageEventStream()
  ;(async () => {
    const output: AssistantMessage = {
      role: "assistant",
      content: [],
      api: model.api,
      provider: model.provider,
      model: model.id,
      usage: createEmptyUsage(),
      stopReason: "stop",
      timestamp: Date.now()
    }

    let client: ClientKind = "claude"
    try {
      const endpoint = resolveEndpoint(getSourceProvider(), model.id)
      client = endpoint.client
      log.debug(`${client === "codex" ? "🤖" : "🧠"} dispatching request`, {
        model: model.id,
        client,
        version: endpoint.fingerprint.version,
        sessionId: SESSION_ID,
        baseUrl: endpoint.baseUrl,
        attribution: endpoint.attribution
      })

      stream.push({ type: "start", partial: output })

      if (client === "codex") {
        await streamCodex(endpoint, model, context, options, output, stream, SESSION_ID)
      } else {
        await streamClaude(endpoint, model, context, options, output, stream, SESSION_ID)
      }

      if (options?.signal?.aborted) throw new Error("Request was aborted")
      stream.push({
        type: "done",
        reason: output.stopReason as "stop" | "length" | "toolUse",
        message: output
      })
      stream.end()
    } catch (error) {
      output.stopReason = options?.signal?.aborted ? "aborted" : "error"
      output.errorMessage = `${LOG_PREFIX} ${errorText(error)}`
      log.error(`❌ request failed`, {
        model: model.id,
        client,
        stopReason: output.stopReason,
        error: errorText(error)
      })
      writeDebugFileRaw("error", model.id, undefined, {
        client,
        stopReason: output.stopReason,
        errorMessage: output.errorMessage
      })
      stream.push({ type: "error", reason: output.stopReason, error: output })
      stream.end()
    }
  })()
  return stream
}

export default function (pi: ExtensionAPI) {
  // 接入 omp 的 file logger，日志随 omp 落到 ~/.omp/logs/omp.<date>.<pid>.log
  setLogger(pi.logger)
  pi.setLabel("AnyRouter")

  try {
    const source = getSourceProvider()
    const models = source.models.map((model) => ({
      id: model.id,
      name: model.name ? `${model.name} (AnyRouter)` : `${model.id} (AnyRouter)`,
      api: API_ID,
      reasoning: model.reasoning ?? true,
      input: model.input ?? ["text"],
      cost: {
        input: model.cost?.input ?? 0,
        output: model.cost?.output ?? 0,
        cacheRead: model.cost?.cacheRead ?? 0,
        cacheWrite: model.cost?.cacheWrite ?? 0
      },
      contextWindow: model.contextWindow ?? 200000,
      maxTokens: model.maxTokens ?? 32000
    }))

    // provider 级凭据只作为兜底，实际请求按模型解析（见 resolveEndpoint）
    pi.registerProvider(PROVIDER_NAME, {
      baseUrl: source.baseUrl || source.models[0].baseUrl,
      apiKey: source.apiKey || source.models[0].apiKey,
      api: API_ID,
      models,
      streamSimple: streamAnyRouter
    })

    log.info(`✅ provider registered`, {
      provider: PROVIDER_NAME,
      configPath: CONFIG_PATH,
      claudeVersion: source.fingerprints.claude.version,
      codexVersion: source.fingerprints.codex.version,
      configTtlMs: CONFIG_TTL_MS,
      models: source.models.map((model) => `${model.id}:${model.client}`)
    })
  } catch (error) {
    log.error(`🚫 failed to register provider`, {
      configPath: CONFIG_PATH,
      error: errorText(error),
      hint: "override with PI_ANYROUTER_CONFIG / PI_ANYROUTER_BASE_URL / PI_ANYROUTER_API_KEY"
    })
  }
}
