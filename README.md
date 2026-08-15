# omp-anyrouter
oh-my-pi extension for anyrouter

## Code By CC
根据omp.sh的插件开发规范，增加一个anyrouter的provider支持，用来模拟claude、codex的原生请求到特定的中转站。
支持通过日志的形式输出异常，日志的目录跟随omp

插件可以参考的实现：`https://github.com/xifan2333/pi-anyrouter/blob/main/index.ts`， 但这个是PI的插件，需要支持OMP的插件形式

插件开发文档：`https://github.com/can1357/oh-my-pi/blob/main/docs/extensions.md`

配置文件格式，用户可以指定使用的模拟客户端，默认可以根据模型名字来判断：

```json
{
  "baseUrl": "https://xxx",
  "apiKey": "sk-xxxx",
  "models": [
    {
      "id": "claude-opus-5[1m]",
      "name": "Claude Opus 5 (1M context)",
      "reasoning": true,
      "client": "claude|codex"
      "input": ["text"],
      "cost": {
        "input": 0,
        "output": 0,
        "cacheRead": 0,
        "cacheWrite": 0
      },
      "contextWindow": 1000000,
      "maxTokens": 64000
    }
  ]
}


```

第三方有用的模拟转发实现，可以参考：
`https://github.com/router-for-me/CLIProxyAPI`
`https://github.com/Wei-Shaw/sub2api`


### 第二轮
目前项目可以运行，需要进行优化：

1. 项目中存在大量的`anyrouter-cc` ，这个不需要，没有`-cc`
2. 日志的输出与否，增加一个环境变量控制
3. 每个模型的apikey 和baseurl 可以增加一个配置，覆盖全局的
4. 更新readme，说明项目的作用，以及使用方法

### 第三轮

- 需要增加一个配置，参考`CLAUDE_CODE_ATTRIBUTION_HEADER`的配置作用，控制是否开启特别的标识，默认不开

- 优化一下日志输出，增加一些emoj开头标识，让终端更加活泼

- 现在对`https://github.com/can1357/oh-my-pi`增加一个新插件，支持让模型的baseUrl配置支持环境变量读取，目前key是支持。是否可以通过插件机制增加这个特性。

## 参考
https://github.com/xifan2333/pi-anyrouter

https://github.com/zccrs/pi-anyrouter (支持omp)

https://github.com/yeahnangua/pi-anyrouter

https://github.com/JCloud77/pi-anyrouter
