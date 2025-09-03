# MLX LM Adopter with Tool Calling 
This project demonstrates [opanai api compatible tool calling](https://platform.openai.com/docs/guides/function-calling) on [MLX LM](https://github.com/ml-explore/mlx-lm).

Supported models are:
- gpt-oss: Tested with [inferencerlabs/openai-gpt-oss-20b-MLX-6.5bit](https://huggingface.co/inferencerlabs/openai-gpt-oss-20b-MLX-6.5bit)
- gemma-3: Tested with [mlx-community/gemma-3-12b-it-qat-8bit](https://huggingface.co/mlx-community/gemma-3-12b-it-qat-8bit). Additional chat template used to enable tool listing and parsing.

Functionalities:
- Tool related parsing (tool calls and tool responses) for supported model series

Not working:
- streaming

## Problem Statement
In order to build an AI agent application using [Google ADK](https://google.github.io/adk-docs), leveraging local LLM models on a Mac device, model server with [MLX](https://github.com/ml-explore/mlx) support was needed. 
While [MLX LM](https://github.com/ml-explore/mlx-lm) is one of them, it lacks of flexible tool-call parser supports, only working with Qwen series when matched with Google ADK.

## Solution
Added parsing adoptor on APIHandler class to parse model output and format into openai api compatible JSON body. 'handle_chat_completion' class method has been overridden.
LLMOutputParser class handles identification and parsing LLM output.

## Comments
Handling and parsing of LLM output may be possible to be done on agent server side, but in this case, **MLX LM server had some issue with loosing part of LLM output** (as of 2025-08-17, v0.26.3) because of its embedded tool call parsing logic, making it unusable when trying to parse on agent server side.



