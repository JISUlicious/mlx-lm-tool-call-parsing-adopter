

import argparse
import json
import logging
from typing import List
import uuid

from mlx_lm.sample_utils import make_sampler, make_logits_processors
from mlx_lm.generate import stream_generate
from mlx_lm.server import stopping_criteria, process_message_content, convert_chat
from mlx_lm.server import APIHandler, ModelProvider
import mlx_lm.server

from src.parser.llm_output_parser import LLMOutputParser, identify_template_style


class AdapterAPIHandler(APIHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The handler now holds an instance of our adapter
        self.output_parser = LLMOutputParser()

    def handle_completion(
            self: APIHandler,
            prompt: List[int],
            stop_id_sequences: List[List[int]],
        ):
            """
            Generate a response to a prompt and send it to the client in a single batch.

            Args:
                prompt (List[int]): The tokenized prompt.
                stop_id_sequences (List[List[int]]): A list of stop words passed
                to the stopping_criteria function
            """
            # tokens = []
            # finish_reason = "length"
            stop_sequence_suffix = None
            if self.stream:
                self.end_headers()
                logging.debug(f"Starting stream:")
            else:
                logging.debug(f"Starting completion:")
            token_logprobs = []
            top_tokens = []

            prompt_text = self.tokenizer.decode(prompt)
            prompt = self.get_prompt_cache(prompt)

            style = identify_template_style(prompt_text)
            
            # Get the correct worker function from the adapter
            output_parser = LLMOutputParser()
            parser_worker = output_parser.get_parser(style)
            
            logging.info(f"Identified style '{style}'. Using parser '{parser_worker.__name__}'.")

            # --- Setup and Generation Loop (unchanged) ---
            # sampler = make_sampler(...)
            # logits_processors = make_logits_processors(...)


            # text = ""
            # tic = time.perf_counter()
            sampler = make_sampler(
                self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p,
                xtc_probability=self.xtc_probability,
                xtc_threshold=self.xtc_threshold,
                xtc_special_tokens=[
                    self.tokenizer.eos_token_id,
                    self.tokenizer.encode("\n"),
                ],
            )
            logits_processors = make_logits_processors(
                self.logit_bias,
                self.repetition_penalty,
                self.repetition_context_size,
            )

            # tool_calls = []
            # tool_text = ""
            # in_tool_call = False
            # segment = ""

            raw_output_buffer = ""
            tokens = []
            finish_reason = "length"
            for gen_response in stream_generate(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                max_tokens=self.max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
                prompt_cache=self.prompt_cache.cache,
                draft_model=self.model_provider.draft_model,
                num_draft_tokens=self.num_draft_tokens,
            ):
                logging.debug(gen_response.text)
                token = gen_response.token
                if token == self.tokenizer.eos_token_id:
                    finish_reason = "stop"
                    break
                
                tokens.append(token)
                # We decode only the last token for efficiency
                raw_output_buffer += self.tokenizer.decode([token])

                # Check for stop conditions (this part is unchanged)
                stop_condition = stopping_criteria(tokens, stop_id_sequences, self.tokenizer.eos_token_id)
                if stop_condition.stop_met:
                    finish_reason = "stop"
                    break
                
                if len(tokens) >= self.max_tokens:
                    break
            logging.debug(f"Raw Output Buffer: {raw_output_buffer}")
        
            structured_data = parser_worker(raw_output_buffer)
            logging.debug(f"Structured Data: {structured_data}")
            # --- Populate final variables from the structured data ---
            final_text = structured_data.get("final_answer", "")
            final_tool_calls = structured_data.get("tool_calls", [])
            logging.debug(f"Final Text: {final_text}")  
            logging.debug(f"Final Tool Calls: {final_tool_calls}, {len(final_tool_calls)}")
            # --- Final Response Formatting (same as original) ---
            response = self.generate_response(
                    final_text if len(final_tool_calls) == 0 else "\n\n",
                    finish_reason,
                    len(prompt),
                    len(tokens),
                    token_logprobs=token_logprobs,
                    top_tokens=top_tokens,
                    tokens=tokens,
                    tool_calls=final_tool_calls,
                )
            
            response_json = json.dumps(response).encode()
            indent = "\t"  # Backslashes can't be inside of f-strings
            logging.debug(f"Outgoing Response: {json.dumps(response, indent=indent)}")

            # Send an additional Content-Length header when it is known
            self.send_header("Content-Length", str(len(response_json)))
            self.end_headers()
            self.wfile.write(response_json)
            self.wfile.flush()

    def handle_chat_completions(self) -> List[int]:
        """
        Handle a chat completion request.

        Returns:
            mx.array: A mx.array of the tokenized prompt from the request body
        """
        body = self.body
        assert "messages" in body, "Request did not contain messages"

        # Determine response type
        self.request_id = f"chatcmpl-{uuid.uuid4()}"
        self.object_type = "chat.completion.chunk" if self.stream else "chat.completion"
        if self.tokenizer.chat_template:
            messages = body["messages"]
            process_message_content(messages)
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tools=body.get("tools"),
                add_generation_prompt=True,
                **self.model_provider.cli_args.chat_template_args,
            )
            logging.debug(f"Chat prompt: {self.tokenizer.decode(prompt)}")
        else:
            prompt = convert_chat(body["messages"], body.get("role_mapping"))
            prompt = self.tokenizer.encode(prompt)
        
        return prompt


def main():
    parser = argparse.ArgumentParser(description="MLX Http Server.")
    parser.add_argument(
        "--model",
        type=str,
        help="The path to the MLX model weights, tokenizer, and config",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        help="Optional path for the trained adapter weights and config.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host for the HTTP server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the HTTP server (default: 8080)",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        help="A model to be used for speculative decoding.",
        default=None,
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        help="Number of tokens to draft when using speculative decoding.",
        default=3,
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Enable trusting remote code for tokenizer",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default="",
        help="Specify a chat template for the tokenizer",
        required=False,
    )
    parser.add_argument(
        "--use-default-chat-template",
        action="store_true",
        help="Use the default chat template",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Default sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Default nucleus sampling top-p (default: 1.0)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Default top-k sampling (default: 0, disables top-k)",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="Default min-p sampling (default: 0.0, disables min-p)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Default maximum number of tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--chat-template-args",
        type=json.loads,
        help="""A JSON formatted string of arguments for the tokenizer's apply_chat_template, e.g. '{"enable_thinking":false}'""",
        default="{}",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), None),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    mlx_lm.server.run(args.host, args.port, ModelProvider(args), handler_class=AdapterAPIHandler)


if __name__ == "__main__":
    main()
