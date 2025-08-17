
import json
import logging
import re


def parse_qwen_output(raw_output: str) -> dict:
    """
    Worker for Qwen. It now fully parses the JSON content of tool calls
    and extracts the chain-of-thought from <think> blocks.
    """
    print(">>> Qwen Worker (with thinking) Called <<<")

    # 1. Use a simple, reliable regex to extract everything between the tags.
    tool_pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    analysis_content = ""
    think_match = think_pattern.search(raw_output)
    if think_match:
        analysis_content = think_match.group(1).strip()

    parsed_tool_calls = []
    for match in tool_pattern.finditer(raw_output):
        tool_content_str = match.group(1).strip()

        # 2. Clean the extracted string in Python.
        # Check for the Qwen2.5 double-brace format and fix it.
        if tool_content_str.startswith("{{") and tool_content_str.endswith("}}"):
            # Slice the string to remove the outer braces, leaving valid JSON.
            tool_content_str = tool_content_str[1:-1]

        # 3. Now, append the cleaned string.
        parsed_tool_calls.append(tool_content_str)


    # 4. Create the clean final answer by removing BOTH blocks
    # First, remove the thinking block
    clean_text = think_pattern.sub("", raw_output)
    # Then, from that result, remove the tool calls
    final_answer = tool_pattern.sub("", clean_text).strip() if len(parsed_tool_calls) == 0 else "\n\n"
    
    logging.debug(f"Analysis: {analysis_content}")
    logging.debug(f"Final Tool Calls: {parsed_tool_calls}")
    logging.debug(f"Final Answer: {final_answer}")

    # 5. Return the complete, standardized dictionary
    return {
        "analysis": analysis_content,
        "tool_calls": parsed_tool_calls,
        "final_answer": final_answer
    }

def parse_openai_harmony_output(raw_output: str) -> dict:
    """
    Parses the raw string output from a openai harmony response with a 
    regex for handling real-world tool call formats.
    """
    parsed_data = {
        "analysis": "",
        "tool_calls": [],
        "final_answer": None
    }

    if not raw_output.endswith("<|end|>"):
        logging.warning("Raw output does not end with '<|end|>'. This may lead to parsing issues.")
        raw_output += "<|end|>"

    # Regex for text channels (this pattern remains correct)
    text_channel_pattern = re.compile(
        r"(?:<\|start\|>assistant)?<\|channel\|>(?P<channel>final|analysis)<\|message\|>(?P<content>.*?)<\|end\|>",
        re.DOTALL
    )

    # --- THE CORRECTED REGEX FOR TOOL CALLS ---
    # This new pattern correctly matches the structure from your example.
    tool_call_pattern = re.compile(
        r"<\|start\|>assistant<\|channel\|>commentary.*?to=functions\.(?P<name>[\w_]+)"
        r".*?<\|message\|>(?P<args>\{.*?\})<\|call\|>",
        re.DOTALL
    )

    # Find and process all text channel blocks
    for match in text_channel_pattern.finditer(raw_output):
        channel = match.group("channel")
        content = match.group("content").strip()
        if channel == "analysis":
            # Append to analysis, ensuring we don't overwrite previous content
            parsed_data["analysis"] = parsed_data["analysis"] + "\n" + content
        elif channel == "final":
            # For final, we replace the content if it exists
            parsed_data["final_answer"] = content

    # Find and process all tool call blocks with the new pattern
    for match in tool_call_pattern.finditer(raw_output):
        tool_name = match.group("name")
        tool_args_str = match.group("args").strip()
        try:
            arguments = json.loads(tool_args_str)
            parsed_data["tool_calls"].append(json.dumps({
                "name": tool_name,
                "arguments": arguments
            }))
        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON arguments for tool '{tool_name}': {tool_args_str}")
            continue
            
    if not parsed_data["final_answer"] or len(parsed_data["tool_calls"]) != 0:
        # If there are tool calls, we don't return the final answer
        parsed_data["final_answer"] = "\n\n"
    return parsed_data

def parse_gemma_output(raw_output: str) -> dict:
    """
    Worker for Gemma family models. Parses Python-like function call syntax
    for tool calls, handling named, positional, multiple, and empty arguments.
    """
    print(">>> Gemma Worker Called <<<")

    def _parse_gemma_args(arg_string: str) -> dict:
        """A helper function to parse Gemma's complex argument syntax."""
        if not arg_string.strip():
            return {}

        parsed_args = {}
        positional_arg_count = 0
        
        # Split by comma, but be careful of commas inside quotes (this is a simplification)
        # A more advanced parser might use ast.literal_eval for safety.
        args = arg_string.split(',')

        for arg in args:
            arg = arg.strip()
            if "=" in arg:
                # Handle named argument: arg_name=arg_value
                key, value = arg.split('=', 1)
                key = key.strip()
                # Strip quotes from value if they exist
                value = value.strip().strip("'\"")
                parsed_args[key] = value
            else:
                # Handle positional argument: arg_value
                key = f"arg{positional_arg_count}"
                value = arg.strip().strip("'\"")
                parsed_args[key] = value
                positional_arg_count += 1
        return parsed_args

    # This regex finds patterns like `function_name( ... )`
    # and captures the name and the argument string.
    tool_call_pattern = re.compile(r"([\w_]+)\s*\((.*?)\)")
    
    parsed_tool_calls = []
    for match in tool_call_pattern.finditer(raw_output):
        function_name = match.group(1)
        raw_args = match.group(2)
        
        # Use the helper to parse the arguments
        arguments = _parse_gemma_args(raw_args)
        
        # We dump back to a JSON string to match our standard output format
        parsed_tool_calls.append(json.dumps({
            "name": function_name,
            "arguments": arguments
        }))

    # The final answer is the raw text with all tool calls removed
    final_answer = tool_call_pattern.sub("", raw_output).strip() if len(parsed_tool_calls) == 0 else "\n\n"

    return {
        "analysis": "", # Gemma format doesn't have a standard <think> block
        "tool_calls": parsed_tool_calls,
        "final_answer": final_answer
    }


def identify_template_style(prompt_text: str) -> str:
    """
    Correctly identifies the chat template style based only on tokens
    found in the formatted PROMPT text.
    """
    if "<|start|>" in prompt_text and "<|end|>" in prompt_text:
        return "openai_harmony"

    if "<|im_start|>" in prompt_text:
        return "qwen-chatml"

    if "<start_of_turn>" in prompt_text:
        return "gemma"

    if "[INST]" in prompt_text:
        return "llama-instruct"

    return "default"

class LLMOutputParser:
    # ... (as defined before, with registered parsers)
    def __init__(self):
        self._parsers = {
            "qwen-chatml": parse_qwen_output,
            "openai_harmony": parse_openai_harmony_output,
            "gemma": parse_gemma_output,
            "default": lambda text: {"final_answer": text.strip()}
        }
    def get_parser(self, style_name: str):
        return self._parsers.get(style_name, self._parsers["default"])
