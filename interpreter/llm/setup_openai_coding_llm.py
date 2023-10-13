import litellm

from ..rag.llamaindexmanager import LlamaIndexManager
from ..utils.merge_deltas import merge_deltas
from ..utils.parse_partial_json import parse_partial_json
from ..utils.convert_to_openai_messages import convert_to_openai_messages
from ..utils.display_markdown_message import display_markdown_message
from ..plugins.function_manager import extend_functions
import tokentrim as tt

from ..rag.index_manager import IndexManager

def setup_openai_coding_llm(interpreter):
    """
    Takes an Interpreter (which includes a ton of LLM settings),
    returns a OI Coding LLM (a generator that takes OI messages and streams deltas with `message`, `language`, and `code`).
    """

    functions = extend_functions(interpreter)

    def coding_llm(messages):

        # Convert messages
        messages = convert_to_openai_messages(messages)

        # Add OpenAI's recommended function message
        messages[0]["content"] += "\n\nOnly use the functions you have been provided with."

        # Seperate out the system_message from messages
        # (We expect the first message to always be a system_message)
        system_message = messages[0]["content"]
        messages = messages[1:]

        # Trim messages, preserving the system_message
        try:
            messages = tt.trim(messages=messages, system_message=system_message, model=interpreter.model)
        except:
            if interpreter.context_window:
                messages = tt.trim(messages=messages, system_message=system_message,
                                   max_tokens=interpreter.context_window)
            else:
                display_markdown_message("""
                **We were unable to determine the context window of this model.** Defaulting to 3000.
                If your model can handle more, run `interpreter --context_window {token limit}` or `interpreter.context_window = {token limit}`.
                """)
                messages = tt.trim(messages=messages, system_message=system_message, max_tokens=3000)

        if interpreter.debug_mode:
            print("Sending this to the OpenAI LLM:", messages)

        # Create LiteLLM generator
        params = {
            'model': interpreter.model,
            'messages': messages,
            'stream': True,
            'functions': functions
        }

        # Optional inputs
        if interpreter.api_base:
            params["api_base"] = interpreter.api_base
        if interpreter.api_key:
            params["api_key"] = interpreter.api_key
        if interpreter.max_tokens:
            params["max_tokens"] = interpreter.max_tokens
        if interpreter.temperature:
            params["temperature"] = interpreter.temperature

        # These are set directly on LiteLLM
        if interpreter.max_budget:
            litellm.max_budget = interpreter.max_budget
        if interpreter.debug_mode:
            litellm.set_verbose = True

        # Report what we're sending to LiteLLM
        if interpreter.debug_mode:
            print("Sending this to LiteLLM:", params)

        response = litellm.completion(**params)

        accumulated_deltas = {}
        language = None
        code = ""
        accumulated_arguments = {}

        for chunk in response:

            if ('choices' not in chunk or len(chunk['choices']) == 0):
                # This happens sometimes
                continue

            delta = chunk["choices"][0]["delta"]

            # Accumulate deltas
            accumulated_deltas = merge_deltas(accumulated_deltas, delta)

            if "content" in delta and delta["content"]:
                yield {"message": delta["content"]}

            if ("function_call" in accumulated_deltas
                    and "arguments" in accumulated_deltas["function_call"]):
                if interpreter.debug_mode:
                    print("function_call\t -- Accumulated deltas:", str(accumulated_deltas))
                arguments = accumulated_deltas["function_call"]["arguments"]
                arguments = parse_partial_json(arguments)

                if arguments:
                    accumulated_arguments = {**accumulated_arguments, **arguments}

                    if accumulated_deltas["function_call"]["name"] == "execute":
                        if (language is None
                                and "language" in arguments
                                and "code" in arguments  # <- This ensures we're *finished* typing language, as opposed to partially done
                                and arguments["language"]):
                            language = arguments["language"]
                            yield {"language": language}

                        if language is not None and "code" in arguments:
                            # Calculate the delta (new characters only)
                            code_delta = arguments["code"][len(code):]
                            # Update the code
                            code = arguments["code"]
                            # Yield the delta
                            if code_delta:
                                yield {"code": code_delta}

        # # execute plugin function if we have one
        if "function_call" in accumulated_deltas and accumulated_deltas["function_call"]["name"] != "execute" and \
                accumulated_deltas["function_call"]["name"] in interpreter.functions:
            if interpreter.debug_mode:
                print(f"\nArguments: {str(arguments)}")
                print("function_call\t -- Accumulated deltas:", str(accumulated_deltas))
            # Execute the function
            function_name = accumulated_deltas["function_call"]["name"]
            parameters = accumulated_arguments
            function = interpreter.functions[function_name]
            results = function(function_name=function_name, parameters=parameters)

            IndexManager().set_stored_message(results)
            if interpreter.debug_mode:
                print("Storing function output: ", results)

            display_markdown_message("Storing function output: ```" + str(results) + "```")

            # params['messages'].append({"message": results, "role": "system"})
            # Exclude this function from the next params... HACKHACKHACK
            # params['functions'] = [f for f in params['functions'] if f['name'] != function_name]

            interpreter.messages.append({"message": results, "role": "system"})

            yield {"output": results}



    return coding_llm
