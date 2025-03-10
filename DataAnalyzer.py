import json
import pandas as pd

class DataAnalyzer:
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "data_analyzer",
                "description": "A tool to use for calculating correlations of the given.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "query string how the data is analyzed.",
                        },
                        "fpath": {
                            "type": "string",
                            "description": "File path of the data to be analyzed.",
                        },
                    },
                    "required": ["query", "fpath"],
                },
            }
        }
    ]

    def __init__(self, aoai_client, chat_model):
        self.aoai_client = aoai_client
        self.chat_model = chat_model

    def data_analyzer(self, query, fpath):
        print("data_analyzer: query=", query)
        print("data_analyzer: fpath=", fpath)

        df = pd.read_csv(fpath)
        columns = df.columns[1:-1]  # Exclude the first column ("time") and the last column ("test_data")
        anomaly_types = columns.tolist()        

        correlations = {}
        for c in anomaly_types:
            corr_value = df['test_data'].corr(df[c])
            correlations[c] = corr_value

        return "correlations=" + json.dumps(correlations)


    def o1_query(self, messages, tools=None, tool_choice=None):
        response = self.aoai_client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            reasoning_effort="high",
            tools=tools,
            tool_choice=tool_choice
        )

        response_message = response.choices[0].message
        messages.append(response_message)

        return response_message, messages

    def call_o1_model(self, messages, tools=None, tool_choice=None, user_context=None):
        response_message, messages = self.o1_query(messages, tools=tools, tool_choice=tool_choice)

        answer = response_message.content
        tool_calls = response_message.tool_calls
        print("tool_calls:", tool_calls)

        if tool_calls:
            self.checkandrun_function_calling(tool_calls, messages)
            response_message, messages = self.o1_query(messages)
        else:
            print("No tool calls were made by the model.")

        return response_message, messages

    def checkandrun_function_calling(self, tool_calls, messages):
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            print("function_name: ", function_name)
            function_args = json.loads(tool_call.function.arguments)
            print("function_args: ", function_args)

            if function_name == "data_analyzer":
                function_response = self.data_analyzer(
                    query=function_args.get("query"),
                    fpath=function_args.get("fpath")
                )
                print("function_response: ", function_response)
            else:
                function_response = json.dumps({"error": "Unknown function"})

            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )

        return