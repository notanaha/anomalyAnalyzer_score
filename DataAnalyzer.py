import json
import pandas as pd
import numpy as np

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
        },
        {
            "type": "function",
            "function": {
                "name": "signatures_distances",
                "description": "A tool to use for calculating signatures distances, where signatures consist of \
                    np.mean(data), np.std(data), np.min(data),np.max(data), np.max(data) - np.min(data), np.median(data), np.sum(data ** 2).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "anomaly_files": {
                            "type": "array",
                            "description": "A list of files with anomaly type: [(<path to a file>, <anomaly type>), ...]",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "file_path": {
                                        "type": "string",
                                        "description": "Path to the file"
                                    },
                                    "anomaly_type": {
                                        "type": "string",
                                        "description": "Anomaly type"
                                    }
                                },
                                "required": ["file_path", "anomaly_type"]
                            },
                        },
                        "test_file": {
                            "type": "string",
                            "description": "test file to predict the type of anomaly.",
                        },
                    },
                    "required": ["anomaly_files", "test_file"],
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
            elif function_name == "signatures_distances":
                function_response = self.signature_distances(
                    anomaly_files=function_args.get("anomaly_files"),
                    test_file=function_args.get("test_file")
                )
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

    def signature_distances(self, anomaly_files:list, test_file:str):
        anomaly_files_tuples = [(item["file_path"], item["anomaly_type"]) for item in anomaly_files]

        def compute_signature(df):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found to compute signature.")

            signature = {}
            for col in numeric_cols:
                if col == "time":
                    continue
                data = df[col].values
                signature[f"{col}_mean"]   = np.mean(data)
                signature[f"{col}_std"]    = np.std(data)
                signature[f"{col}_min"]    = np.min(data)
                signature[f"{col}_max"]    = np.max(data)
                signature[f"{col}_range"]  = np.max(data) - np.min(data)
                signature[f"{col}_median"] = np.median(data)
                signature[f"{col}_energy"] = np.sum(data ** 2)
            return signature

        known_signatures = []
        for filepath, label in anomaly_files_tuples:
            df = pd.read_csv(filepath)
            signature = compute_signature(df)
            known_signatures.append((signature, label))

        # ------------------------------------------------------------
        # 2. Compute signature for the test file
        # ------------------------------------------------------------
        test_df = pd.read_csv(test_file)
        test_signature = compute_signature(test_df)

        # ------------------------------------------------------------
        # 3. Compare with each known signature (distance-based)
        # ------------------------------------------------------------
        def euclidean_distance(sig1, sig2):
            dist_sq = 0.0
            for k in sig1.keys():
                # If all files have exactly the same numeric columns,
                # sig1 and sig2 will have the same keys:
                dist_sq += (sig1[k] - sig2[k])**2
            return np.sqrt(dist_sq)

        distances = []
        for sig, label in known_signatures:
            dist = euclidean_distance(test_signature, sig)
            distances.append((dist, label))
            #print(f"Distance to {label}: {dist:.4f}")

        distances.sort(key=lambda x: x[0])  # sort ascending by distance
        closest_distance, closest_label = distances[0]
        print(f"Closest distance: {closest_distance:.4f} to {closest_label}")
        return json.dumps({
            "distances": distances,
            "closest_distance": closest_distance,
            "closest_label": closest_label
        })
    