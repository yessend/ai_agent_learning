import json

# Define function to get the JSON array of relevant node_ids
def extract_json_array(text: str):
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON array found")

    candidate = text[start:end + 1]
    data = json.loads(candidate)

    if not isinstance(data, list):
        raise ValueError("JSON is not a list")

    return data