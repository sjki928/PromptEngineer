def get_prompt(messages):
    conversation = []
    instruction = ''
    for entry in messages:
        role = entry["role"]
        content = entry["content"]
        
        if role == "user":
            conversation.append(f"### user:\n{content}")

        elif role == "assistant":
            conversation.append(f"### assitant:\n{content}")
        
        else:
            instruction = content

    while len(instruction + "\n" + "\n".join(conversation) + "\n### user:") > 128e3:
        conversation.pop(0)
        
    return instruction + "\n".join(conversation) + "\n### assistant:"