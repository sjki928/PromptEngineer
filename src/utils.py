def get_prompt(messages):
    conversation = []
    instruction = ''
    for entry in messages:
        role = entry["role"]
        content = entry["content"]
        
        if role == "user":
            conversation.append(f"user:\n{content}")

        elif role == "assistant":
            conversation.append(f"assitant:\n{content}")
        
        else:
            instruction = content

    while len("system:"+instruction + "\n" + "\n".join(conversation) + "\nassistant:") > 128e3:
        conversation.pop(0)
        
    return "system:" + instruction + "\n" + "\n".join(conversation) + "\nassistant:"