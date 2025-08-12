import base64
import requests

# OpenAI API Key
API_KEY = ""


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def prompt_make(prompt_path, ex_prompt):
    with open(prompt_path, "r", encoding='utf-8') as f:
        txt = f.readlines()
        prompt_system = txt[1]
        prompt = txt[3]
        if len(txt) > 4:
            for i in range(4, len(txt)):
                prompt = prompt + txt[i]
        prompt = prompt + ex_prompt
        return prompt_system, prompt


def gpt_4o_mini(prompt_path, ex_prompt, img_path=None):
    api_key = API_KEY

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    prompt_system, prompt = prompt_make(prompt_path, ex_prompt)
    content = [{"type": "text", "text": prompt}]
    if img_path:
        base64_image = encode_image(img_path)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
    payload = {
        "model": "gpt-4o-mini",       
        "messages": [
            {"role": "system", "content": prompt_system},
            {"role": "user", "content": content}
        ]}
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    output = response.json()
    return output["choices"][0]['message']["content"]
