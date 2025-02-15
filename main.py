from flask import Flask, request, jsonify
from diffusers import StableDiffusionPipeline
import torch

app = Flask(__name__)

model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
model.to("cuda")

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data['text']
    image = model(prompt).images[0]
    image.save("output.png")
    return jsonify({"image_url": "https://your-repl-url/output.png"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
