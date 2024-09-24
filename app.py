from flask import Flask, request, jsonify
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline

app = Flask(__name__)

# Load the pre-trained model
model_id = "CompVis/stable-diffusion-v-1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

@app.route('/generate', methods=['POST'])
def generate_image():
    data = request.json
    prompt = data.get('prompt', '')
    
    # Generate the image
    image = pipe(prompt).images[0]
    
    # Save and return the image
    image_path = 'output.png'
    image.save(image_path)
    
    return jsonify({'image_url': image_path})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
