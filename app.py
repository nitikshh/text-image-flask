from flask import Flask, request, jsonify
from PIL import Image
import io
import os
import torch
from diffusers import StableDiffusionPipeline

app = Flask(__name__)

# Load the pre-trained model
model_id = "CompVis/stable-diffusion-v-1-4"
try:
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cpu")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

@app.route('/generate', methods=['POST'])
def generate_image():
    data = request.json
    prompt = data.get('prompt', '')
    
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    try:
        # Generate the image
        image = pipe(prompt).images[0]
        
        # Save and return the image
        image_path = 'output.png'
        image.save(image_path)
        
        return jsonify({'image_url': image_path})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
