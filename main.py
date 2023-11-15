from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load pre-trained DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Function to generate a response
def generate_response(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output_ids = model.generate(input_ids, max_length=max_length, num_beams=5, temperature=0.7)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    bot_response = generate_response(user_input)
    return render_template('index.html', user_input=user_input, bot_response=bot_response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
