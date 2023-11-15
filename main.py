from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Load pre-trained GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']

    # Tokenize the user's input
    input_ids = tokenizer.encode(user_input, return_tensors="pt")

    # Generate code using GPT-2 model
    output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.5)
    # Decode the generated code
    generated_code = tokenizer.decode(output[0], skip_special_tokens=True)

    return render_template('index.html', user_input=user_input, bot_response=generated_code)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=80)
