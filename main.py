from flask import Flask, request, render_template
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    conversation_history = f"You: {user_input}\nBot:"
    
    # Tokenize the conversation history
    input_ids = tokenizer.encode(conversation_history, return_tensors="pt")
    
    # Generate a response using the model
    output = model.generate(input_ids, max_length=150, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95, temperature=0.7)
    
    # Decode the generated output
    generated_response = tokenizer.decode(output[0], skip_special_tokens=True)

    return render_template('index.html', user_input=user_input, bot_response=generated_response)

if __name__ == '__main__':
    app.run(debug=True, host="172.174.215.233", port=80)
