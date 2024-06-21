from flask import Flask, request, jsonify
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

app = Flask(__name__)

# Load GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

@app.route('/generate', methods=['POST'])
def generate_text():
    input_text = request.json['text']
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate text based on input
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return jsonify({'response': generated_text})

if __name__ == '__main__':
    app.run('127.0.0.1', port=5000)
