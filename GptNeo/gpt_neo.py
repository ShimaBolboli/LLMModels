from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Define the model name you want to use
model_name = "EleutherAI/gpt-neo-1.3B"

# Download tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

# Optionally, adjust model and tokenizer settings
# tokenizer.do_lower_case = True  # Example adjustment

# Example prompt for generating text
prompt = "Once upon a time"

# Encode the prompt to tensor format expected by model
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# Generate text based on input
output = model.generate(input_ids, max_length=100, num_return_sequences=1)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Text:")
print(generated_text)
