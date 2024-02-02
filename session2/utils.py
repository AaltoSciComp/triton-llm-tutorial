# +

import json

def text_dataset(train_file)
   
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_file,
        block_size=128)
    return train_datast

def tokenize_function(examples):
    text = examples['conversation'][0][0]["input"] + examples['conversation'][0][0]['output']
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="pt",
        padding='max_length',
        truncation=True,
        max_length=1024
    )
    return tokenized_inputs

def preprocess_intents_json(intents_file):
    with open(intents_file, "r") as f:
        data = json.load(f)
    
    preprocessed_data = []
    
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            preprocessed_data.append(f"User: {pattern}\n")
            for response in intent["responses"]:
                preprocessed_data.append(f"Assistant: {response}\n")
    
    return "".join(preprocessed_data)

def save_preprocessed_data(preprocessed_data, output_file):
    with open(output_file, "w") as f:
        f.write(preprocessed_data)
        
def inference(text, model, tokenizer, max_input_tokens=1000, max_output_tokens=100):
  # Tokenize
    input_ids = tokenizer.encode(
          text,
          return_tensors="pt",
          truncation=True,
          max_length=max_input_tokens
    )

    # Generate
    device = model.device
    generated_tokens_with_prompt = model.generate(
    input_ids=input_ids.to(device),
    max_length=max_output_tokens
    )

    # Decode
    generated_text_with_prompt = tokenizer.batch_decode(generated_tokens_with_prompt, skip_special_tokens=True)

    # Strip the prompt
    generated_text_answer = generated_text_with_prompt[0][len(text):]

    return generated_text_answer

# intents_file = "intents.json"
# # output_file = "mental_health_data.txt"

# preprocessed_data = preprocess_intents_json(intents_file)
# # save_preprocessed_data(preprocessed_data, output_file)



