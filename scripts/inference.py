from hf_olmo import *
from transformers import AutoModelForCausalLM, AutoTokenizer

olmo = AutoModelForCausalLM.from_pretrained("../save_sft/step157-unsharded/")
tokenizer = AutoTokenizer.from_pretrained("../save_sft/step157-unsharded/")

while True:
    message = input('Enter your question (type \'quit\' to exit): ')
    # Check if the user wants to quit the loop
    if message.lower() == 'quit':
        break
    # Tokenize the input message
    inputs = tokenizer([message], return_tensors='pt', return_token_type_ids=False)

    # Generate a response using the model
    response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)

    print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
