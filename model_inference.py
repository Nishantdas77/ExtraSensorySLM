# # from transformers import AutoModelForCausalLM, AutoTokenizer

# # model_name = "Qwen/Qwen3-4B"

# # # load the tokenizer and the model
# # tokenizer = AutoTokenizer.from_pretrained(model_name)
# # model = AutoModelForCausalLM.from_pretrained(
# #     model_name,
# #     torch_dtype="auto",
# #     device_map="auto"
# # )

# # # prepare the model input
# # prompt = "You are a specialist in understanding IMU signals. I will give you signal classification and their evidence, your job is to answer them. "
# # messages = [
# #     {"role": "user", "content": prompt}
# # ]
# # text = tokenizer.apply_chat_template(
# #     messages,
# #     tokenize=False,
# #     add_generation_prompt=True,
# #     enable_thinking=True # Switches between thinking and non-thinking modes. Default is True.
# # )
# # model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# # # conduct text completion
# # generated_ids = model.generate(
# #     **model_inputs,
# #     max_new_tokens=1024
# # )
# # output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# # # parsing thinking content
# # try:
# #     # rindex finding 151668 (</think>)
# #     index = len(output_ids) - output_ids[::-1].index(151668)
# # except ValueError:
# #     index = 0

# # thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
# # content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

# # print("thinking content:", thinking_content)
# # print("content:", content)


# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# torch.random.manual_seed(0)

# model = AutoModelForCausalLM.from_pretrained(
#     "microsoft/Phi-3.5-mini-instruct", 
#     device_map="cuda", 
#     torch_dtype="auto", 
    
# )
# tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

# messages = [
#     {"role": "system", "content": "You are a helpful AI assistant."},
#     {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
#     {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
#     {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
# ]

# pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
# )

# generation_args = {
#     "max_new_tokens": 500,
#     "return_full_text": False,
#     "do_sample": False,
# }

# output = pipe(messages, **generation_args)
# print(output[0]['generated_text'])

import torch
from transformers import Mistral3ForConditionalGeneration, MistralCommonBackend

model_id = "mistralai/Ministral-3-3B-Instruct-2512"

tokenizer = MistralCommonBackend.from_pretrained(model_id)
model = Mistral3ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

image_url = "https://static.wikia.nocookie.net/essentialsdocs/images/7/70/Battle.png/revision/latest?cb=20220523172438"

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What action do you think I should take in this situation? List all the possible actions and explain why you think they are good or bad.",
            },
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    },
]

tokenized = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)

tokenized["input_ids"] = tokenized["input_ids"].to(device="cuda")
tokenized["pixel_values"] = tokenized["pixel_values"].to(dtype=torch.bfloat16, device="cuda")
image_sizes = [tokenized["pixel_values"].shape[-2:]]

output = model.generate(
    **tokenized,
    image_sizes=image_sizes,
    max_new_tokens=512,
)[0]

decoded_output = tokenizer.decode(output[len(tokenized["input_ids"][0]):])
print(decoded_output)

