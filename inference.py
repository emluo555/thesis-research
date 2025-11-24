# Load model directly
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

MODEL_DIR = "/scratch/gpfs/ZHUANGL/el5267/thesis-research/models/models--Xkev--Llama-3.2V-11B-cot"

processor = AutoProcessor.from_pretrained(MODEL_DIR)
model = AutoModelForVision2Seq.from_pretrained(MODEL_DIR)
image_path = "pastry.png"
image = Image.open(image_path)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            # {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "How to make this pastry?"}
        ]
    },
]
inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))