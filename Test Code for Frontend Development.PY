# Install required packages
!pip install unsloth

!python -m pip install --no-deps git+https://github.com/huggingface/transformers.git@v4.49.0

import transformers
transformers.__version__


!ngrok config add-authtoken 2uClxbpe2zbawFVmjnzfuUeBPFo_3G6kcv6dQ47sXNzCwMkhf

!pip install streamlit pyngrok peft nest_asyncio torch

# Verify transformers version
!python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# Write the Streamlit app to a file
with open("app.py", "w") as f:
    f.write("""
import streamlit as st
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from unsloth import FastVisionModel
from peft import PeftModel


# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit", # Llama 3.2 vision support
    "unsloth/Llama-3.2-11B-Vision-bnb-4bit",
    "unsloth/Llama-3.2-90B-Vision-Instruct-bnb-4bit", # Can fit in a 80GB card!
    "unsloth/Llama-3.2-90B-Vision-bnb-4bit",

    "unsloth/Pixtral-12B-2409-bnb-4bit",              # Pixtral fits in 16GB!
    "unsloth/Pixtral-12B-Base-2409-bnb-4bit",         # Pixtral base model

    "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",          # Qwen2 VL support
    "unsloth/Qwen2-VL-7B-Instruct-bnb-4bit",
    "unsloth/Qwen2-VL-72B-Instruct-bnb-4bit",

    "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit",      # Any Llava variant works!
    "unsloth/llava-1.5-7b-hf-bnb-4bit",
] # More models at https://huggingface.co/unsloth



# Load the fine-tuned model
@st.cache_resource
def load_model():
    # Step 1: Load the base model first (without adapters)
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit",
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth"
    )

    # Step 2: Load the adapter weights separately
    model = PeftModel.from_pretrained(model, "/content/content/vinayak_model_new")

    # Step 3: Load the image processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    return model, tokenizer, processor

# Generate reasoning from image and question
def generate_reasoning(model, tokenizer, processor, image, question):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "I am providing an image that contains a question with multiple choice options. "
                        "Please extract all the details from the image and output your detailed answer as a single string. "
                        "Include the complete question, all answer choices, the correct answer (one of 'A', 'B', 'C', or 'D'), "
                        "and a step-by-step explanation of your reasoning from understanding the question to reaching a conclusion. "
                        "For example, your output should contain the full question text, the available options, and a comprehensive explanation. "
                        "User question: " + question
                    )
                }
            ]
        }
    ]
    # Convert to model format
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    # Process image and text
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # Generate response
    output_ids = model.generate(**inputs, max_new_tokens=1024)
    # Extract generated text
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs["input_ids"], output_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    return output_text


# Streamlit UI
st.title("Vinayak's Vision-Language Assistant")
st.write("Upload an image and ask a question to get a detailed explanation.")

# Load model on startup
model, tokenizer, processor = load_model()

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width =True)

    # Text input for question
    question = st.text_input("What would you like to know about this image?")

    # Process when button is clicked
    if st.button("Generate Answer") and question:
        with st.spinner("Generating detailed explanation..."):
            answer = generate_reasoning(model, tokenizer, processor, image, question)
            st.write("### Explanation")
            st.write(answer)


""")

# Run Streamlit in the background
import subprocess
subprocess.Popen(["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=True"])

# Import and expose via ngrok
import nest_asyncio
nest_asyncio.apply()  # Fix asyncio event loop issues in Colab
from pyngrok import ngrok

# Define the tunnel configuration with 'addr' instead of 'port'
tunnel = ngrok.connect(addr="8501", proto="http")  # Specify protocol as 'http'

# Get the public URL
url = tunnel.public_url
print("Streamlit App URL:", url)
