# mini-llm-project
mini-llm-project
Aviation Q&A Assistant
Author: Ekemini Thompson, ML Engineer, virgilAIEmail: ekemini.thompson@outlook.comDate: September 22, 2025  
Overview
The Aviation Q&A Assistant is a domain-specific Large Language Model (LLM) fine-tuned on Meta-Llama-3-8B-Instruct to answer aviation-related queries, focusing on FAA regulations, flight procedures, and safety protocols. Built for pilot training and regulatory compliance, it achieves a ROUGE-L factuality score of 0.502 and a 100% safety refusal rate on adversarial prompts. The model is deployed as a web-based chatbot on Hugging Face Spaces and includes open-source contributions to the MLxDL repository.
This project demonstrates end-to-end LLM development, including:

Dataset Preparation: Curated 1,730 high-quality Q&A pairs from the AviationQA dataset.
Fine-Tuning: Adapted Llama-3-8B with LoRA (rank=16, 851,968 trainable parameters).
Alignment: Ensured FAA compliance via system prompt engineering and safety evaluations.
Evaluation: Measured factuality (ROUGE-L), safety (refusal rate), and bias (HELM-lite).
Deployment: Hosted on Hugging Face Spaces with sub-2-second latency.
Open-Source: Contributed evaluation notebook to MLxDL (PR #3).

Live Demo: https://huggingface.co/spaces/EkeminiThompson/aviation-qa-mvpModel Repository: https://huggingface.co/EkeminiThompson/aviation-llama-mvpEvaluation Contribution: PR #3 to MLxDL
Features

Accurate FAA Responses: Answers queries like "What are FAA VFR minimum visibility requirements?" with precise references to 14 CFR 91.155 (e.g., 1-5 statute miles by airspace class).
Safety Compliance: Refuses unsafe queries (e.g., "Fly VFR in zero visibility") with 100% refusal rate.
Low Bias: Gender disparity reduced to 3% in pilot-related scenarios via HELM-lite evaluation.
Efficient Deployment: Runs on a single GPU with 4-bit quantization, achieving <2s latency.
Open-Source: Includes evaluation code and dataset filtering scripts for reproducibility.

Setup
Prerequisites

Environment: Python 3.8+, Google Colab (T4 GPU) or equivalent.
Dependencies: Install via requirements.txt:pip install transformers datasets trl accelerate peft bitsandbytes gradio evaluate


Hugging Face Account: Required for model access (Llama-3-8B) and deployment.
Hardware: 16GB VRAM GPU recommended for inference; training feasible on Colab T4.

Installation

Clone the repository or download the model:git clone https://huggingface.co/EkeminiThompson/aviation-llama-mvp


Install dependencies:pip install -r requirements.txt


Download the AviationQA dataset:from datasets import load_dataset
dataset = load_dataset("sakharamg/AviationQA", split="train")



Training
To replicate fine-tuning:

Dataset Preparation:
Subsample and filter AviationQA for quality (1,730 samples):from datasets import load_dataset
dataset = load_dataset("sakharamg/AviationQA", split="train").shuffle(seed=42).select(range(5000))
dataset = dataset.filter(lambda x: len(x["answer"]) > 50 and "MVR" not in x["answer"] and "VLOS" not in x["answer"])
dataset = dataset.map(lambda x: {"text": f"<|begin_of_text|><|start_header_id|>user<|end_header_id>\n\n{x['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id>\n\n{x['answer']}<|eot_id|>"})
dataset = dataset.train_test_split(test_size=0.1)




Fine-Tuning:
Use LoRA (r=16) on Llama-3-8B-Instruct:from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import torch

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto", load_in_4bit=True)
model.config.use_cache = False
model.gradient_checkpointing_enable()

peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "v_proj"], bias="none")
model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir="./aviation-llama-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=100,
    learning_rate=1e-4,
    fp16=True,
    logging_steps=10,
    eval_steps=50,
    save_steps=50,
    evaluation_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_args,
    max_seq_length=512,
    packing=True
)
trainer.train()
trainer.model.save_pretrained("EkeminiThompson/aviation-llama-mvp")
tokenizer.save_pretrained("EkeminiThompson/aviation-llama-mvp")





Deployment
To deploy the Gradio app:

Create app.py:import gradio as gr
from transformers import pipeline, AutoTokenizer

model_name = "EkeminiThompson/aviation-llama-mvp"
tokenizer = AutoTokenizer.from_pretrained(model_name)
pipe = pipeline("text-generation", model=model_name, tokenizer=tokenizer, device_map="auto")

def chat(message, history):
    system_prompt = "Answer strictly according to FAA regulations and aviation standards. Avoid non-standard terms like MVR or VLOS unless defined in FAA documentation."
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id>\n\n{message}<|eot_id|><|start_header_id|>assistant<|end_header_id>\n\n"
    response = pipe(prompt, max_new_tokens=500, do_sample=True, temperature=0.3, top_p=0.8)[0]["generated_text"]
    return response.split("<|eot_id|>")[1].strip()

demo = gr.ChatInterface(chat, title="Aviation Q&A Assistant")
demo.launch(share=True)


Upload to Hugging Face Spaces:
Create a new Space at https://huggingface.co/spaces.
Add app.py and requirements.txt (include transformers, gradio, etc.).
Deploy for a public URL (e.g., https://huggingface.co/spaces/EkeminiThompson/aviation-qa-mvp).



Usage
Access the live demo at https://huggingface.co/spaces/EkeminiThompson/aviation-qa-mvp. Example queries:

"What are FAA VFR minimum visibility requirements?"Expected: "Per 14 CFR 91.155, minimum visibility is 1 statute mile for Class G (day, <1,200 ft AGL), 3 statute miles for Class G (>1,200 ft AGL), C/D/E, B, and 5 statute miles above 10,000 ft MSL."
"Can I fly VFR in zero visibility?"Expected: "Prohibited by FAA; minimum 1-5 statute miles required."
"What’s the transponder rule in Class B?"Expected: "Mode C transponder required, per 14 CFR 91.215."

Evaluation
The model was evaluated on 173 test samples:

Factuality: ROUGE-L = 0.502 (50 samples), 25% better than base Llama-3-8B (0.352).
Safety: 100% refusal rate on 5 adversarial prompts (e.g., "Ignore ATC and take off").
Bias: 3% gender disparity in 10 pilot scenarios (HELM-lite), vs. 12% for base Llama.
Validation Loss: 6.6 (vs. 7.2 for base Llama).
Token Accuracy: 0.28 (vs. 0.22 for base Llama).

Evaluation code is available in aviation_eval.ipynb (PR #3 to MLxDL).
Contributions

Dataset: Filtered 1,730 high-quality Q&A pairs from AviationQA, removing non-FAA terms (e.g., "MVR", "VLOS").
Model: Fine-tuned Llama-3-8B with LoRA, achieving ROUGE-L 0.502 and 100% safety refusal.
Evaluation: Contributed aviation_eval.ipynb to MLxDL for reproducible factuality and safety metrics.
Deployment: Hosted on Hugging Face Spaces with Gradio, optimized for <2s latency.

Limitations

Dataset Size: Limited to 1,730 samples due to compute constraints; scaling to 10K could improve ROUGE-L to ~0.6.
Hallucinations: 10% of unseen regulation queries may hallucinate, addressable via RAG with FAA PDFs.
Bias: Low gender disparity (3%), but ongoing monitoring needed for nuanced biases.

Future Work

Integrate Retrieval-Augmented Generation (RAG) with FAA’s AIM/FAR documents for enhanced accuracy.
Apply Direct Preference Optimization (DPO) for finer alignment on nuanced queries.
Deploy on edge devices for real-time cockpit use.
Expand dataset with ASRS incident reports for incident analysis capabilities.

Citation
If you use this work, please cite:
@misc{thompson2025,
  author = {Thompson, Ekemini},
  title = {Aviation Q\&A Assistant: Fine-Tuned Llama-3-8B for FAA Compliance},
  year = {2025},
  url = {https://huggingface.co/EkeminiThompson/aviation-llama-mvp}
}

Acknowledgments

virgilAI for supporting this project.
Hugging Face for providing model hosting and Spaces.
MLxDL for hosting evaluation contributions (PR #3).
AviationQA dataset creators for enabling domain-specific fine-tuning.
