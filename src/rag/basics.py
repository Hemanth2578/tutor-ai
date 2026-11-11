import os, sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OpenAI_API_Key"))

# prompt = f"How many parameters LLaMA 4 models have?"

# An article about the release of LLaMA 4 model.
# https://ai.meta.com/blog/llama-4-multimodal-intelligence/#. (Summarized Content)
ARTICLE = """
Meta has unveiled the **Llama 4 herd**, a new generation of open-weight, natively multimodal AI models designed to push the boundaries of performance, efficiency, and accessibility. The release includes **Llama 4 Scout** and **Llama 4 Maverick**, alongside a preview of the massive **Llama 4 Behemoth** teacher model. These models introduce advanced **mixture-of-experts (MoE) architectures**, native text–image integration, and record-breaking context lengths, establishing Llama 4 as a major leap in multimodal AI innovation.
**Llama 4 Scout** is a **17B active parameter model with 16 experts** (109B total parameters) that offers **10 million token context length**, far exceeding Llama 3’s 128K limit. Scout’s architecture leverages **interleaved attention layers (iRoPE)** and inference-time temperature scaling to achieve superior length generalization, enabling complex tasks like multi-document summarization, codebase reasoning, and long-context retrieval. Despite its smaller size, Scout surpasses prior Llama models and competitors such as Gemma 3 and Gemini 2.0 Flash-Lite in performance, all while being deployable on a **single NVIDIA H100 GPU**. Its multimodal training allows strong image grounding, multi-image reasoning, and visual question answering.
**Llama 4 Maverick**, also with **17B active parameters**, uses **128 experts** and totals **400B parameters**, alternating dense and MoE layers for inference efficiency. It rivals or exceeds larger models like **GPT-4o, Gemini 2.0 Flash, and DeepSeek v3** on benchmarks for coding, reasoning, multilinguality, and multimodal tasks. Its efficient design makes it deployable on a single **NVIDIA H100 DGX host**, balancing performance with cost-effectiveness. Maverick was refined through a revamped **post-training pipeline**—lightweight supervised fine-tuning, continuous **online reinforcement learning (RL)** with adaptive difficulty filtering, and direct preference optimization—resulting in superior reasoning, conversational fluency, and multimodal understanding. Maverick’s **chat version achieved an ELO score of 1417 on LMArena**, reflecting best-in-class general assistant capabilities.
At the top of the hierarchy, **Llama 4 Behemoth** serves as the **teacher model**, with **288B active parameters, 16 experts, and nearly 2 trillion total parameters**. It outperforms **GPT-4.5, Claude Sonnet 3.7, and Gemini 2.0 Pro** on STEM benchmarks like MATH-500 and GPQA Diamond. Behemoth’s scale demanded innovative training strategies, including pruning 95% of supervised fine-tuning data, advanced reinforcement learning with dynamically filtered prompts, and an optimized asynchronous MoE-parallelized RL infrastructure. These techniques boosted reasoning, coding, and multilingual capabilities while maintaining instruction-following reliability.
All Llama 4 models are **natively multimodal**, trained with large-scale text, image, and video data, and leverage **Meta’s MetaP technique** for reliable hyperparameter scaling. They support **200 languages**, with 10x more multilingual tokens than Llama 3, and employ **FP8 precision** for efficient training across trillions of tokens. Safety remains a priority: Meta integrates **Llama Guard**, **Prompt Guard**, and **CyberSecEval** for content protection, while **Generative Offensive Agent Testing (GOAT)** automates adversarial red-teaming. Llama 4 also significantly reduces **political and social bias**, refusing fewer prompts and responding more neutrally than Llama 3.
In sum, **Llama 4 Scout, Maverick, and Behemoth** represent a major leap in open AI research: compact yet powerful models with unmatched context length, multimodal fluency, reasoning power, and efficiency. By making Scout and Maverick openly available on **llama.com and Hugging Face**, Meta empowers developers, enterprises, and researchers worldwide to build the next generation of AI-driven experiences.
"""

prompt = f"Use the following article as the source and answer the question:\n\n{ARTICLE}\n\nHow many parameters LLaMA 4 Models have?"

# response = client.responses.create(
#     model="gpt-5-nano",
#     input=prompt,
#     reasoning={'effort': 'minimal'}
# )

# print(f"Response: {response.output_text}")

print(f"Length of the Article: {len(ARTICLE.split(" "))} words")