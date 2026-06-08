import torch
import warnings
from transformers import GPT2Config, LlamaConfig, LlamaForCausalLM, GPT2LMHeadModel

warnings.filter_warnings("ignore")

def simulate_memory_for_table(model_name, config):
    with torch.device("meta"):
        if "GPT-2" in model_name:
            model = GPT2LMHeadModel(config)
        else:
            model = LlamaForCausalLM(config)
            
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adamw_memory_gb = (total_params * 4 * 2) / (1024 ** 3)
    adam_mini_memory_gb = (total_params * 4) / (1024 ** 3)
    saving = (1 - adam_mini_memory_gb / adamw_memory_gb) * 100
    print(f"| {model_name:<15} | {adamw_memory_gb:>8.2f} GB | {adam_mini_memory_gb:>12.2f} GB | {saving:>6.1f}% ↓ |")

def run_experiment():
    print("-" * 60)
    print(f"| {'Model':<15} | {'AdamW (GB)':<10} | {'Adam-mini (GB)':<14} | {'Economy':<8} |")
    print("-" * 60)
    
    gpt2_xl_config = GPT2Config.from_pretrained("gpt2-xl")
    simulate_memory_for_table("GPT-2-1.5B", gpt2_xl_config)
    
    llama_1b = LlamaConfig(vocab_size=32000, hidden_size=2048, intermediate_size=5632, num_hidden_layers=22, num_attention_heads=32)
    simulate_memory_for_table("Llama 2-1B", llama_1b)
    
    llama_7b = LlamaConfig(vocab_size=32000, hidden_size=4096, intermediate_size=11008, num_hidden_layers=32, num_attention_heads=32)
    simulate_memory_for_table("Llama 2-7B", llama_7b)
    
    llama_3_8b = LlamaConfig(vocab_size=128256, hidden_size=4096, intermediate_size=14336, num_hidden_layers=32, num_attention_heads=32, num_key_value_heads=8)
    simulate_memory_for_table("Llama 3-8B", llama_3_8b)
    
    llama_13b = LlamaConfig(vocab_size=32000, hidden_size=5120, intermediate_size=13824, num_hidden_layers=40, num_attention_heads=40)
    simulate_memory_for_table("Llama 2-13B", llama_13b)
    print("-" * 60)

if __name__ == "__main__":
    run_experiment()
