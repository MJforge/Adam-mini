import torch
import gc
import warnings
from transformers import GPT2Config, GPT2LMHeadModel
from adam_mini import Adam_mini

warnings.filter_warnings("ignore")

def get_cuda_memory_mb():
    return torch.cuda.memory_allocated() / (1024 ** 2)

def run_test_for_dtype(dtype_str):
    device = "cuda"
    dtype = torch.float32 if dtype_str == "float32" else torch.float16
    config = GPT2Config.from_pretrained("gpt2")
    
    # AdamW
    model = GPT2LMHeadModel(config).to(dtype).to(device)
    input_ids = torch.randint(0, config.vocab_size, (2, 512)).to(device)
    labels = input_ids.clone()
    
    base_memory = get_cuda_memory_mb()
    optimizer_adamw = torch.optim.AdamW(model.parameters(), lr=1e-4)
    outputs = model(input_ids, labels=labels)
    outputs.loss.backward()
    optimizer_adamw.step()
    
    adamw_state_memory = get_cuda_memory_mb() - base_memory
    del optimizer_adamw, model
    gc.collect()
    torch.cuda.empty_cache()
    
    # Adam-mini
    model = GPT2LMHeadModel(config).to(dtype).to(device)
    base_memory_mini = get_cuda_memory_mb()
    optimizer_mini = Adam_mini(model.named_parameters(), lr=1e-4, weight_decay=0.1, betas=(0.9, 0.95), eps=1e-8)
    outputs = model(input_ids, labels=labels)
    outputs.loss.backward()
    optimizer_mini.step()
    
    mini_state_memory = get_cuda_memory_mb() - base_memory_mini
    del optimizer_mini, model
    gc.collect()
    torch.cuda.empty_cache()
    
    saving = (1 - mini_state_memory / adamw_state_memory) * 100
    return adamw_state_memory, mini_state_memory, saving

def run_real_gpu_experiment():
    if not torch.cuda.is_available():
        print("CUDA GPU not found!")
        return

    m_adamw_32, m_mini_32, save_32 = run_test_for_dtype("float32")
    m_adamw_16, m_mini_16, save_16 = run_test_for_dtype("float16")

    print("\n" + "="*50)
    print(f"| Format   | AdamW Memory | Adam-mini Memory | Economy |")
    print(f"| float32  | {m_adamw_32:>9.2f} MB | {m_mini_32:>14.2f} MB | {save_32:>7.1f}% |")
    print(f"| float16  | {m_adamw_16:>9.2f} MB | {m_mini_16:>14.2f} MB | {save_16:>7.1f}% |")
    print("="*50)

if __name__ == "__main__":
    run_real_gpu_experiment()
