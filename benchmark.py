import argparse
import gc
import time

import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

from torchao.quantization import quant_api


CALIBRATION_PROMPT = "It was a bright cold day in April, and the clocks were striking thirteen."
"Winston Smith, his chin nuzzled into his breast in an effort to escape the vile wind, slipped"
" quickly through the glass doors of Victory Mansions, though not quickly enough to prevent a"
" swirl of gritty dust from entering along with him."
"The hallway smelt of boiled cabbage and old rag mats. At one end of it a coloured poster, too"
" large for indoor display, had been tacked to the wall. It depicted simply an enormous face, more"
" than a metre wide: the face of a man of about forty-five, with a heavy black moustache and ruggedly"
" handsome features. Winston made for the stairs. It was no use trying the lift. Even at the best of"
" times it was seldom working, and at present the electric current was cut off during daylight hours."
"It was part of the economy drive in preparation for Hate Week. The flat was seven flights up, and"
" Winston, who was thirty-nine and had a varicose ulcer above his right ankle, went slowly, resting"
" several times on the way. On each landing, opposite the lift-shaft, the poster with the enormous"
" face gazed from the wall. It was one of those pictures which are so contrived that the eyes follow"
" you about when you move. BIG BROTHER IS WATCHING YOU, the caption beneath it ran."


@torch.no_grad()
def generate(model, tokenizer, device, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs.to(device)
    start = time.time()
    outputs = model.generate(
        input_ids=inputs.input_ids,
        max_new_tokens=20,
        attention_mask=inputs.attention_mask,
        do_sample=True,
        top_k=50,
        top_p=0.9,
    )
    end = time.time()
    generated_text = tokenizer.decode(outputs[0])
    return generated_text, (end - start)


def timing(model, tokenizer, device, batch_size=1, prompt_length=512, nb_tokens=512, iterations=10):
    def synchronize(device):
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
        else:
            torch.cpu.synchronize()

    def timing_event(device):
        if device.type == "cuda":
            return torch.cuda.Event(enable_timing=True)
        elif device.type == "mps":
            return torch.mps.Event(enable_timing=True)

        class CPUEvent:
            def __init__(self):
                self.time = None

            def record(self):
                self.time = time.time()

            def elapsed_time(self, other):
                assert self.time is not None
                assert other.time is not None
                return (other.time - self.time) * 1000

        return CPUEvent()

    generation_config = GenerationConfig(
        max_new_tokens=nb_tokens,
        min_new_tokens=nb_tokens,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        num_beams=1,
        do_sample=False,
        eos_token_id=None,  # This is required for min_new_tokens to actually have an effect.
    )
    model.generation_config.eos_token_id = None  # greedy_search falls back on this eos_token_id that we need to set to None as well for min_new_tokens to have an effect.

    synchronize(device)

    latencies = []
    input_ids = torch.randint(1, model.config.vocab_size - 1, size=(batch_size, prompt_length)).to(device)
    masks = torch.ones(batch_size, prompt_length, dtype=torch.int32).to(device)

    # mean over 10 batches
    for _ in tqdm(range(iterations)):
        start_event = timing_event(device)
        end_event = timing_event(device)
        synchronize(device)
        start_event.record()

        _ = model.generate(input_ids, attention_mask=masks, generation_config=generation_config)
        end_event.record()
        synchronize(device)

        latency_ms = start_event.elapsed_time(end_event)
        latencies.append(latency_ms)

    return np.mean(latencies) / generation_config.min_new_tokens


def get_device_memory(device):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        return torch.cuda.memory_allocated()
    elif device.type == "mps":
        torch.mps.empty_cache()
        return torch.mps.current_allocated_memory()
    return None


def main():
    parser = argparse.ArgumentParser(description="Generate bechmark")
    parser.add_argument(
        "--model", type=str, default="princeton-nlp/Sheared-LLaMA-1.3B", help="The model to use for benchmark"
    )
    parser.add_argument("--device", type=str, default=None, help="The device to use for benchmark.")
    parser.add_argument("--compile", action='store_true', help="Compile the model for improved performance.")
    parser.add_argument("--it", type=int, default=10, help="The number of benchmark iterations")
    parser.add_argument(
        "--quantization",
        type=str,
        default="none",
        choices=["w8a16", "w8a8"],
        help="One of none, w8a16, w8a8.",
    )
    args = parser.parse_args()
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    dtype = torch.float32 if device.type == "cpu" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device)

    if args.quantization != "none":
        print("quantizing")
        start = time.time()
        if args.quantization == "w8a16":
            quant_api.change_linear_weights_to_int8_dqtensors(model)
        elif args.quantization == "w8a8":
            quant_api.change_linear_weights_to_int8_woqtensors(model)
        print(f"Finished: {time.time()-start:.2f}")

    if args.compile:
        torch._inductor.config.use_mixed_mm = True
        model = torch.compile(model, mode='max-autotune')

    memory = get_device_memory(device)
    if memory is not None:
        print(f"Device memory: {memory / (2 ** 30):.4f} GB")

    prompt = "One of my fondest memory is"
    output, latency = generate(model, tokenizer, device, prompt)
    print(f"Sample generation for sanity check: '{output}' in [{latency:.2f} s]")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    mean_latency = timing(model, tokenizer, device, batch_size=1, prompt_length=512, nb_tokens=512, iterations=args.it)
    print(f"\nLatency per token: {mean_latency:.3f} ms")
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated()
        print(f"Peak memory during benchmark: {peak_memory / (2 ** 30):.4f} GB")


if __name__ == "__main__":
    main()
