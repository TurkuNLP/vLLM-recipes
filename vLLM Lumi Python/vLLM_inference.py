from vllm import LLM, SamplingParams

def generate_text(llm, prompt):
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.9,
        top_k=50,
        max_tokens=200,
    )

    # Generate text
    outputs = llm.generate([prompt], sampling_params)
    response = outputs[0].outputs[0].text.strip()
    return response

def main():
    # Input everything you need
    prompt = "Once upon a time" #Adjust your prompt
    model_id = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF" #choose your model
    cache_dir = "/scratch/project_462000642/joonatan/shared_models" #choose your cache directory
    tensor_parallel_size = 8  # Adjust based on available GPUs 

    # Initialize the model with specified parameters
    llm = LLM(
        model=model_id,
        download_dir=cache_dir,
        tensor_parallel_size=tensor_parallel_size,
    )

    # Generate text
    result = generate_text(llm, prompt)
    print("\nGenerated Text:\n")
    print(result)

if __name__ == '__main__':
    main()
