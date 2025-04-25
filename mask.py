

#! Notes: The model include it's last answer in the last part of the reasoning
#! so masking that may be important
import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from dotenv import load_dotenv
from nnsight import CONFIG, LanguageModel
import os
import pandas as pd
import re
import glob
import os.path

END_OF_THINK_TOKEN = 128014

remote = False
device = ""


def load_model_and_tokenizer(model_name=""):
    global remote, device
    """
    Load model and tokenizer from HuggingFace
    """
    if model_name == "":
        model_id, remote, device = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B", True, "auto"
    else: 
        model_id, remote, device = model_name, True, "auto"
    print(f"Loading model and tokenizer: {model_name}")
    CONFIG.API.APIKEY = os.getenv("NDIF_KEY")
    
    model = LanguageModel(model_id, device_map=device, trust_remote_code=True)
    device = model.device
    print(f"Using device: {device}")
    
    model = model
    tokenizer = model.tokenizer
        
    return model, tokenizer
def mask_attention_between_tokens(input_tokens, attention_mask, start_token, end_token, start_token_instance=0, end_token_instance=-1, start_offset=0, end_offset=0):
    """
    Masks attention values between specified start and end tokens, with optional offsets.
    
    Args:
        input_tokens: Tensor of input token IDs (must be 1D, no batch dimension)
        attention_mask: Tensor of attention mask values (1 for unmasked, 0 for masked)
        start_token: Token ID or list of token IDs marking the start of the region to mask
        end_token: Token ID or list of token IDs marking the end of the region to mask
        start_token_instance: Which instance of the start token to use (0-based index, default: 0)
        end_token_instance: Which instance of the end token to use (0-based index, default: -1 for last instance)
        start_offset: Number of tokens to offset from the start token (positive = after, negative = before)
        end_offset: Number of tokens to offset from the end token (positive = after, negative = before)
        
    Returns:
        Modified attention mask with values between start and end tokens (plus offsets) set to 0
    """
    # Convert to tensors if they aren't already
    if not isinstance(input_tokens, torch.Tensor):
        input_tokens = torch.tensor(input_tokens)
    if not isinstance(attention_mask, torch.Tensor):
        attention_mask = torch.tensor(attention_mask)
    if isinstance(start_token, list):
        start_token = torch.tensor(start_token)
    if isinstance(end_token, list):
        end_token = torch.tensor(end_token)
    
    # Assert that input is 1D (no batch dimension)
    assert len(input_tokens.shape) == 1, "Input tokens must be 1D (no batch dimension)"
    # Assert that attention mask shape matches input shape
    assert attention_mask.shape == input_tokens.shape, "Attention mask shape must match input tokens shape"
    
    # Find the indices of start and end tokens
    start_indices = (input_tokens == start_token).nonzero()
    end_indices = (input_tokens == end_token).nonzero()
    
    if len(start_indices) == 0 or len(end_indices) == 0:
        raise Exception("Did not find start and end tokens")
    
    # Get the specified instance of start and end tokens
    try:
        start_idx = start_indices[start_token_instance].item()
    except IndexError:
        raise Exception(f"Start token instance {start_token_instance} not found. Found {len(start_indices)} instances.")
    
    try:
        end_idx = end_indices[end_token_instance].item()
    except IndexError:
        raise Exception(f"End token instance {end_token_instance} not found. Found {len(end_indices)} instances.")
    
    # Apply offsets
    start_idx += start_offset
    end_idx += end_offset
    
    # Check if indices are within bounds
    if start_idx < 0 or start_idx >= len(input_tokens):
        raise Exception(f"Start index {start_idx} (after offset) is out of bounds")
    if end_idx < 0 or end_idx >= len(input_tokens):
        raise Exception(f"End index {end_idx} (after offset) is out of bounds")
    
    # Create a copy of the attention mask
    masked_attention = attention_mask.clone()
    
    # Check if start index is after end index
    if start_idx > end_idx:
        raise Exception(f"Start token at position {start_idx=} appears after end token at position {end_idx=}, {start_token=}, {end_token=}, {start_token_instance=}, {end_token_instance=}")
    
    # Set attention values between start and end tokens to 0
    masked_attention[start_idx:end_idx+1] = 0
    
    return masked_attention

def generate_model_response(model, tokenizer, inputs, max_new_tokens=5000):
    """
    Generate a response from the model given a prompt
    """

    with model.generate(inputs, remote=remote, max_new_tokens=max_new_tokens):
    # with model.generate(input="test", remote=remote, max_new_tokens=max_new_tokens):
        outputs = model.generator.output.save()

    print(f"{outputs=}")
    print(f"{outputs.shape=}")
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

def count_instances_of_token(tensor, token):
    """
    Counts the number of times a token appears in a tensor.
    
    Args:
        tensor: Input tensor of token IDs (must be 1D, no batch dimension)
        token: Token ID or list of token IDs to count
        
    Returns:
        Number of occurrences of the token in the tensor
    """
    # Convert to tensors if they aren't already
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor)
    if isinstance(token, list):
        token = torch.tensor(token)
    
    # Assert that input is 1D (no batch dimension)
    assert len(tensor.shape) == 1, "Input tensor must be 1D (no batch dimension)"
    
    # Count occurrences
    if isinstance(token, torch.Tensor) and len(token.shape) > 0:
        # For multi-token sequences, we need to find all matches
        matches = (tensor == token[0])
        for i in range(1, len(token)):
            matches = matches & (tensor[i:] == token[i])
        return matches.sum().item()
    else:
        # For single tokens, just count direct matches
        return (tensor == token).sum().item()


def generate_with_masking(model, tokenizer, original_prompt, token_to_mask, correct_answer, prompt_basename, max_new_tokens=5000):
    inputs = tokenizer(original_prompt, return_tensors="pt")
    inputs_lst = []
    respones_lst = []
    for k, t in inputs.items():
        inputs[k] = t.squeeze(0)
    
    num_iter = count_instances_of_token(inputs["input_ids"], token_to_mask)
    num_instances = 0
    for i in range(num_iter):
        for j in range(i + 1, num_iter):
            num_instances += 1
    print(f"\n\nrunning for {num_instances} instances")

    # add back batch dimension
    for k, t in inputs.items():
        inputs[k] = t.unsqueeze(0)
    
    # Set up results storage
    results = []
    from datetime import datetime
    import pytz
    
    # Get current date and time in Boston
    boston_tz = pytz.timezone('America/New_York')
    current_time = datetime.now(boston_tz)
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create filename with dataset info and timestamp
    results_dir = "masking_output"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{prompt_basename}_{time_str}.csv")
    
    for i in range(num_iter):
        for j in range(i, num_iter):
            curr_inputs = dict()
            # clone and remove batch dimension
            for k, t in inputs.items():
                print(f"{k=}")
                curr_inputs[k] = t.clone().squeeze(0)
            
            # mask everything from `token_to_mask` till the end of the COT
            if i == j:
                curr_inputs["attention_mask"] = mask_attention_between_tokens(
                    curr_inputs["input_ids"],
                    curr_inputs["attention_mask"],
                    token_to_mask,
                    END_OF_THINK_TOKEN,
                    start_token_instance=i,
                    end_token_instance=-1,
                    start_offset=0,
                    end_offset=-1, # include the end of think token
                )
            else:
                curr_inputs["attention_mask"] = mask_attention_between_tokens(
                    curr_inputs["input_ids"],
                    curr_inputs["attention_mask"],
                    token_to_mask,
                    token_to_mask,
                    start_token_instance=i,
                    end_token_instance=j,
                    start_offset=0,
                    end_offset=0
                )

            for k, t in curr_inputs.items():
                curr_inputs[k] = t.unsqueeze(0)
            inputs_lst.append(curr_inputs)
            response = generate_model_response(model, tokenizer, curr_inputs, max_new_tokens=max_new_tokens)
            respones_lst.append(response)
            
            # Extract model's answer
            model_answer = extract_model_answer(response)
            
            # Determine if answer is correct
            is_correct = model_answer == correct_answer if model_answer is not None else False
            
            # Store results for this question
            result = {
                "original_prompt": original_prompt,
                "response": response,
                "model_answer": model_answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct,
                "input_ids": curr_inputs["input_ids"].tolist(),
                "attention_mask": curr_inputs["attention_mask"].tolist(),
                "masking_pair": f"{i}-{j}"
            }
            
            # Add to results and save immediately
            results.append(result)
            save_results(results, results_file)
            
            print(f"Saved response for masking pair {i}-{j}")
    
    return dict(inputs_lst=inputs_lst, respones_lst=respones_lst)
            


def extract_model_answer(response):
    """
    Extract the model's answer from the response by looking for A, B, C, or D
    """
    # Look for the last occurrence of A, B, C, or D in the response
    # This assumes the model's final answer will be the last letter it mentions
    matches = re.findall(r'[A-D]', response)
    if matches:
        return matches[-1]  # Return the last occurrence
    return None

def save_results(results, filename="mmlu_reasoning_lengths.csv"):
    """
    Save results to CSV file
    """
    df = pd.DataFrame(results)
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def main():
    # Load model and tokenizer
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    model, tokenizer = load_model_and_tokenizer()

    # Define default parameters (these might be overridden by prompt file info)
    # correct_answer = "B" # Removed, will be read from file
    token_to_mask = 14524 # Assuming this is constant
    max_new_tokens = 200 # Assuming this is constant

    # Get list of prompt files
    prompt_dir = "prompts"
    prompt_files = glob.glob(os.path.join(prompt_dir, "*.txt"))

    if not prompt_files:
        print(f"No .txt files found in the '{prompt_dir}' directory.")
        return

    # Process each prompt file
    for prompt_file_path in prompt_files:
        print(f"\n--- Processing prompt file: {prompt_file_path} ---")
        
        # Extract basename for filename generation
        prompt_basename = os.path.splitext(os.path.basename(prompt_file_path))[0]
        
        # Read the prompt content
        try:
            with open(prompt_file_path, "r") as f:
                lines = f.readlines()
            
            if not lines:
                print(f"Error: Prompt file {prompt_file_path} is empty.")
                continue

            # Prompt file includes answer in last line, extract answer from the
            # last line
            correct_answer = lines[-1].strip()
            if not correct_answer: # Check if the last line was just whitespace
                 print(f"Error: Last line of {prompt_file_path} is empty or whitespace. Cannot determine correct answer.")
                 continue

            # Use all lines except the last as the prompt
            original_prompt = "".join(lines[:-1])
            if not original_prompt.strip(): # Check if prompt content is empty
                print(f"Warning: Prompt content in {prompt_file_path} (excluding last line) is empty.")
                # Decide if you want to continue or skip
                # continue 

        except Exception as e:
            print(f"Error reading file {prompt_file_path}: {e}")
            continue # Skip to the next file if reading fails
        
        # Generate response using the new function, passing the basename and extracted answer
        try:
            print(f"Using correct answer: '{correct_answer=}', {original_prompt[-20:]=}")
            out = generate_with_masking(model, tokenizer, original_prompt, token_to_mask, correct_answer, prompt_basename, max_new_tokens=max_new_tokens)
            print(f"Finished processing {prompt_file_path}")
        except Exception as e:
            print(f"Error processing prompt from file {prompt_file_path}: {e}")
            # Decide if you want to stop or continue with the next file
            continue 
        
    
if __name__ == "__main__":
    main()