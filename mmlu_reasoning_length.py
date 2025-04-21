
#? -what temprature should I use
import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from dotenv import load_dotenv
from nnsight import CONFIG, LanguageModel
import os
import pandas as pd
import re

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

def load_mmlu_abstract_math():
    """
    Load the abstract math category from MMLU dataset
    """
    print("Loading MMLU abstract math dataset")
    dataset = load_dataset("cais/mmlu", "abstract_algebra")
    return dataset["test"]

def create_user_prompt(question, choices):
    """
    Create a prompt for the model that encourages reasoning,
    tailored for deepseek r1 llama 8b distill model
    """
    options = ["A", "B", "C", "D"]
    formatted_choices = "\n".join([f"{opt}. {choices[i]}" for i, opt in enumerate(options)])
    
    # Using a prompt template that works well with deepseek models
    prompt = f"""Question: {question}

Options:
{formatted_choices}
Output only the answer (A or B or C or D) with nothing else after the </think> token
"""
    return prompt, options

def get_thinking_words(model) :
    # temp = [{"role": "assistant", "content": ""}, {"role": "user", "content": ""}, {"role"}]
    # assistant = model.tokenizer.apply_chat_template(temp, tokenize=False, add_generation_prompt=True)
    # print(assistant)
    think = "<think>\n"
    end_of_think = "</think>\n"
    return think, end_of_think

def extract_reasoning_length(model, response, tokenizer):
    """
    Measure the length of the reasoning part of the response between 
    specific reasoning start and end markers for the deepseek r1 llama 8b distill model
    """
    # Start patterns to identify where reasoning begins
    think, end_of_think = get_thinking_words(model)
    start_patterns = [think]
    
    # End patterns that signal the end of reasoning and start of the answer
    end_patterns = [end_of_think]
    
    extraction_method = "full"
    start_idx = 0
    end_idx = len(response)
    
    # Try to find the start of reasoning
    for pattern in start_patterns:
        start_match = re.search(pattern, response, re.IGNORECASE)
        if start_match:
            start_idx = start_match.start()
            extraction_method = "start_pattern"
            break
    
    # Try to find the end of reasoning
    for pattern in end_patterns:
        end_match = re.search(pattern, response, re.IGNORECASE)
        if end_match:
            end_idx = end_match.end()  # Changed from start() to end() to include the end token
            extraction_method = "start_end_pattern" if extraction_method == "start_pattern" else "end_pattern"
            break
    
    valid =  extraction_method == "start_end_pattern"
    
    # Extract the reasoning part
    reasoning = response[start_idx:end_idx].strip()
    
    # Count words in the reasoning
    word_count = len(reasoning.split())
    
    # Count tokens in the reasoning using the model's tokenizer
    tokens = tokenizer.encode(reasoning, add_special_tokens=False)
    token_count = len(tokens)
    
    print(f"Extracted reasoning: {word_count} words / {token_count} tokens (method: {extraction_method})")
    
    # Return a dictionary with both word and token counts
    return {
        "word_count": word_count,
        "token_count": token_count,
        "reasoning_text": reasoning,
        "extraction_method": extraction_method,
        "valid": valid,
    }

def generate_model_response(model, tokenizer, prompt, max_new_tokens=5000, temperature=0.7):
    """
    Generate a response from the model given a prompt
    """
    template = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(template, add_generation_prompt=True, return_tensors="pt")
    # input_ids = inputs["input_ids"]
    print(f"{inputs}")
    # print(f"{tokenizer.decode(inputs["input_ids"])=}")
    # print(f"{inputs["input_ids"]=}")

    with model.generate(inputs, remote=remote, max_new_tokens=max_new_tokens):
    # with model.generate(input="test", remote=remote, max_new_tokens=max_new_tokens):
        outputs = model.generator.output.save()

    print(f"{outputs=}")
    print(f"{outputs.shape=}")
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    return response

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
    
    # Load MMLU abstract math dataset
    dataset = load_mmlu_abstract_math()
    
    # Set up results storage
    results = []
    from datetime import datetime
    import pytz
    
    # Get current date and time in Boston
    boston_tz = pytz.timezone('America/New_York')
    current_time = datetime.now(boston_tz)
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create filename with dataset info and timestamp
    dataset_name = "mmlu"
    category = "abstract_algebra"
    results_file = f"output/{dataset_name}_{category}_{time_str}.csv"
    
    # Process a subset for testing (adjust as needed)
    sample_size = min(1000, len(dataset))
    
    print(f"Processing {sample_size} questions with {model_name}...")
    
    try:
        for i in range(sample_size):
            question = dataset[i]["question"]
            choices = [dataset[i]["choices"][j] for j in range(4)]
            correct_answer_idx = dataset[i]["answer"]
            
            # Create prompt
            prompt, options = create_user_prompt(question, choices)
            correct_answer = options[correct_answer_idx]
            
            # Generate response using the new function
            response = generate_model_response(model, tokenizer, prompt)
            print(f"\n\n\n{response=}")
            
            # Measure reasoning length
            reasoning_data = extract_reasoning_length(model, response, tokenizer)
            
            # Extract model's answer
            model_answer = extract_model_answer(response)
            
            # Determine if answer is correct
            is_correct = model_answer == correct_answer if model_answer is not None and reasoning_data["valid"] else False
            
            # Store results for this question
            result = {
                "question_id": i,
                "question": question,
                "response": response,
                "reasoning_text": reasoning_data["reasoning_text"],
                "reasoning_word_count": reasoning_data["word_count"],
                "reasoning_token_count": reasoning_data["token_count"],
                "extraction_method": reasoning_data["extraction_method"],
                "valid": reasoning_data["valid"],
                "model_answer": model_answer,
                "correct_answer": correct_answer,
                "is_correct": is_correct
            }
            
            # Add to results and save immediately
            results.append(result)
            save_results(results, results_file)
            
            print(f"Question {i+1}/{sample_size}: Reasoning length = {reasoning_data['word_count']} words / {reasoning_data['token_count']} tokens")
            print(f"Model answer: {model_answer}, Correct answer: {correct_answer}, Correct: {is_correct}")
        
        # Print final summary statistics
        df = pd.DataFrame(results)
        print("\nSummary Statistics for Reasoning Length:")
        print(f"Mean Word Count: {df['reasoning_word_count'].mean():.2f} words")
        print(f"Median Word Count: {df['reasoning_word_count'].median():.2f} words")
        print(f"Min Word Count: {df['reasoning_word_count'].min()} words")
        print(f"Max Word Count: {df['reasoning_word_count'].max()} words")
        
        print("\nSummary Statistics for Token Count:")
        print(f"Mean Token Count: {df['reasoning_token_count'].mean():.2f} tokens")
        print(f"Median Token Count: {df['reasoning_token_count'].median():.2f} tokens")
        print(f"Min Token Count: {df['reasoning_token_count'].min()} tokens")
        print(f"Max Token Count: {df['reasoning_token_count'].max()} tokens")
        
        print("\nAccuracy Statistics:")
        print(f"Total Questions: {len(df)}")
        print(f"Correct Answers: {df['is_correct'].sum()}")
        print(f"Accuracy: {(df['is_correct'].sum() / len(df)) * 100:.2f}%")
    
    except KeyboardInterrupt:
        print("\nScript interrupted. Saving current results...")
        save_results(results, results_file)
        print("Results saved. Exiting...")
    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Saving current results...")
        save_results(results, results_file)
        print("Results saved. Exiting...")
        raise

if __name__ == "__main__":
    main()