import pandas as pd
import argparse
from pathlib import Path
import os

def save_analysis(csv_path):
    """
    Analyze the MMLU reasoning CSV file and save response and reasoning text
    to separate files in an analysis directory, organized by token count.
    Only saves top 5 entries that are both valid and correct.
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Filter for valid and correct responses
    df = df[(df['valid'] == True) & (df['is_correct'] == True)]
    
    # Sort by reasoning token count and take top 5
    df = df.sort_values('reasoning_token_count', ascending=False).head(5)
    
    # Create analysis directory
    analysis_dir = Path("analysis")
    analysis_dir.mkdir(exist_ok=True)
    
    # Create subdirectories for responses and reasoning
    responses_dir = analysis_dir / "responses"
    reasoning_dir = analysis_dir / "reasoning"
    responses_dir.mkdir(exist_ok=True)
    reasoning_dir.mkdir(exist_ok=True)
    
    # Create summary file
    summary_file = analysis_dir / "summary.txt"
    with open(summary_file, 'w') as f:
        f.write("Analysis of MMLU Reasoning Lengths (Top 5 Valid & Correct Responses)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total entries analyzed: {len(df)}\n")
        f.write(f"Average token count: {df['reasoning_token_count'].mean():.2f}\n")
        f.write(f"Median token count: {df['reasoning_token_count'].median():.2f}\n")
        f.write(f"Min token count: {df['reasoning_token_count'].min()}\n")
        f.write(f"Max token count: {df['reasoning_token_count'].max()}\n\n")
    
    # Save individual files
    for idx, row in df.iterrows():
        # Create filename with token count and index
        base_filename = f"entry_{idx+1}_tokens_{row['reasoning_token_count']}"
        
        # Save response
        response_file = responses_dir / f"{base_filename}_response.txt"
        with open(response_file, 'w') as f:
            f.write(row['response'])
        
        # Save reasoning
        reasoning_file = reasoning_dir / f"{base_filename}_reasoning.txt"
        with open(reasoning_file, 'w') as f:
            f.write(row['reasoning_text'])
        
        # Add to summary
        with open(summary_file, 'a') as f:
            f.write(f"\nEntry {idx + 1}:\n")
            f.write(f"Question ID: {row['question_id']}\n")
            f.write(f"Token Count: {row['reasoning_token_count']}\n")
            f.write(f"Response saved to: {response_file}\n")
            f.write(f"Reasoning saved to: {reasoning_file}\n")
            f.write("-" * 40 + "\n")
    
    print(f"\nAnalysis complete! Files saved in '{analysis_dir}' directory")
    print(f"Summary statistics saved to: {summary_file}")

def main():
    csv_path = "output/mmlu_abstract_algebra_2025-04-17_23-40-41.csv"
    # Check if file exists
    if not Path(csv_path).exists():
        print(f"Error: File {csv_path} does not exist")
        return
    
    save_analysis(csv_path)

if __name__ == "__main__":
    main() 