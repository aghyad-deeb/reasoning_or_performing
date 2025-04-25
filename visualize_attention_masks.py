import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def visualize_attention_masks(csv_file, output_dir="attention_visualizations"):
    """
    Visualize attention masks for incorrect answers in the CSV file.
    
    Args:
        csv_file: Path to the CSV file containing the data
        output_dir: Directory to save the visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    print(f"Reading file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # Filter for incorrect answers
    incorrect_df = df[df["is_correct"] == False]
    print(f"Found {len(incorrect_df)} incorrect answers")
    
    if len(incorrect_df) == 0:
        print("No incorrect answers found.")
        return
    
    # Create a custom colormap from white to blue
    cmap = LinearSegmentedColormap.from_list("custom_blue", ["#ffffff", "#0343df"])
    
    # Process each incorrect answer
    for idx, row in incorrect_df.iterrows():
        try:
            # Extract attention mask (assuming it's stored as a string representation of a list)
            attention_mask_str = row["attention_mask"]
            
            # Parse the attention mask
            # If it's a string representation of a list, convert it to an actual list
            if isinstance(attention_mask_str, str):
                attention_mask = ast.literal_eval(attention_mask_str)
            else:
                attention_mask = attention_mask_str
                
            # If attention_mask is a list with one element (a nested list)
            if isinstance(attention_mask, list) and len(attention_mask) == 1:
                attention_mask = attention_mask[0]
            
            # Convert to numpy array if it's not already
            attention_mask = np.array(attention_mask)
            
            # Get model and correct answers
            model_answer = row.get("model_answer", "Unknown")
            correct_answer = row.get("correct_answer", "Unknown")
            
            # Create the visualization
            plt.figure(figsize=(12, 2))
            sns.heatmap(
                attention_mask.reshape(1, -1), 
                cmap=cmap,
                cbar=True,
                xticklabels=False,
                yticklabels=False,
                vmin=0, 
                vmax=1
            )
            
            # Set title with model answer and correct answer
            plt.title(f"Incorrect Answer: {model_answer} (should be {correct_answer})")
            
            # Add x-axis label explaining that it represents token positions
            plt.xlabel("Token Positions")
            
            # Save the visualization
            file_name = f"attention_mask_idx_{idx}.png"
            plt.savefig(os.path.join(output_dir, file_name), bbox_inches="tight")
            plt.close()
            
            print(f"Saved visualization for index {idx} to {file_name}")
            
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
    
    print(f"Saved {len(incorrect_df)} attention mask visualizations to {output_dir}")
    
    # Create a summary visualization with all attention masks stacked
    try:
        print("Creating summary visualization...")
        all_masks = []
        labels = []
        
        for idx, row in incorrect_df.iterrows():
            try:
                # Extract and parse attention mask
                attention_mask_str = row["attention_mask"]
                if isinstance(attention_mask_str, str):
                    attention_mask = ast.literal_eval(attention_mask_str)
                else:
                    attention_mask = attention_mask_str
                    
                # Handle nested list case
                if isinstance(attention_mask, list) and len(attention_mask) == 1:
                    attention_mask = attention_mask[0]
                
                # Convert to numpy array and reshape
                mask_array = np.array(attention_mask)
                all_masks.append(mask_array)
                
                # Create label with model and correct answers
                model_answer = row.get("model_answer", "Unknown")
                correct_answer = row.get("correct_answer", "Unknown")
                labels.append(f"{model_answer}→{correct_answer}")
                
            except Exception as e:
                print(f"Error adding mask for row {idx} to summary: {str(e)}")
        
        if all_masks:
            # Ensure all masks are the same length by padding with zeros
            max_length = max(len(mask) for mask in all_masks)
            padded_masks = []
            for mask in all_masks:
                if len(mask) < max_length:
                    padded = np.pad(mask, (0, max_length - len(mask)), 'constant')
                    padded_masks.append(padded)
                else:
                    padded_masks.append(mask)
            
            # Stack masks for visualization
            stacked_masks = np.vstack(padded_masks)
            
            # Create the visualization
            plt.figure(figsize=(14, len(all_masks) * 0.5 + 2))
            ax = sns.heatmap(
                stacked_masks, 
                cmap=cmap,
                cbar=True,
                xticklabels=False,
                yticklabels=labels,
                vmin=0, 
                vmax=1
            )
            
            # Improve y-tick labels appearance
            plt.yticks(rotation=0)
            ax.set_title("Attention Masks for All Incorrect Answers")
            plt.xlabel("Token Positions")
            
            # Save the summary visualization
            summary_file = "attention_masks_summary.png"
            plt.savefig(os.path.join(output_dir, summary_file), bbox_inches="tight")
            plt.close()
            
            print(f"Saved summary visualization to {summary_file}")
        else:
            print("Could not create summary visualization - no valid masks processed")
            
    except Exception as e:
        print(f"Error creating summary visualization: {str(e)}")

if __name__ == "__main__":
    # Set the directory and file
    directory = "masking_output"
    file = "entry_76_tokens_4512_response_2025-04-21_00-31-27.csv"
    
    # Full path to the CSV file
    csv_file = os.path.join(directory, file)
    
    # Visualize attention masks
    visualize_attention_masks(csv_file) 