import os
import pandas as pd
import glob
import datetime
import matplotlib.pyplot as plt
import numpy as np

def analyze_csv_files(directory_path, save_results=True, create_visualizations=True):
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(directory_path, "*.csv"))
    
    total_answers = 0
    total_incorrect = 0
    files_analyzed = 0
    file_results = []
    error_types = {}
    
    print(f"Found {len(csv_files)} CSV files in {directory_path}")
    
    for file_path in csv_files:
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Check if the file has the expected structure
            if 'is_correct' in df.columns:
                # Count total and incorrect answers
                file_total = len(df)
                file_incorrect = len(df[~df['is_correct']])
                
                # Calculate error rate
                error_rate = file_incorrect / file_total * 100 if file_total > 0 else 0
                
                # If model_answer and correct_answer columns exist, analyze error types
                if 'model_answer' in df.columns and 'correct_answer' in df.columns:
                    incorrect_df = df[~df['is_correct']]
                    for _, row in incorrect_df.iterrows():
                        model_ans = str(row['model_answer']).strip() if not pd.isna(row['model_answer']) else "N/A"
                        correct_ans = str(row['correct_answer']).strip() if not pd.isna(row['correct_answer']) else "N/A"
                        error_key = f"{model_ans} (should be {correct_ans})"
                        error_types[error_key] = error_types.get(error_key, 0) + 1
                
                file_results.append({
                    'file': os.path.basename(file_path),
                    'total': file_total,
                    'incorrect': file_incorrect,
                    'error_rate': error_rate
                })
                
                total_answers += file_total
                total_incorrect += file_incorrect
                files_analyzed += 1
                
                print(f"Analyzed {os.path.basename(file_path)}: {file_incorrect}/{file_total} wrong ({error_rate:.2f}%)")
            else:
                print(f"Skipping {os.path.basename(file_path)}: missing 'is_correct' column")
        
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {str(e)}")
    
    # Calculate overall results
    if total_answers > 0:
        overall_error_rate = total_incorrect / total_answers * 100
        print("\nOverall Results:")
        print(f"Files analyzed: {files_analyzed}/{len(csv_files)}")
        print(f"Total answers: {total_answers}")
        print(f"Total incorrect: {total_incorrect}")
        print(f"Overall error rate: {overall_error_rate:.2f}%")
        
        # Generate more detailed reports
        print("\nDetailed Results by File (sorted by error rate):")
        for result in sorted(file_results, key=lambda x: x['error_rate'], reverse=True):
            print(f"{result['file']}: {result['incorrect']}/{result['total']} wrong ({result['error_rate']:.2f}%)")
        
        # Show common error types
        if error_types:
            print("\nCommon Error Types (Top 10):")
            for error, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"{error}: {count} occurrences")
        
        # Create visualizations if requested
        if create_visualizations:
            create_analysis_visualizations(file_results, error_types, total_answers, total_incorrect)
        
        # Save results to CSV if requested
        if save_results:
            save_analysis_results(file_results, error_types, total_answers, total_incorrect, overall_error_rate)
    else:
        print("No valid data found in any of the CSV files.")

def create_analysis_visualizations(file_results, error_types, total_answers, total_incorrect):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs('analysis_visualizations', exist_ok=True)
    
    # 1. Error rate by file (bar chart)
    plt.figure(figsize=(10, 6))
    sorted_results = sorted(file_results, key=lambda x: x['error_rate'], reverse=True)
    files = [result['file'] for result in sorted_results]
    error_rates = [result['error_rate'] for result in sorted_results]
    
    plt.bar(range(len(files)), error_rates, color='salmon')
    plt.xlabel('File')
    plt.ylabel('Error Rate (%)')
    plt.title('Error Rate by File')
    plt.xticks(range(len(files)), files, rotation=90)
    plt.tight_layout()
    plt.savefig(f'analysis_visualizations/error_rate_by_file_{timestamp}.png')
    plt.close()
    
    # 2. Correct vs Incorrect answers (pie chart)
    plt.figure(figsize=(8, 8))
    labels = ['Correct', 'Incorrect']
    sizes = [total_answers - total_incorrect, total_incorrect]
    colors = ['lightgreen', 'salmon']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Overall Accuracy')
    plt.savefig(f'analysis_visualizations/overall_accuracy_{timestamp}.png')
    plt.close()
    
    # 3. Top error types (horizontal bar chart)
    if error_types:
        plt.figure(figsize=(12, 8))
        sorted_errors = sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:10]
        error_labels = [error[:30] + '...' if len(error) > 30 else error for error, _ in sorted_errors]
        error_counts = [count for _, count in sorted_errors]
        
        y_pos = np.arange(len(error_labels))
        plt.barh(y_pos, error_counts, color='skyblue')
        plt.yticks(y_pos, error_labels)
        plt.xlabel('Count')
        plt.title('Top 10 Error Types')
        plt.tight_layout()
        plt.savefig(f'analysis_visualizations/top_error_types_{timestamp}.png')
        plt.close()
    
    print(f"\nVisualizations saved to the 'analysis_visualizations' directory")

def save_analysis_results(file_results, error_types, total_answers, total_incorrect, overall_error_rate):
    # Create a timestamp for the results file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Save file-level results
    file_df = pd.DataFrame(file_results)
    file_df = file_df.sort_values('error_rate', ascending=False)
    file_results_path = f"analysis_results_by_file_{timestamp}.csv"
    file_df.to_csv(file_results_path, index=False)
    
    # Save error type results
    if error_types:
        error_data = [{"error_type": error, "count": count} for error, count in error_types.items()]
        error_df = pd.DataFrame(error_data)
        error_df = error_df.sort_values('count', ascending=False)
        error_types_path = f"analysis_results_error_types_{timestamp}.csv"
        error_df.to_csv(error_types_path, index=False)
    
    # Save summary results
    summary_data = {
        "total_answers": [total_answers],
        "total_incorrect": [total_incorrect],
        "overall_error_rate": [overall_error_rate]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_path = f"analysis_results_summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\nResults saved to:")
    print(f"- {file_results_path}")
    if error_types:
        print(f"- {error_types_path}")
    print(f"- {summary_path}")

if __name__ == "__main__":
    # Path to the directory containing the CSV files
    directory_path = "masking_output"
    analyze_csv_files(directory_path) 