# MMLU Reasoning Length Measurement

This project measures the reasoning length of language models on the abstract math category of MMLU (Massive Multitask Language Understanding).

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Test different measuring components:

```bash
# Test a small model (~160M parameters)
python test_small_model.py

# Test the reasoning extraction functionality
python test_reasoning_extraction.py
```

3. Run the main scripts:

```bash
# General script (works with any model)
python mmlu_reasoning_length.py

# Specialized script for deepseek r1 llama 8b distill
python measure_deepseek_reasoning.py
```

## Files

- `mmlu_reasoning_length.py` - General script that measures reasoning length on MMLU
- `measure_deepseek_reasoning.py` - Specialized script for deepseek r1 llama 8b distill model
- `test_small_model.py` - Test script for small models (under 200M parameters)
- `test_reasoning_extraction.py` - Test script for the reasoning extraction functionality
- `requirements.txt` - Dependencies required to run the code

## Reasoning Extraction

This project extracts reasoning length using pattern-based delimiters to identify:
1. Where the reasoning chain starts (typically after phrases like "Let me think step by step")
2. Where the reasoning chain ends (typically before phrases like "Therefore, the answer is")

The extraction gives both word count and token count metrics for the reasoning portion only.

## Customization

- To use a different model, modify the `model_name` variable in the respective script
- Adjust `sample_size` to process more or fewer questions 
- Results are saved to CSV files for further analysis