from src.model import DecentralizedLLM
from src.grammar import LonScriptGrammar

def main():
    # Initialize the model
    model = DecentralizedLLM("gpt2")  # Using GPT-2 as base model, can be changed
    grammar = LonScriptGrammar()
    
    # Example usage
    input_text = "Analyze the impact of renewable energy on climate change"
    context = "Current global climate trends and renewable energy adoption rates"
    
    # Get model response with deep reasoning
    response = model.reason(context, input_text)
    
    # Apply LonScript grammar for enhanced understanding
    enhanced_response = grammar.apply_grammar_rules(response)
    
    print("Enhanced Response:", enhanced_response)

if __name__ == "__main__":
    main()