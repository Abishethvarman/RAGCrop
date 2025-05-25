import csv
import subprocess
import random
from collections import defaultdict

def invoke_llm(prompt):
    """Invoke LLaMA 3.1 locally and return the generated response."""
    try:
        result = subprocess.run(
            ['ollama', 'run', 'llama3.1:8b', prompt],
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except Exception as e:
        return f"Error invoking LLM: {e}"

def generate_txt_from_csv(csv_file_path, output_txt_file):
    """Reads a CSV file, selects 5 examples per category, and generates structured agricultural recommendations."""
    
    # Read CSV and categorize data
    category_dict = defaultdict(list)
    
    with open(csv_file_path, 'r', newline='', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            category_dict[row["label"]].append(row)
    
    # Select up to 5 examples per category
    selected_data = []
    for category, rows in category_dict.items():
        selected_data.extend(random.sample(rows, min(5, len(rows))))

    # Process and generate recommendations
    with open(output_txt_file, 'w', encoding='utf-8') as txt_file:
        for row in selected_data:
            input_features = {
                "N": row["N"],
                "P": row["P"],
                "K": row["K"],
                "pH": row["ph"],
                "Rainfall": row["rainfall"],
                "Humidity": row["humidity"],
                "Temperature": row["temperature"]
            }
            label = row["label"]  # Primary crop recommendation

            # Refined prompt for LLaMA 3.1
            llm_prompt = (
                "You are an assistant for an agricultural application. "
                "Based on the given soil and weather conditions, generate structured farming suggestions. "
                "Provide only relevant details in a key-value format.\n\n"
                "### Input Features ###\n"
                f"- Nitrogen: {input_features['N']} ppm\n"
                f"- Phosphorus: {input_features['P']} ppm\n"
                f"- Potassium: {input_features['K']} ppm\n"
                f"- pH Level: {input_features['pH']}\n"
                f"- Rainfall: {input_features['Rainfall']} mm\n"
                f"- Humidity: {input_features['Humidity']}%\n"
                f"- Temperature: {input_features['Temperature']}°C\n\n"
                "### Generate Suggestions ###\n"
                f"Primary Crop: {label}\n"
                "Provide:\n"
                "- Secondary Crop\n"
                "- Best Irrigation Method\n"
                "- Fertilizer Type & Dosage\n"
                "- Essential Care Tips\n\n"
                "Output in key-value pairs."
            )

            # Invoke LLaMA 3.1
            llm_response = invoke_llm(llm_prompt)

            # Write to text file in key-value format
            txt_file.write(f"Primary Crop: {label}\n")
            txt_file.write(f"Input Conditions: {input_features}\n")
            txt_file.write(f"LLaMA Suggestions:\n{llm_response}\n")
            txt_file.write("=" * 80 + "\n\n")  # Separator for readability

    print(f"Processed and saved structured recommendations to {output_txt_file}")

if __name__ == "__main__":
    csv_file_path = "crop_recommendation.csv"  # Ensure this CSV file exists
    output_txt_file = "crop_suggestions_grownius.txt"

    generate_txt_from_csv(csv_file_path, output_txt_file)