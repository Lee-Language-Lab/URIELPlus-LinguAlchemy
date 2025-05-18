# Computes average metrics across languages in the MASAKHANEWS, MASSIVE, SEMREL datasets experiments

import os
import json
import csv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "outputs"))

# Store results: list of tuples (dataset, model, file, avg_accuracy)
results = []

for dataset in os.listdir(OUTPUTS_DIR):
    dataset_path = os.path.join(OUTPUTS_DIR, dataset)
    if not os.path.isdir(dataset_path):
        continue

    for model in os.listdir(dataset_path):
        model_path = os.path.join(dataset_path, model)
        if not os.path.isdir(model_path):
            continue

        for filename in os.listdir(model_path):
            if filename.endswith('.json'):
                file_path = os.path.join(model_path, filename)

                # Load JSON and extract accuracy
                with open(file_path, 'r') as f:
                    try:
                        data = json.load(f)
                        if dataset == "massive" or dataset == "masakhanews":
                            metric = "accuracy"
                        elif dataset == "semrel":
                            metric = "pearson"
                        metrics = [lang_data[metric] for lang_data in data.values()]
                        avg_metrics = sum(metrics) / len(metrics)
                        results.append((dataset, model, filename, avg_metrics))
                    except Exception as e:
                        print(f"Error in file {file_path}: {e}")

# Sort for readability
results.sort()

# Print as table
print(f"{'Dataset':<15} {'Model':<30} {'File':<15} {'Avg Metrics':<15}")
print('-' * 80)
for dataset, model, filename, accuracy in results:
    print(f"{dataset:<15} {model:<30} {filename:<15} {accuracy:<15.4f}")

# Save table to CSV in outputs folder
output_csv_path = os.path.join(OUTPUTS_DIR, "metrics_table.csv")
with open(output_csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Dataset', 'Model', 'File', 'Avg Metrics'])
    writer.writerows(results)

print(f"\nMetrics table saved to: {output_csv_path}")