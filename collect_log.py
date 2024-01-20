import re
from collections import defaultdict
import argparse
import numpy as np

# Create an argument parser
parser = argparse.ArgumentParser(description="Process log file and extract F1 scores.")

# Add a required argument for the input file
parser.add_argument("--input_file", help="Path to the log file")

# Parse the command-line arguments
args = parser.parse_args()

def own_log():
    with open(args.input_file) as f:
        data = f.read().splitlines()

    result = defaultdict(float)
    result_list = defaultdict(list)
    fold = "Fold 0 Party 0"
    party = 0
    for row in data:
        pattern = r"Fold (.*?)$"
        match = re.search(pattern, row)
        if match:
            fold = match.group(0) + " " + f"Party {party}"

        pattern = r"best test f1 (\d+\.\d+)"

        match = re.search(pattern, row)

        if match:
            f1_score = float(match.group(1))

            if (result[fold] < f1_score):
                result[fold] = f1_score
        
        if "-------------------------------------------------" in row:
            party += 1

    for key, value in dict(result).items():
        print(key + " :", value)
        
    
    for key, value in dict(result).items():
        result_list[key.split(" ")[-1]].append(value)
    
    for key, value in dict(result_list).items():
        print(key + ":")
        print(np.mean(value))
    
    

def origin_log():
    
    f1_dict = defaultdict(float)
    
    with open(args.input_file) as f:
        data = f.read().splitlines()

    pattern = r'.*?\bf1 for active party\b.*?[.!?]'
    
    score_pattern = r'\b\d+\.\d+\b'

    # Use re.findall to collect all matching text
    for text in data:
        if re.search(pattern, text, re.IGNORECASE):
            list_f1 = text.split(":")[1]
            list_f1 = list_f1.replace("[", "").replace("]", "")
            list_f1 = list_f1.split(",")
            list_f1 = [float(i) for i in list_f1]
            mean_f1 = sum(list_f1)/len(list_f1)
            float_numbers = re.findall(score_pattern, text.split(":")[0])
            # Convert the extracted strings to float values
            float_values = [float(num) for num in float_numbers][-1]
            f1_dict[float_values] = mean_f1
            print(text)
            print("mean f1: ", mean_f1)
    sorted_dict = dict(sorted(f1_dict.items()))
    print("fed f1: ", sorted_dict)
    print("combine f1: ", "0.9481608219329713")
    
def uci_log():
        
    with open(args.input_file) as f:
        data = f.read().splitlines()
    
    pattern = re.compile(r".*F1 mean=\d+(\.\d+)?,.*$")

    for text in data:
        if pattern.match(text):
            print(text)
if __name__ == "__main__": 
#     origin_log()
#     own_log()
    uci_log()