import re
from collections import defaultdict
import argparse
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
import heapq


# Create an argument parser
parser = argparse.ArgumentParser(description="Process log file and extract F1 scores.")

# Add a required argument for the input file
parser.add_argument("--input_file", help="Path to the log file")
parser.add_argument("--input_folder", help="Path to the input folder")

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

#     pattern = re.compile(r".*Accuracy mean=\d+(\.\d+)?,.*$")
    
    res = {}
    for text in data:
        if pattern.match(text):
            print(text)
#             key = re.findall(r"mean=([0-9]+\.[0-9]+)", text)[0]
#             res[key] =  text
#     largest_f1 = heapq.nlargest(10, res)
#     for key in largest_f1:
#         print(key, res[key])

def plot_ucilog():

    result = {}
    ## get full path of all files in the folder
    mypath = args.input_folder
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    print(onlyfiles)

    for file in onlyfiles:
        result_file = {}
        with open(mypath + "/" + file) as f:
            data = f.read().splitlines()
        pattern = re.compile(r".*F1 mean=\d+(\.\d+)?,.*$")
        for text in data:
            if pattern.match(text):
                ## get the final float number before the colon as the key
                key = re.findall(r"(\d+\.\d+)[,:]", text)[0]
                value = re.findall(r"mean=([0-9]+\.[0-9]+)", text)[0]
                result_file[float(key)] = float(value)
        result[file.split(".")[0]] = result_file
    print(result)
    plot(result)

def plot(overall_result):
    for key, value in overall_result.items():
        dict = sorted(value.items(), key=lambda x: x[0], reverse=True)
        ratios, f1_means = zip(*dict)
        plt.plot(ratios, f1_means, marker='o', linestyle='-', label=key)
        # Thiết lập các thuộc tính của biểu đồ
        plt.xlabel('Ratio (Descending)')
        plt.ylabel('F1 Mean')
        plt.title('Comparison of F1 Mean - Descending Ratios')
        plt.legend()
        plt.grid(True)

    # Hiển thị biểu đồ
    plt.savefig('comparison_chart.png')

if __name__ == "__main__": 
#     origin_log()
#     own_log()
    uci_log()
#     plot_ucilog()