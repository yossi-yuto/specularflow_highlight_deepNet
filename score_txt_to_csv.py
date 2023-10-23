import os
import csv
import re
import pdb
import statistics as st
from argparse import ArgumentParser

import pandas as pd


def parse_arg():
    parser = ArgumentParser()
    parser.add_argument('-input_dir', type=str)
    parser.add_argument('-output_file', type=str, default="output")
    return parser.parse_args()


def txt_to_num(text):
    lines = text.split('\n')
    # Initialize variables to hold the scores
    pmd_fbeta = 0.0
    pmd_mae = 0.0
    refine_fbeta = 0.0
    refine_mae = 0.0

    # Loop through the last 4 lines to extract the scores
    for line in lines[-4:]:
        if "pmd" in line:
            continue
        elif "refine" in line:
            continue
        elif "Fbeta" in line and "MAE" in line:
            parts = line.split(",")
            if "pmd" in lines[lines.index(line) - 1]:
                pmd_fbeta = float(parts[0].split(":")[1].strip())
                pmd_mae = float(parts[1].split(":")[1].strip())
            elif "refine" in lines[lines.index(line) - 1]:
                refine_fbeta = float(parts[0].split(":")[1].strip())
                refine_mae = float(parts[1].split(":")[1].strip())
    return pmd_fbeta, pmd_mae, refine_fbeta, refine_mae


def main(args):
    # ヘッダーを書き込む
    input_dir = args.input_dir    
    output_csv_path = args.output_file + ".csv"
    
    df = pd.DataFrame(columns=['type', 'ex_num' ,'PMD_Fbeta', 'PMD_MAE', 'SSFH_Fbeta', 'SSFH_MAE'])

    for type in os.listdir(input_dir):
        type_path = os.path.join(input_dir, type)

        avg_pmd_fbeta = []
        avg_pmd_mae = []
        avg_ssfh_fbeta = []
        avg_ssfh_mae = []
        
        for i, ex_num in enumerate(os.listdir(type_path)):
            ex_path = os.path.join(type_path, ex_num)
            txtfile_path = ex_path + "/eval.txt"
            if os.path.exists(txtfile_path):
                with open(txtfile_path, 'r') as infile:
                    string = infile.read()
                    pmd_fbeta, pmd_mae, ssfh_fbeta, ssfh_mae = txt_to_num(string)
                    
                    avg_pmd_fbeta.append(pmd_fbeta)
                    avg_pmd_mae.append(pmd_mae)
                    avg_ssfh_fbeta.append(ssfh_fbeta)
                    avg_ssfh_mae.append(ssfh_mae)
                    if i == 0:
                        df.loc[len(df.index)] = [type, ex_num, pmd_fbeta, pmd_mae, ssfh_fbeta, ssfh_mae]
                    else:
                        df.loc[len(df.index)] = ['', ex_num, pmd_fbeta, pmd_mae, ssfh_fbeta, ssfh_mae]
            else:
                print(f"No such a text file: {txtfile_path}")
        df.loc[len(df.index)] = ['', 'Avg', 
                                round(st.mean(avg_pmd_fbeta), 3),
                                round(st.mean(avg_pmd_mae), 3),
                                round(st.mean(avg_ssfh_fbeta), 3),
                                round(st.mean(avg_ssfh_mae), 3)]
        df.loc[len(df.index)] = ['', 'Std', st.stdev(avg_pmd_fbeta), st.stdev(avg_pmd_mae), st.stdev(avg_ssfh_fbeta), st.stdev(avg_ssfh_mae)]

    df.to_csv(output_csv_path, index=False)

if __name__ == "__main__":
    args = parse_arg()
    main(args)

