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
        if type == 'coin_case':
            continue
        
        for i, ex_num in enumerate(os.listdir(type_path)):
            ex_path = os.path.join(type_path, ex_num)
            txtfile_path = ex_path + "/eval.txt"
            if os.path.exists(txtfile_path):
                with open(txtfile_path, 'r') as infile:
                    string = infile.read()
                    parts = [item.strip() for item in re.split('[\,:\n]', string) if item != '']
                    
                    avg_pmd_fbeta.append(float(parts[-3]))
                    avg_pmd_mae.append(float(parts[-1]))
                    avg_ssfh_fbeta.append(float(parts[2]))
                    avg_ssfh_mae.append(float(parts[4]))
                    if i == 0:
                        df.loc[len(df.index)] = [type, ex_num, parts[-3], parts[-1], parts[2], parts[4]]
                    else:
                        df.loc[len(df.index)] = ['', ex_num, parts[-3], parts[-1], parts[2], parts[4]]
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
    main()

