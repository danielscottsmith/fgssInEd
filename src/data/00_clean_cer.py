import glob
import csv
import pandas as pd
import argparse
from tqdm import tqdm
import json
import os
from sklearn.model_selection import train_test_split


parser = argparse.ArgumentParser(description="Gets full text of sampled CER articles.")
parser.add_argument(
    "-i",
    "--input_dir",
    type=str,
    help="path to raw data",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to interim data",
)


MODES = "train dev test".split()


# def get_data_dir(input_dir):
#     data_dir = os.path.dirname(input_dir)
#     data_dir = os.path.dirname(data_dir)
#     return data_dir


def read_sample_csvs(input_dir):
    """
    CSVs are qualtrics exports of CER survey.
    """
    all_data = []
    csv_files = glob.glob(input_dir + "*.csv")
    for csv_file in csv_files:
        with open(csv_file, "r") as f:
            for line in csv.DictReader(f):
                all_data.append(line)
    df = pd.DataFrame.from_dict(all_data)
    return df


def get_population_df(output_dir):
    """
    Population df is full population of education
    articles in JSTOR, already processed in ''
    """
    df = pd.read_feather(output_dir + "interim_articles_df.feather")
    df = df[df["journal"] == "Comparative Education Review"]
    df = df[["year", "volume", "issue", "title", "text"]]
    print(df.iloc[10])
    return df


def _dichotomize_label(analysis_type):
    """
    Helper function to convert string
    into dummy.
    """
    if "quantitative" in analysis_type:
        return 1
    else:
        return 0


def code_sample(df):
    """
    Recasts sample year to int and
    Labels analysis type.
    """
    df["year"] = pd.to_numeric(df["year"])
    df["quant"] = df["analysis_type"].apply(_dichotomize_label)
    coded_df = df[["year", "volume", "issue", "title", "quant"]]
    return coded_df


def get_full_texts(sample_df, population_df):
    """
    Merges full texts from population df into
    sample df.
    """
    sample_df = pd.merge(
        sample_df, population_df, how="left", on=["year", "volume", "issue", "title"]
    )
    return sample_df


def split_sample(sample_df, MODES, data_dir):
    """
    Splits labeled data up and saves into
    train, dev, test subdirectories
    """
    train_dev, test = train_test_split(output_dir, test_size=20, random_state=72)
    train, dev = train_test_split(train_dev, test_size=0.25, random_state=72)
    samples = [train, dev, test]
    for i, mode in enumerate(MODES):
        # sub_dir = data_dir + f"/{mode}/"
        # if not os.path.exists(sub_dir):
        #     os.makedirs(sub_dir)
        samples[i].to_csv(output_dir + f"{mode}.csv")


def main():
    args = parser.parse_args()

    print("Loading data...")
    sample_df = read_sample_csvs(args.input_dir)
    # population_df = get_population_df(args.output_dir)
    print("\u2713", "Data loaded!\n")

    print("Prepping data...")
    sample_df = code_sample(sample_df)
    # sample_df = get_full_texts(sample_df, population_df)
    split_sample(sample_df, MODES, data_dir)

    # print(u'\u2713', "Data cleaned!\n")
    # print("All Done~")


if __name__ == "__main__":
    main()
