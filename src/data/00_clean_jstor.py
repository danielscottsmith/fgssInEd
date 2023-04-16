import glob
import gzip
import os
import shutil
import pandas as pd
import argparse
from tqdm import tqdm
import json
import numpy as np


parser = argparse.ArgumentParser(
    description="Writes an interim df containing JSTOR articles with standardized col names"
)
parser.add_argument(
    "-i",
    "--in_dir",
    type=str,
    help="raw data subdirectory",
)
parser.add_argument(
    "-o",
    "--out_dir",
    type=str,
    help="population or sample?",
)


def unzip_files(input_dir):
    zip_files = glob.glob(input_dir + "*.gz")
    for zip_file in tqdm(zip_files):
        unzip_file = input_dir + os.path.basename(zip_file).split(".")[0] + ".jsonl"
        with gzip.open(zip_file, "rb") as f_in:
            with open(unzip_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)


def read_raw_data_json(input_dir, output_dir):
    raw_articles = []
    json_files = glob.glob(input_dir + "*.jsonl")
    for json_file in json_files:
        with open(json_file) as f:
            for line in tqdm(f):
                raw_articles.append(json.loads(line))
    raw_articles_df = pd.DataFrame.from_dict(raw_articles)
    print(raw_articles_df.columns)
    raw_articles_df.to_feather(output_dir + "raw.feather")


def clean_df(output_dir):
    feather_file = glob.glob(output_dir + "raw*.feather")[0]
    df = pd.read_feather(feather_file)
    df = df[df["docSubType"] == "research-article"]
    all_articles = []
    article_idx = 0
    for _, article in tqdm(df.iterrows(), total=df.shape[0]):
        article_idx += 1
        article_dct = {
            "article_id": f"jstor{article_idx}",
            "journal": article["isPartOf"],
            "year": article["publicationYear"],
            "volume": article["volumeNumber"],
            "issue": article["issueNumber"],
            "type": article["docSubType"],
            "authors": article["creator"],
            "title": article["title"],
            "nwords": article["wordCount"],
            "npages": article["pageCount"],
            "text": article["fullText"],
        }

        if type(article["creator"]) is np.ndarray:
            authors = article["creator"].tolist()

            if "" in authors:
                authors.remove("")

            if len(authors) > 0:
                article_dct["nauthors"] = len(authors)
                article_dct["auth1_name"] = authors[0].split().pop(0)
            else:
                print("list:")
                print(authors)
                article_dct["nauthors"] = None
                article_dct["auth1_name"] = None

        else:
            print()
            print("raw, not np.ndarray")
            print(article["creator"])
            article_dct["nauthors"] = None
            article_dct["auth1_name"] = None

        all_articles.append(article_dct)
    # all_articles = pd.DataFrame.from_dict(all_articles)
    # all_articles.to_feather(output_dir + "cleaned.feather")


def main():
    args = parser.parse_args()

    print("Decompressing files...")
    # unzip_files(args.in_dir)
    print("\u2713", "Files decompressed!\n")

    print("Reading jsonl files...")
    # raw_df = read_raw_data_json(args.in_dir, args.out_dir)
    print("\u2713", "Raw data loaded!\n")

    print("Cleaning data...")
    clean_df(args.out_dir)
    print("\u2713", "Data cleaned!\n")
    print("All Done~")


if __name__ == "__main__":
    main()
