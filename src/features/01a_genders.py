from genderize import Genderize
import pandas as pd
from tqdm import tqdm
import argparse
import json
import glob
import csv

parser = argparse.ArgumentParser(
    description="Writes an interim df containing articles with standardized col names"
)
parser.add_argument(
    "-d",
    "--data_dir",
    type=str,
    help="processed data dir",
)




def _remove_initials(name):
    if name:
        new = name.replace(".", "")
        new = new.strip()
        if len(new) < 2:
            return None
        else:
            return name
    else:
        return None


def get_df(data_dir):
    return pd.read_feather(data_dir + "cleaned.feather")


def clean_name(df):
    df["auth1_name"] = df["auth1_name"].apply(lambda x: _remove_initials(x))
    n_names = len(set(df["auth1_name"].unique()))
    print(f"... {n_names} names total.")
    return df


API_KEY = "bbe489da604ffbc66a8b9f0764dd88ce"
def lookup_gender(df, data_dir):
    name_genders = glob.glob(data_dir + "name_genders*")
    genderize = Genderize(api_key=API_KEY, timeout=600)
    
    # when no records exist of name genders, start from scratch
    if len(name_genders) == 0:
        genderized_names = {}
        with open(data_dir + f"name_genders.csv", "w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["name", "gender", "prob_woman"])
            if name:
                for name in tqdm(set(df["auth1_name"].unique())):
                    sex = genderize.get([name])[0]["gender"]
                    prob_woman = genderize.get([name])[0]["probability"]
                    if sex == "female":
                        gender = "woman"
                    elif sex == "male":
                        gender = "man"
                        prob_woman = 1 - prob_woman
                    genderized_names[name] = [gender, prob_woman]
                    csv_writer.writerow([name, gender, prob_woman])

    # when records already exist, use and update existing records
    else:
        genderized_names = pd.read_csv(name_genders[0])
        genderized_names = genderized_names.set_index("name")[
            ["gender", "prob_woman"]
        ].T.to_dict("list")
        genderized_names[None] = [None, None]
        with open(data_dir + f"name_genders.csv", "a") as f:
            csv_writer = csv.writer(f)
            for name in tqdm(set(df["auth1_name"].unique())):

                # look up only if name is not in record
                if name:
                    if name not in genderized_names:
                        sex = genderize.get([name])[0]["gender"]
                        prob_woman = genderize.get([name])[0]["probability"]
                        if sex == "female":
                            gender = "woman"
                        elif sex == "male":
                            gender = "man"
                            prob_woman = 1 - prob_woman
                        genderized_names[name] = [gender, prob_woman]
                        csv_writer.writerow([name, gender, prob_woman])
    return genderized_names


def get_genders(df, genderized_names, data_dir):
    df["auth1_gender"] = df["auth1_name"].apply(lambda x: genderized_names[x][0])
    df["auth1_probwom"] = df["auth1_name"].apply(lambda x: genderized_names[x][1])
    df.to_feather(data_dir + "cleaned.feather")
    return df


def main():
    args = parser.parse_args()
    print("Loading data...")
    df = get_df(args.data_dir)
    print("\u2713", "Data loaded!")

    print("Cleaning first names...")
    df = clean_name(df)
    print("\u2713", "First names cleaned!")

    print("Classifying gender of names...")
    genderized_names = lookup_gender(df, args.data_dir)
    get_genders(df, genderized_names, args.data_dir)
    print("\u2713", "Names classified!")


if __name__ == "__main__":
    main()
