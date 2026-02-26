def dataset_overview(df):
    print("Number of documents:", len(df))
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

def missing_values(df):
    print("\nMissing values per column:")
    print(df.isnull().sum())

def class_distribution(df):
    counts = df["class_label"].value_counts()
    proportions = df["class_label"].value_counts(normalize=True)

    print("\nClass distribution (counts):")
    print(counts)

    print("\nClass distribution (proportions):")
    print(proportions)

def text_statistics(df, data_info_path, text_column="post"):
    lengths = df[text_column].astype(str).apply(len)

    print("\nText length statistics:")
    print("Min length:", lengths.min())
    print("Max length:", lengths.max())
    print("Mean length:", round(lengths.mean(), 2))
    print("Median length:", lengths.median())

    with open(data_info_path, "w") as f:
        f.write("\nText length statistics:"+ "\n")
        f.write("Min length:" + str(lengths.min())+ "\n")
        f.write("Max length:"+ str(lengths.max())+ "\n")
        f.write("Mean length:" + str(round(lengths.mean(), 2))+ "\n")
        f.write("Median length:" + str(lengths.median())+ "\n")