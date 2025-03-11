import pandas as pd


df = pd.read_csv("captions.txt", delimiter=",")

df_merged = df.groupby('image', as_index=False)['caption'].apply(lambda x: ' '.join(x))

# df_merged.to_csv("flikr_8k_cleaned.csv")

with open('only_caption.txt',"a") as f:
    for caption in list(df_merged['caption']):
        f.write(caption+"\n")