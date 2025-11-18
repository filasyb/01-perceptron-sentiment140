import pandas as pd
from data_preprocessing import preprocess_dataset, load_config

config = load_config()

raw_path = config["paths"]["raw_data"]
output_path = config["paths"]["preprocessed_data"]

print(f"Cargando dataset original desde: {raw_path}")
cols = ['target','ids','date','flag','user','text']

df = pd.read_csv(raw_path, encoding="latin-1", names=cols)
df = df[["target", "text"]]

df["target"] = df["target"].apply(lambda x: 1 if x == 4 else 0)


df_pos = df[df.target == 1].sample(config["dataset"]["sample_size_pos"])
df_neg = df[df.target == 0].sample(config["dataset"]["sample_size_neg"])

df_sample = pd.concat([df_pos, df_neg]).sample(frac=1).reset_index(drop=True)


df_sample = preprocess_dataset(df_sample)

df_sample.to_csv(output_path, index=False)
print(f"Dataset reducido guardado en: {output_path}")
