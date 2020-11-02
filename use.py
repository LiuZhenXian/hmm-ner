from hmm import Hmm
import pandas as pd

data = pd.read_csv("data/协同_未标记实体.csv")
for i in range(data.shape[0]):
    text = data.iloc[i, 2]

model = Hmm(char2idx_path="dicts/char2idx.json", tag2idx_path="dicts/tag2idx.json")
model.fit("corpus/train_data.txt")

res = pd.DataFrame(columns=["标签", "实体", "正文"])

for i in range(data.shape[0]):
    raw = data.iloc[i, 2].strip()
    tmp = model.usage(raw)
    label = data.iloc[i, 0]
    ents = data.iloc[i, 1]
    res = res.append({"标签": label, "实体": ents, "正文": tmp}, ignore_index=True)

res.to_csv("data/协同_已标记实体初版.csv", encoding="utf_8_sig", index=False)
