from hmm import Hmm

model = Hmm(char2idx_path="dicts/otherchar2idx.json", tag2idx_path="dicts/othertag2idx.json")
model.fit("corpus/hmm_疫情其他数据集_20000.txt")

f = open("corpus/疫情分句后的数据.txt", encoding='utf-8')
ents = []
with open("hmm其他抽取结果.txt", "a", encoding="utf-8") as w:
    for line in f:
        line = line.strip()
        if len(line) <= 1:
            continue
        tmp = model.yq_usage(line)
        for t in tmp:
            if t in ents:
                continue
            ents.append(t)
            w.write(t + '\n')
            print(t)

f.close()




