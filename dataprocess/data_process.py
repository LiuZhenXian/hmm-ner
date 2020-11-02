import json


def get_ents(path):
    with open(path, 'r', encoding='utf-8') as r:
        ents = r.read().split('\n')
    return ents


def get_start_end(sen, ent):
    start = sen.index(ent)
    end = start + len(ent)
    return start, end


def mark_sen(sen, ents):
    tup_list = []  # 先对句子中地每个token都标记为O，用tup_list存放
    tup_list = [[sen[i], "O"] for i in range(len(sen))]
    for ent in ents:
        try:
            start, end = get_start_end(sen, ent)
        except:
            continue
        tup_list[start][1] = "B-OTH"
        for j in range(start+1, end-1):
            tup_list[j][1] = "I-OTH"
        tup_list[end-1][1] = "E-OTH"
    return tup_list


def get_tagged_sen(corpus_path, ents_path, save_path, num_examples):
    ents = get_ents(ents_path)
    with open(save_path, 'w', encoding='utf-8') as w:
        f = open(corpus_path, encoding='utf-8')
        i = 0
        for line in f:
            if i == num_examples:
                break
            line = line.strip()
            if len(line) <= 2:
                continue
            res = mark_sen(line, ents)
            for token in res:
                w.write(token[0] + "\t" + token[1] + "\n")
            w.write("\n")
            i += 1
        f.close()


# 接下来是hmm的数据集
def hmm_data(corpus_path, ents_path, save_path, example_num):
    tmp_save_path = save_path[4:]
    get_tagged_sen(corpus_path, ents_path, tmp_save_path, example_num)
    f = open(tmp_save_path, encoding="utf-8")
    char2idx = {}
    data = []
    i = 0
    tmp_dic = {'text': [], 'label': []}
    for line in f:
        i += 1
        if line == '\n':
            data.append(tmp_dic)
            tmp_dic = {'text':[], 'label':[]}
            continue
        line = line.replace('\n', '').split('\t')
        char = line[0]
        label = line[1]
        tmp_dic['text'].append(char)
        tmp_dic['label'].append(label)
        if char in char2idx:
            continue
        idx = len(char2idx)
        char2idx[char] = idx
    f.close()
    json_s = json.dumps(char2idx, ensure_ascii=False)
    with open(ents_path[:-4] + "char2idx.json", "w", encoding="utf-8") as w:
        w.write(json_s)

    with open(save_path, "w", encoding="utf-8") as w:
        for d in data:
            w.write(str(d)+'\n')

    tag2idx = {'O': 0, 'B-OTH': 1, 'I-OTH': 2, 'E-OTH': 3}
    json_s = json.dumps(tag2idx, ensure_ascii=False)
    with open(ents_path[:-4] + "tag2idx.json", "w", encoding="utf-8") as w:
        w.write(json_s)


hmm_data("疫情分句后的数据.txt", "other.txt", "hmm_疫情其他数据集_20000.txt", 20000)
