import json

headlines = []
with open("Sarcasm_Headlines_Dataset_v2.json") as f:
    lines = f.readlines()
    for line in lines:
        cur_data = json.loads(line)
        if cur_data['is_sarcastic'] == 1:
            headlines += [cur_data['headline']]

print(len(headlines))
print(headlines[0])
