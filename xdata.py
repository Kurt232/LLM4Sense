import json
import copy
data_json = json.load(open('openaqa/data/hhar/split.json', 'r'))
activity_info = json.load(open('openaqa/data/hhar/activity_info.json', 'r'))

train_info = {}

for k, v in activity_info.items():
  t = []
  for i in v:
    if i in data_json['train_idx']:
      t.append(i)
  train_info[k] = t

json.dump(train_info, open('openaqa/data/hhar/train_info.json', 'w'), indent=2)