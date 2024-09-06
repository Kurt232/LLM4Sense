import json
data_json = json.load(open('openaqa/data/toy.json', 'r'))
result_json = []
print(len(data_json))
for i in range(len(data_json)):
  data_json[i]['data_id'] = i/2 # duplicate source_data
json.dump(data_json, open('openaqa/data/toy.json', 'w'), indent=2)
