import json
import os
res_path = 'lla_cla_p_10.json'

output_path = res_path.split('.json')[0] + '_fb.json'

if not os.path.exists(output_path):
  res = json.load(open(res_path, 'r'))
  count = 0
  for r in res:
    print('pred: ', '\033[0;31m', r['pred'], '\033[0m')
    print('-'*40)
    print('ref: ', '\033[0;32m', r['ref'], '\033[0m')
    fb = input('Input code: ').strip()
    if fb:
      if fb == '1':
        r['fb'] = 1
        count += 1
      elif fb == '2':
        r['fb'] = 2 # multiple answers
      elif fb == '3':
        r['fb'] = 3 # invalid output
      elif fb == '0':
        r['fb'] = 0 # nonsense
    else:
      r['fb'] = 0 # wrong

  json.dump(res, open(output_path, 'w'), indent=2)

  print('acc: %0.3f' % count/len(res))
else:
  res = json.load(open(output_path, 'r'))
  count = {'all': [0, 0]}
  for r in res:
    count[r['ref']] = count.get(r['ref'], [0, 0])
    count[r['ref']][0] += 1
    count['all'][0] += 1
    try:
      if r['fb'] == 1:
        count[r['ref']][1] += 1
        count['all'][1] += 1
    except:
      print(r)
  
  for k, v in count.items():
    print(k, 'acc: %0.3f' % (v[1]/v[0]))
    
      
      


  