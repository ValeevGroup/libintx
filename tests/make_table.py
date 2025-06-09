#!/usr/bin/python3

import yaml, sys

docs = []
for f in sys.argv[1:]:
  docs.extend(yaml.load_all(open(f)))

raw_data = {}
simd = None
cxx = None
for doc in docs:
  simd = doc.get("simd", simd)
  cxx = doc.get("cxx", cxx)
  if "results" in doc:
    kv = [(k,v) for t in doc["results"] for (k,v) in t.items()]
    #print (kv)
    raw_data.setdefault((simd,cxx), []).append((doc["K"], kv))

table = []

for (simd,cxx),entries in raw_data.items():
  col = []
  col.append(("simd", simd))
  for (K,v) in entries:
    col.append(("K", K))
    #print ("K      & %s" % K)
    for braket,data in v:
      x = data.split("=")[-1]
      col.append(("$%s$" % braket,x))
  table.append([c1 for c1,c2 in col])
  table.append([c2 for c1,c2 in col])

table = zip(*table) # transpose rows/cols

from tabulate import tabulate
print (tabulate(table, tablefmt="latex"))
