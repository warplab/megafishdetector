import json
import sys

filename = sys.argv[1]
j = json.load(open(filename))
print(j["fps"])
