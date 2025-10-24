import json

import ROOT
import uhi.io.json

import hist

h = ROOT.TH1D("h", "h", 10, -5, 5)
h[...] = range(10)
print("\nh=", h)
print("values=", h.values())

ob = json.dumps(h, default=uhi.io.json.default)
ir = json.loads(ob, object_hook=uhi.io.json.object_hook)

h_loaded = hist.Hist(ir)
print("\nh_loaded =\n", h_loaded)
