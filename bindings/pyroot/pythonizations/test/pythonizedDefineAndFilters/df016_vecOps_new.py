import ROOT
import numpy as np

df = ROOT.RDataFrame(1024)

def generatex(len)->'RVecD':
    l = int(len)
    x = np.random.uniform(-1, 1, l)
    return x

def generatey(len)->'RVecD':
    l = int(len)
    y = np.random.uniform(-1, 1, l)
    return y

d = df.PyDefine("len", lambda: np.random.uniform(0, 16))\
        .PyDefine("x", generatex )\
        .PyDefine("y", generatey)


def calc_r(x, y) -> 'RVecD':
    return np.sqrt(x*x + y*y)

d1 = d.PyDefine("r", calc_r)
# d1.Display().Print()

def rInFig(r, x, y)->'RVecB':
    c1 = r>0.4
    c2 = r<0.8
    c12 = np.logical_and(c1, c2)
    c3 = x*y<0
    # return np.logical_and(c1, c2, c3)
    return np.logical_and(c12, c3)


def yFig(y, rInFig)->'RVecD':
    return y[rInFig]

def xFig(x, rInFig)->'RVecD':
    return x[rInFig]

ring_h = d1.PyDefine("rInFig", rInFig)\
        .PyDefine("yFig", yFig)\
        .PyDefine("xFig", xFig)\
        .Histo2D(("fig", "Two quarters of a ring", 64, -1, 1, 64, -1, 1), "xFig", "yFig")
      


# ring_h = d1.Define("rInFig", "r > .4 && r < .8 && x*y < 0")\
#            .Define("yFig", yFig)\
#            .Define("xFig", xFig)\
#            .Histo2D(("fig", "Two quarters of a ring", 64, -1, 1, 64, -1, 1), "xFig", "yFig")
 
cring = ROOT.TCanvas()
ring_h.Draw("Colz")
cring.SaveAs("df016_ring_new.png")
 
print("Saved figure to df016_ring.png")