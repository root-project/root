import ROOT
import numpy as np


def main():
        
    npoints = 10000000
    df = ROOT.RDataFrame(npoints)

    def f3(x, y)->'RVecD':
        return np.array([x,y])
    
    def f4(p):
        r2 = p[0]**2 + p[1]**2
        return np.sqrt(r2)

    # pidf = df.PyDefine("x", f1)
    pidf = df.PyDefine("x", lambda: np.random.uniform(-1, 1) )\
            .PyDefine("y", lambda: np.random.uniform(-1, 1))\
            .PyDefine("p", f3)\
            .PyDefine("r", f4)\
            .Filter(lambda r: r<=1.0)
    print(f"pi is approximately equal to {4*pidf.Count().GetValue()/npoints}.")

if __name__=='__main__':
    main()