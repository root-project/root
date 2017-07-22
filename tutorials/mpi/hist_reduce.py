## \file
## \ingroup tutorial_mpi
##
## Example to generated random numbers to fill a TH1F histogram in every process
## and merging the result through a custom reduce operation.
## to execute this example with 4 processors, do:
##
## ~~~{.cpp}
##  rootmpi -np 4 hist_reduce.py
## ~~~
##
##
## \macro_output
## \macro_code
##
## \author Omar Zapata

from ROOT import Mpi, TH1F, TF1, TFormula, TCanvas
from ROOT.Mpi import TEnvironment, COMM_WORLD

def HSUM(a,b): # histogram sum(custom operation for reduce)
    #returning an object that is a 
    #histograms sum
    c=TH1F(a)
    c.Add(b)
    return c

def hist_reduce(points = 100000):
   env=TEnvironment()

   root = 0
   rank = COMM_WORLD.GetRank()

   if COMM_WORLD.GetSize() == 1:    return # need at least 2 process

   form1 = TFormula("form1", "abs(sin(x)/x)")
   sqroot = TF1("sqroot", "x*gaus(0) + [3]*form1", 0, 10)
   sqroot.SetParameters(10, 4, 1, 20);

   h1f=TH1F("h1f", "Test random numbers", 200, 0, 10)
   h1f.SetFillColor(rank);
   h1f.FillRandom("sqroot", points)

   

   result=COMM_WORLD.Reduce(h1f, HSUM, root)

   if rank == root:
      c1 = TCanvas("c1", "The FillRandom example", 200, 10, 700, 900)
      c1.SetFillColor(18)
      result.Draw()
      c1.SaveAs("hist.png")

if __name__ == "__main__":
    hist_reduce()

