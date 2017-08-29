## \file
## \ingroup tutorial_mpi
##
##  Simple example to show the process id and the name of the host.
##
## ~~~{.cpp}
##  rootmpi -np 2 hello.py
## ~~~
##
##
## \macro_output
## \macro_code
##
## \author Omar Zapata

from ROOT import Mpi
from ROOT.Mpi import TEnvironment, COMM_WORLD

env = TEnvironment() # environment to start communication system
def  hello():
    print("Hello from process %i of %i in host %s"%(COMM_WORLD.GetRank(),COMM_WORLD.GetSize(),env.GetProcessorName()))

if __name__ == "__main__":
    hello()
