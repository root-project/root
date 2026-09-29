## \file
## \ingroup tutorial_tree
## \notebook -nodraw
##
##
## To use this file, generate hsimple.root first:
## ~~~ {.py}
##    IPython [1]: %run hsimple.py # output: hsimple.root
## ~~~
## and follow the python instructions of "hsimpleProxyDriver.py"
## ~~~ {.py}
##    IPython [2]: %run hsimpleProxyDriver.py 
## ~~~
## 
##
## \macro_code
##
## \author Rene Brun
## \translator P. P.


import ROOT
from ROOT import TTree


# double
def hsimpleProxy( ntuple : TTree ) :
   #
   return ntuple.px
   


if __name__ == "__main__":
   # hsimpleProxy()
   pass
