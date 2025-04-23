import sys

# Note: ROOT does not support import * for Python 3.x
# More info at https://sft.its.cern.ch/jira/browse/ROOT-8931

if sys.hexversion >= 0x3000000:
   from ROOT import TChain
else:
   from ROOT import *

class TestTChain:
   def __init__( self ):
      self.fChain = TChain( '' )
