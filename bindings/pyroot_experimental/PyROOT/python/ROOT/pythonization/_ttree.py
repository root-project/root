
from libROOTPython import PythonizeTTree

# TTree iterator
def _TTree__iter__(self):
  i = 0
  bytes_read = self.GetEntry(i)
  while 0 < bytes_read:
     yield self
     i += 1
     bytes_read = self.GetEntry(i)

  if bytes_read == -1:
     raise RuntimeError( "TTree I/O error" )

# Pythonizor function
def ttree_pythonizor(klass, name):
  if name == 'TTree':
     # Pythonic iterator
     klass.__iter__ = _TTree__iter__

     # C++ pythonizations
     # - tree.branch syntax
     PythonizeTTree(klass)
     
  return True


def get_pythonizor():
  return ttree_pythonizor

