
import ROOT

class Derived(ROOT.TH1F): pass

def create_derived(attr):
   return DerivedCustomReduce(attr)

class DerivedCustomReduce(ROOT.TH1F):
   def __init__(self, attr):
      ROOT.TH1F.__init__(self)
      self.attr = attr

   def __reduce__(self):
      return (create_derived, (self.attr,))
