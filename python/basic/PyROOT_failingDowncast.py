source = """

#include "TH1.h"

TObject* f(){return new TH1F("","",100,0,1);}

class Base{
public:
  virtual ~Base(){};
};

class Derived: public Base {
  virtual ~Derived(){};
};

Base* g(){return new Derived();}
"""

header = open("failingDowncast.h","w")
header.write(source)
header.close()

import ROOT
ROOT.gInterpreter.ProcessLine('#include "failingDowncast.h"')

# for an eventual unit test:
if (type(ROOT.g()) != ROOT.Derived):
  print "Expecting TH1F:    ", ROOT.f()
  print "Expecting Derived: ", ROOT.g()
  raise Exception("Implicit downcast of 'Base' to 'Derived' failed")

