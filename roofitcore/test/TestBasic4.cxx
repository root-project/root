#include "RooRealVar.h"
#include "RooGlobalFunc.h"

using namespace RooFit ;

// Elementary operations on a gaussian PDF
class TestBasic4 : public RooFitTestUnit
{
public: 
  TestBasic4(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Conditional product operator p.d.f"

    return kTRUE ;
  }
} ;
