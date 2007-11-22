#include "RooRealVar.h"
#include "RooGlobalFunc.h"

using namespace RooFit ;

// Elementary operations on a gaussian PDF
class TestBasic7 : public RooFitTestUnit
{
public: 
  TestBasic7(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {
    
    // "Addition oper. p.d.f with range transformed fractions"

    return kTRUE ;
  }
} ;
