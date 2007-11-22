#include "RooRealVar.h"
#include "RooGlobalFunc.h"

using namespace RooFit ;

// Elementary operations on a gaussian PDF
class TestBasic11 : public RooFitTestUnit
{
public: 
  TestBasic11(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooFitTestUnit(refFile,writeRef,verbose) {} ;
  Bool_t testCode() {

    // "Workspace persistence"
    return kTRUE ;
  }
} ;
