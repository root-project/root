#include "RooRealVar.h"
#include "TIterator.h"
#include "RooWorkspace.h"
#include "RooStats/HistFactory/FlexibleInterpVar.h"

using namespace RooStats;
using namespace HistFactory;

// code = 0: piece-wise linear
// code = 1: pice-wise log
// code = 2: parabolic interp with linear extrap
// code = 3: parabolic version of log-normal

void ModifyInterpolationForAll(RooWorkspace* ws, int code=1){
  RooArgSet funcs = ws->allFunctions();
  TIterator* it = funcs.createIterator();
  TObject* tempObj=0;
  while((tempObj=it->Next())){    
    FlexibleInterpVar* flex = dynamic_cast<FlexibleInterpVar*>(tempObj);
    if(flex){
      flex->setAllInterpCodes(code);
    }
  }     
}

void ModifyInterpolationForSet(RooArgSet* modifySet, int code = 1){

  TIterator* it = modifySet->createIterator();
  RooRealVar* alpha=0;
  while((alpha=(RooRealVar*)it->Next())){
    TIterator* serverIt = alpha->clientIterator();
    TObject* tempObj=0;
    while((tempObj=serverIt->Next())){
      FlexibleInterpVar* flex = dynamic_cast<FlexibleInterpVar*>(tempObj);
      if(flex){
	flex->printAllInterpCodes();
	flex->setInterpCode(*alpha,code);
	flex->printAllInterpCodes();
      }
    }     
  }
  
}


void CheckInterpolation(RooWorkspace* ws){
  RooArgSet funcs = ws->allFunctions();
  TIterator* it = funcs.createIterator();
  TObject* tempObj=0;
  while((tempObj=it->Next())){    
    FlexibleInterpVar* flex = dynamic_cast<FlexibleInterpVar*>(tempObj);
    if(flex){
      flex->printAllInterpCodes();
    }
  }     
}
