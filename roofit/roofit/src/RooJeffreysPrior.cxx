/*****************************************************************************

 *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
// 
// BEGIN_HTML
// RooJeffreysPrior 
// END_HTML
//


#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include <math.h>

#include "RooJeffreysPrior.h"
#include "RooAbsReal.h"
#include "RooAbsPdf.h"
#include "RooErrorHandler.h"
#include "RooArgSet.h"
#include "RooMsgService.h"
#include "RooFitResult.h"
#include "TMatrixDSym.h"
#include "RooDataHist.h"
#include "RooFitResult.h"
#include "RooNumIntConfig.h"
#include "RooRealVar.h"

using namespace std;

ClassImp(RooJeffreysPrior)
;

using namespace RooFit;

//_____________________________________________________________________________
RooJeffreysPrior::RooJeffreysPrior(const char* name, const char* title, 
			     RooAbsPdf& nominal,
			     const RooArgList& paramSet,
			     const RooArgList& obsSet) :
  RooAbsPdf(name, title),
  _nominal("nominal","nominal",this,nominal,kFALSE,kFALSE),
  _obsSet("!obsSet","obs-side variation",this,kFALSE,kFALSE),
  _paramSet("!paramSet","high-side variation",this)
{
  //_obsSet("!obsSet","obs-side variation",this),
  _obsIter = _obsSet.createIterator() ;
  _paramIter = _paramSet.createIterator() ;


  TIterator* inputIter1 = obsSet.createIterator() ;
  RooAbsArg* comp ;
  while((comp = (RooAbsArg*)inputIter1->Next())) {
    if (!dynamic_cast<RooAbsReal*>(comp)) {
      coutE(InputArguments) << "RooJeffreysPrior::ctor(" << GetName() << ") ERROR: component " << comp->GetName() 
			    << " in first list is not of type RooAbsReal" << endl ;
      RooErrorHandler::softAbort() ;
    }
    _obsSet.add(*comp) ;
    //    if (takeOwnership) {
    //      _ownedList.addOwned(*comp) ;
    //    }
  }
  delete inputIter1 ;



  TIterator* inputIter3 = paramSet.createIterator() ;
  while((comp = (RooAbsArg*)inputIter3->Next())) {
    if (!dynamic_cast<RooAbsReal*>(comp)) {
      coutE(InputArguments) << "RooJeffreysPrior::ctor(" << GetName() << ") ERROR: component " << comp->GetName() 
			    << " in first list is not of type RooAbsReal" << endl ;
      RooErrorHandler::softAbort() ;
    }
    _paramSet.add(*comp) ;
    //    if (takeOwnership) {
    //      _ownedList.addOwned(*comp) ;
    //    }
  }
  delete inputIter3 ;


  // use a different integrator by default.
  if(paramSet.getSize()==1)
    this->specialIntegratorConfig(kTRUE)->method1D().setLabel("RooAdaptiveGaussKronrodIntegrator1D")  ;

}



//_____________________________________________________________________________
RooJeffreysPrior::RooJeffreysPrior(const RooJeffreysPrior& other, const char* name) :
  RooAbsPdf(other, name), 
  _nominal("!nominal",this,other._nominal),
  _obsSet("!obsSet",this,other._obsSet),
  _paramSet("!paramSet",this,other._paramSet)
{
  // Copy constructor
  _obsIter = _obsSet.createIterator() ;
  _paramIter = _paramSet.createIterator() ;

  // Member _ownedList is intentionally not copy-constructed -- ownership is not transferred
}

//_____________________________________________________________________________
RooJeffreysPrior::RooJeffreysPrior() 
{
  // Default constructor
  _obsIter = NULL;
  _paramIter = NULL;

}



//_____________________________________________________________________________
RooJeffreysPrior::~RooJeffreysPrior() 
{
  // Destructor

  if (_obsIter) delete _obsIter ;
  if (_paramIter) delete _paramIter ;
}




//_____________________________________________________________________________
Double_t RooJeffreysPrior::evaluate() const 
{
  // Calculate and return current value of self
  RooFit::MsgLevel msglevel = RooMsgService::instance().globalKillBelow();
  RooMsgService::instance().setGlobalKillBelow(RooFit::WARNING);
  // create Asimov dataset
  //  _paramSet.Print("v");
  //  return sqrt(_paramSet.getRealValue("mu"));
  *(_nominal.arg().getVariables()) = _paramSet;
  /*
  cout << "_______________"<<endl;
  _paramSet.Print("v");
  _nominal->getVariables()->Print("v");
  cout << "_______________"<<endl;
  */
  RooDataHist* data = ((RooAbsPdf&)(_nominal.arg())).generateBinned(_obsSet,ExpectedData());
  //  data->Print("v");
  //RooFitResult* res = _nominal->fitTo(*data, Save(),PrintLevel(-1),Minos(kFALSE),SumW2Error(kTRUE));
  RooFitResult* res = ((RooAbsPdf&)(_nominal.arg())).fitTo(*data, Save(),PrintLevel(-1),Minos(kFALSE),SumW2Error(kFALSE));
  TMatrixDSym cov = res->covarianceMatrix();
  cov.Invert();
  double ret =  sqrt(cov.Determinant());
  
  /*
    // for 1 parameter can avoid making TMatrix etc.
    // but number of params may be > 1 with others held constant
  if(_paramSet.getSize()==1){
    RooRealVar* var = (RooRealVar*) _paramSet.first();
    // also, the _paramSet proxy one does not pick up a different value
    cout << "eval at "<< ret << " " << 1/(var->getError()) << endl; 
    // need to get the actual variable instance out of the pdf like below
    var = (RooRealVar*) _nominal->getVariables()->find(var->GetName());
    cout << "eval at "<< ret << " " << 1/(var->getError()) << endl; 
  }
  */

  //  res->Print();
  delete data;
  delete res;
  RooMsgService::instance().setGlobalKillBelow(msglevel);

  //  cout << "eval at "<< ret << endl; 
  //  _paramSet.Print("v");
  return ret;

}

//_____________________________________________________________________________
Int_t RooJeffreysPrior::getAnalyticalIntegral(RooArgSet& /*allVars*/, RooArgSet& /*analVars*/, const char* /*rangeName*/) const 
{
  //  if (matchArgs(allVars,analVars,x)) return 1 ;
  //  if (matchArgs(allVars,analVars,mean)) return 2 ;
  //  return 1;
  return 0 ;
}



//_____________________________________________________________________________
Double_t RooJeffreysPrior::analyticalIntegral(Int_t code, const char* /*rangeName*/) const 
{
  assert(code==1 );
  //cout << "evaluating analytic integral" << endl;
  return 1.;
}


