/*****************************************************************************
 * Project: RooFit                                                           *
 * author: Max Baak (mbaak@cern.ch)                                          *
 *****************************************************************************/

#ifndef ROO1DMOMENTMORPHFUNC
#define ROO1DMOMENTMORPHFUNC

#include "RooAbsReal.h"
#include "RooRealProxy.h"
#include "RooCategoryProxy.h"
#include "RooAbsReal.h"
#include "RooAbsCategory.h"
#include "RooSetProxy.h"
#include "RooArgList.h"
#include "TMatrixD.h"
#include "TVectorD.h"

#include <vector>
#include <string>
class RooChangeTracker ;

class Roo1DMomentMorphFunction : public RooAbsReal {
public:

  enum Setting { Linear, NonLinear, NonLinearPosFractions, NonLinearLinFractions } ;

  Roo1DMomentMorphFunction() ;

  Roo1DMomentMorphFunction(const char *name, const char *title, RooAbsReal& _m, const RooArgList& varList,
			   const TVectorD& mrefpoints, const Setting& setting = Linear );
  Roo1DMomentMorphFunction(const Roo1DMomentMorphFunction& other, const char* name=0) ;

  virtual TObject* clone(const char* newname) const { return new Roo1DMomentMorphFunction(*this,newname); }
  virtual ~Roo1DMomentMorphFunction();

  void     setMode(const Setting& setting) { _setting = setting; }

protected:

  Double_t evaluate() const ;

  void calculateFractions() const;

  void     initialize();

  inline   Int_t ij(const Int_t& i, const Int_t& j) const { return (i*_varList.getSize()+j); }
  int      idxmin(const double& m) const;
  int      idxmax(const double& m) const;

  RooRealProxy m ;
  RooSetProxy  _varList ;
  mutable TVectorD* _mref;
  mutable TVectorD* _frac; 

  TIterator* _varItr ;   //! do not persist
  mutable TMatrixD* _M; 

  Setting _setting;

  ClassDef(Roo1DMomentMorphFunction,1) // Your description goes here...
};
 
#endif


