/*****************************************************************************
 * Project: RooStats
 * Package: RooStats
 *    File: $Id$
 * author: Kyle Cranmer
 *****************************************************************************/
#ifndef JEFFREYSPRIOR
#define JEFFREYSPRIOR

#include "RooAbsPdf.h"
#include "RooRealProxy.h"
#include "RooListProxy.h"

class RooRealVar;
class RooArgList ;

class RooJeffreysPrior : public RooAbsPdf {
public:

  RooJeffreysPrior() ;
  RooJeffreysPrior(const char *name, const char *title, RooAbsPdf& nominal, const RooArgList& paramSet, const RooArgList& obsSet) ;
  virtual ~RooJeffreysPrior() ;

  RooJeffreysPrior(const RooJeffreysPrior& other, const char* name = 0);
  virtual TObject* clone(const char* newname) const { return new RooJeffreysPrior(*this, newname); }

  const RooArgList& lowList() const { return _obsSet ; }
  const RooArgList& paramList() const { return _paramSet ; }

  Int_t getAnalyticalIntegral(RooArgSet& allVars, RooArgSet& analVars, const char* rangeName=0) const ;
  Double_t analyticalIntegral(Int_t code, const char* rangeName=0) const ;

protected:

  RooRealProxy _nominal;           // The nominal value
  //RooAbsPdf* _nominal;           // The nominal value
  RooArgList   _ownedList ;       // List of owned components
  RooListProxy _obsSet ;            // Low-side variation
  RooListProxy _paramSet ;            // interpolation parameters
  mutable TIterator* _paramIter ;  //! Iterator over paramSet
  mutable TIterator* _obsIter ;  //! Iterator over lowSet

  Double_t evaluate() const;

  ClassDef(RooJeffreysPrior,1) // Sum of RooAbsReal objects
};

#endif
