#ifndef ROO_LANDAU
#define ROO_LANDAU

#include "RooFitCore/RooAbsPdf.hh"
#include "RooFitCore/RooRealProxy.hh"

class RooRealVar;

class RooLandau : public RooAbsPdf {
public:
  RooLandau(const char *name, const char *title, RooAbsReal& _x, RooAbsReal& _mean, RooAbsReal& _sigma);
  RooLandau(const RooLandau& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooLandau(*this,newname); }
  inline virtual ~RooLandau() { }

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t staticInitOK=kTRUE) const;
  void generateEvent(Int_t code);
  
protected:
  
  RooRealProxy x ;
  RooRealProxy mean ;
  RooRealProxy sigma ;
  
  Double_t evaluate() const ;
  
private:
  
  ClassDef(RooLandau,0) // Landau Distribution PDF
};

#endif
