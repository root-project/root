#ifndef ROO_UNBLIND_UNIFORM
#define ROO_UNBLIND_UNIFORM

#include "RooFitCore/RooAbsHiddenReal.hh"
#include "RooFitCore/RooRealProxy.hh"
#include "RooFitModels/RooBlindTools.hh"

class RooUnblindUniform : public RooAbsHiddenReal {
public:
  // Constructors, assignment etc
  RooUnblindUniform() ;
  RooUnblindUniform(const char *name, const char *title, 
		      const char *blindString, Double_t scale, RooAbsReal& blindValue);
  RooUnblindUniform(const RooUnblindUniform& other, const char* name=0);
  virtual TObject* clone(const char* newname) const { return new RooUnblindUniform(*this,newname); }  
  virtual ~RooUnblindUniform();

protected:

  // Function evaluation
  virtual Double_t evaluate() const ;

  RooRealProxy _value ;
  RooBlindTools _blindEngine ;

  ClassDef(RooUnblindUniform,1) // Uniform unblinding transformation
};

#endif
