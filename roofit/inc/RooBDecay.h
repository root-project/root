#ifndef ROO_BDECAY
#define ROO_BDECAY

#include "RooFitCore/RooConvolutedPdf.hh"
#include "RooFitCore/RooRealProxy.hh"

class RooBDecay : public RooConvolutedPdf
{

public:

  enum DecayType { SingleSided, DoubleSided, Flipped };

  //Constructors, assignment etc
  inline RooBDecay() { }
  RooBDecay(const char *name, const char *title, RooRealVar& t,
			RooAbsReal& tau, RooAbsReal& dgamma,
			RooAbsReal& f1, RooAbsReal& f2, 
			RooAbsReal& f3, RooAbsReal& dm, 
			const RooResolutionModel& model,
			DecayType type);
  RooBDecay(const RooBDecay& other, const char* name=0);
  virtual TObject* clone(const char* newname) const 
  { 
    return new RooBDecay(*this,newname);
  }
  virtual ~RooBDecay();

  virtual Double_t coefficient(Int_t basisIndex) const;

  Int_t getGenerator(const RooArgSet& directVars, RooArgSet &generateVars, Bool_t staticInitOK=kTRUE) const;
  void generateEvent(Int_t code);

protected:

  RooRealProxy _t;
  RooRealProxy _tau;
  RooRealProxy _dgamma;
  RooRealProxy _f1;
  RooRealProxy _f2;
  RooRealProxy _f3;
  RooRealProxy _dm;	
  Int_t _basisCosh;
  Int_t _basisSinh;
  Int_t _basisCos;
  Int_t _basisSin;
  Int_t _basisB;
  DecayType _type;

  ClassDef(RooBDecay, 1) //Abstract Resolution Model
    };

#endif

