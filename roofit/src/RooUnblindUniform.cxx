// -- CLASS DESCRIPTION [REAL] --
//
// Implementation of BlindTools' offset blinding method
// A RooUnblindUniform object is a real valued function
// object, constructed from a blind value holder and a 
// set of unblinding parameters. When supplied to a PDF
// in lieu of a regular parameter, the blind value holder
// supplied to the unblinder objects will in a fit be minimized 
// to blind value corresponding to the actual minimum of the
// parameter. The transformation is chosen such that the
// the error on the blind parameters is indentical to that
// of the unblind parameter

#include "RooFitCore/RooArgSet.hh"
#include "RooFitModels/RooUnblindUniform.hh"


ClassImp(RooUnblindUniform)
;


RooUnblindUniform::RooUnblindUniform() : _blindEngine("") 
{
  // Default constructor
}


RooUnblindUniform::RooUnblindUniform(const char *name, const char *title,
					 const char *blindString, Double_t scale, RooAbsReal& cpasym)
  : RooAbsHiddenReal(name,title), _blindEngine(blindString,RooBlindTools::full,0.,scale), _value("value","Uniform blinded value",this,cpasym) 
{  
  // Constructor from a given RooAbsReal (to hold the blind value) and a set of blinding parameters
}


RooUnblindUniform::RooUnblindUniform(const RooUnblindUniform& other, const char* name) : 
  RooAbsHiddenReal(other, name), _blindEngine(other._blindEngine), _value("asym",this,other._value)
{
  // Copy constructor

}


RooUnblindUniform::~RooUnblindUniform() 
{
  // Destructor
}


Double_t RooUnblindUniform::evaluate() const
{
  // Evaluate RooBlindTools unhide-offset method on blind value
  return _blindEngine.UnHideUniform(_value);
}





