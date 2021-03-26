#include "Floats.h"
#include "LinearCombination.h"
ClassImp(RooLagrangianMorphing::LinearCombination);
namespace {
  template<class T> inline void assign(RooLagrangianMorphing::SuperFloat& var, const T& val){
    #ifdef USE_UBLAS
    var.assign(val);
    #else
    var = val;
    #endif
  }
  inline double convertToDouble(const RooLagrangianMorphing::SuperFloat& var){
    #ifdef USE_UBLAS
    return var.convert_to<double>();
    #else
    return var;
    #endif
  }
}

namespace RooLagrangianMorphing {
  LinearCombination::LinearCombination() :
    _actualVars("actualVars","Variables used by formula expression",this),
    _nset(0)
  {
    // constructor
  }

  LinearCombination::LinearCombination(const char* name) :
    RooAbsReal(name,name),
    _actualVars("actualVars","Variables used by formula expression",this),
    _nset(0)
  {
    // constructor
  }

  LinearCombination::LinearCombination(const LinearCombination& other, const char* name) :
    RooAbsReal(other, name),
    _actualVars("actualVars",this,other._actualVars),
    _coefficients(other._coefficients),
    _nset(0)
  {
    // copy constructor
  }

  void LinearCombination::printArgs(std::ostream& os) const {
    // detailed printing method
    os << "[";
    const std::size_t n(this->_actualVars.getSize());
    for(std::size_t i=0;i<n; ++i){
      const RooAbsReal* r = static_cast<const RooAbsReal*>(this->_actualVars.at(i));
      double c(_coefficients[i]);
      if(c>0 && i>0) os<<"+";
      os << c<<"*"<<r->GetTitle();
    }
    os<<"]";
  }

  LinearCombination::~LinearCombination(){
    // destructor
  }
  
  TObject* LinearCombination::clone(const char* newname) const {
    // create a clone (deep copy) of this object
    LinearCombination* retval = new LinearCombination(newname);
    const std::size_t n(this->_actualVars.getSize());
    for(std::size_t i=0;i<n; ++i){
      const RooAbsReal* r = static_cast<const RooAbsReal*>(this->_actualVars.at(i));
      retval->add(this->_coefficients[i],static_cast<RooAbsReal*>(r->clone()));
    }
    return retval;
  }
  
  void LinearCombination::add(SuperFloat c,RooAbsReal* t){
    // add a new term
    _actualVars.add(*t);
    _coefficients.push_back(c);
  }

  void LinearCombination::setCoefficient(size_t idx,SuperFloat c){
    // set the coefficient with the given index
    this->_coefficients[idx]=c;
  }
  
  SuperFloat LinearCombination::getCoefficient(size_t idx){
    // get the coefficient with the given index
      return this->_coefficients[idx];
  }
  
  Double_t LinearCombination::evaluate() const {
    // call the evaluation
    SuperFloat result;
    ::assign(result,0.);
    const std::size_t n(this->_actualVars.getSize());
    for(std::size_t i=0;i<n; ++i){
      SuperFloat tmp;
      ::assign(tmp,static_cast<const RooAbsReal*>(this->_actualVars.at(i))->getVal());
      result += this->_coefficients[i] * tmp;
    }
    return ::convertToDouble(result);
  }
  
  std::list<Double_t>* LinearCombination::binBoundaries(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const {
    // Forward the plot sampling hint from the p.d.f. that defines the observable obs
    RooFIter iter = this->_actualVars.fwdIterator();
    RooAbsReal* func;
    while((func=(RooAbsReal*)iter.next())) {
      std::list<Double_t>* binb = func->binBoundaries(obs,xlo,xhi);
      if (binb) {
        return binb;
      }
    }
    return 0;
  }

  std::list<Double_t>* LinearCombination::plotSamplingHint(RooAbsRealLValue& obs, Double_t xlo, Double_t xhi) const {
    // Forward the plot sampling hint from the p.d.f. that defines the observable obs
    RooFIter iter = this->_actualVars.fwdIterator();
    RooAbsReal* func;
    while((func=(RooAbsReal*)iter.next())){
      std::list<Double_t>* hint = func->plotSamplingHint(obs,xlo,xhi);
      if (hint) {
        return hint;
      }
    }
    return 0;
  }

}
