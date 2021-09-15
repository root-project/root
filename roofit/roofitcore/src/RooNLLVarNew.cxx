/**
\file RooNLLVarNew.cxx
\class RooNLLVarNew
\ingroup Roofitcore

This is a simple class designed to produce the nll values needed by the fitter.
In contrast to the `RooNLLVar` class, any logic except the bare minimum has been
transfered away to other classes, like the `RooFitDriver`. This class also calls
functions from `RooBatchCompute` library to provide faster computation times.
**/

#include "RooNLLVarNew.h"
#include "rbc.h"

#include <numeric>
#include <stdexcept>
#include <vector>

using namespace ROOT::Experimental;

ClassImp(RooNLLVarNew);

/** Contstruct a RooNLLVarNew

\param pdf The pdf for which the nll is computed for
\param observables The observabes of the pdf
\param weight A pointer to the weight variable (if exists)
\param constraints A pointer to the constraints (if exist)
\param isExtended Set to true if this is an extended fit
**/
RooNLLVarNew::RooNLLVarNew(const char *name, const char *title,
                           RooAbsPdf &pdf, RooArgSet const& observables,
                           RooAbsReal* weight, RooAbsReal* constraints, bool isExtended)
   : RooAbsReal(name, title),
    _pdf{"pdf", "pdf", this, pdf},
    _observables{&observables},
    _isExtended{isExtended}
{
   if(weight) {
       _weight = std::make_unique<RooTemplateProxy<RooAbsReal>>("_weight", "_weight", this, *weight);
    }
   if(constraints) {
       _constraints = constraints;
   }
}

RooNLLVarNew::RooNLLVarNew(const RooNLLVarNew &other, const char *name)
   : RooAbsReal(other, name),
    _pdf{"pdf", this, other._pdf},
    _observables{other._observables}
{
    if(other._weight)
        _weight = std::make_unique<RooTemplateProxy<RooAbsReal>>("_weight", this, *other._weight);
    if(other._constraints)
        _constraints = other._constraints;
}

/** Compute multiple negative logs of propabilities

\param dispatch A pointer to the RooBatchCompute library interface used for this computation
\param output An array of doubles where the computation results will be stored
\param nEvents The number of events to be processed
\param dataMap A map containing spans with the input data for the computation
**/
void RooNLLVarNew::computeBatch(rbc::RbcInterface* dispatch, double* output, size_t nEvents, rbc::DataMap& dataMap) const 
{
  rbc::VarVector vars = {&*_pdf};
  if (_weight) {
      vars.push_back(&**_weight);
  }
  rbc::ArgVector args = {static_cast<double>(vars.size()-1)};
  dispatch->compute(rbc::NegativeLogarithms, output, nEvents, dataMap, vars, args);

  if (_isExtended && _sumWeight == 0.0) {
    if(!_weight) {
      _sumWeight = nEvents;
    } else {
      auto weightSpan = dataMap[&**_weight];
      _sumWeight = weightSpan.size() == 1 ? weightSpan[0] * nEvents
                                          : dispatch->sumReduce(dataMap[&**_weight].data(), nEvents);
    }
  }
}

/** Reduce an array of nll values to the sum of them

\param dispatch A pointer to the RooBatchCompute library interface used for this computation
\param input The input array with the nlls to be reduced
\param nEvents the number of events to be processed
**/
double RooNLLVarNew::reduce(rbc::RbcInterface* dispatch, const double* input, size_t nEvents) const
{
  double nll = dispatch->sumReduce(input, nEvents);
  if (_constraints) {
    nll += _constraints->getVal();
  }
  if (_isExtended) {
    assert(_sumWeight != 0.0);
    nll += _pdf->extendedTerm(_sumWeight, _observables);
  }
  return nll;
}

double RooNLLVarNew::getValV(const RooArgSet *) const
{
   throw std::runtime_error("RooNLLVarNew::getValV was called directly which should not happen!");
}

double RooNLLVarNew::evaluate() const
{
   throw std::runtime_error("RooNLLVarNew::evaluate was called directly which should not happen!");
}

RooSpan<double> RooNLLVarNew::evaluateSpan(rbc::RunContext &, const RooArgSet *) const
{
   throw std::runtime_error("RooNLLVarNew::evaluatSpan was called directly which should not happen!");
}

RooSpan<const double> RooNLLVarNew::getValues(rbc::RunContext &, const RooArgSet *) const
{
   throw std::runtime_error("RooNLLVarNew::getValues was called directly which should not happen!");
}

bool RooNLLVarNew::getParameters(const RooArgSet *depList, RooArgSet& outSet, bool stripDisconnected) const
{
   return _pdf->getParameters(depList, outSet, stripDisconnected);
}
