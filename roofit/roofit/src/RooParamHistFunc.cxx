/*
 * Project: RooFit
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

/** \class RooParamHistFunc
 *  \ingroup Roofit
 * A histogram function that assigns scale parameters to every bin. Instead of the bare bin contents,
 * it therefore yields:
 * \f[
 *  \gamma_{i} * \mathrm{bin}_i
 * \f]
 *
 * The \f$ \gamma_i \f$ can therefore be used to parametrise statistical uncertainties of the histogram
 * template. In conjunction with a constraint term, this can be used to implement the Barlow-Beeston method.
 * The constraint can be implemented using RooHistConstraint.
 *
 * See also the tutorial rf709_BarlowBeeston.C
 */

#include <RooParamHistFunc.h>
#include <RooRealVar.h>
#include <RooFitImplHelpers.h>

ClassImp(RooParamHistFunc);

RooParamHistFunc::RooParamHistFunc(const char *name, const char *title, RooDataHist &dh, const RooAbsArg &x,
                                   const RooParamHistFunc *paramSource, bool paramRelative)
   : RooAbsReal(name, title), _x("x", "x", this), _p("p", "p", this), _dh(dh), _relParam(paramRelative)
{
   // Populate x with observables
   _x.add(x);

   if (paramSource) {
      // Now populate p with existing parameters
      _p.add(paramSource->_p);
      return;
   }

   // Now populate p with parameters
   RooArgSet allVars;
   for (Int_t i = 0; i < _dh.numEntries(); i++) {
      _dh.get(i);
      const char *vname = Form("%s_gamma_bin_%i", GetName(), i);
      RooRealVar *var = new RooRealVar(vname, vname, 0, 1000);
      var->setVal(_relParam ? 1 : _dh.weight());
      var->setError(_relParam ? 1 / sqrt(_dh.weight()) : sqrt(_dh.weight()));
      var->setConstant(true);
      allVars.add(*var);
      _p.add(*var);
   }
   addOwnedComponents(allVars);
}

////////////////////////////////////////////////////////////////////////////////

RooParamHistFunc::RooParamHistFunc(const RooParamHistFunc& other, const char* name) :
  RooAbsReal(other,name),
  _x("x",this,other._x),
  _p("p",this,other._p),
  _dh(other._dh),
  _relParam(other._relParam)
{
}

////////////////////////////////////////////////////////////////////////////////

double RooParamHistFunc::evaluate() const
{
  Int_t idx = ((RooDataHist&)_dh).getIndex(_x,true) ;
  double ret = (static_cast<RooAbsReal*>(_p.at(idx)))->getVal() ;
  return _relParam ? ret * getNominal(idx) : ret;
}

void RooParamHistFunc::translate(RooFit::Detail::CodeSquashContext &ctx) const
{
   std::string const &idx = _dh.calculateTreeIndexForCodeSquash(this, ctx, _x);
   std::string arrName = ctx.buildArg(_p);
   std::string result = arrName + "[" + idx + "]";
   if (_relParam) {
      // get weight[idx] * binv[idx]. Here we get the bin volume for the first element as we assume the distribution to
      // be binned uniformly.
      double binV = _dh.binVolume(0);
      result += " * " + _dh.declWeightArrayForCodeSquash(this, ctx, false, idx) + " * " + std::to_string(binV);
   }
   ctx.addResult(this, result);
}

////////////////////////////////////////////////////////////////////////////////

double RooParamHistFunc::getActual(Int_t ibin)
{
  return (static_cast<RooAbsReal&>(_p[ibin])).getVal() ;
}

////////////////////////////////////////////////////////////////////////////////

void RooParamHistFunc::setActual(Int_t ibin, double newVal)
{
  (static_cast<RooRealVar&>(_p[ibin])).setVal(newVal) ;
}

////////////////////////////////////////////////////////////////////////////////

double RooParamHistFunc::getNominal(Int_t ibin) const
{
  _dh.get(ibin) ;
  return _dh.weight() ;
}

////////////////////////////////////////////////////////////////////////////////

double RooParamHistFunc::getNominalError(Int_t ibin) const
{
  _dh.get(ibin) ;
  return _dh.weightError() ;
}

////////////////////////////////////////////////////////////////////////////////
/// Return sampling hint for making curves of (projections) of this function
/// as the recursive division strategy of RooCurve cannot deal efficiently
/// with the vertical lines that occur in a non-interpolated histogram

std::list<double>* RooParamHistFunc::plotSamplingHint(RooAbsRealLValue& obs, double xlo, double xhi) const
{
  // Check that observable is in dataset, if not no hint is generated
  RooAbsLValue* lvarg = dynamic_cast<RooAbsLValue*>(_dh.get()->find(obs.GetName())) ;
  if (!lvarg) {
    return nullptr ;
  }

  // Retrieve position of all bin boundaries
  const RooAbsBinning* binning = lvarg->getBinningPtr(nullptr);
  double* boundaries = binning->array() ;

  std::list<double>* hint = new std::list<double> ;

  // Widen range slightly
  xlo = xlo - 0.01*(xhi-xlo) ;
  xhi = xhi + 0.01*(xhi-xlo) ;

  double delta = (xhi-xlo)*1e-8 ;

  // Construct array with pairs of points positioned epsilon to the left and
  // right of the bin boundaries
  for (Int_t i=0 ; i<binning->numBoundaries() ; i++) {
    if (boundaries[i]>=xlo && boundaries[i]<=xhi) {
      hint->push_back(boundaries[i]-delta) ;
      hint->push_back(boundaries[i]+delta) ;
    }
  }

  return hint ;
}

////////////////////////////////////////////////////////////////////////////////
/// Return sampling hint for making curves of (projections) of this function
/// as the recursive division strategy of RooCurve cannot deal efficiently
/// with the vertical lines that occur in a non-interpolated histogram

std::list<double>* RooParamHistFunc::binBoundaries(RooAbsRealLValue& obs, double xlo, double xhi) const
{
  // Check that observable is in dataset, if not no hint is generated
  RooAbsLValue* lvarg = dynamic_cast<RooAbsLValue*>(_dh.get()->find(obs.GetName())) ;
  if (!lvarg) {
    return nullptr ;
  }

  // Retrieve position of all bin boundaries
  const RooAbsBinning* binning = lvarg->getBinningPtr(nullptr);
  double* boundaries = binning->array() ;

  std::list<double>* hint = new std::list<double> ;

  // Construct array with pairs of points positioned epsilon to the left and
  // right of the bin boundaries
  for (Int_t i=0 ; i<binning->numBoundaries() ; i++) {
    if (boundaries[i]>=xlo && boundaries[i]<=xhi) {
      hint->push_back(boundaries[i]) ;
    }
  }

  return hint ;
}

////////////////////////////////////////////////////////////////////////////////
/// Advertise that all integrals can be handled internally.

Int_t RooParamHistFunc::getAnalyticalIntegralWN(RooArgSet& allVars, RooArgSet& analVars,
                  const RooArgSet* /*normSet*/, const char* /*rangeName*/) const
{
  // Simplest scenario, integrate over all dependents
  std::unique_ptr<RooAbsCollection> allVarsCommon{allVars.selectCommon(_x)};
  bool intAllObs = (allVarsCommon->size()==_x.size()) ;
  if (intAllObs && matchArgs(allVars,analVars,_x)) {
    return 1 ;
  }

  return 0 ;
}

////////////////////////////////////////////////////////////////////////////////
/// Implement analytical integrations by doing appropriate weighting from  component integrals
/// functions to integrators of components

double RooParamHistFunc::analyticalIntegralWN(Int_t code, const RooArgSet* /*normSet2*/,const char* rangeName) const
{
  // Supports only the scenario of integration over all dependents
  R__ASSERT(code==1) ;

  // The logic for summing over the histogram is borrowed from RooHistPdf with some differences:
  //
  //  - a lambda function is used to inject the parameters for bin scaling into the RooDataHist::sum method
  //
  //  - for simplicity, there is no check for the possibility of full-range integration with another overload of
  //    RooDataHist::sum
  std::map<const RooAbsArg*, std::pair<double, double> > ranges;
  for (const auto obs : _x) {
    ranges[obs] = RooHelpers::getRangeOrBinningInterval(obs, rangeName);
  }

  auto getBinScale = [&](int iBin){ return static_cast<const RooAbsReal&>(_p[iBin]).getVal(); };

  RooArgSet sliceSet{};
  return const_cast<RooDataHist&>(_dh).sum(_x, sliceSet, true, false, ranges, getBinScale);
}
