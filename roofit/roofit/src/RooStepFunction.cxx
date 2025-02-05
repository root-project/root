
/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitBabar                                                      *
 * @(#)root/roofit:$Id$
 * Author:                                                                   *
 *    Tristan du Pree, Nikhef, Amsterdam, tdupree@nikhef.nl                  *
 *    Wouter Verkerke, Nikhef, Amsterdam, verkerke@nikhef.nl
 *                                                                           *
 * Copyright (c) 2009, NIKHEF. All rights reserved.                          *
 *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/** \class RooStepFunction
    \ingroup Roofit

The  Step Function is a binned function whose parameters
are the heights of each bin.

This function may be used to describe oddly shaped distributions. A RooStepFunction
has free parameters. In particular, any statistical uncertainty
used to model this efficiency may be understood with these free parameters.

Note that in contrast to RooParametricStepFunction, a RooStepFunction is NOT a PDF,
but a not-normalized function (RooAbsReal)
**/

#include <RooStepFunction.h>

#include <RooArgList.h>
#include <RooCurve.h>
#include <RooFit/Detail/MathFuncs.h>
#include <RooMath.h>
#include <RooMsgService.h>


namespace {

// Get the values for each element in a RooListProxy, using the correct normalization set.
std::span<double const> values(RooListProxy const &listProxy, std::vector<double> &valueCache)
{
   valueCache.resize(listProxy.size());
   for (std::size_t i = 0; i < listProxy.size(); ++i) {
      valueCache[i] = static_cast<RooAbsReal &>(listProxy[i]).getVal(listProxy.nset());
   }
   return valueCache;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
/// Constructor

RooStepFunction::RooStepFunction(const char* name, const char* title,
             RooAbsReal& x, const RooArgList& coefList, const RooArgList& boundaryList, bool interpolate) :
  RooAbsReal(name, title),
  _x("x", "Dependent", this, x),
  _coefList("coefList","List of coefficients",this),
  _boundaryList("boundaryList","List of boundaries",this),
  _interpolate(interpolate)
{
  _coefList.addTyped<RooAbsReal>(coefList);
  _boundaryList.addTyped<RooAbsReal>(boundaryList);

  if (_boundaryList.size()!=_coefList.size()+1) {
    coutE(InputArguments) << "RooStepFunction::ctor(" << GetName() << ") ERROR: Number of boundaries must be number of coefficients plus 1" << std::endl ;
    throw std::invalid_argument("RooStepFunction::ctor() ERROR: Number of boundaries must be number of coefficients plus 1") ;
  }

}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooStepFunction::RooStepFunction(const RooStepFunction& other, const char* name) :
  RooAbsReal(other, name),
  _x("x", this, other._x),
  _coefList("coefList",this,other._coefList),
  _boundaryList("boundaryList",this,other._boundaryList),
  _interpolate(other._interpolate)
{
}


////////////////////////////////////////////////////////////////////////////////
/// Transfer contents to std::vector for use below

double RooStepFunction::evaluate() const
{
  std::span<const double> b = values(_boundaryList, _boundaryCache);
  int nb = b.size();

  // Return zero if outside any boundaries
  if ((_x<b[0]) || (_x>b[nb-1])) return 0 ;

  if (!_interpolate) {

    // No interpolation -- Return values bin-by-bin
    for (int i=0;i<nb-1;i++){
      if (_x>b[i]&&_x<=b[i+1]) {
   return (static_cast<RooAbsReal*>(_coefList.at(i)))->getVal() ;
      }
    }
    return 0 ;

  }

    std::vector<double> c(_coefList.size()+3) ;

    // Interpolation

    // Make array of (b[0],bin centers,b[last])
    c[0] = b[0] ; c[nb] = b[nb-1] ;
    for (int i=0 ; i<nb-1 ; i++) {
      c[i+1] = (b[i]+b[i+1])/2 ;
    }

    // Make array of (0,coefficient values,0)
    int nc(0) ;
    std::vector<double> y(_coefList.size()+3) ;
    y[nc++] = 0 ;
    for(auto * coef : static_range_cast<RooAbsReal*>(_coefList)) {
      y[nc++] = coef->getVal() ;
    }
    y[nc++] = 0 ;

    for (int i=0;i<nc-1;i++){
      if (_x>c[i]&&_x<=c[i+1]) {
   double xx[2] ; xx[0]=c[i] ; xx[1]=c[i+1] ;
   double yy[2] ; yy[0]=y[i] ; yy[1]=y[i+1] ;
   return RooMath::interpolate(xx,yy,2,_x) ;
      }
    }
    return 0;
}


std::list<double> *RooStepFunction::plotSamplingHint(RooAbsRealLValue &obs, double xlo, double xhi) const
{
   if (obs.namePtr() != _x->namePtr()) {
      return nullptr;
   }

   // Use the helper function from RooCurve to make sure to get sampling hints
   // that work with the RooFitPlotting.
   return RooCurve::plotSamplingHintForBinBoundaries(values(_boundaryList, _boundaryCache), xlo, xhi);
}

int RooStepFunction::getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char * /*rangeName*/) const
{
   // We only support analytical integration if we integrate over x and there is
   // no interpolation.
   if (!_interpolate && matchArgs(allVars, analVars, _x))
      return 1;
   return 0;
}

double RooStepFunction::analyticalIntegral(int /*code*/, const char *rangeName) const
{
   return RooFit::Detail::MathFuncs::stepFunctionIntegral(_x.min(rangeName), _x.max(rangeName), _coefList.size(),
                                                          values(_boundaryList, _boundaryCache).data(),
                                                          values(_coefList, _coefCache).data());
}
