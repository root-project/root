/// \cond ROOFIT_INTERNAL

/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/**
\file RooFoamGenerator.cxx
\class RooFoamGenerator
\ingroup Roofitcore

Generic Monte Carlo toy generator that implement
the TFOAM sampling technique on any positively valued function.
The RooFoamGenerator generator is used by the various generator context
classes to take care of generation of observables for which p.d.fs
do not define internal methods.

The foam generator reacts to the following config options:
- nCell[123N]D
- nSample
- chatLevel
Access those using:
    myPdf->specialGeneratorConfig()->getConfigSection("RooFoamGenerator").setRealValue("nSample",1e4);

\see rf902_numgenconfig.C
**/

#include "RooFoamGenerator.h"

#include <RooAbsReal.h>
#include <RooArgSet.h>
#include <RooCategory.h>
#include <RooDataSet.h>
#include <RooErrorHandler.h>
#include <RooMsgService.h>
#include <RooNumGenConfig.h>
#include <RooRandom.h>
#include <RooRealBinding.h>
#include <RooRealVar.h>

#include "RooNumGenFactory.h"

namespace {

// Lightweight interface adaptor that binds a RooAbsPdf to TFOAM.
class RooTFoamBinding : public TFoamIntegrand {
public:
   RooTFoamBinding(const RooAbsReal &pdf, const RooArgSet &observables)
      : _binding(std::make_unique<RooRealBinding>(pdf, observables, &_nset, false, nullptr))
   {
      _nset.add(observables);
   }

   double Density(Int_t ndim, double *xvec) override
   {
      double x[10];
      for (int i = 0; i < ndim; i++) {
         x[i] = xvec[i] * (_binding->getMaxLimit(i) - _binding->getMinLimit(i)) + _binding->getMinLimit(i);
      }
      double ret = (*_binding)(x);
      return ret < 0 ? 0 : ret;
   }

   RooRealBinding &binding() { return *_binding; }

private:
   RooArgSet _nset;
   std::unique_ptr<RooRealBinding> _binding;
};

} // namespace

////////////////////////////////////////////////////////////////////////////////
/// Register RooIntegrator1D, is parameters and capabilities with RooNumIntFactory

void RooFoamGenerator::registerSampler(RooNumGenFactory& fact)
{
  // Register RooIntegrator1D, is parameters and capabilities with RooNumIntFactory
  RooRealVar nSample("nSample","Number of samples per cell",200,0,1e6) ;
  RooRealVar nCell1D("nCell1D","Number of cells for 1-dim generation",30,0,1e6) ;
  RooRealVar nCell2D("nCell2D","Number of cells for 2-dim generation",500,0,1e6) ;
  RooRealVar nCell3D("nCell3D","Number of cells for 3-dim generation",5000,0,1e6) ;
  RooRealVar nCellND("nCellND","Number of cells for N-dim generation",10000,0,1e6) ;
  RooRealVar chatLevel("chatLevel","TFOAM 'chat level' (verbosity)",0,0,2) ;

  RooFoamGenerator* proto = new RooFoamGenerator ;
  fact.storeProtoSampler(proto,RooArgSet(nSample,nCell1D,nCell2D,nCell3D,nCellND,chatLevel)) ;
}




////////////////////////////////////////////////////////////////////////////////

RooFoamGenerator::RooFoamGenerator(const RooAbsReal &func, const RooArgSet &genVars, const RooNumGenConfig& config, bool verbose, const RooAbsReal* maxFuncVal) :
  RooAbsNumGenerator(func,genVars,verbose,maxFuncVal)
{
  _binding = std::make_unique<RooTFoamBinding>(*_funcClone,_realVars) ;

  _tfoam = std::make_unique<TFoam>("TFOAM") ;
  _tfoam->SetkDim(_realVars.size()) ;
  _tfoam->SetRho(_binding.get()) ;
  _tfoam->SetPseRan(RooRandom::randomGenerator()) ;
  switch(_realVars.size()) {
  case 1:_tfoam->SetnCells((Int_t)config.getConfigSection("RooFoamGenerator").getRealValue("nCell1D")) ; break ;
  case 2:_tfoam->SetnCells((Int_t)config.getConfigSection("RooFoamGenerator").getRealValue("nCell2D")) ; break ;
  case 3:_tfoam->SetnCells((Int_t)config.getConfigSection("RooFoamGenerator").getRealValue("nCell3D")) ; break ;
  default:_tfoam->SetnCells((Int_t)config.getConfigSection("RooFoamGenerator").getRealValue("nCellND")) ; break ;
  }
  _tfoam->SetnSampl((Int_t)config.getConfigSection("RooFoamGenerator").getRealValue("nSample")) ;
  _tfoam->SetPseRan(RooRandom::randomGenerator()) ;
  _tfoam->SetChat((Int_t)config.getConfigSection("RooFoamGenerator").getRealValue("chatLevel")) ;
  _tfoam->Initialize() ;

  _vec.resize(_realVars.size());
  _xmin.resize(_realVars.size());
  _range.resize(_realVars.size());

  Int_t i(0) ;
  for (auto *var : static_range_cast<RooRealVar const*>(_realVars)) {
    _xmin[i] = var->getMin() ;
    _range[i] = var->getMax() - var->getMin() ;
    i++ ;
  }
}

////////////////////////////////////////////////////////////////////////////////
/// are we actually generating anything? (the cache always contains at least our function value)

const RooArgSet *RooFoamGenerator::generateEvent(UInt_t /*remaining*/, double& /*resampleRatio*/)
{
  const RooArgSet *event= _cache->get();
  if(event->size() == 1) return event;

  _tfoam->MakeEvent() ;
  _tfoam->GetMCvect(_vec.data()) ;

  // Transfer contents to dataset
  Int_t i(0) ;
  for (auto arg : _realVars) {
    auto var = static_cast<RooRealVar*>(arg);
    var->setVal(_xmin[i] + _range[i]*_vec[i]) ;
    i++ ;
  }
  return &_realVars ;
}

std::string const& RooFoamGenerator::generatorName() const {
   static const std::string name = "RooFoamGenerator";
   return name;
}

/// \endcond
