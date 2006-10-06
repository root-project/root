/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id$
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

// -- CLASS DESCRIPTION [MISC] --
// RooRandomizeParamMCSModule is an add-on modules to RooMCStudy that
// allows you to randomize input generation parameters. Randomized generation
// parameters can be sampled from a uniform or Gaussian distribution.
// For every randomized parameter, an extra variable is added to 
// RooMCStudy::fitParDataSet() named <parname>_gen that indicates the actual
// value used for generation for each trial. 
//
// You can also choose to randomize the sum of N parameters, rather
// than a single parameter. In that case common multiplicative scale
// factor is applied to each component to bring the sum to the desired
// target value taken from either uniform or Gaussian sampling. This
// latter option is for example useful if you want to change the total
// number of expected events of an extended p.d.f/


#include "RooFitCore/RooDataSet.hh"
#include "RooFitCore/RooRealVar.hh"
#include "RooFitCore/RooRandom.hh"
#include "TString.h"
#include "RooFitCore/RooFit.hh"
#include "RooFitCore/RooFitResult.hh"
#include "RooFitCore/RooAddition.hh"
#include "RooFitCore/RooRandomizeParamMCSModule.hh"

ClassImp(RooRandomizeParamMCSModule)
  ;


RooRandomizeParamMCSModule::RooRandomizeParamMCSModule() : 
  RooAbsMCStudyModule("RooRandomizeParamMCSModule","RooRandomizeParamMCSModule"), _data(0)
{
}


RooRandomizeParamMCSModule::RooRandomizeParamMCSModule(const RooRandomizeParamMCSModule& other) : 
  RooAbsMCStudyModule(other), 
  _unifParams(other._unifParams),
  _gausParams(other._gausParams), 
  _data(0)
{
}


RooRandomizeParamMCSModule:: ~RooRandomizeParamMCSModule() 
{
  if (_data) {
    delete _data ;
  }
}



void RooRandomizeParamMCSModule::sampleUniform(RooRealVar& param, Double_t lo, Double_t hi) 
{  
  // Request uniform smearing of param in range [lo,hi] in RooMCStudy generation cycle

  // If we're already attached to a RooMCStudy, check that given param is actual generator model parameter
  // If not attached, this check is repeated at the attachment moment
  if (genParams()) {
    RooRealVar* actualPar = static_cast<RooRealVar*>(genParams()->find(param.GetName())) ;
    if (!actualPar) {
      cout << "RooRandomizeParamMCSModule::initializeInstance: variable " << param.GetName() << " is not a parameter of RooMCStudy model and is ignored!" << endl ;
      return ;
    }
  }

  _unifParams.push_back(UniParam(&param,lo,hi)) ;
}



void RooRandomizeParamMCSModule::sampleGaussian(RooRealVar& param, Double_t mean, Double_t sigma) 
{
  // Request Gaussian smearing of param in with mean 'mean' and width 'sigma' in RooMCStudy generation cycle

  // If we're already attached to a RooMCStudy, check that given param is actual generator model parameter
  // If not attached, this check is repeated at the attachment moment
  if (genParams()) {
    RooRealVar* actualPar = static_cast<RooRealVar*>(genParams()->find(param.GetName())) ;
    if (!actualPar) {
      cout << "RooRandomizeParamMCSModule::initializeInstance: variable " << param.GetName() << " is not a parameter of RooMCStudy model and is ignored!" << endl ;
      return ;
    }
  }

  _gausParams.push_back(GausParam(&param,mean,sigma)) ;
}




void RooRandomizeParamMCSModule::sampleSumUniform(const RooArgSet& paramSet, Double_t lo, Double_t hi) 
{
  // Request uniform smearing of sum of parameters in paramSet uniform smearing in range [lo,hi] in RooMCStudy generation cycle.
  // This option applies a common multiplicative factor to each parameter in paramSet to make the sum of the parameters
  // add up to the sampled value in the range [lo,hi]


  // Check that all args are RooRealVars
  RooArgSet okset ;
  TIterator* iter = paramSet.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    // Check that arg is a RooRealVar
    RooRealVar* rrv = dynamic_cast<RooRealVar*>(arg) ;
    if (!rrv) {
      cout << "RooRandomizeParamMCSModule::sampleSumUniform() ERROR: input parameter " << arg->GetName() << " is not a RooRealVar and is ignored" << endl ;
      continue;
    }
    okset.add(*rrv) ;
  }

  // If we're already attached to a RooMCStudy, check that given param is actual generator model parameter
  // If not attached, this check is repeated at the attachment moment
  RooArgSet okset2 ;
  if (genParams()) {
    TIterator* psiter = okset.createIterator() ;
    RooAbsArg* arg ;
    while ((arg=(RooAbsArg*)psiter->Next())) {
      RooRealVar* actualVar= static_cast<RooRealVar*>(genParams()->find(arg->GetName())) ;
      if (!actualVar) {
	cout << "RooRandomizeParamMCSModule::sampleSumUniform: variable " << arg->GetName() << " is not a parameter of RooMCStudy model and is ignored!" << endl ;	
      } else {
	okset2.add(*actualVar) ;
      }
    }    
    delete psiter ;
  } else {

   // If genParams() are not available, skip this check for now
   okset2.add(okset) ;

  }


  _unifParamSets.push_back(UniParamSet(okset2,lo,hi)) ;

}




void RooRandomizeParamMCSModule::sampleSumGauss(const RooArgSet& paramSet, Double_t mean, Double_t sigma) 
{
  // Request gaussian smearing of sum of parameters in paramSet uniform smearing with mean 'mean' and width 'sigma' in RooMCStudy generation cycle.
  // This option applies a common multiplicative factor to each parameter in paramSet to make the sum of the parameters
  // add up to the sampled value from the gaussian(mean,sigma)

  // Check that all args are RooRealVars
  RooArgSet okset ;
  TIterator* iter = paramSet.createIterator() ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    // Check that arg is a RooRealVar
    RooRealVar* rrv = dynamic_cast<RooRealVar*>(arg) ;
    if (!rrv) {
      cout << "RooRandomizeParamMCSModule::sampleSumGauss() ERROR: input parameter " << arg->GetName() << " is not a RooRealVar and is ignored" << endl ;
      continue;
    }
    okset.add(*rrv) ;
  }

  // If we're already attached to a RooMCStudy, check that given param is actual generator model parameter
  // If not attached, this check is repeated at the attachment moment
  RooArgSet okset2 ;
  if (genParams()) {
    TIterator* psiter = okset.createIterator() ;
    RooAbsArg* arg ;
    while ((arg=(RooAbsArg*)psiter->Next())) {
      RooRealVar* actualVar= static_cast<RooRealVar*>(genParams()->find(arg->GetName())) ;
      if (!actualVar) {
	cout << "RooRandomizeParamMCSModule::sampleSumUniform: variable " << arg->GetName() << " is not a parameter of RooMCStudy model and is ignored!" << endl ;	
      } else {
	okset2.add(*actualVar) ;
      }
    }    
    delete psiter ;
  } else {

   // If genParams() are not available, skip this check for now
   okset2.add(okset) ;

  }

  _gausParamSets.push_back(GausParamSet(okset,mean,sigma)) ;
  
}




Bool_t RooRandomizeParamMCSModule::initializeInstance()
{
  // Initialize module after attachment to RooMCStudy object

  // Loop over all uniform smearing parameters
  std::list<UniParam>::iterator uiter ;
  for (uiter= _unifParams.begin() ; uiter!= _unifParams.end() ; ++uiter) {

    // Check that listed variable is actual generator model parameter
    RooRealVar* actualPar = static_cast<RooRealVar*>(genParams()->find(uiter->_param->GetName())) ;
    if (!actualPar) {
      cout << "RooRandomizeParamMCSModule::initializeInstance: variable " << uiter->_param->GetName() << " is not a parameter of RooMCStudy model and is ignored!" << endl ;
      uiter = _unifParams.erase(uiter) ;
      continue ;
    }
    uiter->_param = actualPar ;

    // Add variable to summary dataset to hold generator value
    TString parName = Form("%s_gen",uiter->_param->GetName()) ;
    TString parTitle = Form("%s as generated",uiter->_param->GetTitle()) ;
    RooRealVar* par_gen = new RooRealVar(parName.Data(),parTitle.Data(),0) ;    
    _genParSet.addOwned(*par_gen) ;
  }
  
  // Loop over all gaussian smearing parameters
  std::list<UniParam>::iterator giter ;
  for (giter= _unifParams.begin() ; giter!= _unifParams.end() ; ++giter) {

    // Check that listed variable is actual generator model parameter
    RooRealVar* actualPar = static_cast<RooRealVar*>(genParams()->find(giter->_param->GetName())) ;
    if (!actualPar) {
      cout << "RooRandomizeParamMCSModule::initializeInstance: variable " << giter->_param->GetName() << " is not a parameter of RooMCStudy model and is ignored!" << endl ;
      giter = _unifParams.erase(giter) ;
      continue ;
    }
    giter->_param = actualPar ;

    // Add variable to summary dataset to hold generator value
    TString parName = Form("%s_gen",giter->_param->GetName()) ;
    TString parTitle = Form("%s as generated",giter->_param->GetTitle()) ;
    RooRealVar* par_gen = new RooRealVar(parName.Data(),parTitle.Data(),0) ;    
    _genParSet.addOwned(*par_gen) ;
  }


  // Loop over all uniform smearing set of parameters
  std::list<UniParamSet>::iterator usiter ;
  for (usiter= _unifParamSets.begin() ; usiter!= _unifParamSets.end() ; ++usiter) {
    
    // Check that all listed variables are actual generator model parameters
    RooArgSet actualPSet ;
    TIterator* psiter = usiter->_pset.createIterator() ;
    RooAbsArg* arg ;
    while ((arg=(RooAbsArg*)psiter->Next())) {
      RooRealVar* actualVar= static_cast<RooRealVar*>(genParams()->find(arg->GetName())) ;
      if (!actualVar) {
	cout << "RooRandomizeParamMCSModule::initializeInstance: variable " << arg->GetName() << " is not a parameter of RooMCStudy model and is ignored!" << endl ;	
      } else {
	actualPSet.add(*actualVar) ;
      }
    }    
    usiter->_pset.removeAll() ;
    usiter->_pset.add(actualPSet) ;

    // Add variables to summary dataset to hold generator values
    TIterator* iter = usiter->_pset.createIterator() ;
    RooRealVar* param ;
    while((param=(RooRealVar*)iter->Next())) {
      TString parName = Form("%s_gen",param->GetName()) ;
      TString parTitle = Form("%s as generated",param->GetTitle()) ;
      RooRealVar* par_gen = new RooRealVar(parName.Data(),parTitle.Data(),0) ;    
      _genParSet.addOwned(*par_gen) ;          
    }
  }
  
  // Loop over all gaussian smearing set of parameters
  std::list<UniParamSet>::iterator ugiter ;
  for (ugiter= _unifParamSets.begin() ; ugiter!= _unifParamSets.end() ; ++ugiter) {

    // Check that all listed variables are actual generator model parameters
    RooArgSet actualPSet ;
    TIterator* psiter = ugiter->_pset.createIterator() ;
    RooAbsArg* arg ;
    while ((arg=(RooAbsArg*)psiter->Next())) {
      RooRealVar* actualVar= static_cast<RooRealVar*>(genParams()->find(arg->GetName())) ;
      if (!actualVar) {
	cout << "RooRandomizeParamMCSModule::initializeInstance: variable " << arg->GetName() << " is not a parameter of RooMCStudy model and is ignored!" << endl ;	
      } else {
	actualPSet.add(*actualVar) ;
      }
    }    
    ugiter->_pset.removeAll() ;
    ugiter->_pset.add(actualPSet) ;

    // Add variables to summary dataset to hold generator values
    TIterator* iter = ugiter->_pset.createIterator() ;
    RooRealVar* param ;
    while((param=(RooRealVar*)iter->Next())) {
      TString parName = Form("%s_gen",param->GetName()) ;
      TString parTitle = Form("%s as generated",param->GetTitle()) ;
      RooRealVar* par_gen = new RooRealVar(parName.Data(),parTitle.Data(),0) ;    
      _genParSet.addOwned(*par_gen) ;          
    }
  }
  
  // Create new dataset to be merged with RooMCStudy::fitParDataSet
  _data = new RooDataSet("DeltaLLSigData","Additional data for Delta(-log(L)) study",_genParSet) ;

  return kTRUE ;
}


Bool_t RooRandomizeParamMCSModule::initializeRun(Int_t /*numSamples*/) 
{
  // Initialize module at beginning of RooCMStudy run

  // Clear dataset at beginning of run
  _data->reset() ;
  return kTRUE ;
}



Bool_t RooRandomizeParamMCSModule::processBeforeGen(Int_t /*sampleNum*/) 
{
  // Apply all smearings to generator parameters 

  // Apply uniform smearing to all generator parameters for which it is requested
  std::list<UniParam>::iterator uiter ;
  for (uiter= _unifParams.begin() ; uiter!= _unifParams.end() ; ++uiter) {
    Double_t newVal = RooRandom::randomGenerator()->Uniform(uiter->_lo,uiter->_hi) ;        
    cout << "RooRandomizeParamMCSModule::processBeforeGen: applying uniform smearing to generator parameter " 
	 << uiter->_param->GetName() << " in range [" << uiter->_lo << "," << uiter->_hi << "], chosen value for this sample is " << newVal << endl ;
    uiter->_param->setVal(newVal) ;

    RooRealVar* genpar = static_cast<RooRealVar*>(_genParSet.find(Form("%s_gen",uiter->_param->GetName()))) ;
    genpar->setVal(newVal) ;
  }

  // Apply gaussian smearing to all generator parameters for which it is requested
  std::list<GausParam>::iterator giter ;
  for (giter= _gausParams.begin() ; giter!= _gausParams.end() ; ++giter) {
    Double_t newVal = RooRandom::randomGenerator()->Gaus(giter->_mean,giter->_sigma) ;
    cout << "RooRandomizeParamMCSModule::processBeforeGen: applying gaussian smearing to generator parameter " 
	 << giter->_param->GetName() << " with a mean of " << giter->_mean << " and a width of " << giter->_sigma << ", chosen value for this sample is " << newVal << endl ;
    giter->_param->setVal(newVal) ;

    RooRealVar* genpar = static_cast<RooRealVar*>(_genParSet.find(Form("%s_gen",giter->_param->GetName()))) ;
    genpar->setVal(newVal) ;
  }

  // Apply uniform smearing to all sets of generator parameters for which it is requested
  std::list<UniParamSet>::iterator usiter ;
  for (usiter= _unifParamSets.begin() ; usiter!= _unifParamSets.end() ; ++usiter) {

    // Calculate new value for sum 
    Double_t newVal = RooRandom::randomGenerator()->Uniform(usiter->_lo,usiter->_hi) ;        
    cout << "RooRandomizeParamMCSModule::processBeforeGen: applying uniform smearing to sum of set of generator parameters " ;
    usiter->_pset.Print("I")  ;
    cout << " in range [" << usiter->_lo << "," << usiter->_hi << "], chosen sum value for this sample is " << newVal << endl ;

    // Determine original value of sum and calculate per-component scale factor to obtain new valye for sum
    RooAddition sumVal("sumVal","sumVal",usiter->_pset) ;
    Double_t compScaleFactor = newVal/sumVal.getVal() ;

    // Apply multiplicative correction to each term of the sum
    TIterator* iter = usiter->_pset.createIterator() ;
    RooRealVar* param ;
    while((param=(RooRealVar*)iter->Next())) {
      param->setVal(param->getVal()*compScaleFactor) ;
      RooRealVar* genpar = static_cast<RooRealVar*>(_genParSet.find(Form("%s_gen",param->GetName()))) ;
      genpar->setVal(param->getVal()) ;
    }
  }

  // Apply gaussian smearing to all sets of generator parameters for which it is requested
  std::list<GausParamSet>::iterator gsiter ;
  for (gsiter= _gausParamSets.begin() ; gsiter!= _gausParamSets.end() ; ++gsiter) {
    
    // Calculate new value for sum 
    Double_t newVal = RooRandom::randomGenerator()->Gaus(gsiter->_mean,gsiter->_sigma) ;        
    cout << "RooRandomizeParamMCSModule::processBeforeGen: applying gaussian smearing to sum of set of generator parameters " ;
    gsiter->_pset.Print("I")  ;
    cout << " with a mean of " << gsiter->_mean << " and a width of " << gsiter->_sigma << ", chosen value for this sample is " << newVal << endl ;

    // Determine original value of sum and calculate per-component scale factor to obtain new valye for sum
    RooAddition sumVal("sumVal","sumVal",gsiter->_pset) ;
    Double_t compScaleFactor = newVal/sumVal.getVal() ;

    // Apply multiplicative correction to each term of the sum
    TIterator* iter = gsiter->_pset.createIterator() ;
    RooRealVar* param ;
    while((param=(RooRealVar*)iter->Next())) {
      param->setVal(param->getVal()*compScaleFactor) ;
      RooRealVar* genpar = static_cast<RooRealVar*>(_genParSet.find(Form("%s_gen",param->GetName()))) ;
      genpar->setVal(param->getVal()) ;
    }
  }
  
  // Store generator values for all modified parameters
  _data->add(_genParSet) ;
  
  return kTRUE ;
}



RooDataSet* RooRandomizeParamMCSModule::finalizeRun() 
{
  // Return auxiliary data of this module so that it is merged with RooMCStudy::fitParDataSet()
  return _data ;
}


