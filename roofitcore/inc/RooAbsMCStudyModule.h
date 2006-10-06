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

#ifndef ROO_ABS_MC_STUDY_MODULE
#define ROO_ABS_MC_STUDY_MODULE

#include "TList.h"
#include "RooFitCore/RooArgSet.hh"
#include "RooFitCore/RooMCStudy.hh"
class RooAbsPdf;
class RooDataSet ;
class RooAbsData ;
class RooAbsGenContext ;
class RooFitResult ;
class RooPlot ;
class RooRealVar ;

class RooAbsMCStudyModule : public TNamed {
public:

  RooAbsMCStudyModule(const char* name, const char* title) ;
  RooAbsMCStudyModule(const RooAbsMCStudyModule& other) ;
  virtual ~RooAbsMCStudyModule() {} ;

  // Initializer method called upon attachement to given RooMCStudy object
  Bool_t doInitializeInstance(RooMCStudy& /*study*/) ; 

  // Initializer called immediately after attachment to RooMCStudy object and initialization of module base class
  virtual Bool_t initializeInstance() { return kTRUE ; } 

  // Method called at the beginning of each RooMCStudy run
  virtual Bool_t initializeRun(Int_t /*numSamples*/) { return kTRUE ; } 

  // Method called at the end of each RooMCStudy run. If a RooDataSet is returned, it must have a length equal to 
  // the number of toy experiments performed and will merged with the fitpar dataset of RooMCStudy.
  virtual RooDataSet* finalizeRun() { return 0 ; }

  // Method called after resetting of generator parameters to initial values and before call to generator context
  // Any modifications to generator parameters will affect next generation operation (only)
  virtual Bool_t processBeforeGen(Int_t /*sampleNum*/) { return kTRUE ; }

  // Method called after generation of toy data sample and resetting of fit parameters to initial values and before
  // actual fit is performed. Any modifications to fit parameters will apply to next fit operation. Note that setConstant
  // flag of fit parameters are not explicitly reset by RooMCStudy, so any changes made to these flags here will persist
  virtual Bool_t processBetweenGenAndFit(Int_t /*sampleNum*/) { return kTRUE ; }

  // Method called after fit has been performed.
  virtual Bool_t processAfterFit(Int_t /*sampleNum*/) { return kTRUE ; }

protected:

   // Interface methods to RooMCStudy objects, 
   // which are only functional after module has been attached to a RooMCStudy object

   // Refit model using orignal or specified data sample
   RooFitResult* refit(RooAbsData* genSample=0) { if (_mcs) return _mcs->refit(genSample) ; else return 0 ; }

   // Accessors for generated dataset and model
   RooDataSet* genSample() { return _mcs ? _mcs->_genSample : 0 ; }
   RooAbsPdf* genModel() { return _mcs ? _mcs->_genModel : 0 ; }

   // Accessor for generator context, generator parameters,	prototype data and projected dependents
   RooAbsGenContext* genContext() { return _mcs ? _mcs->_genContext : 0 ; }
   RooArgSet* genInitParams() { return _mcs ? _mcs->_genInitParams : 0 ; } 
   RooArgSet* genParams() { return _mcs ? _mcs->_genParams : 0 ; } 
   const RooDataSet* genProtoData() { return _mcs ? _mcs->_genProtoData : 0 ; }
   RooArgSet* projDeps() { return _mcs ? &_mcs->_projDeps : 0 ; }

   // Accessors for fit observables, fit model, current and initial fit parameters and NLL value
   RooArgSet* dependents() { return _mcs ? &_mcs->_dependents : 0 ; } 
   RooArgSet* allDependents() { return _mcs ? &_mcs->_allDependents : 0 ; }
   RooAbsPdf* fitModel() { return _mcs ? _mcs->_fitModel : 0 ; }
   RooArgSet* fitInitParams() { return _mcs ? _mcs->_fitInitParams : 0 ; }
   RooArgSet* fitParams() { return _mcs ? _mcs-> _fitParams : 0 ; }
   RooRealVar* nllVar() { return _mcs ? _mcs->_nllVar : 0 ; }
  
   // Accessors for fit options, generator annd MCstudy configuration flags
   const char* fitOptions() { return _mcs ? _mcs->_fitOptions.Data() : 0 ; }
   RooLinkedList* fitOptList() { return _mcs ? &_mcs->_fitOptList : 0 ; }
   Bool_t extendedGen() { return _mcs ? _mcs->_extendedGen : 0 ; }
   Bool_t binGenData() { return _mcs ? _mcs->_binGenData : 0 ; }
   Double_t numExpGen() { return _mcs ? _mcs->_nExpGen : 0 ; }
   Bool_t randProto() { return _mcs ? _mcs->_randProto : 0 ; }
   Bool_t verboseGen() { return _mcs ? _mcs->_verboseGen : 0 ; }

private:

  // Pointer to RooMCStudy object module is attached to
  RooMCStudy* _mcs ;
	
  ClassDef(RooAbsMCStudyModule,0) // Monte Carlo study manager add-on module
} ;


#endif

