/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooAbsMCStudyModule.h,v 1.2 2007/05/11 09:11:30 verkerke Exp $
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

#include "RooArgSet.h"
#include "RooMCStudy.h"
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
  ~RooAbsMCStudyModule() override {} ;

  /// Initializer method called upon attachement to given RooMCStudy object
  bool doInitializeInstance(RooMCStudy& /*study*/) ;

  /// Initializer called immediately after attachment to RooMCStudy object and initialization of module base class
  virtual bool initializeInstance() {
    return true ;
  }

  /// Method called at the beginning of each RooMCStudy run
  virtual bool initializeRun(Int_t /*numSamples*/) {
    return true ;
  }

  /// Method called at the end of each RooMCStudy run. If a RooDataSet is returned, it must have a length equal to
  /// the number of toy experiments performed and will merged with the fitpar dataset of RooMCStudy.
  virtual RooDataSet* finalizeRun() {
    return 0 ;
  }

  /// Method called after resetting of generator parameters to initial values and before call to generator context
  /// Any modifications to generator parameters will affect next generation operation (only)
  virtual bool processBeforeGen(Int_t /*sampleNum*/) {
    return true ;
  }

  /// Method called after generation of toy data sample and resetting of fit parameters to initial values and before
  /// actual fit is performed. Any modifications to fit parameters will apply to next fit operation. Note that setConstant
  /// flag of fit parameters are not explicitly reset by RooMCStudy, so any changes made to these flags here will persist
  virtual bool processBetweenGenAndFit(Int_t /*sampleNum*/) {
    return true ;
  }

  /// Method called after fit has been performed.
  virtual bool processAfterFit(Int_t /*sampleNum*/) {
    return true ;
  }

protected:

   // Interface methods to RooMCStudy objects,
   // which are only functional after module has been attached to a RooMCStudy object

   /// Refit model using orignal or specified data sample
   RooFitResult* refit(RooAbsData* inGenSample=0) {
     if (_mcs) return _mcs->refit(inGenSample) ; else return 0 ;
   }

   /// Return generate sample
   RooAbsData* genSample() {
     return _mcs ? _mcs->_genSample : 0 ;
   }

   /// Return generator pdf
   RooAbsPdf* genModel() {
     return _mcs ? _mcs->_genModel : 0 ;
   }

   // Accessor for generator context, generator parameters, prototype data and projected dependents.
   RooAbsGenContext* genContext() {
     return _mcs ? _mcs->_genContext : 0 ;
   }

   /// Return initial value of generator model parameters
   RooArgSet* genInitParams() {
     return _mcs ? _mcs->_genInitParams : 0 ;
   }

   /// Return current value of generator model parameters
   RooArgSet* genParams() {
     return _mcs ? _mcs->_genParams : 0 ;
   }

   /// Return generator prototype data provided by user
   const RooDataSet* genProtoData() {
     return _mcs ? _mcs->_genProtoData : 0 ;
   }

   /// Return projected observables
   RooArgSet* projDeps() {
     return _mcs ? &_mcs->_projDeps : 0 ;
   }

   // Accessors for fit observables, fit model, current and initial fit parameters and NLL value

   /// Return fit model observables
   RooArgSet* dependents() {
     return _mcs ? &_mcs->_dependents : 0 ;
   }

   /// Return all observables
   RooArgSet* allDependents() {
     return _mcs ? &_mcs->_allDependents : 0 ;
   }

   /// Return fit model
   RooAbsPdf* fitModel() {
     return _mcs ? _mcs->_fitModel : 0 ;
   }

   /// Return initial value of parameters of fit model
   RooArgSet* fitInitParams() {
     return _mcs ? _mcs->_fitInitParams : 0 ;
   }

   /// Return current value of parameters of fit model
   RooArgSet* fitParams() {
     return _mcs ? _mcs-> _fitParams : 0 ;
   }

   /// Return pointer to RooRealVar holding minimized -log(L) value
   RooRealVar* nllVar() {
     return _mcs ? _mcs->_nllVar : 0 ;
   }

   // Accessors for fit options, generator and MCstudy configuration flags

   /// Return list of fit options provided by user
   RooLinkedList* fitOptList() {
     return _mcs ? &_mcs->_fitOptList : 0 ;
   }

   /// If true extended mode generation is requested
   bool extendedGen() {
     return _mcs ? _mcs->_extendedGen : 0 ;
   }

   /// If true binning of data between generating and fitting is requested
   bool binGenData() {
     return _mcs ? _mcs->_binGenData : 0 ;
   }

   /// Return expected number of events from generator model
   double numExpGen() {
     return _mcs ? _mcs->_nExpGen : 0 ;
   }

   /// If true randomization of prototype data order is requested
   bool randProto() {
     return _mcs ? _mcs->_randProto : 0 ;
   }

   /// If true verbose message in the generation step is requested
   bool verboseGen() {
     return _mcs ? _mcs->_verboseGen : 0 ;
   }

private:

  RooMCStudy* _mcs ; ///< Pointer to RooMCStudy object module is attached to

  ClassDefOverride(RooAbsMCStudyModule,0) // Monte Carlo study manager add-on module
} ;


#endif

