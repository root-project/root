/*****************************************************************************
 * Project: RooFit                                                           *
 *                                                                           *
 * authors:                                                                  *
 *  Lydia Brenner (lbrenner@cern.ch), Carsten Burgard (cburgard@cern.ch)     *
 *  Katharina Ecker (kecker@cern.ch), Adam Kaluza      (akaluza@cern.ch)     *
 * Copyright (c) 2000-2007, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 *****************************************************************************/

#ifndef ROO_HC_MORPH
#define ROO_HC_MORPH

#include "RooAbsPdf.h"
#include "RooRealSumPdf.h"
#include "RooRealSumFunc.h"
#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooArgList.h"
#include "RooProduct.h"
#include "TMatrixD.h"
#include "RooAbsArg.h"
#include "RooLagrangianMorphing.h"

class RooWorkspace;
class RooParamHistFunc;
class TPair;
class TFolder;
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

  // these are a couple of helper functions for use with the Higgs Characterization (HC) Model
  // arXiv: 1306.6464
  RooArgSet makeHCggFCouplings(RooAbsCollection& kappas);
  RooArgSet makeHCVBFCouplings(RooAbsCollection& kappas);
  RooArgSet makeHCHWWCouplings(RooAbsCollection& kappas);
  RooArgSet makeHCHZZCouplings(RooAbsCollection& kappas);
  RooArgSet makeHCHllCouplings(RooAbsCollection& kappas);

#define MAKE_ROOLAGRANGIANMORPH(CLASSNAME) public:            \
  CLASSNAME(const char *name, const char *title, const char* fileName, const char* obsName, const RooLagrangianMorphing::RooLagrangianMorphConfig& config, const char* basefolder, const RooArgList& folders, const char* objFilter = 0, bool allowNegativeYields=true) : RooLagrangianMorph(name,title,fileName,obsName,config,basefolder,folders,objFilter,allowNegativeYields){}; \
  CLASSNAME(const char *name, const char *title, const char* fileName, const char* obsName,const RooLagrangianMorphing::RooLagrangianMorphConfig& config, const RooArgList& folders, const char* objFilter = 0, bool allowNegativeYields=true) : RooLagrangianMorph(name,title,fileName,obsName,config,folders,objFilter,allowNegativeYields){} \
  CLASSNAME(const char *name, const char *title, const char* fileName, const char* obsName, const RooArgList& folders, const char* objFilter = 0, bool allowNegativeYields=true) : RooLagrangianMorph (name,title,fileName,obsName,RooLagrangianMorphing::RooLagrangianMorphConfig(),folders,objFilter,allowNegativeYields){this->makeCouplings();} \
  CLASSNAME(const CLASSNAME& other, const char* newname) : RooLagrangianMorph(other,newname){ }; \
  CLASSNAME():RooLagrangianMorph (){ };                          \
  virtual ~CLASSNAME(){};                        \
  virtual TObject* clone(const char* newname) const override { return new CLASSNAME(*this,newname); };

////////////////////////////////////////////////////////////////////////////////////////////////
// DERIVED CLASSES to implement specific PHYSICS ///////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////

class RooHCggfWWMorph : public RooLagrangianMorphing::RooLagrangianMorph {
  MAKE_ROOLAGRANGIANMORPH(RooHCggfWWMorph)
  ClassDefOverride(RooHCggfWWMorph,1)
  protected:
  void makeCouplings(){ 
    RooArgSet kappas("ggfWW");
    this->_config.setCouplings(makeHCggFCouplings(kappas),makeHCHWWCouplings(kappas));
    this->setup(true);
  }
};

class RooHCvbfWWMorph : public RooLagrangianMorphing::RooLagrangianMorph {
  MAKE_ROOLAGRANGIANMORPH(RooHCvbfWWMorph)
  ClassDefOverride(RooHCvbfWWMorph,1)
  protected:
  void makeCouplings(){
    RooArgSet kappas("vbfWW");
    this->_config.setCouplings(makeHCVBFCouplings(kappas),makeHCHWWCouplings(kappas));
    this->setup(true);
  }
};
class RooHCggfZZMorph : public RooLagrangianMorphing::RooLagrangianMorph {
  MAKE_ROOLAGRANGIANMORPH(RooHCggfZZMorph)
  ClassDefOverride(RooHCggfZZMorph,1)
  protected:
  void makeCouplings(){
    RooArgSet kappas("ggfZZ");
    this->_config.setCouplings(makeHCggFCouplings(kappas),makeHCHZZCouplings(kappas));
    this->setup(true);
  }
};
class RooHCvbfZZMorph : public RooLagrangianMorphing::RooLagrangianMorph {
  MAKE_ROOLAGRANGIANMORPH(RooHCvbfZZMorph)
  ClassDefOverride(RooHCvbfZZMorph,1)
  protected:
  void makeCouplings(){
    RooArgSet kappas("vbfZZ");
    this->_config.setCouplings(makeHCVBFCouplings(kappas),makeHCHZZCouplings(kappas));
    this->setup(true);
  }
};
class RooHCvbfMuMuMorph : public RooLagrangianMorphing::RooLagrangianMorph {
  MAKE_ROOLAGRANGIANMORPH(RooHCvbfMuMuMorph)
  ClassDefOverride(RooHCvbfMuMuMorph,1)
  protected:
  void makeCouplings(){
    RooArgSet kappas("vbfMuMu");
    this->_config.setCouplings(makeHCVBFCouplings(kappas),makeHCHllCouplings(kappas));
    this->setup(true);
  }
};

#ifndef __CINT__
ClassImp(RooHCggfWWMorph)
ClassImp(RooHCvbfWWMorph)
ClassImp(RooHCggfZZMorph)
ClassImp(RooHCvbfZZMorph)
ClassImp(RooHCvbfMuMuMorph)
#endif
//
//////////////////////////////////////////////////////////////////////////////////////////////////
//// DERIVED CLASSES to implement specific PHYSICS ///////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//class RooSMEFTggfMorph : public RooLagrangianMorphing::RooLagrangianMorph {
//  MAKE_ROOLAGRANGIANMORPH(RooSMEFTggfMorph)
//  ClassDefOverride(RooSMEFTggfMorph,1)
//  protected:
//  void makeCouplings(){
//    RooArgSet kappas("ggf");
//    this->_config.setCouplings(makeSMEFTggFCouplings(kappas));
//    this->setup(true);
//  }
//};
//
//class RooSMEFTvbfMorph : public RooLagrangianMorphing::RooLagrangianMorph {
//  MAKE_ROOLAGRANGIANMORPH(RooSMEFTvbfMorph)
//  ClassDefOverride(RooSMEFTvbfMorph,1)
//  protected:
//  void makeCouplings(){
//    RooArgSet kappas("vbf");
//    this->_config.setCouplings(makeSMEFTVBFCouplings(kappas));
//    this->setup(true);
//  }
//};
//
//class RooSMEFTggfWWMorph : public RooLagrangianMorphing::RooLagrangianMorph {
//  MAKE_ROOLAGRANGIANMORPH(RooSMEFTggfWWMorph)
//  ClassDefOverride(RooSMEFTggfWWMorph,1)
//  protected:
//  void makeCouplings(){
//    RooArgSet kappas("ggfWW");
//    this->_config.setCouplings(makeSMEFTggFCouplings(kappas),makeSMEFTHWWCouplings(kappas));
//    this->setup(true);
//  }
//};
//
//class RooSMEFTvbfWWMorph : public RooLagrangianMorphing::RooLagrangianMorph {
//  MAKE_ROOLAGRANGIANMORPH(RooSMEFTvbfWWMorph)
//  ClassDefOverride(RooSMEFTvbfWWMorph,1)
//  protected:
//  void makeCouplings(){
//    RooArgSet kappas("vbfWW");
//    this->_config.setCouplings(makeSMEFTVBFCouplings(kappas),makeSMEFTHWWCouplings(kappas));
//    this->setup(true);
//  }
//};
//
//#ifndef __CINT__
//ClassImp(RooSMEFTggfMorph)
//ClassImp(RooSMEFTvbfMorph)
////ClassImp(RooSMEFTzhlepMorph)
////ClassImp(RooSMEFTwhlepMorph)
////ClassImp(RooSMEFTtthMorph)
//
//ClassImp(RooSMEFTggfMorph)
//ClassImp(RooSMEFTvbfMorph)
////ClassImp(RooSMEFTzhlepMorph)
////ClassImp(RooSMEFTwhlepMorph)
////ClassImp(RooSMEFTtthMorph)
//
//ClassImp(RooSMEFTggfWWMorph)
//ClassImp(RooSMEFTvbfWWMorph)
//
//ClassImp(RooSMEFTggfWWMorph)
//ClassImp(RooSMEFTvbfWWMorph)
//#endif

#endif
