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

  
////////////////////////////////////////////////////////////////////////////////////////////////
//
// RooLagrangianMorph
//
// The RooLagrangianMorph is a type of RooAbsReal that allows to morph
// different input EFT samples to some arbitrary output EFT
// sample, as long as the desired set of output parameters lie
// within the realm spanned by the input samples. More
// specifically, it expects as an input a TFile (or TDirectory)
// with the following layout:
//
// TDirectory
//  |-sample1
//  | |-param_card     // a TH1 which encodes the EFT parameter values used for this sample
//  | | histogram      // a TH1 with a distribution of some physical observable
//  | |-subfolder1     // a subfolder (optional)
//  | | |-histogram1   // another TH1 with a distribution of some physical observable
//  | | |-histogram2   // another TH1 with a distribution of some physical observalbe
//  | | ...            // more of these
//  |-sample2
//  | |-param_card     // a TH1 which encodes the EFT parameter values used for this sample
//  | | histogram      // a TH1 with a distribution of the same physical observable as above
//  | | ...
//  | ...
//
// The RooLagrangianMorph operates on this structure, extracts data
// and meta-data and produces a morphing result as a RooRealSum
// consisting of the input histograms with appropriate prefactors.
//
// The histograms to be morphed can be accessed via their paths in
// the respective sample, e.g. using
//    "histogram"
// or "subfolder1/histogram1"
// or "some/deep/path/to/some/subfolder/histname"
//
////////////////////////////////////////////////////////////////////////////////////////////////


#ifndef ROO_LAGRANGIAN_MORPH
#define ROO_LAGRANGIAN_MORPH

#include "RooAbsReal.h"
#include "RooRealSumPdf.h"
#include "RooRealSumFunc.h"
#include "RooWrapperPdf.h"
#include "RooAbsReal.h"
#include "RooRealVar.h"
#include "RooArgList.h"
#include "RooProduct.h"
#include "TMatrixD.h"
#include "RooAbsArg.h"

class RooWorkspace;
class RooParamHistFunc;
class TPair;
class TFolder;
#include <vector>
#include <string>
#include <iostream>
#include <fstream>

namespace RooLagrangianMorphing {
 class RooLagrangianMorph;
 typedef std::map<const std::string,double> ParamSet;
  typedef std::map<const std::string,int> FlagSet;
  typedef std::map<const std::string,RooLagrangianMorphing::ParamSet > ParamMap;
  typedef std::map<const std::string,RooLagrangianMorphing::FlagSet > FlagMap;
  extern bool gAllowExceptions;
  double implementedPrecision();
  RooWorkspace* makeCleanWorkspace(RooWorkspace* oldWS, const char* newName = 0, const char* mcname = "ModelConfig", bool keepData = false);
  void importToWorkspace(RooWorkspace* ws, const RooAbsReal* object);
  void importToWorkspace(RooWorkspace* ws, RooAbsData* object);
  void append(RooLagrangianMorphing::ParamSet& set, const char* str, double val);
  void append(RooLagrangianMorphing::ParamMap& map, const char* str, RooLagrangianMorphing::ParamSet& set);
    
  void writeMatrixToFile(const TMatrixD& matrix, const char* fname);
  void writeMatrixToStream(const TMatrixD& matrix, std::ostream& stream);
  TMatrixD readMatrixFromFile(const char* fname);
  TMatrixD readMatrixFromStream(std::istream& stream);

  RooDataHist* makeDataHistogram(TH1* hist, RooRealVar* observable, const char* histname = NULL);
  void setDataHistogram(TH1* hist, RooRealVar* observable, RooDataHist* dh);
  void printDataHistogram(RooDataHist* hist, RooRealVar* obs);

  int countSamples(std::vector<RooArgList*>& vertices);
  int countSamples(int nprod, int ndec, int nboth);

  TPair* makeCrosssectionContainer(double xs, double unc);
  std::map<std::string,std::string> createWeightStrings(const RooLagrangianMorphing::ParamMap& inputs, const std::vector<std::string>& couplings);
  std::map<std::string,std::string> createWeightStrings(const RooLagrangianMorphing::ParamMap& inputs, const std::vector<std::vector<std::string> >& vertices);
  std::map<std::string,std::string> createWeightStrings(const RooLagrangianMorphing::ParamMap& inputs, const std::vector<RooArgList*>& vertices, RooArgList& couplings);
  std::map<std::string,std::string> createWeightStrings(const RooLagrangianMorphing::ParamMap& inputs, const std::vector<RooArgList*>& vertices, RooArgList& couplings, const RooLagrangianMorphing::FlagMap& flagValues, const RooArgList& flags, const std::vector<RooArgList*>& nonInterfering);
  RooArgSet createWeights(const RooLagrangianMorphing::ParamMap& inputs, const std::vector<RooArgList*>& vertices, RooArgList& couplings, const RooLagrangianMorphing::FlagMap& inputFlags, const RooArgList& flags, const std::vector<RooArgList*>& nonInterfering);
  RooArgSet createWeights(const RooLagrangianMorphing::ParamMap& inputs, const std::vector<RooArgList*>& vertices, RooArgList& couplings);

  // some helpers to make the template mapping work
  template<class Base> struct Internal;
  template<> struct Internal<RooAbsReal> { typedef RooRealSumFunc Type; };

 class RooLagrangianMorphConfig {
  friend class RooLagrangianMorph;
  public:

  RooLagrangianMorphConfig();
  RooLagrangianMorphConfig(const RooAbsCollection& couplings);
  RooLagrangianMorphConfig(const RooAbsCollection& prodCouplings, const RooAbsCollection& decCouplings);
  RooLagrangianMorphConfig(const RooLagrangianMorphConfig& other);
  void setCouplings(const RooAbsCollection& couplings);
  void setCouplings(const RooAbsCollection& prodCouplings, const RooAbsCollection& decCouplings);
  template <class T> void setVertices(const std::vector<T>& vertices);
  template <class T> void setDiagrams(const std::vector<std::vector<T> >& diagrams);
  template <class T> void setNonInterfering(const std::vector<T*>& nonInterfering);
  template <class T> void addDiagrams(const std::vector<T>& diagrams);

  virtual ~RooLagrangianMorphConfig();

  protected:
  std::vector<RooListProxy*> _vertices;
  RooListProxy _couplings;
  RooListProxy _prodCouplings;
  RooListProxy _decCouplings;
  std::vector<std::vector<RooListProxy*> > _configDiagrams;
  std::vector<RooListProxy*> _nonInterfering;
};

  class RooLagrangianMorph : public RooAbsReal {
    using InternalType = typename Internal<RooAbsReal>::Type;
  public:
    RooLagrangianMorph();
    RooLagrangianMorph(const char *name, const char *title, const char* fileName, const char* obsName, const RooLagrangianMorphConfig& config, const char* objFilter = 0, bool allowNegativeYields=true);
    RooLagrangianMorph(const char *name, const char *title, const char* fileName, const char* obsName, const RooLagrangianMorphConfig& config, const char* basefolder, const RooArgList& folders, const char* objFilter = 0, bool allowNegativeYields=true);
    RooLagrangianMorph(const char *name, const char *title, const char* fileName, const char* obsName, const RooLagrangianMorphConfig& config, const RooArgList& folders, const char* objFilter = 0, bool allowNegativeYields=true);
    RooLagrangianMorph(const char *name, const char *title, const char* fileName, const char* obsName, const char* objFilter = 0, bool allowNegativeYields=true);
    RooLagrangianMorph(const char *name, const char *title, const char* fileName, const char* obsName, const char* basefolder, const RooArgList& folders, const char* objFilter = 0, bool allowNegativeYields=true);
    RooLagrangianMorph(const char *name, const char *title, const char* fileName, const char* obsName, const RooArgList& folders, const char* objFilter = 0, bool allowNegativeYields=true);
    RooLagrangianMorph(const RooLagrangianMorph& other, const char *newName);
 
    virtual ~RooLagrangianMorph();

    virtual std::list<Double_t>* binBoundaries(RooAbsRealLValue& /*obs*/, Double_t /*xlo*/, Double_t /*xhi*/) const override;
    virtual std::list<Double_t>* plotSamplingHint(RooAbsRealLValue& /*obs*/, Double_t /*xlo*/, Double_t /*xhi*/) const override;
    virtual Bool_t isBinnedDistribution(const RooArgSet& obs) const override;
    virtual Double_t evaluate() const override;
    virtual TObject* clone(const char* newname) const override;
    virtual Double_t getValV(const RooArgSet* set=0) const override;

    virtual Bool_t checkObservables(const RooArgSet *nset) const override;
    virtual Bool_t forceAnalyticalInt(const RooAbsArg &arg) const override;
    virtual Int_t getAnalyticalIntegralWN(RooArgSet &allVars, RooArgSet &numVars, const RooArgSet *normSet, const char *rangeName = 0) const override;
    virtual Double_t analyticalIntegralWN(Int_t code, const RooArgSet *normSet, const char *rangeName = 0) const override;
    virtual void printMetaArgs(std::ostream &os) const override;
    virtual RooAbsArg::CacheMode canNodeBeCached() const override;
    virtual void setCacheAndTrackHints(RooArgSet &) override;

    void insert(RooWorkspace* ws);
    
    void resetFlags();
    void setParameters(const char* foldername);
    void setParameters(TH1* paramhist);
    void setParameter(const char* name, double value);
    void setFlag(const char* name, double value);
    void setParameters(const ParamSet& params);
    void setParameters(const RooArgList* list);
    double getParameterValue(const char* name) const;
    RooRealVar* getParameter(const char* name) const;
    RooRealVar* getFlag(const char* name) const;
    bool hasParameter(const char* paramname) const;
    bool isParameterUsed(const char* paramname) const;
    bool isParameterConstant(const char* paramname) const;
    void setParameterConstant(const char* paramname, bool constant) const;
    void setParameter(const char* name, double value, double min, double max);
    void setParameter(const char* name, double value, double min, double max, double error);
    void randomizeParameters(double z);
    const RooArgList* getParameterSet() const;
    using RooAbsArg::getParameters;
    ParamSet getParameters(const char* foldername) const;
    ParamSet getParameters() const;

    int nParameters() const;
    int nSamples() const;
    int nPolynomials() const;

    bool isCouplingUsed(const char* couplname) const;
    const RooArgList* getCouplingSet() const;
    ParamSet getCouplings() const;

    TMatrixD getMatrix() const;
    TMatrixD getInvertedMatrix() const;
    double getCondition() const;
    
    RooRealVar* getObservable() const;
    RooRealVar* getBinWidth() const;
 
    void printEvaluation() const;
    void printCouplings() const;
    void printParameters() const;
    void printFlags() const;
    void printParameters(const char* samplename) const;
    void printSamples() const;
    void printPhysics() const;

    RooProduct* getSumElement(const char* name) const;
  
    const std::vector<std::string>& getSamples() const;
  
    double expectedUncertainty() const;
    TH1* createTH1(const std::string& name, RooFitResult* r = NULL);
    TH1* createTH1(const std::string& name, bool correlateErrors, RooFitResult* r = NULL);
  
  protected:

    class CacheElem;
    void init();
    void printAuthors() const;
    void setup(bool ownParams = true);
    bool _ownParameters = false;
  
    void disableInterference(const std::vector<const char*>& nonInterfering);
    void disableInterferences(const std::vector<std::vector<const char*> >& nonInterfering);

    mutable RooObjCacheManager _cacheMgr; //! The cache manager
 
    void addFolders(const RooArgList& folders);

    InternalType* getInternal() const;
  
    bool hasCache() const;
    RooLagrangianMorph::CacheElem* getCache(const RooArgSet* nset) const;
    void readParameters(TDirectory* f);
    void collectInputs(TDirectory* f);
    void updateSampleWeights();
    RooRealVar* setupObservable(const char* obsname,TClass* mode,TObject* inputExample);
    
  public:
  
    bool updateCoefficients();
    bool useCoefficients(const TMatrixD& inverse);
    bool useCoefficients(const char* filename);
    bool writeCoefficients(const char* filename);
  
    int countContributingFormulas() const;
    RooParamHistFunc* getBaseTemplate();
    RooAbsReal* getSampleWeight(const char* name);
    void printSampleWeights() const;
    void printWeights() const;

    void setScale(double val);
    double getScale();
    
  RooRealSumFunc* getFunc() const;
  RooRealSumFunc* cloneFunc() const;
  RooAbsPdf::ExtendMode extendMode() const ;
  Double_t expectedEvents(const RooArgSet* nset) const ;
  Double_t expectedEvents(const RooArgSet& nset) const ;
  Double_t expectedEvents() const;
  RooWrapperPdf* getPdf() const;
  RooWrapperPdf* clonePdf() const;
  Bool_t selfNormalized() const ;
  protected:
    double _scale = 1;
    std::string _fileName;
    std::string _obsName;
    std::string _objFilter;
    std::string _baseFolder;
    bool _allowNegativeYields;
    std::vector<std::string>  _folderNames;
    ParamMap _paramCards;
    FlagMap _flagValues;
    std::map<std::string,int>  _sampleMap;
    RooListProxy _physics;
    RooListProxy _operators;
    RooListProxy _observables ;
    RooListProxy _binWidths ;
    RooListProxy _flags;
    RooLagrangianMorphConfig _config;
    std::vector<std::vector<RooListProxy*> > _diagrams;

    mutable const RooArgSet* _curNormSet ; //!

  public:

    ClassDefOverride(RooLagrangianMorph,1)
  };

}

#endif
