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
// RooLagrangianMorphFunc
//
// The RooLagrangianMorphFunc is a type of RooAbsReal that allows to morph
// different input EFT samples to some arbitrary output EFT
// sample, as long as the desired set of output parameters lie
// within the realm spanned by the input samples. More
// specifically, it expects as an input a TFile (or TDirectory)
// with the following layout:
//
// TDirectory
//  |-sample1
//  | |-param_card    // TH1 EFT parameter values of sample1
//  | | histogram1    // TH1 of some physics distribution
//  | |-subfolder1    // a subfolder (optional)
//  | | |-histogram2  // TH1 of some physics distribution
//  | | |-....
//  |-sample2
//  | |-param_card     // TH1 of EFT parameter values of sample2
//  | | histogram1     // TH1 of some physics distribution
//  | |-subfolder1     // same folder structure as before
//  | | |-histogram2  // TH1 of some physics distribution
//  | | |-....
//  |-sampleN
// The RooLagrangianMorphFunc operates on this structure, extracts data
// and meta-data and produces a morphing result as a RooRealSumFunc
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

#include "RooFit/Floats.h"
#include "RooAbsArg.h"
#include "RooAbsReal.h"
#include "RooArgList.h"
#include "RooRatio.h"
#include "RooRealSumFunc.h"
#include "RooRealSumPdf.h"
#include "RooSetProxy.h"
#include "RooWrapperPdf.h"
#include "TMatrixD.h"

class RooWorkspace;
class RooParamHistFunc;
class RooProduct;
class RooRealVar;
class TPair;
class TFolder;
class RooLagrangianMorphFunc;

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

class RooLagrangianMorphFunc : public RooAbsReal {

public:
   typedef std::map<const std::string, double> ParamSet;
   typedef std::map<const std::string, int> FlagSet;
   typedef std::map<const std::string, ParamSet> ParamMap;
   typedef std::map<const std::string, FlagSet> FlagMap;

   struct Config {

      std::string observableName;
      std::string fileName;
      ParamMap paramCards;
      FlagMap flagValues;
      std::vector<std::string> folderNames;
      RooArgList couplings;
      RooArgList decCouplings;
      RooArgList prodCouplings;
      RooArgList folders;
      std::vector<RooArgList *> vertices;
      std::vector<std::vector<const char *>> nonInterfering;
      bool allowNegativeYields = true;
   };

   RooLagrangianMorphFunc();
   RooLagrangianMorphFunc(const char *name, const char *title, const char *filename, const char *observableName,
                          const RooArgSet &couplings, const RooArgSet &inputs);
   RooLagrangianMorphFunc(const char *name, const char *title, const Config &config);
   RooLagrangianMorphFunc(const RooLagrangianMorphFunc &other, const char *newName);

   virtual ~RooLagrangianMorphFunc();

   std::list<double> *binBoundaries(RooAbsRealLValue & /*obs*/, double /*xlo*/, double /*xhi*/) const override;
   std::list<double> *plotSamplingHint(RooAbsRealLValue & /*obs*/, double /*xlo*/, double /*xhi*/) const override;
   bool isBinnedDistribution(const RooArgSet &obs) const override;
   double evaluate() const override;
   TObject *clone(const char *newname) const override;

   bool checkObservables(const RooArgSet *nset) const override;
   bool forceAnalyticalInt(const RooAbsArg &arg) const override;
   Int_t getAnalyticalIntegralWN(RooArgSet &allVars, RooArgSet &numVars, const RooArgSet *normSet,
                                 const char *rangeName = 0) const override;
   double analyticalIntegralWN(Int_t code, const RooArgSet *normSet, const char *rangeName = 0) const override;
   void printMetaArgs(std::ostream &os) const override;
   RooAbsArg::CacheMode canNodeBeCached() const override;
   void setCacheAndTrackHints(RooArgSet &) override;

   void insert(RooWorkspace *ws);

   void setParameters(const char *foldername);
   void setParameters(TH1 *paramhist);
   void setParameter(const char *name, double value);
   void setFlag(const char *name, double value);
   void setParameters(const ParamSet &params);
   void setParameters(const RooArgList *list);
   double getParameterValue(const char *name) const;
   RooRealVar *getParameter(const char *name) const;
   RooRealVar *getFlag(const char *name) const;
   bool hasParameter(const char *paramname) const;
   bool isParameterUsed(const char *paramname) const;
   bool isParameterConstant(const char *paramname) const;
   void setParameterConstant(const char *paramname, bool constant) const;
   void setParameter(const char *name, double value, double min, double max);
   void setParameter(const char *name, double value, double min, double max, double error);
   void randomizeParameters(double z);
   const RooArgSet *getParameterSet() const;
   ParamSet getMorphParameters(const char *foldername) const;
   ParamSet getMorphParameters() const;

   RooLagrangianMorphFunc *getLinear() const;

   int nParameters() const;
   int nPolynomials() const;

   bool isCouplingUsed(const char *couplname);
   const RooArgList *getCouplingSet() const;
   ParamSet getCouplings() const;

   TMatrixD getMatrix() const;
   TMatrixD getInvertedMatrix() const;
   double getCondition() const;

   RooRealVar *getObservable() const;
   RooRealVar *getBinWidth() const;

   void printEvaluation() const;
   void printCouplings() const;
   void printFlags() const;
   void printPhysics() const;

   RooProduct *getSumElement(const char *name) const;

   std::vector<std::string> getSamples() const;

   double expectedUncertainty() const;
   TH1 *createTH1(const std::string &name);
   TH1 *createTH1(const std::string &name, bool correlateErrors);

private:
   class CacheElem;
   void init();
   void setup(bool ownParams = true);
   void disableInterference(const std::vector<const char *> &nonInterfering);
   void disableInterferences(const std::vector<std::vector<const char *>> &nonInterfering);

   void addFolders(const RooArgList &folders);

   bool hasCache() const;
   RooLagrangianMorphFunc::CacheElem *getCache() const;
   void updateSampleWeights();

   RooRealVar *setupObservable(const char *obsname, TClass *mode, TObject *inputExample);

public:
   /// length of floating point digits precision supported by implementation.
   static constexpr double implementedPrecision = RooFit::SuperFloatPrecision::digits10;

   void writeMatrixToFile(const TMatrixD &matrix, const char *fname);
   void writeMatrixToStream(const TMatrixD &matrix, std::ostream &stream);
   TMatrixD readMatrixFromFile(const char *fname);
   TMatrixD readMatrixFromStream(std::istream &stream);

   int countSamples(std::vector<RooArgList *> &vertices);
   int countSamples(int nprod, int ndec, int nboth);

   std::map<std::string, std::string>
   createWeightStrings(const ParamMap &inputs, const std::vector<std::vector<std::string>> &vertices);
   std::map<std::string, std::string>
   createWeightStrings(const ParamMap &inputs, const std::vector<RooArgList *> &vertices, RooArgList &couplings);
   std::map<std::string, std::string>
   createWeightStrings(const ParamMap &inputs, const std::vector<RooArgList *> &vertices, RooArgList &couplings,
                       const FlagMap &flagValues, const RooArgList &flags,
                       const std::vector<RooArgList *> &nonInterfering);
   RooArgSet createWeights(const ParamMap &inputs, const std::vector<RooArgList *> &vertices, RooArgList &couplings,
                           const FlagMap &inputFlags, const RooArgList &flags,
                           const std::vector<RooArgList *> &nonInterfering);
   RooArgSet createWeights(const ParamMap &inputs, const std::vector<RooArgList *> &vertices, RooArgList &couplings);

   bool updateCoefficients();
   bool useCoefficients(const TMatrixD &inverse);
   bool useCoefficients(const char *filename);
   bool writeCoefficients(const char *filename);

   int countContributingFormulas() const;
   RooAbsReal *getSampleWeight(const char *name);
   void printParameters(const char *samplename) const;
   void printParameters() const;
   void printSamples() const;
   void printSampleWeights() const;
   void printWeights() const;

   void setScale(double val);
   double getScale();

   int nSamples() const { return this->_config.folderNames.size(); }

   RooRealSumFunc *getFunc() const;
   std::unique_ptr<RooWrapperPdf> createPdf() const;

   RooAbsPdf::ExtendMode extendMode() const;
   Double_t expectedEvents(const RooArgSet *nset) const;
   Double_t expectedEvents(const RooArgSet &nset) const;
   Double_t expectedEvents() const;
   Bool_t selfNormalized() const { return true; }

   void readParameters(TDirectory *f);
   void collectInputs(TDirectory *f);

   static std::unique_ptr<RooRatio> makeRatio(const char *name, const char *title, RooArgList &nr, RooArgList &dr);

private:

   mutable RooObjCacheManager _cacheMgr; //! The cache manager
   double _scale = 1.0;
   std::map<std::string, int> _sampleMap;
   RooListProxy _physics;
   RooSetProxy _operators;
   RooListProxy _observables;
   RooListProxy _binWidths;
   RooListProxy _flags;
   Config _config;
   std::vector<std::vector<RooListProxy *>> _diagrams;
   std::vector<RooListProxy *> _nonInterfering;

   ClassDefOverride(RooLagrangianMorphFunc, 1)
};

#endif
