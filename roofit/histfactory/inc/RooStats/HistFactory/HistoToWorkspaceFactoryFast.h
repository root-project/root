// @(#)root/roostats:$Id:  cranmer $
// Author: Kyle Cranmer, Akira Shibata
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_HISTOTOWORKSPACEFACTORYFAST
#define ROOSTATS_HISTOTOWORKSPACEFACTORYFAST

#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <sstream>
#include <memory>

#include <RooPlot.h>
#include <RooArgSet.h>
#include <RooFitResult.h>
#include <RooAbsReal.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>
#include <TObject.h>
#include <TH1.h>
#include <TDirectory.h>

#include "RooStats/HistFactory/Systematics.h"
class ParamHistFunc;
class RooProduct;
class RooHistFunc;

namespace RooStats{
  namespace HistFactory{

    // Forward Declarations FTW
    class Measurement;
    class Channel;
    class Sample;

    class HistoToWorkspaceFactoryFast: public TObject {

    public:


      HistoToWorkspaceFactoryFast();
      HistoToWorkspaceFactoryFast(  RooStats::HistFactory::Measurement& Meas );
      virtual ~HistoToWorkspaceFactoryFast();

      static void ConfigureWorkspaceForMeasurement( const std::string& ModelName,
                      RooWorkspace* ws_single,
                      Measurement& measurement );

      RooWorkspace* MakeSingleChannelModel( Measurement& measurement, Channel& channel );
      RooWorkspace*  MakeCombinedModel(std::vector<std::string>, std::vector<std::unique_ptr<RooWorkspace>>&);

      static RooWorkspace* MakeCombinedModel( Measurement& measurement );
      static void PrintCovarianceMatrix(RooFitResult* result, RooArgSet* params,
               std::string filename);

      void SetFunctionsToPreprocess(std::vector<std::string> lines) { fPreprocessFunctions=lines; }

    protected:

       void AddConstraintTerms(RooWorkspace* proto, Measurement& measurement, std::string prefix, std::string interpName,
               std::vector<OverallSys>& systList,
               std::vector<std::string>& likelihoodTermNames,
               std::vector<std::string>& totSystTermNames);

      std::unique_ptr<RooProduct> CreateNormFactor(RooWorkspace* proto, std::string& channel,
            std::string& sigmaEpsilon, Sample& sample, bool doRatio);

      void AddMultiVarGaussConstraint(RooWorkspace* proto, std::string prefix,
                  int lowBin, int highBin,
                  std::vector<std::string>& likelihoodTermNames);

      void AddPoissonTerms(RooWorkspace* proto, std::string prefix, std::string obsPrefix,
            std::string expPrefix, int lowBin, int highBin,
            std::vector<std::string>& likelihoodTermNames);

      static void EditSyst(RooWorkspace* proto, const char* pdfNameChar,
            std::map<std::string,double> gammaSyst,
            std::map<std::string,double> uniformSyst,
            std::map<std::string,double> logNormSyst,
            std::map<std::string,double> noSyst);

      RooWorkspace* MakeSingleChannelWorkspace(Measurement& measurement, Channel& channel);

      void MakeTotalExpected(RooWorkspace* proto, const std::string& totName,
              const std::vector<RooProduct*>& sampleScaleFactors,
              std::vector<std::vector<RooAbsArg*>>&  sampleHistFuncs) const;

      RooDataSet* MergeDataSets(RooWorkspace* combined,
            std::vector<std::unique_ptr<RooWorkspace>>& wspace_vec,
            std::vector<std::string> const& channel_names,
            std::string const& dataSetName,
            RooArgList obsList,
            RooCategory* channelCat);

      RooHistFunc* MakeExpectedHistFunc(const TH1* hist, RooWorkspace* proto, std::string prefix,
          const RooArgList& observables) const;

      void SetObsToExpected(RooWorkspace* proto, std::string obsPrefix, std::string expPrefix,
             int lowBin, int highBin);

      std::unique_ptr<TH1> MakeScaledUncertaintyHist(const std::string& Name,
                 std::vector< std::pair<const TH1*, std::unique_ptr<TH1>> > const& HistVec ) const;

      TH1* MakeAbsolUncertaintyHist( const std::string& Name, const TH1* Hist );

      RooArgList createStatConstraintTerms( RooWorkspace* proto,
                   std::vector<std::string>& constraintTerms,
                   ParamHistFunc& paramHist, const TH1* uncertHist,
                   Constraint::Type type, Double_t minSigma );

      void ConfigureHistFactoryDataset(RooDataSet& obsData, TH1 const& nominal, RooWorkspace& proto,
                   std::vector<std::string> const& obsNameVec);

      std::vector<std::string> fSystToFix;
      std::map<std::string, double> fParamValues;
      double fNomLumi;
      double fLumiError;
      int fLowBin;
      int fHighBin;

    private:

      void GuessObsNameVec(const TH1* hist);

      std::vector<std::string> fObsNameVec;
      std::string fObsName;
      std::vector<std::string> fPreprocessFunctions;

      RooArgList createObservables(const TH1 *hist, RooWorkspace *proto) const;

      ClassDef(RooStats::HistFactory::HistoToWorkspaceFactoryFast,3)
    };

  }
}

#endif
