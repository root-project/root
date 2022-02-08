// @(#)root/roostats:$Id:  cranmer $
// Author: Kyle Cranmer, Akira Shibata
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_HISTOTOWORKSPACEFACTORY
#define ROOSTATS_HISTOTOWORKSPACEFACTORY

#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <sstream>

#include <RooPlot.h>
#include <RooArgSet.h>
#include <RooFitResult.h>
#include <RooAbsReal.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>
#include <TObject.h>
#include <TH1.h>
#include <TDirectory.h>


namespace RooStats{
namespace HistFactory{

  struct EstimateSummary;

  class HistoToWorkspaceFactory: public TObject {

    public:

     HistoToWorkspaceFactory(  std::string, std::string , std::vector<std::string> , double =200, double =20, int =0, int =6, TFile * =0);
      HistoToWorkspaceFactory();
      ~HistoToWorkspaceFactory() override;

      void AddEfficiencyTerms(RooWorkspace* proto, std::string prefix, std::string interpName,
            std::map<std::string,std::pair<double,double> > systMap,
            std::vector<std::string>& likelihoodTermNames, std::vector<std::string>& totSystTermNames);

      std::string AddNormFactor(RooWorkspace *, std::string & , std::string & , EstimateSummary & , bool );

      void AddMultiVarGaussConstraint(RooWorkspace* proto, std::string prefix,int lowBin, int highBin, std::vector<std::string>& likelihoodTermNames);

      void AddPoissonTerms(RooWorkspace* proto, std::string prefix, std::string obsPrefix, std::string expPrefix, int lowBin, int highBin,
               std::vector<std::string>& likelihoodTermNames);

      //void Combine_old();

      RooWorkspace *  MakeCombinedModel(std::vector<std::string>, std::vector<RooWorkspace*>);

      //void Combine_ratio(std::vector<std::string> , std::vector<RooWorkspace*>);

      void Customize(RooWorkspace* proto, const char* pdfNameChar, std::map<std::string,std::string> renameMap);

      void EditSyst(RooWorkspace* proto, const char* pdfNameChar, std::map<std::string,double> gammaSyst, std::map<std::string,double> uniformSyst, std::map<std::string,double> logNormSyst);

      void FormatFrameForLikelihood(RooPlot* frame, std::string XTitle=std::string("#sigma / #sigma_{SM}"), std::string YTitle=std::string("-log likelihood"));


      void LinInterpWithConstraint(RooWorkspace* proto, TH1* nominal, std::vector<TH1*> lowHist, std::vector<TH1*> highHist,
                 std::vector<std::string> sourceName, std::string prefix, std::string productPrefix, std::string systTerm,
                 int lowBin, int highBin, std::vector<std::string>& likelihoodTermNames);

      TDirectory* Makedirs( TDirectory* file, std::vector<std::string> names );

      RooWorkspace* MakeSingleChannelModel(std::vector<RooStats::HistFactory::EstimateSummary> summary, std::vector<std::string> systToFix, bool doRatio=false);

      void  MakeTotalExpected(RooWorkspace* proto, std::string totName, std::string /**/, std::string /**/,
            int lowBin, int highBin, std::vector<std::string>& syst_x_expectedPrefixNames,
            std::vector<std::string>& normByNames);

      TDirectory* Mkdir( TDirectory * file, std::string name );

      void PrintCovarianceMatrix(RooFitResult* result, RooArgSet* params, std::string filename);
      void ProcessExpectedHisto(TH1* hist,RooWorkspace* proto, std::string prefix, std::string productPrefix, std::string systTerm, double low, double high, int lowBin, int highBin);
      void SetObsToExpected(RooWorkspace* proto, std::string obsPrefix, std::string expPrefix, int lowBin, int highBin);
      void FitModel(RooWorkspace *, std::string, std::string, std::string, bool=false  );
      std::string FilePrefixStr(std::string);


      std::string fFileNamePrefix;
      std::string fRowTitle;
      std::vector<std::string> fSystToFix;
      double fNomLumi, fLumiError;
      int  fLowBin, fHighBin;
      std::stringstream fResultsPrefixStr;
      TFile * fOut_f;
      FILE * pFile;

      ClassDefOverride(RooStats::HistFactory::HistoToWorkspaceFactory,1)
  };

}
}

#endif
