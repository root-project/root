// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss,Or Cohen

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodCompositeBase                                                   *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Virtual base class for all MVA method                                     *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Andreas Hoecker    <Andreas.Hocker@cern.ch> - CERN, Switzerland           *
 *      Joerg Stelzer      <Joerg.Stelzer@cern.ch>  - CERN, Switzerland           *
 *      Helge Voss         <Helge.Voss@cern.ch>     - MPI-K Heidelberg, Germany   *
 *      Kai Voss           <Kai.Voss@cern.ch>       - U. of Victoria, Canada      *
 *      Or Cohen           <orcohenor@gmail.com>    - Weizmann Inst., Israel      *
 *      Eckhard v. Toerne  <evt@uni-bonn.de>        - U of Bonn, Germany          *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      LAPP, Annecy, France                                                      *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodBoost
#define ROOT_TMVA_MethodBoost

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodBoost                                                          //
//                                                                      //
// Class for boosting a TMVA method                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <iosfwd>
#include <vector>

#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif

#ifndef ROOT_TMVA_MethodCompositeBase
#include "TMVA/MethodCompositeBase.h"
#endif

namespace TMVA {

   class Factory;  // DSMTEST
   class Reader;   // DSMTEST
   class DataSetManager;  // DSMTEST

   class MethodBoost : public MethodCompositeBase {

   public :

      // constructors
      MethodBoost( const TString& jobName,
                   const TString& methodTitle,
                   DataSetInfo& theData,
                   const TString& theOption = "",
                   TDirectory* theTargetDir = NULL );

      MethodBoost( DataSetInfo& dsi,
                   const TString& theWeightFile,
                   TDirectory* theTargetDir = NULL );

      virtual ~MethodBoost( void );

      virtual Bool_t HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t /*numberTargets*/ );

      // training and boosting all the classifiers
      void Train( void );

      // ranking of input variables
      const Ranking* CreateRanking();

      // saves the name and options string of the boosted classifier
      Bool_t BookMethod( Types::EMVA theMethod, TString methodTitle, TString theOption );
      void SetBoostedMethodName ( TString methodName )     { fBoostedMethodName  = methodName; }

      Int_t          GetBoostNum() { return fBoostNum; }

      // gives the monitoring historgram from the vector according to index of the
      // histrogram added in the MonitorBoost function
      TH1*           GetMonitoringHist( Int_t histInd ) { return (*fMonitorHist)[fDefaultHistNum+histInd]; }

      void           AddMonitoringHist( TH1* hist )     { return fMonitorHist->push_back(hist); }

      Types::EBoostStage    GetBoostStage() { return fBoostStage; }

      void CleanBoostOptions();

      Double_t GetMvaValue( Double_t* err=0, Double_t* errUpper = 0 );

   private :
      // clean up
      void ClearAll();

      // print fit results
      void PrintResults( const TString&, std::vector<Double_t>&, const Double_t ) const;

      // initializing mostly monitoring tools of the boost process
      void Init();
      void InitHistos();
      void CheckSetup();

      // the option handling methods
      void DeclareOptions();
      void ProcessOptions();

      MethodBoost* SetStage( Types::EBoostStage stage ) { fBoostStage = stage; return this; }

      //training a single classifier
      void SingleTrain();

      //calculating a boosting weight from the classifier, storing it in the next one
      void SingleBoost();

      // calculate weight of single method
      void CalcMethodWeight();

      // return ROC integral on training/testing sample
      Double_t GetBoostROCIntegral(Bool_t, Types::ETreeType, Bool_t CalcOverlapIntergral=kFALSE);

      //writing the monitoring histograms and tree to a file
      void WriteMonitoringHistosToFile( void ) const;

      // write evaluation histograms into target file
      virtual void WriteEvaluationHistosToFile(Types::ETreeType treetype);

      // performs the MethodBase testing + testing of each boosted classifier
      virtual void TestClassification();

      //finding the MVA to cut between sig and bgd according to fMVACutPerc,fMVACutType
      void FindMVACut();

      //setting all the boost weights to 1
      void ResetBoostWeights();

      //creating the vectors of histogram for monitoring MVA response of each classifier
      void CreateMVAHistorgrams();

      // calculate MVA values of current trained method on training
      // sample
      void CalcMVAValues();

      //Number of times the classifier is boosted (set by the user)
      Int_t             fBoostNum;
      // string specifying the boost type (AdaBoost / Bagging )
      TString           fBoostType;

      // string specifying the boost type ( ByError,Average,LastMethod )
      TString           fMethodWeightType;

      //estimation of the level error of the classifier analysing the train dataset
      Double_t          fMethodError;
      //estimation of the level error of the classifier analysing the train dataset (with unboosted weights)
      Double_t          fOrigMethodError;

      //the weight used to boost the next classifier
      Double_t          fBoostWeight;

      // min and max values for the classifier response
      TString fTransformString;

      //ADA boost parameter, default is 1
      Double_t          fADABoostBeta;

      // seed for random number generator used for bagging
      UInt_t            fRandomSeed;

      // details of the boosted classifier
      TString           fBoostedMethodName;
      TString           fBoostedMethodTitle;
      TString           fBoostedMethodOptions;

      // histograms to monitor values during the boosting
      std::vector<TH1*>* fMonitorHist;

      //whether to monitor the MVA response of every classifier using the
      Bool_t                fMonitorBoostedMethod;

      //MVA output from each classifier over the training hist, using orignal events weights
      std::vector< TH1* >   fTrainSigMVAHist;
      std::vector< TH1* >   fTrainBgdMVAHist;
      //MVA output from each classifier over the training hist, using boosted events weights
      std::vector< TH1* >   fBTrainSigMVAHist;
      std::vector< TH1* >   fBTrainBgdMVAHist;
      //MVA output from each classifier over the testing hist
      std::vector< TH1* >   fTestSigMVAHist;
      std::vector< TH1* >   fTestBgdMVAHist;

      // tree  to monitor values during the boosting
      TTree*            fMonitorTree;

      // the stage of the boosting
      Types::EBoostStage fBoostStage;

      //the number of histogram filled for every type of boosted classifier
      Int_t             fDefaultHistNum;

      //whether to recalculate the MVA cut at every boosting step
      Bool_t            fRecalculateMVACut;

      // roc integral of last trained method (on training sample)
      Double_t          fROC_training;

      // overlap integral of mva distributions for signal and
      // background (training sample)
      Double_t          fOverlap_integral;

      // mva values for the last trained method (on training sample)
      std::vector<Float_t> *fMVAvalues;

      DataSetManager* fDataSetManager; // DSMTEST
      friend class Factory; // DSMTEST
      friend class Reader;  // DSMTEST





   protected:

      // get help message text
      void GetHelpMessage() const;

      ClassDef(MethodBoost,0)
   };
}

#endif
