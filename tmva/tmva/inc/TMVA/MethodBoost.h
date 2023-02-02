// @(#)root/tmva $Id$
// Author: Andreas Hoecker, Joerg Stelzer, Helge Voss, Kai Voss,Or Cohen, Jan Therhaag, Eckhard von Toerne

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
 *      Andreas Hoecker    <Andreas.Hocker@cern.ch>   - CERN, Switzerland         *
 *      Peter Speckmayer   <Peter.Speckmazer@cern.ch> - CERN, Switzerland         *
 *      Joerg Stelzer      <Joerg.Stelzer@cern.ch>    - CERN, Switzerland         *
 *      Helge Voss         <Helge.Voss@cern.ch>       - MPI-K Heidelberg, Germany *
 *      Jan Therhaag       <Jan.Therhaag@cern.ch>     - U of Bonn, Germany        *
 *      Eckhard v. Toerne  <evt@uni-bonn.de>          - U of Bonn, Germany        *
 *                                                                                *
 * Copyright (c) 2005-2011:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
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

#include "TMVA/MethodBase.h"

#include "TMVA/MethodCompositeBase.h"

namespace TMVA {

   class Factory;  // DSMTEST
   class Reader;   // DSMTEST
   class DataSetManager;  // DSMTEST
   namespace Experimental {
   class Classification;
   }
   class MethodBoost : public MethodCompositeBase {
      friend class Factory; // DSMTEST
      friend class Reader;  // DSMTEST
      friend class Experimental::Classification;

   public :

      // constructors
      MethodBoost( const TString& jobName,
                   const TString& methodTitle,
                   DataSetInfo& theData,
                   const TString& theOption = "" );

      MethodBoost( DataSetInfo& dsi,
                   const TString& theWeightFile );

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

      void CleanBoostOptions();

      Double_t GetMvaValue( Double_t* err = nullptr, Double_t* errUpper = nullptr );

   private :
      // clean up
      void ClearAll();

      // print fit results
      void PrintResults( const TString&, std::vector<Double_t>&, const Double_t ) const;

      // initializing mostly monitoring tools of the boost process
      void Init();
      void InitHistos();
      void CheckSetup();

      void MonitorBoost( Types::EBoostStage stage, UInt_t methodIdx=0);

      // the option handling methods
      void DeclareOptions();
      void DeclareCompatibilityOptions();
      void ProcessOptions();


      MethodBase*  CurrentMethod(){return fCurrentMethod;}
      UInt_t       CurrentMethodIdx(){return fCurrentMethodIdx;}
      // training a single classifier
      void SingleTrain();

      // calculating a boosting weight from the classifier, storing it in the next one
      Double_t SingleBoost(MethodBase* method);
      Double_t AdaBoost(MethodBase* method, Bool_t useYesNoLeaf );
      Double_t Bagging();


      // calculate weight of single method
      Double_t CalcMethodWeight();

      // return ROC integral on training/testing sample
      Double_t GetBoostROCIntegral(Bool_t, Types::ETreeType, Bool_t CalcOverlapIntergral=kFALSE);

      // writing the monitoring histograms and tree to a file
      void WriteMonitoringHistosToFile( void ) const;

      // write evaluation histograms into target file
      virtual void WriteEvaluationHistosToFile(Types::ETreeType treetype);

      // performs the MethodBase testing + testing of each boosted classifier
      virtual void TestClassification();

      // finding the MVA to cut between sig and bgd according to fMVACutPerc,fMVACutType
      void FindMVACut(MethodBase* method);

      // setting all the boost weights to 1
      void ResetBoostWeights();

      // creating the vectors of histogram for monitoring MVA response of each classifier
      void CreateMVAHistorgrams();

      // calculate MVA values of current trained method on training
      // sample
      void CalcMVAValues();

      UInt_t                 fBoostNum;             ///< Number of times the classifier is boosted
      TString                fBoostType;            ///< string specifying the boost type

      TString                fTransformString;      ///< min and max values for the classifier response
      Bool_t                 fDetailedMonitoring;   ///< produce detailed monitoring histograms (boost-wise)

      Double_t               fAdaBoostBeta;         ///< ADA boost parameter, default is 1
      UInt_t                 fRandomSeed;           ///< seed for random number generator used for bagging
      Double_t               fBaggedSampleFraction; ///< rel.Size of bagged sample

      TString                fBoostedMethodName;    ///< details of the boosted classifier
      TString                fBoostedMethodTitle;   ///< title
      TString                fBoostedMethodOptions; ///< options

      Bool_t                 fMonitorBoostedMethod; ///< monitor the MVA response of every classifier

      // MVA output from each classifier over the training hist, using orignal events weights
      std::vector< TH1* >   fTrainSigMVAHist;
      std::vector< TH1* >   fTrainBgdMVAHist;
      // MVA output from each classifier over the training hist, using boosted events weights
      std::vector< TH1* >   fBTrainSigMVAHist;
      std::vector< TH1* >   fBTrainBgdMVAHist;
      // MVA output from each classifier over the testing hist
      std::vector< TH1* >   fTestSigMVAHist;
      std::vector
         < TH1* >   fTestBgdMVAHist;

      //monitoring tree/ntuple and it's variables
      TTree*                fMonitorTree;     ///< tree  to monitor values during the boosting
      Double_t              fBoostWeight;     ///< the weight used to boost the next classifier
      Double_t              fMethodError;     ///< estimation of the level error of the classifier
      // analysing the train dataset
      Double_t           fROC_training;       ///< roc integral of last trained method (on training sample)

      // overlap integral of mva distributions for signal and
      // background (training sample)
      Double_t           fOverlap_integral;

      std::vector<Float_t> *fMVAvalues;       ///< mva values for the last trained method

      DataSetManager*    fDataSetManager;     ///< DSMTEST
      TString fHistoricOption;                ///< historic variable, only needed for "CompatibilityOptions"
      Bool_t fHistoricBoolOption;             ///< historic variable, only needed for "CompatibilityOptions"

   protected:

      // get help message text
      void GetHelpMessage() const;

      ClassDef(MethodBoost,0);
   };
}

#endif
