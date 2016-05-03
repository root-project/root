// @(#)root/tmva $Id$
// Author: Peter Speckmayer

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodDNN                                                              *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      NeuralNetwork                                                             *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Peter Speckmayer      <peter.speckmayer@gmx.at> - CERN, Switzerland       *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

//#pragma once

#ifndef ROOT_TMVA_MethodDNN
#define ROOT_TMVA_MethodDNN

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodDNN                                                             //
//                                                                      //
// Neural Network implementation                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <vector>
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TTree
#include "TTree.h"
#endif
#ifndef ROOT_TRandom3
#include "TRandom3.h"
#endif
#ifndef ROOT_TH1F
#include "TH1F.h"
#endif
#ifndef ROOT_TMVA_MethodBase
#include "TMVA/MethodBase.h"
#endif
#ifndef TMVA_NEURAL_NET
#include "TMVA/NeuralNet.h"
#endif



namespace TMVA {

   class MethodDNN : public MethodBase
   {

   public:

      // standard constructors
      MethodDNN ( const TString& jobName,
                  const TString&  methodTitle,
                  DataSetInfo& theData,
                  const TString& theOption,
                  TDirectory* theTargetDir = 0 );

      MethodDNN ( DataSetInfo& theData,
                  const TString& theWeightFile,
                  TDirectory* theTargetDir = 0 );

      virtual ~MethodDNN();

      virtual Bool_t HasAnalysisType( Types::EAnalysisType type, UInt_t numberClasses, UInt_t numberTargets );
      std::vector<std::pair<int,TMVA::DNN::EnumFunction>> ParseLayoutString(TString layerSpec);
      std::vector<std::map<TString,TString>> ParseKeyValueString(TString parseString, TString blockDelim, TString tokenDelim);

      void Train();

      virtual Double_t GetMvaValue( Double_t* err=0, Double_t* errUpper=0 );
      virtual const std::vector<Float_t>& GetRegressionValues();
      virtual const std::vector<Float_t>& GetMulticlassValues();

      using MethodBase::ReadWeightsFromStream;

      // write weights to stream
      void AddWeightsXMLTo     ( void* parent ) const;

      // read weights from stream
      void ReadWeightsFromStream( std::istream & i );
      void ReadWeightsFromXML   ( void* wghtnode );

      // ranking of input variables
      const Ranking* CreateRanking();

      // nice output
      void PrintCoefficients( void );

      // write classifier-specific monitoring information to target file
      virtual void     WriteMonitoringHistosToFile() const;

   protected:


      // make ROOT-independent C++ class for classifier response (classifier-specific implementation)
      void MakeClassSpecific( std::ostream&, const TString& ) const;

      // get help message text
      void GetHelpMessage() const;


   private:

      void checkGradients ();

      // the option handling methods
      void DeclareOptions();
      void ProcessOptions();

      // general helper functions
      void     Init();


   private:
      TMVA::DNN::Net fNet;
      std::vector<double> fWeights;

      TString  fLayoutString;
      std::vector<std::pair<int,TMVA::DNN::EnumFunction>> fLayout;
      TString  fErrorStrategy;
      TString  fTrainingStrategy;
      TMVA::DNN::ModeErrorFunction fModeErrorFunction;
      std::shared_ptr<TMVA::Monitoring> fMonitoring;
      double   fSumOfSigWeights_test;
      double   fSumOfBkgWeights_test;
      bool     fResume;
      TString  fWeightInitializationStrategyString;
      TMVA::DNN::WeightInitializationStrategy fWeightInitializationStrategy;

      std::vector<std::shared_ptr<TMVA::DNN::Settings>> fSettings;

      TString  fFileName;
      double fScaleToNumEvents;

      ClassDef(MethodDNN,0); // neural network 
   };

} // namespace TMVA


// make_unqiue is only available with C++14
template <typename T, typename... Args>
   std::unique_ptr<T> make_unique (Args&&... args)
{
   return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

// make_shared is only available with C++14
template <typename T, typename... Args>
   std::shared_ptr<T> make_shared (Args&&... args)
{
   return std::shared_ptr<T>(new T(std::forward<Args>(args)...));
}



#endif
