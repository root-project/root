// @(#)root/tmva $Id$
// Author: Vladimir Ilievski 20/06/2017


/*************************************************************************
 * Copyright (C) 2017, Vladimir Ilievski                                 *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMVA_MethodDL
#define ROOT_TMVA_MethodDL


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodCNN                                                            //
//                                                                      //
// Base class for all Deep Learning Modules                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TString.h"
#include "TMVA/MethodBase.h"
#include "TMVA/Types.h"
#include "TMVA/DNN/Functions.h"

#include <vector>

namespace TMVA
{
   
///< All of the options that can be specified in the training string
struct TTrainingSettings
{
   size_t                batchSize;
   size_t                testInterval;
   size_t                convergenceSteps;
   DNN::ERegularization       regularization;
   Double_t              learningRate;
   Double_t              momentum;
   Double_t              weightDecay;
   std::vector<Double_t> dropoutProbabilities;
   bool                  multithreading;
};

class MethodDL : public MethodBase
{

private:
   using KeyValueVector_t = std::vector<std::map<TString, TString>>;
    
   // the option handling methods
   virtual void DeclareOptions();
   virtual void ProcessOptions();
    
   virtual void Init() = 0;
    
    
   DNN::EInitialization   fWeightInitialization;    ///< The initialization method
   DNN::EOutputFunction   fOutputFunction;          ///< The output function for making the predictions
   DNN::ELossFunction     fLossFunction;            ///< The loss function
    
   TString   fErrorStrategy;                   ///< The string defining the error strategy for training
   TString   fTrainingStrategyString;          ///< The string defining the training strategy
   TString   fWeightInitializationString;      ///< The string defining the weight initialization method
   TString   fArchitectureString;              ///< The string defining the architecure: CPU or GPU
   bool      fResume;
    
   std::vector<TTrainingSettings> fTrainingSettings;     ///< The vector defining each training strategy
   KeyValueVector_t fSettings;                           ///< Map for the training strategy
    
   ClassDef(MethodDL,0);

protected:
   // provide a help message
   virtual void GetHelpMessage() const = 0;

public:

   MethodDL(const TString& jobName,
            Types::EMVA mvaType,
            const TString&  methodTitle,
            DataSetInfo& theData,
            const TString& theOption);
    
   MethodDL(Types::EMVA mvaType,
            DataSetInfo& theData,
            const TString& theWeightFile);

   virtual ~MethodDL();
    
   // Check the type of analysis the deep learning network can do
   virtual Bool_t HasAnalysisType(Types::EAnalysisType type,
                                  UInt_t numberClasses,
                                  UInt_t numberTargets ) = 0;
   
   // Method for parsing the layout specifications of the deep learning network
   virtual void ParseLayoutString(TString layerSpec) = 0;
   virtual KeyValueVector_t ParseKeyValueString(TString parseString,
                                                TString blockDelim,
                                                TString tokenDelim);
   
   // Methods for training the deep learning network
   virtual void Train() = 0;
   virtual void TrainGpu() = 0;
   virtual void TrainCpu() = 0;
    
   virtual Double_t GetMvaValue( Double_t* err=0, Double_t* errUpper=0 ) = 0;
   
   // Methods for writing and reading weights
   using MethodBase::ReadWeightsFromStream;
   virtual void AddWeightsXMLTo      ( void* parent ) const = 0;
   virtual void ReadWeightsFromXML   ( void* wghtnode ) = 0;
   virtual void ReadWeightsFromStream( std::istream& ) = 0;

   // create ranking
   virtual const Ranking* CreateRanking() = 0;

   void CallDeclareOptions();
   void CallProcessOptions();
    
   /* Getters */
   DNN::EInitialization GetWeightInitialization() {return fWeightInitialization;}
   DNN::EOutputFunction GetOutputFunction()       {return fOutputFunction;}
   DNN::ELossFunction GetLossFunction()           {return fLossFunction;}

    
   
   TString GetErrorStrategyString()        {return fErrorStrategy;}
   TString GetTrainingStrategyString()     {return fTrainingStrategyString;}
   TString GetWeightInitializationString() {return fWeightInitializationString;}
   TString GetArchitectureString()         {return fArchitectureString;}
    
    
   const std::vector<TTrainingSettings>& GetTrainingSettings() const  {return fTrainingSettings;}
   std::vector<TTrainingSettings>& GetTrainingSettings()              {return fTrainingSettings;}
   const KeyValueVector_t& GetKeyValueSettings() const                {return fSettings;}
   KeyValueVector_t& GetKeyValueSettings()                            {return fSettings;}
    
    
   /** Setters */
   void SetWeightInitialization(DNN::EInitialization weightInitialization){
      fWeightInitialization = weightInitialization;
   }
   void SetOutputFunction(DNN::EOutputFunction outputFunction){
      fOutputFunction = outputFunction;
   }
   void SetErrorStrategyString(TString errorStrategy) {
      fErrorStrategy = errorStrategy;
   }
   void SetTrainingStrategyString(TString trainingStrategyString){
      fTrainingStrategyString = trainingStrategyString;
   }
   void SetWeightInitializationString(TString weightInitializationString){
      fWeightInitializationString = weightInitializationString;
   }
   void SetArchitectureString(TString architectureString){
      fArchitectureString = architectureString;
   }
};
    
} // namespace TMVA


#endif

