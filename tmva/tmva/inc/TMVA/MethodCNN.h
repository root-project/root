// @(#)root/tmva $Id$
// Author: Vladimir Ilievski 30/05/2017


/*************************************************************************
 * Copyright (C) 2017, Vladimir Ilievski                                 *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TMVA_MethodCNN
#define ROOT_TMVA_MethodCNN

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodCNN                                                            //
//                                                                      //
// Convolutional Neural Network implementation                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TString.h"


#include "TMVA/MethodBase.h"
#include "TMVA/DNN/CNN/CNNLayer.h"
#include "TMVA/DNN/CNN/ConvNet.h"

#include "TMVA/DNN/Architectures/Reference.h"

#ifdef DNNCPU
#include "TMVA/DNN/Architectures/Cpu.h"
#endif

#ifdef DNNCUDA
#include "TMVA/DNN/Architectures/Cuda.h"
#endif

#include <vector>
#include <tuple>


using namespace TMVA::DNN::CNN;
using namespace TMVA::DNN;

namespace TMVA
{

class MethodCNN: public MethodBase
{
public:
    
   using Architecture_t  = TReference<Double_t>;
   using ConvNet_t       = TConvNet<Architecture_t>;
   using Matrix_t        = typename Architecture_t::Matrix_t;

private:
    
   using LayoutVector_t   = std::vector<std::tuple<int, int, int, EActivationFunction>>;
    
   struct TTrainingSettings
    {
        size_t                batchSize;
        size_t                testInterval;
        size_t                convergenceSteps;
        ERegularization       regularization;
        Double_t              learningRate;
        Double_t              momentum;
        Double_t              weightDecay;
        std::vector<Double_t> dropoutProbabilities;
        bool                  multithreading;
    };
    
   ConvNet_t         fConvNet;                 ///< The convolutional neural net to train
   EInitialization   fWeightInitialization;    ///< The initialization method
   EOutputFunction   fOutputFunction;          ///< The output function for making the predictions
    
   TString   fLayoutString;                    ///< The string defining the layout of the CNN
   TString   fErrorStrategy;                   ///< The string defining the error strategy for training
   TString   fTrainingStrategyString;          ///< The string defining the training strategy
   TString   fWeightInitializationString;      ///< The string defining the weight initialization method
   TString   fArchitectureString;              ///< The string defining the architecure: CPU or GPU
   bool      fResume;
   
   LayoutVector_t                 fLayout;               ///< Dimensions and activation functions of each layer
   std::vector<TTrainingSettings> fTrainingSettings;     ///< The vector defining each training strategy
   
    
   void Init();
    
   // the option handling methods
   void DeclareOptions();
   void ProcessOptions();
   

   // Write and read weights from an XML file
   static inline void WriteMatrixXML(void *parent, const char *name,
                                     const TMatrixT<Double_t> &X);
   static inline void ReadMatrixXML(void *xml, const char *name,
                                    TMatrixT<Double_t> &X);
    
   ClassDef(MethodCNN,0);

protected:
   void GetHelpMessage() const;
    
public:
    
   // Standard Constructors
   MethodCNN(const TString& jobName,
             const TString&  methodTitle,
             DataSetInfo& theData,
             const TString& theOption);
    
   MethodCNN(DataSetInfo& theData,
             const TString& theWeightFile);
    
   virtual ~MethodCNN();
    
   virtual Bool_t HasAnalysisType(Types::EAnalysisType type,
                                  UInt_t numberClasses,
                                  UInt_t numberTargets );
    
   LayoutVector_t   ParseLayoutString(TString layerSpec);
    
   void Train();
   void TrainGpu();
   void TrainCpu();
    
   virtual Double_t GetMvaValue( Double_t* err=0, Double_t* errUpper=0 );
   // virtual const std::vector<Float_t>& GetRegressionValues();
   // virtual const std::vector<Float_t>& GetMulticlassValues();
    
   using MethodBase::ReadWeightsFromStream;
    
   // write weights to stream
   void AddWeightsXMLTo     ( void* parent ) const;
    
   // read weights from stream
   void ReadWeightsFromStream( std::istream & i );
   void ReadWeightsFromXML   ( void* wghtnode );
    
   // ranking of input variables
   const Ranking* CreateRanking();

};
} // namespace TMVA




#endif
