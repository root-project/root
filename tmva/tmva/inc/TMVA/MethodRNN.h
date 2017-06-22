// @(#)root/tmva/:$Id$
// Author: Saurav Shekhar 21/06/17

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : MethodRNN                                                             *
 *                                                                                *
 * Description:                                                                   *
 *      NeuralNetwork                                                             *
 *                                                                                *
 * Authors (alphabetical):                                                        * *      Saurav Shekhar    <sauravshekhar01@gmail.com> - ETH Zurich, Switzerland   *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 * All rights reserved.                                                           *
 *      CERN, Switzerland                                                         *
 *                                                                                *
 * For the licensing terms see $ROOTSYS/LICENSE.                                  *
 * For the list of contributors see $ROOTSYS/README/CREDITS.                      *
 **********************************************************************************/

#ifndef ROOT_TMVA_MethodRNN
#define ROOT_TMVA_MethodRNN

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// MethodRNN                                                            //
//                                                                      //
// Recurrent Neural Network implementation                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TString.h"
#include "TMVA/MethodBase.h"
#include "TMVA/DNN/RNN/RNNLayer.h"

#include "TMVA/DNN/Architectures/Reference.h"

#ifdef DNNCPU
#include "TMVA/DNN/Architectures/Cpu.h"
#endif

#ifdef DNNCUDA
#include "TMVA/DNN/Architectures/Cuda.h"
#endif

#include <vector>
#include <tuple>

namespace TMVA 
{

class MethodRNN : public MethodBase
{
  using Architecture_t = DNN::TReference<Double_t>;
  //using RecurrentNet_t = DNN::RNN::TRecurrentNet<Architecture_t>;
  using Matrix_t       = typename Architecture_t::Matrix_t;

private:
  // TimeSteps, stateSize, inputSize, Activationfn
  using LayoutVector_t   = std::tuple<int, int, int, DNN::EActivationFunction>;  
  using KeyValueVector_t = std::vector<std::map<TString, TString>>;

  struct TTrainingSettings
    {
        size_t                batchSize;
        size_t                testInterval;
        size_t                convergenceSteps;
        DNN::ERegularization  regularization;
        Double_t              learningRate;
        Double_t              momentum;
        Double_t              weightDecay;
        Double_t              dropoutProbability;
        bool                  multithreading;
    };

  //RecurrentNet_t       fRecurrentNet;           ///< Recurrent net to train
  DNN::EInitialization fWeightInitialization;   ///< Initialization
  DNN::EOutputFunction fOutputFunction;         ///< Output fn for making predictions

  TString   fLayoutString;                    ///< String defining the layout of the RNN
  TString   fErrorStrategy;                   ///< String defining the error strategy for training
  TString   fTrainingStrategyString;          ///< String defining the training strategy
  TString   fWeightInitializationString;      ///< String defining the weight initialization method
  TString   fArchitectureString;              ///< String defining the architecure: CPU or GPU
  bool      fResume;
  
  LayoutVector_t                 fLayout;             ///< Dimensions and activation function of Layer
  std::vector<TTrainingSettings> fTrainingSettings;   ///< The vector defining each training strategy
  
   
  void Init();
   
  // the option handling methods
  void DeclareOptions();
  void ProcessOptions();
  
  // Write and read weights from an XML file
  static inline void WriteMatrixXML(void *parent, const char *name,
                                    const TMatrixT<Double_t> &X);
  static inline void ReadMatrixXML(void *xml, const char *name,
                                   TMatrixT<Double_t> &X);
   
  ClassDef(MethodRNN,0); // neural network

protected:

   void MakeClassSpecific( std::ostream&, const TString& ) const;
   void GetHelpMessage() const;

public:
    
   // Standard Constructors
   MethodRNN(const TString& jobName,
             const TString&  methodTitle,
             DataSetInfo& theData,
             const TString& theOption);
    
   MethodRNN(DataSetInfo& theData,
             const TString& theWeightFile);
    
   virtual ~MethodRNN();
    
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
