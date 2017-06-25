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


#include "TMVA/MethodDL.h"
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

enum ECNNLayerType
{
   kConv = 1,
   kPool = 2,
   kFC =   3
};
    
    
class MethodCNN: public MethodDL
{
public:
    
   using Architecture_t  = TReference<Double_t>;
   using ConvNet_t       = TConvNet<Architecture_t>;
   using Matrix_t        = typename Architecture_t::Matrix_t;

private:
   // TYPE | DEPTH (WIDTH) | FLTH (0) | FLTW (0) | STRR (0) | STRC (0) | ZPADH (0) | ZPADW (0) |ACTFNC
   using LayoutVector_t   = std::vector<std::tuple<ECNNLayerType, int, int,
                                                   int, int, int, int, int, EActivationFunction>>;
    
   TString   fLayoutString;                    ///< The string defining the layout of the CNN
    
   // the option handling methods
   void DeclareOptions();
   void ProcessOptions();
    
   void Init();
    
   ConvNet_t fConvNet;                 ///< The convolutional neural net to train
   LayoutVector_t fLayout;             ///< Dimensions and activation functions of each layer in the CNN
    
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
    
   // Check the type of analysis the CNN can do
   virtual Bool_t HasAnalysisType(Types::EAnalysisType type,
                                  UInt_t numberClasses,
                                  UInt_t numberTargets );
    
   // Method for parsing the layout specifications of the CNN
   void ParseLayoutString(TString layerSpec);
    
   // Methods for training the CNN
   void Train();
   void TrainGpu();
   void TrainCpu();
    
   virtual Double_t GetMvaValue( Double_t* err=0, Double_t* errUpper=0 );
   // virtual const std::vector<Float_t>& GetRegressionValues();
   // virtual const std::vector<Float_t>& GetMulticlassValues();
    
    
   // Methods for writing and reading weights
   void AddWeightsXMLTo     ( void* parent ) const;
   void ReadWeightsFromStream( std::istream & i );
   void ReadWeightsFromXML   ( void* wghtnode );
    
   // ranking of input variables
   const Ranking* CreateRanking();
    
   TString GetLayoutString()   {return fLayoutString;}
    
   void SetLayoutString(TString layoutString) {
      fLayoutString = layoutString;
   }

};
} // namespace TMVA




#endif
