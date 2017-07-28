// @(#)root/tmva/tmva/dnn:$Id$
// Author: Akshay Vashistha

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TLogisticRegressionLayer                                              *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Logistic Regression Layer                                                 *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Akshay Vashistha <akshayvashistha1995@gmail.com>  - CERN, Switzerland     *
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


 #ifndef TMVA_DAE_LOGISTIC_REGRESSION_LAYER
 #define TMVA_DAE_LOGISTIC_REGRESSION_LAYER

 #include "TMatrix.h"

 #include "TMVA/DNN/GeneralLayer.h"
 #include "TMVA/DNN/Functions.h"

 #include <iostream>
 #include <vector>

 namespace TMVA {
 namespace DNN {
 namespace DAE {

 /** \class TLogisticRegressionLayer
      LogisticRegression Layer is used in the finetune step of training for AutoEncoders.
      This is the supervised learning step to classify output.
 */

template <typename Architecture_t>
class TLogisticRegressionLayer : public VGeneralLayer<Architecture_t> {
public:
   using Matrix_t = typename Architecture_t::Matrix_t;
   using Scalar_t = typename Architecture_t::Scalar_t;

   size_t fInputUnits; ///< Number of input units

   size_t fOutputUnits; ///< Number of output Units

   Matrix_t fWeights; ///< Weights associated with layer

   Matrix_t fBiases; ///<  Bias associated

   size_t fTestDataBatchSize; ///< Number of testing units in testing btch


   /* constructor */
   TLogisticRegressionLayer(size_t BatchSize, size_t InputUnits, size_t OutputUnits, size_t TestDataBatchSize);

   /*! Copy the denoise layer provided as a pointer */
   TLogisticRegressionLayer(TLogisticRegressionLayer<Architecture_t> *layer);

   /* copy constructor */
   TLogisticRegressionLayer(const TLogisticRegressionLayer &);

   /* had to use initialize as initialize in GeneralLayer initializes its weights and biases*/
   void Initialize(DNN::EInitialization m);

   void Forward(std::vector<Matrix_t> input, bool applyDropout = false);

   void Backward(std::vector<Matrix_t> &gradients_backward,
                 const std::vector<Matrix_t> &activations_backward);

   /*  Getters */
   size_t GetInputUnits()         const {return fInputUnits;}
   size_t GetOutputUnits()        const {return fOutputUnits;}
   size_t GetTestDataBatchSize()  const {return fTestDataBatchSize;}

   const Matrix_t & GetWeights() const {return fWeights;}
   Matrix_t & GetWeights() {return fWeights;}

   const Matrix_t & GetBiases() const {return fBiases;}
   Matrix_t & GetBiases() {return fBiases;}

   /* Train the Logistic Regression Layer */
   void TrainLogReg(std::vector<Matrix_t> &input, std::vector<Matrix_t> &output, Scalar_t learningRate);

   /* Predict output of Logistic Regression Layer, should be used as a
      successive call  after TrainLogReg() */
   std::vector<Matrix_t> PredictLogReg(std::vector<Matrix_t> &input, Scalar_t learningRate);

   void Print() const;

};
//______________________________________________________________________________

template <typename Architecture_t>
TLogisticRegressionLayer<Architecture_t>::TLogisticRegressionLayer(size_t batchSize, size_t inputUnits,
                                                                   size_t outputUnits, size_t testDataBatchSize)
   : VGeneralLayer<Architecture_t>(batchSize, 1, 1, 0, 0, 0, 0, 0,0,0, 0, 0,
   0, batchSize, outputUnits, 1, EInitialization::kUniform),
   fInputUnits(inputUnits), fOutputUnits(outputUnits),
   fWeights(outputUnits,inputUnits), fBiases(outputUnits,1), fTestDataBatchSize(testDataBatchSize)

{
  // Output Tensor will be created in General Layer
}



//______________________________________________________________________________
template <typename Architecture_t>
TLogisticRegressionLayer<Architecture_t>::TLogisticRegressionLayer(TLogisticRegressionLayer<Architecture_t> *layer)
   : VGeneralLayer<Architecture_t>(layer),
   fInputUnits(layer->GetInputUnits()), fOutputUnits(layer->GetOutputUnits()),
   fWeights(layer->GetOutputUnits(),layer->GetInputUnits()), fBiases(layer->GetOutputUnits(),1),
   fTestDataBatchSize(layer->GetTestDataBatchSize())
{
  // Output Tensor will be created in General Layer
}

//______________________________________________________________________________
template <typename Architecture_t>
TLogisticRegressionLayer<Architecture_t>::TLogisticRegressionLayer(const TLogisticRegressionLayer &logistic)
   : VGeneralLayer<Architecture_t>(logistic),
   fInputUnits(logistic.GetInputUnits()), fOutputUnits(logistic.GetOutputUnits()),
   fWeights(logistic.GetOutputUnits(),logistic.GetInputUnits()), fBiases(logistic.GetOutputUnits(),1),
   fTestDataBatchSize(logistic.GetTestDataBatchSize())

{
  // Output Tensor will be created in General Layer

}
//______________________________________________________________________________
template <typename Architecture_t>
auto TLogisticRegressionLayer<Architecture_t>::TrainLogReg(std::vector<Matrix_t> &input,
                                                          std::vector<Matrix_t> &outputLabel,
                                                          Scalar_t learningRate)
-> void
{
   Matrix_t p(this->GetOutputUnits(), 1);
   Matrix_t difference(this->GetOutputUnits(), 1);
   for(size_t i=0; i<this->GetBatchSize(); i++)
   {
      Architecture_t::ForwardLogReg(input[i], p, this->GetWeights());
      Architecture_t::AddBiases(p, this->GetBiases());
      Architecture_t::SoftmaxAE(p);
      Architecture_t::UpdateParamsLogReg(input[i], outputLabel[i], difference, p,
                                        this->GetWeights(), this->GetBiases(),
                                        learningRate, this->GetBatchSize());
   }
}

//______________________________________________________________________________
template <typename Architecture_t>
auto TLogisticRegressionLayer<Architecture_t>::PredictLogReg(std::vector<Matrix_t> &input,
                                                Scalar_t learningRate)
-> std::vector<Matrix_t>
{
   for(size_t i=0; i<this->GetTestDataBatchSize(); i++)
   {
      Architecture_t::ForwardLogReg(input[i], this->GetOutputAt(i), this->GetWeights());
      Architecture_t::SoftmaxAE(this->GetOutputAt(i));
   }
   return this->GetOutput();
}

//______________________________________________________________________________
template<typename Architecture_t>
auto TLogisticRegressionLayer<Architecture_t>::Print() const
-> void
{
   std::cout << "Input Batch Size: " << this->GetBatchSize() << "\n"
            << "Output Batch size: " << this->GetTestDataBatchSize() << "\n"
            << "Input Units: " << this->GetInputUnits() << "\n"
            << "Output Units: " << this->GetOutputUnits() << "\n";
}
//______________________________________________________________________________
template <typename Architecture_t>
auto TLogisticRegressionLayer<Architecture_t>::Initialize(DNN::EInitialization m)
-> void

{
   DNN::initialize<Architecture_t>(fWeights, m);
   DNN::initialize<Architecture_t>(fBiases, DNN::EInitialization::kZero);
}
//______________________________________________________________________________
template <typename Architecture_t>
auto inline TLogisticRegressionLayer<Architecture_t>::Backward(std::vector<Matrix_t> &gradients_backward,
                                                     const std::vector<Matrix_t> &activations_backward)
-> void
{
}

//______________________________________________________________________________
template <typename Architecture_t>
auto TLogisticRegressionLayer<Architecture_t>::Forward(std::vector<Matrix_t> input, bool applyDropout)
-> void
{
}
//______________________________________________________________________________

}// namespace DAE
}//namespace DNN
}//namespace TMVA
#endif /* TMVA_DAE_LOGISTIC_REGRESSION_LAYER */
