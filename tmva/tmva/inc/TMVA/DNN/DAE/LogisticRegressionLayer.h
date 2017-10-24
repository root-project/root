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

   size_t fTestDataBatchSize; ///< Number of testing units in testing btch

   Scalar_t fLearningRate; ///< Learning Rate


   /* constructor */
   TLogisticRegressionLayer(size_t BatchSize, size_t InputUnits, size_t OutputUnits, size_t TestDataBatchSize,
                            Scalar_t LearningRate);

   /*! Copy the denoise layer provided as a pointer */
   TLogisticRegressionLayer(TLogisticRegressionLayer<Architecture_t> *layer);

   /* copy constructor */
   TLogisticRegressionLayer(const TLogisticRegressionLayer &);

   // This is basically the prediction step. Can be used in DeepNet to Predict output.
   void Forward(std::vector<Matrix_t> &input, bool applyDropout = false);

   void Backward(std::vector<Matrix_t> &gradients_backward,
                 const std::vector<Matrix_t> &activations_backward,
                 std::vector<Matrix_t> &inp1,
                 std::vector<Matrix_t> &inp2);

   /*  Getters */
   size_t GetInputUnits()         const {return fInputUnits;}
   size_t GetOutputUnits()        const {return fOutputUnits;}
   size_t GetTestDataBatchSize()  const {return fTestDataBatchSize;}
   Scalar_t GetLearningRate()     const {return fLearningRate;}


   /* Train the Logistic Regression Layer */

   /* Predict output of Logistic Regression Layer, should be used as a
      successive call  after TrainLogReg() */
   //std::vector<Matrix_t> PredictLogReg(std::vector<Matrix_t> &input);

   void Print() const;

};
//______________________________________________________________________________

template <typename Architecture_t>
TLogisticRegressionLayer<Architecture_t>::TLogisticRegressionLayer(size_t batchSize, size_t inputUnits,
                                                                   size_t outputUnits, size_t testDataBatchSize,
                                                                   Scalar_t learningRate)
   : VGeneralLayer<Architecture_t>(batchSize, 1, 1, 0, 0, 0, 0, 1, {outputUnits}, {inputUnits}, 1, {outputUnits},
   {1}, testDataBatchSize, outputUnits, 1, EInitialization::kUniform),
   fInputUnits(inputUnits), fOutputUnits(outputUnits),
   fTestDataBatchSize(testDataBatchSize), fLearningRate(learningRate)

{
  // Output Tensor will be created in General Layer
}



//______________________________________________________________________________
template <typename Architecture_t>
TLogisticRegressionLayer<Architecture_t>::TLogisticRegressionLayer(TLogisticRegressionLayer<Architecture_t> *layer)
   : VGeneralLayer<Architecture_t>(layer),
   fInputUnits(layer->GetInputUnits()), fOutputUnits(layer->GetOutputUnits()),
   fTestDataBatchSize(layer->GetTestDataBatchSize()), fLearningRate(layer->GetLearningRate())
{
  // Output Tensor will be created in General Layer
}

//______________________________________________________________________________
template <typename Architecture_t>
TLogisticRegressionLayer<Architecture_t>::TLogisticRegressionLayer(const TLogisticRegressionLayer &logistic)
   : VGeneralLayer<Architecture_t>(logistic),
   fInputUnits(logistic.fInputUnits), fOutputUnits(logistic.fOutputUnits),
   fTestDataBatchSize(logistic.fTestDataBatchSize), fLearningRate(logistic.fLearningRate)

{
  // Output Tensor will be created in General Layer

}
//______________________________________________________________________________
//______________________________________________________________________________
template <typename Architecture_t>
auto inline TLogisticRegressionLayer<Architecture_t>::Backward(std::vector<Matrix_t> &inputLabel,
                                                               const std::vector<Matrix_t> & /*inp1*/,
                                                               std::vector<Matrix_t> &input, std::vector<Matrix_t> &
                                                               /*inp2*/) -> void
{
   for(size_t i=0; i<this->GetBatchSize(); i++)
   {
      Matrix_t p(this->GetOutputUnits(), 1);
      Matrix_t difference(this->GetOutputUnits(), 1);
      for(size_t j=0; j<(size_t)p.GetNrows(); j++)
      {
         for(size_t k=0; k<(size_t)p.GetNcols(); k++)
         {
            p(j,k)=0;
            difference(j,k)=0;
         }
      }
      Architecture_t::ForwardLogReg(input[i], p, this->GetWeightsAt(0));
      Architecture_t::AddBiases(p, this->GetBiasesAt(0));
      Architecture_t::SoftmaxAE(p);
      Architecture_t::UpdateParamsLogReg(input[i], inputLabel[i], difference, p,
                                         this->GetWeightsAt(0), this->GetBiasesAt(0),
                                         this->GetLearningRate(), this->GetBatchSize());

   }

}

//______________________________________________________________________________
template <typename Architecture_t>
auto TLogisticRegressionLayer<Architecture_t>::Forward(std::vector<Matrix_t> &input, bool /*applyDropout*/) -> void
{
   for(size_t i=0; i<this->GetTestDataBatchSize(); i++)
   {
      Architecture_t::ForwardLogReg(input[i], this->GetOutputAt(i), this->GetWeightsAt(0));
      Architecture_t::AddBiases(this->GetOutputAt(i), this->GetBiasesAt(0));
      Architecture_t::SoftmaxAE(this->GetOutputAt(i));

   }
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

            std::cout<<"output: "<<std::endl;
            for(size_t i=0; i<this->GetOutput().size(); i++)
            {
               for (Int_t j = 0; j < this->GetOutputAt(i).GetNrows(); j++) {
                  for (Int_t k = 0; k < this->GetOutputAt(i).GetNcols(); k++) {
                     std::cout << this->GetOutputAt(i)(j, k) << "\t";
                  }
                  std::cout << std::endl;
                 }
            }
}

//______________________________________________________________________________


//______________________________________________________________________________

//______________________________________________________________________________

}// namespace DAE
}//namespace DNN
}//namespace TMVA
#endif /* TMVA_DAE_LOGISTIC_REGRESSION_LAYER */
