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

   std::vector<Matrix_t> fInput;

   size_t fEpochs;


   /* constructor */
   TLogisticRegressionLayer(size_t BatchSize, size_t InputUnits, size_t OutputUnits, size_t TestDataBatchSize,
                            Scalar_t LearningRate, size_t epochs);

   /*! Copy the denoise layer provided as a pointer */
   TLogisticRegressionLayer(TLogisticRegressionLayer<Architecture_t> *layer);

   /* copy constructor */
   TLogisticRegressionLayer(const TLogisticRegressionLayer &);

   /* had to use initialize as initialize in GeneralLayer initializes its weights and biases*/
   //void Initialize(DNN::EInitialization m);
   // actually a predict log reg fn.
   void Forward(std::vector<Matrix_t> input, bool applyDropout = false);

   void Backward(std::vector<Matrix_t> &gradients_backward,
                 const std::vector<Matrix_t> &activations_backward);

   /*  Getters */
   size_t GetInputUnits()         const {return fInputUnits;}
   size_t GetOutputUnits()        const {return fOutputUnits;}
   size_t GetTestDataBatchSize()  const {return fTestDataBatchSize;}
   Scalar_t GetLearningRate()     const {return fLearningRate;}
   size_t GetEpochs()             const {return fEpochs;}

   const std::vector<Matrix_t> &GetInput() const { return fInput; }
   std::vector<Matrix_t> &GetInput() { return fInput; }

   Matrix_t &GetInputAt(size_t i) { return fInput[i]; }
   const Matrix_t &GetInputAt(size_t i) const { return fInput[i]; }

   /* Train the Logistic Regression Layer */
   void TrainLogReg(std::vector<Matrix_t> &input, std::vector<Matrix_t> &output);

   /* Predict output of Logistic Regression Layer, should be used as a
      successive call  after TrainLogReg() */
   //std::vector<Matrix_t> PredictLogReg(std::vector<Matrix_t> &input);

   void Print() const;

};
//______________________________________________________________________________

template <typename Architecture_t>
TLogisticRegressionLayer<Architecture_t>::TLogisticRegressionLayer(size_t batchSize, size_t inputUnits,
                                                                   size_t outputUnits, size_t testDataBatchSize,
                                                                   Scalar_t learningRate, size_t epochs)
   : VGeneralLayer<Architecture_t>(batchSize, 1, 1, 0, 0, 0, 0, 1, {outputUnits}, {inputUnits}, 1, {outputUnits},
   {1}, testDataBatchSize, outputUnits, 1, EInitialization::kUniform),
   fInputUnits(inputUnits), fOutputUnits(outputUnits),
   fTestDataBatchSize(testDataBatchSize), fLearningRate(learningRate),
   fInput(), fEpochs(epochs)

{
  for (size_t i = 0; i < batchSize; i++)
  {
     fInput.emplace_back(inputUnits,1);
  }
  // Output Tensor will be created in General Layer
}



//______________________________________________________________________________
template <typename Architecture_t>
TLogisticRegressionLayer<Architecture_t>::TLogisticRegressionLayer(TLogisticRegressionLayer<Architecture_t> *layer)
   : VGeneralLayer<Architecture_t>(layer),
   fInputUnits(layer->GetInputUnits()), fOutputUnits(layer->GetOutputUnits()),
   fTestDataBatchSize(layer->GetTestDataBatchSize()), fLearningRate(layer->GetLearningRate()),
   fEpochs(layer->GetEpochs())
{
  for (size_t i = 0; i < layer->GetBatchSize() ; i++)
  {
     this->GetInput().emplace_back(layer->GetInputUnits(),1);
  }
  // Output Tensor will be created in General Layer
}

//______________________________________________________________________________
template <typename Architecture_t>
TLogisticRegressionLayer<Architecture_t>::TLogisticRegressionLayer(const TLogisticRegressionLayer &logistic)
   : VGeneralLayer<Architecture_t>(logistic),
   fInputUnits(logistic.fInputUnits), fOutputUnits(logistic.fOutputUnits),
   fTestDataBatchSize(logistic.fTestDataBatchSize), fLearningRate(logistic.fLearningRate),
   fEpochs(logistic.fEpochs)

{
  for (size_t i = 0; i < logistic.GetBatchSize() ; i++)
  {
     this->GetInput().emplace_back(logistic.GetInputUnits(),1);
  }
  // Output Tensor will be created in General Layer

}
//______________________________________________________________________________
template <typename Architecture_t>
auto inline TLogisticRegressionLayer<Architecture_t>::Backward(std::vector<Matrix_t> &outputLabel,
                                                     const std::vector<Matrix_t> &input)
-> void
{
   for(size_t i=0; i<this->GetBatchSize(); i++)
   {
      Architecture_t::Copy(this->GetInputAt(i), input[i]);
   }

   for(size_t epoch=0; epoch<this->GetEpochs(); epoch++)
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
         Architecture_t::ForwardLogReg(this->GetInputAt(i), p, this->GetWeightsAt(0));
         Architecture_t::AddBiases(p, this->GetBiasesAt(0));
         Architecture_t::SoftmaxAE(p);
         Architecture_t::UpdateParamsLogReg(this->GetInputAt(i), outputLabel[i], difference, p,
                                            this->GetWeightsAt(0), this->GetBiasesAt(0),
                                            this->GetLearningRate(), this->GetBatchSize());

      }
   }
}

//______________________________________________________________________________
template <typename Architecture_t>
auto TLogisticRegressionLayer<Architecture_t>::Forward(std::vector<Matrix_t> input, bool applyDropout)
-> void
{
   for(size_t i=0; i<this->GetTestDataBatchSize(); i++)
   {
      Architecture_t::ForwardLogReg(input[i], this->GetOutputAt(i), this->GetWeightsAt(0));
      Architecture_t::AddBiases(this->GetOutputAt(i), this->GetBiasesAt(0));
      Architecture_t::SoftmaxAE(this->GetOutputAt(i));

   }
   //return this->GetOutput();
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
                for(size_t j=0; j<this->GetOutputAt(i).GetNrows(); j++)
                {
                   for(size_t k=0; k<this->GetOutputAt(i).GetNcols(); k++)
                   {
                      std::cout<<this->GetOutputAt(i)(j,k)<<"\t";
                   }
                   std::cout<<std::endl;
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
