// @(#)root/tmva/tmva/dnn:$Id$
// Author: Akshay Vashistha

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TransformLayer                                                                  *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Transform Layer for DeepAutoEncoders                                      *
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

#ifndef TMVA_TRANSFORMLAYER
#define TMVA_TRANSFORMLAYER

#include "TMVA/DNN/GeneralLayer.h"
#include "DenoiseLayer.h"
#include "TMVA/DNN/Functions.h"
#include "TMatrix.h"

#include <cmath>
#include <iostream>
#include <cstdlib>
#include <vector>

namespace TMVA {
namespace DNN {
namespace DAE {


/** \class Transform Layer
     Transform Layer is used next to the Denoise Layer. It is used to get
     compressed output for each set of input. The updated weights and biases
     are shared from Denoise Layer to Transform Layer. The compressed input is
     then passed to next Denoise Layer or LogisticRegression layer in Deep
     Network.
*/


template <typename Architecture_t>
class TransformLayer : public VGeneralLayer<Architecture_t> {
public:

  using Matrix_t = typename Architecture_t::Matrix_t;
  using Scalar_t = typename Architecture_t::Scalar_t;
  using Tensor_t = std::vector<Matrix_t>

  size_t fInputUnits; ///< Number of input units to preceding Denoise Layer.

  size_t fOutputUnits; ///< Number of compressed units specified.

  Matrix_t fWeights; ///< To store weights received from preceding Denoise Layer.

  Matrix_t fBiases; ///< To store biases received from preceding Denoise Layer.

  /* constructor */
  TransformLayer(size_t BatchSize, size_t InputUnits, size_t OutputUnits);

  /*! Copy the denoise layer provided as a pointer */
  TransformLayer(TransformLayer<Architecture_t> *layer);

  /* copy constructor */
  TransformLayer(const TransformLayer &);

  /*
   Transform the matrix from Input to Transformed.
   This transformed matrix is same as the hidden layer we get in TDAE class.
   This transformed matrix will be the input to next layer in stacking steps.
  */
  void Transform(Tensor_t &input, Tensor_t &transformed);

  /*
      Initialize the Weights and Bias Matrices of this layer with same weight
      and bias as in the previously trained TDAE layer.
  */
  void Initialize(Matrix_t &, Matrix_t &);

  // Getters
  size_t GetInputUnits()                      const {return fInputUnits;}
  size_t GetOutputUnits()                     const {return fOutputUnits;}
  const Matrix_t & GetWeights()               const {return fWeights;}
  Matrix_t & GetWeights()                           {return fWeights;}
  const Matrix_t & GetBiases()                const {return fBiases;}
  Matrix_t & GetBiases()                      {return fBiases;}

};


//
//
//  TransformLayer Class - Implementation
//______________________________________________________________________________

template<typename Architecture_t>
TransformLayer<Architecture_t>::TransformLayer(size_t batchSize,
                                         size_t inputDepth,size_t inputHeight,
                                         size_t inputWidth, size_t outputUnits)
                           :VGeneralLayer<Architecture_t>(batchSize),
                            fInputUnits(inputHeight * inputWidth * inputDepth),
                            fOutputUnits(outputUnits),
                            fWeights(outputUnits,inputUnits),
                            fBiases(outputUnits,1)

{

}

//______________________________________________________________________________

template <typename Architecture_t>
TransformLayer<Architecture_t>::TransformLayer(TransformLayer<Architecture_t> *layer)
                  : VGeneralLayer<Architecture_t>(layer),
                    fInputUnits(layer->GetInputUnits()),
                    fOutputUnits(layer->GetOutputUnits()),
                    fWeights(layer->GetOutputUnits(), layer->GetInputUnits()),
                    fBiases(layer->GetOutputUnits(), 1)

{
  Architecture_t::Copy(fWeights, layer.GetWeights());
  Architecture_t::Copy(fBiases, layer.GetHBiases());
}

//______________________________________________________________________________

template<typename Architecture_t>
TransformLayer<Architecture_t>::TransformLayer(const TransformLayer &trans)
                               :VGeneralLayer<Architecture_t>(trans),
                                fInputUnits(trans.fInputUnits),
                                fOutputUnits(trans.fOutputUnits),
                                fWeights(trans.fOutputUnits,trans.fInputUnits),
                                fBiases(trans.fOutputUnits,1)
{

}

//______________________________________________________________________________

template <typename Architecture_t>
auto TransformLayer<Architecture_t>::Initialize(Matrix_t &Weights,
                                                Matrix_t &Biases)
-> void
{
  Architecture_t::Copy(fWeights, Weights);
  Architecture_t::Copy(fBiases, Biases);
}

//______________________________________________________________________________

template <typename Architecture_t>
auto TransformLayer<Architecture_t>::Transform(Tensor_t &input,
                                               Tensor_t &transformed)
-> void
{
  for (size_t i = 0; i < this->GetBatchSize(); i++)
  {
    Architecture_t::Transform(input[i], transformed[i],
                              this->GetWeights(), this->GetBiases());
    Architecture_t::Sigmoid(transformed[i]);
  }
}

//______________________________________________________________________________
template<typename Architecture_t>
auto TransformLayer<Architecture_t>::Print() const
-> void
{
   std::cout << "Batch Size: " << this->GetBatchSize() << "\n"
             << "Input Units: " << this->GetInputUnits() << "\n"
             << "compressed Units: " << this->GetOutputUnits() << "\n";
}
//______________________________________________________________________________

}// namespace DAE
}//namespace DNN
}//namespace TMVA
#endif /* TMVA_TRANSFORMLAYER */
