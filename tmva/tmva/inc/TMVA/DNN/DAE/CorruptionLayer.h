// @(#)root/tmva/tmva/dnn:$Id$
// Author: Akshay Vashistha

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TCorruptionLayer                                                                  *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Corruption Layer for DeepAutoEncoders                                      *
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


#ifndef TMVA_DAE_CORRUPTION_LAYER
#define TMVA_DAE_CORRUPTION_LAYER

#include "TMatrix.h"

#include "TMVA/DNN/GeneralLayer.h"
#include "TMVA/DNN/Functions.h"

#include <iostream>
#include <vector>

namespace TMVA {
namespace DNN {
namespace DAE {

  /** \class TCorruptionLayer
    *  Used to corrupt the input values according to defined Corruption level.
  */

template <typename Architecture_t>
class TCorruptionLayer : public VGeneralLayer<Architecture_t> {
public:
   using Matrix_t = typename Architecture_t::Matrix_t;
   using Scalar_t = typename Architecture_t::Scalar_t;


   size_t fVisibleUnits; ///< total number of visible units

   Scalar_t fDropoutProbability; ///< Probability that an input is active.

   size_t fType; ///< Type of layer

   Scalar_t  fCorruptionLevel; ///<Corruption level for layer

   /*! Constructor. */
   TCorruptionLayer(size_t BatchSize, size_t VisibleUnits,Scalar_t DropoutProbability,
                    Scalar_t CorruptionLevel);

   /*! Copy the denoise layer provided as a pointer */
   TCorruptionLayer(TCorruptionLayer<Architecture_t> *layer);

   /*! Copy constructor. */
   TCorruptionLayer(const TCorruptionLayer &);

   void Forward(std::vector<Matrix_t> input, bool applyDropout = false);

   void Backward(std::vector<Matrix_t> &gradients_backward,
                 const std::vector<Matrix_t> &activations_backward);

   void Print() const;

   size_t GetVisibleUnits() const { return fVisibleUnits; }
   size_t GetType() const {return fType;}
   Scalar_t GetDropoutProbability() const { return fDropoutProbability; }
   Scalar_t GetCorruptionLevel() const {return fCorruptionLevel;}

};

//______________________________________________________________________________
template <typename Architecture_t>
TCorruptionLayer<Architecture_t>::TCorruptionLayer(size_t batchSize, size_t visibleUnits,
                           Scalar_t dropoutProbability, Scalar_t corruptionLevel)
   : VGeneralLayer<Architecture_t>(batchSize, 1, 1, 0, 0, 0, 0, 0, 0, 0,
   0,0,0, batchSize, visibleUnits, 1, EInitialization::kUniform),
   fVisibleUnits(visibleUnits), fDropoutProbability(dropoutProbability),
   fType(1), fCorruptionLevel(corruptionLevel)

{

}
//______________________________________________________________________________
template <typename Architecture_t>
TCorruptionLayer<Architecture_t>::TCorruptionLayer(TCorruptionLayer<Architecture_t> *layer)
   : VGeneralLayer<Architecture_t>(layer),
   fVisibleUnits(layer->GetVisibleUnits()),fType(1),
   fDropoutProbability(layer->GetDropoutProbability()),
   fCorruptionLevel(layer->GetCorruptionLevel())
{
  // Output Tensor will be created in General Layer
}
//______________________________________________________________________________
template <typename Architecture_t>
TCorruptionLayer<Architecture_t>::TCorruptionLayer(const TCorruptionLayer &corrupt)
   : VGeneralLayer<Architecture_t>(corrupt),
   fVisibleUnits(corrupt.fVisibleUnits),fType(1),
   fDropoutProbability(corrupt.fDropoutProbability),
   fCorruptionLevel(corrupt.fCorruptionLevel)


{
  // Output Tensor will be created in General Layer

}
//______________________________________________________________________________
template <typename Architecture_t>
auto TCorruptionLayer<Architecture_t>::Forward(std::vector<Matrix_t> input, bool applyDropout)
-> void
{
   for (size_t i = 0; i < this->GetBatchSize(); i++)
   {
      if (applyDropout && (this->GetDropoutProbability() != 1.0))
      {
         Architecture_t::Dropout(input[i], this->GetDropoutProbability());
      }
      Architecture_t::CorruptInput(input[i], this->GetOutputAt(i), this->GetCorruptionLevel());
   }
}
//______________________________________________________________________________
template <typename Architecture_t>
auto inline TCorruptionLayer<Architecture_t>::Backward(std::vector<Matrix_t> &gradients_backward,
                                                     const std::vector<Matrix_t> &activations_backward)
-> void
{
}
//______________________________________________________________________________
template<typename Architecture_t>
auto TCorruptionLayer<Architecture_t>::Print() const
-> void
{
   std::cout << "Batch Size: " << this->GetBatchSize() << "\n"
            << "Input Units: " << this->GetVisibleUnits() << "\n";

   for(size_t i=0; i<this->GetBatchSize(); i++)
   {
      for(size_t j=0; j<this->GetOutputAt(i).GetNrows(); j++)
      {
         for(size_t k=0; k<this->GetOutputAt(i).GetNcols(); k++)
         {
            std::cout<<this->GetOutputAt(i)(j,k)<<"\t";
      	 }
         std::cout<<std::endl;
      }
      std::cout<<std::endl;
      std::cout<<std::endl;
   }
}
//______________________________________________________________________________

}// namespace DAE
}//namespace DNN
}//namespace TMVA
#endif /* TMVA_DAE_CORRUPTION_LAYER*/
