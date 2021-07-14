// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 10/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

 //////////////////////////////////////////////////////////////////
 // Implementation of the activation functions for the reference //
 // implementation.                                              //
 //////////////////////////////////////////////////////////////////


#include "TMVA/DNN/Architectures/Reference.h"
#include "TRandom.h"
#include <time.h>

namespace TMVA
{
namespace DNN
{

//______________________________________________________________________________

template<typename Real_t>
void TReference<Real_t>::DropoutForward(TMatrixT<Real_t> & B, TDescriptors*, TWorkspace*, Real_t dropoutProbability)
{
   size_t m,n;
   m = B.GetNrows();
   n = B.GetNcols();

   TRandom rand(time(nullptr));

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         Real_t r = rand.Uniform();
         if (r >= dropoutProbability) {
            B(i,j) = 0.0;
         } else {
            B(i,j) /= dropoutProbability;
         }
      }
   }
}

}
}
