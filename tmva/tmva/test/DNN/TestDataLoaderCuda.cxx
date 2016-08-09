// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 08/08/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////
// Test the generic data loader for the CUDA implementation. //
///////////////////////////////////////////////////////////////

#include "TMVA/DNN/Net.h"
#include "TMVA/DNN/DataLoader.h"
#include "TMVA/DNN/Architectures/Cuda.h"
#include "Utility.h"
#include "TMatrix.h"

using namespace TMVA::DNN;

int main()
{
   using Matrix_t     = typename TCuda::Matrix_t;
   using DataLoader_t = TDataLoader<MatrixInput_t, TCuda>;

   TMatrixT<Double_t> A(20, 5), B(20, 5);
   randomMatrix(A);
   randomMatrix(B);

   for(size_t i = 0; i < 20; i++) {
      for (size_t j = 0; j < 5; j++) {
         A(i,j) = i;
         B(i,j) = j;
      }
   }

   MatrixInput_t data(A, B);
   DataLoader_t loader(data, 20, 5, 5, 5, 4);

   TNet<TCuda> net(5, 5, ELossFunction::MEANSQUAREDERROR, ERegularization::NONE);
   net.AddLayer(5, EActivationFunction::IDENTITY);
   net.Initialize(EInitialization::IDENTITY);

   std::vector<TNet<TCuda>> nets{};

   std::vector<TCudaMatrix> ms{};
   nets.reserve(4);
   ms.reserve(4);
   for (size_t i = 0; i < 4; i++) {
      nets.push_back(net);
      ms.push_back(TCudaMatrix(5,5));
      for (size_t j = 0; j < net.GetDepth(); j++) {
         TCuda::Copy(nets[i].GetLayer(j).GetWeights(), net.GetLayer(j).GetWeights());
         TCuda::Copy(nets[i].GetLayer(j).GetBiases(), net.GetLayer(j).GetBiases());
      }
   }

   std::cout << "net size: " << nets.size() << std::endl;
   for (size_t i = 0; i < 4; i++) {
   }
   std::cout << "ms size: " << ms.size() << std::endl;

   for (size_t i = 0; i < 4; i++) {
      auto b = loader.GetBatch();
      nets[i].Forward(b.GetInput());
      TCuda::Copy(ms[i], nets[i].GetOutput());
   }
   for (size_t i = 0; i < 4; i++) {
      ((TMatrixT<Double_t>) ms[i]).Print();
   }
}





