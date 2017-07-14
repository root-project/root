// @(#)root/tmva/tmva/dnn:$Id$
// Author: Akshay Vashistha (ajatgd)

/*************************************************************************
 * Copyright (C) 2016, ajatgd                                            *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////
// Generic test for DataLoader implementations. //
//////////////////////////////////////////////////

#include "../Utility.h"
#include "TMVA/DNN/DAE/DataLoader.h"
#include "TMVA/DNN/DAE/DenoiseAE.h"

namespace TMVA {
namespace DNN {
namespace DAE {
template <typename Architecture_t> auto testDataLoader() -> void {
  using Scalar_t = typename Architecture_t::Scalar_t;
  using Matrix_t = typename Architecture_t::Matrix_t;
  using DataLoader_t = TDataLoader<MatrixInput_t, Architecture_t>;

  TMatrixT<Double_t> X(10, 1);
  MatrixInput_t input(X, X);

  DataLoader_t loader(input, 10, 5, 1, 1);

  Matrix_t A(loader.GetInput());

  for (size_t i = 0; i < (size_t)A.GetNrows(); i++) {
    for (size_t j = 0; j < (size_t)A.GetNcols(); j++) {
      std::cout << A(i, j) << "\t";
    }
    std::cout << std::endl;
  }
}
}
}
}
