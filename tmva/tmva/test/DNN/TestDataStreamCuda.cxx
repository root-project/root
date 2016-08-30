// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////
// Test TCudaDataLoader class for streaming data to CUDA devices. //
///////////////////////////////////////////////////////////////////

#include "TMatrix.h"
#include "TMVA/DNN/Architectures/Cuda.h"
#include "Utility.h"

using namespace TMVA::DNN;

int main()
{
    TMatrixT<Double_t> X(20,5), Y(20,10);
    randomMatrix(X);
    randomMatrix(Y);
    MatrixInput_t inputData(X, Y);

    TCudaDataLoader<MatrixInput_t> dataLoader(inputData, 20, 5, 10, 2, 2, 1);

    CudaDouble_t maximumError = 0.0;
    CudaDouble_t error        = 0.0;

    for (auto b : dataLoader) {
        TMatrixT<CudaDouble_t> X2(b.GetInput());
        TMatrixT<CudaDouble_t> Y2(b.GetOutput());

        std::cout << "::: BATCH :::" << std::endl;

        X.Print();
        X2.Print();
        Y.Print();
        Y2.Print();
        maximumError = std::max(error, maximumError);
        maximumError = std::max(error, maximumError);
    }

    std::cout << "Test data stream: Max. rel. error = " << maximumError << std::endl;

}
