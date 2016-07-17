// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 12/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////
// Generic test for DataLoader implementations. //
//////////////////////////////////////////////////

#include "TMVA/DNN/Net.h"
#include "Utility.h"

namespace TMVA
{
namespace DNN
{

template <typename Architecture_t>
auto testDataLoader()
    -> typename Architecture_t::Scalar_t
{

    using Scalar_t     = typename Architecture_t::Scalar_t;
    using Matrix_t     = typename Architecture_t::Matrix_t;
    using Net_t        = TNet<Architecture_t>;

    using DataLoader_t = typename Architecture_t::template DataLoader_t<MatrixInput_t>;

    Matrix_t X(2000, 100); randomMatrix(X);
    MatrixInput_t input(X, X);
    DataLoader_t loader(input, 2000, 20, 100, 100);

    Net_t net(20, 100, ELossFunction::MEANSQUAREDERROR);
    net.AddLayer(100,  EActivationFunction::IDENTITY);
    net.AddLayer(100,  EActivationFunction::IDENTITY);
    net.Initialize(EInitialization::IDENTITY);

    Scalar_t maximumError = 0.0;
    for (auto b : loader) {
        Matrix_t inputMatrix  = b.GetInput();
        Matrix_t outputMatrix = b.GetOutput();
        Scalar_t error = net.Loss(inputMatrix, outputMatrix);
        maximumError = std::max(error, maximumError);
    }

    return maximumError;
}

} // namespace DNN
} // namespace TMVA
