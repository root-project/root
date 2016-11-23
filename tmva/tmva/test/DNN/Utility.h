#ifndef TMVA_TEST_DNN_UTILITY
#define TMVA_TEST_DNN_UTILITY

#include <iostream>
#include <sstream>
#include <type_traits>
#include "stdlib.h"
#include "TRandom.h"
#include "TMVA/DNN/Architectures/Reference.h"
#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/Net.h"

namespace TMVA
{
namespace DNN
{

/** Construct a random linear neural network with up to five layers.*/
//______________________________________________________________________________
template <typename AArchitecture>
void constructRandomLinearNet(TNet<AArchitecture> & net)
{
    int nlayers = rand() % 5 + 1;

    std::vector<EActivationFunction> ActivationFunctions
    = {EActivationFunction::kIdentity};

    for (int i = 0; i < nlayers; i++) {
        int width = rand() % 20 + 1;
        EActivationFunction f =
        ActivationFunctions[rand() % ActivationFunctions.size()];
        net.AddLayer(width, f);
    }
}

/*! Set matrix to the identity matrix */
//______________________________________________________________________________
template <typename AMatrix>
void identityMatrix(AMatrix &X)
{
    size_t m, n;
    m = X.GetNrows();
    n = X.GetNcols();


    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
        X(i,j) = 0.0;
        }
        if (i < n) {
        X(i,i) = 1.0;
        }
    }
}

/*! Fill matrix with random, Gaussian-distributed values. */
//______________________________________________________________________________
template <typename AMatrix>
void randomMatrix(AMatrix &X)
{
    size_t m,n;
    m = X.GetNrows();
    n = X.GetNcols();

    TRandom rand(clock());

    Double_t sigma = sqrt(10.0);

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
        X(i,j) = rand.Gaus(0.0, sigma);
        }
    }
}

/*! Generate a random batch as input for a neural net. */
//______________________________________________________________________________
template <typename AMatrix>
void randomBatch(AMatrix &X)
{
    randomMatrix(X);
}

/*! Generate a random batch as input for a neural net. */
//______________________________________________________________________________
template <typename AMatrix>
void copyMatrix(AMatrix &X, const AMatrix &Y)
{
    size_t m,n;
    m = X.GetNrows();
    n = X.GetNcols();

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
        X(i,j) = Y(i,j);
        }
    }
}

/*! Apply functional to each element in the matrix. */
//______________________________________________________________________________
template <typename AMatrix, typename F>
void applyMatrix(AMatrix &X, F f)
{
    size_t m,n;
    m = X.GetNrows();
    n = X.GetNcols();

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
        X(i,j) = f(X(i,j));
        }
    }
}

/*! Combine elements of two given matrices into a single matrix using
 *  the given function f. */
//______________________________________________________________________________
template <typename AMatrix, typename F>
void zipWithMatrix(AMatrix &Z,
                    F f,
                    const AMatrix &X,
                    const AMatrix &Y)
{
    size_t m,n;
    m = X.GetNrows();
    n = X.GetNcols();

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
        Z(i,j) = f(X(i,j), Y(i,j));
        }
    }
}

/** Generate a random batch as input for a neural net. */
//______________________________________________________________________________
template <typename AMatrix, typename AFloat, typename F>
AFloat reduce(F f, AFloat start, const AMatrix &X)
{
    size_t m,n;
    m = X.GetNrows();
    n = X.GetNcols();

    AFloat result = start;

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
        result = f(result, X(i,j));
        }
    }
    return result;
}

/** Apply function to matrix element-wise and compute the mean of the resulting
 *  element values */
//______________________________________________________________________________
template <typename AMatrix, typename AFloat, typename F>
AFloat reduceMean(F f, AFloat start, const AMatrix &X)
{
    size_t m,n;
    m = X.GetNrows();
    n = X.GetNcols();

    AFloat result = start;

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
        result = f(result, X(i,j));
        }
    }
    return result / (AFloat) (m * n);
}

/** Compute the relative error of x and y normalized by y. Specialized for
 *  float and double to make sure both arguments are above expected machine
 *  precision (1e-5 and 1e-10). */
//______________________________________________________________________________
template <typename AFloat>
inline AFloat relativeError(const AFloat &x,
                            const AFloat &y);


//______________________________________________________________________________
template <>
inline Double_t relativeError(const Double_t &x,
                              const Double_t &y)
{
    if ((std::abs(x) > 1e-10) && (std::abs(y) > 1e-10)) {
        return std::fabs((x - y) / y);
    } else {
        return std::fabs(x - y);
    }
}

//______________________________________________________________________________
template <>
inline Real_t relativeError(const Real_t &x,
                            const Real_t &y)
{
    if ((std::abs(x) > 1e-5) && (std::abs(y) > 1e-5)) {
        return std::fabs((x - y) / y);
    } else {
        return std::fabs(x - y);
    }
}

/*! Compute the maximum, element-wise relative error of the matrices
*  X and Y normalized by the element of Y. Protected against division
*  by zero. */
//______________________________________________________________________________
template <typename AMatrix>
auto maximumRelativeError(const AMatrix &X,
                          const AMatrix &Y)
-> decltype(X(0,0))
{

    using AFloat = decltype(X(0,0));

    size_t m,n;
    m = X.GetNrows();
    n = X.GetNcols();

    AFloat maximumError = 0.0;

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
        AFloat error = relativeError(X(i,j), Y(i,j));
        maximumError = std::max(error, maximumError);
        }
    }
    return maximumError;
}

/*! Numerically compute the derivative of the functional f using finite
*  differences. */
//______________________________________________________________________________
template <typename F, typename AFloat>
inline AFloat finiteDifference(F f, AFloat dx)
{
    return f(dx) - f(0.0 - dx);
}

/*! Color code error. */
//______________________________________________________________________________
template <typename AFloat>
std::string print_error(AFloat &e)
{
    std::ostringstream out{};

    out << ("\e[");

    if (e > 1e-5)
        out << "31m";
    else if (e > 1e-9)
        out << "33m";
    else
        out << "32m";

    out << e;
    out << "\e[39m";

    return out.str();
}

}
}

#endif
