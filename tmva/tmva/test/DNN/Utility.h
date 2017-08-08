#ifndef TMVA_TEST_DNN_UTILITY
#define TMVA_TEST_DNN_UTILITY

#include <cassert>
#include <iostream>
#include <sstream>
#include <type_traits>
#include "stdlib.h"
#include "TRandom.h"
#include "TMVA/DNN/Architectures/Reference.h"
#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/Net.h"
#include "TMVA/DNN/DeepNet.h"

namespace TMVA {
namespace DNN {

/** Construct a convolutional neural network with one convolutional layer,
 *  one pooling layer and two fully connected layers. The dimensions are
 *  predetermined. The activation functions are chosen randomly. */
//______________________________________________________________________________
template <typename AArchitecture>
void constructConvNet(TDeepNet<AArchitecture> &net)
{
   /* For random selection */
   std::vector<EActivationFunction> ActivationFunctions = {EActivationFunction::kIdentity, EActivationFunction::kRelu,
                                                           EActivationFunction::kSigmoid, EActivationFunction::kTanh};

   size_t depth = 12;
   size_t filterHeightConv = 2;
   size_t filterWidthConv = 2;
   size_t strideRowsConv = 1;
   size_t strideColsConv = 1;
   size_t zeroPaddingHeight = 1;
   size_t zeroPaddingWidth = 1;

   EActivationFunction fConv = ActivationFunctions[rand() % ActivationFunctions.size()];

   net.AddConvLayer(depth, filterHeightConv, filterWidthConv, strideRowsConv, strideColsConv, zeroPaddingHeight,
                    zeroPaddingWidth, fConv);

   size_t filterHeightPool = 6;
   size_t filterWidthPool = 6;
   size_t strideRowsPool = 1;
   size_t strideColsPool = 1;

   net.AddMaxPoolLayer(filterHeightPool, filterWidthPool, strideRowsPool, strideColsPool);

   size_t depthReshape = 1;
   size_t heightReshape = 1;
   size_t widthReshape = net.GetLayerAt(net.GetDepth() - 1)->GetDepth() *
                         net.GetLayerAt(net.GetDepth() - 1)->GetHeight() *
                         net.GetLayerAt(net.GetDepth() - 1)->GetWidth();

   net.AddReshapeLayer(depthReshape, heightReshape, widthReshape);

   size_t widthFC1 = 20;
   EActivationFunction fFC1 = ActivationFunctions[rand() % ActivationFunctions.size()];
   net.AddDenseLayer(widthFC1, fFC1);

   size_t widthFC2 = 5;
   EActivationFunction fFC2 = EActivationFunction::kIdentity;
   net.AddDenseLayer(widthFC2, fFC2);
}

/** Construct a linear convolutional neural network with one convolutional layer,
 *  one pooling layer and two fully connected layers. The dimensions are
 *  predetermined. The activation functions are all linear.  */
//______________________________________________________________________________
template <typename AArchitecture>
void constructLinearConvNet(TDeepNet<AArchitecture> &net)
{
   size_t depth = 12;
   size_t filterHeightConv = 2;
   size_t filterWidthConv = 2;
   size_t strideRowsConv = 1;
   size_t strideColsConv = 1;
   size_t zeroPaddingHeight = 1;
   size_t zeroPaddingWidth = 1;

   EActivationFunction fConv = EActivationFunction::kIdentity;

   net.AddConvLayer(depth, filterHeightConv, filterWidthConv, strideRowsConv, strideColsConv, zeroPaddingHeight,
                    zeroPaddingWidth, fConv);

   size_t filterHeightPool = 6;
   size_t filterWidthPool = 6;
   size_t strideRowsPool = 1;
   size_t strideColsPool = 1;

   net.AddMaxPoolLayer(filterHeightPool, filterWidthPool, strideRowsPool, strideColsPool);

   size_t depthReshape = 1;
   size_t heightReshape = 1;
   size_t widthReshape = net.GetLayerAt(net.GetDepth() - 1)->GetDepth() *
                         net.GetLayerAt(net.GetDepth() - 1)->GetHeight() *
                         net.GetLayerAt(net.GetDepth() - 1)->GetWidth();

   net.AddReshapeLayer(depthReshape, heightReshape, widthReshape);

   size_t widthFC1 = 20;
   EActivationFunction fFC1 = EActivationFunction::kIdentity;
   net.AddDenseLayer(widthFC1, fFC1);

   size_t widthFC2 = 5;
   EActivationFunction fFC2 = EActivationFunction::kIdentity;
   net.AddDenseLayer(widthFC2, fFC2);
}

/** Construct a random linear neural network with up to five layers.*/
//______________________________________________________________________________
template <typename AArchitecture>
void constructRandomLinearNet(TNet<AArchitecture> &net)
{
   int nlayers = rand() % 5 + 1;

   std::vector<EActivationFunction> ActivationFunctions = {EActivationFunction::kIdentity};

   for (int i = 0; i < nlayers; i++) {
      int width = rand() % 20 + 1;
      EActivationFunction f = ActivationFunctions[rand() % ActivationFunctions.size()];
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
         X(i, j) = 0.0;
      }
      if (i < n) {
         X(i, i) = 1.0;
      }
   }
}

/*! Fill matrix with given value.*/
//______________________________________________________________________________
template <typename AMatrix, typename AReal>
void fillMatrix(AMatrix &X, AReal x)
{
   size_t m, n;
   m = X.GetNrows();
   n = X.GetNcols();

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         X(i, j) = x;
      }
   }
}

/*! Fill matrix with random, Gaussian-distributed values. */
//______________________________________________________________________________
template <typename AMatrix>
void randomMatrix(AMatrix &X)
{
   size_t m, n;
   m = X.GetNrows();
   n = X.GetNcols();

   TRandom rand(clock());

   Double_t sigma = sqrt(10.0);

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         X(i, j) = rand.Gaus(0.0, sigma);
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
   size_t m, n;
   m = X.GetNrows();
   n = X.GetNcols();

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         X(i, j) = Y(i, j);
      }
   }
}

/*! Apply functional to each element in the matrix. */
//______________________________________________________________________________
template <typename AMatrix, typename F>
void applyMatrix(AMatrix &X, F f)
{
   size_t m, n;
   m = X.GetNrows();
   n = X.GetNcols();

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         X(i, j) = f(X(i, j));
      }
   }
}

/*! Combine elements of two given matrices into a single matrix using
 *  the given function f. */
//______________________________________________________________________________
template <typename AMatrix, typename F>
void zipWithMatrix(AMatrix &Z, F f, const AMatrix &X, const AMatrix &Y)
{
   size_t m, n;
   m = X.GetNrows();
   n = X.GetNcols();

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         Z(i, j) = f(X(i, j), Y(i, j));
      }
   }
}

/** Generate a random batch as input for a neural net. */
//______________________________________________________________________________
template <typename AMatrix, typename AFloat, typename F>
AFloat reduce(F f, AFloat start, const AMatrix &X)
{
   size_t m, n;
   m = X.GetNrows();
   n = X.GetNcols();

   AFloat result = start;

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         result = f(result, X(i, j));
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
   size_t m, n;
   m = X.GetNrows();
   n = X.GetNcols();

   AFloat result = start;

   for (size_t i = 0; i < m; i++) {
      for (size_t j = 0; j < n; j++) {
         result = f(result, X(i, j));
      }
   }
   return result / (AFloat)(m * n);
}

/** Compute the relative error of x and y */
//______________________________________________________________________________
template <typename T>
inline T relativeError(const T &x, const T &y)
{
   using std::abs;

   if (x == y) return T(0.0);

   T diff = abs(x - y);

   if (x * y == T(0.0) || diff < std::numeric_limits<T>::epsilon()) return diff;

   return diff / (abs(x) + abs(y));
}

/*! Compute the maximum, element-wise relative error of the matrices
*  X and Y normalized by the element of Y. Protected against division
*  by zero. */
//______________________________________________________________________________
template <typename Matrix1, typename Matrix2>
auto maximumRelativeError(const Matrix1 &X, const Matrix2 &Y) -> Double_t
{
   Double_t curError, maxError = 0.0;

   Int_t m = X.GetNrows();
   Int_t n = X.GetNcols();

   assert(m == Y.GetNrows());
   assert(n == Y.GetNcols());

   for (Int_t i = 0; i < m; i++) {
      for (Int_t j = 0; j < n; j++) {
         curError = relativeError<Double_t>(X(i, j), Y(i, j));
         maxError = std::max(curError, maxError);
      }
   }

   return maxError;
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
