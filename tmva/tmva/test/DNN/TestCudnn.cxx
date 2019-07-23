#include "Utility.h"
#include "TMVA/DNN/Architectures/Cudnn.h"
#include "TMVA/DNN/Architectures/Reference.h"
#include <stdlib.h>

using namespace TMVA::DNN;

//_________________________________________________________________________________
Double_t testMultiply()
{
    const size_t ntests = 100;

    Double_t maximumError = 0;

    for (size_t i = 0; i < ntests; i++) {
        size_t m, k;
        m = rand() % 50 + 1;
        k = rand() % 50 + 1;

        TMatrixT<Double_t> A(m,k), B(m,k), C(m,k);
        randomMatrix(A);
        randomMatrix(B);
        randomMatrix(C);
        std::vector<size_t> shapeA {1,1,m,k};
        std::vector<TMatrixT<Double_t> > matrixVecA {A};
        std::vector<size_t> shapeB {1,1,m,k};
        std::vector<TMatrixT<Double_t> > matrixVecB {B};
        std::vector<size_t> shapeC {1,1,m,k};
        std::vector<TMatrixT<Double_t> > matrixVecC {C};
        TCudaTensor<Double_t> ACudnn(matrixVecA, 4, shapeA, 0, 3),
                              BCudnn(matrixVecB, 4, shapeB, 0, 3), 
                              CCudnn(matrixVecC, 4, shapeC, 0, 3);

        TCudnn<Double_t>::Multiply(CCudnn, ACudnn, BCudnn);
    }
    return maximumError;
}

//_________________________________________________________________________________
/*Double_t testAddRowWise()
{
   const size_t ntests = 10;

   Double_t maximumError = 0;

   for (size_t i = 0; i < ntests; i++) {
      size_t m, n;
      m = rand() % 50 + 1;
      n = rand() % 50 + 1;

      TMatrixT<Double_t> A(m,n), B(m,n), theta(n,1);
      //randomMatrix(A);
      randomMatrix(theta);
      TCudaMatrix ACuda(A), BCuda(B), thetaCuda(theta);

      TReference<Double_t>::AddRowWise(A, theta);
      TCuda<false>::AddRowWise(ACuda,thetaCuda);
      TMatrixT<Double_t> ARef(ACuda);

      Double_t error = maximumRelativeError(A, ARef);
      maximumError   = std::max(error, maximumError);
   }
   return maximumError;
}*/

//_________________________________________________________________________________
/*Double_t testHadamard()
{
   const size_t ntests = 10;
   Double_t maximumError = 0;

   for (size_t i = 0; i < ntests; i++) {
      size_t m, n;
      m = rand() % 10 + 1;
      n = rand() % 10 + 1;

      TMatrixT<Double_t> A(m,n), B(m,n);
      randomMatrix(A);
      randomMatrix(B);
      TCudaMatrix ACuda(A), BCuda(B);

      for (size_t j = 0; j < (size_t) A.GetNrows(); j++) {
         for (size_t k = 0; k < (size_t) A.GetNcols(); k++) {
             A(j,k) *= B(j,k);
         }
      }

      TCuda<false>::Hadamard(ACuda, BCuda);
      TMatrixT<Double_t> ARef(ACuda);
      Double_t error = maximumRelativeError(A, ARef);
      maximumError   = std::max(error, maximumError);
   }
   return maximumError;
}*/

//_________________________________________________________________________________
Double_t testReduction()
{
   const size_t ntests = 1;
   Double_t maximumError = 0;

   for (size_t i = 0; i < ntests; i++) {
      size_t m, n;
      m = rand() % 50 + 1;
      n = rand() % 50 + 1;

      TMatrixT<Double_t> A(m,n);

      for (size_t j = 0; j < m; j++) {
         for (size_t k = 0; k < n; k++) {
            A(j,k) = 11.0;
         }
      }
      std::vector<size_t> shapeA {1,1,m,n};
      std::vector<TMatrixT<Double_t> > matrixVecA {A};
      TCudaTensor<Double_t> ACudnn(matrixVecA, 4, shapeA);

      //std::vector<size_t> shapeB {1,1,1,n};
      //TCudaTensor<Double_t> BCudnn(1,n);
      //TCuda<Double_t>::InitializeZero(BCudnn);
      Double_t s  = TCudnn<Double_t>::Sum(ACudnn);
      //TMatrixT<Double_t> B(BCuda);
      Double_t error = s - ((Double_t) m * n);
      maximumError   = std::max(fabs(error), maximumError);

      /*for (size_t j = 0; j < n; j++) {
         //std::cout << B(0,j) << " / " << j * m << std::endl;
         error = std::abs(B(0,j) - m);
         maximumError   = std::max(error, maximumError);
      }*/
   }
   return maximumError;
}

//_________________________________________________________________________________
/*Double_t testScaleAdd()
{
   const size_t ntests   = 10;
   Double_t maximumError = 0;

   for (size_t i = 0; i < ntests; i++) {
      size_t m, n;
      m = rand() % 1000 + 1;
      n = rand() % 1000 + 1;

      TMatrixT<Double_t> A(m,n), B(m,n);

      randomMatrix(A);
      randomMatrix(B);

      TCudaMatrix ACuda(A);
      TCudaMatrix BCuda(B);

      Double_t beta = ((Double_t) rand()) / ((Double_t) RAND_MAX);
      TReference<Double_t>::ScaleAdd(A, B, beta);
      TCuda<false>::ScaleAdd(ACuda, BCuda, beta);

      Double_t error = maximumRelativeError(A, (TMatrixT<Double_t>) ACuda);
      maximumError   = std::max(error, maximumError);
   }
   return maximumError;
}*/

//_________________________________________________________________________________
int main()
{
    Double_t error;
    error = testReduction();
    std::cout << "Testing reduction: max. rel. error = ";
    std::cout << error << std::endl;

    /*error = testScaleAdd();
    std::cout << "Testing scale_add: max. rel. error = ";
    std::cout << error << std::endl;*/

    /*error = testHadamard();
    std::cout << "Testing hadamard: max. rel. error = ";
    std::cout << error << std::endl;*/

    /*error = testMultiply();
    std::cout << "Testing multiplication: max. rel. error = ";
    std::cout << error << std::endl;*/

    /*error = testAddRowWise();
    std::cout << "Testing add_row_wise: max. rel. error = ";
    std::cout << error << std::endl;*/
}
