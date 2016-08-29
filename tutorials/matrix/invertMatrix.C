/// \file
/// \ingroup tutorial_matrix
/// \notebook -nodraw
/// This macro shows several ways to invert a matrix . Each  method
/// is a trade-off between accuracy of the inversion and speed.
/// Which method to chose depends on "how well-behaved" the matrix is.
/// This is best checked through a call to Condition(), available in each
/// decomposition class. A second possibility (less preferred) would be to
/// check the determinant
///
///  #### USAGE
///
/// This macro can be executed with Cling or ACLIC
///  - via the interpretor, do
/// ~~~{.cpp}
///    root > .x invertMatrix.C
/// ~~~
///  - via ACLIC
/// ~~~{.cpp}
///    root > gSystem->Load("libMatrix");
///    root > .x invertMatrix.C+
/// ~~~
///
/// \macro_output
/// \macro_code
///
/// \author Eddy Offermann

#include <iostream>
#include "TMath.h"
#include "TMatrixD.h"
#include "TMatrixDLazy.h"
#include "TVectorD.h"
#include "TDecompLU.h"
#include "TDecompSVD.h"

void invertMatrix(Int_t msize=6)
{
   if (msize < 2 || msize > 10) {
      std::cout << "2 <= msize <= 10" <<std::endl;
      return;
   }
   std::cout << "--------------------------------------------------------" <<std::endl;
   std::cout << "Inversion results for a ("<<msize<<","<<msize<<") matrix" <<std::endl;
   std::cout << "For each inversion procedure we check the maximum size  " <<std::endl;
   std::cout << "of the off-diagonal elements of Inv(A) * A              " <<std::endl;
   std::cout << "--------------------------------------------------------" <<std::endl;

   TMatrixD H_square = THilbertMatrixD(msize,msize);

   // ### 1. InvertFast(Double_t *det=0)
   // It is identical to Invert() for sizes > 6 x 6 but for smaller sizes, the
   // inversion is performed according to Cramer's rule by explicitly calculating
   // all Jacobi's sub-determinants . For instance for a 6 x 6 matrix this means:
   // \# of 5 x 5 determinant : 36
   // \# of 4 x 4 determinant : 75
   // \# of 3 x 3 determinant : 80
   // \# of 2 x 2 determinant : 45    (see TMatrixD/FCramerInv.cxx)
   //
   // The only "quality" control in this process is to check whether the 6 x 6
   // determinant is unequal 0 . But speed gains are significant compared to Invert() ,
   // up to an order of magnitude for sizes <= 4 x 4
   //
   // The inversion is done "in place", so the original matrix will be overwritten
   // If a pointer to a Double_t is supplied the determinant is calculated
   //

   std::cout << "1. Use .InvertFast(&det)" <<std::endl;
   if (msize > 6)
      std::cout << " for ("<<msize<<","<<msize<<") this is identical to .Invert(&det)" <<std::endl;

   Double_t det1;
   TMatrixD H1 = H_square;
   H1.InvertFast(&det1);

   // Get the maximum off-diagonal matrix value . One way to do this is to set the
   // diagonal to zero .

   TMatrixD U1(H1,TMatrixD::kMult,H_square);
   TMatrixDDiag diag1(U1); diag1 = 0.0;
   const Double_t U1_max_offdiag = (U1.Abs()).Max();
   std::cout << "  Maximum off-diagonal = " << U1_max_offdiag << std::endl;
   std::cout << "  Determinant          = " << det1 << std::endl;

   // ### 2. Invert(Double_t *det=0)
   // Again the inversion is performed in place .
   // It consists out of a sequence of calls to the decomposition classes . For instance
   // for the general dense matrix TMatrixD the LU decomposition is invoked:
   // - The matrix is decomposed using a scheme according to Crout which involves
   //   "implicit partial pivoting", see for instance Num. Recip. (we have also available
   //    a decomposition scheme that does not the scaling and is therefore even slightly
   //    faster but less stable)
   //    With each decomposition, a tolerance has to be specified . If this tolerance
   //    requirement is not met, the matrix is regarded as being singular. The value
   //    passed to this decomposition, is the data member fTol of the matrix . Its
   //    default value is DBL_EPSILON, which is defined as the smallest number so that
   //    1+DBL_EPSILON > 1
   // - The last step is a standard forward/backward substitution .
   //
   // It is important to realize that both InvertFast() and Invert() are "one-shot" deals , speed
   // comes at a price . If something goes wrong because the matrix is (near) singular, you have
   // overwritten your original matrix and  no factorization is available anymore to get more
   // information like condition number or change the tolerance number .
   //
   // All other calls in the matrix classes involving inversion like the ones with the "smart"
   // constructors (kInverted,kInvMult...) use this inversion method .
   //

   std::cout << "2. Use .Invert(&det)" << std::endl;

   Double_t det2;
   TMatrixD H2 = H_square;
   H2.Invert(&det2);

   TMatrixD U2(H2,TMatrixD::kMult,H_square);
   TMatrixDDiag diag2(U2); diag2 = 0.0;
   const Double_t U2_max_offdiag = (U2.Abs()).Max();
   std::cout << "  Maximum off-diagonal = " << U2_max_offdiag << std::endl;
   std::cout << "  Determinant          = " << det2 << std::endl;

   // ### 3. Inversion through LU decomposition
   // The (default) algorithms used are similar to 2. (Not identical because in 2, the whole
   // calculation is done "in-place". Here the original matrix is copied (so more memory
   // management => slower) and several operations can be performed without having to repeat
   // the decomposition step .
   // Inverting a matrix is nothing else than solving a set of equations where the rhs is given
   // by the unit matrix, so the steps to take are identical to those solving a linear equation :
   //

   std::cout << "3. Use TDecompLU" << std::endl;

   TMatrixD H3 = H_square;
   TDecompLU lu(H_square);

   // Any operation that requires a decomposition will trigger it . The class keeps
   // an internal state so that following operations will not perform the decomposition again
   // unless the matrix is changed through SetMatrix(..)
   // One might want to proceed more cautiously by invoking first Decompose() and check its
   // return value before proceeding....

   lu.Invert(H3);
   Double_t d1_lu; Double_t d2_lu;
   lu.Det(d1_lu,d2_lu);
   Double_t det3 = d1_lu*TMath::Power(2.,d2_lu);

   TMatrixD U3(H3,TMatrixD::kMult,H_square);
   TMatrixDDiag diag3(U3); diag3 = 0.0;
   const Double_t U3_max_offdiag = (U3.Abs()).Max();
   std::cout << "  Maximum off-diagonal = " << U3_max_offdiag << std::endl;
   std::cout << "  Determinant          = " << det3 << std::endl;

   // ### 4. Inversion through SVD decomposition
   // For SVD and QRH, the (n x m) matrix does only have to fulfill n >=m . In case n > m
   // a pseudo-inverse is calculated
   std::cout << "4. Use TDecompSVD on non-square matrix" << std::endl;

   TMatrixD H_nsquare = THilbertMatrixD(msize,msize-1);

   TDecompSVD svd(H_nsquare);

   TMatrixD H4 = svd.Invert();
   Double_t d1_svd; Double_t d2_svd;
   svd.Det(d1_svd,d2_svd);
   Double_t det4 = d1_svd*TMath::Power(2.,d2_svd);

   TMatrixD U4(H4,TMatrixD::kMult,H_nsquare);
   TMatrixDDiag diag4(U4); diag4 = 0.0;
   const Double_t U4_max_offdiag = (U4.Abs()).Max();
   std::cout << "  Maximum off-diagonal = " << U4_max_offdiag << std::endl;
   std::cout << "  Determinant          = " << det4 << std::endl;
}
