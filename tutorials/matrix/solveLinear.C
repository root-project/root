/// \file
/// \ingroup tutorial_matrix
/// \notebook -nodraw
/// This macro shows several ways to perform a linear least-squares
/// analysis . To keep things simple we fit a straight line to 4
/// data points
/// The first 4 methods use the linear algebra package to find
///  x  such that min \f$ (A x - b)^T (A x - b) \f$ where A and b
///  are calculated with the data points  and the functional expression :
///
///  1. Normal equations:
///   Expanding the expression \f$ (A x - b)^T (A x - b) \f$ and taking the
///   derivative wrt x leads to the "Normal Equations":
///   \f$ A^T A x = A^T b \f$ where \f$ A^T A \f$ is a positive definite matrix. Therefore,
///   a Cholesky decomposition scheme can be used to calculate its inverse .
///   This leads to the solution \f$ x = (A^T A)^-1 A^T b \f$ . All this is done in
///   routine NormalEqn . We made it a bit more complicated by giving the
///   data weights .
///   Numerically this is not the best way to proceed because effectively the
///   condition number of \f$ A^T A \f$ is twice as large as that of A, making inversion
///   more difficult
///
///  2. SVD
///   One can show that solving \f$ A x = b \f$ for x with A of size \f$ (m x n) \f$
///   and \f$ m > n \f$  through a Singular Value Decomposition is equivalent to minimizing
///   \f$ (A x - b)^T (A x - b) \f$ Numerically , this is the most stable method of all 5
///
///  3. Pseudo Inverse
///   Here we calculate the generalized matrix inverse ("pseudo inverse") by
///   solving \f$ A X = Unit \f$ for matrix \f$ X \f$ through an SVD . The formal expression for
///   is \f$ X = (A^T A)^-1 A^T \f$ . Then we multiply it by \f$ b \f$ .
///   Numerically, not as good as 2 and not as fast . In general it is not a
///   good idea to solve a set of linear equations with a matrix inversion .
///
///  4. Pseudo Inverse , brute force
///   The pseudo inverse is calculated brute force through a series of matrix
///   manipulations . It shows nicely some operations in the matrix package,
///   but is otherwise a big "no no" .
///
///  5. Least-squares analysis with Minuit
///   An objective function L is minimized by Minuit, where
///    \f$ L = sum_i { (y - c_0 -c_1 * x / e)^2 } \f$
///   Minuit will calculate numerically the derivative of L wrt c_0 and c_1 .
///   It has not been told that these derivatives are linear in the parameters
///   c_0 and c_1 .
///   For ill-conditioned linear problems it is better to use the fact it is
///   a linear fit as in 2 .
///
/// Another interesting thing is the way we assign data to the vectors and
/// matrices through adoption .
/// This allows data assignment without physically moving bytes around .
///
///  #### USAGE
///
/// This macro can be executed via CINT or via ACLIC
/// - via the interpretor, do
/// ~~~{.cpp}
///    root > .x solveLinear.C
/// ~~~
/// - via ACLIC
/// ~~~{.cpp}
///    root > gSystem->Load("libMatrix");
///    root > gSystem->Load("libGpad");
///    root > .x solveLinear.C+
/// ~~~
///
/// \macro_output
/// \macro_code
///
/// \author Eddy Offermann

#include "Riostream.h"
#include "TMatrixD.h"
#include "TVectorD.h"
#include "TGraphErrors.h"
#include "TDecompChol.h"
#include "TDecompSVD.h"
#include "TF1.h"


void solveLinear(Double_t eps = 1.e-12)
{
   cout << "Perform the fit  y = c0 + c1 * x in four different ways" << endl;

   const Int_t nrVar  = 2;
   const Int_t nrPnts = 4;

   Double_t ax[] = {0.0,1.0,2.0,3.0};
   Double_t ay[] = {1.4,1.5,3.7,4.1};
   Double_t ae[] = {0.5,0.2,1.0,0.5};

   // Make the vectors 'Use" the data : they are not copied, the vector data
   // pointer is just set appropriately

   TVectorD x; x.Use(nrPnts,ax);
   TVectorD y; y.Use(nrPnts,ay);
   TVectorD e; e.Use(nrPnts,ae);

   TMatrixD A(nrPnts,nrVar);
   TMatrixDColumn(A,0) = 1.0;
   TMatrixDColumn(A,1) = x;

   cout << " - 1. solve through Normal Equations" << endl;

   const TVectorD c_norm = NormalEqn(A,y,e);

   cout << " - 2. solve through SVD" << endl;
   // numerically  preferred method

   // first bring the weights in place
   TMatrixD Aw = A;
   TVectorD yw = y;
   for (Int_t irow = 0; irow < A.GetNrows(); irow++) {
      TMatrixDRow(Aw,irow) *= 1/e(irow);
      yw(irow) /= e(irow);
   }

   TDecompSVD svd(Aw);
   Bool_t ok;
   const TVectorD c_svd = svd.Solve(yw,ok);

   cout << " - 3. solve with pseudo inverse" << endl;

   const TMatrixD pseudo1  = svd.Invert();
   TVectorD c_pseudo1 = yw;
   c_pseudo1 *= pseudo1;

   cout << " - 4. solve with pseudo inverse, calculated brute force" << endl;

   TMatrixDSym AtA(TMatrixDSym::kAtA,Aw);
   const TMatrixD pseudo2 = AtA.Invert() * Aw.T();
   TVectorD c_pseudo2 = yw;
   c_pseudo2 *= pseudo2;

   cout << " - 5. Minuit through TGraph" << endl;

   TGraphErrors *gr = new TGraphErrors(nrPnts,ax,ay,0,ae);
   TF1 *f1 = new TF1("f1","pol1",0,5);
   gr->Fit("f1","Q");
   TVectorD c_graph(nrVar);
   c_graph(0) = f1->GetParameter(0);
   c_graph(1) = f1->GetParameter(1);

   // Check that all 4 answers are identical within a certain
   // tolerance . The 1e-12 is somewhat arbitrary . It turns out that
   // the TGraph fit is different by a few times 1e-13.

   Bool_t same = kTRUE;
   same &= VerifyVectorIdentity(c_norm,c_svd,0,eps);
   same &= VerifyVectorIdentity(c_norm,c_pseudo1,0,eps);
   same &= VerifyVectorIdentity(c_norm,c_pseudo2,0,eps);
   same &= VerifyVectorIdentity(c_norm,c_graph,0,eps);
   if (same)
      cout << " All solutions are the same within tolerance of " << eps << endl;
   else
      cout << " Some solutions differ more than the allowed tolerance of " << eps << endl;
}
