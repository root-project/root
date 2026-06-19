# \file
# \ingroup tutorial_matrix
# \notebook -nodraw
# This macro shows several ways to invert a matrix . Each  method
# is a trade-off between accuracy of the inversion and speed.
# Which method to chose depends on "how well-behaved" the matrix is.
# This is best checked through a call to Condition(), available in each
# decomposition class. A second possibility (less preferred) would be to
# check the determinant
#
#  #### USAGE
#
# This macro can be executed with python3 or ipython3
#  - via the bash, do
# ~~~{.py}
#    bash > python3 invertMatrix.py
# ~~~
#  - via ipython3
# ~~~{.py}
#    IP[1] > %run invertMatrix.py
# ~~~
#
# \macro_output
# \macro_code
#
# \author Eddy Offermann
# \translator P. P. 

import ROOT
import ctypes

from ROOT import iostream, TMath, TMatrixD, TMatrixDLazy, TVectorD, TDecompLU, TDecompSVD

THilbertMatrixD = ROOT.THilbertMatrixD
TMatrixDDiag = ROOT.TMatrixDDiag 
TDecompSVD = ROOT.TDecompSVD

Double_t = ROOT.Double_t 
c_double = ctypes.c_double

Power = TMath.Power

def invertMatrix(msize=6):

   if msize < 2 or msize > 10:
      print("msize should be inside : 2 <= msize <= 10" )
      return
      
   print(f"--------------------------------------------------------" )
   print(f"Inversion results for a (",msize,",",msize,") matrix" )
   print(f"For each inversion procedure we check the maximum size  " )
   print(f" of the off-diagonal elements of Inv(A)  A              " )
   print(f"--------------------------------------------------------" )
   
   H_square = THilbertMatrixD(msize,msize) #type(H_square) is TMatrixD
   
   # ### 1. InvertFast(det=0)
   # It is identical to Invert() for sizes > 6 x 6 but for smaller sizes, the
   # inversion is performed according to Cramer's rule by explicitly calculating
   # all Jacobi's sub-determinants . For instance for a 6 x 6 matrix this means:
   # \# of 5 x 5 determinant : 36
   # \# of 4 x 4 determinant : 75
   # \# of 3 x 3 determinant : 80
   # \# of 2 x 2 determinant : 45    (see TMatrixD/FCramerInv.cxx)
   #                                 (or see help(ROOT.TMatrixD)
   #
   # The only "quality" control in this process is to check whether the 6 x 6
   # determinant is unequal 0 . But speed gains are significant compared to Invert() ,
   # up to an order of magnitude for sizes <= 4 x 4
   #
   # The inversion is done "in place", so the original matrix will be overwritten
   # If a pointer to a Double_t is supplied the determinant is calculated
   #
   
   print(f"1. Use .InvertFast(&det)" )
   if msize > 6:
      print(f" for (", msize, ",", msize, ") this is identical to .Invert(&det)" )
   
   det1 = c_double()
   H1 = TMatrixD(H_square) # type(H1) is TMatrixD
   H1.InvertFast(det1)
   
   # Get the maximum off-diagonal matrix value . One way to do this is to set the
   # diagonal to zero .
   
   U1 = TMatrixD(H1, TMatrixD.kMult, H_square) # type(U1) is TMatrixD 
   diag1 = TMatrixDDiag(U1)
   #Note: This syntax functions only in C++. In Python, we use the .__assign__ method.
   #diag1 = 0.0 
   diag1.__assign__(0.0) 
   U1_max_offdiag = (U1.Abs()).Max()
   print( "  Maximum off-diagonal =", U1_max_offdiag )
   print( "  Determinant          =", det1.value )
   
   # ### 2. Invert(Double_t *det=0)
   # Again the inversion is performed in place .
   # It consists out of a sequence of calls to the decomposition classes . For instance
   # for the general dense matrix TMatrixD the LU decomposition is invoked:
   # - The matrix is decomposed using a scheme according to Crout which involves
   #   "implicit partial pivoting", see for instance Num. Recip. (we have also available
   #    a decomposition scheme that does not the scaling and is therefore even slightly
   #    faster but less stable)
   #    With each decomposition, a tolerance has to be specified . If this tolerance
   #    requirement is not met, the matrix is regarded as being singular. The value
   #    passed to this decomposition, is the data member fTol of the matrix . Its
   #    default value is DBL_EPSILON, which is defined as the smallest number so that
   #    1+DBL_EPSILON > 1
   # - The last step is a standard forward/backward substitution .
   #
   # It is important to realize that both InvertFast() and Invert() are "one-shot" deals , speed
   # comes at a price . If something goes wrong because the matrix is (near) singular, you have
   # overwritten your original matrix and  no factorization is available anymore to get more
   # information like condition number or change the tolerance number .
   #
   # All other calls in the matrix classes involving inversion like the ones with the "smart"
   # constructors (kInverted,kInvMult...) use this inversion method .
   #
   
   print(f"2. Use .Invert(&det)")
   
   det2 = c_double() 
   H2 = TMatrixD(H_square) # Converting to TMatrixD type for Invert method.
   H2.Invert(det2)
   
   U2 = TMatrixD(H2, TMatrixD.kMult, H_square) # type(U2) is TMatrixD 
   diag2 = TMatrixDDiag(U2)
   # diag2 = 0.0
   diag2.__assign__(0.0)
   U2_max_offdiag = (U2.Abs()).Max()
   print(f"  Maximum off-diagonal = {U2_max_offdiag}")
   print(f"  Determinant          = {det2.value}")
   
   # ### 3. Inversion through LU decomposition
   # The (default) algorithms used are similar to 2. (Not identical because in 2, the whole
   # calculation is done "in-place". Here the original matrix is copied (so more memory
   # management => slower) and several operations can be performed without having to repeat
   # the decomposition step .
   # Inverting a matrix is nothing else than solving a set of equations where the rhs is given
   # by the unit matrix, so the steps to take are identical to those solving a linear equation :
   #
   
   print(f"3. Use TDecompLU")
   
   H3 = TMatrixD(H_square) # Converting to TMatrixD type for method Invert.
   lu = TDecompLU(H_square) # type(lu) is TDecompLU 
   
   # Any operation that requires a decomposition will trigger it . The class keeps
   # an internal state so that following operations will not perform the decomposition again
   # unless the matrix is changed through SetMatrix(..)
   # One might want to proceed more cautiously by invoking first Decompose() and check its
   # return value before proceeding....
   
   lu.Invert(H3)
   d1_lu , d2_lu = [c_double() for i in range(2)] 
   lu.Det(d1_lu, d2_lu)
   det3 = d1_lu.value * Power(2., d2_lu.value)
   
   U3 = TMatrixD(H3,TMatrixD.kMult,H_square) # type(U3) is TMatrixD 
   diag3 = TMatrixDDiag(U3) # type(diag3) TMatrixDDiag 
   #diag3 = 0.0 # Initializes all its value to zero.
   diag3.__assign__(0.0)
   U3_max_offdiag = (U3.Abs()).Max()
   print(f"  Maximum off-diagonal = {U3_max_offdiag}")
   print(f"  Determinant          = {det3}")
   
   # ### 4. Inversion through SVD decomposition
   # For SVD and QRH, the (n x m) matrix does only have to fulfill n >=m . In case n > m
   # a pseudo-inverse is calculated
   print(f"4. Use TDecompSVD on non-square matrix")
   
   H_nsquare = THilbertMatrixD(msize, msize-1) # type(H_nsquare) TMatrixD 
   
   svd = TDecompSVD(H_nsquare) # type(svd) is TDecompSVD 
   
   H4 = svd.Invert()
   d1_svd , d2_svd = [c_double() for i in range(2)]
   svd.Det(d1_svd, d2_svd)
   det4 = d1_svd.value * Power(2., d2_svd.value)
   
   U4 = TMatrixD(H4, TMatrixD.kMult, H_nsquare) # type(U4) is TMatrixD 
   diag4 = TMatrixDDiag(U4)
   #diag4 = 0.0
   diag4.__assign__(0.0) 
   U4_max_offdiag = (U4.Abs()).Max()
   print(f"  Maximum off-diagonal = {U4_max_offdiag}")
   print(f"  Determinant          = {det4}")
   
if __name__ == "__main__":
  invertMatrix() 
