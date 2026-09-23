# \file
# \ingroup tutorial_matrix
# \notebook -nodraw
# This macro shows several ways to perform a linear least-squares
# analysis . To keep things simple we fit a straight line to 4
# data points
# The first 4 methods use the linear algebra package to find
#  x  such that min \f$ (A x - b)^T (A x - b) \f$ where A and b
#  are calculated with the data points  and the functional expression :
#
#  1. Normal equations:
#   Expanding the expression \f$ (A x - b)^T (A x - b) \f$ and taking the
#   derivative wrt x leads to the "Normal Equations":
#   \f$ A^T A x = A^T b \f$ where \f$ A^T A \f$ is a positive definite matrix. Therefore,
#   a Cholesky decomposition scheme can be used to calculate its inverse .
#   This leads to the solution \f$ x = (A^T A)^-1 A^T b \f$ . All this is done in
#   routine NormalEqn . We made it a bit more complicated by giving the
#   data weights .
#   Numerically this is not the best way to proceed because effectively the
#   condition number of \f$ A^T A \f$ is twice as large as that of A, making inversion
#   more difficult
#
#  2. SVD
#   One can show that ssolvingolving \f$ A x = b \f$ for x with A of size \f$ (m x n) \f$
#   and \f$ m > n \f$  through a Singular Value Decomposition is equivalent to minimizing
#   \f$ (A x - b)^T (A x - b) \f$ Numerically , this is the most stable method of all 5
#
#  3. Pseudo Inverse
#   Here we calculate the generalized matrix inverse ("pseudo inverse") by
#   solving \f$ A X = Unit \f$ for matrix \f$ X \f$ through an SVD . The formal expression for
#   is \f$ X = (A^T A)^-1 A^T \f$ . Then we multiply it by \f$ b \f$ .
#   Numerically, not as good as 2 and not as fast . In general it is not a
#   good idea to solve a set of linear equations with a matrix inversion .
#
#  4. Pseudo Inverse , brute force
#   The pseudo inverse is calculated brute force through a series of matrix
#   manipulations . It shows nicely some operations in the matrix package,
#   but is otherwise a big "no no" .
#
#  5. Least-squares analysis with Minuit
#   An objective function L is minimized by Minuit, where
#    \f$ L = sum_i { (y - c_0 -c_1 * x / e)^2 } \f$
#   Minuit will calculate numerically the derivative of L wrt c_0 and c_1 .
#   It has not been told that these derivatives are linear in the parameters
#   c_0 and c_1 .
#   For ill-conditioned linear problems it is better to use the fact it is
#   a linear fit as in 2 .
#
# Another interesting thing is the way we assign data to the vectors and
# matrices through adoption .
# This allows data assignment without physically moving bytes around .
#
#  #### USAGE
#
# This macro can be executed via bash with python3 or via interpreter ipython3:
# - via the bash interpreter, do
# ~~~{.py}
#    bash > ipython3 solveLinear.py
# ~~~
# - or via ipython3 interpreter, do
# ~~~{.py}
#    IP[1]: %run solveLinear.py
# ~~~
#
# \macro_output
# \macro_code
#
# \author Eddy Offermann
# \translator P. P.


import ROOT
import ctypes

from ROOT import TMatrixD, TVectorD, TGraphErrors, TDecompChol, TDecompSVD, TF1

c_bool = ctypes.c_bool
c_double = ctypes.c_double

TMatrixDColumn = ROOT.TMatrixDColumn
NormalEqn = ROOT.NormalEqn
TMatrixDRow = ROOT.TMatrixDRow 
TMatrixDSym = ROOT.TMatrixDSym

kTRUE = ROOT.kTRUE  

VerifyVectorIdentity = ROOT.VerifyVectorIdentity

def solveLinear(eps = 1.e-12):

   print( "Perform the fit  y = c0 + c1 * x in four different ways") 
   
   nrVar = 2
   nrPnts = 4
   
   ax = [0.0,1.0,2.0,3.0]
   ay = [1.4,1.5,3.7,4.1]
   ae = [0.5,0.2,1.0,0.5]
   
   c_ax = (c_double*nrPnts)(*ax)
   c_ay = (c_double*nrPnts)(*ay)
   c_ae = (c_double*nrPnts)(*ae)
   # Make the vectors 'Use" the data : they are not copied, the vector data
   # pointer is just set appropriately
   
   x = TVectorD() 
   x.Use(nrPnts, c_ax)
   y = TVectorD() 
   y.Use(nrPnts, c_ay)
   e = TVectorD() 
   e.Use(nrPnts, c_ae)
   
   A = TMatrixD(nrPnts, nrVar)
   #Not to use: TMatrixDColumn(A,0) = 1.0
   TMatrixDColumn(A,0).__assign__( 1.0)
   #Not to use:TMatrixDColumn(A,1) = x
   TMatrixDColumn(A,1).__assign__( x )
   
   print(f" - 1. solve through Normal Equations")
   
   c_norm = NormalEqn(A,y,e) # TVectorD 
   
   print(f" - 2. solve through SVD")
   # numerically  preferred method
   
   # first bring the weights in place
   # Initializaing Aw and yw ...
   Aw = TMatrixD(A) #TMatrixD 
   yw = TVectorD(y) #TVectorD
   
   for irow in range( A.GetNrows()):
      # Note: whichever of the following notations is equivalent to:
      #       C++ :TMatrixDRow(Aw,irow) *= 1/e(irow);
      TMatrixDRow(Aw,irow).__imul__( 1/e(irow))
      #TMatrixDRow(Aw,irow).__imul__( 1/e[irow])
      #       C++ :yw( irow) /= e(irow)
      yw[irow] /= e(irow)
   
   svd = TDecompSVD(Aw)
   ok = c_bool()
   c_svd = svd.Solve(yw, ok)
   
   print(f" - 3. solve with pseudo inverse")
   
   pseudo1 = svd.Invert()
   c_pseudo1 = TVectorD(yw) #
   c_pseudo1 *= pseudo1 #
   
   print(f" - 4. solve with pseudo inverse, calculated brute force")
   AtA = TMatrixDSym(TMatrixDSym.kAtA, Aw)
   AtA_inv = TMatrixD(AtA).Invert()
   # Note: 1th. we initialize 2th. We transpose 3th. We save
   #       Otherwise, if we use simply Aw.T(), the elements of Aw transpose itself.
   #       Be cautios about this behaviour.
   #       Also, the operator '=' is overwrited; causing a reference to same 
   #       TMatrix-Object variable, as in C++, different to copying-a-reference as 
   #       in Python. So, initialize your variable first and then use it.
   #Initialize Transposed Matrix n,m (notice that it is not m,n) 
   Aw_trans = TMatrixD(Aw.GetNcols(), Aw.GetNrows()) 
   Aw_trans.Transpose(Aw)  #Saving the transposed elements into. Aw is untouched.
   # Be Cautios with a one-line-syntax like :
   # Aw_trans = TMatrixD(Aw.T())
   m = AtA_inv.GetNrows() # Rows 
   n = Aw_trans.GetNcols() # Columns
   # Remeber A[m, l]*B[l,n] = C[m,n]
   # If you want to see how it works ... 
   # print(m,n)
   # jAtA_inv.Print()
   # Aw_trans.Print()
   pseudo2 = TMatrixD(m, n)
   # Note: A*B matrix multiplication is not implemented as a python operation yet.
   #       We will instead use the .Mult method. C.Mult(A, B) is C=A*B.
   # Not to use: pseudo2 = AtA.Invert() * Aw.T() # TMatrixD 
   # Not to use: pseudo2 = AtA_inv * Aw_trans # AtA_inv * Aw.T() # TMatrixD
   pseudo2.Mult(AtA_inv, Aw_trans)
   c_pseudo2 = TVectorD(yw) #
   c_pseudo2 *= pseudo2 # review
   
   print(f" - 5. Minuit through TGraph")
   
   #Not to use: gr =  TGraphErrors(nrPnts,ax,ay,0,ae)
   gr =  TGraphErrors(nrPnts, c_ax, c_ay, 0, c_ae)
   f1 =  TF1("f1","pol1",0,5)
   gr.Fit("f1","Q")
   c_graph = TVectorD(nrVar)
   c_graph[0] = f1.GetParameter(0)
   c_graph[1] = f1.GetParameter(1)
   
   # Check that all 4 answers are identical within a certain
   # tolerance . The 1e-12 is somewhat arbitrary . It turns out that
   # the TGraph fit is different by a few times 1e-13.
   
   same = kTRUE
   same &= VerifyVectorIdentity(c_norm,c_svd,0,eps)
   same &= VerifyVectorIdentity(c_norm,c_pseudo1,0,eps)
   same &= VerifyVectorIdentity(c_norm,c_pseudo2,0,eps)
   same &= VerifyVectorIdentity(c_norm,c_graph,0,eps)
   if same:
      print(f" All solutions are the same within tolerance of ", eps)
   else:
      print(f" Some solutions differ more than the allowed tolerance of ", eps)
   
if __name__ == "__main__":
   solveLinear()
