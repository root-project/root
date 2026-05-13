# \file
# \ingroup tutorial_matrix
# \notebook -nodraw
# This tutorial shows how to decompose a matrix A in an orthogonal matrix Q and an upper
# triangular matrix R using QR Householder decomposition with the TDecompQRH class.
# We use the same matrix as in this example: <https:#en.wikipedia.org/wiki/QR_decomposition#Example_2>


import ROOT
import ctypes

TMath = 		 ROOT.TMath
TDecompQRH = 		 ROOT.TDecompQRH

c_double = ctypes.c_double
TMatrixT = ROOT.TMatrixT
TMatrixD = ROOT.TMatrixD
double = ROOT.double
AreEqualAbs = TMath.AreEqualAbs
Error = ROOT.Error


def decomposeQR():
   
   n = 3
   
   a = [12, -51, 4, 6, 167, -68, -4, 24, -41]
   c_a = (c_double * len(a))(*a)
   
   A = TMatrixT(double)(3, 3, c_a)
   
   print("initial matrix A ")
   
   A.Print()
   
   decomp = TDecompQRH(A)
   
   ret = decomp.Decompose() # type(ret) is bool
   
   print("Orthogonal Q matrix ")
   
   # note that decomp.GetQ()  returns an internal matrix which is not Q defined as A = QR
   Q = decomp.GetOrthogonalMatrix()
   Q.Print()
   
   print("Upper Triangular R matrix ")
   R = decomp.GetR()
   
   R.Print()
   
   # check that we have a correct Q-R decomposition
   
   gQ = Q
   gR = R
   # Note: Matrix Multiplication operation * is not implemented yet.
   # Not to use : Q * R 
   # compA = Q * R # type(comA) is TMatrixT(double) 
   # Instead use the Mult-method as C.Mult(A, B) 
   m_row = Q.GetNrows()
   n_col = R.GetNcols()
   #compA = TMatrixT(double)(m_row, n_col) #Equivalent to:
   compA = TMatrixD(m_row, n_col)
   # compA = TMatrix(3,3) #only works in this particular case
   compA.Mult(Q, R)
   # compA.Print() # m_row, n_col 

   
   print("Computed A matrix from Q * R ")
   compA.Print()
   
   for i in range( A.GetNrows() ):
      for j in range( A.GetNcols() ):
         if not TMath.AreEqualAbs( compA(i,j), A(i,j), 1.E-6) :
            print("Tolerance Error:")
            print(f"at position entry (i,j): ({i}, {j}) ...")
            Error("decomposeQR",
            """Remonstrate(decomposed) matrix is not equal to the original :
             {:f} different than {:f}""".format(compA(i,j), A(i,j)) )
         
      
   
   # check also that Q is orthogonal (Q^T * Q = I)
   # Not to use: QT = Q # Transposed Q is QT. 
   # SpecialCase: QT = TMatrixD(3,3)
   QT = TMatrixD(Q.GetNrows(),Q.GetNcols()) # Initialize QT Matrix 
   QT.Transpose(Q)
   ## Debugging
   #print("Q")
   #Q.Print() 
   #print("QT")
   #QT.Print()
   #Not to use: qtq = QT * Q
   #qtq = TMatrixD(QT.GetNrows(), Q.GetNcols())
   qtq = TMatrixD(3,3) # Initialize qtq Matrix
   qtq.Mult(QT, Q)
   print("Analyzing... QTQ Matrix:")     
   is_looping = True
   for i in range(Q.GetNrows()):
      for j in range(Q.GetNcols()):
         if (i == j and not AreEqualAbs(qtq(i, i), 1., 1.E-6)) or \
            (i != j and not AreEqualAbs(qtq(i, j), 0., 1.E-6)) :
            print(f"At position entry (i,j): ({i}, {j}) ...")
            Error("decomposeQR", "Q matrix is not orthogonal ")
            #qtq.Print()
            is_looping = False
            break
      if is_looping == False: 
         break

   if is_looping == False:
      qtq.Print()            
   else :
      print("Good. Q matrix is orthogonal with error less than 1.E-6")  
         
if __name__ == "__main__":
   decomposeQR()      
