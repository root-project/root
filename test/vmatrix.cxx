// @(#)root/test:$Name:  $:$Id: vmatrix.cxx,v 1.6 2001/05/09 18:01:49 rdm Exp $
// Author: Fons Rademakers   14/11/97

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Linear Algebra Package -- Matrix Verifications.                      //
//                                                                      //
// This file implements a large set of TMatrix operation tests.         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <iostream.h>
#include <math.h>
#include <float.h>

#include "TApplication.h"
#include "TFile.h"
#include "TMatrix.h"


//
//------------------------------------------------------------------------
//            Service validation functions
//
void verify_matrix_identity(const TMatrix &m1, const TMatrix &m2)
{ VerifyMatrixIdentity(m1,m2); }

void verify_vector_identity(const TVector &v1, const TVector &v2)
{ VerifyVectorIdentity(v1,v2); }

void verify_element_value(const TMatrix &m, Real_t val)
{ VerifyElementValue(m, val); }

void are_compatible(const TMatrix &m1, const TMatrix &m2)
{
   if (AreCompatible(m1, m2))
      cout << "matrices are compatible" << endl;
   else
      cout << "matrices are NOT compatible" << endl;
}

//
//------------------------------------------------------------------------
//          Test allocation functions and compatibility check
//
void test_allocation()
{

   cout << "\n\n---> Test allocation and compatibility check" << endl;

   TMatrix m1(4,20);
   TMatrix m2(0,3,0,19);
   TMatrix m3(1,4,0,19);
   TMatrix m4(m1);

   cout << "\nStatus information reported for matrix m3:" << endl;
   cout << "  Row lower bound ... " << m3.GetRowLwb() << endl;
   cout << "  Row upper bound ... " << m3.GetRowUpb() << endl;
   cout << "  Col lower bound ... " << m3.GetColLwb() << endl;
   cout << "  Col upper bound ... " << m3.GetColUpb() << endl;
   cout << "  No. rows ..........." << m3.GetNrows()  << endl;
   cout << "  No. cols ..........." << m3.GetNcols()  << endl;
   cout << "  No. of elements ...." << m3.GetNoElements() << endl;

   cout << "\nCheck matrices 1 & 2 for compatibility" << endl;
   are_compatible(m1,m2);

   cout << "Check matrices 1 & 4 for compatibility" << endl;
   are_compatible(m1,m4);

   cout << "m2 has to be compatible with m3 after resizing to m3" << endl;
   m2.ResizeTo(m3);
   are_compatible(m2,m3);

   TMatrix m5(m1.GetNrows()+1,m1.GetNcols()+5);
   cout << "m1 has to be compatible with m5 after resizing to m5" << endl;
   m1.ResizeTo(m5.GetNrows(),m5.GetNcols());
   are_compatible(m1,m5);

   cout << "\nDone\n" << endl;
}

//
//------------------------------------------------------------------------
//                Test uniform element operations
//
class FillMatrix : public TElementPosAction {
   int no_elems, no_cols;
   void Operation(Real_t &element)
      { element = 4*TMath::Pi()/no_elems * (fI*no_cols+fJ); }
public:
   FillMatrix(const TMatrix &m) :
         no_elems(m.GetNoElements()), no_cols(m.GetNcols()) { }
};

typedef  double (*dfunc)(double);
class ApplyFunction : public TElementAction {
   dfunc fFunc;
   void Operation(Real_t &element) { element = fFunc(double(element)); }
public:
   ApplyFunction(dfunc func) : fFunc(func) { }
};

void test_element_op(int rsize, int csize)
{
   const double pattern = 8.625;
   int i,j;

   cout << "\n---> Test operations that treat each element uniformly" << endl;

   TMatrix m(-1,rsize-2,1,csize);

   cout << "\nWriting zeros to m..." << endl;
   for (i = m.GetRowLwb(); i <= m.GetRowUpb(); i++)
      for(j = m.GetColLwb(); j <= m.GetColUpb(); j++)
         m(i,j) = 0;
   verify_element_value(m,0);

   cout << "Creating zero m1 ..." << endl;
   TMatrix m1(TMatrix::kZero, m);
   verify_element_value(m1,0);

   cout << "Comparing m1 with 0 ..." << endl;
   Assert(m1 == 0);
   Assert(!(m1 != 0));

   cout << "Writing a pattern " << pattern << " by assigning to m(i,j)..." << endl;
   for (i = m.GetRowLwb(); i <= m.GetRowUpb(); i++)
      for (j = m.GetColLwb(); j <= m.GetColUpb(); j++)
         m(i,j) = pattern;
   verify_element_value(m,pattern);

   cout << "Writing the pattern by assigning to m1 as a whole ..."  << endl;
   m1 = pattern;
   verify_element_value(m1,pattern);

   cout << "Comparing m and m1 ..." << endl;
   Assert(m == m1);
   cout << "Comparing (m=0) and m1 ..." << endl;
   Assert(!(m.Zero() == m1));

   cout << "Clearing m1 ..." << endl;
   m1.Zero();
   verify_element_value(m1,0);

   cout << "\nClear m and add the pattern" << endl;
   m.Zero();
   m += pattern;
   verify_element_value(m,pattern);
   cout << "   add the doubled pattern with the negative sign" << endl;
   m += -2*pattern;
   verify_element_value(m,-pattern);
   cout << "   subtract the trippled pattern with the negative sign" << endl;
   m -= -3*pattern;
   verify_element_value(m,2*pattern);

   cout << "\nVerify comparison operations when all elems are the same" << endl;
   m = pattern;
   Assert( m == pattern && !(m != pattern) );
   Assert( m > 0 && m >= pattern && m <= pattern );
   Assert( m > -pattern && m >= -pattern );
   Assert( m <= pattern && !(m < pattern) );
   m -= 2*pattern;
   Assert( m  < -pattern/2 && m <= -pattern/2 );
   Assert( m  >= -pattern && !(m > -pattern) );

   cout << "\nVerify comparison operations when not all elems are the same" << endl;
   m = pattern; m(m.GetRowUpb(),m.GetColUpb()) = pattern-1;
   Assert( !(m == pattern) && !(m != pattern) );
   Assert( m != 0 );                   // none of elements are 0
   Assert( !(m >= pattern) && m <= pattern && !(m<pattern) );
   Assert( !(m <= pattern-1) && m >= pattern-1 && !(m>pattern-1) );

   cout << "\nAssign 2*pattern to m by repeating additions" << endl;
   m = 0; m += pattern; m += pattern;
   cout << "Assign 2*pattern to m1 by multiplying by two " << endl;
   m1 = pattern; m1 *= 2;
   verify_element_value(m1,2*pattern);
   Assert( m == m1 );
   cout << "Multiply m1 by one half returning it to the 1*pattern" << endl;
   m1 *= 1/2.;
   verify_element_value(m1,pattern);

   cout << "\nAssign -pattern to m and m1" << endl;
   m.Zero(); m -= pattern; m1 = -pattern;
   verify_element_value(m,-pattern);
   Assert( m == m1 );
   cout << "m = sqrt(sqr(m)); m1 = abs(m1); Now m and m1 have to be the same" << endl;
   m.Sqr();
   verify_element_value(m,pattern*pattern);
   m.Sqrt();
   verify_element_value(m,pattern);
   m1.Abs();
   verify_element_value(m1,pattern);
   Assert( m == m1 );

   cout << "\nCheck out to see that sin^2(x) + cos^2(x) = 1" << endl;
   FillMatrix f(m);
   m.Apply(f);
   m1 = m;
   ApplyFunction s(&TMath::Sin);
   ApplyFunction c(&TMath::Cos);
   m.Apply(s);
   m1.Apply(c);
   m.Sqr();
   m1.Sqr();
   m += m1;
   verify_element_value(m,1);

   cout << "\nDone\n" << endl;
}

//
//------------------------------------------------------------------------
//        Test binary matrix element-by-element operations
//
void test_binary_ebe_op(int rsize, int csize)
{
   const double pattern = 4.25;
   int i, j;

   cout << "\n---> Test Binary Matrix element-by-element operations" << endl;

   TMatrix m(2,rsize+1,0,csize-1);
   TMatrix m1(TMatrix::kZero,m);
   TMatrix mp(TMatrix::kZero,m);

   for (i = mp.GetRowLwb(); i <= mp.GetRowUpb(); i++)
      for (j = mp.GetColLwb(); j <= mp.GetColUpb(); j++)
         mp(i,j) = (i-m.GetNrows()/2.)*j*pattern;

   cout << "\nVerify assignment of a matrix to the matrix" << endl;
   m = pattern;
   m1.Zero();
   m1 = m;
   verify_element_value(m1,pattern);
   Assert( m1 == m );

   cout << "\nAdding the matrix to itself, uniform pattern " << pattern << endl;
   m.Zero(); m = pattern;
   m1 = m; m1 += m1;
   verify_element_value(m1,2*pattern);
   cout << "  subtracting two matrices ..." << endl;
   m1 -= m;
   verify_element_value(m1,pattern);
   cout << "  subtracting the matrix from itself" << endl;
   m1 -= m1;
   verify_element_value(m1,0);
   cout << "  adding two matrices together" << endl;
   m1 += m;
   verify_element_value(m1,pattern);

   cout << "\nArithmetic operations on matrices with not the same elements" << endl;
   cout << "   adding mp to the zero matrix..." << endl;
   m.Zero(); m += mp;
   verify_matrix_identity(m,mp);
   m1 = m;
   cout << "   making m = 3*mp and m1 = 3*mp, via add() and succesive mult" << endl;
   Add(m,2,mp);
   m1 += m1; m1 += mp;
   verify_matrix_identity(m,m1);
   cout << "   clear both m and m1, by subtracting from itself and via add()" << endl;
   m1 -= m1;
   Add(m,-3,mp);
   verify_matrix_identity(m,m1);

   cout << "\nTesting element-by-element multiplications and divisions" << endl;
   cout << "   squaring each element with sqr() and via multiplication" << endl;
   m = mp; m1 = mp;
   m.Sqr();
   ElementMult(m1,m1);
   verify_matrix_identity(m,m1);
   cout << "   compare (m = pattern^2)/pattern with pattern" << endl;
   m = pattern; m1 = pattern;
   m.Sqr();
   ElementDiv(m,m1);
   verify_element_value(m,pattern);
   Compare(m1,m);

   cout << "\nDone\n" << endl;
}

//
//------------------------------------------------------------------------
//              Verify matrix transposition
//
void test_transposition(int msize)
{
   cout << "\n---> Verify matrix transpose "
           "for matrices of a characteristic size " << msize << endl;

   {
      cout << "\nCheck to see that a square UnitMatrix stays the same";
      TMatrix m(msize,msize);
      m.UnitMatrix();
      TMatrix mt(TMatrix::kTransposed,m);
      Assert( m == mt );
   }

   {
      cout << "\nTest a non-square UnitMatrix";
      TMatrix m(msize,msize+1);
      m.UnitMatrix();
      TMatrix mt(TMatrix::kTransposed,m);
      Assert(m.GetNrows() == mt.GetNcols() && m.GetNcols() == mt.GetNrows() );
      int i,j;
      for (i = m.GetRowLwb(); i <= TMath::Min(m.GetRowUpb(),m.GetColUpb()); i++)
         for (j = m.GetColLwb(); j <= TMath::Min(m.GetRowUpb(),m.GetColUpb()); j++)
            Assert( m(i,j) == mt(i,j) );
   }

   {
      cout << "\nCheck to see that a symmetric (Hilbert)Matrix stays the same";
      TMatrix m(msize,msize);
      m.HilbertMatrix();
      TMatrix mt(TMatrix::kTransposed,m);
      Assert( m == mt );
   }

   {
      cout << "\nCheck transposing a non-symmetric matrix";
      TMatrix m(msize+1,msize);
      m.HilbertMatrix();
      m(1,2) = TMath::Pi();
      TMatrix mt(TMatrix::kTransposed,m);
      Assert(m.GetNrows() == mt.GetNcols() && m.GetNcols() == mt.GetNrows());
      Assert(mt(2,1) == (Real_t)TMath::Pi() && mt(1,2) != (Real_t)TMath::Pi());

      cout << "\nCheck double transposing a non-symmetric matrix" << endl;
      TMatrix mtt(TMatrix::kTransposed,mt);
      Assert( m == mtt );
   }

   cout << "\nDone\n" << endl;
}

//
//------------------------------------------------------------------------
//           Test special matrix creation
//
class MakeHilbert : public TElementPosAction {
   void Operation(Real_t &element) { element = 1./(fI+fJ+1); }
public:
   MakeHilbert() { }
};

class TestUnit : public TElementPosAction {
   int fIsUnit;
   void Operation(Real_t &element)
      { if (fIsUnit) fIsUnit = fI==fJ ? element == 1.0 : element == 0; }
public:
   TestUnit() : fIsUnit(0==0) { }
   int is_indeed_unit() const { return fIsUnit; }
};

void test_special_creation(int dim)
{
   cout << "\n---> Check creating some special matrices of dimension " <<
   dim << endl;

   {
      cout << "\ntest creating Hilbert matrices" << endl;
      TMatrix m(dim+1,dim);
      TMatrix m1(TMatrix::kZero,m);
      m.HilbertMatrix();
      Assert( !(m == m1) );
      Assert( m != 0 );
      MakeHilbert mh;
      m1.Apply(mh);
      Assert( m1 != 0 );
      Assert( m == m1 );
   }

   {
      cout << "test creating zero matrix and copy constructor" << endl;
      TMatrix m(dim,dim+1);
      m.HilbertMatrix();
      Assert( m != 0 );
      TMatrix m1(m);               // Applying the copy constructor
      Assert( m1 == m );
      TMatrix m2(TMatrix::kZero,m);
      Assert( m2 == 0 );
      Assert( m != 0 );
   }

   {
      cout << "test creating unit matrices" << endl;
      TMatrix m(dim,dim);
      {
         TestUnit test_unit;
         m.Apply(test_unit);
         Assert( !test_unit.is_indeed_unit() );
      }
      m.UnitMatrix();
      {
         TestUnit test_unit;
         m.Apply(test_unit);
         Assert( test_unit.is_indeed_unit() );
      }
      m.ResizeTo(dim-1,dim);
      TMatrix m2(TMatrix::kUnit,m);
      {
         TestUnit test_unit;
         m2.Apply(test_unit);
         Assert( test_unit.is_indeed_unit() );
      }
      m.ResizeTo(dim,dim-2);
      m.UnitMatrix();
      {
         TestUnit test_unit;
         m.Apply(test_unit);
         Assert( test_unit.is_indeed_unit() );
      }
   }

   {
      cout << "check to see that Haar matrix has *exactly* orthogonal columns"
           << endl;
      const int order = 5;
      TMatrix haar = THaarMatrix(order);
      Assert( haar.GetNrows() == (1<<order) && haar.GetNrows() == haar.GetNcols() );
      TVector colj(1<<order), coll(1<<order);
      int j;
      for (j = haar.GetColLwb(); j <= haar.GetColUpb(); j++) {
         colj = TMatrixColumn(haar,j);
         Assert(TMath::Abs(TMath::Abs(colj*colj - 1.0)) <= FLT_EPSILON );
         for (int l=j+1; l <= haar.GetColUpb(); l++) {
            coll = TMatrixColumn(haar,l);
            Assert( colj*coll == 0 );
         }
      }
      cout << "make Haar (sub)matrix and test it *is* a submatrix" << endl;
      const int no_sub_cols = (1<<order) - 3;
      TMatrix haar_sub = THaarMatrix(order,no_sub_cols);
      Assert( haar_sub.GetNrows() == (1<<order) &&
              haar_sub.GetNcols() == no_sub_cols );
      for (j = haar_sub.GetColLwb(); j <= haar_sub.GetColUpb(); j++) {
         colj = TMatrixColumn(haar,j);
         coll = TMatrixColumn(haar_sub,j);
         verify_vector_identity(colj,coll);
      }
   }

   cout << "\nDone\n" << endl;
}

//
//------------------------------------------------------------------------
//           Test matrix promises
//
class hilbert_matrix_promise : public TLazyMatrix {
   void FillIn(TMatrix &m) const { m.HilbertMatrix(); }
public:
   hilbert_matrix_promise(int nrows, int ncols)
      : TLazyMatrix(nrows,ncols) {}
   hilbert_matrix_promise(int row_lwb, int row_upb,
                          int col_lwb, int col_upb)
      : TLazyMatrix(row_lwb,row_upb,col_lwb,col_upb) { }
};

void test_matrix_promises(int dim)
{
   cout << "\n---> Check making/forcing promises, (lazy)matrices of dimension " <<
           dim << endl;

   {
      cout << "\nmake a promise and force it by a constructor" << endl;
      TMatrix m = hilbert_matrix_promise(dim,dim+1);
      TMatrix m1(TMatrix::kZero,m);
      Assert( !(m1 == m) && m1 == 0 );
      m1.HilbertMatrix();
      verify_matrix_identity(m,m1);
   }

   {
      cout << "make a promise and force it by an assignment" << endl;
      TMatrix m(-1,dim,0,dim);
      TMatrix m1(TMatrix::kZero,m);
      m = hilbert_matrix_promise(-1,dim,0,dim);
      Assert( !(m1 == m) && m1 == 0 );
      m1.HilbertMatrix();
      verify_matrix_identity(m,m1);
   }

   cout << "\nDone\n" << endl;
}

//
//------------------------------------------------------------------------
//             Verify the norm calculation
//
void test_norms(int rsize, int csize)
{
   cout << "\n---> Verify norm calculations" << endl;

   const double pattern = 10.25;

   if (rsize % 2 == 1 || csize %2 == 1)
      Fatal("test_norms",
            "Sorry, size of the matrix to test must be even for this test\n");

   TMatrix m(rsize,csize);

   cout << "\nAssign " << pattern << " to all the elements and check norms" << endl;
   m = pattern;
   cout << "  1. (col) norm should be pattern*nrows" << endl;
   Assert( m.Norm1() == pattern*m.GetNrows() );
   Assert( m.Norm1() == m.ColNorm() );
   cout << "  Inf (row) norm should be pattern*ncols" << endl;
   Assert( m.NormInf() == pattern*m.GetNcols() );
   Assert( m.NormInf() == m.RowNorm() );
   cout << "  Square of the Eucl norm has got to be pattern^2 * no_elems" << endl;
   Assert( m.E2Norm() == (pattern*pattern)*m.GetNoElements() );
   TMatrix m1(TMatrix::kZero,m);
   Assert( m.E2Norm() == E2Norm(m,m1) );

   cout << "\nDone\n" << endl;
}

//
//------------------------------------------------------------------------
//              Verify the determinant evaluation
//
void test_determinant(int msize)
{
   cout << "\n---> Verify determinant evaluation "
           "for a square matrix of size " << msize << endl;

   TMatrix m(msize,msize);

   cout << "\nCheck to see that the determinant of the unit matrix is one";
   m.UnitMatrix();
   cout << "\n	determinant is " << m.Determinant();
   Assert( m.Determinant() == 1 );

   const double pattern = 2.5;
   cout << "\nCheck the determinant for the matrix with " << pattern <<
           " at the diagonal";
   int i, j;
   for (i = m.GetRowLwb(); i <= m.GetRowUpb(); i++)
      for (j = m.GetColLwb(); j <= m.GetColUpb(); j++)
         m(i,j) = ( i==j ? pattern : 0 );
   cout << "\n	determinant is " << m.Determinant();
   Assert( m.Determinant() == TMath::Power(pattern,(double)m.GetNrows()) );

   cout << "\nCheck the determinant of the transposed matrix";
   m.UnitMatrix();
   m(1,2) = pattern;
   TMatrix m_tran(TMatrix::kTransposed,m);
   Assert( !(m == m_tran) );
   Assert( m.Determinant() == m_tran.Determinant() );

   {
      cout << "\nswap two rows/cols of a matrix and watch det's sign";
      m.UnitMatrix();
      TMatrixRow(m,3) = pattern;
      const double det1 = m.Determinant();
      TMatrixRow row1(m,1);
      TVector vrow1(m.GetRowLwb(),m.GetRowUpb()); vrow1 = row1;
      TVector vrow3(m.GetRowLwb(),m.GetRowUpb()); vrow3 = TMatrixRow(m,3);
      row1 = vrow3; TMatrixRow(m,3) = vrow1;
      Assert( m.Determinant() == -det1 );
      TMatrixColumn col2(m,2);
      TVector vcol2(m.GetRowLwb(),m.GetRowUpb()); vcol2 = col2;
      TVector vcol4(m.GetRowLwb(),m.GetRowUpb()); vcol4 = TMatrixColumn(m,4);
      col2 = vcol4; TMatrixColumn(m,4) = vcol2;
      Assert( m.Determinant() == det1 );
   }

   cout << "\nCheck the determinant for the matrix with " << pattern <<
           " at the anti-diagonal";
   for (i = m.GetRowLwb(); i <= m.GetRowUpb(); i++)
      for (j = m.GetColLwb(); j <= m.GetColUpb(); j++)
         m(i,j) = ( i==(m.GetColUpb()+m.GetColLwb()-j) ? pattern : 0 );
   Assert( m.Determinant() == TMath::Power(pattern,(double)m.GetNrows()) *
         ( m.GetNrows()*(m.GetNrows()-1)/2 & 1 ? -1 : 1 ) );

   cout << "\nCheck the determinant for the singular matrix"
           "\n	defined as above with zero first row";
   m.Zero();
   for (i = m.GetRowLwb()+1; i <= m.GetRowUpb(); i++)
      for (j = m.GetColLwb(); j <= m.GetColUpb(); j++)
         m(i,j) = ( i==(m.GetColUpb()+m.GetColLwb()-j) ? pattern : 0 );
   cout << "\n	determinant is " << m.Determinant();
   Assert( m.Determinant() == 0 );

   cout << "\nCheck out the determinant of the Hilbert matrix";
   TMatrix H(3,3);
   H.HilbertMatrix();
   cout << "\n    3x3 Hilbert matrix: exact determinant 1/2160 ";
   cout << "\n                              computed    1/"<< 1/H.Determinant();

   H.ResizeTo(4,4);
   H.HilbertMatrix();
   cout << "\n    4x4 Hilbert matrix: exact determinant 1/6048000 ";
   cout << "\n                              computed    1/"<< 1/H.Determinant();

   H.ResizeTo(5,5);
   H.HilbertMatrix();
   cout << "\n    5x5 Hilbert matrix: exact determinant 3.749295e-12";
   cout << "\n                              computed    "<< H.Determinant();

   H.ResizeTo(7,7);
   H.HilbertMatrix();
   cout << "\n    7x7 Hilbert matrix: exact determinant 4.8358e-25";
   cout << "\n                              computed    "<< H.Determinant();

   H.ResizeTo(9,9);
   H.HilbertMatrix();
   cout << "\n    9x9 Hilbert matrix: exact determinant 9.72023e-43";
   cout << "\n                              computed    "<< H.Determinant();

   H.ResizeTo(10,10);
   H.HilbertMatrix();
   cout << "\n    10x10 Hilbert matrix: exact determinant 2.16418e-53";
   cout << "\n                              computed    "<< H.Determinant()
        << endl;

   cout << "\nDone\n" << endl;
}

//
//------------------------------------------------------------------------
//               Verify matrix multiplications
//
void test_mm_multiplications(int msize)
{
   cout << "\n---> Verify matrix multiplications "
           "for matrices of the characteristic size " << msize << endl;

   {
      cout << "\nTest inline multiplications of the UnitMatrix" << endl;
      TMatrix m(-1,msize,-1,msize);
      TMatrix u(TMatrix::kUnit,m);
      m.HilbertMatrix(); m(3,1) = TMath::Pi();
      u *= m;
      verify_matrix_identity(u,m);
   }

   {
      cout << "Test inline multiplications by a DiagMatrix" << endl;
      TMatrix m(msize+3,msize);
      m.HilbertMatrix(); m(1,3) = TMath::Pi();
      TVector v(msize);
      int i;
      for (i = v.GetLwb(); i <= v.GetUpb(); i++)
         v(i) = 1+i;
      TMatrix diag(msize,msize);
      //(TMatrixDiag)diag = v;
      TMatrixDiag td = diag;
      td = v;
      TMatrix eth = m;
      for (i = eth.GetRowLwb(); i <= eth.GetRowUpb(); i++)
         for (int j = eth.GetColLwb(); j <= eth.GetColUpb(); j++)
            eth(i,j) *= v(j);
      m *= diag;
      verify_matrix_identity(m,eth);
   }

   {
      cout << "Test XPP = X where P is a permutation matrix" << endl;
      TMatrix m(msize-1,msize);
      m.HilbertMatrix(); m(2,3) = TMath::Pi();
      TMatrix eth = m;
      TMatrix p(msize,msize);
      for (int i = p.GetRowLwb(); i <= p.GetRowUpb(); i++)
         p(p.GetRowUpb()+p.GetRowLwb()-i,i) = 1;
      m *= p;
      m *= p;
      verify_matrix_identity(m,eth);
   }

   {
      cout << "Test general matrix multiplication through inline mult" << endl;
      TMatrix m(msize-2,msize);
      m.HilbertMatrix(); m(3,3) = TMath::Pi();
      TMatrix mt(TMatrix::kTransposed,m);
      TMatrix p(msize,msize);
      p.HilbertMatrix();
      TMatrixDiag(p) += 1;
      TMatrix mp(m,TMatrix::kMult,p);
      TMatrix m1 = m;
      m *= p;
      verify_matrix_identity(m,mp);
      TMatrix mp1(mt,TMatrix::kTransposeMult,p);
      verify_matrix_identity(m,mp1);
      Assert( !(m1 == m) );
      TMatrix mp2(TMatrix::kZero,m1);
      Assert( mp2 == 0 );
      mp2.Mult(m1,p);
      verify_matrix_identity(m,mp2);
   }

   {
      cout << "Check to see UU' = U'U = E when U is the Haar matrix" << endl;
      const int order = 5;
      const int no_sub_cols = (1<<order) - 5;
      TMatrix haar_sub = THaarMatrix(5,no_sub_cols);
      TMatrix haar_sub_t(TMatrix::kTransposed,haar_sub);
      TMatrix hsths(haar_sub_t,TMatrix::kMult,haar_sub);
      TMatrix hsths1(TMatrix::kZero,hsths); hsths1.Mult(haar_sub_t,haar_sub);
      TMatrix hsths_eth(TMatrix::kUnit,hsths);
      Assert( hsths.GetNrows() == no_sub_cols && hsths.GetNcols() == no_sub_cols );
      verify_matrix_identity(hsths,hsths_eth);
      verify_matrix_identity(hsths1,hsths_eth);

      TMatrix haar = THaarMatrix(5);
      TMatrix unit(TMatrix::kUnit,haar);
      TMatrix haar_t(TMatrix::kTransposed,haar);
      TMatrix hth(haar,TMatrix::kTransposeMult,haar);
      TMatrix hht(haar,TMatrix::kMult,haar_t);
      TMatrix hht1 = haar; hht1 *= haar_t;
      TMatrix hht2(TMatrix::kZero,haar); hht2.Mult(haar,haar_t);
      verify_matrix_identity(unit,hth);
      verify_matrix_identity(unit,hht);
      verify_matrix_identity(unit,hht1);
      verify_matrix_identity(unit,hht2);
   }
   cout << "\nDone\n" << endl;
}

//
//------------------------------------------------------------------------
//               Verify vector-matrix multiplications
//
void test_vm_multiplications(int msize)
{
   cout << "\n---> Verify vector-matrix multiplications "
          "for matrices of the characteristic size " << msize << endl;
   {
      cout << "\nCheck shrinking a vector by multiplying by a non-sq unit matrix"
           << endl;
      TVector vb(-2,msize);
      for (int i = vb.GetLwb(); i <= vb.GetUpb(); i++)
         vb(i) = TMath::Pi() - i;
      Assert( vb != 0 );
      TMatrix mc(1,msize-2,-2,msize);       // contracting matrix
      mc.UnitMatrix();
      TVector v1 = vb;
      TVector v2 = vb;
      v1 *= mc;
      v2.ResizeTo(1,msize-2);
      verify_vector_identity(v1,v2);
   }

   {
      cout << "Check expanding a vector by multiplying by a non-sq unit matrix"
           << endl;
      TVector vb(msize);
      for (int i = vb.GetLwb(); i <= vb.GetUpb(); i++)
         vb(i) = TMath::Pi() + i;
      Assert( vb != 0 );
      TMatrix me(2,msize+5,0,msize-1);    // expanding matrix
      me.UnitMatrix();
      TVector v1 = vb;
      TVector v2 = vb;
      v1 *= me;
      v2.ResizeTo(v1);
      verify_vector_identity(v1,v2);
   }

   {
      cout << "Check general matrix-vector multiplication" << endl;
      TVector vb(msize);
      for (int i = vb.GetLwb(); i <= vb.GetUpb(); i++)
         vb(i) = TMath::Pi() + i;
      TMatrix vm(msize,1);
      TMatrixColumn(vm,0) = vb;
      TMatrix m(0,msize,0,msize-1);
      m.HilbertMatrix();
      vb *= m;
      Assert( vb.GetLwb() == 0 );
      TMatrix mvm(m,TMatrix::kMult,vm);
      TMatrix mvb(msize+1,1);
      TMatrixColumn(mvb,0) = vb;
      verify_matrix_identity(mvb,mvm);
   }

   cout << "\nDone\n" << endl;
}

//
//------------------------------------------------------------------------
//               Verify matrix inversion
//
void test_inversion(int msize)
{
   cout << "\n---> Verify matrix inversion for square matrices "
           "of size " << msize << endl;
   {
      cout << "\nTest inversion of a diagonal matrix" << endl;
      TMatrix m(-1,msize,-1,msize);
      TMatrix mi(TMatrix::kZero,m);
      for (int i = m.GetRowLwb(); i <= m.GetRowUpb(); i++)
         mi(i,i) = 1/(m(i,i) = i-m.GetRowLwb()+1);
      TMatrix mi1(TMatrix::kInverted,m);
      m.Invert();
      verify_matrix_identity(m,mi);
      verify_matrix_identity(mi1,mi);
   }

   {
      cout << "Test inversion of an orthonormal (Haar) matrix" << endl;
      TMatrix m = THaarMatrix(3);
      TMatrix morig = m;
      TMatrix mt(TMatrix::kTransposed,m);
      double det = -1;         // init to a wrong val to see if it's changed
      m.Invert(&det);
      Assert( TMath::Abs(det-1) <= FLT_EPSILON );
      verify_matrix_identity(m,mt);
      TMatrix mti(TMatrix::kInverted,mt);
      verify_matrix_identity(mti,morig);
   }

   {
      cout << "Test inversion of a good matrix with diagonal dominance" << endl;
      TMatrix m(msize,msize);
      m.HilbertMatrix();
      TMatrixDiag(m) += 1;
      TMatrix morig = m;
      double det_inv = 0;
      const double det_comp = m.Determinant();
      m.Invert(&det_inv);
      cout << "\tcomputed determinant             " << det_comp << endl;
      cout << "\tdeterminant returned by Invert() " << det_inv << endl;

      cout << "\tcheck to see M^(-1) * M is E" << endl;
      TMatrix mim(m,TMatrix::kMult,morig);
      TMatrix unit(TMatrix::kUnit,m);
      verify_matrix_identity(mim,unit);

      cout << "\tcheck to see M * M^(-1) is E" << endl;
      TMatrix mmi = morig; mmi *= m;
      verify_matrix_identity(mmi,unit);
   }

   cout << "\nDone\n" << endl;
}

//
//------------------------------------------------------------------------
//           Test matrix I/O
//
void test_matrix_io()
{
   const double pattern = TMath::Pi();

   cout << "\n---> Test matrix I/O" << endl;

   TMatrix m(40,40);
   m = pattern;

   cout << "\nWrite matrix m to database" << endl;

   TFile *f = new TFile("vmatrix.root", "RECREATE");

   m.Write("m");

   cout << "\nClose database" << endl;
   delete f;

   cout << "\nOpen database in read-only mode and read matrix" << endl;
   TFile *f1 = new TFile("vmatrix.root");

   TMatrix *mr = (TMatrix*) f1->Get("m");

   cout << "\nRead matrix should be same as original still in memory" << endl;
   Assert((*mr) == m);

   delete f1;

   cout << "\nDone\n" << endl;
}

//
//------------------------------------------------------------------------
//                    Main module
//
int main()
{
   // Make sure all registered dictionaries have been initialized
   TApplication app("vmatrix", 0, 0);

   cout<< "\n\n" <<
          "----------------------------------------------------------------" <<
          "\n\t\tVerify Operations on Matrices" << endl;

   test_allocation();
   test_element_op(20,10);
   test_binary_ebe_op(10,20);
   test_transposition(20);
   test_special_creation(20);
   test_matrix_promises(20);
   test_norms(40,20);

   // test advanced matrix operations
   test_determinant(20);
   test_mm_multiplications(20);
   test_vm_multiplications(20);
   test_inversion(20);

   test_matrix_io();

   return 0;
}
