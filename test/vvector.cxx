// @(#)root/test:$Name:  $:$Id: vvector.cxx,v 1.2 2000/07/11 18:05:26 rdm Exp $
// Author: Fons Rademakers   14/11/97

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Linear Algebra Package -- Vector Verifications.                      //
//                                                                      //
// This file implements a large set of TVector operation tests.         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#include "TFile.h"
#include "TMatrix.h"
#include <iostream.h>

//
//------------------------------------------------------------------------
//            Service validation functions
//
void verify_vector_identity(const TVector &v1, const TVector &v2)
{ VerifyVectorIdentity(v1,v2); }

void verify_element_value(const TVector &v, Real_t val)
{ VerifyElementValue(v, val); }

void are_compatible(const TVector &v1, const TVector &v2)
{
   if (AreCompatible(v1, v2))
      cout << "vectors are compatible" << endl;
   else
      cout << "vectors are NOT compatible" << endl;
}

//
//------------------------------------------------------------------------
//          Test allocation functions and compatibility check
//

void test_allocation()
{

   cout << "\n\n---> Test allocation and compatibility check" << endl;

   TVector v1(20);
   TVector v2(0,19);
   TVector v3(1,20);
   TVector v4(v1);

   cout << "\nStatus information reported for vector v3:" << endl;
   cout << "  Lower bound ... " << v3.GetLwb() << endl;
   cout << "  Upper bound ... " << v3.GetUpb() << endl;
   cout << "  No. of elements " << v3.GetNoElements() << endl;

   cout << "\nCheck vectors 1 & 2 for compatibility" << endl;
   are_compatible(v1,v2);

   cout << "Check vectors 1 & 4 for compatibility" << endl;
   are_compatible(v1,v4);

   cout << "v2 has to be compatible with v3 after resizing to v3" << endl;
   v2.ResizeTo(v3);
   are_compatible(v2,v3);

   TVector v5(v1.GetUpb()+5);
   cout << "v1 has to be compatible with v5 after resizing to v5.upb" << endl;
   v1.ResizeTo(v5.GetNoElements());
   are_compatible(v1,v5);

   {
      cout << "Check that shrinking does not change remaining elements" << endl;
      TVector vb(-1,20);
      int i;
      for (i = vb.GetLwb(); i <= vb.GetUpb(); i++)
         vb(i) = i+TMath::Pi();
      TVector v = vb;
      Assert( v == vb );
      Assert( v != 0 );
      v.ResizeTo(0,10);
      for (i = v.GetLwb(); i <= v.GetUpb(); i++)
         Assert( v(i) == vb(i-v.GetLwb()+vb.GetLwb()) );
      cout << "Check that expansion expands by zeros" << endl;
      const int old_nelems = v.GetUpb() - v.GetLwb() + 1;
      v.ResizeTo(vb);
      Assert( !(v == vb) );
      for (i = v.GetLwb(); i < v.GetLwb()+old_nelems; i++)
         Assert( v(i) == vb(i) );
      for (; i <= v.GetUpb(); i++)
         Assert( v(i) == 0 );
   }
   cout << "\nDone\n" << endl;
}

//
//------------------------------------------------------------------------
//                Test uniform element operations
//
class SinAction : public TElementAction {
   void Operation(Real_t &element) { element = TMath::Sin(element); }
   public:
      SinAction() { }
};

class CosAction : public TElementPosAction {
   Double_t factor;
   void Operation(Real_t &element) { element = TMath::Cos(factor*fI); }
   public:
      CosAction(Int_t no_elems): factor(2*TMath::Pi()/no_elems) { }
};

void test_element_op(int vsize)
{
   const double pattern = TMath::Pi();
   int i;

   cout << "\n---> Test operations that treat each element uniformly" << endl;

   TVector v(-1,vsize-2);
   TVector v1(v);

   cout << "\nWriting zeros to v..." << endl;
   for (i = v.GetLwb(); i <= v.GetUpb(); i++)
      v(i) = 0;
   verify_element_value(v,0);

   cout << "Clearing v1 ..." << endl;
   v1.Zero();
   verify_element_value(v1,0);

   cout << "Comparing v1 with 0 ..." << endl;
   Assert(v1 == 0);

   cout << "Writing a pattern " << pattern << " by assigning to v(i)..." << endl;
   for (i = v.GetLwb(); i <= v.GetUpb(); i++)
      v(i) = pattern;
   verify_element_value(v,pattern);

   cout << "Writing the pattern by assigning to v1 as a whole ..." << endl;
   v1 = pattern;
   verify_element_value(v1,pattern);

   cout << "Comparing v and v1 ..." << endl;
   Assert(v == v1);
   cout << "Comparing (v=0) and v1 ..." << endl;
   Assert(!(v.Zero() == v1));

   cout << "\nClear v and add the pattern" << endl;
   v.Zero();
   v += pattern;
   verify_element_value(v,pattern);
   cout << "   add the doubled pattern with the negative sign" << endl;
   v += -2*pattern;
   verify_element_value(v,-pattern);
   cout << "   subtract the trippled pattern with the negative sign" << endl;
   v -= -3*pattern;
   verify_element_value(v,2*pattern);

   cout << "\nVerify comparison operations" << endl;
   v = pattern;
   Assert( v == pattern && !(v != pattern) && v >= pattern && v <= pattern );
   Assert( v > 0 && v >= 0 );
   Assert( v > -pattern && v >= -pattern );
   Assert( v < pattern+1 && v <= pattern+1 );
   v(v.GetUpb()) += 1;
   Assert( !(v==pattern) && !(v != pattern) && v != pattern-1 );
   Assert( v >= pattern && !(v > pattern) && !(v >= pattern+1) );
   Assert( v <= pattern+1.001 && !(v < pattern+1) && !(v <= pattern) );

   cout << "\nAssign 2*pattern to v by repeating additions" << endl;
   v = 0; v += pattern; v += pattern;
   cout << "Assign 2*pattern to v1 by multiplying by two" << endl;
   v1 = pattern; v1 *= 2;
   verify_element_value(v1,2*pattern);
   Assert( v == v1 );
   cout << "Multiply v1 by one half returning it to the 1*pattern" << endl;
   v1 *= 1/2.;
   verify_element_value(v1,pattern);

   cout << "\nAssign -pattern to v and v1" << endl;
   v.Zero(); v -= pattern; v1 = -pattern;
   verify_element_value(v,-pattern);
   Assert( v == v1 );
   cout << "v = sqrt(sqr(v)); v1 = abs(v1); Now v and v1 have to be the same" << endl;
   v.Sqr();
   verify_element_value(v,pattern*pattern);
   v.Sqrt();
   verify_element_value(v,pattern);
   v1.Abs();
   verify_element_value(v1,pattern);
   Assert( v == v1 );

   {
      cout << "\nCheck out to see that sin^2(x) + cos^2(x) = 1" << endl;
      for (i = v.GetLwb(); i <= v.GetUpb(); i++)
         v(i) = 2*TMath::Pi()/v.GetNoElements() * i;
      SinAction s;
      v.Apply(s);
      TVector v2 = v;

      CosAction c(v.GetNoElements());
      v1.Apply(c);
      TVector v3 = v1;
      v.Sqr();
      v1.Sqr();
      v += v1;
      verify_element_value(v,1);

#if 0
      cout << "\n\tdo it again through TLazyMatrix promise of a vector" << endl;
      class square_add : public TLazyMatrix, public TElementAction {
         const TVector &v1; TVector &v2;
         void Operation(Real_t &element)
              { Assert(j==1); element = v1(i)*v1(i) + v2(i)*v2(i); }
         void FillIn(TMatrix &m) const { m.Apply(*this); }
      public:
         SquareAdd(const TVector &v1, TVector &v2) :
         TLazyMatrix(v1.GetRowLwb(), v1.GetRowUpb(),1,1),
                     fV1(v1), fV2(v2) {}
      };
      TVector vres = SquareAdd(v2,v3);
      TVector vres1 = v2; Assert( !(vres1 == vres) );
      verify_element_value(vres,1);
      vres1 = SquareAdd(v2,v3);
      verify_element_value(vres1,1);
#endif
   }

   cout << "\nVerify constructor with initialization" << endl;
   TVector vi(0,4,0.0,1.0,2.0,3.0,4.0,"END");
   TVector vit(5);
   for (i = vit.GetLwb(); i <= vit.GetUpb(); i++)
      vit(i) = i;
   verify_vector_identity(vi,vit);

   cout << "\nDone\n" << endl;
}

//
//------------------------------------------------------------------------
//                 Test binary vector operations
//
void test_binary_op(int vsize)
{
   const double pattern = TMath::Pi();
   int i;

   cout << "\n---> Test Binary Vector operations" << endl;

   TVector v(2,vsize+1);
   TVector v1(v);
   TVector vp(v);

   for (i = v.GetLwb(); i <= v.GetUpb(); i++)
      vp(i) = (i-v.GetNoElements()/2.)*pattern;

   cout << "\nVerify assignment of a vector to the vector" << endl;
   v = pattern;
   v1.Zero();
   v1 = v;
   verify_element_value(v1,pattern);
   Assert( v1 == v );

   cout << "\nAdding one vector to itself, uniform pattern " << pattern << endl;
   v.Zero(); v = pattern;
   v1 = v; v1 += v1;
   verify_element_value(v1,2*pattern);
   cout << "  subtracting two vectors ..." << endl;
   v1 -= v;
   verify_element_value(v1,pattern);
   cout << "  subtracting the vector from itself" << endl;
   v1 -= v1;
   verify_element_value(v1,0);
   cout << "  adding two vectors together" << endl;
   v1 += v;
   verify_element_value(v1,pattern);

   cout << "\nArithmetic operations on vectors with not the same elements" << endl;
   cout << "   adding vp to the zero vector..." << endl;
   v.Zero(); v += vp;
//  verify_vector_identity(v,vp);
   Assert( v == vp );
   v1 = v;
   cout << "   making v = 3*vp and v1 = 3*vp, via add() and succesive mult" << endl;
   Add(v,2,vp);
   v1 += v1; v1 += vp;
   verify_vector_identity(v,v1);
   cout << "   clear both v and v1, by subtracting from itself and via add()" << endl;
   v1 -= v1;
   Add(v,-3,vp);
   verify_vector_identity(v,v1);

   cout << "\nTesting element-by-element multiplications and divisions" << endl;
   cout << "   squaring each element with sqr() and via multiplication" << endl;
   v = vp; v1 = vp;
   v.Sqr();
   ElementMult(v1,v1);
   verify_vector_identity(v,v1);
   cout << "   compare (v = pattern^2)/pattern with pattern" << endl;
   v = pattern; v1 = pattern;
   v.Sqr();
   ElementDiv(v,v1);
   verify_element_value(v,pattern);
   Compare(v1,v);

   cout << "\nDone\n" << endl;
}

//
//------------------------------------------------------------------------
//               Verify the norm calculation
//
void test_norms(int vsize)
{
   cout << "\n---> Verify norm calculations" << endl;

   const double pattern = 10.25;

   if ( vsize % 2 == 1 )
      Fatal("test_norms", "size of the vector to test must be even for this test\n");

   TVector v(vsize);
   TVector v1(v);

   cout << "\nAssign " << pattern << " to all the elements and check norms" << endl;
   v = pattern;
   cout << "  1. norm should be pattern*no_elems" << endl;
   Assert( v.Norm1() == pattern*v.GetNoElements() );
   cout << "  Square of the 2. norm has got to be pattern^2 * no_elems" << endl;
   Assert( v.Norm2Sqr() == (pattern*pattern)*v.GetNoElements() );
   cout << "  Inf norm should be pattern itself" << endl;
   Assert( v.NormInf() == pattern );
   cout << "  Scalar product of vector by itself is the sqr(2. vector norm)" << endl;
   Assert( v.Norm2Sqr() == v * v );

   double ap_step = 1;
   double ap_a0   = -pattern;
   int n = v.GetNoElements();
   cout << "\nAssign the arithm progression with 1. term " << ap_a0 <<
           "\nand the difference " << ap_step << endl;
   int i;
   for (i = v.GetLwb(); i <= v.GetUpb(); i++)
      v(i) = (i-v.GetLwb())*ap_step + ap_a0;
   int l = TMath::Min(TMath::Max((int)TMath::Ceil(-ap_a0/ap_step),0),n);
   double norm = (2*ap_a0 + (l+n-1)*ap_step)/2*(n-l) +
                 (-2*ap_a0-(l-1)*ap_step)/2*l;
   cout << "  1. norm should be " << norm << endl;
   Assert( v.Norm1() == norm );
   norm = n*( (ap_a0*ap_a0) + ap_a0*ap_step*(n-1) + (ap_step*ap_step)*(n-1)*(2*n-1)/6);
   cout << "  Square of the 2. norm has got to be "
           "n*[ a0^2 + a0*q*(n-1) + q^2/6*(n-1)*(2n-1) ], or " << norm << endl;
   Assert( v.Norm2Sqr() == norm );
   norm = TMath::Max(TMath::Abs(v(v.GetLwb())),TMath::Abs(v(v.GetUpb())));
   cout << "  Inf norm should be max(abs(a0),abs(a0+(n-1)*q)), ie " << norm
        << endl;
   Assert( v.NormInf() == norm );
   cout << "  Scalar product of vector by itself is the sqr(2. vector norm)" << endl;
   Assert( v.Norm2Sqr() == v * v );

#if 0
   v1.Zero();
   Compare(v,v1);  // they are not equal (of course)
#endif

   cout << "\nConstruct v1 to be orthogonal to v as v(n), -v(n-1), v(n-2)..." << endl;
   for (i = 0; i < v1.GetNoElements(); i++)
      v1(i+v1.GetLwb()) = v(v.GetUpb()-i) * ( i % 2 == 1 ? -1 : 1 );
   cout << "||v1|| has got to be equal ||v|| regardless of the norm def" << endl;
   Assert( v1.Norm1() == v.Norm1() );
   Assert( v1.Norm2Sqr() == v.Norm2Sqr() );
   Assert( v1.NormInf() == v.NormInf() );
   cout << "But the scalar product has to be zero" << endl;
   Assert( v1 * v == 0 );

   cout << "\nDone\n" << endl;
}

//
//------------------------------------------------------------------------
//           Test operations with vectors and matrix slices
//
void test_matrix_slices(int vsize)
{
   const Real_t pattern = 8.625;
   int i;

   cout << "\n---> Test operations with vectors and matrix slices" << endl;

   TVector vc(0,vsize);
   TVector vr(0,vsize+1);
   TMatrix m(0,vsize,0,vsize+1);

   cout << "\nCheck modifying the matrix column-by-column" << endl;
   m = pattern;
   Assert( m == pattern );
   for (i = m.GetColLwb(); i <= m.GetColUpb(); i++) {
      TMatrixColumn(m,i) = pattern-1;
      Assert( !( m == pattern ) && !( m != pattern ) );
      TMatrixColumn(m,i) *= 2;
      vc = TMatrixColumn(m,i);
      verify_element_value(vc,2*(pattern-1));
      vc = TMatrixColumn(m, i+1 > m.GetColUpb() ? m.GetColLwb() : i+1);
      verify_element_value(vc,pattern);
      TMatrixColumn(m,i) *= 0.5;
      TMatrixColumn(m,i) += 1;
      Assert( m == pattern );
   }

   Assert( m == pattern );
   for (i = m.GetColLwb(); i <= m.GetColUpb(); i++) {
      vc = pattern+1;
      TMatrixColumn(m,i) = vc;
      Assert( !( m == pattern ) && !( m != pattern ) );
      {
         TMatrixColumn mc(m,i);
         for (int j = m.GetRowLwb(); j <= m.GetRowUpb(); j++)
            mc(j) *= 4;
      }
      vc = TMatrixColumn(m,i);
      verify_element_value(vc,4*(pattern+1));
      TMatrixColumn(m,i) *= 0.25;
      TMatrixColumn(m,i) += -1;
      vc = TMatrixColumn(m,i);
      verify_element_value(vc,pattern);
      Assert( m == pattern );
   }

   cout << "\nCheck modifying the matrix row-by-row" << endl;
   m = pattern;
   Assert( m == pattern );
   for (i = m.GetRowLwb(); i <= m.GetRowUpb(); i++) {
      TMatrixRow(m,i) = pattern+2;
      Assert( !( m == pattern ) && !( m != pattern ) );
      vr = TMatrixRow(m,i);
      verify_element_value(vr,pattern+2);
      vr = TMatrixRow(m,i+1 > m.GetRowUpb() ? m.GetRowLwb() : i+1);
      verify_element_value(vr,pattern);
      TMatrixRow(m,i) += -2;
      Assert( m == pattern );
   }

   Assert( m == pattern );
   for (i = m.GetRowLwb(); i <= m.GetRowUpb(); i++) {
      vr = pattern-2;
      TMatrixRow(m,i) = vr;
      Assert( !( m == pattern ) && !( m != pattern ) );
      {
         TMatrixRow mr(m,i);
         for (int j = m.GetColLwb(); j <= m.GetColUpb(); j++)
            mr(j) *= 8;
      }
      vr = TMatrixRow(m,i);
      verify_element_value(vr,8*(pattern-2));
      TMatrixRow(m,i) *= 1./8;
      TMatrixRow(m,i) += 2;
      vr = TMatrixRow(m,i);
      verify_element_value(vr,pattern);
      Assert( m == pattern );
   }

   cout << "\nCheck modifying the matrix diagonal" << endl;
   m = pattern;
   //(TMatrixDiag)m = pattern-3;
   TMatrixDiag td = m;
   td = pattern-3;
   Assert( !( m == pattern ) && !( m != pattern ) );
   vc = TMatrixDiag(m);
   verify_element_value(vc,pattern-3);
   //TMatrixDiag(m) += 3;
   td += 3;
   Assert( m == pattern );
   vc = pattern+3;
   //(TMatrixDiag)m = vc;
   td = vc;
   Assert( !( m == pattern ) && !( m != pattern ) );
   {
      TMatrixDiag md(m);
      for (int j = 1; j <= md.GetNdiags(); j++)
         md(j) /= 1.5;
   }
   vc = TMatrixDiag(m);
   verify_element_value(vc,(pattern+3)/1.5);
   TMatrixDiag(m) *= 1.5;
   TMatrixDiag(m) += -3;
   vc = TMatrixDiag(m);
   verify_element_value(vc,pattern);
   Assert( m == pattern );

   cout << "\nCheck out to see that multiplying by diagonal is column-wise"
           "\nmatrix multiplication" << endl;
   TMatrix mm(m);
   TMatrix m1(m.GetRowLwb(),TMath::Max(m.GetRowUpb(),m.GetColUpb()),
              m.GetColLwb(),TMath::Max(m.GetRowUpb(),m.GetColUpb()));
   TVector vc1(vc), vc2(vc);
   for (i = m.GetRowLwb(); i < m.GetRowUpb(); i++)
      TMatrixRow(m,i) = pattern+i;      // Make a multiplicand
   mm = m;                              // Save it

   m1 = pattern+10;
   for (i = vr.GetLwb(); i <= vr.GetUpb(); i++)
      vr(i) = i+2;
   //(TMatrixDiag)m1 = vr;               // Make the other multiplicand
   TMatrixDiag td2 = m1;
   td2 = vr;
   Assert( !(m1 == pattern+10) );

   m *= TMatrixDiag(m1);
   for (i = m.GetColLwb(); i <= m.GetColUpb(); i++) {
      vc1 = TMatrixColumn(mm,i);
      vc1 *= vr(i);                    // Do a column-wise multiplication
      vc2 = TMatrixColumn(m,i);
      verify_vector_identity(vc1, vc2);
   }

   cout << "\nDone\n" << endl;
}

//
//------------------------------------------------------------------------
//           Test vector I/O
//
void test_vector_io()
{
   const double pattern = TMath::Pi();

   cout << "\n---> Test vector I/O" << endl;

   TVector v(40);
   v = pattern;

   cout << "\nWrite vector v to database" << endl;

   TFile *f = new TFile("vvector.root", "RECREATE");

   v.Write("v");

   cout << "\nClose database" << endl;
   delete f;

   cout << "\nOpen database in read-only mode and read vector" << endl;
   TFile *f1 = new TFile("vvector.root");

   TVector *vr = (TVector*) f1->Get("v");

   cout << "\nRead vector should be same as original still in memory" << endl;
   Assert((*vr) == v);

   delete f1;

   cout << "\nDone\n" << endl;
}

//
//------------------------------------------------------------------------
//                    Main module
//
int main()
{
   TROOT vec("vector","verify vectors");
   cout<< "\n\n" <<
          "----------------------------------------------------------------" <<
          "\n\t\tVerify Operations on Vectors" << endl;

   test_allocation();
   test_element_op(20);
   test_binary_op(20);
   test_norms(20);
   test_matrix_slices(20);
   test_vector_io();

   return 0;
}
