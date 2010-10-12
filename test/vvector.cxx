// @(#)root/test:$Id$
// Author: Fons Rademakers and Eddy Offermann  Nov 2003

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Linear Algebra Package -- Vector Verifications.                      //
//                                                                      //
// This file implements a large set of TVectorD operation tests.        //
// *******************************************************************  //
// *  Starting  Vector - S T R E S S suite                              //
// *******************************************************************  //
// Test  1 : Allocation, Filling, Resizing......................... OK  //
// Test  2 : Uniform vector operations............................. OK  //
// Test  3 : Binary vector element-by-element operations............OK  //
// Test  4 : Vector Norms...........................................OK  //
// Test  5 : Matrix Slices to Vectors...............................OK  //
// Test  6 : Vector Persistence.....................................OK  //
// *******************************************************************  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//_____________________________batch only_____________________
#ifndef __CINT__

#include "TROOT.h"
#include "TFile.h"
#include "Riostream.h"
#include "TVectorD.h"
#include "TMath.h"

void stress_vector       (Int_t verbose);
void stress_allocation   ();
void stress_element_op   (Int_t vsize);
void stress_binary_op    (Int_t vsize);
void stress_norms        (Int_t vsize);
void stress_matrix_slices(Int_t vsize);
void stress_vector_io    ();

int main(int argc,char **argv)
{
  Int_t verbose = 0;
  Char_t c;
  while (argc > 1 && argv[1][0] == '-' && argv[1][1] != 0) {
    for (Int_t i = 1; (c = argv[1][i]) != 0; i++) {
      switch (c) {
        case 'v':
          verbose = 1;
          break;
        default:
          Error("vvector", "unknown flag -%c",c);
          break;
      }
    }
    argc--;
    argv++;
  }
  stress_vector(verbose);
  return 0;
}

#endif

#define EPSILON 1.0e-14

Int_t gVerbose = 0;

//________________________________common part_________________________

void stress_vector(Int_t verbose=0)
{
  cout << "******************************************************************" <<endl;
  cout << "*  Starting  Vector - S T R E S S suite                          *" <<endl;
  cout << "******************************************************************" <<endl;

  gVerbose = verbose;
  stress_allocation();
  stress_element_op(20);
  stress_binary_op(20);
  stress_norms(20);
  stress_matrix_slices(20);
  stress_vector_io();

  cout << "******************************************************************" <<endl;
}

void StatusPrint(Int_t id,const Char_t *title,Bool_t status)
{
  // Print test program number and its title
//   const Int_t kMAX = 65;
//   TString header = TString("Test ")+Form("%2d",id)+" : "+title;
//   const Int_t nch = header.Length();
//   for (Int_t i = nch; i < kMAX; i++) header += '.';
//   cout << header << (status ? "OK" : "FAILED") << endl;
  // Print test program number and its title
   const Int_t kMAX = 65;
   char header[80];
   sprintf(header,"Test %2d : %s",id,title);
   Int_t nch = strlen(header);
   for (Int_t i=nch;i<kMAX;i++) header[i] = '.';
   header[kMAX] = 0;
   header[kMAX-1] = ' ';
   cout << header << (status ? "OK" : "FAILED") << endl;
}

//------------------------------------------------------------------------
//          Test allocation functions and compatibility check
//

void stress_allocation()
{
  if (gVerbose)
    cout << "\n\n---> Test allocation and compatibility check" << endl;

  Bool_t ok = kTRUE;
  TVectorD v1(20);
  TVectorD v2(0,19);
  TVectorD v3(1,20);
  TVectorD v4(v1);

  if (gVerbose) {
    cout << "\nStatus information reported for vector v3:" << endl;
    cout << "  Lower bound ... " << v3.GetLwb() << endl;
    cout << "  Upper bound ... " << v3.GetUpb() << endl;
    cout << "  No. of elements " << v3.GetNoElements() << endl;
  }

  if (gVerbose)
    cout << "\nCheck vectors 1 & 2 for compatibility" << endl;
  ok &= AreCompatible(v1,v2,gVerbose);

  if (gVerbose)
    cout << "Check vectors 1 & 4 for compatibility" << endl;
  ok &= AreCompatible(v1,v4,gVerbose);

  if (gVerbose)
    cout << "v2 has to be compatible with v3 after resizing to v3" << endl;
  v2.ResizeTo(v3);
  ok &= AreCompatible(v2,v3,gVerbose);

  TVectorD v5(v1.GetUpb()+5);
  if (gVerbose)
    cout << "v1 has to be compatible with v5 after resizing to v5.upb" << endl;
  v1.ResizeTo(v5.GetNoElements());
  ok &= AreCompatible(v1,v5,gVerbose);

  {
    if (gVerbose)
      cout << "Check that shrinking does not change remaining elements" << endl;
    TVectorD vb(-1,20);
    Int_t i;
    for (i = vb.GetLwb(); i <= vb.GetUpb(); i++)
      vb(i) = i+TMath::Pi();
    TVectorD v = vb;
    ok &= ( v == vb ) ? kTRUE : kFALSE;
    ok &= ( v != 0 ) ? kTRUE : kFALSE;
    v.ResizeTo(0,10);
    for (i = v.GetLwb(); i <= v.GetUpb(); i++)
      ok &= ( v(i) == vb(i) ) ? kTRUE : kFALSE;
    if (gVerbose)
      cout << "Check that expansion expands by zeros" << endl;
    const Int_t old_nelems = v.GetNoElements();
    const Int_t old_lwb    = v.GetLwb();
    v.ResizeTo(vb);
    ok &= ( !(v == vb) ) ? kTRUE : kFALSE;
    for (i = old_lwb; i < old_lwb+old_nelems; i++)
      ok &= ( v(i) == vb(i) ) ? kTRUE : kFALSE;
    for (i = v.GetLwb(); i < old_lwb; i++)
      ok &= ( v(i) == 0 ) ? kTRUE : kFALSE;
    for (i = old_lwb+old_nelems; i <= v.GetUpb(); i++)
      ok &= ( v(i) == 0 ) ? kTRUE : kFALSE;
  }

  if (gVerbose)
    cout << "\nDone\n" << endl;
  StatusPrint(1,"Allocation, Filling, Resizing",ok);
}

//
//------------------------------------------------------------------------
//                Test uniform element operations
//
class SinAction : public TElementActionD {
  void Operation(Double_t &element) const { element = TMath::Sin(element); }
  public:
    SinAction() { }
};

class CosAction : public TElementPosActionD {
  Double_t factor;
  void Operation(Double_t &element) const { element = TMath::Cos(factor*fI); }
  public:
    CosAction() { }
    CosAction(Int_t no_elems): factor(2*TMath::Pi()/no_elems) { }
};

void stress_element_op(Int_t vsize)
{
  if (gVerbose)
    cout << "\n---> Test operations that treat each element uniformly" << endl;

  Bool_t ok = kTRUE;
  const double pattern = TMath::Pi();

  TVectorD v(-1,vsize-2);
  TVectorD v1(v);

  if (gVerbose)
    cout << "\nWriting zeros to v..." << endl;
  for (Int_t i = v.GetLwb(); i <= v.GetUpb(); i++)
    v(i) = 0;
  ok &= VerifyVectorValue(v,0.0,gVerbose,EPSILON);

  if (gVerbose)
    cout << "Clearing v1 ..." << endl;
  v1.Zero();
  ok &= VerifyVectorValue(v1,0.0,gVerbose,EPSILON);

  if (gVerbose)
    cout << "Comparing v1 with 0 ..." << endl;
  ok &= (v1 == 0) ? kTRUE : kFALSE;

  if (gVerbose)
    cout << "Writing a pattern " << pattern << " by assigning to v(i)..." << endl;
  {
    for (Int_t i = v.GetLwb(); i <= v.GetUpb(); i++)
      v(i) = pattern;
    ok &= VerifyVectorValue(v,pattern,gVerbose,EPSILON);
  }

  if (gVerbose)
    cout << "Writing the pattern by assigning to v1 as a whole ..." << endl;
  v1 = pattern;
  ok &= VerifyVectorValue(v1,pattern,gVerbose,EPSILON);

  if (gVerbose)
    cout << "Comparing v and v1 ..." << endl;
  ok &= (v == v1) ? kTRUE : kFALSE;
  if (gVerbose)
    cout << "Comparing (v=0) and v1 ..." << endl;
  ok &= (!(v.Zero() == v1)) ? kTRUE : kFALSE;

  if (gVerbose)
    cout << "\nClear v and add the pattern" << endl;
  v.Zero();
  v += pattern;
  ok &= VerifyVectorValue(v,pattern,gVerbose,EPSILON);
  if (gVerbose)
    cout << "   add the doubled pattern with the negative sign" << endl;
  v += -2*pattern;
  ok &= VerifyVectorValue(v,-pattern,gVerbose,EPSILON);
  if (gVerbose)
    cout << "   subtract the trippled pattern with the negative sign" << endl;
  v -= -3*pattern;
  ok &= VerifyVectorValue(v,2*pattern,gVerbose,EPSILON);

  if (gVerbose)
    cout << "\nVerify comparison operations" << endl;
  v = pattern;
  ok &= ( v == pattern && !(v != pattern) && v >= pattern && v <= pattern ) ? kTRUE : kFALSE;
  ok &= ( v > 0 && v >= 0 ) ? kTRUE : kFALSE;
  ok &= ( v > -pattern && v >= -pattern ) ? kTRUE : kFALSE;
  ok &= ( v < pattern+1 && v <= pattern+1 ) ? kTRUE : kFALSE;
  v(v.GetUpb()) += 1;
  ok &= ( !(v==pattern)      && !(v != pattern)  && v != pattern-1 ) ? kTRUE : kFALSE;
  ok &= ( v >= pattern       && !(v > pattern)   && !(v >= pattern+1) ) ? kTRUE : kFALSE;
  ok &= ( v <= pattern+1.001 && !(v < pattern+1) && !(v <= pattern) ) ? kTRUE : kFALSE;

  if (gVerbose)
    cout << "\nAssign 2*pattern to v by repeating additions" << endl;
  v = 0; v += pattern; v += pattern;
  if (gVerbose)
    cout << "Assign 2*pattern to v1 by multiplying by two" << endl;
  v1 = pattern; v1 *= 2;
  ok &= VerifyVectorValue(v1,2*pattern,gVerbose,EPSILON);
  ok &= ( v == v1 ) ? kTRUE : kFALSE;
  if (gVerbose)
    cout << "Multiply v1 by one half returning it to the 1*pattern" << endl;
  v1 *= 1/2.;
  ok &= VerifyVectorValue(v1,pattern,gVerbose,EPSILON);

  if (gVerbose)
    cout << "\nAssign -pattern to v and v1" << endl;
  v.Zero(); v -= pattern; v1 = -pattern;
  ok &= VerifyVectorValue(v,-pattern,gVerbose,EPSILON);
  ok &= ( v == v1 ) ? kTRUE : kFALSE;
  if (gVerbose)
    cout << "v = sqrt(sqr(v)); v1 = abs(v1); Now v and v1 have to be the same" << endl;
  v.Sqr();
  ok &= VerifyVectorValue(v,pattern*pattern,gVerbose,EPSILON);
  v.Sqrt();
  ok &= VerifyVectorValue(v,pattern,gVerbose,EPSILON);
  v1.Abs();
  ok &= VerifyVectorValue(v1,pattern,gVerbose,EPSILON);
  ok &= ( v == v1 ) ? kTRUE : kFALSE;

  {
    if (gVerbose)
      cout << "\nCheck out to see that sin^2(x) + cos^2(x) = 1" << endl;
    for (Int_t i = v.GetLwb(); i <= v.GetUpb(); i++)
      v(i) = 2*TMath::Pi()/v.GetNoElements()*i;
#ifndef __CINT__
    SinAction s;
    v.Apply(s);
    CosAction c(v.GetNoElements());
    v1.Apply(c);
#else
    for (Int_t i = v.GetLwb(); i <= v.GetUpb(); i++) {
      v(i)  = TMath::Sin(v(i));
      v1(i) = TMath::Cos(2*TMath::Pi()/v1.GetNrows()*i);
    }
#endif
    TVectorD v2 = v;
    TVectorD v3 = v1;
    v.Sqr();
    v1.Sqr();
    v += v1;
    ok &= VerifyVectorValue(v,1.,gVerbose,EPSILON);
  }

  if (gVerbose)
    cout << "\nVerify constructor with initialization" << endl;
#ifndef __CINT__
  TVectorD vi(0,4,0.0,1.0,2.0,3.0,4.0,"END");
#else
  Double_t vval[] = {0.0,1.0,2.0,3.0,4.0};
  TVectorD vi(5,vval);
#endif
  TVectorD vit(5);
  {
    for (Int_t i = vit.GetLwb(); i <= vit.GetUpb(); i++)
      vit(i) = Double_t(i);
    ok &= VerifyVectorIdentity(vi,vit,gVerbose,EPSILON);
  }

  if (gVerbose)
    cout << "\nDone\n" << endl;
  StatusPrint(2,"Uniform vector operations",ok);
}

//
//------------------------------------------------------------------------
//                 Test binary vector operations
//
void stress_binary_op(Int_t vsize)
{
  if (gVerbose)
    cout << "\n---> Test Binary Vector operations" << endl;

  Bool_t ok = kTRUE;
  const double pattern = TMath::Pi();

  const Double_t epsilon = EPSILON*vsize/10;

  TVectorD v(2,vsize+1);
  TVectorD v1(v);

  if (gVerbose)
    cout << "\nVerify assignment of a vector to the vector" << endl;
  v = pattern;
  v1.Zero();
  v1 = v;
  ok &= VerifyVectorValue(v1,pattern,gVerbose,EPSILON);
  ok &= ( v1 == v ) ? kTRUE : kFALSE;

  if (gVerbose)
    cout << "\nAdding one vector to itself, uniform pattern " << pattern << endl;
  v.Zero(); v = pattern;
  v1 = v; v1 += v1;
  ok &= VerifyVectorValue(v1,2*pattern,gVerbose,EPSILON);
  if (gVerbose)
    cout << "  subtracting two vectors ..." << endl;
  v1 -= v;
  ok &= VerifyVectorValue(v1,pattern,gVerbose,EPSILON);
  if (gVerbose)
    cout << "  subtracting the vector from itself" << endl;
  v1 -= v1;
  ok &= VerifyVectorValue(v1,0.,gVerbose,EPSILON);
  if (gVerbose)
    cout << "  adding two vectors together" << endl;
  v1 += v;
  ok &= VerifyVectorValue(v1,pattern,gVerbose,EPSILON);

  TVectorD vp(2,vsize+1);
  {
    for (Int_t i = vp.GetLwb(); i <= vp.GetUpb(); i++)
      vp(i) = (i-vp.GetNoElements()/2.)*pattern;
  }

  if (gVerbose) {
    cout << "\nArithmetic operations on vectors with not the same elements" << endl;
    cout << "   adding vp to the zero vector..." << endl;
  }
  v.Zero();
  ok &= ( v == 0.0 ) ? kTRUE : kFALSE;
  v += vp;
  ok &= VerifyVectorIdentity(v,vp,gVerbose,epsilon);
//  ok &= ( v == vp ) ? kTRUE : kFALSE;
  v1 = v;
  if (gVerbose)
    cout << "   making v = 3*vp and v1 = 3*vp, via add() and succesive mult" << endl;
  Add(v,2.,vp);
  v1 += v1; v1 += vp;
  ok &= VerifyVectorIdentity(v,v1,gVerbose,epsilon);
  if (gVerbose)
    cout << "   clear both v and v1, by subtracting from itself and via add()" << endl;
  v1 -= v1;
  Add(v,-3.,vp);
  ok &= VerifyVectorIdentity(v,v1,gVerbose,epsilon);

  if (gVerbose) {
    cout << "\nTesting element-by-element multiplications and divisions" << endl;
    cout << "   squaring each element with sqr() and via multiplication" << endl;
  }
  v = vp; v1 = vp;
  v.Sqr();
  ElementMult(v1,v1);
  ok &= VerifyVectorIdentity(v,v1,gVerbose,epsilon);
  if (gVerbose)
    cout << "   compare (v = pattern^2)/pattern with pattern" << endl;
  v = pattern; v1 = pattern;
  v.Sqr();
  ElementDiv(v,v1);
  ok &= VerifyVectorValue(v,pattern,gVerbose,epsilon);
  if (gVerbose)
   Compare(v1,v);

  if (gVerbose)
    cout << "\nDone\n" << endl;
  StatusPrint(3,"Binary vector element-by-element operations",ok);
}

//
//------------------------------------------------------------------------
//               Verify the norm calculation
//
void stress_norms(Int_t vsize)
{
  if (gVerbose)
    cout << "\n---> Verify norm calculations" << endl;

  Bool_t ok = kTRUE;
  const double pattern = 10.25;

  if ( vsize % 2 == 1 )
    Fatal("stress_norms", "size of the vector to test must be even for this test\n");

  TVectorD v(vsize);
  TVectorD v1(v);

  if (gVerbose)
    cout << "\nAssign " << pattern << " to all the elements and check norms" << endl;
  v = pattern;
  if (gVerbose)
    cout << "  1. norm should be pattern*no_elems" << endl;
  ok &= ( v.Norm1() == pattern*v.GetNoElements() ) ? kTRUE : kFALSE;
  if (gVerbose)
    cout << "  Square of the 2. norm has got to be pattern^2 * no_elems" << endl;
  ok &= ( v.Norm2Sqr() == (pattern*pattern)*v.GetNoElements() ) ? kTRUE : kFALSE;
  if (gVerbose)
    cout << "  Inf norm should be pattern itself" << endl;
  ok &= ( v.NormInf() == pattern ) ? kTRUE : kFALSE;
  if (gVerbose)
    cout << "  Scalar product of vector by itself is the sqr(2. vector norm)" << endl;
  ok &= ( v.Norm2Sqr() == v*v ) ? kTRUE : kFALSE;

  Double_t ap_step = 1;
  Double_t ap_a0   = -pattern;
  Int_t n = v.GetNoElements();
  if (gVerbose) {
    cout << "\nAssign the arithm progression with 1. term " << ap_a0 <<
            "\nand the difference " << ap_step << endl;
  }
  {
    for (Int_t i = v.GetLwb(); i <= v.GetUpb(); i++)
      v(i) = (i-v.GetLwb())*ap_step + ap_a0;
  }
  Int_t l = TMath::Min(TMath::Max((int)TMath::Ceil(-ap_a0/ap_step),0),n);
  Double_t norm = (2*ap_a0+(l+n-1)*ap_step)/2*(n-l) +
                  (-2*ap_a0-(l-1)*ap_step)/2*l;
  if (gVerbose)
    cout << "  1. norm should be " << norm << endl;
  ok &= ( v.Norm1() == norm ) ? kTRUE : kFALSE;
  norm = n*( (ap_a0*ap_a0)+ap_a0*ap_step*(n-1)+(ap_step*ap_step)*(n-1)*(2*n-1)/6);
  if (gVerbose) {
    cout << "  Square of the 2. norm has got to be "
            "n*[ a0^2 + a0*q*(n-1) + q^2/6*(n-1)*(2n-1) ], or " << norm << endl;
  }
  ok &= ( TMath::Abs( (v.Norm2Sqr()-norm)/norm ) < 1e-15 ) ? kTRUE : kFALSE;

  norm = TMath::Max(TMath::Abs(v(v.GetLwb())),TMath::Abs(v(v.GetUpb())));
  if (gVerbose)
    cout << "  Inf norm should be max(abs(a0),abs(a0+(n-1)*q)), ie " << norm << endl;
  ok &= ( v.NormInf() == norm ) ? kTRUE : kFALSE;
  if (gVerbose)
    cout << "  Scalar product of vector by itself is the sqr(2. vector norm)" << endl;
  ok &= ( v.Norm2Sqr() == v*v ) ? kTRUE : kFALSE;

#if 0
  v1.Zero();
  Compare(v,v1);  // they are not equal (of course)
#endif

  if (gVerbose)
    cout << "\nConstruct v1 to be orthogonal to v as v(n), -v(n-1), v(n-2)..." << endl;
  {
    for (Int_t i = 0; i < v1.GetNoElements(); i++)
      v1(i+v1.GetLwb()) = v(v.GetUpb()-i) * ( i % 2 == 1 ? -1 : 1 );
  }
  if (gVerbose)
    cout << "||v1|| has got to be equal ||v|| regardless of the norm def" << endl;
  ok &= ( v1.Norm1()    == v.Norm1() ) ? kTRUE : kFALSE;
  ok &= ( v1.Norm2Sqr() == v.Norm2Sqr() ) ? kTRUE : kFALSE;
  ok &= ( v1.NormInf()  == v.NormInf() ) ? kTRUE : kFALSE;
  if (gVerbose)
    cout << "But the scalar product has to be zero" << endl;
  ok &= ( v1 * v == 0 ) ? kTRUE : kFALSE;

  if (gVerbose)
    cout << "\nDone\n" << endl;
  StatusPrint(4,"Vector Norms",ok);
}

//
//------------------------------------------------------------------------
//           Test operations with vectors and matrix slices
//
void stress_matrix_slices(Int_t vsize)
{
  if (gVerbose)
    cout << "\n---> Test operations with vectors and matrix slices" << endl;

  Bool_t ok = kTRUE;
  const Double_t pattern = 8.625;

  TVectorD vc(0,vsize);
  TVectorD vr(0,vsize+1);
  TMatrixD m(0,vsize,0,vsize+1);

  Int_t i,j;
  if (gVerbose)
    cout << "\nCheck modifying the matrix column-by-column" << endl;
  m = pattern;
  ok &= ( m == pattern ) ? kTRUE : kFALSE;
  for (i = m.GetColLwb(); i <= m.GetColUpb(); i++) {
    TMatrixDColumn(m,i) = pattern-1;
    ok &= ( !( m == pattern ) && !( m != pattern ) ) ? kTRUE : kFALSE;
    TMatrixDColumn(m,i) *= 2;
    vc = TMatrixDColumn(m,i);
    ok &= VerifyVectorValue(vc,2*(pattern-1),gVerbose);
    vc = TMatrixDColumn(m,i+1 > m.GetColUpb() ? m.GetColLwb() : i+1);
    ok &= VerifyVectorValue(vc,pattern,gVerbose,EPSILON);
    TMatrixDColumn(m,i) *= 0.5;
    TMatrixDColumn(m,i) += 1;
    ok &= ( m == pattern ) ? kTRUE : kFALSE;
  }

  ok &= ( m == pattern ) ? kTRUE : kFALSE;
  for (i = m.GetColLwb(); i <= m.GetColUpb(); i++) {
    vc = pattern+1;
    TMatrixDColumn(m,i) = vc;
    ok &= ( !( m == pattern ) && !( m != pattern ) ) ? kTRUE : kFALSE;
    {
      TMatrixDColumn mc(m,i);
      for (j = m.GetRowLwb(); j <= m.GetRowUpb(); j++)
        mc(j) *= 4;
    }
    vc = TMatrixDColumn(m,i);
    ok &= VerifyVectorValue(vc,4*(pattern+1),gVerbose,EPSILON);
    TMatrixDColumn(m,i) *= 0.25;
    TMatrixDColumn(m,i) += -1;
    vc = TMatrixDColumn(m,i);
    ok &= VerifyVectorValue(vc,pattern,gVerbose,EPSILON);
    ok &= ( m == pattern ) ? kTRUE : kFALSE;
  }

  if (gVerbose)
    cout << "\nCheck modifying the matrix row-by-row" << endl;
  m = pattern;
  ok &= ( m == pattern ) ? kTRUE : kFALSE;
  for (i = m.GetRowLwb(); i <= m.GetRowUpb(); i++) {
    TMatrixDRow(m,i) = pattern+2;
    ok &= ( !( m == pattern ) && !( m != pattern ) ) ? kTRUE : kFALSE;
    vr = TMatrixDRow(m,i);
    ok &= VerifyVectorValue(vr,pattern+2,gVerbose,EPSILON);
    vr = TMatrixDRow(m,i+1 > m.GetRowUpb() ? m.GetRowLwb() : i+1);
    ok &= VerifyVectorValue(vr,pattern,gVerbose,EPSILON);
    TMatrixDRow(m,i) += -2;
    ok &= ( m == pattern ) ? kTRUE : kFALSE;
  }

  ok &= ( m == pattern ) ? kTRUE : kFALSE;
  for (i = m.GetRowLwb(); i <= m.GetRowUpb(); i++) {
    vr = pattern-2;
    TMatrixDRow(m,i) = vr;
    ok &= ( !( m == pattern ) && !( m != pattern ) ) ? kTRUE : kFALSE;
    {
      TMatrixDRow mr(m,i);
      for (j = m.GetColLwb(); j <= m.GetColUpb(); j++)
        mr(j) *= 8;
    }
    vr = TMatrixDRow(m,i);
    ok &= VerifyVectorValue(vr,8*(pattern-2),gVerbose,EPSILON);
    TMatrixDRow(m,i) *= 1./8;
    TMatrixDRow(m,i) += 2;
    vr = TMatrixDRow(m,i);
    ok &= VerifyVectorValue(vr,pattern,gVerbose,EPSILON);
    ok &= ( m == pattern ) ? kTRUE : kFALSE;
  }

  if (gVerbose)
    cout << "\nCheck modifying the matrix diagonal" << endl;
  m = pattern;
  TMatrixDDiag td = m;
  td = pattern-3;
  ok &= ( !( m == pattern ) && !( m != pattern ) ) ? kTRUE : kFALSE;
  vc = TMatrixDDiag(m);
  ok &= VerifyVectorValue(vc,pattern-3,gVerbose,EPSILON);
  td += 3;
  ok &= ( m == pattern ) ? kTRUE : kFALSE;
  vc = pattern+3;
  td = vc;
  ok &= ( !( m == pattern ) && !( m != pattern ) ) ? kTRUE : kFALSE;
  {
    TMatrixDDiag md(m);
    for (j = 0; j < md.GetNdiags(); j++)
      md(j) /= 1.5;
  }
  vc = TMatrixDDiag(m);
  ok &= VerifyVectorValue(vc,(pattern+3)/1.5,gVerbose,EPSILON);
  TMatrixDDiag(m) *= 1.5;
  TMatrixDDiag(m) += -3;
  vc = TMatrixDDiag(m);
  ok &= VerifyVectorValue(vc,pattern,gVerbose,EPSILON);
  ok &= ( m == pattern ) ? kTRUE : kFALSE;

  if (gVerbose) {
    cout << "\nCheck out to see that multiplying by diagonal is column-wise"
            "\nmatrix multiplication" << endl;
  }
  TMatrixD mm(m);
  TMatrixD m1(m.GetRowLwb(),TMath::Max(m.GetRowUpb(),m.GetColUpb()),
              m.GetColLwb(),TMath::Max(m.GetRowUpb(),m.GetColUpb()));
  TVectorD vc1(vc),vc2(vc);
  for (i = m.GetRowLwb(); i < m.GetRowUpb(); i++)
    TMatrixDRow(m,i) = pattern+i;      // Make a multiplicand
  mm = m;                          // Save it

  m1 = pattern+10;
  for (i = vr.GetLwb(); i <= vr.GetUpb(); i++)
    vr(i) = i+2;
  TMatrixDDiag td2 = m1;
  td2 = vr;
  ok &= ( !(m1 == pattern+10) ) ? kTRUE : kFALSE;

  m *= TMatrixDDiag(m1);
  for (i = m.GetColLwb(); i <= m.GetColUpb(); i++) {
    vc1 = TMatrixDColumn(mm,i);
    vc1 *= vr(i);                    // Do a column-wise multiplication
    vc2 = TMatrixDColumn(m,i);
    ok &= VerifyVectorIdentity(vc1,vc2,gVerbose,EPSILON);
  }

  if (gVerbose)
    cout << "\nDone\n" << endl;
  StatusPrint(5,"Matrix Slices to Vectors",ok);
}

//
//------------------------------------------------------------------------
//           Test vector I/O
//
void stress_vector_io()
{
  if (gVerbose)
    cout << "\n---> Test vector I/O" << endl;

  Bool_t ok = kTRUE;
  const double pattern = TMath::Pi();

  TVectorD v(40);
  v = pattern;

  if (gVerbose)
    cout << "\nWrite vector v to database" << endl;

  TFile *f = new TFile("vvector.root","RECREATE");

  v.Write("v");

  if (gVerbose)
    cout << "\nClose database" << endl;
  delete f;

  if (gVerbose)
    cout << "\nOpen database in read-only mode and read vector" << endl;
  TFile *f1 = new TFile("vvector.root");

  TVectorD *vr = (TVectorD*) f1->Get("v");

  if (gVerbose)
    cout << "\nRead vector should be same as original still in memory" << endl;
  ok &= ((*vr) == v) ? kTRUE : kFALSE;

  delete f1;

  if (gVerbose)
    cout << "\nDone\n" << endl;
  StatusPrint(6,"Vector Persistence",ok);
}
