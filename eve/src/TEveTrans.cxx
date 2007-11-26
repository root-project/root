// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//______________________________________________________________________________
// TEveTrans
//
// TEveTrans is a 4x4 transformation matrix for homogeneous coordinates
// stored internaly in a column-major order to allow direct usage by
// GL. The element type is Double32_t as statically the floats would
// be precise enough but continuous operations on the matrix must
// retain precision of column vectors.
//
// Cartan angles in mA[1-3] (+z, -y, +x) are stored for backward
// compatibility and will probably be removed soon.
//
// Direct  element access (first two should be used with care):
// operator[i]    direct access to elements,   i:0->15
// CM(i,j)        element 4*j + i;           i,j:0->3    { CM ~ c-matrix }
// operator(i,j)  element 4*(j-1) + i - 1    i,j:1->4
//
// Column-vector access:
// USet Get/SetBaseVec(), Get/SetPos() and Arr[XYZT]() methods.
//
// For all methods taking the matrix indices:
// 1->X, 2->Y, 3->Z; 4->Position (if applicable). 0 reserved for time.
//
// Shorthands in method-names:
// LF ~ LocalFrame; PF ~ ParentFrame; IP ~ InPlace

#include <TEveTrans.h>
#include <TEveUtil.h>
#include <TMath.h>
#include <TClass.h>

#include <ctype.h>

#define F00  0
#define F01  4
#define F02  8
#define F03 12

#define F10  1
#define F11  5
#define F12  9
#define F13 13

#define F20  2
#define F21  6
#define F22 10
#define F23 14

#define F30  3
#define F31  7
#define F32 11
#define F33 15

ClassImp(TEveTrans)

//______________________________________________________________________________
TEveTrans::TEveTrans() :
   TObject(),
   mA1(0), mA2(0), mA3(0), bAsOK(kFALSE),
   fUseTrans (kTRUE),
   fEditTrans(kFALSE)
{
   UnitTrans();
}

//______________________________________________________________________________
TEveTrans::TEveTrans(const TEveTrans& t) :
   TObject(),
   mA1(t.mA1), mA2(t.mA2), mA3(t.mA3), bAsOK(t.bAsOK),
   fUseTrans (t.fUseTrans),
   fEditTrans(t.fEditTrans)
{
   SetTrans(t, kFALSE);
}

//______________________________________________________________________________
TEveTrans::TEveTrans(const Double_t arr[16]) :
   TObject(),
   mA1(0), mA2(0), mA3(0), bAsOK(kFALSE),
   fUseTrans (kTRUE),
   fEditTrans(kFALSE)
{
   SetFromArray(arr);
}

//______________________________________________________________________________
TEveTrans::TEveTrans(const Float_t  arr[16]) :
   TObject(),
   mA1(0), mA2(0), mA3(0), bAsOK(kFALSE),
   fUseTrans (kTRUE),
   fEditTrans(kFALSE)
{
   SetFromArray(arr);
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrans::UnitTrans()
{
   // Reset matrix to unity.

   memset(M, 0, 16*sizeof(Double_t));
   M[F00] = M[F11] = M[F22] = M[F33] = 1;
   mA1 = mA2 = mA3 = 0;
   bAsOK = kTRUE;
}

//______________________________________________________________________________
void TEveTrans::UnitRot()
{
   // Reset rotation part of the matrix to unity.

   memset(M, 0, 12*sizeof(Double_t));
   M[F00] = M[F11] = M[F22] = 1;
   mA1 = mA2 = mA3 = 0;
   bAsOK = kTRUE;
}

//______________________________________________________________________________
void TEveTrans::SetTrans(const TEveTrans& t, Bool_t copyAngles)
{
   memcpy(M, t.M, sizeof(M));
   if (copyAngles && t.bAsOK) {
      bAsOK = kTRUE;
      mA1 = t.mA1; mA2 = t.mA2; mA3 = t.mA3;
   } else {
      bAsOK = kFALSE;
   }
}

//______________________________________________________________________________
void TEveTrans::SetFromArray(const Double_t arr[16])
{
   for(Int_t i=0; i<16; ++i) M[i] = arr[i];
   bAsOK = false;
}

//______________________________________________________________________________
void TEveTrans::SetFromArray(const Float_t  arr[16])
{
   for(Int_t i=0; i<16; ++i) M[i] = arr[i];
   bAsOK = false;
}

//______________________________________________________________________________
void TEveTrans::SetupRotation(Int_t i, Int_t j, Double_t f)
{
   // Setup the matrix as an elementary rotation.
   // Optimized versions of left/right multiplication with an elementary
   // rotation matrix are implemented in RotatePF/RotateLF.
   // Expects identity matrix.

   if(i == j) return;
   TEveTrans& M = *this;
   M(i,i) = M(j,j) = TMath::Cos(f);
   Double_t s = TMath::Sin(f);
   M(i,j) = -s; M(j,i) = s;
   bAsOK = kFALSE;
}

/******************************************************************************/

// OrtoNorm3 and Invert are near the bottom.

/******************************************************************************/

//______________________________________________________________________________
void TEveTrans::MultLeft(const TEveTrans& t)
{
   Double_t  B[4];
   Double_t* C = M;
   for(int c=0; c<4; ++c, C+=4) {
      const Double_t* T = t.M;
      for(int r=0; r<4; ++r, ++T)
         B[r] = T[0]*C[0] + T[4]*C[1] + T[8]*C[2] + T[12]*C[3];
      C[0] = B[0]; C[1] = B[1]; C[2] = B[2]; C[3] = B[3];
   }
   bAsOK = kFALSE;
}

//______________________________________________________________________________
void TEveTrans::MultRight(const TEveTrans& t)
{
   Double_t  B[4];
   Double_t* C = M;
   for(int r=0; r<4; ++r, ++C) {
      const Double_t* T = t.M;
      for(int c=0; c<4; ++c, T+=4)
         B[c] = C[0]*T[0] + C[4]*T[1] + C[8]*T[2] + C[12]*T[3];
      C[0] = B[0]; C[4] = B[1]; C[8] = B[2]; C[12] = B[3];
   }
   bAsOK = kFALSE;
}

//______________________________________________________________________________
TEveTrans TEveTrans::operator*(const TEveTrans& t)
{
   TEveTrans b(*this);
   b.MultRight(t);
   return b;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrans::TransposeRotationPart()
{
   Double_t x;
   x = M[F01]; M[F01] = M[F10]; M[F10] = x;
   x = M[F02]; M[F02] = M[F20]; M[F20] = x;
   x = M[F12]; M[F12] = M[F21]; M[F21] = x;
   bAsOK = false;
}

/******************************************************************************/
// Move & Rotate
/******************************************************************************/

//______________________________________________________________________________
void TEveTrans::MoveLF(Int_t ai, Double_t amount)
{
   const Double_t *C = M + 4*--ai;
   M[F03] += amount*C[0]; M[F13] += amount*C[1]; M[F23] += amount*C[2];
}

//______________________________________________________________________________
void TEveTrans::Move3LF(Double_t x, Double_t y, Double_t z)
{
   M[F03] += x*M[0] + y*M[4] + z*M[8];
   M[F13] += x*M[1] + y*M[5] + z*M[9];
   M[F23] += x*M[2] + y*M[6] + z*M[10];
}

//______________________________________________________________________________
void TEveTrans::RotateLF(Int_t i1, Int_t i2, Double_t amount)
{
   // Rotate in local frame. Does optimised version of MultRight.

   if(i1 == i2) return;
   // Algorithm: TEveTrans a; a.SetupRotation(i1, i2, amount); MultRight(a);
   // Optimized version:
   const Double_t cos = TMath::Cos(amount), sin = TMath::Sin(amount);
   Double_t  b1, b2;
   Double_t* C = M;
   --i1 <<= 2; --i2 <<= 2; // column major
   for(int r=0; r<4; ++r, ++C) {
      b1 = cos*C[i1] + sin*C[i2];
      b2 = cos*C[i2] - sin*C[i1];
      C[i1] = b1; C[i2] = b2;
   }
   bAsOK = kFALSE;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrans::MovePF(Int_t ai, Double_t amount)
{
   M[F03 + --ai] += amount;
}

//______________________________________________________________________________
void TEveTrans::Move3PF(Double_t x, Double_t y, Double_t z)
{
   M[F03] += x;
   M[F13] += y;
   M[F23] += z;
}

//______________________________________________________________________________
void TEveTrans::RotatePF(Int_t i1, Int_t i2, Double_t amount)
{
   // Rotate in parent frame. Does optimised version of MultLeft.

   if(i1 == i2) return;
   // Algorithm: TEveTrans a; a.SetupRotation(i1, i2, amount); MultLeft(a);

   // Optimized version:
   const Double_t cos = TMath::Cos(amount), sin = TMath::Sin(amount);
   Double_t  b1, b2;
   Double_t* C = M;
   --i1; --i2;
   for(int c=0; c<4; ++c, C+=4) {
      b1 = cos*C[i1] - sin*C[i2];
      b2 = cos*C[i2] + sin*C[i1];
      C[i1] = b1; C[i2] = b2;
   }
   bAsOK = kFALSE;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrans::Move(const TEveTrans& a, Int_t ai, Double_t amount)
{
   const Double_t* A = a.M + 4*--ai;
   M[F03] += amount*A[0];
   M[F13] += amount*A[1];
   M[F23] += amount*A[2];
}

//______________________________________________________________________________
void TEveTrans::Move3(const TEveTrans& a, Double_t x, Double_t y, Double_t z)
{
   const Double_t* A = a.M;
   M[F03] += x*A[F00] + y*A[F01] + z*A[F02];
   M[F13] += x*A[F10] + y*A[F11] + z*A[F12];
   M[F23] += x*A[F20] + y*A[F21] + z*A[F22];
}

//______________________________________________________________________________
void TEveTrans::Rotate(const TEveTrans& a, Int_t i1, Int_t i2, Double_t amount)
{
   if(i1 == i2) return;
   TEveTrans X(a);
   X.Invert();
   MultLeft(X);
   RotatePF(i1, i2, amount);
   MultLeft(a);
   bAsOK = kFALSE;
}

/******************************************************************************/
// Base-vector interface
/******************************************************************************/

//______________________________________________________________________________
void TEveTrans::SetBaseVec(Int_t b, Double_t x, Double_t y, Double_t z)
{
   Double_t* C = M + 4*--b;
   C[0] = x; C[1] = y; C[2] = z;
   bAsOK = kFALSE;
}

//______________________________________________________________________________
void TEveTrans::SetBaseVec(Int_t b, const TVector3& v)
{
   Double_t* C = M + 4*--b;
   v.GetXYZ(C);
   bAsOK = kFALSE;
}

//______________________________________________________________________________
TVector3 TEveTrans::GetBaseVec(Int_t b) const
{ return TVector3(&M[4*--b]); }

void TEveTrans::GetBaseVec(Int_t b, TVector3& v) const
{
   const Double_t* C = M + 4*--b;
   v.SetXYZ(C[0], C[1], C[2]);
}

/******************************************************************************/
// Position interface
/******************************************************************************/

//______________________________________________________________________________
void TEveTrans::SetPos(Double_t x, Double_t y, Double_t z)
{ M[F03] = x; M[F13] = y; M[F23] = z; }

void TEveTrans::SetPos(Double_t* x)
{ M[F03] = x[0]; M[F13] = x[1]; M[F23] = x[2]; }

void TEveTrans::SetPos(Float_t* x)
{ M[F03] = x[0]; M[F13] = x[1]; M[F23] = x[2]; }

void TEveTrans::SetPos(const TEveTrans& t)
{
   const Double_t* T = t.M;
   M[F03] = T[F03]; M[F13] = T[F13]; M[F23] = T[F23];
}

//______________________________________________________________________________
void TEveTrans::GetPos(Double_t& x, Double_t& y, Double_t& z) const
{ x = M[F03]; y = M[F13]; z = M[F23]; }

void TEveTrans::GetPos(Double_t* x) const
{ x[0] = M[F03]; x[1] = M[F13]; x[2] = M[F23]; }

void TEveTrans::GetPos(Float_t* x) const
{ x[0] = M[F03]; x[1] = M[F13]; x[2] = M[F23]; }

void TEveTrans::GetPos(TVector3& v) const
{ v.SetXYZ(M[F03], M[F13], M[F23]); }

TVector3 TEveTrans::GetPos() const
{ return TVector3(M[F03], M[F13], M[F23]); }

/******************************************************************************/
// Cardan angle interface
/******************************************************************************/

namespace {
inline void clamp_angle(Float_t& a) {
   while(a < -TMath::TwoPi()) a += TMath::TwoPi();
   while(a >  TMath::TwoPi()) a -= TMath::TwoPi();
}
}

void TEveTrans::SetRotByAngles(Float_t a1, Float_t a2, Float_t a3)
{
   // Sets Rotation part as given by angles:
   // a1 around z, -a2 around y, a3 around x
   clamp_angle(a1); clamp_angle(a2); clamp_angle(a3);

   Double_t A, B, C, D, E, F;
   A = TMath::Cos(a3); B = TMath::Sin(a3);
   C = TMath::Cos(a2); D = TMath::Sin(a2); // should be -sin(a2) for positive direction
   E = TMath::Cos(a1); F = TMath::Sin(a1);
   Double_t AD = A*D, BD = B*D;

   M[F00] = C*E; M[F01] = -BD*E - A*F; M[F02] = -AD*E + B*F;
   M[F10] = C*F; M[F11] = -BD*F + A*E; M[F12] = -AD*F - B*E;
   M[F20] = D;   M[F21] = B*C;         M[F22] = A*C;

   mA1 = a1; mA2 = a2; mA3 = a3;
   bAsOK = true;
}

//______________________________________________________________________________
void TEveTrans::SetRotByAnyAngles(Float_t a1, Float_t a2, Float_t a3,
                                  const Text_t* pat)
{
   // Sets Rotation part as given by angles a1, a1, a3 and pattern pat.
   // Pattern consists of "XxYyZz" characters.
   // eg: x means rotate about x axis, X means rotate in negative direction
   // xYz -> R_x(a3) * R_y(-a2) * R_z(a1); (standard Gled representation)
   // Note that angles and pattern elements have inversed order!
   //
   // Implements Eulerian/Cardanian angles in a uniform way.

   int n = strspn(pat, "XxYyZz"); if(n > 3) n = 3;
   // Build Trans ... assign ...
   Float_t a[] = { a3, a2, a1 };
   UnitRot();
   for(int i=0; i<n; i++) {
      if(isupper(pat[i])) a[i] = -a[i];
      switch(pat[i]) {
         case 'x': case 'X': RotateLF(2, 3, a[i]); break;
         case 'y': case 'Y': RotateLF(3, 1, a[i]); break;
         case 'z': case 'Z': RotateLF(1, 2, a[i]); break;
      }
   }
   bAsOK = kFALSE;
}

//______________________________________________________________________________
void TEveTrans::GetRotAngles(Float_t* x) const
{
   // Get Cardan rotation angles (pattern xYz above).

   if(!bAsOK) {
      Double_t sx, sy, sz;
      GetScale(sx, sy, sz);
      Double_t d = M[F20]/sx;
      if(d>1) d=1; else if(d<-1) d=-1; // Fix numerical errors
      mA2 = TMath::ASin(d);
      Double_t C = TMath::Cos(mA2);
      if(TMath::Abs(C) > 8.7e-6) {
         mA1 = TMath::ATan2(M[F10], M[F00]);
         mA3 = TMath::ATan2(M[F21]/sy, M[F22]/sz);
      } else {
         mA1 = TMath::ATan2(M[F10]/sx, M[F11]/sy);
         mA3 = 0;
      }
      bAsOK = true;
   }
   x[0] = mA1; x[1] = mA2; x[2] = mA3;
}

/******************************************************************************/
// Scaling
/******************************************************************************/

//______________________________________________________________________________
void TEveTrans::Scale(Double_t sx, Double_t sy, Double_t sz)
{
   M[F00] *= sx; M[F10] *= sx; M[F20] *= sx;
   M[F01] *= sy; M[F11] *= sy; M[F21] *= sy;
   M[F02] *= sz; M[F12] *= sz; M[F22] *= sz;
}

//______________________________________________________________________________
void TEveTrans::GetScale(Double_t& sx, Double_t& sy, Double_t& sz) const
{
   sx = TMath::Sqrt( M[F00]*M[F00] + M[F10]*M[F10] + M[F20]*M[F20] );
   sy = TMath::Sqrt( M[F01]*M[F01] + M[F11]*M[F11] + M[F21]*M[F21] );
   sz = TMath::Sqrt( M[F02]*M[F02] + M[F12]*M[F12] + M[F22]*M[F22] );
}

//______________________________________________________________________________
void TEveTrans::Unscale(Double_t& sx, Double_t& sy, Double_t& sz)
{
   GetScale(sx, sy, sz);
   M[F00] /= sx; M[F10] /= sx; M[F20] /= sx;
   M[F01] /= sy; M[F11] /= sy; M[F21] /= sy;
   M[F02] /= sz; M[F12] /= sz; M[F22] /= sz;
}

//______________________________________________________________________________
Double_t TEveTrans::Unscale()
{
   Double_t sx, sy, sz;
   Unscale(sx, sy, sz);
   return (sx + sy + sz)/3;
}

/******************************************************************************/
// Operations on vectors
/******************************************************************************/

//______________________________________________________________________________
void TEveTrans::MultiplyIP(TVector3& v, Double_t w) const
{
   v.SetXYZ(M[F00]*v.x() + M[F01]*v.y() + M[F02]*v.z() + M[F03]*w,
            M[F10]*v.x() + M[F11]*v.y() + M[F12]*v.z() + M[F13]*w,
            M[F20]*v.x() + M[F21]*v.y() + M[F22]*v.z() + M[F23]*w);
}

//______________________________________________________________________________
void TEveTrans::MultiplyIP(Double_t* v, Double_t w) const
{
   Double_t r[3] = { v[0], v[1], v[2] };
   v[0] = M[F00]*r[0] + M[F01]*r[1] + M[F02]*r[2] + M[F03]*w;
   v[1] = M[F10]*r[0] + M[F11]*r[1] + M[F12]*r[2] + M[F13]*w;
   v[2] = M[F20]*r[0] + M[F21]*r[1] + M[F22]*r[2] + M[F23]*w;
}

//______________________________________________________________________________
void TEveTrans::MultiplyIP(Float_t* v, Double_t w) const
{
   Double_t r[3] = { v[0], v[1], v[2] };
   v[0] = M[F00]*r[0] + M[F01]*r[1] + M[F02]*r[2] + M[F03]*w;
   v[1] = M[F10]*r[0] + M[F11]*r[1] + M[F12]*r[2] + M[F13]*w;
   v[2] = M[F20]*r[0] + M[F21]*r[1] + M[F22]*r[2] + M[F23]*w;
}

//______________________________________________________________________________
TVector3 TEveTrans::Multiply(const TVector3& v, Double_t w) const
{
   return TVector3(M[F00]*v.x() + M[F01]*v.y() + M[F02]*v.z() + M[F03]*w,
                   M[F10]*v.x() + M[F11]*v.y() + M[F12]*v.z() + M[F13]*w,
                   M[F20]*v.x() + M[F21]*v.y() + M[F22]*v.z() + M[F23]*w);
}

//______________________________________________________________________________
void TEveTrans::Multiply(const Double_t *vin, Double_t* vout, Double_t w) const
{
   vout[0] = M[F00]*vin[0] + M[F01]*vin[1] + M[F02]*vin[2] + M[F03]*w;
   vout[1] = M[F10]*vin[0] + M[F11]*vin[1] + M[F12]*vin[1] + M[F13]*w;
   vout[2] = M[F20]*vin[0] + M[F21]*vin[1] + M[F22]*vin[1] + M[F23]*w;
}

//______________________________________________________________________________
void TEveTrans::RotateIP(TVector3& v) const
{
   v.SetXYZ(M[F00]*v.x() + M[F01]*v.y() + M[F02]*v.z(),
            M[F10]*v.x() + M[F11]*v.y() + M[F12]*v.z(),
            M[F20]*v.x() + M[F21]*v.y() + M[F22]*v.z());
}

//______________________________________________________________________________
TVector3 TEveTrans::Rotate(const TVector3& v) const
{
   return TVector3(M[F00]*v.x() + M[F01]*v.y() + M[F02]*v.z(),
                   M[F10]*v.x() + M[F11]*v.y() + M[F12]*v.z(),
                   M[F20]*v.x() + M[F21]*v.y() + M[F22]*v.z());
}

/******************************************************************************/
// Normalization, ortogonalization
/******************************************************************************/

//______________________________________________________________________________
Double_t TEveTrans::norm3_column(Int_t col)
{
   Double_t* C = M + 4*--col;
   const Double_t  l = TMath::Sqrt(C[0]*C[0] + C[1]*C[1] + C[2]*C[2]);
   C[0] /= l; C[1] /= l; C[2] /= l;
   return l;
}

//______________________________________________________________________________
Double_t TEveTrans::orto3_column(Int_t col, Int_t ref)
{
   Double_t* C = M + 4*--col;
   Double_t* R = M + 4*--ref;
   const Double_t dp = C[0]*R[0] + C[1]*R[1] + C[2]*R[2];
   C[0] -= R[0]*dp; C[1] -= R[1]*dp; C[2] -= R[2]*dp;
   return dp;
}

//______________________________________________________________________________
void TEveTrans::OrtoNorm3()
{
   norm3_column(1);
   orto3_column(2,1); norm3_column(2);
   M[F02] = M[F10]*M[F21] - M[F11]*M[F20];
   M[F12] = M[F20]*M[F01] - M[F21]*M[F00];
   M[F22] = M[F00]*M[F11] - M[F01]*M[F10];
   // cross-product faster.
   // orto3_column(3,1); orto3_column(3,2); norm3_column(3);
}

/******************************************************************************/
// Inversion
/******************************************************************************/

//______________________________________________________________________________
Double_t TEveTrans::Invert()
{
   // Copied from ROOT's TMatrixFCramerInv.

   static const TEveException _eh("TEveTrans::Invert ");

   // Find all NECESSARY 2x2 dets:  (18 of them)
   const Double_t det2_12_01 = M[F10]*M[F21] - M[F11]*M[F20];
   const Double_t det2_12_02 = M[F10]*M[F22] - M[F12]*M[F20];
   const Double_t det2_12_03 = M[F10]*M[F23] - M[F13]*M[F20];
   const Double_t det2_12_13 = M[F11]*M[F23] - M[F13]*M[F21];
   const Double_t det2_12_23 = M[F12]*M[F23] - M[F13]*M[F22];
   const Double_t det2_12_12 = M[F11]*M[F22] - M[F12]*M[F21];
   const Double_t det2_13_01 = M[F10]*M[F31] - M[F11]*M[F30];
   const Double_t det2_13_02 = M[F10]*M[F32] - M[F12]*M[F30];
   const Double_t det2_13_03 = M[F10]*M[F33] - M[F13]*M[F30];
   const Double_t det2_13_12 = M[F11]*M[F32] - M[F12]*M[F31];
   const Double_t det2_13_13 = M[F11]*M[F33] - M[F13]*M[F31];
   const Double_t det2_13_23 = M[F12]*M[F33] - M[F13]*M[F32];
   const Double_t det2_23_01 = M[F20]*M[F31] - M[F21]*M[F30];
   const Double_t det2_23_02 = M[F20]*M[F32] - M[F22]*M[F30];
   const Double_t det2_23_03 = M[F20]*M[F33] - M[F23]*M[F30];
   const Double_t det2_23_12 = M[F21]*M[F32] - M[F22]*M[F31];
   const Double_t det2_23_13 = M[F21]*M[F33] - M[F23]*M[F31];
   const Double_t det2_23_23 = M[F22]*M[F33] - M[F23]*M[F32];

   // Find all NECESSARY 3x3 dets:   (16 of them)
   const Double_t det3_012_012 = M[F00]*det2_12_12 - M[F01]*det2_12_02 + M[F02]*det2_12_01;
   const Double_t det3_012_013 = M[F00]*det2_12_13 - M[F01]*det2_12_03 + M[F03]*det2_12_01;
   const Double_t det3_012_023 = M[F00]*det2_12_23 - M[F02]*det2_12_03 + M[F03]*det2_12_02;
   const Double_t det3_012_123 = M[F01]*det2_12_23 - M[F02]*det2_12_13 + M[F03]*det2_12_12;
   const Double_t det3_013_012 = M[F00]*det2_13_12 - M[F01]*det2_13_02 + M[F02]*det2_13_01;
   const Double_t det3_013_013 = M[F00]*det2_13_13 - M[F01]*det2_13_03 + M[F03]*det2_13_01;
   const Double_t det3_013_023 = M[F00]*det2_13_23 - M[F02]*det2_13_03 + M[F03]*det2_13_02;
   const Double_t det3_013_123 = M[F01]*det2_13_23 - M[F02]*det2_13_13 + M[F03]*det2_13_12;
   const Double_t det3_023_012 = M[F00]*det2_23_12 - M[F01]*det2_23_02 + M[F02]*det2_23_01;
   const Double_t det3_023_013 = M[F00]*det2_23_13 - M[F01]*det2_23_03 + M[F03]*det2_23_01;
   const Double_t det3_023_023 = M[F00]*det2_23_23 - M[F02]*det2_23_03 + M[F03]*det2_23_02;
   const Double_t det3_023_123 = M[F01]*det2_23_23 - M[F02]*det2_23_13 + M[F03]*det2_23_12;
   const Double_t det3_123_012 = M[F10]*det2_23_12 - M[F11]*det2_23_02 + M[F12]*det2_23_01;
   const Double_t det3_123_013 = M[F10]*det2_23_13 - M[F11]*det2_23_03 + M[F13]*det2_23_01;
   const Double_t det3_123_023 = M[F10]*det2_23_23 - M[F12]*det2_23_03 + M[F13]*det2_23_02;
   const Double_t det3_123_123 = M[F11]*det2_23_23 - M[F12]*det2_23_13 + M[F13]*det2_23_12;

   // Find the 4x4 det:
   const Double_t det = M[F00]*det3_123_123 - M[F01]*det3_123_023 +
      M[F02]*det3_123_013 - M[F03]*det3_123_012;

   if(det == 0) {
      throw(_eh + "matrix is singular.");
   }

   const Double_t oneOverDet = 1.0/det;
   const Double_t mn1OverDet = - oneOverDet;

   M[F00] = det3_123_123 * oneOverDet;
   M[F01] = det3_023_123 * mn1OverDet;
   M[F02] = det3_013_123 * oneOverDet;
   M[F03] = det3_012_123 * mn1OverDet;

   M[F10] = det3_123_023 * mn1OverDet;
   M[F11] = det3_023_023 * oneOverDet;
   M[F12] = det3_013_023 * mn1OverDet;
   M[F13] = det3_012_023 * oneOverDet;

   M[F20] = det3_123_013 * oneOverDet;
   M[F21] = det3_023_013 * mn1OverDet;
   M[F22] = det3_013_013 * oneOverDet;
   M[F23] = det3_012_013 * mn1OverDet;

   M[F30] = det3_123_012 * mn1OverDet;
   M[F31] = det3_023_012 * oneOverDet;
   M[F32] = det3_013_012 * mn1OverDet;
   M[F33] = det3_012_012 * oneOverDet;

   bAsOK = kFALSE;
   return det;
}

/******************************************************************************/

//______________________________________________________________________________
void TEveTrans::Streamer(TBuffer &R__b)
{
   // Stream an object of class TEveTrans.

   if (R__b.IsReading()) {
      TEveTrans::Class()->ReadBuffer(R__b, this);
      bAsOK = kFALSE;
   } else {
      TEveTrans::Class()->WriteBuffer(R__b, this);
   }
}

/******************************************************************************/
/******************************************************************************/

//______________________________________________________________________________
void TEveTrans::Print(Option_t* /*option*/) const
{
   const Double_t* C = M;
   for(Int_t i=0; i<4; ++i, ++C)
      printf("%8.3f %8.3f %8.3f | %8.3f\n", C[0], C[4], C[8], C[12]);
}

#include <iomanip>

//______________________________________________________________________________
ostream& operator<<(ostream& s, const TEveTrans& t) {
   s.setf(std::ios::fixed, std::ios::floatfield);
   s.precision(3);
   for(Int_t i=1; i<=4; i++)
      for(Int_t j=1; j<=4; j++)
         s << t(i,j) << ((j==4) ? "\n" : "\t");
   return s;
}

/******************************************************************************/
// TEveUtil stuff
/******************************************************************************/

#include <TGeoMatrix.h>
#include <TBuffer3D.h>

void TEveTrans::SetFrom(Double_t* carr)
{
   fUseTrans = kTRUE;
   memcpy(M, carr, 16*sizeof(Double_t));
   bAsOK = kFALSE;
}

//______________________________________________________________________________
void TEveTrans::SetFrom(const TGeoMatrix& mat)
{
   fUseTrans = kTRUE;
   const Double_t *r = mat.GetRotationMatrix();
   const Double_t *t = mat.GetTranslation();
   const Double_t *s = mat.GetScale();
   Double_t       *m = M;
   m[0] = r[0]*s[0]; m[1] = r[3]*s[0]; m[2] = r[6]*s[0]; m[3] = 0; m += 4;
   m[0] = r[1]*s[1]; m[1] = r[4]*s[1]; m[2] = r[7]*s[1]; m[3] = 0; m += 4;
   m[0] = r[2]*s[2]; m[1] = r[5]*s[2]; m[2] = r[8]*s[2]; m[3] = 0; m += 4;
   m[0] = t[0];      m[1] = t[1];      m[2] = t[2];      m[3] = 1;
   bAsOK = kFALSE;
}

//______________________________________________________________________________
void TEveTrans::SetGeoHMatrix(TGeoHMatrix& mat)
{
   Double_t *r = mat.GetRotationMatrix();
   Double_t *t = mat.GetTranslation();
   Double_t *s = mat.GetScale();
   if (fUseTrans)
   {
      mat.SetBit(TGeoMatrix::kGeoGenTrans);
      Double_t *m = M;
      GetScale(s[0], s[1], s[2]);
      r[0] = m[0]/s[0]; r[3] = m[1]/s[0]; r[6] = m[2]/s[0]; m += 4;
      r[1] = m[0]/s[1]; r[4] = m[1]/s[1]; r[7] = m[2]/s[1]; m += 4;
      r[2] = m[0]/s[2]; r[5] = m[1]/s[2]; r[8] = m[2]/s[2]; m += 4;
      t[0] = m[0];      t[1] = m[1];      t[2] = m[2];
   } else {
      mat.ResetBit(TGeoMatrix::kGeoGenTrans);
      r[0] = 1; r[3] = 0; r[6] = 0;
      r[1] = 0; r[4] = 1; r[7] = 0;
      r[2] = 0; r[5] = 0; r[8] = 1;
      s[0] = s[1] = s[2] = 1;
      t[0] = t[1] = t[2] = 0;
   }
}

//______________________________________________________________________________
void TEveTrans::SetBuffer3D(TBuffer3D& buff)
{
   buff.fLocalFrame = fUseTrans;
   if (fUseTrans) {
      // In phys-shape ctor the rotation part is transposed, due to
      // TGeo's convention for rotation matrix. So we have to transpose
      // it here, also.
      Double_t *B = buff.fLocalMaster;
      B[0]  = M[0];  B[1]  = M[4];  B[2]  = M[8];  B[3]  = M[3];
      B[4]  = M[1];  B[5]  = M[5];  B[6]  = M[9];  B[7]  = M[7];
      B[8]  = M[2];  B[9]  = M[6];  B[10] = M[10]; B[11] = M[11];
      B[12] = M[12]; B[13] = M[13]; B[14] = M[14]; B[15] = M[15];
      // Otherwise this would do:
      // memcpy(buff.fLocalMaster, M, 16*sizeof(Double_t));
   }
}

//______________________________________________________________________________
Bool_t TEveTrans::IsScale(Double_t low, Double_t high) const
{
   // Test if the transformation is a scale.
   // To be used by ROOT TGLObject descendants that potentially need to
   // use GL_NORMALIZE.
   // The low/high limits are expected to be squares of acutal limits.
   //
   // Ideally this should be done by the TGLViewer [but is not].

   if (!fUseTrans) return kFALSE;
   Double_t s;
   s = M[F00]*M[F00] + M[F10]*M[F10] + M[F20]*M[F20];
   if (s < low || s > high) return kTRUE;
   s = M[F01]*M[F01] + M[F11]*M[F11] + M[F21]*M[F21];
   if (s < low || s > high) return kTRUE;
   s = M[F02]*M[F02] + M[F12]*M[F12] + M[F22]*M[F22];
   if (s < low || s > high) return kTRUE;
   return kFALSE;
}
