// @(#)root/physics:$Id$
// Author: Peter Malzacher   19/06/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLorentzRotation
#define ROOT_TLorentzRotation


#ifndef ROOT_TRotation
#include "TRotation.h"
#endif
#ifndef ROOT_TLorentzVector
#include "TLorentzVector.h"
#endif


class TLorentzRotation : public TObject {


public:

class TLorentzRotationRow {
public:
   inline TLorentzRotationRow(const TLorentzRotation &, int);
   inline Double_t operator [] (int) const;
private:
   const TLorentzRotation * fRR;
   int fII;
};
   // Helper class for implemention of C-style subscripting r[i][j]

   TLorentzRotation();
   // Default constructor. Gives a unit matrix.

   TLorentzRotation(const TRotation &);
   // Constructor for 3d rotations.

   TLorentzRotation(const TLorentzRotation &);
   // Copy constructor.

   TLorentzRotation(Double_t, Double_t, Double_t);
   TLorentzRotation(const TVector3 &);
   // Constructors giving a Lorenz-boost.

   inline Double_t XX() const;
   inline Double_t XY() const;
   inline Double_t XZ() const;
   inline Double_t XT() const;
   inline Double_t YX() const;
   inline Double_t YY() const;
   inline Double_t YZ() const;
   inline Double_t YT() const;
   inline Double_t ZX() const;
   inline Double_t ZY() const;
   inline Double_t ZZ() const;
   inline Double_t ZT() const;
   inline Double_t TX() const;
   inline Double_t TY() const;
   inline Double_t TZ() const;
   inline Double_t TT() const;
   // Elements of the matrix.

   inline TLorentzRotationRow operator [] (int) const;
   // Returns object of the helper class for C-style subscripting r[i][j]


   Double_t operator () (int, int) const;
   // Fortran-style subscriptimg: returns (i,j) element of the matrix.


   inline TLorentzRotation & operator = (const TLorentzRotation &);
   inline TLorentzRotation & operator = (const TRotation &);
   // Assignment.

   inline Bool_t operator == (const TLorentzRotation &) const;
   inline Bool_t operator != (const TLorentzRotation &) const;
   // Comparisons.

   inline Bool_t IsIdentity() const;
   // Returns true if the Identity matrix.

   inline TLorentzVector VectorMultiplication(const TLorentzVector&) const;
   inline TLorentzVector operator * (const TLorentzVector &) const;
   // Multiplication with a Lorentz vector.

   TLorentzRotation MatrixMultiplication(const TLorentzRotation &) const;
   inline TLorentzRotation operator * (const TLorentzRotation &) const;
   inline TLorentzRotation & operator *= (const TLorentzRotation &);
   inline TLorentzRotation & Transform(const TLorentzRotation &);
   inline TLorentzRotation & Transform(const TRotation &);
   // Matrix multiplication.
   // Note: a *= b; <=> a = a * b; while a.Transform(b); <=> a = b * a;

   inline TLorentzRotation Inverse() const;
   // Return the inverse.

   inline TLorentzRotation & Invert();
   // Inverts the LorentzRotation matrix.

   inline TLorentzRotation & Boost(Double_t, Double_t, Double_t);
   inline TLorentzRotation & Boost(const TVector3 &);
   // Lorenz boost.

   inline TLorentzRotation & RotateX(Double_t);
   // Rotation around x-axis.

   inline TLorentzRotation & RotateY(Double_t);
   // Rotation around y-axis.

   inline TLorentzRotation & RotateZ(Double_t);
   // Rotation around z-axis.

   inline TLorentzRotation & Rotate(Double_t, const TVector3 &);
   inline TLorentzRotation & Rotate(Double_t, const TVector3 *);
   // Rotation around specified vector.

protected:

   Double_t fxx, fxy, fxz, fxt,
            fyx, fyy, fyz, fyt,
            fzx, fzy, fzz, fzt,
            ftx, fty, ftz, ftt;
   // The matrix elements.

   void SetBoost(Double_t, Double_t, Double_t);
   // Set elements according to a boost vector.

   TLorentzRotation(Double_t, Double_t, Double_t, Double_t,
                    Double_t, Double_t, Double_t, Double_t,
                    Double_t, Double_t, Double_t, Double_t,
                    Double_t, Double_t, Double_t, Double_t);
   // Protected constructor.

   ClassDef(TLorentzRotation,1) // Lorentz transformations including boosts and rotations

};



inline Double_t TLorentzRotation::XX() const { return fxx; }
inline Double_t TLorentzRotation::XY() const { return fxy; }
inline Double_t TLorentzRotation::XZ() const { return fxz; }
inline Double_t TLorentzRotation::XT() const { return fxt; }
inline Double_t TLorentzRotation::YX() const { return fyx; }
inline Double_t TLorentzRotation::YY() const { return fyy; }
inline Double_t TLorentzRotation::YZ() const { return fyz; }
inline Double_t TLorentzRotation::YT() const { return fyt; }
inline Double_t TLorentzRotation::ZX() const { return fzx; }
inline Double_t TLorentzRotation::ZY() const { return fzy; }
inline Double_t TLorentzRotation::ZZ() const { return fzz; }
inline Double_t TLorentzRotation::ZT() const { return fzt; }
inline Double_t TLorentzRotation::TX() const { return ftx; }
inline Double_t TLorentzRotation::TY() const { return fty; }
inline Double_t TLorentzRotation::TZ() const { return ftz; }
inline Double_t TLorentzRotation::TT() const { return ftt; }

inline TLorentzRotation::TLorentzRotationRow::TLorentzRotationRow
(const TLorentzRotation & r, int i) : fRR(&r), fII(i) {}

inline Double_t TLorentzRotation::TLorentzRotationRow::operator [] (int jj) const {
   return fRR->operator()(fII,jj);
}

inline TLorentzRotation::TLorentzRotationRow TLorentzRotation::operator [] (int i) const {
   return TLorentzRotationRow(*this, i);
}

inline TLorentzRotation & TLorentzRotation::operator = (const TLorentzRotation & r) {
   fxx = r.fxx; fxy = r.fxy; fxz = r.fxz; fxt = r.fxt;
   fyx = r.fyx; fyy = r.fyy; fyz = r.fyz; fyt = r.fyt;
   fzx = r.fzx; fzy = r.fzy; fzz = r.fzz; fzt = r.fzt;
   ftx = r.ftx; fty = r.fty; ftz = r.ftz; ftt = r.ftt;
   return *this;
}

//inline TLorentzRotation &
//TLorentzRotation::operator = (const TRotation & r) {
//  mxx = r.xx(); mxy = r.xy(); mxz = r.xz(); mxt = 0.0;
//  myx = r.yx(); myy = r.yy(); myz = r.yz(); myt = 0.0;
//  mzx = r.zx(); mzy = r.zy(); mzz = r.zz(); mzt = 0.0;
//  mtx = 0.0;    mty = 0.0;    mtz = 0.0;    mtt = 1.0;
//  return *this;
//}

inline TLorentzRotation & TLorentzRotation::operator = (const TRotation & r) {
   fxx = r.XX(); fxy = r.XY(); fxz = r.XZ(); fxt = 0.0;
   fyx = r.YX(); fyy = r.YY(); fyz = r.YZ(); fyt = 0.0;
   fzx = r.ZX(); fzy = r.ZY(); fzz = r.ZZ(); fzt = 0.0;
   ftx = 0.0;    fty = 0.0;    ftz = 0.0;    ftt = 1.0;
   return *this;
}


//inline Bool_t
//TLorentzRotation::operator == (const TLorentzRotation & r) const {
//  return (mxx == r.xx() && mxy == r.xy() && mxz == r.xz() && mxt == r.xt() &&
//          myx == r.yx() && myy == r.yy() && myz == r.yz() && myt == r.yt() &&
//          mzx == r.zx() && mzy == r.zy() && mzz == r.zz() && mzt == r.zt() &&
//          mtx == r.tx() && mty == r.ty() && mtz == r.tz() && mtt == r.tt())
//  ? kTRUE : kFALSE;
//}

inline Bool_t TLorentzRotation::operator == (const TLorentzRotation & r) const {
   return (fxx == r.fxx && fxy == r.fxy && fxz == r.fxz && fxt == r.fxt &&
           fyx == r.fyx && fyy == r.fyy && fyz == r.fyz && fyt == r.fyt &&
           fzx == r.fzx && fzy == r.fzy && fzz == r.fzz && fzt == r.fzt &&
           ftx == r.ftx && fty == r.fty && ftz == r.ftz && ftt == r.ftt)
   ? kTRUE : kFALSE;
}

//inline Bool_t
//TLorentzRotation::operator != (const TLorentzRotation & r) const {
//  return (mxx != r.xx() || mxy != r.xy() || mxz != r.xz() || mxt != r.xt() ||
//          myx != r.yx() || myy != r.yy() || myz != r.yz() || myt != r.yt() ||
//          mzx != r.zx() || mzy != r.zy() || mzz != r.zz() || mzt != r.zt() ||
//          mtx != r.tx() || mty != r.ty() || mtz != r.tz() || mtt != r.tt())
//  ? kTRUE : kFALSE;
//}

inline Bool_t TLorentzRotation::operator != (const TLorentzRotation & r) const {
   return (fxx != r.fxx || fxy != r.fxy || fxz != r.fxz || fxt != r.fxt ||
           fyx != r.fyx || fyy != r.fyy || fyz != r.fyz || fyt != r.fyt ||
           fzx != r.fzx || fzy != r.fzy || fzz != r.fzz || fzt != r.fzt ||
           ftx != r.ftx || fty != r.fty || ftz != r.ftz || ftt != r.ftt)
   ? kTRUE : kFALSE;
}

inline Bool_t TLorentzRotation::IsIdentity() const {
   return (fxx == 1.0 && fxy == 0.0 && fxz == 0.0 && fxt == 0.0 &&
           fyx == 0.0 && fyy == 1.0 && fyz == 0.0 && fyt == 0.0 &&
           fzx == 0.0 && fzy == 0.0 && fzz == 1.0 && fzt == 0.0 &&
           ftx == 0.0 && fty == 0.0 && ftz == 0.0 && ftt == 1.0)
   ? kTRUE : kFALSE;
}


inline TLorentzVector TLorentzRotation::VectorMultiplication(const TLorentzVector & p) const {
   return TLorentzVector(fxx*p.X()+fxy*p.Y()+fxz*p.Z()+fxt*p.T(),
                         fyx*p.X()+fyy*p.Y()+fyz*p.Z()+fyt*p.T(),
                         fzx*p.X()+fzy*p.Y()+fzz*p.Z()+fzt*p.T(),
                         ftx*p.X()+fty*p.Y()+ftz*p.Z()+ftt*p.T());
}

inline TLorentzVector TLorentzRotation::operator * (const TLorentzVector & p) const {
   return VectorMultiplication(p);
}

inline TLorentzRotation TLorentzRotation::operator * (const TLorentzRotation & m) const {
   return MatrixMultiplication(m);
}

inline TLorentzRotation & TLorentzRotation::operator *= (const TLorentzRotation & m) {
   return *this = MatrixMultiplication(m);
}

inline TLorentzRotation & TLorentzRotation::Transform(const TLorentzRotation & m) {
   return *this = m.MatrixMultiplication(*this);
}

inline TLorentzRotation & TLorentzRotation::Transform(const TRotation & m){
   return Transform(TLorentzRotation(m));
}

inline TLorentzRotation TLorentzRotation::Inverse() const {
   return TLorentzRotation( fxx,  fyx,  fzx, -ftx,
                            fxy,  fyy,  fzy, -fty,
                            fxz,  fyz,  fzz, -ftz,
                           -fxt, -fyt, -fzt,  ftt);
}

inline TLorentzRotation & TLorentzRotation::Invert() {
   return *this = Inverse();
}

inline TLorentzRotation & TLorentzRotation::Boost(Double_t bx, Double_t by, Double_t bz) {
   return Transform(TLorentzRotation(bx, by, bz));
}

inline TLorentzRotation & TLorentzRotation::Boost(const TVector3 & b) {
   return Transform(TLorentzRotation(b));
}

inline TLorentzRotation & TLorentzRotation::RotateX(Double_t angle) {
   return Transform(TRotation().RotateX(angle));
}

inline TLorentzRotation & TLorentzRotation::RotateY(Double_t angle) {
   return Transform(TRotation().RotateY(angle));
}

inline TLorentzRotation & TLorentzRotation::RotateZ(Double_t angle) {
   return Transform(TRotation().RotateZ(angle));
}

inline TLorentzRotation & TLorentzRotation::Rotate(Double_t angle, const TVector3 & axis) {
   return Transform(TRotation().Rotate(angle, axis));
}

inline TLorentzRotation & TLorentzRotation::Rotate(Double_t angle, const TVector3 * axis) {
   return Transform(TRotation().Rotate(angle, axis));
}

#endif
