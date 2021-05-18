// @(#)root/physics:$Id$
// Author: Pasha Murat   12/02/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TVector2
#define ROOT_TVector2

#include "TObject.h"


class TVector2 : public TObject {
//------------------------------------------------------------------------------
//  data members
//------------------------------------------------------------------------------
protected:

   Double_t    fX;    // components of the vector
   Double_t    fY;
//------------------------------------------------------------------------------
//  function members
//------------------------------------------------------------------------------
public:

   typedef Double_t Scalar;   // to be able to use it with the ROOT::Math::VectorUtil functions

   TVector2 ();
   TVector2 (const TVector2&) = default;
   TVector2 (Double_t *s);
   TVector2 (Double_t x0, Double_t y0);
   virtual ~TVector2();
                                        // ****** unary operators

   TVector2&       operator  = (TVector2 const & v);
   TVector2&       operator += (TVector2 const & v);
   TVector2&       operator -= (TVector2 const & v);
   Double_t        operator *= (TVector2 const & v);
   TVector2&       operator *= (Double_t s);
   TVector2&       operator /= (Double_t s);

                                        // ****** binary operators

   friend TVector2       operator + (const TVector2&, const TVector2&);
   friend TVector2       operator + (const TVector2&, Double_t  );
   friend TVector2       operator + (Double_t  , const TVector2&);
   friend TVector2       operator - (const TVector2&, const TVector2&);
   friend TVector2       operator - (const TVector2&, Double_t  );
   friend Double_t       operator * (const TVector2&, const TVector2&);
   friend TVector2       operator * (const TVector2&, Double_t  );
   friend TVector2       operator * (Double_t  , const TVector2&);
   friend TVector2       operator / (const TVector2&, Double_t  );
   friend Double_t       operator ^ (const TVector2&, const TVector2&);

                                        // ****** setters
   void Set(const TVector2& v);
   void Set(Double_t x0, Double_t y0);
   void Set(float  x0, float  y0);
   void SetX(Double_t x0);
   void SetY(Double_t y0);
                                        // ****** other member functions

   Double_t Mod2() const { return fX*fX+fY*fY; };
   Double_t Mod () const;

   Double_t Px()   const { return fX; };
   Double_t Py()   const { return fY; };
   Double_t X ()   const { return fX; };
   Double_t Y ()   const { return fY; };

                                        // phi() is defined in [0,TWOPI]

   Double_t Phi           () const;
   Double_t DeltaPhi(const TVector2& v) const;
   void     SetMagPhi(Double_t mag, Double_t phi);

                                        // unit vector in the direction of *this

   TVector2 Unit() const;
   TVector2 Ort () const;

                                        // projection of *this to the direction
                                        // of TVector2 vector `v'

   TVector2 Proj(const TVector2& v) const;

                                        // component of *this normal to `v'

   TVector2 Norm(const TVector2& v) const;

                                        // rotates 2-vector by phi radians
   TVector2 Rotate (Double_t phi) const;

                                        // returns phi angle in the interval [0,2*PI)
   static Double_t Phi_0_2pi(Double_t x);

                                        // returns phi angle in the interval [-PI,PI)
   static Double_t Phi_mpi_pi(Double_t x);


   void Print(Option_t* option="") const;

   ClassDef(TVector2,3)  // A 2D physics vector

};

                                        // ****** unary operators

inline TVector2& TVector2::operator  = (TVector2 const& v) {fX  = v.fX; fY  = v.fY; return *this;}
inline TVector2& TVector2::operator += (TVector2 const& v) {fX += v.fX; fY += v.fY; return *this;}
inline TVector2& TVector2::operator -= (TVector2 const& v) {fX -= v.fX; fY -= v.fY; return *this;}

                                        // scalar product of 2 2-vectors

inline Double_t   TVector2::operator *= (const TVector2& v) { return(fX*v.fX+fY*v.fY); }

inline TVector2& TVector2::operator *= (Double_t s) { fX *=s; fY *=s; return *this; }
inline TVector2& TVector2::operator /= (Double_t s) { fX /=s; fY /=s; return *this; }

                                        // ****** binary operators

inline TVector2  operator + (const TVector2& v1, const TVector2& v2) {
   return TVector2(v1.fX+v2.fX,v1.fY+v2.fY);
}

inline TVector2  operator + (const TVector2& v1, Double_t bias) {
   return TVector2 (v1.fX+bias,v1.fY+bias);
}

inline TVector2  operator + (Double_t bias, const TVector2& v1) {
   return TVector2 (v1.fX+bias,v1.fY+bias);
}

inline TVector2  operator - (const TVector2& v1, const TVector2& v2) {
   return TVector2(v1.fX-v2.fX,v1.fY-v2.fY);
}

inline TVector2  operator - (const TVector2& v1, Double_t bias) {
   return TVector2 (v1.fX-bias,v1.fY-bias);
}

inline TVector2  operator * (const TVector2& v, Double_t s) {
   return TVector2 (v.fX*s,v.fY*s);
}

inline TVector2    operator * (Double_t s, const TVector2& v) {
   return TVector2 (v.fX*s,v.fY*s);
}

inline Double_t operator * (const TVector2& v1, const TVector2& v2) {
   return  v1.fX*v2.fX+v1.fY*v2.fY;
}

inline TVector2     operator / (const TVector2& v, Double_t s) {
   return TVector2 (v.fX/s,v.fY/s);
}

inline Double_t   operator ^ (const TVector2& v1, const TVector2& v2) {
   return  v1.fX*v2.fY-v1.fY*v2.fX;
}

inline  Double_t TVector2::DeltaPhi(const TVector2& v) const { return Phi_mpi_pi(Phi()-v.Phi()); }

inline  TVector2 TVector2::Ort () const { return Unit(); }

inline  TVector2 TVector2::Proj(const TVector2& v) const { return v*(((*this)*v)/v.Mod2()); }

inline  TVector2 TVector2::Norm(const TVector2& v) const {return *this-Proj(v); }

                                     // ****** setters

inline void TVector2::Set(const TVector2& v   )     { fX = v.fX; fY = v.fY; }
inline void TVector2::Set(Double_t x0, Double_t y0) { fX = x0  ; fY = y0 ;  }
inline void TVector2::Set(float  x0, float  y0)     { fX = x0  ; fY = y0 ;  }
inline void TVector2::SetX(Double_t x0)             { fX = x0 ; }
inline void TVector2::SetY(Double_t y0)             { fY = y0 ; }


#endif
