// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT MathLib Team  and                    *
  *                      FNAL LCG ROOT MathLib Team                    *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for class Polar3D
//
// Created by: Lorenzo Moneta  at Mon May 30 11:40:03 2005
// Major revamp:  M. Fischler  at Wed Jun  8 2005
//
// Last update: $Id$
//
#ifndef ROOT_Math_GenVector_Polar3D
#define ROOT_Math_GenVector_Polar3D  1

#include "Math/Math.h"

#include "Math/GenVector/eta.h"

#include <cmath>

namespace ROOT {

namespace Math {


//__________________________________________________________________________________________
   /**
       Class describing a polar coordinate system based on r, theta and phi
       Phi is restricted to be in the range [-PI,PI)

       @ingroup GenVector
   */


template <class T>
class Polar3D {

public :

   typedef T Scalar;

   /**
      Default constructor with r=theta=phi=0
   */
   Polar3D() : fR(0), fTheta(0), fPhi(0) {  }

   /**
      Construct from the polar coordinates:  r, theta and phi
   */
   Polar3D(T r,T theta,T phi) : fR(r), fTheta(theta), fPhi(phi) { Restrict(); }

   /**
      Construct from any Vector or coordinate system implementing
      R(), Theta() and Phi()
   */
   template <class CoordSystem >
   explicit Polar3D( const CoordSystem & v ) :
      fR(v.R() ),  fTheta(v.Theta() ),  fPhi(v.Phi() )  { Restrict(); }

   // for g++  3.2 and 3.4 on 32 bits found that the compiler generated copy ctor and assignment are much slower
   // re-implement them ( there is no no need to have them with g++4)

   /**
      copy constructor
    */
   Polar3D(const Polar3D & v) :
      fR(v.R() ),  fTheta(v.Theta() ),  fPhi(v.Phi() )  {   }

   /**
      assignment operator
    */
   Polar3D & operator= (const Polar3D & v) {
      fR     = v.R();
      fTheta = v.Theta();
      fPhi   = v.Phi();
      return *this;
   }

   /**
      Set internal data based on an array of 3 Scalar numbers
   */
   void SetCoordinates( const Scalar src[] )
   { fR=src[0]; fTheta=src[1]; fPhi=src[2]; Restrict(); }

   /**
      get internal data into an array of 3 Scalar numbers
   */
   void GetCoordinates( Scalar dest[] ) const
   { dest[0] = fR; dest[1] = fTheta; dest[2] = fPhi; }

   /**
      Set internal data based on 3 Scalar numbers
   */
   void SetCoordinates(Scalar r, Scalar  theta, Scalar  phi)
   { fR=r; fTheta=theta; fPhi=phi; Restrict(); }

   /**
      get internal data into 3 Scalar numbers
   */
   void GetCoordinates(Scalar& r, Scalar& theta, Scalar& phi) const {r=fR; theta=fTheta; phi=fPhi;}


   Scalar R()     const { return fR;}
   Scalar Phi()   const { return fPhi; }
   Scalar Theta() const { return fTheta; }
   Scalar Rho() const { using std::sin; return fR * sin(fTheta); }
   Scalar X() const { using std::cos; return Rho() * cos(fPhi); }
   Scalar Y() const { using std::sin; return Rho() * sin(fPhi); }
   Scalar Z() const { using std::cos; return fR * cos(fTheta); }
   Scalar Mag2()  const { return fR*fR;}
   Scalar Perp2() const { return Rho() * Rho(); }

   // pseudorapidity
   Scalar Eta() const
   {
      return Impl::Eta_FromTheta(fTheta, fR);
   }

   // setters (only for data members)


   /**
       set the r coordinate value keeping theta and phi constant
   */
   void SetR(const T & r) {
      fR = r;
   }

   /**
       set the theta coordinate value keeping r and phi constant
   */
   void SetTheta(const T & theta) {
      fTheta = theta;
   }

   /**
       set the phi coordinate value keeping r and theta constant
   */
   void SetPhi(const T & phi) {
      fPhi = phi;
      Restrict();
   }

   /**
       set all values using cartesian coordinates
   */
   void SetXYZ(Scalar x, Scalar y, Scalar z);


private:
   inline static Scalar pi()  { return M_PI; }
   inline void Restrict() {
      using std::floor;
      if (fPhi <= -pi() || fPhi > pi()) fPhi = fPhi - floor(fPhi / (2 * pi()) + .5) * 2 * pi();
   }

public:

   /**
       scale by a scalar quantity - for polar coordinates r changes
   */
   void Scale (T a) {
      if (a < 0) {
         Negate();
         a = -a;
      }
      // angles do not change when scaling by a positive quantity
      fR *= a;
   }

   /**
      negate the vector
   */
   void Negate ( ) {
      fPhi = ( fPhi > 0 ? fPhi - pi() : fPhi + pi() );
      fTheta = pi() - fTheta;
   }

   // assignment operators
   /**
      generic assignment operator from any coordinate system
   */
   template <class CoordSystem >
   Polar3D & operator= ( const CoordSystem & c ) {
      fR     = c.R();
      fTheta = c.Theta();
      fPhi   = c.Phi();
      return *this;
   }

   /**
      Exact equality
   */
   bool operator==(const Polar3D & rhs) const {
      return fR == rhs.fR && fTheta == rhs.fTheta && fPhi == rhs.fPhi;
   }
   bool operator!= (const Polar3D & rhs) const {return !(operator==(rhs));}


   // ============= Compatibility section ==================

   // The following make this coordinate system look enough like a CLHEP
   // vector that an assignment member template can work with either
   T x() const { return X(); }
   T y() const { return Y(); }
   T z() const { return Z(); }

   // ============= Specializations for improved speed ==================

   // (none)

#if defined(__MAKECINT__) || defined(G__DICTIONARY)

   // ====== Set member functions for coordinates in other systems =======

   void SetX(Scalar x);

   void SetY(Scalar y);

   void SetZ(Scalar z);

   void SetRho(Scalar rho);

   void SetEta(Scalar eta);

#endif

private:
   T fR;
   T fTheta;
   T fPhi;
};



  } // end namespace Math

} // end namespace ROOT

// move implementations here to avoid circle dependencies

#include "Math/GenVector/Cartesian3D.h"

#if defined(__MAKECINT__) || defined(G__DICTIONARY)
#include "Math/GenVector/GenVector_exception.h"
#include "Math/GenVector/CylindricalEta3D.h"
#endif


namespace ROOT {

  namespace Math {

template <class T>
void Polar3D<T>::SetXYZ(Scalar xx, Scalar yy, Scalar zz) {
   *this = Cartesian3D<Scalar>(xx, yy, zz);
}

#if defined(__MAKECINT__) || defined(G__DICTIONARY)

  // ====== Set member functions for coordinates in other systems =======


template <class T>
void Polar3D<T>::SetX(Scalar xx) {
   GenVector_exception e("Polar3D::SetX() is not supposed to be called");
   throw e;
   Cartesian3D<Scalar> v(*this); v.SetX(xx); *this = Polar3D<Scalar>(v);
}
template <class T>
void Polar3D<T>::SetY(Scalar yy) {
   GenVector_exception e("Polar3D::SetY() is not supposed to be called");
   throw e;
   Cartesian3D<Scalar> v(*this); v.SetY(yy); *this = Polar3D<Scalar>(v);
}
template <class T>
void Polar3D<T>::SetZ(Scalar zz) {
   GenVector_exception e("Polar3D::SetZ() is not supposed to be called");
   throw e;
   Cartesian3D<Scalar> v(*this); v.SetZ(zz); *this = Polar3D<Scalar>(v);
}
template <class T>
void Polar3D<T>::SetRho(Scalar rho) {
   GenVector_exception e("Polar3D::SetRho() is not supposed to be called");
   throw e;
   CylindricalEta3D<Scalar> v(*this); v.SetRho(rho);
   *this = Polar3D<Scalar>(v);
}
template <class T>
void Polar3D<T>::SetEta(Scalar eta) {
   GenVector_exception e("Polar3D::SetEta() is not supposed to be called");
   throw e;
   CylindricalEta3D<Scalar> v(*this); v.SetEta(eta);
   *this = Polar3D<Scalar>(v);
}

#endif


  } // end namespace Math

} // end namespace ROOT



#endif /* ROOT_Math_GenVector_Polar3D  */
