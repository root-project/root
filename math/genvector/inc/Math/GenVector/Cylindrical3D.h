// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT MathLib Team  and                    *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for class Cylindrica3D
//
// Created by: Lorenzo Moneta  at Tue Dec 06 2005
//
//
#ifndef ROOT_Math_GenVector_Cylindrical3D
#define ROOT_Math_GenVector_Cylindrical3D  1

#include "Math/Math.h"

#include "Math/GenVector/eta.h"

#include <limits>
#include <cmath>

namespace ROOT {

namespace Math {

//__________________________________________________________________________________________
  /**
      Class describing a cylindrical coordinate system based on rho, z and phi.
      The base coordinates are rho (transverse component) , z and phi
      Phi is restricted to be in the range [-PI,PI)

      @ingroup GenVector
  */

template <class T>
class Cylindrical3D {

public :

   typedef T Scalar;

   /**
      Default constructor with rho=z=phi=0
   */
   Cylindrical3D() : fRho(0), fZ(0), fPhi(0) {  }

   /**
      Construct from rho eta and phi values
   */
   Cylindrical3D(Scalar rho, Scalar zz, Scalar phi) :
      fRho(rho), fZ(zz), fPhi(phi) { Restrict(); }

   /**
      Construct from any Vector or coordinate system implementing
      Rho(), Z() and Phi()
   */
   template <class CoordSystem >
   explicit Cylindrical3D( const CoordSystem & v ) :
      fRho( v.Rho() ),  fZ( v.Z() ),  fPhi( v.Phi() ) { Restrict(); }

   // for g++  3.2 and 3.4 on 32 bits found that the compiler generated copy ctor and assignment are much slower
   // re-implement them ( there is no no need to have them with g++4)

   /**
      copy constructor
    */
   Cylindrical3D(const Cylindrical3D & v) :
      fRho(v.Rho() ),  fZ(v.Z() ),  fPhi(v.Phi() )  {   }

   /**
      assignment operator
    */
   Cylindrical3D & operator= (const Cylindrical3D & v) {
      fRho = v.Rho();
      fZ   = v.Z();
      fPhi = v.Phi();
      return *this;
   }

   /**
      Set internal data based on an array of 3 Scalar numbers ( rho, z , phi)
   */
   void SetCoordinates( const Scalar src[] )
   { fRho=src[0]; fZ=src[1]; fPhi=src[2]; Restrict(); }

   /**
      get internal data into an array of 3 Scalar numbers ( rho, z , phi)
   */
   void GetCoordinates( Scalar dest[] ) const
   { dest[0] = fRho; dest[1] = fZ; dest[2] = fPhi; }

   /**
      Set internal data based on 3 Scalar numbers ( rho, z , phi)
   */
   void SetCoordinates(Scalar  rho, Scalar  zz, Scalar  phi)
   { fRho=rho; fZ=zz; fPhi=phi; Restrict(); }

   /**
      get internal data into 3 Scalar numbers ( rho, z , phi)
   */
   void GetCoordinates(Scalar& rho, Scalar& zz, Scalar& phi) const
   {rho=fRho; zz=fZ; phi=fPhi;}

private:
   inline static Scalar pi() { return Scalar(M_PI); }
   inline void          Restrict()
   {
      using std::floor;
      if (fPhi <= -pi() || fPhi > pi()) fPhi = fPhi - floor(fPhi / (2 * pi()) + .5) * 2 * pi();
   }
public:

   // accessors

   Scalar Rho()   const { return fRho; }
   Scalar Z()     const { return fZ;   }
   Scalar Phi()   const { return fPhi; }
   Scalar X() const { using std::cos; return fRho * cos(fPhi); }
   Scalar Y() const { using std::sin; return fRho * sin(fPhi); }

   Scalar Mag2()  const { return fRho*fRho + fZ*fZ;   }
   Scalar R() const { using std::sqrt; return sqrt(Mag2()); }
   Scalar Perp2() const { return fRho*fRho;           }
   Scalar Theta() const { return (fRho == Scalar(0) && fZ == Scalar(0)) ? Scalar(0) : atan2(fRho, fZ); }

   // pseudorapidity - use same implementation as in Cartesian3D
   Scalar Eta() const {
      return Impl::Eta_FromRhoZ( fRho, fZ);
   }

   // setters (only for data members)


   /**
       set the rho coordinate value keeping z and phi constant
   */
   void SetRho(T rho) {
      fRho = rho;
   }

   /**
       set the z coordinate value keeping rho and phi constant
   */
   void SetZ(T zz) {
      fZ = zz;
   }

   /**
       set the phi coordinate value keeping rho and z constant
   */
   void SetPhi(T phi) {
      fPhi = phi;
      Restrict();
   }

   /**
       set all values using cartesian coordinates
   */
   void SetXYZ(Scalar x, Scalar y, Scalar z);

   /**
      scale by a scalar quantity a --
      for cylindrical coords only rho and z change
   */
   void Scale (T a) {
      if (a < 0) {
         Negate();
         a = -a;
      }
      fRho *= a;
      fZ *= a;
   }

   /**
      negate the vector
   */
   void Negate ( ) {
      fPhi = ( fPhi > 0 ? fPhi - pi() : fPhi + pi() );
      fZ = -fZ;
   }

   // assignment operators
   /**
      generic assignment operator from any coordinate system implementing Rho(), Z() and Phi()
   */
   template <class CoordSystem >
   Cylindrical3D & operator= ( const CoordSystem & c ) {
      fRho = c.Rho();
      fZ   = c.Z();
      fPhi = c.Phi();
      return *this;
   }

   /**
      Exact component-by-component equality
   */
   bool operator==(const Cylindrical3D & rhs) const {
      return fRho == rhs.fRho && fZ == rhs.fZ && fPhi == rhs.fPhi;
   }
   bool operator!= (const Cylindrical3D & rhs) const
   {return !(operator==(rhs));}


   // ============= Compatibility section ==================

   // The following make this coordinate system look enough like a CLHEP
   // vector that an assignment member template can work with either
   T x() const { return X();}
   T y() const { return Y();}
   T z() const { return Z(); }

   // ============= Specializations for improved speed ==================

   // (none)

#if defined(__MAKECINT__) || defined(G__DICTIONARY)

   // ====== Set member functions for coordinates in other systems =======

   void SetX(Scalar x);

   void SetY(Scalar y);

   void SetEta(Scalar eta);

   void SetR(Scalar r);

   void SetTheta(Scalar theta);

#endif


private:

   T fRho;
   T fZ;
   T fPhi;

};

  } // end namespace Math

} // end namespace ROOT

// move implementations here to avoid circle dependencies

#include "Math/GenVector/Cartesian3D.h"

#if defined(__MAKECINT__) || defined(G__DICTIONARY)
#include "Math/GenVector/GenVector_exception.h"
#include "Math/GenVector/CylindricalEta3D.h"
#include "Math/GenVector/Polar3D.h"
#endif

namespace ROOT {

  namespace Math {

template <class T>
void Cylindrical3D<T>::SetXYZ(Scalar xx, Scalar yy, Scalar zz) {
   *this = Cartesian3D<Scalar>(xx, yy, zz);
}

#if defined(__MAKECINT__) || defined(G__DICTIONARY)


  // ====== Set member functions for coordinates in other systems =======



template <class T>
void Cylindrical3D<T>::SetX(Scalar xx) {
   GenVector_exception e("Cylindrical3D::SetX() is not supposed to be called");
   throw e;
   Cartesian3D<Scalar> v(*this); v.SetX(xx); *this = Cylindrical3D<Scalar>(v);
}
template <class T>
void Cylindrical3D<T>::SetY(Scalar yy) {
   GenVector_exception e("Cylindrical3D::SetY() is not supposed to be called");
   throw e;
   Cartesian3D<Scalar> v(*this); v.SetY(yy); *this = Cylindrical3D<Scalar>(v);
}
template <class T>
void Cylindrical3D<T>::SetR(Scalar r) {
   GenVector_exception e("Cylindrical3D::SetR() is not supposed to be called");
   throw e;
   Polar3D<Scalar> v(*this); v.SetR(r);
   *this = Cylindrical3D<Scalar>(v);
}
template <class T>
void Cylindrical3D<T>::SetTheta(Scalar theta) {
   GenVector_exception e("Cylindrical3D::SetTheta() is not supposed to be called");
   throw e;
   Polar3D<Scalar> v(*this); v.SetTheta(theta);
   *this = Cylindrical3D<Scalar>(v);
}
template <class T>
void Cylindrical3D<T>::SetEta(Scalar eta) {
   GenVector_exception e("Cylindrical3D::SetEta() is not supposed to be called");
   throw e;
   CylindricalEta3D<Scalar> v(*this); v.SetEta(eta);
   *this = Cylindrical3D<Scalar>(v);
}

#endif

  } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_GenVector_Cylindrical3D  */
