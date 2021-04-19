// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT MathLib Team  and                    *
  *                      FNAL LCG ROOT MathLib Team                    *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for class Polar2D
//
// Created by: Lorenzo Moneta  at Mon May 30 11:40:03 2005
// Major revamp:  M. Fischler  at Wed Jun  8 2005
//
// Last update: $Id$
//
#ifndef ROOT_Math_GenVector_Polar2D
#define ROOT_Math_GenVector_Polar2D  1

#include "Math/Math.h"

#include "Math/GenVector/etaMax.h"



namespace ROOT {

namespace Math {


//__________________________________________________________________________________________
   /**
       Class describing a polar 2D coordinate system based on r and phi
       Phi is restricted to be in the range [-PI,PI)

       @ingroup GenVector
   */


template <class T>
class Polar2D {

public :

   typedef T Scalar;

   /**
      Default constructor with r=1,phi=0
   */
   Polar2D() : fR(1.), fPhi(0) {  }

   /**
      Construct from the polar coordinates:  r and phi
   */
   Polar2D(T r,T phi) : fR(r), fPhi(phi) { Restrict(); }

   /**
      Construct from any Vector or coordinate system implementing
      R() and Phi()
   */
   template <class CoordSystem >
   explicit Polar2D( const CoordSystem & v ) :
      fR(v.R() ),  fPhi(v.Phi() )  { Restrict(); }

   // for g++  3.2 and 3.4 on 32 bits found that the compiler generated copy ctor and assignment are much slower
   // re-implement them ( there is no no need to have them with g++4)

   /**
      copy constructor
    */
   Polar2D(const Polar2D & v) :
      fR(v.R() ),  fPhi(v.Phi() )  {   }

   /**
      assignment operator
    */
   Polar2D & operator= (const Polar2D & v) {
      fR     = v.R();
      fPhi   = v.Phi();
      return *this;
   }


   /**
      Set internal data based on 2 Scalar numbers
   */
   void SetCoordinates(Scalar r, Scalar  phi)
   { fR=r; fPhi=phi; Restrict(); }

   /**
      get internal data into 2 Scalar numbers
   */
   void GetCoordinates(Scalar& r, Scalar& phi) const {r=fR; phi=fPhi;}


   Scalar R()     const { return fR;}
   Scalar Phi()   const { return fPhi; }
   Scalar X() const { return fR * std::cos(fPhi); }
   Scalar Y() const { return fR * std::sin(fPhi); }
   Scalar Mag2()  const { return fR*fR;}


   // setters (only for data members)


   /**
       set the r coordinate value keeping phi constant
   */
   void SetR(const T & r) {
      fR = r;
   }


   /**
       set the phi coordinate value keeping r constant
   */
   void SetPhi(const T & phi) {
      fPhi = phi;
      Restrict();
   }

   /**
       set all values using cartesian coordinates
   */
   void SetXY(Scalar a, Scalar b);


private:
   inline static double pi()  { return M_PI; }

   /**
      restrict abgle hi to be between -PI and PI
    */
   inline void Restrict() {
      if (fPhi <= -pi() || fPhi > pi()) fPhi = fPhi - std::floor(fPhi / (2 * pi()) + .5) * 2 * pi();
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
   }

   /**
      rotate the vector
    */
   void Rotate(T angle) {
      fPhi += angle;
      Restrict();
   }

   // assignment operators
   /**
      generic assignment operator from any coordinate system
   */
   template <class CoordSystem >
   Polar2D & operator= ( const CoordSystem & c ) {
      fR     = c.R();
      fPhi   = c.Phi();
      return *this;
   }

   /**
      Exact equality
   */
   bool operator==(const Polar2D & rhs) const {
      return fR == rhs.fR && fPhi == rhs.fPhi;
   }
   bool operator!= (const Polar2D & rhs) const {return !(operator==(rhs));}


   // ============= Compatibility section ==================

   // The following make this coordinate system look enough like a CLHEP
   // vector that an assignment member template can work with either
   T x() const { return X();}
   T y() const { return Y();}

   // ============= Specializations for improved speed ==================

   // (none)

#if defined(__MAKECINT__) || defined(G__DICTIONARY)

   // ====== Set member functions for coordinates in other systems =======

   void SetX(Scalar a);

   void SetY(Scalar a);

#endif

private:
   T fR;
   T fPhi;
};


   } // end namespace Math

} // end namespace ROOT


// move implementations here to avoid circle dependencies

#include "Math/GenVector/Cartesian2D.h"

#if defined(__MAKECINT__) || defined(G__DICTIONARY)
#include "Math/GenVector/GenVector_exception.h"
#endif

namespace ROOT {

   namespace Math {

template <class T>
void Polar2D<T>::SetXY(Scalar a, Scalar b) {
   *this = Cartesian2D<Scalar>(a, b);
}


#if defined(__MAKECINT__) || defined(G__DICTIONARY)


// ====== Set member functions for coordinates in other systems =======

      template <class T>
      void Polar2D<T>::SetX(Scalar a) {
         GenVector_exception e("Polar2D::SetX() is not supposed to be called");
         throw e;
         Cartesian2D<Scalar> v(*this); v.SetX(a); *this = Polar2D<Scalar>(v);
      }
      template <class T>
      void Polar2D<T>::SetY(Scalar a) {
         GenVector_exception e("Polar2D::SetY() is not supposed to be called");
         throw e;
         Cartesian2D<Scalar> v(*this); v.SetY(a); *this = Polar2D<Scalar>(v);
      }

#endif


   } // end namespace Math

} // end namespace ROOT



#endif /* ROOT_Math_GenVector_Polar2D  */
