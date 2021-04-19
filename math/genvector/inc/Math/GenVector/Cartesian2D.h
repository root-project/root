// @(#)root/mathcore:$Id: b12794c790afad19142e34a401af6c233aba446b $
// Authors: W. Brown, M. Fischler, L. Moneta    2005

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
  *                    & FNAL LCG ROOT Mathlib Team                    *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for class Cartesian2D
//
// Created by: Lorenzo Moneta  at Mon 16 Apr 2007
//
#ifndef ROOT_Math_GenVector_Cartesian2D
#define ROOT_Math_GenVector_Cartesian2D  1

#include "Math/GenVector/Polar2Dfwd.h"

#include "Math/Math.h"


namespace ROOT {

namespace Math {

//__________________________________________________________________________________________
   /**
       Class describing a 2D cartesian coordinate system
       (x, y coordinates)

       @ingroup GenVector
   */

template <class T = double>
class Cartesian2D {

public :

   typedef T Scalar;

   /**
      Default constructor  with x=y=0
   */
   Cartesian2D() : fX(0.0), fY(0.0)  {  }

   /**
      Constructor from x,y  coordinates
   */
   Cartesian2D(Scalar xx, Scalar yy) : fX(xx), fY(yy) {  }

   /**
      Construct from any Vector or coordinate system implementing
      X() and Y()
   */
   template <class CoordSystem>
   explicit Cartesian2D(const CoordSystem & v)
      : fX(v.X()), fY(v.Y()) {  }


   // for g++  3.2 and 3.4 on 32 bits found that the compiler generated copy ctor and assignment are much slower
   // re-implement them ( there is no no need to have them with g++4)
   /**
      copy constructor
    */
   Cartesian2D(const Cartesian2D & v) :
      fX(v.X()), fY(v.Y())  {  }

   /**
      assignment operator
    */
   Cartesian2D & operator= (const Cartesian2D & v) {
      fX = v.X();
      fY = v.Y();
      return *this;
   }

   /**
      Set internal data based on 2 Scalar numbers
   */
   void SetCoordinates(Scalar  xx, Scalar  yy) { fX=xx; fY=yy;  }

   /**
      get internal data into 2 Scalar numbers
   */
   void GetCoordinates(Scalar& xx, Scalar& yy ) const {xx=fX; yy=fY; }

   Scalar X()     const { return fX;}
   Scalar Y()     const { return fY;}
   Scalar Mag2()  const { return fX*fX + fY*fY; }
   Scalar R() const { return std::sqrt(Mag2()); }
   Scalar Phi() const { return (fX == Scalar(0) && fY == Scalar(0)) ? Scalar(0) : atan2(fY, fX); }

   /**
       set the x coordinate value keeping y constant
   */
   void SetX(Scalar a) { fX = a; }

   /**
       set the y coordinate value keeping x constant
   */
   void SetY(Scalar a) { fY = a; }

   /**
       set all values using cartesian coordinates
   */
   void SetXY(Scalar xx, Scalar yy ) {
      fX=xx;
      fY=yy;
   }

   /**
      scale the vector by a scalar quantity a
   */
   void Scale(Scalar a) { fX *= a; fY *= a;  }

   /**
      negate the vector
   */
   void Negate() { fX = -fX; fY = -fY;  }

   /**
       rotate by an angle
    */
   void Rotate(Scalar angle) {
      const Scalar s = std::sin(angle);
      const Scalar c = std::cos(angle);
      SetCoordinates(c * fX - s * fY, s * fX + c * fY);
   }

   /**
      Assignment from any class implementing x(),y()
      (can assign from any coordinate system)
   */
   template <class CoordSystem>
   Cartesian2D & operator = (const CoordSystem & v) {
      fX = v.x();
      fY = v.y();
      return *this;
   }

   /**
      Exact equality
   */
   bool operator == (const Cartesian2D & rhs) const {
      return fX == rhs.fX && fY == rhs.fY;
   }
   bool operator != (const Cartesian2D & rhs) const {return !(operator==(rhs));}


   // ============= Compatibility section ==================

   // The following make this coordinate system look enough like a CLHEP
   // vector that an assignment member template can work with either
   Scalar x() const { return X();}
   Scalar y() const { return Y();}

   // ============= Overloads for improved speed ==================

   template <class T2>
   explicit Cartesian2D( const Polar2D<T2> & v )
   {
      const Scalar r = v.R(); // re-using this instead of calling v.X() and v.Y()
      // is the speed improvement
      fX = r * std::cos(v.Phi());
      fY = r * std::sin(v.Phi());
   }
   // Technical note:  This works even though only Polar2Dfwd.h is
   // included (and in fact, including Polar2D.h would cause circularity
   // problems). It works because any program **using** this ctor must itself
   // be including Polar2D.h.

   template <class T2>
   Cartesian2D & operator = (const Polar2D<T2> & v)
   {
      const Scalar r = v.R();
      fX             = r * std::cos(v.Phi());
      fY             = r * std::sin(v.Phi());
      return *this;
   }



#if defined(__MAKECINT__) || defined(G__DICTIONARY)

   // ====== Set member functions for coordinates in other systems =======

   void SetR(Scalar r);

   void SetPhi(Scalar phi);

#endif


private:

   /**
      (Contiguous) data containing the coordinates values x and y
   */
   T  fX;
   T  fY;

};


   } // end namespace Math

} // end namespace ROOT


#if defined(__MAKECINT__) || defined(G__DICTIONARY)
// need to put here setter methods to resolve nasty cyclical dependencies
// I need to include other coordinate systems only when Cartesian is already defined
// since they depend on it

#include "Math/GenVector/GenVector_exception.h"
#include "Math/GenVector/Polar2D.h"

// ====== Set member functions for coordinates in other systems =======

namespace ROOT {

   namespace Math {

      template <class T>
      void Cartesian2D<T>::SetR(Scalar r) {
         GenVector_exception e("Cartesian2D::SetR() is not supposed to be called");
         throw e;
         Polar2D<Scalar> v(*this); v.SetR(r); *this = Cartesian2D<Scalar>(v);
      }


      template <class T>
      void Cartesian2D<T>::SetPhi(Scalar phi) {
         GenVector_exception e("Cartesian2D::SetPhi() is not supposed to be called");
         throw e;
         Polar2D<Scalar> v(*this); v.SetPhi(phi); *this = Cartesian2D<Scalar>(v);
      }



   } // end namespace Math

} // end namespace ROOT

#endif




#endif /* ROOT_Math_GenVector_Cartesian2D  */
