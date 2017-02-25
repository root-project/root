// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 ROOT FNAL MathLib Team                          *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for BoostY
//
// Created by: Mark Fischler  Mon Nov 1  2005
//
// Last update: $Id$
//
#ifndef ROOT_Math_GenVector_BoostY
#define ROOT_Math_GenVector_BoostY 1

#include "Math/GenVector/LorentzVector.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "Math/GenVector/DisplacementVector3D.h"
#include "Math/GenVector/Cartesian3D.h"

namespace ROOT {
namespace Math {
namespace Impl {
  
//__________________________________________________________________________________________
   /**
      Class representing a Lorentz Boost along the Y axis, by beta.
      For efficiency, gamma is held as well.

      @ingroup GenVector
   */

template< typename T = double>
class BoostY {

public:

   typedef T Scalar;

   enum ELorentzRotationMatrixIndex {
      kLXX =  0, kLXY =  1, kLXZ =  2, kLXT =  3
    , kLYX =  4, kLYY =  5, kLYZ =  6, kLYT =  7
    , kLZX =  8, kLZY =  9, kLZZ = 10, kLZT = 11
    , kLTX = 12, kLTY = 13, kLTZ = 14, kLTT = 15
   };

   enum EBoostMatrixIndex {
      kXX =  0, kXY =  1, kXZ =  2, kXT =  3
    , kYY =  4, kYZ =  5, kYT =  6
    , kZZ =  7, kZT =  8
    , kTT =  9
    , kNElems = 10
   };

   // ========== Constructors and Assignment =====================

   /**
      Default constructor (identity transformation)
   */
   BoostY() : fBeta(0), fGamma(1) {}

   /**
      Construct given a Scalar beta_y
   */
   explicit BoostY(Scalar beta_y) { SetComponents(beta_y); }

   // The compiler-generated copy ctor, copy assignment, and dtor are OK.

   /**
      Re-adjust components to eliminate small deviations from a perfect
      orthosyplectic matrix.
   */
   void Rectify() {
     // Assuming the representation of this is close to a true Lorentz Rotation,
     // but may have drifted due to round-off error from many operations,
     // this forms an "exact" orthosymplectic matrix for the Lorentz Rotation
     // again.
     
     if ( fGamma <= Scalar(0) ) {
       GenVector::Throw (
                         "Attempt to rectify a boost with non-positive gamma");
     }
     else
     {
       Scalar beta = fBeta;
       if ( beta >= Scalar(1) ) {
         beta /= ( beta * ( Scalar( 1.0 + 1.0e-16 ) ) );
       }
       SetComponents ( beta );
     }
   }

   // ======== Components ==============

   /**
      Set components from a Scalar beta_y
   */
   void SetComponents (Scalar by) {
     using namespace std;
     // set component
     const Scalar bp2 = by*by;
     if ( bp2 >= Scalar(1) ) {
       GenVector::Throw("Beta Vector supplied to set BoostY represents speed >= c");
     }
     else
     {
       fBeta = by;
       fGamma = Scalar(1) / sqrt( Scalar(1) -bp2 );
     }
   }

   /**
      Get components into a Scalar beta_y
   */
   void GetComponents (Scalar& by) const {
     // get component
     by = fBeta;
   }
  
   /**
       Retrieve the beta of the Boost
   */
   Scalar Beta() const { return fBeta; }

   /**
       Retrieve the gamma of the Boost
   */
   Scalar Gamma() const { return fGamma; }

   /**
       Set the given beta of the Boost
   */
   void SetBeta(Scalar beta) { SetComponents(beta); }

   /**
      The beta vector for this boost
   */
   typedef  DisplacementVector3D<Cartesian3D<T>, DefaultCoordinateSystemTag > XYZVector;
   XYZVector BetaVector() const {
     // return beta vector
     return DisplacementVector3D< Cartesian3D<Scalar> > ( Scalar(0), fBeta, Scalar(0) );
   }
  
   /**
      Get elements of internal 4x4 symmetric representation, into a data
      array suitable for direct use as the components of a LorentzRotation
      Note -- 16 Scalars will be written into the array; if the array is not
      that large, then this will lead to undefined behavior.
   */
   void GetLorentzRotation (Scalar r[]) const {
     // get corresponding LorentzRotation
     r[kLXX] = Scalar(1);  r[kLXY] = Scalar(0);     r[kLXZ] = Scalar(0);  r[kLXT] = Scalar(0);
     r[kLYX] = Scalar(0);  r[kLYY] = fGamma;        r[kLYZ] = Scalar(0);  r[kLYT] = fGamma*fBeta;
     r[kLZX] = Scalar(0);  r[kLZY] = Scalar(0);     r[kLZZ] = Scalar(1);  r[kLZT] = Scalar(0);
     r[kLTX] = Scalar(0);  r[kLTY] = fGamma*fBeta;  r[kLTZ] = Scalar(0);  r[kLTT] = fGamma;
   }

   // =========== operations ==============

   /**
      Lorentz transformation operation on a Minkowski ('Cartesian')
      LorentzVector
   */
   LorentzVector< ROOT::Math::PxPyPzE4D<T> >
   operator() (const LorentzVector< ROOT::Math::PxPyPzE4D<T> > & v) const {
     // apply boost to a LV
     const Scalar y = v.Py();
     const Scalar t = v.E();
     return LorentzVector< PxPyPzE4D<T> >
       (  v.Px()
        , fGamma*y       + fGamma*fBeta*t
        , v.Pz()
        , fGamma*fBeta*y + fGamma*t );
   }

   /**
      Lorentz transformation operation on a LorentzVector in any
      coordinate system
   */
   template <class CoordSystem>
   LorentzVector<CoordSystem>
   operator() (const LorentzVector<CoordSystem> & v) const {
      LorentzVector< PxPyPzE4D<T> > xyzt(v);
      LorentzVector< PxPyPzE4D<T> > r_xyzt = operator()(xyzt);
      return LorentzVector<CoordSystem> ( r_xyzt );
   }

   /**
      Lorentz transformation operation on an arbitrary 4-vector v.
      Preconditions:  v must implement methods x(), y(), z(), and t()
      and the arbitrary vector type must have a constructor taking (x,y,z,t)
   */
   template <class Foreign4Vector>
   Foreign4Vector
   operator() (const Foreign4Vector & v) const {
      LorentzVector< PxPyPzE4D<T> > xyzt(v);
      LorentzVector< PxPyPzE4D<T> > r_xyzt = operator()(xyzt);
      return Foreign4Vector ( r_xyzt.X(), r_xyzt.Y(), r_xyzt.Z(), r_xyzt.T() );
   }

   /**
      Overload operator * for rotation on a vector
   */
   template <class A4Vector>
   inline
   A4Vector operator* (const A4Vector & v) const
   {
      return operator()(v);
   }

   /**
      Invert a BoostY in place
   */
   void Invert() {
     // invert Boost
     fBeta = -fBeta;
   }

   /**
      Return inverse of  a rotation
   */
   BoostY Inverse() const {
     // return inverse
     BoostY tmp(*this);
     tmp.Invert();
     return tmp;
   }

   /**
      Equality/inequality operators
   */
   bool operator == (const BoostY & rhs) const {
      return ( fBeta  == rhs.fBeta &&
               fGamma == rhs.fGamma );
   }
   bool operator != (const BoostY & rhs) const {
      return ! operator==(rhs);
   }

private:

   Scalar fBeta;    // beta Y of the Boost
   Scalar fGamma;   // gamma of the Boost

};  // BoostY

// ============ Class BoostY ends here ============

/**
   Stream Output and Input
*/
   // TODO - I/O should be put in the manipulator form
template< typename T>
std::ostream & operator<< (std::ostream & os, const BoostY<T> & b) {
  os << " BoostY( beta: " << b.Beta() << ", gamma: " << b.Gamma() << " ) ";
  return os;
}
  
} //namepsace Impl

typedef Impl::BoostY<double> BoostY;
typedef Impl::BoostY<float> BoostYF;  

} //namespace Math
} //namespace ROOT

#endif /* ROOT_Math_GenVector_BoostY  */
