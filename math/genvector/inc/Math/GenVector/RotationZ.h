// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT FNAL MathLib Team                    *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for class RotationZ representing a rotation about the Z axis
//
// Created by: Mark Fischler Mon July 18  2005
//
// Last update: $Id$
//
#ifndef ROOT_Math_GenVector_RotationZ
#define ROOT_Math_GenVector_RotationZ  1


#include "Math/GenVector/Cartesian3D.h"
#include "Math/GenVector/DisplacementVector3D.h"
#include "Math/GenVector/PositionVector3D.h"
#include "Math/GenVector/LorentzVector.h"
#include "Math/GenVector/3DDistances.h"

#include "Math/GenVector/RotationZfwd.h"

#include "TMath.h"
#include <cmath>

namespace ROOT {
namespace Math {


//__________________________________________________________________________________________
   /**
      Rotation class representing a 3D rotation about the Z axis by the angle of rotation.
      For efficiency reason, in addition to the angle, the sine and cosine of the angle are held

      @ingroup GenVector

      @see GenVector
   */

class RotationZ {

public:

   typedef double Scalar;


   // ========== Constructors and Assignment =====================

   /**
      Default constructor (identity rotation)
   */
   RotationZ() : fAngle(0), fSin(0), fCos(1) { }

   /**
      Construct from an angle
   */
   explicit RotationZ( Scalar angle ) :   fAngle(angle),
                                          fSin(std::sin(angle)),
                                          fCos(std::cos(angle))
   {
      Rectify();
   }

   // The compiler-generated copy ctor, copy assignment, and destructor are OK.

   /**
      Rectify makes sure the angle is in (-pi,pi]
   */
   void Rectify()  {
      if (std::fabs(fAngle) >= TMath::Pi()) {
         double x = fAngle / TMath::TwoPi();
         fAngle = TMath::TwoPi() * (x + std::floor(.5 - x));
         fSin = std::sin(fAngle);
         fCos = std::cos(fAngle);
      }
   }

   // ======== Components ==============

   /**
      Set given the angle.
   */
   void SetAngle (Scalar angle) {
      fSin=std::sin(angle);
      fCos=std::cos(angle);
      fAngle= angle;
      Rectify();
   }
   void SetComponents (Scalar angle) { SetAngle(angle); }

   /**
      Get the angle
   */
   void GetAngle(Scalar &angle) const { using std::atan2; angle = atan2(fSin, fCos); }
   void GetComponents ( Scalar & angle ) const { GetAngle(angle); }

   /**
      Angle of rotation
   */
   Scalar Angle() const { using std::atan2; return atan2(fSin, fCos); }

   /**
      Sine or Cosine of the rotation angle
   */
   Scalar SinAngle () const { return fSin; }
   Scalar CosAngle () const { return fCos; }

   // =========== operations ==============

//   /**
//      Rotation operation on a cartesian vector
//    */
//   typedef  DisplacementVector3D< Cartesian3D<double> > XYZVector;
//   XYZVector operator() (const XYZVector & v) const {
//     return XYZVector
//       ( fCos*v.x()-fSin*v.y(), fCos*v.y()+fSin*v.x(), v.z() );
//   }

   /**
      Rotation operation on a displacement vector in any coordinate system
   */
   template <class CoordSystem, class U>
   DisplacementVector3D<CoordSystem,U>
   operator() (const DisplacementVector3D<CoordSystem,U> & v) const {
      DisplacementVector3D< Cartesian3D<double>,U > xyz;
      xyz.SetXYZ( fCos*v.x()-fSin*v.y(), fCos*v.y()+fSin*v.x(), v.z()  );
      return DisplacementVector3D<CoordSystem,U>(xyz);
   }

   /**
      Rotation operation on a position vector in any coordinate system
   */
   template <class CoordSystem, class U>
   PositionVector3D<CoordSystem, U>
   operator() (const PositionVector3D<CoordSystem,U> & v) const {
      DisplacementVector3D< Cartesian3D<double>,U > xyz(v);
      DisplacementVector3D< Cartesian3D<double>,U > rxyz = operator()(xyz);
      return PositionVector3D<CoordSystem,U> ( rxyz );
   }

   /**
      Rotation operation on a Lorentz vector in any 4D coordinate system
   */
   template <class CoordSystem>
   LorentzVector<CoordSystem>
   operator() (const LorentzVector<CoordSystem> & v) const {
      DisplacementVector3D< Cartesian3D<double> > xyz(v.Vect());
      xyz = operator()(xyz);
      LorentzVector< PxPyPzE4D<double> > xyzt (xyz.X(), xyz.Y(), xyz.Z(), v.E());
      return LorentzVector<CoordSystem> ( xyzt );
   }

   /**
      Rotation operation on an arbitrary vector v.
      Preconditions:  v must implement methods x(), y(), and z()
      and the arbitrary vector type must have a constructor taking (x,y,z)
   */
   template <class ForeignVector>
   ForeignVector
   operator() (const  ForeignVector & v) const {
      DisplacementVector3D< Cartesian3D<double> > xyz(v);
      DisplacementVector3D< Cartesian3D<double> > rxyz = operator()(xyz);
      return ForeignVector ( rxyz.X(), rxyz.Y(), rxyz.Z() );
   }

   /**
      Overload operator * for rotation on a vector
   */
   template <class AVector>
   inline
   AVector operator* (const AVector & v) const
   {
      return operator()(v);
   }

   /**
      Invert a rotation in place
   */
   void Invert() { fAngle = -fAngle; fSin = -fSin; }

   /**
      Return inverse of  a rotation
   */
   RotationZ Inverse() const { RotationZ t(*this); t.Invert(); return t; }

   // ========= Multi-Rotation Operations ===============

   /**
      Multiply (combine) two rotations
   */
   RotationZ operator * (const RotationZ & r) const {
      RotationZ ans;
      double x = (fAngle + r.fAngle) / TMath::TwoPi();
      ans.fAngle = TMath::TwoPi() * (x + std::floor(.5 - x));
      ans.fSin   = fSin*r.fCos + fCos*r.fSin;
      ans.fCos   = fCos*r.fCos - fSin*r.fSin;
      return ans;
   }

   /**
      Post-Multiply (on right) by another rotation :  T = T*R
   */
   RotationZ & operator *= (const RotationZ & r) { return *this = (*this)*r; }

   /**
      Equality/inequality operators
   */
   bool operator == (const RotationZ & rhs) const {
      if( fAngle != rhs.fAngle )  return false;
      return true;
   }
   bool operator != (const RotationZ & rhs) const {
      return ! operator==(rhs);
   }

private:

   Scalar fAngle;   // rotation angle
   Scalar fSin;     // sine of the rotation angle
   Scalar fCos;     // cosine of the rotation angle

};  // RotationZ

// ============ Class RotationZ ends here ============

/**
   Distance between two rotations
 */
template <class R>
inline
typename RotationZ::Scalar
Distance ( const RotationZ& r1, const R & r2) {return gv_detail::dist(r1,r2);}

/**
   Stream Output and Input
 */
  // TODO - I/O should be put in the manipulator form

inline
std::ostream & operator<< (std::ostream & os, const RotationZ & r) {
  os << " RotationZ(" << r.Angle() << ") ";
  return os;
}


}  // namespace Math
}  // namespace ROOT

#endif // ROOT_Math_GenVector_RotationZ
