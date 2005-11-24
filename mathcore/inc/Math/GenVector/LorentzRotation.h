// @(#)root/mathcore:$Name:  $:$Id: LorentzRotation.h,v 1.4 2005/11/16 19:30:47 marafino Exp $
// Authors: W. Brown, M. Fischler, L. Moneta    2005  

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 ROOT MathLib Team                               *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for LorentzRotation
// 
// Created by: Mark Fischler  Mon Aug 8  2005
// 
// Last update: $Id: LorentzRotation.h,v 1.4 2005/11/16 19:30:47 marafino Exp $
// 
#ifndef ROOT_Math_GenVector_LorentzRotation 
#define ROOT_Math_GenVector_LorentzRotation  1

#include "Math/GenVector/LorentzRotationfwd.h"

#include "Math/GenVector/LorentzVector.h"
#include "Math/GenVector/PxPyPzE4D.h"

#include "Math/GenVector/Rotation3Dfwd.h"
#include "Math/GenVector/AxisAnglefwd.h"
#include "Math/GenVector/EulerAnglesfwd.h"
#include "Math/GenVector/Quaternionfwd.h"
#include "Math/GenVector/RotationXfwd.h"
#include "Math/GenVector/RotationYfwd.h"
#include "Math/GenVector/RotationZfwd.h"
#include "Math/GenVector/Boost.h"
#include "Math/GenVector/BoostX.h"
#include "Math/GenVector/BoostY.h"
#include "Math/GenVector/BoostZ.h"

namespace ROOT {

  namespace Math {

  /**
     Lorentz transformation class with the (4D) transformation represented by
     a 4x4 orthosymplectic matrix.
     See also Boost, BoostX, BoostY and BoostZ for classes representing
     specialized Lorentz transformations.
     Also, the 3-D rotation classes can be considered to be special Lorentz
     transformations which do not mix space and time components.

     @ingroup GenVector

  */

class LorentzRotation {

public:

  typedef double Scalar;

  enum LorentzRotationMatrixIndex {
      XX =  0, XY =  1, XZ =  2, XT =  3
    , YX =  4, YY =  5, YZ =  6, YT =  7
    , ZX =  8, ZY =  9, ZZ = 10, ZT = 11
    , TX = 12, TY = 13, TZ = 14, TT = 15
  };

  // ========== Constructors and Assignment =====================

  /**
      Default constructor (identity transformation)
  */
  LorentzRotation();

  /**
     Construct given a pair of pointers or iterators defining the
     beginning and end of an array of sixteen Scalars
   */
  template<class IT>
  LorentzRotation(IT begin, IT end) { SetComponents(begin,end); }

  /**
     Construct from a pure boost 
  */

//explicit LorentzRotation( Boost  const &  ) {} // TODO
//explicit LorentzRotation( BoostX const &  ) {} // TODO
//explicit LorentzRotation( BoostY const &  ) {} // TODO
//explicit LorentzRotation( BoostZ const &  ) {} // TODO

  explicit LorentzRotation( Boost  const & b  ) {  b.GetLorentzRotation( fM+0 ); } 
  explicit LorentzRotation( BoostX const & bx ) { bx.GetLorentzRotation( fM+0 ); }
  explicit LorentzRotation( BoostY const & by ) { by.GetLorentzRotation( fM+0 ); }
  explicit LorentzRotation( BoostZ const & bz ) { bz.GetLorentzRotation( fM+0 ); }

  /**
     Construct from a 3-D rotation (no space-time mixing)
  */
  explicit LorentzRotation( Rotation3D  const & r ); 
  explicit LorentzRotation( AxisAngle   const & a ); 
  explicit LorentzRotation( EulerAngles const & e ); 
  explicit LorentzRotation( Quaternion  const & q ); 
  explicit LorentzRotation( RotationX   const & r ); 
  explicit LorentzRotation( RotationY   const & r ); 
  explicit LorentzRotation( RotationZ   const & r ); 

  /**
     Construct from a linear algebra matrix of size at least 4x4,
     which must support operator()(i,j) to obtain elements (0,3) thru (3,3).
     Precondition:  The matrix is assumed to be orthosymplectic.  NO checking
     or re-adjusting is performed.
     Note:  (0,0) refers to the XX component; (3,3) refers to the TT component.
  */
  template<class ForeignMatrix>
  explicit LorentzRotation(const ForeignMatrix & m) { SetComponents(m); }

  /**
     Construct from four orthosymplectic vectors (which must have methods
     x(), y(), z() and t()) which will be used as the columns of the Lorentz
     rotation matrix.  The orthosymplectic conditions will be checked, and 
     values adjusted so that the result will always be a good Lorentz rotation 
     matrix.
  */
  template<class Foreign4Vector>
  LorentzRotation(const Foreign4Vector& v1,
                  const Foreign4Vector& v2,
                  const Foreign4Vector& v3,
                  const Foreign4Vector& v4 ) { SetComponents(v1, v2, v3, v4); }

  // The compiler-generated copy ctor, copy assignment, and dtor are OK.

  /**
     Raw constructor from sixteen Scalar components (without any checking)
  */
  LorentzRotation(Scalar  xx, Scalar  xy, Scalar  xz, Scalar xt,
                  Scalar  yx, Scalar  yy, Scalar  yz, Scalar yt,
                  Scalar  zx, Scalar  zy, Scalar  zz, Scalar zt,
                  Scalar  tx, Scalar  ty, Scalar  tz, Scalar tt)
 {
    SetComponents (xx, xy, xz, xt, 
    		   yx, yy, yz, yt, 
                   zx, zy, zz, zt,
		   tx, ty, tz, tt);
 }

  /**
     Assign from a pure boost 
  */
  LorentzRotation &
  operator=( Boost  const & b ) { return operator=(LorentzRotation(b)); }
  LorentzRotation &
  operator=( BoostX const & b ) { return operator=(LorentzRotation(b)); }
  LorentzRotation &
  operator=( BoostY const & b ) { return operator=(LorentzRotation(b)); }
  LorentzRotation &
  operator=( BoostZ const & b ) { return operator=(LorentzRotation(b)); }

  /**
     Assign from a 3-D rotation 
  */
  LorentzRotation &
  operator=( Rotation3D  const & r ) { return operator=(LorentzRotation(r)); }
  LorentzRotation &
  operator=( AxisAngle   const & a ) { return operator=(LorentzRotation(a)); }
  LorentzRotation &
  operator=( EulerAngles const & e ) { return operator=(LorentzRotation(e)); }
  LorentzRotation &
  operator=( Quaternion  const & q ) { return operator=(LorentzRotation(q)); }
  LorentzRotation &
  operator=( RotationZ   const & r ) { return operator=(LorentzRotation(r)); }
  LorentzRotation &
  operator=( RotationY   const & r ) { return operator=(LorentzRotation(r)); }
  LorentzRotation &
  operator=( RotationX   const & r ) { return operator=(LorentzRotation(r)); }

  /**
     Assign from a linear algebra matrix of size at least 4x4,
     which must support operator()(i,j) to obtain elements (0,3) thru (3,3).
     Precondition:  The matrix is assumed to be orthosymplectic.  NO checking
     or re-adjusting is performed.
  */
  template<class ForeignMatrix>
  LorentzRotation &
  operator=(const ForeignMatrix & m) { SetComponents(m); return *this; }

  /**
     Re-adjust components to eliminate small deviations from a perfect
     orthosyplectic matrix.
   */
  void Rectify();

  // ======== Components ==============

  /**
     Set components from four orthosymplectic vectors (which must have methods
     x(), y(), z(), and t()) which will be used as the columns of the 
     Lorentz rotation matrix.  The values will be adjusted
     so that the result will always be a good Lorentz rotation matrix.
  */
  template<class Foreign4Vector>
  void
  SetComponents (const Foreign4Vector& v1,
                 const Foreign4Vector& v2,
                 const Foreign4Vector& v3,
                 const Foreign4Vector& v4 ) {
    fM[XX]=v1.x();  fM[XY]=v2.x();  fM[XZ]=v3.x();  fM[XT]=v4.x();
    fM[YX]=v1.y();  fM[YY]=v2.y();  fM[YZ]=v3.y();  fM[YT]=v4.y();
    fM[ZX]=v1.z();  fM[ZY]=v2.z();  fM[ZZ]=v3.z();  fM[ZT]=v4.z();
    fM[TX]=v1.t();  fM[TY]=v2.t();  fM[TZ]=v3.t();  fM[TT]=v4.t();
    Rectify();
  }

  /**
     Get components into four 4-vectors which will be the (orthosymplectic) 
     columns of the rotation matrix.  (The 4-vector class must have a 
     constructor from 4 Scalars used as x, y, z, t) 
  */
  template<class Foreign4Vector>
  void
  GetComponents ( Foreign4Vector& v1,
                  Foreign4Vector& v2,
                  Foreign4Vector& v3,
                  Foreign4Vector& v4 ) const {
    v1 = Foreign4Vector ( fM[XX], fM[YX], fM[ZX], fM[TX] );
    v2 = Foreign4Vector ( fM[XY], fM[YY], fM[ZY], fM[TY] );
    v3 = Foreign4Vector ( fM[XZ], fM[YZ], fM[ZZ], fM[TZ] );
    v4 = Foreign4Vector ( fM[XT], fM[YT], fM[ZT], fM[TT] );
  }

  /**
     Set the 16 matrix components given an iterator to the start of
     the desired data, and another to the end (16 past start).
   */
  template<class IT>
  void SetComponents(IT begin, IT end) {
    assert (end==begin+16);
    std::copy ( begin, end, fM+0 );
  }

  /**
     Get the 16 matrix components into data specified by an iterator begin
     and another to the end of the desired data (16 past start).
   */
  template<class IT>
  void GetComponents(IT begin, IT end) const {
    assert (end==begin+16);
    std::copy ( fM+0, fM+16, begin );
  }

  /**
     Set components from a linear algebra matrix of size at least 4x4,
     which must support operator()(i,j) to obtain elements (0,0) thru (3,3).
     Precondition:  The matrix is assumed to be orthosymplectic.  NO checking
     or re-adjusting is performed.
  */
  template<class ForeignMatrix>
  void
  SetComponents (const ForeignMatrix & m) {
    fM[XX]=m(0,0);  fM[XY]=m(0,1);  fM[XZ]=m(0,2);  fM[XT]=m(0,3);
    fM[YX]=m(1,0);  fM[YY]=m(1,1);  fM[YZ]=m(1,2);  fM[YT]=m(1,3);
    fM[ZX]=m(2,0);  fM[ZY]=m(2,1);  fM[ZZ]=m(2,2);  fM[ZT]=m(2,3);
    fM[TX]=m(3,0);  fM[TY]=m(3,1);  fM[TZ]=m(3,2);  fM[TT]=m(3,3);
  }

  /**
     Get components into a linear algebra matrix of size at least 4x4,
     which must support operator()(i,j) for write access to elements
     (0,0) thru (3,3).
  */
  template<class ForeignMatrix>
  void
  GetComponents (ForeignMatrix & m) const {
    m(0,0)=fM[XX];  m(0,1)=fM[XY];  m(0,2)=fM[XZ]; m(0,3)=fM[XT];
    m(1,0)=fM[YX];  m(1,1)=fM[YY];  m(1,2)=fM[YZ]; m(1,3)=fM[YT];
    m(2,0)=fM[ZX];  m(2,1)=fM[ZY];  m(2,2)=fM[ZZ]; m(2,3)=fM[ZT];
    m(3,0)=fM[TX];  m(3,1)=fM[TY];  m(3,2)=fM[TZ]; m(3,3)=fM[TT];
  }

  /**
     Set the components from sixteen scalars -- UNCHECKED for orthosymplectic
   */
  void
  SetComponents (Scalar  xx, Scalar  xy, Scalar  xz, Scalar  xt,
                 Scalar  yx, Scalar  yy, Scalar  yz, Scalar  yt,
                 Scalar  zx, Scalar  zy, Scalar  zz, Scalar  zt,
                 Scalar  tx, Scalar  ty, Scalar  tz, Scalar  tt) {
                 fM[XX]=xx;  fM[XY]=xy;  fM[XZ]=xz;  fM[XT]=xt;
                 fM[YX]=yx;  fM[YY]=yy;  fM[YZ]=yz;  fM[YT]=yt;
                 fM[ZX]=zx;  fM[ZY]=zy;  fM[ZZ]=zz;  fM[ZT]=zt;
                 fM[TX]=tx;  fM[TY]=ty;  fM[TZ]=tz;  fM[TT]=tt;
  }

  /**
     Get the sixteen components into sixteen scalars
   */
  void
  GetComponents (Scalar &xx, Scalar &xy, Scalar &xz, Scalar &xt,
                 Scalar &yx, Scalar &yy, Scalar &yz, Scalar &yt,
                 Scalar &zx, Scalar &zy, Scalar &zz, Scalar &zt,
                 Scalar &tx, Scalar &ty, Scalar &tz, Scalar &tt) const {
                 xx=fM[XX];  xy=fM[XY];  xz=fM[XZ];  xt=fM[XT];
                 yx=fM[YX];  yy=fM[YY];  yz=fM[YZ];  yt=fM[YT];
                 zx=fM[ZX];  zy=fM[ZY];  zz=fM[ZZ];  zt=fM[ZT];
                 tx=fM[TX];  ty=fM[TY];  tz=fM[TZ];  tt=fM[TT];
  }

  // =========== operations ==============

  /**
     Lorentz transformation operation on a Minkowski ('Cartesian') 
     LorentzVector
  */
  LorentzVector< ROOT::Math::PxPyPzE4D<double> >
  operator() (const LorentzVector< ROOT::Math::PxPyPzE4D<double> > & v) const;
  
  /**
     Lorentz transformation operation on a LorentzVector in any 
     coordinate system
   */
  template <class CoordSystem>
  LorentzVector<CoordSystem>
  operator() (const LorentzVector<CoordSystem> & v) const {
    LorentzVector< PxPyPzE4D<double> > xyzt(v);
    LorentzVector< PxPyPzE4D<double> > Rxyzt = operator()(xyzt);
    return LorentzVector<CoordSystem> ( Rxyzt );
  }

  /**
     Lorentz transformation operation on an arbitrary 4-vector v.
     Preconditions:  v must implement methods x(), y(), z(), and t()
     and the arbitrary vector type must have a constructor taking (x,y,z,t)
   */
  template <class Foreign4Vector>
  Foreign4Vector
  operator() (const Foreign4Vector & v) const {
    LorentzVector< PxPyPzE4D<double> > xyzt(v);
    LorentzVector< PxPyPzE4D<double> > Rxyzt = operator()(xyzt);
    return Foreign4Vector ( Rxyzt.X(), Rxyzt.Y(), Rxyzt.Z(), Rxyzt.T() );
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
      Invert a Lorentz rotation in place
   */
  void Invert();

  /**
      Return inverse of  a rotation
   */
  LorentzRotation Inverse() const;

  // ========= Multi-Rotation Operations ===============

  /**
     Multiply (combine) this Lorentz rotation by another LorentzRotation
   */
  LorentzRotation operator * (const LorentzRotation & r) const;

#ifdef TODO_LATER
  /**
     Multiply (combine) this Lorentz rotation by a pure Lorentz boost
   */
  LorentzRotation operator * (const Boost  & b) const; // TODO
  LorentzRotation operator * (const BoostX & b) const; // TODO
  LorentzRotation operator * (const BoostY & b) const; // TODO
  LorentzRotation operator * (const BoostZ & b) const; // TODO

  /**
     Multiply (combine) this Lorentz rotation by a 3-D Rotation
   */
  LorentzRotation operator * (const Rotation3D  & r) const; // TODO
  LorentzRotation operator * (const AxisAngle   & a) const; // TODO
  LorentzRotation operator * (const EulerAngles & e) const; // TODO
  LorentzRotation operator * (const Quaternion  & q) const; // TODO
  LorentzRotation operator * (const RotationX  & rx) const; // TODO
  LorentzRotation operator * (const RotationY  & ry) const; // TODO
  LorentzRotation operator * (const RotationZ  & rz) const; // TODO
#endif

  /**
     Post-Multiply (on right) by another LorentzRotation, Boost, or 
     rotation :  T = T*R
   */
  template <class R>
  LorentzRotation & operator *= (const R & r) { return *this = (*this)*r; }

  /**
     Equality/inequality operators
   */
  bool operator == (const LorentzRotation & rhs) {
    for (unsigned int i=0; i < 16; ++i) {
      if( fM[i] != rhs.fM[i] )  return false;
    }
    return true;
  }
  bool operator != (const LorentzRotation & rhs) {
    return ! operator==(rhs);
  }

private:

  Scalar fM[16];

};  // LorentzRotation

// ============ Class LorentzRotation ends here ============

// ============================================ vetted to here  ============

#ifdef NOTYET
/**
   Distance between two Lorentz rotations
 */
template <class R>
inline
typename Rotation3D::Scalar
Distance ( const Rotation3D& r1, const R & r2) {return gv_detail::dist(r1,r2);}
#endif

} //namespace Math
} //namespace ROOT







#endif /* ROOT_Math_GenVector_LorentzRotation  */
