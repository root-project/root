// @(#)root/mathcore:$Name:  $:$Id: Rotation3D.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: W. Brown, M. Fischler, L. Moneta, A. Zsenei   06/2005 

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT FNAL MathLib Team                    *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for class Rotation in 3 dimensions, represented by 3x3 matrix
//
// Created by: Mark Fischler Thurs June 9  2005
//
// Last update: Wed June 22  2005
//
#ifndef ROOT_Math_Rotation3D 
#define ROOT_Math_Rotation3D 1

#include "MathCore/Vector3Dfwd.h"
#include "MathCore/DisplacementVector3D.h"
#include "MathCore/Cartesian3D.h"


#include <algorithm>
#include <cassert>

namespace ROOT {

  namespace Math {


  /**
     Rotation class with the (3D) rotation represented by
     a 3x3 orthogonal matrix.
     This is the optimal representation for application to vectors.
     See also AxisAngle, EulerAngles, and Quaternion for classes
     which have conversion operators to Rotation3D.
  */

// TODO - remove class Test
class Test {
  template <class T>
  Test (T begin) { }
};

class Rotation3D {

public:

  typedef double Scalar;

  enum Rotation3DMatrixIndex {
      XX = 0, XY = 1, XZ = 2
    , YX = 3, YY = 4, YZ = 5
    , ZX = 6, ZY = 7, ZZ = 8
    };

  // ========== Constructors and Assignment =====================

  /**
      Default constructor (identity rotation)
  */
  Rotation3D();

  /**
     Construct given a pair of pointers or iterators defining the
     beginning and end of an array of nine Scalars
   */
  template<class IT>
  Rotation3D(IT begin, IT end) { SetComponents(begin,end); }

  /**
     Construct from a linear algebra matrix of size at least 3x3,
     which must support operator()(i,j) to obtain elements (0,0) thru (2,2).
     Precondition:  The matrix is assumed to be orthonormal.  NO checking
     or re-adjusting is performed.
  */
  template<class ForeignMatrix>
  explicit Rotation3D(const ForeignMatrix & m) { SetComponents(m); }

  /**
     Construct from three orthonormal vectors (which must have methods
     x(), y() and z()) which will be used as the columns of the rotation
     matrix.  The orthonormality will be checked, and values adjusted
     so that the result will always be a good rotation matrix.
  */
  template<class ForeignVector>
  Rotation3D(const ForeignVector& v1,
             const ForeignVector& v2,
             const ForeignVector& v3 ) {
    fM[XX]=v1.x();  fM[XY]=v2.x();  fM[XZ]=v3.x();
    fM[YX]=v1.y();  fM[YY]=v2.y();  fM[YZ]=v3.y();
    fM[ZX]=v1.z();  fM[ZY]=v2.z();  fM[ZZ]=v3.z();
    Rectify();
  }

  // The compiler-generated copy ctor, copy assignment, and dtor are OK.

  /**
     Assign from an orthonormal linear algebra matrix of size 3x3,
     which must support operator()(i,j) to obtain elements (0,0) thru (2,2).
  */
  template<class ForeignMatrix>
  Rotation3D &
  operator=(const ForeignMatrix & m) { SetComponents(m); return *this; }


  // Assignment from a ForeignMatrix is fine proceeding via the conversion ctor.

  /**
     Re-adjust components to eliminate small deviations from perfect
     orthonormality.
   */
  void Rectify();

  // ======== Components ==============

  /**
     Set the 9 matrix components given an iterator to the start of
     the desired data, and another to the endd (9 past start).
   */
  template<class IT>
  void SetComponents(IT begin, IT end) {
    assert (end==begin+9);
    std::copy ( begin, end, fM );
  }

  /**
     Get the 9 matrix components into data specified by an iterator begin
     and another to the end of the desired data (9 past start).
   */
  template<class IT>
  void GetComponents(IT begin, IT end) const {
    assert (end==begin+9);
    std::copy ( fM, fM+9, begin );
  }

  /**
     Set components from a linear algebra matrix of size at least 3x3,
     which must support operator()(i,j) to obtain elements (0,0) thru (2,2).
     Precondition:  The matrix is assumed to be orthonormal.  NO checking
     or re-adjusting is performed.
  */
  template<class ForeignMatrix>
  void
  SetComponents (const ForeignMatrix & m) {
    fM[XX]=m(0,0);  fM[XY]=m(0,1);  fM[XZ]=m(0,2);
    fM[YX]=m(1,0);  fM[YY]=m(1,1);  fM[YZ]=m(1,2);
    fM[ZX]=m(2,0);  fM[ZY]=m(2,1);  fM[ZZ]=m(2,2);
  }

  /**
     Set components into a linear algebra matrix of size at least 3x3,
     which must support operator()(i,j) for write access to elements
     (0,0) thru (2,2).
  */
  template<class ForeignMatrix>
  void
  GetComponents (ForeignMatrix & m) const {
    m(0,0)=fM[XX];  m(0,1)=fM[XY];  m(0,2)=fM[XZ];
    m(1,0)=fM[YX];  m(1,1)=fM[YY];  m(1,2)=fM[YZ];
    m(2,0)=fM[ZX];  m(2,1)=fM[ZY];  m(2,2)=fM[ZZ];
  }

  // =========== operations ==============

  /**
     Rotation operation on a cartesian vector
   */

//   ::ROOT::Math::DisplacementVector3D<Cartesian3D<double> > 
//   operator() (const DisplacementVector3D<Cartesian3D<double> > & v) const;
// for CINT (why ??) works only if I have XYZVector
  XYZVector 
  operator() (const XYZVector & v) const;

  /**
     Rotation operation on any vector in any coordinate system
     (DisplacementVector3D, PositionVector3D)
   */
  template <class AVector>
  AVector operator() (const AVector & v) const {
    return AVector(operator()( ::ROOT::Math::DisplacementVector3D< Cartesian3D<double> >(v)));
  }

  // TODO - LorentzVector rotation

  /**
     Overload operator * for rotation on a vector
   */
  template <class AVector>
  inline
  AVector operator* (const AVector & v) const
  {
    return this->operator()(v);
  }

  /**
     Multiply (combine) two rotations
   */
  Rotation3D operator * (const Rotation3D & r) const;

  /**
     Post-Multiply (on right) by another rotation :  T = T*R
   */
  Rotation3D & operator *= (const Rotation3D & r) { return *this = (*this)*r; }

  /**
      Invert a rotation in place
   */
  void Invert();

  /**
      Return inverse of  a rotation
   */
  Rotation3D Inverse() const { Rotation3D t(*this); t.Invert(); return t; }


private:

    Scalar fM[9];

  };

} //namespace Math
} //namespace ROOT

#endif // ROOT_Math_Rotation3D 
