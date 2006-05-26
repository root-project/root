// @(#)root/mathcore:$Name:  $:$Id: Boost.h,v 1.3 2006/04/11 13:06:15 moneta Exp $
// Authors: W. Brown, M. Fischler, L. Moneta    2005  

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 ROOT FNAL MathLib Team                          *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for Boost
// 
// Created by: Mark Fischler  Mon Nov 1  2005
// 
// Last update: $Id: Boost.h,v 1.3 2006/04/11 13:06:15 moneta Exp $
// 
#ifndef ROOT_Math_GenVector_Boost
#define ROOT_Math_GenVector_Boost 1

#include "Math/GenVector/LorentzVector.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "Math/GenVector/DisplacementVector3D.h"
#include "Math/GenVector/Cartesian3D.h"

#include "Math/GenVector/BoostX.h"
#include "Math/GenVector/BoostY.h"
#include "Math/GenVector/BoostZ.h"

namespace ROOT {

  namespace Math {

  /**
     Lorentz boost class with the (4D) transformation represented internally
     by a 4x4 orthosymplectic matrix.
     See also BoostX, BoostY and BoostZ for classes representing
     specialized Lorentz boosts.
     Also, the 3-D rotation classes can be considered to be special Lorentz
     transformations which do not mix space and time components.

     @ingroup GenVector

  */

class Boost {

public:

  typedef double Scalar;

  enum LorentzRotationMatrixIndex {
      LXX =  0, LXY =  1, LXZ =  2, LXT =  3
    , LYX =  4, LYY =  5, LYZ =  6, LYT =  7
    , LZX =  8, LZY =  9, LZZ = 10, LZT = 11
    , LTX = 12, LTY = 13, LTZ = 14, LTT = 15
  };

  enum BoostMatrixIndex {
      XX =  0, XY =  1, XZ =  2, XT =  3
    	     , YY =  4, YZ =  5, YT =  6
    		      , ZZ =  7, ZT =  8
    			       , TT =  9
  };

  // ========== Constructors and Assignment =====================

  /**
      Default constructor (identity transformation)
  */
  Boost() { SetIdentity(); }

  /**
     Construct given a three Scalars beta_x, beta_y, and beta_z
   */
  Boost(Scalar beta_x, Scalar beta_y, Scalar beta_z) 
  				{ SetComponents(beta_x, beta_y, beta_z); }

  /**
     Construct given a beta vector (which must have methods x(), y(), z())
   */
  template <class Avector>
  explicit
  Boost(const Avector & beta) { SetComponents(beta); }

  /**
     Construct given a pair of pointers or iterators defining the
     beginning and end of an array of three Scalars to use as beta_x, _y, and _z
   */
  template<class IT>
  Boost(IT begin, IT end) { SetComponents(begin,end); }

  /**
     Construct from an axial boost 
  */

  explicit Boost( BoostX const & bx ) {SetComponents(bx.BetaVector());} 
  explicit Boost( BoostY const & by ) {SetComponents(by.BetaVector());} 
  explicit Boost( BoostZ const & bz ) {SetComponents(bz.BetaVector());} 

  // The compiler-generated copy ctor, copy assignment, and dtor are OK.

  /**
     Assign from an axial pure boost 
  */
  Boost &
  operator=( BoostX const & bx ) { return operator=(Boost(bx)); }
  Boost &
  operator=( BoostY const & by ) { return operator=(Boost(by)); }
  Boost &
  operator=( BoostZ const & bz ) { return operator=(Boost(bz)); }

  /**
     Re-adjust components to eliminate small deviations from a perfect
     orthosyplectic matrix.
   */
  void Rectify();

  // ======== Components ==============

  /**
     Set components from beta_x, beta_y, and beta_z
  */
  void
  SetComponents (Scalar beta_x, Scalar beta_y, Scalar beta_z);

  /**
     Get components into beta_x, beta_y, and beta_z 
  */
  void
  GetComponents (Scalar& beta_x, Scalar& beta_y, Scalar& beta_z) const;

  /**
     Set components from a beta vector 
  */
  template <class Avector>
  void
  SetComponents (const Avector & beta) 
		  	{ SetComponents(beta.x(), beta.y(), beta.z()); } 

  /**
     Set given a pair of pointers or iterators defining the beginning and end of 
     an array of three Scalars to use as beta_x,beta _y, and beta_z
   */
  template<class IT>
  void SetComponents(IT begin, IT end) {
    assert (end==begin+3);
    SetComponents (*begin, *(begin+1), *(begin+2));
  }

  /**
     Get given a pair of pointers or iterators defining the beginning and end of 
     an array of three Scalars into which to place beta_x, beta_y, and beta_z
   */
  template<class IT>
  void GetComponents(IT begin, IT end) const {
    assert (end==begin+3);
    GetComponents (*begin, *(begin+1), *(begin+2));
  }
   
  /**
     The beta vector for this boost
   */
  typedef  DisplacementVector3D<Cartesian3D<double>, DefaultCoordinateSystemTag > XYZVector; 
  XYZVector BetaVector() const;

  /**
     Get elements of internal 4x4 symmetric representation, into a data
     array suitable for direct use as the components of a LorentzRotation
     Note -- 16 Scalars will be written into the array; if the array is not
     that large, then this will lead to undefined behavior.
  */
  void 
  GetLorentzRotation (Scalar r[]) const; 
  
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
      Invert a Boost in place
   */
  void Invert();

  /**
      Return inverse of  a rotation
   */
  Boost Inverse() const;

  /**
     Equality/inequality operators
   */
  bool operator == (const Boost & rhs) {
    for (unsigned int i=0; i < 10; ++i) {
      if( fM[i] != rhs.fM[i] )  return false;
    }
    return true;
  }
  bool operator != (const Boost & rhs) {
    return ! operator==(rhs);
  }

protected: 

  void SetIdentity();

private:

  Scalar fM[10];

};  // Boost

// ============ Class Boost ends here ============

/**
   Stream Output and Input
 */
  // TODO - I/O should be put in the manipulator form 

std::ostream & operator<< (std::ostream & os, const Boost & b);


} //namespace Math
} //namespace ROOT







#endif /* ROOT_Math_GenVector_Boost  */
