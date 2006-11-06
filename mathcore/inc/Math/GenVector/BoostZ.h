// @(#)root/mathcore:$Name:  $:$Id: BoostZ.h,v 1.5 2006/06/15 16:23:44 moneta Exp $
// Authors: W. Brown, M. Fischler, L. Moneta    2005  

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 ROOT FNAL MathLib Team                          *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for BoostZ
// 
// Created by: Mark Fischler  Mon Nov 1  2005
// 
// Last update: $Id: BoostZ.h,v 1.5 2006/06/15 16:23:44 moneta Exp $
// 
#ifndef ROOT_Math_GenVector_BoostZ
#define ROOT_Math_GenVector_BoostZ 1

#include "Math/GenVector/LorentzVector.h"
#include "Math/GenVector/PxPyPzE4D.h"
#include "Math/GenVector/DisplacementVector3D.h"
#include "Math/GenVector/Cartesian3D.h"

namespace ROOT {

  namespace Math {

  /**
     Lorentz boost class with the (4D) transformation represented internally
     by a 4x4 orthosymplectic matrix.
     Also, the 3-D rotation classes can be considered to be special Lorentz
     transformations which do not mix space and time components.

     @ingroup GenVector

  */

class BoostZ {

public:

  typedef double Scalar;

  enum LorentzRotationMatrixIndex {
      kLXX =  0, kLXY =  1, kLXZ =  2, kLXT =  3
    , kLYX =  4, kLYY =  5, kLYZ =  6, kLYT =  7
    , kLZX =  8, kLZY =  9, kLZZ = 10, kLZT = 11
    , kLTX = 12, kLTY = 13, kLTZ = 14, kLTT = 15
  };

  enum BoostMatrixIndex {
      kXX =  0, kXY =  1, kXZ =  2, kXT =  3
    	      , kYY =  4, kYZ =  5, kYT =  6
    		        , kZZ =  7, kZT =  8
    			          , kTT =  9
  };

  // ========== Constructors and Assignment =====================

  /**
      Default constructor (identity transformation)
  */
  BoostZ();

  /**
     Construct given a Scalar beta_z
   */
  BoostZ(Scalar beta_z) { SetComponents(beta_z); }


  // The compiler-generated copy ctor, copy assignment, and dtor are OK.

  /**
     Re-adjust components to eliminate small deviations from a perfect
     orthosyplectic matrix.
   */
  void Rectify();

  // ======== Components ==============

  /**
     Set components from a Scalar beta_z
  */
  void
  SetComponents (Scalar beta_z);

  /**
     Get components into a Scalar beta_z
  */
  void
  GetComponents (Scalar& beta_z) const;


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
  void  SetBeta(Scalar beta) { SetComponents(beta); }
   
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
    LorentzVector< PxPyPzE4D<double> > r_xyzt = operator()(xyzt);
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
    LorentzVector< PxPyPzE4D<double> > xyzt(v);
    LorentzVector< PxPyPzE4D<double> > r_xyzt = operator()(xyzt);
    return Foreign4Vector ( r_xyzt.X(), r_xyzt.Y(), r_xyzt.Z(), r_xyzt.T() );
  }

  /**
     Overload operator * for boost on a vector
   */
  template <class A4Vector>
  inline
  A4Vector operator* (const A4Vector & v) const
  {
    return operator()(v);
  }

  /**
      Invert a BoostZ in place and return a reference to the inverted boost class
   */
  BoostZ & Invert();

  /**
      Return inverse of  a BoostZ
   */
  BoostZ Inverse() const;

  /**
     Equality/inequality operators
   */
  bool operator == (const BoostZ & rhs) const {
    if( fBeta  != rhs.fBeta  ) return false;
    if( fGamma != rhs.fGamma ) return false;
    return true;
  }
  bool operator != (const BoostZ & rhs) const {
    return ! operator==(rhs);
  }

private:

  Scalar fBeta;
  Scalar fGamma;

};  // BoostZ

// ============ Class BoostZ ends here ============

/**
   Stream Output and Input
 */
  // TODO - I/O should be put in the manipulator form 

std::ostream & operator<< (std::ostream & os, const BoostZ & b);


} //namespace Math
} //namespace ROOT







#endif /* ROOT_Math_GenVector_BoostZ  */
