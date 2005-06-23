// @(#)root/mathcore:$Name:  $:$Id: Polar3D.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: Mark Fischler & Lorenzo Moneta   06/2005 

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for class Polar3D
// 
// Created by: Lorenzo Moneta  at Mon May 30 11:40:03 2005
// 
// Last update: mf Tue Jun 16 2005 
// 
#ifndef ROOT_MATH_POLAR3D
#define ROOT_MATH_POLAR3D 1

#include "GenVector/etaMax.h"

#include <cmath>

namespace ROOT { 

  namespace Math { 


  /** 
      Class describing a polar coordinate system based on r, theta and phi
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
  Polar3D(T r,T theta,T phi) : fR(r), fTheta(theta), fPhi(phi) {  }

  /**
     Construct from any Vector or coordinate system implementing 
     R(), Theta() and Phi()
    */ 
  template <class CoordSystem > 
  explicit Polar3D( const CoordSystem & v ) : 
    fR(v.R() ),  fTheta(v.Theta() ),  fPhi(v.Phi() )  {  } 

  // no reason for a cusom destructor  ~Cartesian3D() {}

  /**
     Set internal data based on an array of 3 Scalar numbers
   */ 
  void SetCoordinates( const Scalar * src ) 
  			{ fR=src[0]; fTheta=src[1]; fPhi=src[2]; }

  /**
     get internal data into an array of 3 Scalar numbers
   */ 
  void GetCoordinates( Scalar * dest ) const 
  			{ dest[0] = fR; dest[1] = fTheta; dest[2] = fPhi; }

   
  T R()     const { return fR;}
  T Phi()   const { return fPhi; }
  T Theta() const { return fTheta; } 
  T Rho()   const { return fR*std::sin(fTheta); }
  T X()     const { return Rho()*std::cos(fPhi);}
  T Y()     const { return Rho()*std::sin(fPhi);}
  T Z()     const { return fR*std::cos(fTheta); } 
  T Mag2()  const { return fR*fR;}
  T Perp2() const { return Rho()*Rho(); }

  // pseudorapidity
  T Eta() const 
  { if (fTheta != 0) {
      return -log( tan( fTheta/2.));
      // given that we already have theta, this method 
      // should be faster than the one used from cartesian coordinates
    } else { 
      return fR + etaMax<T>();
      // Note - if fTheta is exaclt pi, we would want to return -fR - etaMax,
      //        but fTheta can never be **exactly** pi 
    }
  }

  // setters (only for data members) 

  /**
     set all the data members ( r, theta, phi) 
   */ 
  void setValues(const T & r, const T & theta, const T & phi) { 
    fR = r;  
    fTheta = theta;  
    fPhi = phi;   
  }
  
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
  }

  /** 
      scale by a scalar quantity - for polar coordinates r changes
  */
  void Scale (const T & a) { 
    // angles do not change when scaling
    fR *= a;     
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

  // ============= Compatibility secition ==================
  
  // The following make this coordinate system look enough like a CLHEP
  // vector that an assignment member template can work with either
  T x() const { return X();}
  T y() const { return Y();}
  T z() const { return Z(); } 
  
  // ============= Specializations for improved speed ==================

  // (none)

private:
  T fR;
  T fTheta;
  T fPhi;
};



  } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_MATH_POLAR3D */
