// @(#)root/mathcore:$Name:  $:$Id: Cartesian3D.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: W. Brown, M. Fischler, L. Moneta, A. Zsenei   06/2005 

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for class Cartesian3D
// 
// Created by: Lorenzo Moneta  at Mon May 30 11:16:56 2005
// 
// Last update: mf Tue Jun 16 2005 
// 
#ifndef ROOT_Math_Cartesian3D 
#define ROOT_Math_Cartesian3D 1

#include "MathCore/etaMax.h"
#include "MathCore/Polar3Dfwd.h"

#include <cmath>

namespace ROOT { 

  namespace Math { 

  /** 
      Class describing a 3D cartesian coordinate system
      (x, y, z coordinates) 
   */ 

template <class T> 
class Cartesian3D { 

public : 

  typedef T Scalar;

  /**
     Default constructor  with x=y=z=0 
   */
  Cartesian3D() : fX(0), fY(0), fZ(0) {  }

  /**
     Constructor from x,y,z coordinates
   */
  Cartesian3D(T x, T y, T z) : fX(x), fY(y), fZ(z) {  } 

  /**
     Construct from any Vector or coordinate system implementing 
     X(), Y() and Z()
  */
  template <class CoordSystem> 
  explicit Cartesian3D(const CoordSystem & v) 
  				: fX(v.X()), fY(v.Y()), fZ(v.Z()) {  }

  // no reason for a custom destructor  ~Cartesian3D() {} and copy constructor

  /**
     Set internal data based on an array of 3 Scalar numbers
   */ 
  void SetCoordinates( const Scalar * src ) { fX=src[0]; fY=src[1]; fZ=src[2]; }

  /**
     get internal data into an array of 3 Scalar numbers
   */ 
  void GetCoordinates( Scalar * dest ) const 
  				{ dest[0] = fX; dest[1] = fY; dest[2] = fZ; }

  /**
     Set internal data based on 3 Scalar numbers
   */ 
  void SetCoordinates(Scalar  x, Scalar  y, Scalar  z) { fX=x; fY=y; fZ=z; }

  /**
     get internal data into 3 Scalar numbers
   */ 
  void GetCoordinates(Scalar& x, Scalar& y, Scalar& z) const {x=fX; y=fY; z=fZ;}  				


  T X()     const { return fX;}
  T Y()     const { return fY;}
  T Z()     const { return fZ;}
  T Mag2()  const { return fX*fX + fY*fY + fZ*fZ;}
  T Perp2() const { return fX*fX + fY*fY ;}
  T Rho()   const { return std::sqrt( Perp2());}
  T R()     const { return std::sqrt( Mag2());}
  T Theta() const { return (fX==0 && fY==0 && fZ==0) ? 0.0 : atan2(Rho(),Z());}
  T Phi()   const { return (fX==0 && fY==0) ? 0.0 : atan2(fY,fX);}
 
  // pseudorapidity
  // T Eta() const { return -log( tan( theta()/2.));} 
  T Eta() const 
  { T rho = Rho();
    if (rho > 0) {
      T z_scaled(fZ/rho);
      return std::log(z_scaled+std::sqrt(z_scaled*z_scaled+1)); // faster 
    } else if (fZ==0) {
      return 0;
    } else if (fZ>0) {
      return fZ + etaMax<T>();
    }  else {
      return fZ - etaMax<T>();
    }
  }

  /** 
      set the x coordinate value keeping y and z constant
   */ 
  void SetX(T x) { fX = x; }

  /** 
      set the y coordinate value keeping x and z constant
   */ 
  void SetY(T y) { fY = y; }

  /** 
      set the z coordinate value keeping x and y constant
   */ 
  void SetZ(T z) { fZ = z; }

  /**
     scale coordinate values by a scalar quantity a
   */
  void Scale(T a) { fX *= a; fY *= a;  fZ *= a; }

  /**
     Assignment from any class implementing X(), Y() and Z()
     (can assign from any coordinate system) 
  */
  template <class CoordSystem> 
  Cartesian3D & operator = (const CoordSystem & v) { 
    fX = v.X();  
    fY = v.Y();  
    fZ = v.Z();  
    return *this;
  }

  // ============= Compatibility secition ==================
  
  // The following make this coordinate system look enough like a CLHEP
  // vector that an assignment member template can work with either
  T x() const { return X();}
  T y() const { return Y();}
  T z() const { return Z(); } 
  
  // ============= Overloads for improved speed ==================

  template <class T2>
  explicit Cartesian3D( const Polar3D<T2> & v ) : fZ (v.Z())
  {
    T rho = v.Rho(); // re-using this instead of calling v.X() and v.Y()
                     // is the speed improvement
    fX = rho * std::cos(v.Phi());
    fY = rho * std::sin(v.Phi());    
  } 

  template <class T2>
  Cartesian3D & operator = (const Polar3D<T2> & v) 
  { 
    T rho = v.Rho(); 
    fX = rho * std::cos(v.Phi());
    fY = rho * std::sin(v.Phi());
    fZ = v.Z();
    return *this;
  }



private:

  /**
     (Contiguous) data containing the coordinates values x,y,z
   */
  T fX;
  T fY;
  T fZ; 
};

  } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_Cartesian3D */
