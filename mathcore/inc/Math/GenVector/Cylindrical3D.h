// @(#)root/mathcore:$Name: v5-10-00 $:$Id: Cylindrical3D.h,v 1.2 2006/02/06 17:22:03 moneta Exp $
// Authors: W. Brown, M. Fischler, L. Moneta    2005  

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT MathLib Team  and                    *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for class Cylindrica3D
// 
// Created by: Lorenzo Moneta  at Tue Dec 06 2005
// 
// 
#ifndef ROOT_Math_GenVector_Cylindrical3D 
#define ROOT_Math_GenVector_Cylindrical3D  1

#include "Math/GenVector/etaMax.h"

#include <cmath>
#include <limits>

#if defined(__MAKECINT__) || defined(G__DICTIONARY) 
#include "Math/GenVector/GenVector_exception.h"
#include "Math/GenVector/Cartesian3D.h"
#include "Math/GenVector/Polar3D.h"
#include "Math/GenVector/CylindricalEta3D.h"
#endif
 

namespace ROOT { 

  namespace Math { 

  /** 
      Class describing a cylindrical coordinate system based on rho, z and phi.   
      The base coordinates are rho (transverse component) , z and phi
  
      @ingroup GenVector
   */ 

template <class T> 
class Cylindrical3D { 

public : 

  typedef T Scalar;

  /**
     Default constructor with rho=z=phi=0
   */
  Cylindrical3D() : fRho(0), fZ(0), fPhi(0) {  }

  /**
     Construct from rho eta and phi values
   */
  Cylindrical3D(Scalar rho, Scalar z, Scalar phi) :  
    fRho(rho), fZ(z), fPhi(phi) { Restrict(); }

  /**
     Construct from any Vector or coordinate system implementing 
     Rho(), Z() and Phi()
  */ 
  template <class CoordSystem > 
  explicit Cylindrical3D( const CoordSystem & v ) : 
    fRho( v.Rho() ),  fZ( v.Z() ),  fPhi( v.Phi() ) { Restrict(); } 

  // no reason for a custom destructor  ~Cartesian3D() {}

  /**
     Set internal data based on an array of 3 Scalar numbers ( rho, z , phi)
   */ 
  void SetCoordinates( const Scalar src[] ) 
  			{ fRho=src[0]; fZ=src[1]; fPhi=src[2]; Restrict(); }

  /**
     get internal data into an array of 3 Scalar numbers ( rho, z , phi)
   */ 
  void GetCoordinates( Scalar dest[] ) const 
  			{ dest[0] = fRho; dest[1] = fZ; dest[2] = fPhi; }

  /**
     Set internal data based on 3 Scalar numbers ( rho, z , phi)
   */ 
  void SetCoordinates(Scalar  rho, Scalar  z, Scalar  phi) 
  			{ fRho=rho; fZ=z; fPhi=phi; Restrict(); }

  /**
     get internal data into 3 Scalar numbers ( rho, z , phi)
   */ 
  void GetCoordinates(Scalar& rho, Scalar& z, Scalar& phi) const 
  			{rho=fRho; z=fZ; phi=fPhi;}  				

  private:
    inline static double pi() { return 3.14159265358979323; } 
    inline void Restrict() {
      if ( fPhi <= -pi() || fPhi > pi() ) 
        fPhi = fPhi - std::floor( fPhi/(2*pi()) +.5 ) * 2*pi();
    return;
    } 
  public:
   
  // accessors
 
  Scalar Rho()   const { return fRho; }
  Scalar Z()     const { return fZ;   }
  Scalar Phi()   const { return fPhi; }
  
  Scalar X()     const { return fRho*std::cos(fPhi); }
  Scalar Y()     const { return fRho*std::sin(fPhi); }

  Scalar Mag2()  const { return fRho*fRho + fZ*fZ;   }
  Scalar R()     const { return std::sqrt( Mag2());  }
  Scalar Perp2() const { return fRho*fRho;           }
  Scalar Theta() const { return (fRho==0 && fZ==0 ) ? 0.0 : atan2(fRho,fZ); }

  // pseudorapidity - same implementation as in Cartesian3D
  Scalar Eta() const 
  { Scalar rho = Rho();
  /* static */ const Scalar big_z_scaled = 
      std::pow(std::numeric_limits<Scalar>::epsilon(),static_cast<Scalar>(-.6)); 
    if (rho > 0) {
      Scalar z_scaled(fZ/rho);
      if (std::fabs(z_scaled) < big_z_scaled) {
        return std::log(z_scaled+std::sqrt(z_scaled*z_scaled+1)); 
      } else {
        return  fZ>0 ? std::log(2.0*z_scaled) : -std::log(-2.0*z_scaled);
      }
    } else if (fZ==0) {
      return 0;
    } else if (fZ>0) {
      return fZ + etaMax<Scalar>();
    }  else {
      return fZ - etaMax<Scalar>();
    }
  }

 
  // setters (only for data members) 

  /**
     set all the data members ( rho, eta, phi) 
   */ 
  void setValues(T rho, T z, T phi) { 
    fRho = rho;  
    fZ   =   z;  
    fPhi = phi;   
    Restrict();
  }
  
  /** 
      set the rho coordinate value keeping z and phi constant
   */ 
  void SetRho(T rho) { 
        fRho = rho;      
  }

  /** 
      set the z coordinate value keeping rho and phi constant
   */ 
  void SetZ(T z) { 
        fZ = z;      
  }

  /** 
      set the phi coordinate value keeping rho and z constant
   */ 
  void SetPhi(T phi) { 
        fPhi = phi;      
	Restrict();
  }

  /**
     scale by a scalar quantity a -- 
     for cylindrical coords only rho and z change
  */ 
  void Scale (T a) {   
    if (a < 0) {
      Negate();
      a = -a;
    }
    fRho *= a; 
    fZ *= a;
  }

  /**
     negate the vector
  */ 
  void Negate ( ) { 
    fPhi = ( fPhi > 0 ? fPhi - pi() : fPhi + pi() );
    fZ = -fZ;
  }

  // assignment operators
  /**
    generic assignment operator from any coordinate system implementing Rho(), Z() and Phi()
  */ 
  template <class CoordSystem > 
  Cylindrical3D & operator= ( const CoordSystem & c ) { 
    fRho = c.Rho();  
    fZ   = c.Z(); 
    fPhi = c.Phi(); 
    return *this;
  }

  /**
    Exact component-by-component equality 
    */  
  bool operator==(const Cylindrical3D & rhs) const {
    return fRho == rhs.fRho && fZ == rhs.fZ && fPhi == rhs.fPhi;
  }
  bool operator!= (const Cylindrical3D & rhs) const 
  						{return !(operator==(rhs));}
  

  // ============= Compatibility section ==================
  
  // The following make this coordinate system look enough like a CLHEP
  // vector that an assignment member template can work with either
  T x() const { return X();}
  T y() const { return Y();}
  T z() const { return Z(); } 
  
  // ============= Specializations for improved speed ==================

  // (none)

#if defined(__MAKECINT__) || defined(G__DICTIONARY) 

  // ====== Set member functions for coordinates in other systems =======

  void SetX(Scalar x) {  
    GenVector_exception e("Cylindrical3D::SetX() is not supposed to be called");
    Throw(e);
    Cartesian3D<Scalar> v(*this); v.SetX(x); 
    *this = Cylindrical3D<Scalar>(v);
  }
  void SetY(Scalar y) {  
    GenVector_exception e("Cylindrical3D::SetY() is not supposed to be called");
    Throw(e);
    Cartesian3D<Scalar> v(*this); v.SetY(y); 
    *this = Cylindrical3D<Scalar>(v);
  }
  void SetEta(Scalar eta) {  
    GenVector_exception e("Cylindrical3D::SetEta() is not supposed to be called");
    Throw(e);
    CylindricalEta3D<Scalar> v(*this); v.SetEta(eta); 
    *this = Cylindrical3D<Scalar>(v);
  }
  void SetR(Scalar r) {  
    GenVector_exception e("Cylindrical3D::SetR() is not supposed to be called");
    Throw(e);
    Polar3D<Scalar> v(*this); v.SetR(r); 
    *this = Cylindrical3D<Scalar>(v);
  }
  void SetTheta(Scalar theta) {  
    GenVector_exception e("Cylindrical3D::SetTheta() is not supposed to be called");
    Throw(e);
    Polar3D<Scalar> v(*this); v.SetTheta(theta); 
    *this = Cylindrical3D<Scalar>(v);
  }

#endif


private:

  T fRho;
  T fZ;
  T fPhi;

};

  } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_GenVector_Cylindrical3D  */
