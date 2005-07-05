// @(#)root/mathcore:$Name:  $:$Id: CylindricalEta3D.h,v 1.1 2005/06/24 18:54:24 brun Exp $
// Authors: W. Brown, M. Fischler, L. Moneta, A. Zsenei   06/2005 

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
  *                                                                    *
  *                                                                    *
  **********************************************************************/


// Header file for class CylindricalEta3D
// 
// Created by: Lorenzo Moneta  at Mon May 30 11:58:46 2005
// 
// Last update: mf Tue Jun 16 2005
// 
#ifndef ROOT_Math_CylindricalEta3D 
#define ROOT_Math_CylindricalEta3D 1

#include "MathCore/etaMax.h"

namespace ROOT { 

  namespace Math { 

  /** 
      Class describing a cylindrical coordinate system based on eta (pseudorapidity) instead of z.  
      The base coordinates are rho (transverse component) , eta and phi
  
      @ingroup GenVector
   */ 

template <class T> 
class CylindricalEta3D { 

public : 

  typedef T Scalar;

  /**
     Default constructor with rho=eta=phi=0
   */
  CylindricalEta3D() : fRho(0), fEta(0), fPhi(0) {  }

  /**
     Construct from rho eta and phi values
   */
  CylindricalEta3D(T rho, T eta, T phi) :  fRho(rho), fEta(eta), fPhi(phi) {  }

  /**
     Construct from any Vector or coordinate system implementing 
     Rho(), Eta() and Phi()
    */ 
  template <class CoordSystem > 
  explicit CylindricalEta3D( const CoordSystem & v ) : 
    fRho(v.Rho() ),  fEta(v.Eta() ),  fPhi(v.Pphi() )  {  } 

  // no reason for a custom destructor  ~Cartesian3D() {}

  /**
     Set internal data based on an array of 3 Scalar numbers
   */ 
  void SetCoordinates( const Scalar * src ) 
  			{ fRho=src[0]; fEta=src[1]; fPhi=src[2]; }

  /**
     get internal data into an array of 3 Scalar numbers
   */ 
  void GetCoordinates( Scalar * dest ) const 
  			{ dest[0] = fRho; dest[1] = fEta; dest[2] = fPhi; }

  /**
     Set internal data based on 3 Scalar numbers
   */ 
  void SetCoordinates(Scalar  rho, Scalar  eta, Scalar  phi) { fRho=rho; fEta=eta; fPhi=phi; }

  /**
     get internal data into 3 Scalar numbers
   */ 
  void GetCoordinates(Scalar& rho, Scalar& eta, Scalar& phi) const {rho=fRho; eta=fEta; phi=fPhi;}  				

   
  // accessors
 
  T Rho()   const { return fRho; }
  T Eta()   const { return fEta; } 
  T Phi()   const { return fPhi; }
  T X()     const { return fRho*std::cos(fPhi); }
  T Y()     const { return fRho*std::sin(fPhi); }
  T Z()     const { return fRho >  0 ? fRho*std::sinh(fEta) : 
                           fEta == 0 ? 0                    :
                           fEta >  0 ? fEta - etaMax<T>()   :
		                       fEta + etaMax<T>()   ; }
  T R()     const { return fRho*std::cosh(fEta); }
  T Mag2()  const { return R()*R();              }
  T Perp2() const { return fRho*fRho;            }
  T Theta() const { return fEta==0 ? 0 : 2* std::atan( std::exp( - fEta ) ); }

  // setters (only for data members) 

  /**
     set all the data members ( rho, eta, phi) 
   */ 
  void setValues(T rho, T eta, T phi) { 
    fRho = rho;  
    fEta = eta;  
    fPhi = phi;   
  }
  
  /** 
      set the rho coordinate value keeping eta and phi constant
   */ 
  void SetRho(T rho) { 
        fRho = rho;      
  }

  /** 
      set the eta coordinate value keeping rho and phi constant
   */ 
  void SetEta(T eta) { 
        fEta = eta;      
  }

  /** 
      set the phi coordinate value keeping rho and eta constant
   */ 
  void SetPhi(T phi) { 
        fPhi = phi;      
  }

  /**
     scale by a scalar quantity - for cylindrical eta coords, only rho changes!
  */ 
  void Scale (const T & a) { 
    // angles do not change when scaling
    fRho *= a;     
  }

  // assignment operators
  /**
    generic assignment operator from any coordinate system 
  */ 
  template <class CoordSystem > 
  CylindricalEta3D & operator= ( const CoordSystem & c ) { 
    fRho = c.Rho();  
    fEta = c.Eta(); 
    fPhi = c.Phi(); 
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
  T fRho;
  T fEta;
  T fPhi;
};

  } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_CylindricalEta3D */
