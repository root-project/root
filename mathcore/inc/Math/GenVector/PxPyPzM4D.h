// @(#)root/mathcore:$Name:  $:$Id: PxPyPzM4D.h,v 1.3 2007/02/05 09:40:19 moneta Exp $
// Authors: W. Brown, M. Fischler, L. Moneta    2005  

/**********************************************************************
*                                                                    *
* Copyright (c) 2005 , LCG ROOT MathLib Team                         *
*                                                                    *
*                                                                    *
**********************************************************************/

// Header file for class PxPyPzM4D
// 
// Created by: fischler at Wed Jul 20   2005
//   (starting from PxPyPzM4D by moneta)
// 
// Last update: $Id: PxPyPzM4D.h,v 1.3 2007/02/05 09:40:19 moneta Exp $
// 
#ifndef ROOT_Math_GenVector_PxPyPzM4D 
#define ROOT_Math_GenVector_PxPyPzM4D  1

#include "Math/GenVector/etaMax.h"
#include "Math/GenVector/GenVector_exception.h"


#if defined(__MAKECINT__) || defined(G__DICTIONARY) 
#include "Math/GenVector/PtEtaPhiE4D.h"
#include "Math/GenVector/PxPyPzE4D.h"
#endif


#include <cmath>

namespace ROOT { 
  
  namespace Math { 
    
/** 
   Class describing a 4D coordinate system 
   or momentum-energy vectors stored as (Px, Py, Pz, M).
   This system is useful to describe ultra-relativistic particles 
   (like electrons at LHC) to avoid numerical errors evaluating the mass 
   when E >>> m
   The metric used is (-,-,-,+)
   Spacelike particles (M2 < 0) are described with negative mass values, 
   but in this case m2 must alwasy be less than P2 to preserve a positive value of E2

   @ingroup GenVector
*/ 

template <class ScalarType = double> 
class PxPyPzM4D { 

public : 

  typedef ScalarType Scalar;

  // --------- Constructors ---------------

  /**
  Default constructor  with x=y=z=m=0 
   */
  PxPyPzM4D() : fX(0), fY(0), fZ(0), fM(0) { }


  /**
    Constructor  from x, y , z , m values
   */
  PxPyPzM4D(Scalar x, Scalar y, Scalar z, Scalar m) : 
      				    fX(x), fY(y), fZ(z), fM(m) { 
     
    if (fM < 0) RestrictNegMass();
  }

  /**
    construct from any 4D  coordinate system class 
    implementing X(), Y(), X() and M()
   */
  template <class CoordSystem> 
  explicit PxPyPzM4D(const CoordSystem & v) : 
    fX( v.X() ), fY( v.Y() ), fZ( v.Z() ), fM( v.M() )  
   { }

  // no reason for a custom destructor  ~Cartesian3D() {} and copy constructor

  /**
    Set internal data based on an array of 4 Scalar numbers
   */ 
  void SetCoordinates( const Scalar src[] ) { 
     fX=src[0]; fY=src[1]; fZ=src[2]; fM=src[3]; 
     if (fM < 0) RestrictNegMass();
  }

  /**
    get internal data into an array of 4 Scalar numbers
   */ 
  void GetCoordinates( Scalar dest[] ) const 
  		{ dest[0] = fX; dest[1] = fY; dest[2] = fZ; dest[3] = fM; }

  /**
    Set internal data based on 4 Scalar numbers
   */ 
  void SetCoordinates(Scalar  x, Scalar  y, Scalar  z, Scalar m) { 
     fX=x; fY=y; fZ=z; fM=m;
     if (fM < 0) RestrictNegMass();
  }

  /**
    get internal data into 4 Scalar numbers
   */ 
  void GetCoordinates(Scalar& x, Scalar& y, Scalar& z, Scalar& m) const 
  		{ x=fX; y=fY; z=fZ; m=fM;}  				

  // --------- Coordinates and Coordinate-like Scalar properties -------------

  // cartesian (Minkowski)coordinate accessors 

  Scalar Px() const { return fX;}
  Scalar Py() const { return fY;}
  Scalar Pz() const { return fZ;}
  Scalar M() const  { return fM; }

  Scalar X() const { return fX;}
  Scalar Y() const { return fY;}
  Scalar Z() const { return fZ;}

  // other coordinate representation
  /**
     Energy 
   */ 				  
   Scalar E()  const { return std::sqrt(E2() ); }

  Scalar T() const { return E();}

  /**
    squared magnitude of spatial components
   */
  Scalar P2() const { return fX*fX + fY*fY + fZ*fZ; } 

  /**
    magnitude of spatial components (magnitude of 3-momentum)
   */
  Scalar P() const { return std::sqrt(P2()); } 
  Scalar R() const { return P(); } 

  /**
    vector magnitude squared (or mass squared)
    In case of negative mass (spacelike particles return negative values)
   */
  Scalar M2() const   { 
     return ( fM  >= 0 ) ?  fM*fM :  -fM*fM; 
  }
  Scalar Mag2() const { return M2(); } 

  Scalar Mag() const    { return M(); }

  /**
     energy squared
   */
  Scalar E2() const { 
     Scalar e2 =  P2() + M2(); 
     // protect against numerical errors when M2() is negative
     return e2 > 0 ? e2 : 0; 
  }

  /** 
    transverse spatial component squared  
    */
  Scalar Pt2()   const { return fX*fX + fY*fY;}
  Scalar Perp2() const { return Pt2();}

  /**
    Transverse spatial component (P_perp or rho)
   */
  Scalar Pt()   const { return std::sqrt(Perp2());}
  Scalar Perp() const { return Pt();}
  Scalar Rho()  const { return Pt();}

  /** 
    transverse mass squared
    */
  Scalar Mt2() const { return E2() - fZ*fZ; } 

  /**
    transverse mass
   */
  Scalar Mt() const { 
    Scalar mm = Mt2();
    if (mm >= 0) {
      return std::sqrt(mm);
    } else {
      GenVector_exception e ("PxPyPzM4D::Mt() - Tachyonic:\n"
      		"    Pz^2 > E^2 so the transverse mass would be imaginary");
      Throw(e);  
      return -std::sqrt(-mm);
    }
  } 

  /** 
    transverse energy squared
    */
  Scalar Et2() const {  // is (E^2 * pt ^2) / p^2 
  			// but it is faster to form p^2 from pt^2
    Scalar pt2 = Pt2();
    return pt2 == 0 ? 0 : E2() * pt2/( pt2 + fZ*fZ );
  }

  /**
    transverse energy
   */
  Scalar Et() const { 
    Scalar etet = Et2();
    return std::sqrt(etet);
  }

  /**
    azimuthal angle 
   */
  Scalar Phi() const  { 
    return (fX == 0.0 && fY == 0.0) ? 0.0 : std::atan2(fY,fX);
  }

  /**
    polar angle
   */
  Scalar Theta() const {
    return (fX == 0.0 && fY == 0.0 && fZ == 0.0) ? 0.0 : std::atan2(Pt(),fZ);
  }

  /** 
    pseudorapidity
    */
  Scalar Eta() const { 
    Scalar rho = Pt();
    if (rho > 0) {
      Scalar z_scaled(fZ/rho);
      return std::log(z_scaled+std::sqrt(z_scaled*z_scaled+1)); // faster 
    } else if (fZ==0) {
      return 0;
    } else if (fZ>0) {
      return fZ + etaMax<Scalar>( );
    }  else {
      return fZ - etaMax<Scalar>( );
    }
  }

  // --------- Set Coordinates of this system  ---------------


  /**
    set X value 
   */
  void SetPx( Scalar  x) { 
    fX = x; 
  }
  /**
    set Y value 
   */
  void SetPy( Scalar  y) { 
    fY = y; 
  }
  /**
    set Z value 
   */
  void SetPz( Scalar  z) { 
    fZ = z; 
  }
  /**
    set T value 
   */
  void SetM( Scalar  m) { 
    fM = m; 
    if (fM < 0) RestrictNegMass();
  }


  // ------ Manipulations -------------
  
  /**
     negate the 4-vector
   */
  void Negate( ) { fX = -fX; fY = -fY;  fZ = -fZ; fM = -fM;}

  /**
    scale coordinate values by a scalar quantity a
   */
  void Scale( const Scalar & a) { 
    fX *= a; 
    fY *= a; 
    fZ *= a; 
    fM *= a; 
  }
 
  /**
    Assignment from a generic coordinate system implementing 
    x(), y(), z() and t()
   */
  template <class AnyCoordSystem> 
    PxPyPzM4D & operator = (const AnyCoordSystem & v) { 
      fX = v.x();  
      fY = v.y();  
      fZ = v.z();  
      fM = std::sqrt ( v.t()*v.t() - P2() );
      return *this;
    }

  /**
    Exact equality
   */  
  bool operator == (const PxPyPzM4D & rhs) const {
    return fX == rhs.fX && fY == rhs.fY && fZ == rhs.fZ && fM == rhs.fM;
  }
  bool operator != (const PxPyPzM4D & rhs) const {return !(operator==(rhs));}


  // ============= Compatibility section ==================

  // The following make this coordinate system look enough like a CLHEP
  // vector that an assignment member template can work with either
  Scalar x() const { return X(); }
  Scalar y() const { return Y(); }
  Scalar z() const { return Z(); } 
  Scalar t() const { return E(); } 



#if defined(__MAKECINT__) || defined(G__DICTIONARY) 

  // ====== Set member functions for coordinates in other systems =======

  void SetPt(Scalar pt) {  
    GenVector_exception e("PxPyPzM4D::SetPt() is not supposed to be called");
    Throw(e);
    PtEtaPhiE4D<Scalar> v(*this); v.SetPt(pt); *this = PxPyPzM4D<Scalar>(v);
  }
  void SetEta(Scalar eta) {  
    GenVector_exception e("PxPyPzM4D::SetEta() is not supposed to be called");
    Throw(e);
    PtEtaPhiE4D<Scalar> v(*this); v.SetEta(eta); *this = PxPyPzM4D<Scalar>(v);
  }
  void SetPhi(Scalar phi) {  
    GenVector_exception e("PxPyPzM4D::SetPhi() is not supposed to be called");
    Throw(e);
    PtEtaPhiE4D<Scalar> v(*this); v.SetPhi(phi); *this = PxPyPzM4D<Scalar>(v);
  }
  void SetE(Scalar t) {  
    GenVector_exception e("PxPyPzM4D::SetM() is not supposed to be called");
    Throw(e);
    PxPyPzE4D<Scalar> v(*this); v.SetE(t); 
    *this = PxPyPzM4D<Scalar>(v);
  }

#endif

private:

   // restrict the value of negative mass to avoid unphysical negative E2 values 
   // M2 must be less than P2 for the tachionic particles - otherwise use positive values
   inline void RestrictNegMass() {
    if ( fM >=0 ) return;
    if ( P2() - fM*fM  < 0 ) { 
       GenVector_exception e("PxPyPzM4D::unphysical value of mass, set to closest physical value");
       Throw(e);
       fM = - P();
    }
    return;
  } 


  /**
    (contigous) data containing the coordinate values x,y,z,t
  */

  ScalarType fX;
  ScalarType fY;
  ScalarType fZ;
  ScalarType fM;

}; 
    
} // end namespace Math  
} // end namespace ROOT

#endif // ROOT_Math_GenVector_PxPyPzM4D 
