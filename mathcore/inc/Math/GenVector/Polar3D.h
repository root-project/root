// @(#)root/mathcore:$Name:  $:$Id: Polar3D.h,v 1.2 2005/09/19 16:43:07 brun Exp $
// Authors: W. Brown, M. Fischler, L. Moneta    2005  

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT MathLib Team  and                    *
  *                      FNAL LCG ROOT MathLib Team                    *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for class Polar3D
// 
// Created by: Lorenzo Moneta  at Mon May 30 11:40:03 2005
// Major revamp:  M. Fischler  at Wed Jun  8 2005
// 
// Last update: $Id: Polar3D.h,v 1.2 2005/09/19 16:43:07 brun Exp $
// 
#ifndef ROOT_Math_GenVector_Polar3D 
#define ROOT_Math_GenVector_Polar3D  1

#include "Math/GenVector/etaMax.h"

#include <cmath>
#include <limits>

#if defined(__MAKECINT__) || defined(G__DICTIONARY) 
#endif
 
namespace ROOT { 

  namespace Math { 


  /** 
      Class describing a polar coordinate system based on r, theta and phi

      @ingroup GenVector
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
  void SetCoordinates( const Scalar src[] ) 
  			{ fR=src[0]; fTheta=src[1]; fPhi=src[2]; }

  /**
     get internal data into an array of 3 Scalar numbers
   */ 
  void GetCoordinates( Scalar dest[] ) const 
  			{ dest[0] = fR; dest[1] = fTheta; dest[2] = fPhi; }

  /**
     Set internal data based on 3 Scalar numbers
   */ 
  void SetCoordinates(Scalar r, Scalar  theta, Scalar  phi) 
  					{ fR=r; fTheta=theta; fPhi=phi; }

  /**
     get internal data into 3 Scalar numbers
   */ 
  void GetCoordinates(Scalar& r, Scalar& theta, Scalar& phi) const {r=fR; theta=fTheta; phi=fPhi;}  				

   
  Scalar R()     const { return fR;}
  Scalar Phi()   const { return fPhi; }
  Scalar Theta() const { return fTheta; } 
  Scalar Rho()   const { return fR*std::sin(fTheta); }
  Scalar X()     const { return Rho()*std::cos(fPhi);}
  Scalar Y()     const { return Rho()*std::sin(fPhi);}
  Scalar Z()     const { return fR*std::cos(fTheta); } 
  Scalar Mag2()  const { return fR*fR;}
  Scalar Perp2() const { return Rho()*Rho(); }

  // pseudorapidity
  Scalar Eta() const 
  { Scalar tanThetaOver2 = std::tan( fTheta/2.);
    if (tanThetaOver2 == 0) {
      return fR + etaMax<Scalar>();
    } else if (tanThetaOver2 > std::numeric_limits<Scalar>::max()) {
      return -fR - etaMax<Scalar>();
    } else {
      return -std::log(tanThetaOver2);
      // given that we already have theta, this method 
      // should be faster than the one used from cartesian coordinates
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

  private:
    inline static double pi()  { return 3.14159265358979323; } 
  public:   

  /** 
      scale by a scalar quantity - for polar coordinates r changes
  */
  void Scale (T a) { 
    if (a < 0) {
      Negate();
      a = -a;
    }
    // angles do not change when scaling by a positive quantity
    fR *= a;     
  }

  /**
     negate the vector
  */ 
  void Negate ( ) { 
    fPhi = ( fPhi > 0 ? fPhi - pi() : fPhi + pi() );
    fTheta = pi() - fTheta;
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

  /**
    Exact equality
   */  
  bool operator==(const Polar3D & rhs) const {
    return fR == rhs.fR && fTheta == rhs.fTheta && fPhi == rhs.fPhi;
  }
  bool operator!= (const Polar3D & rhs) const {return !(operator==(rhs));}
  

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

  void SetX(Scalar x);

  void SetY(Scalar y); 

  void SetZ(Scalar z);

  void SetRho(Scalar rho);  

  void SetEta(Scalar eta); 

#endif

private:
  T fR;
  T fTheta;
  T fPhi;
};



  } // end namespace Math

} // end namespace ROOT



#if defined(__MAKECINT__) || defined(G__DICTIONARY) 

  // ====== Set member functions for coordinates in other systems =======


#include "Math/GenVector/GenVector_exception.h"
#include "Math/GenVector/Cartesian3D.h"
#include "Math/GenVector/CylindricalEta3D.h"


namespace ROOT { 

  namespace Math { 

template <class T>  
void Polar3D<T>::SetX(Scalar x) {  
    GenVector_exception e("Polar3D::SetX() is not supposed to be called");
    Throw(e);
    Cartesian3D<Scalar> v(*this); v.SetX(x); *this = Polar3D<Scalar>(v);
  }
template <class T>  
void Polar3D<T>::SetY(Scalar y) {  
    GenVector_exception e("Polar3D::SetY() is not supposed to be called");
    Throw(e);
    Cartesian3D<Scalar> v(*this); v.SetY(y); *this = Polar3D<Scalar>(v);
  }
template <class T>  
void Polar3D<T>::SetZ(Scalar z) {  
    GenVector_exception e("Polar3D::SetZ() is not supposed to be called");
    Throw(e);
    Cartesian3D<Scalar> v(*this); v.SetZ(z); *this = Polar3D<Scalar>(v);
  }
template <class T>  
void Polar3D<T>::SetRho(Scalar rho) {  
    GenVector_exception e("Polar3D::SetRho() is not supposed to be called");
    Throw(e);
    CylindricalEta3D<Scalar> v(*this); v.SetRho(rho); 
    *this = Polar3D<Scalar>(v);
  }
template <class T>  
void Polar3D<T>::SetEta(Scalar eta) {  
    GenVector_exception e("Polar3D::SetEta() is not supposed to be called");
    Throw(e);
    CylindricalEta3D<Scalar> v(*this); v.SetEta(eta); 
    *this = Polar3D<Scalar>(v);
  }



  } // end namespace Math

} // end namespace ROOT

#endif  


#endif /* ROOT_Math_GenVector_Polar3D  */
