// @(#)root/mathcore:$Name:  $:$Id: PtEtaPhiE4D.h,v 1.5 2006/11/25 09:30:08 moneta Exp $
// Authors: W. Brown, M. Fischler, L. Moneta    2005  

/**********************************************************************
*                                                                    *
* Copyright (c) 2005 , LCG ROOT FNAL MathLib Team                    *
*                                                                    *
*                                                                    *
**********************************************************************/

// Header file for class PtEtaPhiE4D
// 
// Created by: fischler at Wed Jul 20 2005
//   based on CylindricalEta4D by moneta
// 
// Last update: $Id: PtEtaPhiE4D.h,v 1.5 2006/11/25 09:30:08 moneta Exp $
// 
#ifndef ROOT_Math_GenVector_PtEtaPhiE4D 
#define ROOT_Math_GenVector_PtEtaPhiE4D  1

#include "Math/GenVector/etaMax.h"
#include "Math/GenVector/GenVector_exception.h"

#include <cmath>

//#define TRACE_CE
#ifdef TRACE_CE
#include <iostream>
#endif



namespace ROOT {   
namespace Math { 
       
/** 
   Class describing a 4D cylindrical coordinate system
   using Pt , Phi, Eta and E (or rho, phi, eta , T) 
   The metric used is (-,-,-,+). 

   @ingroup GenVector
*/ 

template <class ScalarType> 
class PtEtaPhiE4D { 

public : 

  typedef ScalarType Scalar;

  // --------- Constructors ---------------

  /**
  Default constructor gives zero 4-vector  
   */
  PtEtaPhiE4D() : fPt(0), fEta(0), fPhi(0), fE(0) { }

  /**
    Constructor  from pt, eta, phi, e values
   */
  PtEtaPhiE4D(Scalar pt, Scalar eta, Scalar phi, Scalar e) : 
      			fPt(pt), fEta(eta), fPhi(phi), fE(e) { Restrict(); }

  /**
    Generic constructor from any 4D coordinate system implementing 
    Pt(), Eta(), Phi() and E()  
   */ 
  template <class CoordSystem > 
    explicit PtEtaPhiE4D(const CoordSystem & c) : 
    fPt(c.Pt()), fEta(c.Eta()), fPhi(c.Phi()), fE(c.E())  { }

  // no need for customized copy constructor and destructor 

  /**
    Set internal data based on an array of 4 Scalar numbers
   */ 
  void SetCoordinates( const Scalar src[] ) 
    { fPt=src[0]; fEta=src[1]; fPhi=src[2]; fE=src[3]; Restrict(); }

  /**
    get internal data into an array of 4 Scalar numbers
   */ 
  void GetCoordinates( Scalar dest[] ) const 
    { dest[0] = fPt; dest[1] = fEta; dest[2] = fPhi; dest[3] = fE; }

  /**
    Set internal data based on 4 Scalar numbers
   */ 
  void SetCoordinates(Scalar pt, Scalar eta, Scalar phi, Scalar e) 
    { fPt=pt; fEta = eta; fPhi = phi; fE = e; Restrict(); }

  /**
    get internal data into 4 Scalar numbers
   */ 
  void 
  GetCoordinates(Scalar& pt, Scalar & eta, Scalar & phi, Scalar& e) const 
    { pt=fPt; eta=fEta; phi = fPhi; e = fE; }

  // --------- Coordinates and Coordinate-like Scalar properties -------------

  // 4-D Cylindrical eta coordinate accessors  

  Scalar Pt()  const { return fPt;  }
  Scalar Eta() const { return fEta; }
  Scalar Phi() const { return fPhi; }
  Scalar E()   const { return fE;   }

  Scalar Perp()const { return Pt(); }
  Scalar Rho() const { return Pt(); }
  Scalar T()   const { return E();  }
  
  // other coordinate representation

  Scalar Px() const { return fPt*cos(fPhi);}
  Scalar X () const { return Px();         }
  Scalar Py() const { return fPt*sin(fPhi);}
  Scalar Y () const { return Py();         }
  Scalar Pz() const {
    return fPt >   0 ? fPt*std::sinh(fEta)     : 
           fEta == 0 ? 0                       :
           fEta >  0 ? fEta - etaMax<Scalar>() :
                       fEta + etaMax<Scalar>() ; 
  }
  Scalar Z () const { return Pz(); }

  /** 
     magnitude of momentum
   */
  Scalar P() const { 
    return  fPt  > 0                 ?  fPt*std::cosh(fEta)       :
  	    fEta >  etaMax<Scalar>() ?  fEta - etaMax<Scalar>()   :
 	    fEta < -etaMax<Scalar>() ? -fEta - etaMax<Scalar>()   :
			                0                         ; 
  }
  Scalar R() const { return P(); }

  /** 
    squared magnitude of spatial components (momentum squared)
   */
  Scalar P2() const { Scalar p = P(); return p*p; }

  /**
    vector magnitude squared (or mass squared)
   */
  Scalar M2() const { Scalar p = P(); return fE*fE - p*p; }
  Scalar Mag2() const { return M2(); } 

  /**
    invariant mass 
   */
  Scalar M() const    { 
    Scalar mm = M2();
    if (mm >= 0) {
      return std::sqrt(mm);
    } else {
      GenVector_exception e ("PtEtaPhiE4D::M() - Tachyonic:\n"
  "    Pt and Eta give P such that P^2 > E^2, so the mass would be imaginary");
      Throw(e);  
      return -std::sqrt(-mm);
    }
  }
  Scalar Mag() const    { return M(); }

  /** 
    transverse spatial component squared  
    */
  Scalar Pt2()   const { return fPt*fPt;}
  Scalar Perp2() const { return Pt2();  }

  /** 
    transverse mass squared
    */
  Scalar Mt2() const {  Scalar pz = Pz(); return fE*fE  - pz*pz; } 

  /**
    transverse mass
   */
  Scalar Mt() const { 
    Scalar mm = Mt2();
    if (mm >= 0) {
      return std::sqrt(mm);
    } else {
      GenVector_exception e ("PtEtaPhiE4D::Mt() - Tachyonic:\n"
 "    Pt and Eta give Pz such that Pz^2 > E^2, so the mass would be imaginary");
     Throw(e);  
      return -std::sqrt(-mm);
    }
  } 

  /**
    transverse energy
   */
  /**
    transverse energy
   */
  Scalar Et() const { 
    return fE / std::cosh(fEta); // faster using eta
  }

  /** 
    transverse energy squared
    */
  Scalar Et2() const { Scalar et = Et(); return et*et; }


private:
  inline static double pi() { return 3.14159265358979323; } 
  inline void Restrict() {
    if ( fPhi <= -pi() || fPhi > pi() ) 
      fPhi = fPhi - std::floor( fPhi/(2*pi()) +.5 ) * 2*pi();
  return;
  } 
public:

  /**
    polar angle
   */
  Scalar Theta() const {
    if (fPt  >  0) return 2* std::atan( exp( - fEta ) );
    if (fEta >= 0) return 0;
    return pi();
  }

  // --------- Set Coordinates of this system  ---------------

  /**
    set Pt value 
   */
  void SetPt( Scalar  pt) { 
    fPt = pt; 
  }
  /**
    set eta value 
   */
  void SetEta( Scalar  eta) { 
    fEta = eta; 
  }
  /**
    set phi value 
   */
  void SetPhi( Scalar  phi) { 
    fPhi = phi; 
    Restrict();
  }
  /**
    set E value 
   */
  void SetE( Scalar  e) { 
    fE = e; 
  }

  // ------ Manipulations -------------

  /**
     negate the 4-vector
   */
  void Negate( ) { fPhi = - fPhi; fEta = - fEta; fE = - fE; }

  /**
    Scale coordinate values by a scalar quantity a
   */
  void Scale( Scalar a) { 
    if (a < 0) {
	Negate(); a = -a;
    }
    fPt *= a; 
    fE  *= a; 
  }

  /**
    Assignment from a generic coordinate system implementing 
    Pt(), Eta(), Phi() and E()  
   */ 
  template <class CoordSystem > 
    PtEtaPhiE4D & operator = (const CoordSystem & c) { 
      fPt  = c.Pt(); 
      fEta = c.Eta();
      fPhi = c.Phi(); 
      fE   = c.E(); 
      return *this;
    }

  /**
    Exact equality
   */  
  bool operator == (const PtEtaPhiE4D & rhs) const {
    return fPt == rhs.fPt && fEta == rhs.fEta 
    			  && fPhi == rhs.fPhi && fE == rhs.fE;
  }
  bool operator != (const PtEtaPhiE4D & rhs) const {return !(operator==(rhs));}

    // ============= Compatibility secition ==================

  // The following make this coordinate system look enough like a CLHEP
  // vector that an assignment member template can work with either
  Scalar x() const { return X(); }
  Scalar y() const { return Y(); }
  Scalar z() const { return Z(); } 
  Scalar t() const { return E(); } 



#if defined(__MAKECINT__) || defined(G__DICTIONARY) 

  // ====== Set member functions for coordinates in other systems =======

  void SetPx(Scalar px);  

  void SetPy(Scalar py);

  void SetPz(Scalar pz);

  void SetM(Scalar m);


#endif

private:

  ScalarType fPt;
  ScalarType fEta;
  ScalarType fPhi;
  ScalarType fE; 

};    
    
    
} // end namespace Math  
} // end namespace ROOT



#if defined(__MAKECINT__) || defined(G__DICTIONARY) 
// move implementations here to avoid circle dependencies

#include "Math/GenVector/PxPyPzE4D.h"
#include "Math/GenVector/PtEtaPhiM4D.h"

namespace ROOT { 

namespace Math { 

     
  // ====== Set member functions for coordinates in other systems =======

template <class ScalarType>  
void PtEtaPhiE4D<ScalarType>::SetPx(Scalar px) {  
    GenVector_exception e("PtEtaPhiE4D::SetPx() is not supposed to be called");
    Throw(e);
    PxPyPzE4D<Scalar> v(*this); v.SetPx(px); *this = PtEtaPhiE4D<Scalar>(v);
}
template <class ScalarType>  
void PtEtaPhiE4D<ScalarType>::SetPy(Scalar py) {  
    GenVector_exception e("PtEtaPhiE4D::SetPx() is not supposed to be called");
    Throw(e);
    PxPyPzE4D<Scalar> v(*this); v.SetPy(py); *this = PtEtaPhiE4D<Scalar>(v);
  }
template <class ScalarType>  
void PtEtaPhiE4D<ScalarType>::SetPz(Scalar pz) {  
    GenVector_exception e("PtEtaPhiE4D::SetPx() is not supposed to be called");
    Throw(e);
    PxPyPzE4D<Scalar> v(*this); v.SetPz(pz); *this = PtEtaPhiE4D<Scalar>(v);
  }
template <class ScalarType>  
void PtEtaPhiE4D<ScalarType>::SetM(Scalar m) {  
    GenVector_exception e("PtEtaPhiE4D::SetM() is not supposed to be called");
    Throw(e);
    PtEtaPhiM4D<Scalar> v(*this); v.SetM(m); 
    *this = PtEtaPhiE4D<Scalar>(v);
}

} // end namespace Math

} // end namespace ROOT

#endif  // endif __MAKE__CINT || G__DICTIONARY



#endif // ROOT_Math_GenVector_PtEtaPhiE4D 

