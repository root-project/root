// @(#)root/mathcore:$Name:  $:$Id: CylindricalEta4D.h,v 1.1 2005/06/24 18:54:24 brun Exp $
// Authors: W. Brown, M. Fischler, L. Moneta, A. Zsenei   06/2005 

/**********************************************************************
*                                                                    *
* Copyright (c) 2005 , LCG ROOT MathLib Team                         *
*                                                                    *
*                                                                    *
**********************************************************************/

// Header file for class CylindricalEta4D
// 
// Created by: moneta  at Tue May 31 20:53:01 2005
// 
// Last update: moneta Jun 24 2005
// 
#ifndef ROOT_Math_CylindricalEta4D 
#define ROOT_Math_CylindricalEta4D 1

#include "MathCore/etaMax.h"

#include <cmath>


namespace ROOT { 
  
  namespace Math { 
    
    
    /** 
	Class describing a 4D cylindrical coordinate system
	using Pt , Phi, Eta and E (or rho, phi, eta , T) 
	The metric used is (-,-,-,+). 
	Note that the data are not stored contigously in memory 
	threfore there is NO method returning a pointer to the data
    */ 
    
    template <class ValueType> 
    class CylindricalEta4D { 
      
    public : 
      
      typedef ValueType Scalar;
      
      CylindricalEta4D() : fPt(0), fEta(0), fPhi(0), fE(0) {}
      
      CylindricalEta4D(Scalar  pt, Scalar  eta, Scalar  phi, Scalar  e) :   
        fPt(pt),
        fEta(eta),
        fPhi(phi),
        fE(e)
      {}
      
      /**
        Generic constructor from any 4D coordinate system implementing Pt(), Eta(), Phi() and T()  
       */ 
      template <class CoordSystem > 
        explicit CylindricalEta4D(const CoordSystem & c) : 
        fPt(c.Pt()),
        fEta(c.Eta()),
        fPhi(c.Phi()),
        fE(c.T())
      {}
      
      
      // -- non need for customized copy constructor and destructor 
      
      /**
        Set internal data based on an array of 4 Scalar numbers
       */ 
      void SetCoordinates( const Scalar * src ) { fPt=src[0]; fEta=src[1]; fPhi=src[2]; fE=src[3]; }
      
      /**
        get internal data into an array of 3 Scalar numbers
       */ 
      void GetCoordinates( Scalar * dest ) const 
      { dest[0] = fPt; dest[1] = fEta; dest[2] = fPhi; dest[3] = fE; }
      
      /**
        Set internal data based on 3 Scalar numbers
       */ 
      void SetCoordinates(Scalar  pt, Scalar  eta, Scalar  phi, Scalar  e) { fPt=pt; fEta = eta; fPhi = phi; fE = e; }
      
      /**
        get internal data into 3 Scalar numbers
       */ 
      void GetCoordinates(Scalar & pt, Scalar & eta, Scalar & phi, Scalar & e) const { pt=fPt; eta=fEta; phi = fPhi; e = fE; }
  	
      
      
      Scalar X() const { return fPt*cos(fPhi);}

      Scalar Y() const { return fPt*sin(fPhi);}

      Scalar Z()     const {
        return fPt >  0 ? fPt*std::sinh(fEta) : 
        fEta == 0 ? 0                    :
        fEta >  0 ? fEta - etaMax<ValueType>()   :
        fEta + etaMax<ValueType>(); 
      }
      
      Scalar T() const { return fE; }
      
      
      Scalar R() const { return fPt*std::cosh(fEta); } 
      
      Scalar M2() const { return fE*fE - R()*R(); }
      
      Scalar M() const  { 
        double mm = M2();
        return mm < 0.0 ? -std::sqrt(-mm) : std::sqrt(mm);
      }  
      
      
      Scalar Perp2() const { return fPt*fPt;}
      
      Scalar Rho() const { return fPt;} 
      
      Scalar Mt2() const { return fE*fE  - Z()*Z(); } 
      
      Scalar Mt() const { 
        double mm = Mt2();
        return mm < 0.0 ? -std::sqrt(-mm) : std::sqrt(mm);
      } 
      
      Scalar Et2() const { return Et()*Et(); }
      
      Scalar Et() const { 
        return fE / std::cosh(fEta); // not sure if this is fastest impl.
      }
      
      Scalar Phi() const  { return fPhi;}
      
      Scalar Theta() const {
        return  2* std::atan( exp( - fEta ) );
      }
      
      // pseudorapidity
      Scalar Eta() const { return fEta; } 
      
      // setters 
      
      /**
        Set the coordinate value Pt, Eta, Phi and E
       */
      void SetValues(const Scalar & Pt, const Scalar & Eta, const Scalar & Phi, const Scalar & E) { 
        fPt = Pt;  
        fEta = Eta; 
        fPhi = Phi; 
        fE = E; 
      }

      // Set Data members
      void SetPt(const Scalar & Pt) { fPt = Pt; }
      
      void SetEta(const Scalar & Eta) { fEta = Eta; }
      
      void SetPhi(const Scalar & Phi) { fPhi = Phi; }
      
      void SetE(const Scalar & E) { fE = E; }
      
      /**
        Scale coordinate values by a scalar quantity a
       */
      void Scale( const Scalar & a) { 
        fPt *= a; 
        fE *= a; 
      }
      
      
      // assignment 
      
      /**
        generic assignment from any 4D coordinate system implementing Pt(), Eta(), Phi() and t()  
       */ 
      template <class CoordSystem > 
        CylindricalEta4D & operator = (const CoordSystem & c) { 
          fPt = c.Pt(); 
          fEta = c.Eta();
          fPhi = c.Phi(); 
          fE = c.T(); 
          return *this;
        }
      
        // ============= Compatibility secition ==================
  
      // The following make this coordinate system look enough like a CLHEP
      // vector that an assignment member template can work with either
      Scalar x() const { return X();}
      Scalar y() const { return Y();}
      Scalar z() const { return Z(); } 
      Scalar t() const { return T(); } 

      
      
    private:
        
      Scalar fPt;
      Scalar fEta;
      Scalar fPhi;
      Scalar fE; 
      
    };
    
    
    
  } // end namespace Math
  
} // end namespace ROOT


#endif 

