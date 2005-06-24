// @(#)root/mathcore:$Name:  $:$Id: CylindricalEta4D.hv 1.0 2005/06/23 12:00:00 moneta Exp $
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
    
    template <class T> 
    class CylindricalEta4D { 
      
    public : 
      
      typedef T Scalar;
      
      CylindricalEta4D() : fPt(0), fEta(0), fPhi(0), fE(0) {}
      
      CylindricalEta4D(T  pt, T  eta, T  phi, T  e) :   
        fPt(pt),
        fEta(eta),
        fPhi(phi),
        fE(e)
      {}
      
      /**
        Generic constructor from any 4D coordinate system implementing Pt(), Eta(), Phi() and E()  
       */ 
      template <class CoordSystem > 
        explicit CylindricalEta4D(const CoordSystem & c) : 
        fPt(c.Pt()),
        fEta(c.Eta()),
        fPhi(c.Phi()),
        fE(c.E())
      {}
      
      
      // -- non need for customized copy constructor and destructor 
      
      /**
        Set internal data based on an array of 4 Scalar numbers
       */ 
      void SetCoordinates( const T * src ) { fPt=src[0]; fEta=src[1]; fPhi=src[2]; fE=src[3]; }
      
      /**
        get internal data into an array of 3 Scalar numbers
       */ 
      void GetCoordinates( T * dest ) const 
      { dest[0] = fPt; dest[1] = fEta; dest[2] = fPhi; dest[3] = fE; }
      
      /**
        Set internal data based on 3 Scalar numbers
       */ 
      void SetCoordinates(T  pt, T  eta, T  phi, T  e) { fPt=pt; fEta = eta; fPhi = phi; fE = e; }
      
      /**
        get internal data into 3 Scalar numbers
       */ 
      void GetCoordinates(T& pt, T & eta, T & phi, T& e) const { pt=fPt; eta=fEta; phi = fPhi; e = fE; }
  	
      
      
      T X() const { return fPt*cos(fPhi);}

      T Y() const { return fPt*sin(fPhi);}

      T Z()     const {
        return fPt >  0 ? fPt*std::sinh(fEta) : 
        fEta == 0 ? 0                    :
        fEta >  0 ? fEta - etaMax<T>()   :
        fEta + etaMax<T>(); 
      }
      
      T E() const { return fE; }
      
      
      T R() const { return fPt*std::cosh(fEta); } 
      
      T M2() const { return fE*fE - R()*R(); }
      
      T M() const  { 
        double mm = M2();
        return mm < 0.0 ? -std::sqrt(-mm) : std::sqrt(mm);
      }  
      
      
      T Perp2() const { return fPt*fPt;}
      
      T Rho() const { return fPt;} 
      
      T Mt2() const { return fE*fE  - Z()*Z(); } 
      
      T Mt() const { 
        double mm = Mt2();
        return mm < 0.0 ? -std::sqrt(-mm) : std::sqrt(mm);
      } 
      
      T Et2() const { return Et()*Et(); }
      
      T Et() const { 
        return fE / std::cosh(fEta); // not sure if this is fastest impl.
      }
      
      T Phi() const  { return fPhi;}
      
      T Theta() const {
        return  2* std::atan( exp( - fEta ) );
      }
      
      // pseudorapidity
      T Eta() const { return fEta; } 
      
      // setters 
      
      /**
        Set the coordinate value Pt, Eta, Phi and E
       */
      void SetValues(const T & Pt, const T & Eta, const T & Phi, const T & E) { 
        fPt = Pt;  
        fEta = Eta; 
        fPhi = Phi; 
        fE = E; 
      }

      // Set Data members
      void SetPt(const T & Pt) { fPt = Pt; }
      
      void SetEta(const T & Eta) { fEta = Eta; }
      
      void SetPhi(const T & Phi) { fPhi = Phi; }
      
      void SetE(const T & E) { fE = E; }
      
      /**
        Scale coordinate values by a scalar quantity a
       */
      void Scale( const T & a) { 
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
          fE = c.E(); 
          return *this;
        }
      
        // ============= Compatibility secition ==================
  
      // The following make this coordinate system look enough like a CLHEP
      // vector that an assignment member template can work with either
      T x() const { return X();}
      T y() const { return Y();}
      T z() const { return Z(); } 
      T t() const { return E(); } 

      
      
    private:
        
      T fPt;
      T fEta;
      T fPhi;
      T fE; 
      
    };
    
    
    
  } // end namespace Math
  
} // end namespace ROOT


#endif 

