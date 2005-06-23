// @(#)root/mathcore:$Name:  $:$Id: PtEtaPhiMSystem.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: Mark Fischler & Lorenzo Moneta   06/2005 

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class PtEtaPhiMSystem
// 
// Created by: moneta  at Tue May 31 21:10:29 2005
// 
// Last update: Tue May 31 21:10:29 2005
// 
#ifndef ROOT_MATH_PTETAPHIMSYSTEM
#define ROOT_MATH_PTETAPHIMSYSTEM 1

#include <cmath>

#include "GenVector/Cartesian4D.h"

namespace ROOT { 

  namespace Math { 


    /** 
	Class describing a 4D cylindrical coordinate system
	using Pt , Eta, Phi and mass   
	The metric used is (-,-,-,+) for (X,y,z,t) 
	Note that the data are not stored contigously in memory 
	threfore there is NO method returning a pointer to the data
    */ 

    template <class T> 
    class PtEtaPhiMSystem { 
      
    public : 

      typedef T Scalar;
      
      PtEtaPhiMSystem() : fPt(0), fEta(0), fPhi(0), fMass(0) {}
      
      PtEtaPhiMSystem(const T & Pt, const T & Eta, const T & Phi, const T & M) : 
	fPt(Pt),
	fEta(Eta),
	fPhi(Phi),
	fMass(M)
      {}
      
  
      /**
	 Generic constructor from any 4D coordinate system implementing Pt(), Eta(), Phi() and M()  
      */ 
      template <class AnyCoordSystem > 
      PtEtaPhiMSystem(const AnyCoordSystem & c) : 
	fPt(c.Pt()),
	fEta(c.Eta()),
	fPhi(c.Phi()),
	fMass(c.M())
      {}

    
      ~PtEtaPhiMSystem() {}

 
      T Px() const { return fPt*cos(fPhi);}
      T Py() const { return fPt*sin(fPhi);}
      T Pz() const { return fPt*std::sinh(fEta);}
      T E() const { return std::sqrt(fMass*fMass + p()*p());}


      T p() const { return fPt*std::cosh(fEta); } 
      
 
      T M2() const { return fMass*fMass; }
      T M() const  { return fMass; } 


      T Perp2() const { return fPt*fPt;}
      
      T Pt() const { return fPt;} 
      
      T Mt2() const { return fMass*fMass + fPt*fPt; } 

      
      T Mt() const { 
	T mm = Mt2();
	return mm < 0.0 ? -std::sqrt(-mm) : std::sqrt(mm);
      } 
      
      // what is the sign for et 
      
      T et2() const { return et()*et(); }
      /// or return fPt*fPt * ( 1. + fMass*fMass/p()/p() ); 
      
      T et() const { 
	return E() / std::cosh(fEta); // not sure if this is fastest impl.
      }
      
      T Phi() const  { return fPhi;}
      
      T Theta() const {
	return  2* std::atan( exp( - fEta ) ); 
	// return acos( CosTheta() );
      }
      
      // pseudorapidity
      T Eta() const { return fEta; } 
      
      void SetValues(const T & Pt, const T & Eta, const T & Phi, const T & M) { 
	fPt = Pt;  
	fEta = Eta; 
	fPhi = Phi; 
	fMass = M; 
      }
    
      void setPt(const T & Pt) { fPt = Pt; }

      void setEta(const T & Eta) { fEta = Eta; }

      void setPhi(const T & Phi) { fPhi = Phi; }

      void setM(const T & M) { fMass = M; }

      /**
	 Scale coordinate values by a scalar quantity a
      */
      void Scale( const T & a) { 
	fPt *= a; 
	fMass *= a; 
      }
      
 
      /**
	 Generic asignment from any 4D coordinate system implementing Pt(), Eta(), Phi() and M()  
      */ 
      template <class AnyCoordSystem > 
      PtEtaPhiMSystem & operator = (const AnyCoordSystem & c) { 
	fPt = c.Pt(); 
	fEta = c.Eta();
	fPhi = c.Phi(); 
	fMass = c.M(); 
	return *this;
      }


    private:
      
      T fPt;
      T fEta;
      T fPhi;
      T fMass; 
    };


  } // end namespace Math

} // end namespace ROOT


#endif
