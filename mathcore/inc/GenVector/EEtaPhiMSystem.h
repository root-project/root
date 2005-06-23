// @(#)root/mathcore:$Name:  $:$Id: EEtaPhiMSystem.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: Mark Fischler & Lorenzo Moneta   06/2005 

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// Header file for class EEtaPhiMSystem
// 
// Created by: moneta  at Tue May 31 21:15:19 2005
// 
// Last update: Tue May 31 21:15:19 2005
// 
#ifndef ROOT_MATH_EETAPHIMSYSTEM
#define ROOT_MATH_EETAPHIMSYSTEM 1

#include <cmath>
#include <assert.h>

#include "GenVector/Cartesian4D.h"

namespace ROOT { 

  namespace Math { 


    /** 
	Class describing a 4D coordinate system based on E (t) , Eta, Phi and Mass . 
	The metric used is (-,-,-,+) refered to  (X, Y, Z, t Coordinates) 
	Note that the data are not stored contigously in memory 
	threfore there is NO method returning a pointer to the data
    */ 

    template <class T> 
    class EEtaPhiMSystem { 
      
    public : 

      typedef T Scalar;

      
      EEtaPhiMSystem() : fE(0), fEta(0), fPhi(0), fMass(0) {}
      
      EEtaPhiMSystem(const T & E, const T & Eta, const T & Phi, const T & mass) : 
	fE(E),
	fEta(Eta),
	fPhi(Phi),
	fMass(mass)
      {}


      /**
	 Generic constructor from any 4D coordinate system implementing t(), Eta(), Phi() and M()  
      */ 
      template <class AnyCoordSystem > 
      EEtaPhiMSystem(const AnyCoordSystem & c) : 
	fE(c.E()),
	fEta(c.Eta()),
	fPhi(c.Phi()),
	fMass(c.M())
      {}

    

      ~EEtaPhiMSystem() {}

 
      T Px() const { return Pt()*cos(fPhi);}
      T Py() const { return Pt()*sin(fPhi);}
      T Pz() const { return p() * CosTheta();}
      T E() const { return fE;}


      T p() const { 
	if (fMass == 0) return fE; 
	//here one has to be careful for E > M for M > 0 (time-like vectors) 
	if (fMass > 0)  { 
	  assert ( fE >= fMass ); 
	  // sign according to E ? 
	  return std::sqrt( fE*fE - fMass*fMass ); 
	}
	else 
	  // for negative masses 
	  return  std::sqrt( fE*fE + fMass*fMass );
      } 

 
      T M2() const { return fMass*fMass;}

      T M() const  { return fMass; }  
      
      
      T Perp2() const { return Pt() * Pt();}

      T Pt() const { return p()*sinTheta();}  // not sure if this fastest (Atlas does this) 

      T Mt2() const { return fE*fE  - p2() * CosTheta() * CosTheta(); } 
      
      T Mt() const { 
	double mm = Mt2();
	return mm < 0.0 ? -std::sqrt(-mm) : std::sqrt(mm);
      } 

      T et2() const { return et()*et(); }
  

      T et() const { 
	return fE* sinTheta(); // not sure if this is fastest impl.
      }
      
      T Phi() const  { return fPhi;}
      
      T Theta() const {
	return  2* atan( exp( - fEta ) ); 
	// return acos( CosTheta() );
      }
      
      // pseudorapidity
      T Eta() const { return fEta; } 
      

      // useful in this context 
      T sinTheta() const { return 1./std::cosh(fEta); }
      
      T CosTheta() const { return std::tanh(fEta); }
      
      T p2() const { 
	if (fMass == 0) return fE*fE; 
	return fE*fE - fMass*fMass; 
      }

      // setters 

      /**
	 Set the coordinate values E, Eta, Phi and M
       */
      void SetValues(const T & E, const T & Eta, const T & Phi, const T & M) { 
	fE = E;  
	fEta = Eta; 
	fPhi = Phi; 
	fMass = M; 
      }
      // Set Data members
      void setE(const T & E) { fE = E; }

      void setEta(const T & Eta) { fEta = Eta; }

      void setPhi(const T & Phi) { fPhi = Phi; }

      void setM(const T & M) { fMass = M; }

      /**
	 Scale coordinate values by a scalar quantity a
      */
      void Scale( const T & a) { 
	fE *= a; 
	fMass *= a; 
      }
      
      /**
	 Generic asignment from any 4D coordinate system implementing t(), Eta(), Phi() and M()  
      */ 
      template <class AnyCoordSystem > 
      EEtaPhiMSystem & operator = (const AnyCoordSystem & c) { 
	fE = c.E(); 
	fEta = c.Eta();
	fPhi = c.Phi(); 
	fMass = c.M(); 
	return *this;
      }


    private:
      T fE;
      T fEta;
      T fPhi;
      T fMass; 
    };

    

  } // end namespace Math

} // end namespace ROOT


#endif
