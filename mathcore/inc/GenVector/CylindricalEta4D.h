// @(#)root/mathcore:$Name:  $:$Id: CylindricalEta4D.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: Mark Fischler & Lorenzo Moneta   06/2005 

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
// Last update: Tue May 31 20:53:01 2005
// 
#ifndef ROOT_MATH_CYLINDRICALETA4D
#define ROOT_MATH_CYLINDRICALETA4D 1


#include <cmath>


namespace ROOT { 

  namespace Math { 

    
    /** 
	Class describing a 4D cylindrical coordinate system
	using Pt , Phi, Eta and E .   
	The metric used is (-,-,-,+). 
	Note that the data are not stored contigously in memory 
	threfore there is NO method returning a pointer to the data
    */ 

    template <class T> 
    class CylindricalEta4D { 

    public : 

      typedef T Scalar;
      
      CylindricalEta4D() : fPt(0), fEta(0), fPhi(0), fE(0) {}
      
      CylindricalEta4D(const T & Pt, const T & Eta, const T & Phi, const T & E) : 
	fPt(Pt),
	fEta(Eta),
	fPhi(Phi),
	fE(E)
      {}
      
      /**
	 Generic constructor from any 4D coordinate system implementing Pt(), Eta(), Phi() and E()  
      */ 
      template <class AnyCoordSystem > 
      CylindricalEta4D(const AnyCoordSystem & c) : 
	fPt(c.Pt()),
	fEta(c.Eta()),
	fPhi(c.Phi()),
	fE(c.E())
      {}

    

      ~CylindricalEta4D() {}

 
      T Px() const { return fPt*cos(fPhi);}
      T Py() const { return fPt*sin(fPhi);}
      T Pz() const { return fPt*std::sinh(fEta);}
      T E() const { return fE;}


      T p() const { return fPt*std::cosh(fEta); } 
      
      
      T M2() const { return fE*fE - p()*p(); }

      T M() const  { 
	double mm = M2();
	return mm < 0.0 ? -std::sqrt(-mm) : std::sqrt(mm);
      }  


      T Perp2() const { return fPt*fPt;}

      T Pt() const { return fPt;} 

      T Mt2() const { return fE*fE  - Pz() * Pz(); } 
      
      T Mt() const { 
	double mm = Mt2();
	return mm < 0.0 ? -std::sqrt(-mm) : std::sqrt(mm);
      } 
      
      T et2() const { return et()*et(); }
      
      
      T et() const { 
	return fE / std::cosh(fEta); // not sure if this is fastest impl.
      }
      
      T Phi() const  { return fPhi;}
      
      T Theta() const {
	return  2* std::atan( exp( - fEta ) ); 
	// return acos( CosTheta() );
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
      template <class AnyCoordSystem > 
      CylindricalEta4D & operator = (const AnyCoordSystem & c) { 
	fPt = c.Pt(); 
	fEta = c.Eta();
	fPhi = c.Phi(); 
	fE = c.E(); 
	return *this;
      }



    private:
      
      T fPt;
      T fEta;
      T fPhi;
      T fE; 

    };



  } // end namespace Math

} // end namespace ROOT


#endif 

