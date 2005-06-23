// @(#)root/mathcore:$Name:  $:$Id: Cartesian4D.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: Mark Fischler & Lorenzo Moneta   06/2005 

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for class Cartesian4D
// 
// Created by: moneta  at Tue May 31 17:07:12 2005
// 
// Last update: Tue May 31 17:07:12 2005
// 
#ifndef ROOT_MATH_CARTESIAN4D
#define ROOT_MATH_CARTESIAN4D 1

#include <cmath>

namespace ROOT { 

  namespace Math { 

    /** 
	Class describing a 4D cartesian coordinate system (X, Y, Z, t Coordinates). 
	The metric used is (-,-,-,+)
    */ 

    template <class T> 
    class Cartesian4D { 
      
    public : 
      
      typedef T Scalar;


      /**
	 Default constructor  with X=y=z=t=0 
      */
      Cartesian4D() { 
	fV[0] = 0; 
	fV[1] = 0;
	fV[2] = 0; 
	fV[3] = 0; 
      }

      /**
	 Constructor  from X, Y , Z , t values
      */
      Cartesian4D(const T & X, const T & Y, const T & Z, const T & t) { 
	fV[0] = X; 
	fV[1] = Y; 
	fV[2] = Z; 
	fV[3] = t; 
      }
      
      /**
	 Copy constructor
      */
      Cartesian4D(const Cartesian4D & c) { 
	fV[0] = c.fV[0];
	fV[1] = c.fV[1]; 
	fV[2] = c.fV[2]; 
	fV[3] = c.fV[3];
      }


      /**
	 construct from a generic coordinate system class implementing X(), Y() and Z() and t()
      */
      template <class AnyCoordSystem> 
      explicit Cartesian4D(const AnyCoordSystem & v) { 
	fV[0] = v.X(); 
	fV[1] = v.Y(); 
	fV[2] = v.Z();
	fV[3] = v.t();
      }

    
      /**
	 Destructor (no operations)
      */
      ~Cartesian4D() {}
      
      /**
	 get internal data (non const version)
      */ 
      T * data() { return fV; }
      
      /**
	 get internal data (const version)
      */ 
      const T * data() const { return fV; }
      
 
      // coordinate accessors 

      T Px() const { return fV[0];}
      T Py() const { return fV[1];}
      T Pz() const { return fV[2];}
      T E() const { return fV[3];}

      // other coordinate representation

      /**
	 momentum , magnitude of spatial components
       */
      T p() const { return std::sqrt( fV[0]*fV[0] + fV[1]*fV[1] + fV[2]*fV[2] ); } 

      /**
	 invariant mass squared 
      */
 
      T M2() const { return fV[3]*fV[3] - fV[0]*fV[0] - fV[1]*fV[1] - fV[2]*fV[2];}

      /**
	 invariant mass 
      */
      T M() const    { 
	T mm = M2();
	return mm < 0.0 ? -std::sqrt(-mm) : std::sqrt(mm);
      }

      /** 
	  transverse momentum squared  
	  ( Perp2 for CLHEP interface compatibility )
      */
      T Perp2() const { return fV[0]*fV[0] + fV[1]*fV[1];}

      /**
	 transverse momentum
       */
      T Pt() const { return std::sqrt( Perp2());}

      /** 
	  transverse mass squared
      */
      T Mt2() const { return fV[3]*fV[3] - fV[2]*fV[2]; } 

      /**
	 transverse mass
      */
      T Mt() const { 
	T mm = Mt2();
	return mm < 0.0 ? -std::sqrt(-mm) : std::sqrt(mm);
      } 

      /** 
	  transverse energy squared
      */
      T et2() const {  // is (E^2 * Pt ^2) / p^2 but faster to express p in terms of Pt
	T pt2 = Perp2();
	return pt2 == 0 ? 0 : fV[3]*fV[3] * pt2/( pt2 + fV[2]*fV[2] );
      }

      /**
	 transverse energy
      */
      T et() const { 
	T etet = et2();
	return fV[3] < 0.0 ? -std::sqrt(etet) : std::sqrt(etet);
      }

      /**
	 azimuthal Angle 
       */
      T Phi() const  { return fV[0] == 0.0 && fV[1] == 0.0 ? 0.0 : std::atan2(fV[1],fV[0]);}

      /**
	 polar Angle
       */
      T Theta() const {
	return fV[0] == 0.0 && fV[1] == 0.0 && fV[2] == 0.0 ? 0.0 : std::atan2(Pt(),fV[2]);
      }

      /** 
	  pseudorapidity
      */
      T Eta() const { 
	T xx ( Pz() / Pt()); 
	return log(xx+sqrt(xx*xx+1));   // faster 
	// this is faster (as CLHEP) on mac but slower than clhep on intel (why ??)
	//     T M = mag();
	//     if ( M==  0   ) return  0.0;   
	//     if ( M==  fV[2] ) return  1.0E72;// for double 
	//     if ( M== -fV[2] ) return -1.0E72;
	//     return 0.5*log( (M + fV[2])/(M - fV[2] ) );
      }

      /** 
	  Set all values 
      */
      void SetValues(const T & Px, const T & Py, const T & Pz, const T & E) { 
	fV[0] = Px;  
	fV[1] = Py; 
	fV[2] = Pz; 
	fV[3] = E; 
      }

      /**
	 Set X value 
      */
      void setPx( const T & X) { 
	fV[0] = X; 
      }
      /**
	 Set Y value 
      */
      void setPy( const T & Y) { 
	fV[0] = Y; 
      }
      /**
	 Set Z value 
      */
      void setPz( const T & Z) { 
	fV[0] = Z; 
      }

      /**
	 Scale coordinate values by a scalar quantity a
      */
      void Scale( const T & a) { 
	fV[0] *= a; 
	fV[1] *= a; 
	fV[2] *= a; 
	fV[3] *= a; 
      }


      /**
	 Assignment from a generic coordinate system implementing X(), Y(), Z() and t()
      */
      template <class AnyCoordSystem> 
      Cartesian4D & operator = (const AnyCoordSystem & v) { 
	fV[0] = v.X();  
	fV[1] = v.Y();  
	fV[2] = v.Z();  
	fV[3] = v.t();
	return *this;
      }


    private:
      /**
	 array containing the coordinate values X,y,z,t
      */
      T fV[4];

    }; 

  } // end namespace Math

} // end namespace ROOT

#endif
