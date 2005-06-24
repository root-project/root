// @(#)root/mathcore:$Name:  $:$Id: Cartesian4D.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: W. Brown, M. Fischler, L. Moneta, A. Zsenei   06/2005 

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
// Last update: moneta  Jun 24 2005
// 
#ifndef ROOT_Math_Cartesian4D 
#define ROOT_Math_Cartesian4D 1

#include "MathCore/etaMax.h"

#include <cmath>

namespace ROOT { 
  
  namespace Math { 
    
    /** 
    Class describing a 4D cartesian coordinate system (x, y, z, t coordinates). 
    The metric used is (-,-,-,+)
    */ 
    
    template <class T> 
    class Cartesian4D { 
      
    public : 
      
      typedef T Scalar;
      
      
      /**
      Default constructor  with x=y=z=t=0 
       */
      Cartesian4D() : fX(0), fY(0), fZ(0), fT(0) {}
      
      
      /**
        Constructor  from x, y , z , t values
       */
      Cartesian4D(T  x, T  y, T  z, T  t) : fX(x), fY(y), fZ(z), fT(t) {}
      
      
      /**
        construct from any vector of  coordinate system class implementing X(), Y() and Z() and E()
       */
      template <class CoordSystem> 
        explicit Cartesian4D(const CoordSystem & v) : 
        fX( v.X() ), fY( v.Y() ), fZ( v.Z() ), fT( v.E() )  { }
      
      // no reason for a custom destructor  ~Cartesian3D() {} and copy constructor
      
      /**
        Set internal data based on an array of 4 Scalar numbers
       */ 
      void SetCoordinates( const Scalar * src ) { fX=src[0]; fY=src[1]; fZ=src[2]; fT=src[3]; }
      
      /**
        get internal data into an array of 3 Scalar numbers
       */ 
      void GetCoordinates( Scalar * dest ) const 
      { dest[0] = fX; dest[1] = fY; dest[2] = fZ; dest[3] = fT; }
      
      /**
        Set internal data based on 3 Scalar numbers
       */ 
      void SetCoordinates(Scalar  x, Scalar  y, Scalar  z, Scalar t) { fX=x; fY=y; fZ=z; fT=t;}
      
      /**
        get internal data into 3 Scalar numbers
       */ 
      void GetCoordinates(Scalar& x, Scalar& y, Scalar& z, Scalar& t) const {x=fX; y=fY; z=fZ; t=fT;}  				
      
      
      // coordinate accessors 
      
      T X() const { return fX;}
      T Y() const { return fY;}
      T Z() const { return fZ;}
      T E() const { return fT;}
      
      // other coordinate representation
      
      /**
        magnitude of spatial components
       */
      T R() const { return std::sqrt( fX*fX + fY*fY + fZ*fZ ); } 
      
      /**
        vector magnitude square  
       */
      
      T M2() const { return fT*fT - fX*fX - fY*fY - fZ*fZ;}
      
      /**
        vector invariant mass 
       */
      T M() const    { 
        T mm = M2();
        return mm < 0.0 ? -std::sqrt(-mm) : std::sqrt(mm);
      }
      
      /** 
        transverse spatial component squared  
        */
      T Perp2() const { return fX*fX + fY*fY;}
      
      /**
        Transverse spatial component (rho)
       */
      T Rho() const { return std::sqrt( Perp2());}
      T Pt() const { return std::sqrt( Perp2());}

      
      /** 
        transverse mass squared
        */
      T Mt2() const { return fT*fT - fZ*fZ; } 
      
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
      T Et2() const {  // is (E^2 * pt ^2) / p^2 but faster to express p in terms of pt
        T pt2 = Perp2();
        return pt2 == 0 ? 0 : fT*fT * pt2/( pt2 + fZ*fZ );
      }
      
      /**
        transverse energy
       */
      T Et() const { 
        T etet = Et2();
        return fT < 0.0 ? -std::sqrt(etet) : std::sqrt(etet);
      }
      
      /**
        azimuthal angle 
       */
      T Phi() const  { return fX == 0.0 && fY == 0.0 ? 0.0 : std::atan2(fY,fX);}
      
      /**
        polar angle
       */
      T Theta() const {
        return fX == 0.0 && fY == 0.0 && fZ == 0.0 ? 0.0 : std::atan2(Rho(),fZ);
      }
      
      /** 
        pseudorapidity
        */
      T Eta() const { 
        T rho = Rho();
        if (rho > 0) {
          T z_scaled(fZ/rho);
          return std::log(z_scaled+std::sqrt(z_scaled*z_scaled+1)); // faster 
        } else if (fZ==0) {
          return 0;
        } else if (fZ>0) {
          return fZ + etaMax<T>();
        }  else {
          return fZ - etaMax<T>();
        }
      }
      
      
      /**
        set X value 
       */
      void setX( T  x) { 
        fX = x; 
      }
      /**
        set Y value 
       */
      void setY( T  y) { 
        fX = y; 
      }
      /**
        set Z value 
       */
      void setZ( T  z) { 
        fX = z; 
      }
      /**
        set T value 
       */
      void setT( T  t) { 
        fT = t; 
      }
      
      /**
        scale coordinate values by a scalar quantity a
       */
      void Scale( const T & a) { 
        fX *= a; 
        fY *= a; 
        fZ *= a; 
        fT *= a; 
      }
      
      
      /**
        Assignment from a generic coordinate system implementing X(), Y(), Z() and E()
       */
      template <class AnyCoordSystem> 
        Cartesian4D & operator = (const AnyCoordSystem & v) { 
          fX = v.X();  
          fY = v.Y();  
          fZ = v.Z();  
          fT = v.E();
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

      /**
        (contigous) data containing the coordinate values x,y,z,t
      */

      T fX;
      T fY;
      T fZ;
      T fT;
      
    }; 
    
  } // end namespace Math
  
} // end namespace ROOT

#endif
