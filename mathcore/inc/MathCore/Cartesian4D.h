// @(#)root/mathcore:$Name:  $:$Id: Cartesian4D.h,v 1.1 2005/06/24 18:54:24 brun Exp $
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
    
    template <class ValueType> 
    class Cartesian4D { 
      
    public : 
      
      typedef ValueType Scalar;
      
      
      /**
      Default constructor  with x=y=z=t=0 
       */
      Cartesian4D() : fX(0), fY(0), fZ(0), fT(0) {}
      
      
      /**
        Constructor  from x, y , z , t values
       */
      Cartesian4D(Scalar  x, Scalar  y, Scalar  z, Scalar  t) : fX(x), fY(y), fZ(z), fT(t) {}
      
      
      /**
        construct from any vector of  coordinate system class implementing X(), Y() and Z() and T()
       */
      template <class CoordSystem> 
        explicit Cartesian4D(const CoordSystem & v) : 
        fX( v.X() ), fY( v.Y() ), fZ( v.Z() ), fT( v.T() )  { }
      
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
      
      Scalar X() const { return fX;}
      Scalar Y() const { return fY;}
      Scalar Z() const { return fZ;}
      Scalar T() const { return fT;}
      
      // other coordinate representation
      
      /**
        magnitude of spatial components
       */
      Scalar R() const { return std::sqrt( fX*fX + fY*fY + fZ*fZ ); } 
      
      /**
        vector magnitude square  
       */
      
      Scalar M2() const { return fT*fT - fX*fX - fY*fY - fZ*fZ;}
      
      /**
        vector invariant mass 
       */
      Scalar M() const    { 
        Scalar mm = M2();
        return mm < 0.0 ? -std::sqrt(-mm) : std::sqrt(mm);
      }
      
      /** 
        transverse spatial component squared  
        */
      Scalar Perp2() const { return fX*fX + fY*fY;}
      
      /**
        Transverse spatial component (rho)
       */
      Scalar Rho() const { return std::sqrt( Perp2());}
      Scalar Pt() const { return std::sqrt( Perp2());}

      
      /** 
        transverse mass squared
        */
      Scalar Mt2() const { return fT*fT - fZ*fZ; } 
      
      /**
        transverse mass
       */
      Scalar Mt() const { 
        Scalar mm = Mt2();
        return mm < 0.0 ? -std::sqrt(-mm) : std::sqrt(mm);
      } 
      
      /** 
        transverse energy squared
        */
      Scalar Et2() const {  // is (E^2 * pt ^2) / p^2 but faster to express p in terms of pt
        Scalar pt2 = Perp2();
        return pt2 == 0 ? 0 : fT*fT * pt2/( pt2 + fZ*fZ );
      }
      
      /**
        transverse energy
       */
      Scalar Et() const { 
        Scalar etet = Et2();
        return fT < 0.0 ? -std::sqrt(etet) : std::sqrt(etet);
      }
      
      /**
        azimuthal angle 
       */
      Scalar Phi() const  { return fX == 0.0 && fY == 0.0 ? 0.0 : std::atan2(fY,fX);}
      
      /**
        polar angle
       */
      Scalar Theta() const {
        return fX == 0.0 && fY == 0.0 && fZ == 0.0 ? 0.0 : std::atan2(Rho(),fZ);
      }
      
      /** 
        pseudorapidity
        */
      Scalar Eta() const { 
        Scalar rho = Rho();
        if (rho > 0) {
          Scalar z_scaled(fZ/rho);
          return std::log(z_scaled+std::sqrt(z_scaled*z_scaled+1)); // faster 
        } else if (fZ==0) {
          return 0;
        } else if (fZ>0) {
          return fZ + etaMax<ValueType>();
        }  else {
          return fZ - etaMax<ValueType>();
        }
      }
      
      
      /**
        set X value 
       */
      void setX( Scalar  x) { 
        fX = x; 
      }
      /**
        set Y value 
       */
      void setY( Scalar  y) { 
        fX = y; 
      }
      /**
        set Z value 
       */
      void setZ( Scalar  z) { 
        fX = z; 
      }
      /**
        set T value 
       */
      void setT( Scalar  t) { 
        fT = t; 
      }
      
      /**
        scale coordinate values by a scalar quantity a
       */
      void Scale( const Scalar & a) { 
        fX *= a; 
        fY *= a; 
        fZ *= a; 
        fT *= a; 
      }
      
      
      /**
        Assignment from a generic coordinate system implementing X(), Y(), Z() and T()
       */
      template <class AnyCoordSystem> 
        Cartesian4D & operator = (const AnyCoordSystem & v) { 
          fX = v.X();  
          fY = v.Y();  
          fZ = v.Z();  
          fT = v.T();
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

      /**
        (contigous) data containing the coordinate values x,y,z,t
      */

      Scalar fX;
      Scalar fY;
      Scalar fZ;
      Scalar fT;
      
    }; 
    
  } // end namespace Math
  
} // end namespace ROOT

#endif
