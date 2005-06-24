// @(#)root/mathcore:$Name:  $:$Id: VectorUtil.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: W. Brown, M. Fischler, L. Moneta, A. Zsenei   06/2005 

// @(#)root/mathcore:$Name:  $:$Id: VectorUtil.h,v 1.6 2005/06/24 11:22:34 moneta Exp $
// Authors: Mark Fischler & Lorenzo Moneta   06/2005 

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for Vector Utility functions
// 
// Created by: moneta  at Tue May 31 21:10:29 2005
// 
// Last update: Tue May 31 21:10:29 2005
// 
#ifndef ROOT_Math_VectorUtil 
#define ROOT_Math_VectorUtil 1


#ifdef _WIN32
#define _USE_MATH_DEFINES 
#endif
#include <cmath>
#ifndef M_PI
#define M_PI        3.14159265358979323846   /* pi */
#endif


namespace ROOT { 

  namespace Math { 


    // utility functions for vector classes 



    /** 
	Global Helper functions for generic Vector classes. Any Vector classes implementing some defined member functions, 
	like  Phi() or Eta() or mag() can use these functions.   
	The functions returning a scalar value, returns always double precision number even if the vector are 
	based on another precision type
    */ 
    
    
    namespace VectorUtil { 
    
      // methods for 3D vectors 

      /**
	 Find aximutal Angle difference between two generic vectors ( v2.Phi() - v1.Phi() ) 
	 The only requirements on the Vector classes is that they implement the Phi() method
	 \param v1  Vector of any type implementing the Phi() operator
	 \param v2  Vector of any type implementing the Phi() operator
	 \return  Phi difference
	 \f[ \Delta \phi = \phi_2 - \phi_1 \f]
      */
      template <class Vector1, class Vector2> 
      double DeltaPhi( const Vector1 & v1, const Vector2 & v2) { 
	double dphi = v2.Phi() - v1.Phi(); 
	if ( dphi > M_PI ) {
	  dphi -= 2.0*M_PI;
	} else if ( dphi <= -M_PI ) {
	  dphi += 2.0*M_PI;
	}
	return dphi;
      }
      
      
      
    /**
       Find difference in pseudorapidity (Eta) and Phi betwen two generic vectors
       The only requirements on the Vector classes is that they implement the Phi() and Eta() method
       \param v1  Vector 1  
       \param v2  Vector 2
       \return   Angle between the two vectors
       \f[ \Delta R = \sqrt{  ( \Delta \phi )^2 + ( \Delta \eta )^2 } \f]
    */ 
      template <class Vector1, class Vector2> 
      double DeltaR( const Vector1 & v1, const Vector2 & v2) { 
	double dphi = DeltaPhi(v1,v2); 
	double deta = v2.Eta() - v1.Eta(); 
	return std::sqrt( dphi*dphi + deta*deta ); 
      }


      
      /**
	 Find CosTheta Angle between two generic 3D vectors 
	 pre-requisite: vectors implement the X(), Y() and Z() 
	 \param v1  Vector v1  
	 \param v2  Vector v2
	 \return   cosine of Angle between the two vectors
	 \f[ \cos \theta = \frac { \vec{v1} \cDot \vec{v2} }{ | \vec{v1} | | \vec{v2} | } \f]
      */ 
      // this cannot be made all generic since Mag2() for 2, 3 or 4 D is different 
      // need to have a specialization for polar Coordinates ??
      template <class Vector1, class Vector2>
      double CosTheta( const Vector1 &  v1, const Vector2  & v2) { 
        double arg;
	double v1_r2 = v1.X()*v1.X() + v1.Y()*v1.Y() + v1.Z()*v1.Z();
	double v2_r2 = v2.X()*v2.X() + v2.Y()*v2.Y() + v2.Z()*v2.Z();
	double ptot2 = v1_r2*v2_r2;
	if(ptot2 <= 0) {
	  arg = 0.0;
	}else{
	  double pdot = v1.X()*v2.X() + v1.Y()*v2.Y() + v1.Z()*v2.Z();
	  arg = pdot/std::sqrt(ptot2);
	  if(arg >  1.0) arg =  1.0;
	  if(arg < -1.0) arg = -1.0;
	}
	return arg;
      }


      /**
	 Find Angle between two vectors. 
	 Use the CosTheta() function 
	 \param v1  Vector v1  
	 \param v2  Vector v2
	 \return   Angle between the two vectors
	 \f[ \theta = \cos ^{-1} \frac { \vec{v1} \cDot \vec{v2} }{ | \vec{v1} | | \vec{v2} | } \f]
      */ 
      template <class Vector1, class Vector2> 
      double Angle( const  Vector1 & v1, const Vector2 & v2) { 
	return std::acos( CosTheta(v1, v2) ); 
      }

      // Lorentz Vector functions


    /**
       return the invariant mass of two LorentzVector 
       The only requirement on the LorentzVector is that they need to implement the  
       X() , Y(), Z() and E() methods. 
       \param v1 LorenzVector 1
       \param v2 LorenzVector 2
       \return invariant mass M 
       \f[ M_{12} = \sqrt{ (\vec{v1} + \vec{v2} ) \cDot (\vec{v1} + \vec{v2} ) } \f]
    */ 
      template <class Vector1, class Vector2> 
      double InvariantMass( const Vector1 & v1, const Vector2 & v2) { 
	double ee = (v1.E() + v2.E() );
	double xx = (v1.X() + v2.X() );
	double yy = (v1.Y() + v2.Y() );
	double zz = (v1.Z() + v2.Z() );
	double mm2 = ee*ee - xx*xx - yy*yy - zz*zz; 
	return mm2 < 0.0 ? -std::sqrt(-mm2) : std::sqrt(mm2);
	Cartesian4D<double> q(xx,yy,zz,ee); 
	return q.M();
	//return ( v1 + v2).mag(); 
      }

      // rotation and transformations
      
      
#ifndef __CINT__       
      /** 
	  rotation along X axis for a generic vector by an Angle alpha 
	  returning a new vector. 
	  The only pre requisite on the Vector is that it has to implement the X() , Y() and Z() 
	  operators and can be constructed from X,y,z
      */ 
      template <class Vector> 
      Vector RotateX(const Vector & v, double alpha) { 
	double sina = sin(alpha);
	double cosa = cos(alpha);
	double y2 = v.Y() * cosa - v.Z()*sina;
	double z2 = v.Z() * cosa + v.Y() * sina; 
	return Vector(v.X(), y2, z2);
      }

      /** 
	  rotation along Y axis for a generic vector by an Angle alpha 
	  returning a new vector. 
	  The only pre requisite on the Vector is that it has to implement the X() , Y() and Z() 
	  operators and can be constructed from X,y,z
      */ 
      template <class Vector> 
      Vector RotateY(const Vector & v, double alpha) { 
	double sina = sin(alpha);
	double cosa = cos(alpha);
	double x2 = v.X() * cosa + v.Z() * sina; 
	double z2 = v.Z() * cosa - v.X() * sina;
	return Vector(x2, v.Y(), z2);
      }

      /** 
	  rotation along Z axis for a generic vector by an Angle alpha 
	  returning a new vector. 
	  The only pre requisite on the Vector is that it has to implement the X() , Y() and Z() 
	  operators and can be constructed from X,y,z
      */ 
      template <class Vector> 
      Vector RotateZ(const Vector & v, double alpha) { 
	double sina = sin(alpha);
	double cosa = cos(alpha);
	double x2 = v.X() * cosa - v.Y() * sina; 
	double y2 = v.Y() * cosa - v.X() * sina;
	return Vector(x2, y2, v.Z() );
      }
      

      /**
	 rotation on a generic vector using a generic rotation class.
	 The only requirement on the vector is that implements the 
	 X(), Y(), Z() methods and be constructed from X,y,z values
	 The requirement on the rotation is that need to implement the 
	 (i,j) operator returning the matrix element with R(0,0) = xx element
      */
      template<class Vector, class Rotation> 
      Vector Rotate(const Vector &v, const Rotation & rot) { 
	register double xX = v.X();
	register double yY = v.Y();
	register double zZ = v.Z();
	double x2 =  rot(0,0)*xX + rot(0,1)*yY + rot(0,2)*zZ;
	double y2 =  rot(1,0)*xX + rot(1,1)*yY + rot(1,2)*zZ;
	double z2 =  rot(2,0)*xX + rot(2,1)*yY + rot(2,2)*zZ;
	return Vector(x2,y2,z2);
      }
#endif      


    }  // end namespace Vector Util

  } // end namespace Math
  
} // end namespace ROOT


#endif /* ROOT_Math_VectorUtil */
