// @(#)root/mathcore:$Name:  $:$Id: Point3Dfwd.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: Mark Fischler & Lorenzo Moneta   06/2005 

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file Point3Dfwd
// 
// Created by: Lorenzo Moneta  at Mon May 30 18:12:14 2005
// 
// Last update: Mon May 30 18:12:14 2005
// 
#ifndef ROOT_MATH_POINT3DFWD
#define ROOT_MATH_POINT3DFWD 1

// forward declareations of position vectors (Points) and type defs definitions

namespace ROOT { 

  namespace Math { 

    template<class CoordSystem> class PositionVector3D; 

    template<typename T> class Cartesian3D;  
    template<typename T> class CylindricalEta3D;  
    template<typename T> class Polar3D;  

 
    /**
       3D Point based on the cartesian Coordinates X,y,z in double precision
    */
    typedef PositionVector3D< Cartesian3D<double> > XYZPoint; 

    /**
       3D Point based on the cartesian corrdinates X,y,z in single precision
    */
    typedef PositionVector3D< Cartesian3D<double> > XYZPointF; 
    typedef XYZPoint XYZPointD; 
    
    /**
       3D Point based on the Eta based cylindrical Coordinates Rho, Eta, Phi in double precision. 
    */
    typedef PositionVector3D< CylindricalEta3D<double> > RhoEtaPhiPoint; 
    /**
       3D Point based on the Eta based cylindrical Coordinates Rho, Eta, Phi in single precision. 
    */
    typedef PositionVector3D< CylindricalEta3D<float> > RhoEtaPhiPointF; 
    typedef RhoEtaPhiPoint RhoEtaPhiPointD; 

    /**
       3D Point based on the polar Coordinates Rho, Theta, Phi in double precision. 
    */
    typedef PositionVector3D< Polar3D<double> > Polar3DPoint; 
    /**
     3D Point based on the polar Coordinates Rho, Theta, Phi in single precision. 
    */
    typedef PositionVector3D< Polar3D<float> > Polar3DPointF; 
    typedef Polar3DPoint Polar3DPointD; 

  } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_MATH_POINT3DFWD */
