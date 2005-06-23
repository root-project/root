// @(#)root/mathcore:$Name:  $:$Id: Vector3Dfwd.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: Mark Fischler & Lorenzo Moneta   06/2005 

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file Vector3Dfwd
// 
// Created by: Lorenzo Moneta  at Mon May 30 18:08:35 2005
// 
// Last update: Mon May 30 18:08:35 2005
// 
#ifndef ROOT_MATH_VECTOR3DFWD
#define ROOT_MATH_VECTOR3DFWD 1

// forward declarations of displacement vectors (Vectors) and type defs definitions

namespace ROOT { 

  namespace Math { 


    template<class CoordSystem> class DisplacementVector3D; 

    template<typename T> class Cartesian3D;  
    template<typename T> class CylindricalEta3D;  
    template<typename T> class Polar3D;  


    /**
       3D Vector based on the cartesian Coordinates X,y,z in double precision
    */
    typedef DisplacementVector3D< Cartesian3D<double> > XYZVector; 
    /**
       3D Vector based on the cartesian corrdinates X,y,z in single precision
    */
    typedef DisplacementVector3D< Cartesian3D<float> > XYZVectorF; 
    typedef XYZVector XYZVectorD; 

    /**
       3D Vector based on the Eta based cylindrical Coordinates Rho, Eta, Phi in double precision. 
    */
    typedef DisplacementVector3D< CylindricalEta3D<double> > RhoEtaPhiVector; 
  /**
     3D Vector based on the Eta based cylindrical Coordinates Rho, Eta, Phi in single precision. 
   */
    typedef DisplacementVector3D< CylindricalEta3D<float> > RhoEtaPhiVectorF; 
    typedef RhoEtaPhiVector RhoEtaPhiVectorD; 

    /**
       3D Vector based on the polar Coordinates Rho, Theta, Phi in double precision. 
    */
    typedef DisplacementVector3D< Polar3D<double> > Polar3DVector; 
    /**
       3D Vector based on the polar Coordinates Rho, Theta, Phi in single precision. 
    */
    typedef DisplacementVector3D< Polar3D<float> > Polar3DVectorF; 
    typedef Polar3DVector Polar3DVectorD; 
    

  } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_MATH_VECTOR3DFWD */
