// @(#)root/mathcore:$Name:  $:$Id: Vector3Dfwd.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: W. Brown, M. Fischler, L. Moneta    2005  

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
#ifndef ROOT_Math_Vector3Dfwd 
#define ROOT_Math_Vector3Dfwd  1

// forward declarations of displacement vectors (Vectors) and type defs definitions

namespace ROOT { 

  namespace Math { 


    template<class CoordSystem> class DisplacementVector3D; 

    template<typename T> class Cartesian3D;  
    template<typename T> class CylindricalEta3D;  
    template<typename T> class Polar3D;  


    /**
       3D Vector based on the cartesian coordinates x,y,z in double precision
    */
    typedef DisplacementVector3D< Cartesian3D<double> > XYZVector; 
    /**
       3D Vector based on the cartesian corrdinates x,y,z in single precision
    */
    typedef DisplacementVector3D< Cartesian3D<float> > XYZVectorF; 
    typedef XYZVector XYZVectorD; 

    /**
       3D Vector based on the eta based cylindrical coordinates rho, eta, phi in double precision. 
    */
    typedef DisplacementVector3D< CylindricalEta3D<double> > RhoEtaPhiVector; 
  /**
     3D Vector based on the eta based cylindrical coordinates rho, eta, phi in single precision. 
   */
    typedef DisplacementVector3D< CylindricalEta3D<float> > RhoEtaPhiVectorF; 
    typedef RhoEtaPhiVector RhoEtaPhiVectorD; 

    /**
       3D Vector based on the polar coordinates rho, theta, phi in double precision. 
    */
    typedef DisplacementVector3D< Polar3D<double> > Polar3DVector; 
    /**
       3D Vector based on the polar coordinates rho, theta, phi in single precision. 
    */
    typedef DisplacementVector3D< Polar3D<float> > Polar3DVectorF; 
    typedef Polar3DVector Polar3DVectorD; 
    

  } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_Vector3Dfwd  */
