// @(#)root/mathcore:$Name:  $:$Id: LorentzVectorfwd.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: W. Brown, M. Fischler, L. Moneta, A. Zsenei   06/2005 

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for class LorentzVectorfwd
// 
// Created by: moneta  at Tue May 31 21:06:43 2005
// 
// Last update: Tue May 31 21:06:43 2005
// 
#ifndef ROOT_Math_LorentzVectorfwd 
#define ROOT_Math_LorentzVectorfwd 1


namespace ROOT { 

  namespace Math { 


    // forward declaretions of Lorentz Vectors and type defs definitions

    template<class CoordSystem> class LorentzVector; 

    template<typename T> class Cartesian4D;  
    template<typename T> class CylindricalEta4D;  
//     template<typename T> class PtEtaPhiMSystem;  
//     template<typename T> class EEtaPhiMSystem;  


    // for LorentzVector have only double classes (define the vector in the global ref frame) 

    /**
       LorentzVector based on x,y,x,t (or px,py,pz,E) coordinates in double precision with metric (-,-,-,+) 
    */
    typedef LorentzVector<Cartesian4D<double> > XYZTVector;

    /**
     LorentzVector based on x,y,x,t (or px,py,pz,E) coordinates in float precision with metric (-,-,-,+) 
    */
    typedef LorentzVector< Cartesian4D <float> > XYZTVectorF;
    

    /**
       LorentzVector based on the cylindrical coordinates Pt, eta, phi and E (rho, eta, phi, t) in double precision
    */
    typedef LorentzVector<CylindricalEta4D<double> > PtEtaPhiEVector;

//     /**
//        LorentzVector based on the cylindrical coordinates pt, eta, phi and Mass in double precision
//     */
//     typedef BasicLorentzVector<PtEtaPhiMSystem<double> > LorentzVectorPtEtaPhiM;
    
//     /**
//        LorentzVector based on the coordinates E, Eta, Phi and Mass in double precision. These coordinates are normally used to represents a cluster objects in a calorimeter at a collider experiment. 
//     */
//     typedef BasicLorentzVector<EEtaPhiMSystem<double> > LorentzVectorEEtaPhiM;
    


  } // end namespace Math

} // end namespace ROOT

#endif

