// @(#)root/mathcore:$Name:  $:$Id: LorentzVectorfwd.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: Mark Fischler & Lorenzo Moneta   06/2005 

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
#ifndef ROOT_MATH_LORENTZVECTORFWD
#define ROOT_MATH_LORENTZVECTORFWD 1


namespace ROOT { 

  namespace Math { 


    // forward declaretions of Lorentz Vectors and type defs definitions

    template<class CoordSystem> class BasicLorentzVector; 

    template<typename T> class Cartesian4D;  
    template<typename T> class CylindricalEta4D;  
    template<typename T> class PtEtaPhiMSystem;  
    template<typename T> class EEtaPhiMSystem;  


    // for LorentzVector have only double classes (define the vector in the global ref frame) 

    /**
       LorentzVector based on X,y,x,t (or Px,pY,pZ,E) Coordinates in double precision with metric (-,-,-,+) 
    */
    typedef BasicLorentzVector<Cartesian4D<double> > LorentzVector;

    /**
     LorentzVector based on X,y,x,t (or Px,pY,pZ,E) Coordinates in float precision with metric (-,-,-,+) 
    */
    typedef BasicLorentzVector< Cartesian4D <float> > LorentzVectorF;
    

    /**
       LorentzVector based on the cylindrical Coordinates Pt, Eta, Phi and E (Rho, Eta, Phi, t) in double precision
    */
    typedef BasicLorentzVector<CylindricalEta4D<double> > LorentzVectorPtEtaPhiE;

    /**
       LorentzVector based on the cylindrical Coordinates Pt, Eta, Phi and Mass in double precision
    */
    typedef BasicLorentzVector<PtEtaPhiMSystem<double> > LorentzVectorPtEtaPhiM;
    
    /**
       LorentzVector based on the Coordinates E, Eta, Phi and Mass in double precision. These Coordinates are normally used to represents a cluster objects in a calorimeter at a collider experiment. 
    */
    typedef BasicLorentzVector<EEtaPhiMSystem<double> > LorentzVectorEEtaPhiM;
    


  } // end namespace Math

} // end namespace ROOT

#endif

