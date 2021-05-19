// @(#)root/mathcore:$Id$
// Authors: W. Brown, M. Fischler, L. Moneta    2005

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
#ifndef ROOT_Math_Vector4Dfwd
#define ROOT_Math_Vector4Dfwd  1


namespace ROOT {

  namespace Math {


    // forward declarations of Lorentz Vectors and type defs definitions

    template<class CoordSystem> class LorentzVector;

    template<typename T> class PxPyPzE4D;
    template<typename T> class PtEtaPhiE4D;
    template<typename T> class PxPyPzM4D;
    template<typename T> class PtEtaPhiM4D;
//     template<typename T> class EEtaPhiMSystem;


    // for LorentzVector have only double classes (define the vector in the global ref frame)

    /**
       LorentzVector based on x,y,x,t (or px,py,pz,E) coordinates in double precision with metric (-,-,-,+)

       To use it use `#include <Vector4D.h>`

       See the documentation on the LorentzVector page.
    */
    typedef LorentzVector<PxPyPzE4D<double> > XYZTVector;
    // for consistency
    typedef LorentzVector<PxPyPzE4D<double> > PxPyPzEVector;


    /**
       LorentzVector based on x,y,x,t (or px,py,pz,E) coordinates in float precision with metric (-,-,-,+)

       To use it use `#include <Vector4D.h>`

       See the documentation on the LorentzVector page.
    */
    typedef LorentzVector< PxPyPzE4D <float> > XYZTVectorF;


    /**
       LorentzVector based on the x, y, z,  and Mass in double precision

       To use it use `#include <Vector4D.h>`

       See the documentation on the LorentzVector page.
    */
    typedef LorentzVector<PxPyPzM4D<double> > PxPyPzMVector;

    /**
       LorentzVector based on the cylindrical coordinates Pt, eta, phi and E (rho, eta, phi, t) in double precision

       To use it use `#include <Vector4D.h>`

       See the documentation on the LorentzVector page.
    */
    typedef LorentzVector<PtEtaPhiE4D<double> > PtEtaPhiEVector;

    /**
       LorentzVector based on the cylindrical coordinates pt, eta, phi and Mass in double precision

       To use it use `#include <Vector4D.h>`

       See the documentation on the LorentzVector page.
    */
    typedef LorentzVector<PtEtaPhiM4D<double> > PtEtaPhiMVector;

//     /**
//        LorentzVector based on the coordinates E, Eta, Phi and Mass in double precision. These coordinates are normally used to represents a cluster objects in a calorimeter at a collider experiment.
//     */
//     typedef BasicLorentzVector<EEtaPhiMSystem<double> > LorentzVectorEEtaPhiM;



  } // end namespace Math

} // end namespace ROOT

#endif

