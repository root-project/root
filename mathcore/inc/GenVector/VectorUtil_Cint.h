// @(#)root/mathcore:$Name:  $:$Id: VectorUtil_Cint.hv 1.0 2005/06/23 12:00:00 moneta Exp $
// Authors: Mark Fischler & Lorenzo Moneta   06/2005 

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2005 , LCG ROOT MathLib Team                         *
  *                                                                    *
  *                                                                    *
  **********************************************************************/

// Header file for Vector Utility Function (for CINT) 
// 
// Created by: moneta  at Tue May 31 21:10:29 2005
// 
// Last update: Tue May 31 21:10:29 2005
// 
#ifndef ROOT_MATH_VECTORUTIL_CINT
#define ROOT_MATH_VECTORUTIL_CINT 1

// functions for CINT without using templates 


#include "GenVector/VectorUtil.h"
#include "GenVector/Vector3D.h"
#include "GenVector/Point3D.h"
#include "GenVector/LorentzVector.h"

namespace ROOT { 

  namespace Math { 


    // utility functions for vector classes in global space 

    XYZVector operator * (double a, XYZVector v) { 
      return v *= a;
    }
    


        
    namespace VectorUtil { 


      double DeltaPhi(const XYZVector & v1, const XYZVector & v2) { 
	return DeltaPhi<XYZVector, XYZVector>(v1,v2);
      }

      double DeltaPhi(const RhoEtaPhiVector & v1, const RhoEtaPhiVector & v2) { 
	return DeltaPhi<RhoEtaPhiVector, RhoEtaPhiVector>(v1,v2);
      }

      double DeltaPhi(const Polar3DVector & v1, const Polar3DVector & v2) { 
	return DeltaPhi<Polar3DVector, Polar3DVector>(v1,v2);
      }

      double DeltaPhi(const XYZPoint & v1, const XYZPoint & v2) { 
	return DeltaPhi<XYZPoint, XYZPoint>(v1,v2);
      }

      double DeltaPhi(const Polar3DPoint & v1, const Polar3DPoint & v2) { 
	return DeltaPhi<Polar3DPoint, Polar3DPoint>(v1,v2);
      }

      double DeltaPhi(const RhoEtaPhiPoint & v1, const RhoEtaPhiPoint & v2) { 
	return DeltaPhi<RhoEtaPhiPoint, RhoEtaPhiPoint>(v1,v2);
      }

      double DeltaPhi(const LorentzVector & v1, const LorentzVector & v2) { 
	return DeltaPhi<LorentzVector, LorentzVector>(v1,v2);
      }

      double DeltaPhi(const LorentzVectorPtEtaPhiM & v1, const LorentzVectorPtEtaPhiM & v2) { 
	return DeltaPhi<LorentzVectorPtEtaPhiM, LorentzVectorPtEtaPhiM>(v1,v2);
      }

      // delta R


      double DeltaR(const XYZVector & v1, const XYZVector & v2) { 
	return DeltaR<XYZVector, XYZVector>(v1,v2);
      }

      double DeltaR(const RhoEtaPhiVector & v1, const RhoEtaPhiVector & v2) { 
	return DeltaR<RhoEtaPhiVector, RhoEtaPhiVector>(v1,v2);
      }

      double DeltaR(const Polar3DVector & v1, const Polar3DVector & v2) { 
	return DeltaR<Polar3DVector, Polar3DVector>(v1,v2);
      }

      double DeltaR(const XYZPoint & v1, const XYZPoint & v2) { 
	return DeltaR<XYZPoint, XYZPoint>(v1,v2);
      }

      double DeltaR(const Polar3DPoint & v1, const Polar3DPoint & v2) { 
	return DeltaR<Polar3DPoint, Polar3DPoint>(v1,v2);
      }

      double DeltaR(const RhoEtaPhiPoint & v1, const RhoEtaPhiPoint & v2) { 
	return DeltaR<RhoEtaPhiPoint, RhoEtaPhiPoint>(v1,v2);
      }

      double DeltaR(const LorentzVector & v1, const LorentzVector & v2) { 
	return DeltaR<LorentzVector, LorentzVector>(v1,v2);
      }

      double DeltaR(const LorentzVectorPtEtaPhiM & v1, const LorentzVectorPtEtaPhiM & v2) { 
	return DeltaR<LorentzVectorPtEtaPhiM, LorentzVectorPtEtaPhiM>(v1,v2);
      }

      // cosTheta v1 v2 

      double CosTheta(const XYZVector & v1, const XYZVector & v2) { 
	return CosTheta<XYZVector, XYZVector>(v1,v2);
      }

      double CosTheta(const RhoEtaPhiVector & v1, const RhoEtaPhiVector & v2) { 
	return CosTheta<RhoEtaPhiVector, RhoEtaPhiVector>(v1,v2);
      }

      double CosTheta(const Polar3DVector & v1, const Polar3DVector & v2) { 
	return CosTheta<Polar3DVector, Polar3DVector>(v1,v2);
      }

      double CosTheta(const XYZPoint & v1, const XYZPoint & v2) { 
	return CosTheta<XYZPoint, XYZPoint>(v1,v2);
      }

      double CosTheta(const Polar3DPoint & v1, const Polar3DPoint & v2) { 
	return CosTheta<Polar3DPoint, Polar3DPoint>(v1,v2);
      }

      double CosTheta(const RhoEtaPhiPoint & v1, const RhoEtaPhiPoint & v2) { 
	return CosTheta<RhoEtaPhiPoint, RhoEtaPhiPoint>(v1,v2);
      }

      double CosTheta(const LorentzVector & v1, const LorentzVector & v2) { 
	return CosTheta<LorentzVector, LorentzVector>(v1,v2);
      }

      double CosTheta(const LorentzVectorPtEtaPhiM & v1, const LorentzVectorPtEtaPhiM & v2) { 
	return CosTheta<LorentzVectorPtEtaPhiM, LorentzVectorPtEtaPhiM>(v1,v2);
      }

      // angle v1 v2

      double Angle(const XYZVector & v1, const XYZVector & v2) { 
	return Angle<XYZVector, XYZVector>(v1,v2);
      }

      double Angle(const RhoEtaPhiVector & v1, const RhoEtaPhiVector & v2) { 
	return Angle<RhoEtaPhiVector, RhoEtaPhiVector>(v1,v2);
      }

      double Angle(const Polar3DVector & v1, const Polar3DVector & v2) { 
	return Angle<Polar3DVector, Polar3DVector>(v1,v2);
      }

      double Angle(const XYZPoint & v1, const XYZPoint & v2) { 
	return Angle<XYZPoint, XYZPoint>(v1,v2);
      }

      double Angle(const Polar3DPoint & v1, const Polar3DPoint & v2) { 
	return Angle<Polar3DPoint, Polar3DPoint>(v1,v2);
      }

      double Angle(const RhoEtaPhiPoint & v1, const RhoEtaPhiPoint & v2) { 
	return Angle<RhoEtaPhiPoint, RhoEtaPhiPoint>(v1,v2);
      }

      double Angle(const LorentzVector & v1, const LorentzVector & v2) { 
	return Angle<LorentzVector, LorentzVector>(v1,v2);
      }

      double Angle(const LorentzVectorPtEtaPhiM & v1, const LorentzVectorPtEtaPhiM & v2) { 
	return Angle<LorentzVectorPtEtaPhiM, LorentzVectorPtEtaPhiM>(v1,v2);
      }

      // invariant mass v1 v2 

      double InvariantMass(const LorentzVector & v1, const LorentzVector & v2) { 
	return InvariantMass<LorentzVector, LorentzVector>(v1,v2);
      }

      double InvariantMass(const LorentzVectorPtEtaPhiM & v1, const LorentzVectorPtEtaPhiM & v2) { 
	return InvariantMass<LorentzVectorPtEtaPhiM, LorentzVectorPtEtaPhiM>(v1,v2);
      }

      double InvariantMass(const LorentzVector & v1, const LorentzVectorPtEtaPhiM & v2) { 
	return InvariantMass<LorentzVector, LorentzVectorPtEtaPhiM>(v1,v2);
      }

      double InvariantMass(const LorentzVectorPtEtaPhiM & v1, const LorentzVector & v2) { 
	return InvariantMass<LorentzVectorPtEtaPhiM, LorentzVector>(v1,v2);
      }


    }  // end namespace Vector Util

   
  } // end namespace Math
  
} // end namespace ROOT




#endif
