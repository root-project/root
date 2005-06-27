// @(#)root/mathcore:$Name:  $:$Id: VectorUtil_Cint.h,v 1.1 2005/06/24 18:54:24 brun Exp $
// Authors: W. Brown, M. Fischler, L. Moneta, A. Zsenei   06/2005 

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
#ifndef ROOT_Math_VectorUtil_Cint 
#define ROOT_Math_VectorUtil_Cint 1

// functions for CINT without using templates 


#include "MathCore/VectorUtil.h"
#include "MathCore/Vector3D.h"
#include "MathCore/Point3D.h"
#include "MathCore/Vector4D.h"

namespace ROOT { 

  namespace Math { 


    // utility functions for vector classes in global space 

    XYZVector operator * (double a, XYZVector v) { 
      return v *= a;
    }

    XYZTVector operator * (double a, XYZTVector v) { 
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

      double DeltaPhi(const XYZTVector & v1, const XYZTVector & v2) { 
	return DeltaPhi<XYZTVector, XYZTVector>(v1,v2);
      }

      double DeltaPhi(const PtEtaPhiEVector & v1, const PtEtaPhiEVector & v2) { 
	return DeltaPhi<PtEtaPhiEVector, PtEtaPhiEVector>(v1,v2);
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

      double DeltaR(const XYZTVector & v1, const XYZTVector & v2) { 
	return DeltaR<XYZTVector, XYZTVector>(v1,v2);
      }

      double DeltaR(const PtEtaPhiEVector & v1, const PtEtaPhiEVector & v2) { 
	return DeltaR<PtEtaPhiEVector, PtEtaPhiEVector>(v1,v2);
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

      double CosTheta(const XYZTVector & v1, const XYZTVector & v2) { 
	return CosTheta<XYZTVector, XYZTVector>(v1,v2);
      }

      double CosTheta(const PtEtaPhiEVector & v1, const PtEtaPhiEVector & v2) { 
	return CosTheta<PtEtaPhiEVector, PtEtaPhiEVector>(v1,v2);
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

      double Angle(const XYZTVector & v1, const XYZTVector & v2) { 
	return Angle<XYZTVector, XYZTVector>(v1,v2);
      }

      double Angle(const PtEtaPhiEVector & v1, const PtEtaPhiEVector & v2) { 
	return Angle<PtEtaPhiEVector, PtEtaPhiEVector>(v1,v2);
      }

      // invariant mass v1 v2 

      double InvariantMass(const XYZTVector & v1, const XYZTVector & v2) { 
	return InvariantMass<XYZTVector, XYZTVector>(v1,v2);
      }

      double InvariantMass(const PtEtaPhiEVector & v1, const PtEtaPhiEVector & v2) { 
	return InvariantMass<PtEtaPhiEVector, PtEtaPhiEVector>(v1,v2);
      }

      double InvariantMass(const XYZTVector & v1, const PtEtaPhiEVector & v2) { 
	return InvariantMass<XYZTVector, PtEtaPhiEVector>(v1,v2);
      }

      double InvariantMass(const PtEtaPhiEVector & v1, const XYZTVector & v2) { 
	return InvariantMass<PtEtaPhiEVector, XYZTVector>(v1,v2);
      }


    }  // end namespace Vector Util

   
  } // end namespace Math
  
} // end namespace ROOT




#endif
