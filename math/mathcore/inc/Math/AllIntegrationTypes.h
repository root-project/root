// @(#)root/mathmore:$Id$
// Author: Magdalena Slawinska  10/2007

 /**********************************************************************
  *                                                                    *
  * Copyright (c) 2007 ROOT Foundation,  CERN/PH-SFT                   *
  *                                                                    *
  *                                                                    *
  **********************************************************************/


// Integration types for
// one and multidimensional integration
// eith a common interface

#ifndef ROOT_Math_AllIntegrationTypes
#define ROOT_Math_AllIntegrationTypes



namespace ROOT {
namespace Math {


    // type of integration


  
    //for 1-dim integration
  namespace IntegrationOneDim {


    /**
	 enumeration specifying the integration types.
	 <ul>
         <li>kGAUSS: simple Gauss integration method with fixed rule
	 <li>kNONADAPTIVE : to be used for smooth functions
	 <li>kADAPTIVE : to be used for general functions without singularities.
	 <li>kADAPTIVESINGULAR: default adaptive integration type which can be used in the case of the presence of singularities.
	 </ul>
	 @ingroup Integration
    */
     enum Type { kGAUSS, kADAPTIVE, kADAPTIVESINGULAR, kNONADAPTIVE};

  }

    //for multi-dim integration
  namespace IntegrationMultiDim {


    /**
	 enumeration specifying the integration types.
	 <ul>
         <li>ADAPTIVE : adaptive multi-dimensional integration
	 <li>PLAIN    MC integration
	 <li>MISER    MC integration 
	 <li>VEGAS    MC integration
	 </ul>
	 @ingroup MCIntegration
    */

     enum Type {kADAPTIVE, kVEGAS, kMISER, kPLAIN};

  }  


} // namespace Math
} // namespace ROOT

#endif /* ROOT_Math_AllIntegrationTypes */
