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
         <li>kDEFAULT: default type specifiend in the static options
         <li>kGAUSS: simple Gauss integration method with fixed rule
         <li>kLEGENDRE: Gauss-Legendre integration
     <li>kNONADAPTIVE : to be used for smooth functions
     <li>kADAPTIVE : to be used for general functions without singularities.
     <li>kADAPTIVESINGULAR: default adaptive integration type which can be used in the case of the presence of singularities.
     </ul>
     @ingroup Integration
    */
     enum Type { kDEFAULT = -1, kGAUSS, kLEGENDRE, kADAPTIVE, kADAPTIVESINGULAR, kNONADAPTIVE};

  }

    //for multi-dim integration
  namespace IntegrationMultiDim {


    /**
     enumeration specifying the integration types.
     <ul>
     <li>kDEFAULT  : default type specified in the static option
     <li>kADAPTIVE : adaptive multi-dimensional integration
     <li>kPLAIN    MC integration
     <li>kMISER    MC integration
     <li>kVEGAS    MC integration
     </ul>
     @ingroup MCIntegration
     */

     enum Type {kDEFAULT = -1, kADAPTIVE, kVEGAS, kMISER, kPLAIN};

  }


} // namespace Math
} // namespace ROOT

#endif /* ROOT_Math_AllIntegrationTypes */
