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

     /// enumeration specifying the integration types.
     /// @ingroup Integration
     enum Type {
        kDEFAULT = -1,     ///< default type specifiend in the static options
        kGAUSS,            ///< simple Gauss integration method with fixed rule
        kLEGENDRE,         ///< Gauss-Legendre integration
        kADAPTIVE,         ///< to be used for general functions without singularities
        kADAPTIVESINGULAR, ///< default adaptive integration type which can be used in the case of the presence of singularities.
        kNONADAPTIVE       ///< to be used for smooth functions
     };
  }

    //for multi-dim integration
  namespace IntegrationMultiDim {

     /// enumeration specifying the integration types.
     /// @ingroup MCIntegration
     enum Type {
        kDEFAULT = -1, ///< default type specified in the static option
        kADAPTIVE,     ///< adaptive multi-dimensional integration
        kVEGAS,        ///< MC integration
        kMISER,        ///< MC integration
        kPLAIN         ///< MC integration
     };
  }


} // namespace Math
} // namespace ROOT

#endif /* ROOT_Math_AllIntegrationTypes */
