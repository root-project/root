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
  namespace IntegrationOneDim {

    //for 1-dim integration
     enum Type {ADAPTIVE, ADAPTIVESINGULAR, NONADAPTIVE};

  }

    //for multi-dim integration
  namespace IntegrationMultiDim {

    //for 1-dim integration
     enum Type {ADAPTIVE, VEGAS, MISER, PLAIN};

  }  


} // namespace Math
} // namespace ROOT

#endif /* ROOT_Math_AllIntegrationTypes */
