// @(#)root/mathcore:$Name:  $:$Id: Math.h,v 1.1 2006/11/17 18:18:47 moneta Exp $
// Author: L. Moneta Tue Nov 14 15:44:38 2006

/**********************************************************************
 *                                                                    *
 * Copyright (c) 2006  LCG ROOT Math Team, CERN/PH-SFT                *
 *                                                                    *
 *                                                                    *
 **********************************************************************/

// mathematical constants like Pi

#ifndef ROOT_Math_Math
#define ROOT_Math_Math

#ifdef _WIN32
#define _USE_MATH_DEFINES 
#endif
#include <cmath>
#ifndef M_PI
#define M_PI        3.14159265358979323846   /* pi */
#endif

namespace ROOT { 

   namespace Math { 

/** 
    Mathematical constants 
*/ 
inline double Pi() { return M_PI; } 
      
   } // end namespace Math

} // end namespace ROOT


#endif /* ROOT_Math_Math */
