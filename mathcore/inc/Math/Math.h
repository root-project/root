// @(#)root/mathcore:$Name:  $:$Id: Math.h,v 1.1 2007/05/22 13:35:16 moneta Exp $
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
#define M_PI       3.14159265358979323846264338328      // Pi 
#endif

#ifndef M_PI_2
#define M_PI_2     1.57079632679489661923132169164      // Pi/2 
#endif

#ifndef M_PI_4
#define M_PI_4     0.78539816339744830961566084582      // Pi/4 
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
