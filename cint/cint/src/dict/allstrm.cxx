/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/********************************************************
* allstrm.cxx
*
*  Switches different iostream files automatically.
*  This is only experimental.
********************************************************/

#if defined(__KCC)        /* KCC  C++ compiler */

#include "kccstrm.cxx"


#elif defined(__INTEL_COMPILER) /* icc and ecc C++ compilers */

#include "iccstrm.cxx"


#elif defined(__GNUC__)  /* gcc/g++  GNU C/C++ compiler major version */

#if (__GNUC__>=3)
#include "gcc3strm.cxx"
#else
#include "libstrm.cxx"
#endif


#elif defined(__HP_aCC)     /* HP aCC C++ compiler */

#include "libstrm.cxx"


#elif defined(__hpux)     /* HP CC C++ compiler */

#include "libstrm.cxx"


#elif defined(__SUNPRO_CC)  /* Sun C++ compiler */

#if (__SUNPRO_CC>=5)
#include "sun5strm.cxx"
#else
#include "sunstrm.cxx"
#endif


#elif defined(_MSC_VER)     /* Microsoft Visual C++ version */

#include "vcstrm.cxx"


#elif defined(__SC__)       /* Symantec C/C++ compiler */

#include "libstrm.cxx"


#elif defined(__BCPLUSPLUS__)  /* Borland C++ compiler */

#include "cbstrm.cpp"


#elif defined(__alpha__) /* DEC/Compac Alpha */

#include "libstrm.cxx"


#else

#include "libstrm.cxx"


#endif

