
#ifndef __CINT_INTERNAL_CPP__

/* Using external C/C++ preprocessor with -p or +P option */
#if defined(__GNUC__) || defined(G__GNUC)
#include_next "sys/cdefs.h"
#else
#include "/usr/include/sys/cdefs.h"
#endif

#else /*  __CINT_INTERNAL_CPP__ */

/* Using Cint's internal preprocessor which has limitation */
/* nothing */

#endif

