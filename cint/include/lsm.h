/*************************************************************************
* lsm.h
*
* Least square method library
*
*************************************************************************/

#ifndef G__LSM_H
#define G__LSM_H

#ifndef G__LSMSL 
# ifdef G__SHAREDLIB
#include <lsm.dll>
# else
#include <lsm.c>
# endif
#endif // G__LSMSL

#endif
