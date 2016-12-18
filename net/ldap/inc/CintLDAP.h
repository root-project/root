// @(#)root/ldap:$Id$
// Author: Oleksandr Grebenyuk   21/09/2001

/*************************************************************************
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_CintLDAP
#define ROOT_CintLDAP

#ifdef __CLING__

// Loaded inside Cling, we need to mitigate duplication
// ourselves.

#include <ldap.h>

#ifndef LBER_CLASS_UNIVERSAL
#include <lber.h>   // needed for older versions of ldap.h
#endif

#else
// Regular section, the user must make sure explicitly that the
// correct set of header is included (or not).
#include <lber.h>   // needed for older versions of ldap.h
#include <ldap.h>

#endif

#endif // ROOT_CintLDAP
