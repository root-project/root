// @(#)root/ldap:$Id$
// Author: Oleksandr Grebenyuk   21/09/2001

/*************************************************************************
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_CintLDAP
#define ROOT_CintLDAP

#if !defined(__CINT__) && !defined(G__DICTIONARY)

#include <lber.h>   // needed for older versions of ldap.h
#include <ldap.h>

#else

struct LDAP;
struct LDAPMessage;
struct LDAPMod;

#define LDAP_VERSION1	1
#define LDAP_VERSION2	2
#define LDAP_VERSION3	3

const int LDAP_PORT              = 389;

const int LDAP_SCOPE_DEFAULT     = -1;
const int LDAP_SCOPE_BASE        = 0x0000;
const int LDAP_SCOPE_ONELEVEL    = 0x0001;
const int LDAP_SCOPE_SUBTREE     = 0x0002;

const int LDAP_MOD_ADD           = 0x0000;
const int LDAP_MOD_DELETE        = 0x0001;
const int LDAP_MOD_REPLACE       = 0x0002;
const int LDAP_MOD_BVALUES       = 0x0080;

const int LDAP_SERVER_DOWN       = 0x51;

#endif

#endif
