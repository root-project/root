// $Id$
#ifndef __SECPWD_PLATFORM_
#define __SECPWD_PLATFORM_
/******************************************************************************/
/*                                                                            */
/*                 X r d S e c p w d P l a t f o r m. h h                     */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//
// crypt
//
#if defined(__solaris__)
#include <crypt.h>
#endif
#if defined(__osf__) || defined(__sgi) || defined(__macos__)
extern "C" char *crypt(const char *, const char *);
#endif

//
// shadow passwords
//
#include <grp.h>
// For shadow passwords
#if defined(__solaris__)
#ifndef R__SHADOWPW
#define R__SHADOWPW
#endif
#endif
#ifdef R__SHADOWPW
#include <shadow.h>
#endif

#endif
