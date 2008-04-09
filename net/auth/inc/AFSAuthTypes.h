// @(#)root/auth:$Id$
// Author: G. Ganis, Nov 2006

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_AFSAuthTypes
#define ROOT_AFSAuthTypes


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// AFSAuthTypes                                                         //
//                                                                      //
// Sugnatures for the utility functions to acquire / handle AFS tokens. //
// Needed when loading dynamically the library.                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

// AFS token getter: arguments are
//     1st         user name
//     2nd         password or encrypted key buffer
//     3rd         length of encrypted key buffer or <=0 for plain password
//     4th         lifetime in seconds (-1 for default - 1 day)
//     5th         reason message in case of failure
// On success a token is returned as opaque information.
// On error / failure, 0 is returned; if emsg != 0, *emsg points to an
// error message.
typedef void *(*GetAFSToken_t)(const char *, const char *, int, int, char **);

// Verify validity an AFS token. The opaque input information is the one
// returned by a successful call to GetAFSToken.
// The remaining lifetime is returned, i.e. <=0 if expired.
typedef int (*VerifyAFSToken_t)(void *);

// Delete an AFS token returned by a successful call to GetAFSToken.
typedef void (*DeleteAFSToken_t)(void *);

// Returns a pointer to a string with the local cell. The string must
// not be freed or deleted.
typedef char *(*AFSLocalCell_t)();

#endif

