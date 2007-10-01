// @(#)root/auth:$Id$
// Author: G. Ganis, Nov 2006

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_AFSAuth
#define ROOT_AFSAuth


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// AFSAuth                                                              //
//                                                                      //
// Utility functions to acquire and handle AFS tokens.                  //
// These functions are available as separate plugin, libAFSAuth.so,     //
// depending aonly on the AFS libraries.                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

// One day as default lifetime
#define DFLTTOKENLIFETIME (24*3600)

extern "C" {
// Get AFS token for the local cell for 'usr'. The meaning of the
// information passed at 'pwd' depends on 'pwlen'. For 'pwlen <= 0'
// 'pwd' is interpreted as the plain password (null terminated string).
// For 'pwlen > 0', the 'pwlen' bytes at 'pwd' contain the password in
// for of encryption key (struct ktc_encryptionKey).
// On success a token is returned as opaque information.
// On error / failure, 0 is returned; if emsg != 0, *emsg points to an
// error message.
void *GetAFSToken(const char *usr,
                  const char *pwd, int pwlen = -1,
                  int life = DFLTTOKENLIFETIME, char **emsg = 0);

// Verify validity an AFS token. The opaque input information is the one
// returned by a successful call to GetAFSToken.
// The remaining lifetime is returned, i.e. <=0 if expired.
int VerifyAFSToken(void *token);

// Delete an AFS token returned by a successful call to GetAFSToken.
void DeleteAFSToken(void *token);

// Returns a pointer to a string with the local cell. The string must
// not be freed or deleted.
char *AFSLocalCell();

}
#endif

