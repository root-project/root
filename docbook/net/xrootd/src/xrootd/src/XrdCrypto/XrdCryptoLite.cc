/******************************************************************************/
/*                                                                            */
/*                      X r d C r y p t o L i t e . c c                       */
/*                                                                            */
/* (c) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$

const char *XrdCryptoLiteCVSID = "$Id$";

#include <errno.h>
#include <string.h>

#include "XrdCrypto/XrdCryptoLite.hh"

/******************************************************************************/
/*                                C r e a t e                                 */
/******************************************************************************/

/* This is simply a landing pattern for all supported crypto methods; to avoid
   requiring the client to include specific implementation include files. Add
   your implementation in the following way:
   1. Define an external function who's signature follows:
      XrdCryptoLite *XrdCryptoLite_New_xxxx(const char Type)
      where 'xxxx' corresponds to the passed Name argument.
   2. Insert the extern to the function.
   3. Insert the code segment that calls the function.
*/
  
XrdCryptoLite *XrdCryptoLite::Create(int &rc, const char *Name, const char Type)
{
   extern XrdCryptoLite *XrdCryptoLite_New_bf32(const char Type);
   XrdCryptoLite *cryptoP = 0;

   if (!strcmp(Name, "bf32"))     cryptoP = XrdCryptoLite_New_bf32(Type);

// Return appropriately
//
   rc = (cryptoP ? 0 : EPROTONOSUPPORT);
   return cryptoP;
}
