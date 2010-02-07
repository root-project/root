// $Id$
#ifndef __CRYPTO_MSGDGST_H__
#define __CRYPTO_MSGDGST_H__
/******************************************************************************/
/*                                                                            */
/*                 X r d C r y p t o M s g D i g e s t . h h                  */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* Abstract interface for Message Digest crypto functionality.                */
/* Allows to plug-in modules based on different crypto implementation         */
/* (OpenSSL, Botan, ...)                                                      */
/*                                                                            */
/* ************************************************************************** */

#include <XrdCrypto/XrdCryptoBasic.hh>

// ---------------------------------------------------------------------------//
//
// Message Digest abstract buffer
//
// ---------------------------------------------------------------------------//
class XrdCryptoMsgDigest : public XrdCryptoBasic
{

public:
   XrdCryptoMsgDigest() : XrdCryptoBasic() { }
   virtual ~XrdCryptoMsgDigest() { }

   // Validity
   virtual bool IsValid();

   // Methods
   virtual int Reset(const char *dgst);
   virtual int Update(const char *b, int l);
   virtual int Final();

   // Equality operator
   bool operator==(const XrdCryptoMsgDigest md);
};

#endif
