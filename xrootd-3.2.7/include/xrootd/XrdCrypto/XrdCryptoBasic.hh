// $Id$
#ifndef __CRYPTO_BASIC_H__
#define __CRYPTO_BASIC_H__
/******************************************************************************/
/*                                                                            */
/*                     X r d C r y p t o B a s i c. h h                       */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/


/* ************************************************************************** */
/*                                                                            */
/* Generic buffer for crypto functions needed in XrdCrypto                    */
/* Different crypto implementation (OpenSSL, Botan, ...) available as plug-in */
/*                                                                            */
/* ************************************************************************** */

#include <XProtocol/XProtocol.hh>
#include <XrdSut/XrdSutBucket.hh>

// ---------------------------------------------------------------------------//
//
// Basic buffer
//
// ---------------------------------------------------------------------------//
class XrdCryptoBasic
{
public:
   // ctor
   XrdCryptoBasic(const char *t = 0, int l = 0, const char *b = 0);
   // dtor
   virtual ~XrdCryptoBasic() 
          { if (type) delete[] type; if (membuf) delete[] membuf; }
   // getters
   virtual XrdSutBucket *AsBucket();
   char *AsHexString();
   virtual int   Length() const { return lenbuf; }
   virtual char *Buffer() const { return membuf; }
   virtual char *Type() const { return type; }
   // setters
   virtual int   FromHex(const char *hex);
   virtual int   SetLength(int l);
   virtual int   SetBuffer(int l, const char *b);
   virtual int   SetType(const char *t);
   // special setter to avoid buffer re-allocation
   virtual void  UseBuffer(int l, const char *b)
          { if (membuf) delete[] membuf; membuf = (char *)b; lenbuf = l; }

private:
   kXR_int32  lenbuf;
   char      *membuf;
   char      *type;
};

#endif
