// @(#)root/base:$Name:$:$Id:$
// Author: Fons Rademakers   29/9/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMD5
#define ROOT_TMD5

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMD5                                                                 //
//                                                                      //
// This code implements the MD5 message-digest algorithm.               //
// The algorithm is due to Ron Rivest. This code was                    //
// written by Colin Plumb in 1993, no copyright is claimed.             //
// This code is in the public domain; do with it what you wish.         //
//                                                                      //
// Equivalent code is available from RSA Data Security, Inc.            //
// This code has been tested against that, and is equivalent,           //
// except that you don't need to include two pages of legalese          //
// with every copy.                                                     //
//                                                                      //
// To compute the message digest of a chunk of bytes, create an         //
// TMD5 object, call Update() as needed on buffers full of bytes, and   //
// then call Final(), which will, optinally, fill a supplied 16-byte    //
// array with the digest.                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

// forward declaration
class TMD5;
Bool_t operator==(const TMD5 &m1, const TMD5 &m2);


class TMD5 {

friend Bool_t operator==(const TMD5 &m1, const TMD5 &m2);

private:
   UInt_t    fBuf[4];     //!temp buffer
   UInt_t    fBits[2];    //!temp buffer
   UChar_t   fIn[64];     //!temp buffer
   UChar_t   fDigest[16]; //message digest
   Bool_t    fFinalized;  //true if message digest has been finalized

   void Transform(UInt_t buf[4], const UInt_t in[16]);
   void ByteReverse(UChar_t *buf, UInt_t longs);

public:
   TMD5();

   void        Update(const UChar_t *buf, UInt_t len);
   void        Final();
   void        Final(UChar_t digest[16]);
   void        Print() const;
   const char *AsString() const;

   ClassDef(TMD5,1)  // MD5 cryptographic hash functions with a 128 bit output
};


inline Bool_t operator!=(const TMD5 &m1, const TMD5 &m2)
{ return !(m1 == m2); }


#endif
