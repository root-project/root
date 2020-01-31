// @(#)root/base:$Id$
// Author: Gerardo Ganis + Fons Rademakers   15/5/2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TBase64
\ingroup Base

This code implements the Base64 encoding and decoding.

Base64 encoded messages are typically used in authentication
protocols and to pack binary data in HTTP messages.
*/

#include "TBase64.h"

#include <ROOT/RConfig.hxx>

ClassImp(TBase64);

////////////////////////////////////////////////////////////////////////////////
/// Base64 encoding of 3 bytes from in.
/// Output (4 bytes) saved in out (not null terminated).

static void ToB64low(const char *in, char *out, int mod)
{
   static char b64ref[64] = {
      'A','B','C','D','E','F','G','H','I','J',
      'K','L','M','N','O','P','Q','R','S','T',
      'U','V','W','X','Y','Z',
      'a','b','c','d','e','f','g','h','i','j',
      'k','l','m','n','o','p','q','r','s','t',
      'u','v','w','x','y','z',
      '0','1','2','3','4','5','6','7','8','9',
      '+','/'
   };

   if (R__likely(mod > 2)) {
      *out++ = b64ref[ (int)(0x3F & (in[0] >> 2)) ];
      *out++ = b64ref[ 0x3F & ((0x30 & (in[0] << 4)) | (0x0F & (in[1] >> 4))) ];
      *out++ = b64ref[ 0x3F & ((0x3C & (in[1] << 2)) | (0x03 & (in[2] >> 6))) ];
      *out++ = b64ref[ 0x3F & in[2] ];
   } else if (mod == 1) {
      *out++ = b64ref[ 0x3F & (in[0] >> 2) ];
      *out++ = b64ref[ 0x3F & (0x30 & (in[0] << 4)) ];
      *out++ = '=';
      *out++ = '=';
   } else if (mod == 2) {
      *out++ = b64ref[ 0x3F & (in[0] >> 2) ];
      *out++ = b64ref[ 0x3F & ((0x30 & (in[0] << 4)) | (0x0F & (in[1] >> 4))) ];
      *out++ = b64ref[ 0x3F & (0x3C & (in[1] << 2)) ];
      *out++ = '=';
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Base64 decoding of 4 bytes from in.
/// Output (3 bytes) appended to out.
/// No check for base64-ness of input characters.

static void FromB64low(const char *in, TString &out)
{
   static int b64inv[256] = {
      -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
      -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
      -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
      52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-2,-1,-1,
      -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
      15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
      -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
      41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
      -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
      -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
      -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
      -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
      -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
      -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
      -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
      -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
   };

   const UInt_t i0 = (UInt_t)(in[0]);
   const UInt_t i1 = (UInt_t)(in[1]);
   const UInt_t i2 = (UInt_t)(in[2]);
   const UInt_t i3 = (UInt_t)(in[3]);
   if (R__likely(in[3] != '=')) {
      out.Append((char)(0xFC & (b64inv[i0] << 2)) | (0x03 & (b64inv[i1] >> 4)));
      out.Append((char)(0xF0 & (b64inv[i1] << 4)) | (0x0F & (b64inv[i2] >> 2)));
      out.Append((char)(0xC0 & (b64inv[i2] << 6)) | (0x3F &  b64inv[i3]));
   } else if (in[2] == '=') {
      out.Append((char)(0xFC & (b64inv[i0] << 2)) | (0x03 & (b64inv[i1] >> 4)));
   } else {
      out.Append((char)(0xFC & (b64inv[i0] << 2)) | (0x03 & (b64inv[i1] >> 4)));
      out.Append((char)(0xF0 & (b64inv[i1] << 4)) | (0x0F & (b64inv[i2] >> 2)));
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Transform data into a null terminated base64 string.

TString TBase64::Encode(const char *data)
{
   return Encode(data, strlen(data));
}

////////////////////////////////////////////////////////////////////////////////
/// Transform len bytes from data into a null terminated base64 string.

TString TBase64::Encode(const char *data, Int_t len)
{
   TString ret((len < 8) ? 16 : (int) (len * 1.25 + 8)); // every 3 bytes coded in 4 base64

   char oo[4];
   for (int i = 0; i < len; i += 3) {
      ToB64low(data+i, oo, len-i);
      ret.Append(oo, 4);
   }
   return ret;
}

////////////////////////////////////////////////////////////////////////////////
/// Decode a base64 string date into a generic TString.
/// No check for base64-ness of input characters.

TString TBase64::Decode(const char *data)
{
   int len = strlen(data);
   TString ret(len);

   for (int i = 0; i < len; i += 4)
      FromB64low(data+i, ret);

   return ret;
}
