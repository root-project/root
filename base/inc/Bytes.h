/* @(#)root/base:$Name$:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Bytes
#define ROOT_Bytes


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Bytes                                                                //
//                                                                      //
// A set of inline byte handling routines.                              //
//                                                                      //
// The set of tobuf() and frombuf() routines take care of packing a     //
// basic type value into a buffer in network byte order (i.e. they      //
// perform byte swapping when needed). The buffer does not have to      //
// start on a machine (long) word boundary.                             //
//                                                                      //
// For __GNUC__ on linux on i486 processors and up                      //
// use the `bswap' opcode provided by the GNU C Library.                //
//                                                                      //
// The set of host2net() and net2host() routines convert a basic type   //
// value from host to network byte order and vice versa. On BIG ENDIAN  //
// machines this is a no op.                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#ifndef __CINT__
#include <string.h>
#endif


#if defined(__linux) && defined(__i386__) && !defined __CINT__
#include "Byteswap.h"
#endif

//______________________________________________________________________________
inline void tobuf(char *&buf, UChar_t x)
{
   *buf++ = x;
}

inline void tobuf(char *&buf, UShort_t x)
{
#ifdef R__BYTESWAP
# if defined(__linux) && defined(__i386__)
   *((UShort_t *)buf) = bswap_16(x);
# else
   char *sw = (char *)&x;
   buf[0] = sw[1];
   buf[1] = sw[0];
# endif
#else
   memcpy(buf, &x, sizeof(UShort_t));
#endif
   buf += sizeof(UShort_t);
}

inline void tobuf(char *&buf, UInt_t x)
{
#ifdef R__BYTESWAP
# if defined(__linux) && defined(__i386__)
   *((UInt_t *)buf) = bswap_32(x);
# else
   char *sw = (char *)&x;
   buf[0] = sw[3];
   buf[1] = sw[2];
   buf[2] = sw[1];
   buf[3] = sw[0];
# endif
#else
   memcpy(buf, &x, sizeof(UInt_t));
#endif
   buf += sizeof(UInt_t);
}

inline void tobuf(char *&buf, ULong_t x)
{
#ifdef R__BYTESWAP
#ifdef R__B64
# if defined(__linux) && defined(__i386__) && defined __GNUC__ && __GNUC__ >= 2
   *((ULong_t *)buf) = bswap_64(x);
# else
   char *sw = (char *)&x;
   buf[0] = sw[7];
   buf[1] = sw[6];
   buf[2] = sw[5];
   buf[3] = sw[4];
   buf[4] = sw[3];
   buf[5] = sw[2];
   buf[6] = sw[1];
   buf[7] = sw[0];
# endif
#else
# if defined(__linux) && defined(__i386__)
   *((ULong_t *)buf) = bswap_32(x);
# else
   char *sw = (char *)&x;
   buf[0] = sw[3];
   buf[1] = sw[2];
   buf[2] = sw[1];
   buf[3] = sw[0];
# endif
#endif
#else
   memcpy(buf, &x, sizeof(ULong_t));
#endif
   buf += sizeof(ULong_t);
}

inline void tobuf(char *&buf, Float_t x)
{
#ifdef R__BYTESWAP
# if defined(__linux) && defined(__i386__)
   *((UInt_t *)buf) = bswap_32(*((UInt_t *)&x));
# else
   char *sw = (char *)&x;
   buf[0] = sw[3];
   buf[1] = sw[2];
   buf[2] = sw[1];
   buf[3] = sw[0];
# endif
#else
   memcpy(buf, &x, sizeof(Float_t));
#endif
   buf += sizeof(Float_t);
}

inline void tobuf(char *&buf, Double_t x)
{
#ifdef R__BYTESWAP
# if defined(__linux) && defined(__i386__) && defined __GNUC__ && __GNUC__ >= 2
   *((unsigned long long *)buf) = bswap_64(*((unsigned long long *)&x));
# else
   char *sw = (char *)&x;
   buf[0] = sw[7];
   buf[1] = sw[6];
   buf[2] = sw[5];
   buf[3] = sw[4];
   buf[4] = sw[3];
   buf[5] = sw[2];
   buf[6] = sw[1];
   buf[7] = sw[0];
# endif
#else
   memcpy(buf, &x, sizeof(Double_t));
#endif
   buf += sizeof(Double_t);
}

inline void frombuf(char *&buf, UChar_t *x)
{
   *x = *buf++;
}

inline void frombuf(char *&buf, UShort_t *x)
{
#ifdef R__BYTESWAP
# if defined(__linux) && defined(__i386__)
   *x = bswap_16(*((UShort_t *)buf));
# else
   char *sw = (char *)x;
   sw[0] = buf[1];
   sw[1] = buf[0];
# endif
#else
   memcpy(x, buf, sizeof(UShort_t));
#endif
   buf += sizeof(UShort_t);
}

inline void frombuf(char *&buf, UInt_t *x)
{
#ifdef R__BYTESWAP
# if defined(__linux) && defined(__i386__)
   *x = bswap_32(*((UInt_t *)buf));
# else
   char *sw = (char *)x;
   sw[0] = buf[3];
   sw[1] = buf[2];
   sw[2] = buf[1];
   sw[3] = buf[0];
# endif
#else
   memcpy(x, buf, sizeof(UInt_t));
#endif
   buf += sizeof(UInt_t);
}

inline void frombuf(char *&buf, ULong_t *x)
{
#ifdef R__BYTESWAP
#ifdef R__B64
# if defined(__linux) && defined(__i386__) && defined __GNUC__ && __GNUC__ >= 2
   *x = bswap_64(*((ULong_t *)buf));
# else
   char *sw = (char *)x;
   sw[0] = buf[7];
   sw[1] = buf[6];
   sw[2] = buf[5];
   sw[3] = buf[4];
   sw[4] = buf[3];
   sw[5] = buf[2];
   sw[6] = buf[1];
   sw[7] = buf[0];
# endif
#else
# if defined(__linux) && defined(__i386__)
   *x = bswap_32(*((ULong_t *)buf));
# else
   char *sw = (char *)x;
   sw[0] = buf[3];
   sw[1] = buf[2];
   sw[2] = buf[1];
   sw[3] = buf[0];
# endif
#endif
#else
   memcpy(x, buf, sizeof(ULong_t));
#endif
   buf += sizeof(ULong_t);
}

inline void frombuf(char *&buf, Float_t *x)
{
#ifdef R__BYTESWAP
# if defined(__linux) && defined(__i386__)
   *((UInt_t*)x) = bswap_32(*((UInt_t *)buf));
# else
   char *sw = (char *)x;
   sw[0] = buf[3];
   sw[1] = buf[2];
   sw[2] = buf[1];
   sw[3] = buf[0];
# endif
#else
   memcpy(x, buf, sizeof(Float_t));
#endif
   buf += sizeof(Float_t);
}

inline void frombuf(char *&buf, Double_t *x)
{
#ifdef R__BYTESWAP
# if defined(__linux) && defined(__i386__) && defined __GNUC__ && __GNUC__ >= 2
   *((unsigned long long*)x) = bswap_64(*((unsigned long long *)buf));
# else
   char *sw = (char *)x;
   sw[0] = buf[7];
   sw[1] = buf[6];
   sw[2] = buf[5];
   sw[3] = buf[4];
   sw[4] = buf[3];
   sw[5] = buf[2];
   sw[6] = buf[1];
   sw[7] = buf[0];
# endif
#else
   memcpy(x, buf, sizeof(Double_t));
#endif
   buf += sizeof(Double_t);
}

inline void tobuf(char *&buf, Char_t x)  { tobuf(buf, (UChar_t) x); }
inline void tobuf(char *&buf, Short_t x) { tobuf(buf, (UShort_t) x); }
inline void tobuf(char *&buf, Int_t x)   { tobuf(buf, (UInt_t) x); }
inline void tobuf(char *&buf, Long_t x)  { tobuf(buf, (ULong_t) x); }

inline void frombuf(char *&buf, Char_t *x)  { frombuf(buf, (UChar_t *) x); }
inline void frombuf(char *&buf, Short_t *x) { frombuf(buf, (UShort_t *) x); }
inline void frombuf(char *&buf, Int_t *x)   { frombuf(buf, (UInt_t *) x); }
inline void frombuf(char *&buf, Long_t *x)  { frombuf(buf, (ULong_t *) x); }


//______________________________________________________________________________
#ifdef R__BYTESWAP
inline UShort_t host2net(UShort_t x)
{
# if defined(__linux) && defined(__i386__)
   return bswap_16(x);
# else
   return (((x & 0x00ff) << 8) | ((x & 0xff00) >> 8));
#endif
}

inline UInt_t host2net(UInt_t x)
{
# if defined(__linux) && defined(__i386__)
   return bswap_32(x);
# else
   return (((x & 0x000000ffU) << 24) | ((x & 0x0000ff00U) <<  8) |
           ((x & 0x00ff0000U) >>  8) | ((x & 0xff000000U) >> 24));
#endif
}

inline ULong_t host2net(ULong_t x)
{
#ifdef R__B64
# if defined(__linux) && defined(__i386__) && defined __GNUC__ && __GNUC__ >= 2
   return bswap_64(x);
# else
   char sw[sizeof(ULong_t)];
   *(ULong_t *)sw = x;

   char *sb = (char *)&x;
   sb[0] = sw[7];
   sb[1] = sw[6];
   sb[2] = sw[5];
   sb[3] = sw[4];
   sb[4] = sw[3];
   sb[5] = sw[2];
   sb[6] = sw[1];
   sb[7] = sw[0];
   return x;
#endif
#else
   return (ULong_t)host2net((UInt_t) x);
#endif
}

inline Float_t host2net(Float_t xx)
{
# if defined(__linux) && defined(__i386__)
   return bswap_32(*((UInt_t *)&xx));
# else
   UInt_t *x = (UInt_t *)&xx;
   *x = (((*x & 0x000000ffU) << 24) | ((*x & 0x0000ff00U) <<  8) |
         ((*x & 0x00ff0000U) >>  8) | ((*x & 0xff000000U) >> 24));
   return xx;
#endif
}

inline Double_t host2net(Double_t x)
{
# if defined(__linux) && defined(__i386__) && defined __GNUC__ && __GNUC__ >= 2
   return bswap_64(*((unsigned long long *)&x));
# else
   char sw[sizeof(Double_t)];
   *(Double_t *)sw = x;

   char *sb = (char *)&x;
   sb[0] = sw[7];
   sb[1] = sw[6];
   sb[2] = sw[5];
   sb[3] = sw[4];
   sb[4] = sw[3];
   sb[5] = sw[2];
   sb[6] = sw[1];
   sb[7] = sw[0];
   return x;
#endif
}
#else  /* R__BYTESWAP */
inline UShort_t host2net(UShort_t x) { return x; }
inline UInt_t   host2net(UInt_t x)   { return x; }
inline ULong_t  host2net(ULong_t x)  { return x; }
inline Float_t  host2net(Float_t x)  { return x; }
inline Double_t host2net(Double_t x) { return x; }
#endif

inline Short_t  host2net(Short_t x) { return host2net((UShort_t)x); }
inline Int_t    host2net(Int_t x)   { return host2net((UInt_t)x); }
inline Long_t   host2net(Long_t x)  { return host2net((ULong_t)x); }

inline UShort_t net2host(UShort_t x) { return host2net(x); }
inline Short_t  net2host(Short_t x)  { return host2net(x); }
inline UInt_t   net2host(UInt_t x)   { return host2net(x); }
inline Int_t    net2host(Int_t x)    { return host2net(x); }
inline ULong_t  net2host(ULong_t x)  { return host2net(x); }
inline Long_t   net2host(Long_t x)   { return host2net(x); }
inline Float_t  net2host(Float_t x)  { return host2net(x); }
inline Double_t net2host(Double_t x) { return host2net(x); }

#endif
