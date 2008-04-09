// $Id$
#ifndef __CRYPTO_SSLAUX_H__
#define __CRYPTO_SSLAUX_H__
/******************************************************************************/
/*                                                                            */
/*                  X r d C r y p t o S s l A u x . h h                       */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* OpenSSL utility functions                                                  */
/*                                                                            */
/* ************************************************************************** */

#include <XrdCrypto/XrdCryptoAux.hh>
#include <XrdCrypto/XrdCryptoX509Chain.hh>
#include <openssl/asn1.h>

#define kSslKDFunDefLen  24

//
// Password-Based Key Derivation Function 2, specified in PKCS #5
//
int XrdCryptosslKDFunLen(); // default buffer length
int XrdCryptosslKDFun(const char *pass, int plen, const char *salt, int slen,
                      char *key, int len);
//
// X509 manipulation: certificate verification
bool XrdCryptosslX509VerifyCert(XrdCryptoX509 *c, XrdCryptoX509 *r);
// chain verification
bool XrdCryptosslX509VerifyChain(XrdCryptoX509Chain *chain, int &errcode);
// chain export to bucket
XrdSutBucket *XrdCryptosslX509ExportChain(XrdCryptoX509Chain *c, bool key = 0);
// chain export to file (proxy file creation)
int XrdCryptosslX509ChainToFile(XrdCryptoX509Chain *c, const char *fn);
// certificates from file parsing
int XrdCryptosslX509ParseFile(const char *fname, XrdCryptoX509Chain *c);
// certificates from bucket parsing
int XrdCryptosslX509ParseBucket(XrdSutBucket *b, XrdCryptoX509Chain *c);
//
// Function to convert from ASN1 time format into UTC since Epoch (Jan 1, 1970) 
int XrdCryptosslASN1toUTC(ASN1_TIME *tsn1);

/******************************************************************************/
/*          E r r o r   L o g g i n g / T r a c i n g   F l a g s             */
/******************************************************************************/
#define sslTRACE_ALL       0x0007
#define sslTRACE_Dump      0x0004
#define sslTRACE_Debug     0x0002
#define sslTRACE_Notify    0x0001

#endif

