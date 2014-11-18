// $Id$
#ifndef __CRYPTO_AUX_H__
#define __CRYPTO_AUX_H__

#include <stdio.h>
#ifndef WIN32
#include "XrdSys/XrdSysHeaders.hh"
#endif
#include <XProtocol/XProtocol.hh>

/******************************************************************************/
/*                 M i s c e l l a n e o u s   D e f i n e s                  */
/******************************************************************************/
#define ABSTRACTMETHOD(x) {cerr <<"Method "<<x<<" must be overridden!" <<endl;}

/******************************************************************************/
/*          E r r o r   L o g g i n g / T r a c i n g   F l a g s             */
/******************************************************************************/
#define cryptoTRACE_ALL       0x0007
#define cryptoTRACE_Dump      0x0004
#define cryptoTRACE_Debug     0x0002
#define cryptoTRACE_Notify    0x0001

// RSA parameters
#define XrdCryptoMinRSABits 512
#define XrdCryptoDefRSABits 1024
#define XrdCryptoDefRSAExp  0x10001

/******************************************************************************/
/*                     U t i l i t y   F u n c t i o n s                      */
/******************************************************************************/
typedef int (*XrdCryptoKDFunLen_t)();
typedef int (*XrdCryptoKDFun_t)(const char *pass, int plen,
                                const char *salt, int slen,
                                char *key, int klen);
int XrdCryptoKDFunLen();
int XrdCryptoKDFun(const char *pass, int plen, const char *salt, int slen,
                   char *key, int klen);


/******************************************************************************/
/*  X r d C r y p t o S e t T r a c e                                         */
/*                                                                            */
/*  Set trace flags according to 'trace'                                      */
/*                                                                            */
/******************************************************************************/
//______________________________________________________________________________
void XrdCryptoSetTrace(kXR_int32 trace);

#endif
