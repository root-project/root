// $Id$
#ifndef __CRYPTO_FACTORY_H__
#define __CRYPTO_FACTORY_H__
/******************************************************************************/
/*                                                                            */
/*                 X r d C r y p t o F a c t o r y . h h                      */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* Abstract interface for a crypto factory                                    */
/* Allows to plug-in modules based on different crypto implementation         */
/* (OpenSSL, Botan, ...)                                                      */
/*                                                                            */
/* ************************************************************************** */

#include <XrdCrypto/XrdCryptoAux.hh>

#define MAXFACTORYNAMELEN  10
// ---------------------------------------------------------------------------//
//
// Abstract Crypto Factory
//
// ---------------------------------------------------------------------------//

class XrdSutBucket;
class XrdCryptoCipher;
class XrdCryptoMsgDigest;
class XrdCryptoRSA;
class XrdCryptoX509;
class XrdCryptoX509Chain;
class XrdCryptoX509Crl;
class XrdCryptoX509Req;

#if 1
//
// Prototypes for some Utility Functions

// Key derivation function
typedef int (*XrdCryptoKDFunLen_t)();
typedef int (*XrdCryptoKDFun_t)(const char *pass, int plen,
                                const char *salt, int slen,
                                char *key, int klen);

// X509 manipulation: certificate verification
typedef bool (*XrdCryptoX509VerifyCert_t)(XrdCryptoX509 *c, XrdCryptoX509 *r);
// chain verification
typedef bool (*XrdCryptoX509VerifyChain_t)(XrdCryptoX509Chain *chain,
                                           int &errcode);
// chain export
typedef XrdSutBucket *(*XrdCryptoX509ExportChain_t)(XrdCryptoX509Chain *, bool);

// chain to file
typedef int (*XrdCryptoX509ChainToFile_t)(XrdCryptoX509Chain *, const char *);

// certificates from file parsing
typedef int (*XrdCryptoX509ParseFile_t)(const char *fname,
                                        XrdCryptoX509Chain *);
// certificates from bucket parsing
typedef int (*XrdCryptoX509ParseBucket_t)(XrdSutBucket *,
                                          XrdCryptoX509Chain *);
#endif

class XrdCryptoFactory
{
private:
   char    name[MAXFACTORYNAMELEN];
   int     fID;
public:
   XrdCryptoFactory(const char *n = "Unknown", int id = -1);
   virtual ~XrdCryptoFactory() { }

   // Set trace flags
   virtual void SetTrace(kXR_int32 trace);

   // Get the factory name
   char *Name() const { return (char *)&name[0]; }
   int   ID() const { return fID; }

   // Get the right factory
   static XrdCryptoFactory *GetCryptoFactory(const char *factoryname);

   // Hook to a Key Derivation Function (PBKDF2 when possible)
   virtual XrdCryptoKDFunLen_t KDFunLen(); // Length of buffer
   virtual XrdCryptoKDFun_t KDFun();

   // Cipher constructors
   virtual bool SupportedCipher(const char *t);
   virtual XrdCryptoCipher *Cipher(const char *t, int l = 0);
   virtual XrdCryptoCipher *Cipher(const char *t, int l, const char *k, 
                                   int liv, const char *iv);
   virtual XrdCryptoCipher *Cipher(XrdSutBucket *b);
   virtual XrdCryptoCipher *Cipher(int bits, char *pub, int lpub, const char *t = 0);
   virtual XrdCryptoCipher *Cipher(const XrdCryptoCipher &c);

   // MsgDigest constructors
   virtual bool SupportedMsgDigest(const char *dgst);
   virtual XrdCryptoMsgDigest *MsgDigest(const char *dgst);

   // RSA constructors
   virtual XrdCryptoRSA *RSA(int b = 0, int e = 0);
   virtual XrdCryptoRSA *RSA(const char *p, int l = 0);
   virtual XrdCryptoRSA *RSA(const XrdCryptoRSA &r);

   // X509 constructors
   virtual XrdCryptoX509 *X509(const char *cf, const char *kf = 0);
   virtual XrdCryptoX509 *X509(XrdSutBucket *b);

   // X509 CRL constructors
   virtual XrdCryptoX509Crl *X509Crl(const char *crlfile, int opt = 0);
   virtual XrdCryptoX509Crl *X509Crl(XrdCryptoX509 *cacert);

   // X509 REQ constructors
   virtual XrdCryptoX509Req *X509Req(XrdSutBucket *bck);

   // Hooks to handle X509 certificates
   virtual XrdCryptoX509VerifyCert_t X509VerifyCert();
   virtual XrdCryptoX509VerifyChain_t X509VerifyChain();
   virtual XrdCryptoX509ParseFile_t X509ParseFile();
   virtual XrdCryptoX509ParseBucket_t X509ParseBucket();
   virtual XrdCryptoX509ExportChain_t X509ExportChain();
   virtual XrdCryptoX509ChainToFile_t X509ChainToFile();

   // Equality operator
   bool operator==(const XrdCryptoFactory factory);
};
#endif
