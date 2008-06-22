// $Id$
#ifndef __CRYPTO_X509CHAIN_H__
#define __CRYPTO_X509CHAIN_H__
/******************************************************************************/
/*                                                                            */
/*                   X r d C r y p t o X 5 0 9 C h a i n . h h                */
/*                                                                            */
/* (c) 2005 G. Ganis , CERN                                                   */
/*                                                                            */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* Chain of X509 certificates.                                                */
/*                                                                            */
/* ************************************************************************** */

#include <XrdSut/XrdSutBucket.hh>
#include <XrdCrypto/XrdCryptoX509.hh>
#include <XrdCrypto/XrdCryptoX509Crl.hh>

// ---------------------------------------------------------------------------//
//                                                                            //
// XrdCryptoX509Chain                                                         //
//                                                                            //
// Light single-linked list for managing stacks of XrdCryptoX509* objects     //
//                                                                            //
// ---------------------------------------------------------------------------//

//
// Description of options for verify
typedef struct {
   int  opt;            // option container
   int  when;           // time of verification (UTC)
   int  pathlen;        // max allowed path length of chain
   XrdCryptoX509Crl *crl; // CRL
} x509ChainVerifyOpt_t;

const int kOptsCheckSelfSigned = 0x2;    // CA ckecking option

//
// Node definition
//
class XrdCryptoX509ChainNode {

private:
   XrdCryptoX509          *cert;
   XrdCryptoX509ChainNode *next;
public:
   XrdCryptoX509ChainNode(XrdCryptoX509 *c = 0, XrdCryptoX509ChainNode *n = 0)
        { cert = c; next = n;}
   virtual ~XrdCryptoX509ChainNode() { }

   XrdCryptoX509          *Cert() const { return cert; }
   XrdCryptoX509ChainNode *Next() const { return next; }

   void SetNext(XrdCryptoX509ChainNode *n) { next = n; }
};

class XrdCryptoX509Chain {

   friend class XrdCryptosslgsiX509Chain;

   enum ESearchMode { kExact = 0, kBegin = 1, kEnd = 2 };

public:
   XrdCryptoX509Chain(XrdCryptoX509 *c = 0);
   XrdCryptoX509Chain(XrdCryptoX509Chain *ch);
   virtual ~XrdCryptoX509Chain();

   // CA status
   enum ECAStatus { kUnknown = 0, kAbsent, kInvalid, kValid};

   // Error codes
   enum EX509ChainErr { kNone = 0, kInconsistent, kTooMany, kNoCA,
                        kNoCertificate, kInvalidType, kInvalidNames,
                        kRevoked, kExpired, kMissingExtension,
                        kVerifyFail, kInvalidSign, kCANotAutoSigned };

   // In case or error
   const char         *X509ChainError(EX509ChainErr e);
   const char         *LastError() const { return lastError.c_str(); }

   // Dump content
   void Dump();

   // Access information
   int                 Size() const { return size; }
   XrdCryptoX509      *End() const { return end->Cert(); }
   ECAStatus           StatusCA() const { return statusCA; }
   const char         *CAname();
   const char         *EECname();
   const char         *CAhash();
   const char         *EEChash();

   // Modifiers
   void                InsertAfter(XrdCryptoX509 *c, XrdCryptoX509 *cp);
   void                PutInFront(XrdCryptoX509 *c);
   void                PushBack(XrdCryptoX509 *c);
   void                Remove(XrdCryptoX509 *c);
   bool                CheckCA(bool checkselfsigned = 1);
   void                Cleanup(bool keepCA = 0);
   void                SetStatusCA(ECAStatus st) { statusCA = st; }

   // Search
   XrdCryptoX509      *SearchByIssuer(const char *issuer,
                                      ESearchMode mode = kExact);
   XrdCryptoX509      *SearchBySubject(const char *subject,
                                       ESearchMode mode = kExact);

   // Check validity in time
   virtual int         CheckValidity(bool outatfirst = 1, int when = 0);

   // Reorder (C(n) issuer of C(n+1)) 
   virtual int         Reorder();

   // Verify chain
   virtual bool        Verify(EX509ChainErr &e, x509ChainVerifyOpt_t *vopt = 0);

   // Pseudo - iterator functionality
   XrdCryptoX509       *Begin();
   XrdCryptoX509       *Next();

private:


   XrdCryptoX509ChainNode *begin;
   XrdCryptoX509ChainNode *current;
   XrdCryptoX509ChainNode *end;
   XrdCryptoX509ChainNode *previous;
   int                     size;
   XrdOucString            lastError;
   XrdOucString            caname;
   XrdOucString            eecname;
   XrdOucString            cahash;
   XrdOucString            eechash;
   ECAStatus               statusCA;

   XrdCryptoX509ChainNode *Find(XrdCryptoX509 *c);
   XrdCryptoX509ChainNode *FindIssuer(const char *issuer,
                                      ESearchMode mode = kExact,
                                      XrdCryptoX509ChainNode **p = 0);
   XrdCryptoX509ChainNode *FindSubject(const char *subject,
                                       ESearchMode mode = kExact,
                                       XrdCryptoX509ChainNode **p = 0);
   bool Verify(EX509ChainErr &e, const char *msg,
               XrdCryptoX509::EX509Type type, int when,
               XrdCryptoX509 *xcer, XrdCryptoX509 *xsig,
               XrdCryptoX509Crl *crl = 0);

};

#endif
