// $Id$
#ifndef __CRYPTO_SSLGSIX509CHAIN_H__
#define __CRYPTO_SSLGSIX509CHAIN_H__
/******************************************************************************/
/*                                                                            */
/*           X r d C r y p t o s s l g s i X 5 0 9 C h a i n . h h            */
/*                                                                            */
/* (c) 2005 G. Ganis , CERN                                                   */
/*                                                                            */
/******************************************************************************/

/* ************************************************************************** */
/*                                                                            */
/* Chain of X509 certificates following GSI policy(ies).                      */
/*                                                                            */
/* ************************************************************************** */

#include <XrdCrypto/XrdCryptoX509Chain.hh>

// ---------------------------------------------------------------------------//
//                                                                            //
// XrdCryptosslgsiX509Chain                                                   //
//                                                                            //
// Enforce GSI policies on X509 certificate chains                            //
//                                                                            //
// ---------------------------------------------------------------------------//

const int kOptsRfc3820 = 0x1;

class XrdCryptosslgsiX509Chain : public XrdCryptoX509Chain {

public:
   XrdCryptosslgsiX509Chain(XrdCryptoX509 *c = 0) : XrdCryptoX509Chain(c) { }
   XrdCryptosslgsiX509Chain(XrdCryptosslgsiX509Chain *c) : XrdCryptoX509Chain(c) { }
   virtual ~XrdCryptosslgsiX509Chain() { }

   // Verify chain
   bool Verify(EX509ChainErr &e, x509ChainVerifyOpt_t *vopt = 0);

private:

   // Proxy naming rules 
   bool SubjectOK(EX509ChainErr &e, XrdCryptoX509 *xcer);
};

#endif
