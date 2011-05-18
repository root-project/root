// $Id$

const char *XrdCryptosslgsiX509ChainCVSID = "$Id$";
/******************************************************************************/
/*                                                                            */
/*           X r d C r y p t o s s l g s i X 5 0 9 C h a i n . h h            */
/*                                                                            */
/* (c) 2005  G. Ganis, CERN                                                   */
/*                                                                            */
/******************************************************************************/
#include <string.h>
#include <time.h>

#include <XrdCrypto/XrdCryptosslgsiAux.hh>
#include <XrdCrypto/XrdCryptosslgsiX509Chain.hh>
#include <XrdCrypto/XrdCryptoTrace.hh>

// ---------------------------------------------------------------------------//
//                                                                            //
// XrdCryptosslgsiX509Chain                                                   //
//                                                                            //
// Enforce GSI policies on X509 certificate chains                            //
//                                                                            //
// ---------------------------------------------------------------------------//

//___________________________________________________________________________
bool XrdCryptosslgsiX509Chain::Verify(EX509ChainErr &errcode, x509ChainVerifyOpt_t *vopt)
{
   // Verify the chain
   EPNAME("X509Chain::Verify");
   errcode = kNone; 

   // There must be at least a CA and a EEC. 
   if (size < 2) {
      DEBUG("Nothing to verify (size: "<<size<<")");
      return 0;
   }
   if (QTRACE(Dump)) { Dump(); }

   //
   // Reorder if needed
   if (Reorder() != 0) {
      errcode = kInconsistent;
      lastError = ":";
      lastError += X509ChainError(errcode);
      return 0;
   }

   //
   // Verification options
   int opt  = (vopt) ? vopt->opt : 0;
   int when = (vopt) ? vopt->when : (int)time(0);
   int plen = (vopt) ? vopt->pathlen : -1;
   XrdCryptoX509Crl *crl = (vopt) ? vopt->crl : 0;

   //
   // Global path depth length consistency check
   if (plen > -1 && plen < size) {
      errcode = kTooMany;
      lastError = "checking path depth: ";
      lastError += X509ChainError(errcode);
   }

   //
   // Check the first certificate: it MUST be of CA type, valid,
   // self-signed
   XrdCryptoX509ChainNode *node = begin;
   XrdCryptoX509 *xcer = node->Cert();      // Certificate under exam
   XrdCryptoX509 *xsig = xcer;              // Signing certificate
   if (statusCA == kUnknown) {
      if (!XrdCryptoX509Chain::Verify(errcode, "CA: ",
                                      XrdCryptoX509::kCA, when, xcer, xsig))
         return 0;
      statusCA = kValid;
   } else if (statusCA == kAbsent || statusCA == kInvalid) {
      errcode = kNoCA;
      lastError = X509ChainError(errcode);
      return 0;
   }

   //
   // Update the max path depth len
   if (plen > -1)
      plen -= 1;
   //
   // Check the end-point entity (or sub-CA) certificate
   while (node->Next() && strcmp(node->Next()->Cert()->Type(), "Proxy")) {
      xsig = xcer;
      node = node->Next();
      xcer = node->Cert();
      if (!XrdCryptoX509Chain::Verify(errcode, "EEC or sub-CA: ",
                                      XrdCryptoX509::kUnknown,
                                      when, xcer, xsig, crl))
         return 0;
      //
      // Update the max path depth len
      if (plen > -1)
         plen -= 1;
   }

   //
   // There are proxy certificates
   xsig = xcer;
   node = node->Next();
   while (node && (plen == -1 || plen > 0)) {

      // Attache to certificate
      xcer = node->Cert();

      // Proxy subject name must follow some rules
      if (!SubjectOK(errcode, xcer))
         return 0;

      // Check if ProxyCertInfo extension is there (required by RFC3820)
      int pxplen = -1;
      if (opt & kOptsRfc3820) {
         const void *extdata = xcer->GetExtension(gsiProxyCertInfo_OID);
         if (!extdata || !XrdSslgsiProxyCertInfo(extdata, pxplen)) {
            errcode = kMissingExtension;
            lastError = "rfc3820: ";
            lastError += X509ChainError(errcode);
            return 0;
         }
      }
      // Update plen, if needed
      if (plen == -1) {
         plen = (pxplen > -1) ? pxplen : plen;
      } else {
         plen--;
         // Aply stricter rules if required
         plen = (pxplen > -1 && pxplen < plen) ? pxplen : plen;
      }

      // Standard verification
      if (!XrdCryptoX509Chain::Verify(errcode, "Proxy: ",
                                      XrdCryptoX509::kProxy, when, xcer, xsig))
         return 0;

      // Get next
      xsig = xcer;
      node = node->Next();
   }

   // We are done (successfully!)
   return 1;
}


//___________________________________________________________________________
bool XrdCryptosslgsiX509Chain::SubjectOK(EX509ChainErr &errcode, XrdCryptoX509 *xcer)
{
   // Apply GSI rules for proxy subject names

   // Check inputs
   if (!xcer) {
      errcode = kNoCertificate;
      lastError = "subject check:";
      lastError += X509ChainError(errcode);
      return 0;
   }

   // This applies only to proxies
   if (xcer->type != XrdCryptoX509::kProxy)
      return 1;

   // Pointers to names
   if (!(xcer->Subject()) || !(xcer->Issuer())) {
      errcode = kInvalidNames;
      lastError = "subject check:";
      lastError += X509ChainError(errcode);
      return 0;
   }

   // Subject name must start with issuer name.
   // We need the length of the common part between issuer and subject.
   // We allow proxies issued by other proxies. In such cases we must
   // ignore the last '/CN=' in the issuer name; this explains the need
   // for the following gymnastic.

   int ilen = strlen(xcer->Issuer());
   if (strncmp(xcer->Subject(), xcer->Issuer(), ilen)) {
      // Check if the issuer is a proxy: ignore the last 'CN='
      char *pcn = (char *) strstr(xcer->Issuer(), "/CN=");
      if (pcn) {
         char *pcnn = 0;
         while ((pcnn = (char *) strstr(pcn+1,"/CN=")))
	    pcn = pcnn;
         ilen = (int)(pcn - xcer->Issuer());
      }
      if (strncmp(xcer->Subject() + ilen,"/CN=",4)) {
         errcode = kInvalidNames;
         lastError = "proxy subject check: found additional chars :";
         lastError += X509ChainError(errcode);
         return 0;
      }
      if (strncmp(xcer->Subject(), xcer->Issuer(), ilen)) {
         errcode = kInvalidNames;
         lastError = "proxy issuer check: issuer not found in subject :";
         lastError += X509ChainError(errcode);
         return 0;
      }
   }

   // A common name must be appendend
   char *pp = (char *)strstr(xcer->Subject()+ilen, "CN=");
   if (!pp) {
      errcode = kInvalidNames;
      lastError = "proxy subject check: no appended 'CN='";
      lastError += X509ChainError(errcode);
      return 0;
   }

   // But only one
   pp = strstr(pp+strlen("CN="), "CN=");
   if (pp) {
      errcode = kInvalidNames;
      lastError = "proxy subject check: too many appended 'CN='s";
      lastError += X509ChainError(errcode);
      return 0;
   }

   // We are done
   return 1;
}


