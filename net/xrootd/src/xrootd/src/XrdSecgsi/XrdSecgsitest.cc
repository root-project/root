// $Id$
//
//  Test program for XrdSecgsi
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <XrdOuc/XrdOucString.hh>
#include <XrdSys/XrdSysLogger.hh>
#include <XrdSys/XrdSysError.hh>

#include <XrdSut/XrdSutAux.hh>

#include <XrdCrypto/XrdCryptoAux.hh>
#include <XrdCrypto/XrdCryptoFactory.hh>
#include <XrdCrypto/XrdCryptoX509.hh>
#include <XrdCrypto/XrdCryptoX509Req.hh>
#include <XrdCrypto/XrdCryptoX509Chain.hh>
#include <XrdCrypto/XrdCryptoX509Crl.hh>

#include <XrdCrypto/XrdCryptosslgsiX509Chain.hh>
#include <XrdCrypto/XrdCryptosslgsiAux.hh>

#include <XrdSecgsi/XrdSecgsiTrace.hh>

#include <openssl/x509v3.h>

//
// Globals 

// #define PRINT(x) {cerr <<x <<endl;}
XrdCryptoFactory *gCryptoFactory = 0;

const char *CAcert = "/etc/grid-security/certificates/1b3f034a.0";
const char *CAbad  = "/etc/grid-security/certificates/42864e48.0";
const char *CAwrg  = "/etc/grid-security/certificates/globus-host-ssl.conf.1b3f034a";
const char *EEcert = "/home/ganis/.globus/usercert.pem";
const char *EEkey  = "/home/ganis/.globus/userkey.pem";
const char *PXcert = "/tmp/x509up_u2759";
const char *PXcrtp = "/tmp/x509up_u2759p";
const char *HOcert = "/etc/grid-security/root/rootcert.pem";
const char *HOkey  = "/etc/grid-security/root/rootkey.pem";
const char *CAcer1  = "/etc/grid-security/certificates/fa3af1d7.0";
const char *CRLcer1 = "/etc/grid-security/certificates/fa3af1d7_r0";

//
// For error logging and tracing
static XrdSysLogger Logger;
static XrdSysError eDest(0,"gsitest_");
XrdOucTrace *gsiTrace = 0;

int main( int argc, char **argv )
{
   // Test implemented functionality
   EPNAME("main");
   char cryptomod[64] = "ssl";
   char outname[256] = {0};

   //
   // Initiate error logging and tracing
   eDest.logger(&Logger);
   if (!gsiTrace)
      gsiTrace = new XrdOucTrace(&eDest);
   if (gsiTrace) {
      // Medium level
      gsiTrace->What |= (TRACE_Authen | TRACE_Debug);
   }
   //
   // Set debug flags in other modules
   XrdSutSetTrace(sutTRACE_Debug);
   XrdCryptoSetTrace(cryptoTRACE_Debug);

   //
   // Determine application name
   char *p = argv[0];
   int k = strlen(argv[0]);
   while (k--)
      if (p[k] == '/') break;
   strcpy(outname,p+k+1);

   //
   // Load the crypto factory
   if (!(gCryptoFactory = XrdCryptoFactory::GetCryptoFactory(cryptomod))) {
      PRINT(": cannot instantiate factory "<<cryptomod);
      exit(1);
   }
   gCryptoFactory->SetTrace(cryptoTRACE_Debug);

   PRINT(": --------------------------------------------------- ");

   //
   // Test certificate loading
   XrdCryptoX509 *xCA = gCryptoFactory->X509(CAcert);
   if (xCA) {
      xCA->Dump();
   } else {
      PRINT( ": problems loading CA cert");
   }
   XrdCryptoX509 *xCB = gCryptoFactory->X509(CAbad);
   if (xCB) {
      xCB->Dump();
   } else {
      PRINT( ": problems loading CB cert");
   }
   XrdCryptoX509 *xEE = gCryptoFactory->X509(EEcert);
   if (xEE) {
      xEE->Dump();
   } else {
      PRINT( ": problems loading EE cert");
   }
   XrdCryptoX509 *xPX = gCryptoFactory->X509(PXcert);
   if (xPX) {
      xPX->Dump();
   } else {
      PRINT( ": problems loading PX cert");
   }
   //
   PRINT(": --------------------------------------------------- ");
   PRINT(": Testing ParseFile ... ");
   XrdCryptoX509ParseFile_t ParseFile = gCryptoFactory->X509ParseFile();
   XrdCryptoRSA *key = 0;
   XrdCryptoX509Chain *chain = new XrdCryptoX509Chain();
   if (ParseFile) {
      int nci = (*ParseFile)(PXcert, chain);
      key = chain->Begin()->PKI();
      PRINT(nci <<" certificates found parsing file");
      chain->Dump();
      chain->PushBack(xCA);
      chain->Dump();
      int rorc = chain->Reorder();
      chain->Dump();
      PRINT(": form reorder: "<<rorc);
      XrdCryptoX509Chain::EX509ChainErr ecod = XrdCryptoX509Chain::kNone;
      int verc = chain->Verify(ecod);
      PRINT(": form verify: "<<verc);
   } else {
      PRINT( ": problems attaching to X509ParseFile");
      exit (1);
   }
   //
   PRINT(": Testing ExportChain ... ");
   XrdCryptoX509ExportChain_t ExportChain = gCryptoFactory->X509ExportChain();
   XrdSutBucket *chainbck = 0;
   if (ExportChain) {
      chainbck = (*ExportChain)(chain);
   } else {
      PRINT( ": problems attaching to X509ExportChain");
      exit (1);
   }
   //
   PRINT(": Testing Chain import ... ");
   XrdCryptoX509ParseBucket_t ParseBucket = gCryptoFactory->X509ParseBucket();
   // Init new chain with CA certificate 
   XrdCryptoX509Chain *CAchain = new XrdCryptoX509Chain(xCA);
   if (ParseBucket && CAchain) {
      int nci = (*ParseBucket)(chainbck, CAchain);
      PRINT(nci <<" certificates found parsing bucket");
      CAchain->Dump();
      int rorc = CAchain->Reorder();
      PRINT(": form reorder: "<<rorc);
      CAchain->Dump();
      XrdCryptoX509Chain::EX509ChainErr ecod = XrdCryptoX509Chain::kNone;
      int verc = CAchain->Verify(ecod);
      PRINT(": form verify: "<<verc);
      CAchain->PushBack(xCB);
      CAchain->Dump();
      rorc = CAchain->Reorder();
      PRINT(": form reorder: "<<rorc);
   } else {
      PRINT( ": problems creating new X509Chain" <<
                       " or attaching to X509ParseBucket");
      exit (1);
   }

   //
   PRINT(": Testing GSI chain import and verification ... ");
   // Init new GSI chain with CA certificate 
   XrdCryptosslgsiX509Chain *GSIchain = new XrdCryptosslgsiX509Chain(xCA);
   if (ParseBucket && GSIchain) {
      int nci = (*ParseBucket)(chainbck, GSIchain);
      PRINT(nci <<" certificates found parsing bucket");
      GSIchain->Dump();
      XrdCryptoX509Chain::EX509ChainErr ecod = XrdCryptoX509Chain::kNone;
      x509ChainVerifyOpt_t vopt = { kOptsRfc3820, 0, -1 };
      int verc = GSIchain->Verify(ecod, &vopt);
      PRINT(": form verify: "<<verc);
      GSIchain->Dump();
   } else {
      PRINT( ": problems creating new gsiX509Chain");
      exit (1);
   }

   //
   PRINT(": Testing GSI chain copy ... ");
   // Init new GSI chain with CA certificate 
   XrdCryptosslgsiX509Chain *GSInew = new XrdCryptosslgsiX509Chain(GSIchain);
   if (GSInew) {
      GSInew->Dump();
   } else {
      PRINT( ": problems creating new gsiX509Chain with copy");
      exit (1);
   }

   //
   PRINT(": Testing Cert verification ... ");
   XrdCryptoX509VerifyCert_t VerifyCert = gCryptoFactory->X509VerifyCert();
   if (VerifyCert) {
      bool ok = xEE->Verify(xCA);
      PRINT( ": verify cert: EE signed by CA? " <<ok);
      ok = xPX->Verify(xEE);
      PRINT( ": verify cert: PX signed by EE? " <<ok);
      ok = xPX->Verify(xCA);
      PRINT( ": verify cert: PX signed by CA? " <<ok);
   } else {
      PRINT( ": problems attaching to X509VerifyCert");
      exit (1);
   }

   //
   PRINT(": --------------------------------------------------- ");
   PRINT(": Testing ParseFile on wrong file ... ");
   XrdCryptosslgsiX509Chain *GSIwrg = new XrdCryptosslgsiX509Chain();
   if (GSIwrg) {
      int nci = (*ParseFile)(CAwrg, GSIwrg);
      PRINT(nci <<" certificates found parsing file");
      GSIwrg->Dump();
   } else {
      PRINT( ": problems creating a gsiX509Chain instance");
      exit (1);
   }

   //
   PRINT(": --------------------------------------------------- ");
   PRINT(": Testing loading of cert + key ... host");
   XrdCryptoX509 *xHO = gCryptoFactory->X509(HOcert, HOkey);
   if (xHO) {
      xHO->Dump();
   } else {
      PRINT( ": problems loading HO cert");
   }

   //
   PRINT(": --------------------------------------------------- ");
   PRINT(": Testing proxy creation ");
//   XrdProxyOpt_t *pxopt = 0;   // defaults
   XrdProxyOpt_t pxopt = {1024, 3600, 5};   // 1024 bits, valid=1h, depthlen=5 
   XrdCryptosslgsiX509Chain *cPXp = new XrdCryptosslgsiX509Chain();
   XrdCryptoRSA *kPXp = 0;
   XrdCryptoX509 *xPXp = 0;
   X509_EXTENSION *ext = 0;
   int prc = XrdSslgsiX509CreateProxy(EEcert,EEkey,&pxopt,cPXp,&kPXp,PXcrtp);
   if (prc == 0) {
      cPXp->Dump();
      xPXp = (XrdCryptoX509 *)(cPXp->Begin());
      ext = (X509_EXTENSION *)(xPXp->GetExtension("1.3.6.1.4.1.3536.1.222"));
   } else {
      PRINT( ": problems creating proxy");
      exit(1);
   }

   //
   PRINT(": --------------------------------------------------- ");
   PRINT(": Testing request creation ");
   XrdCryptoX509Req *rPXp = 0;
   XrdCryptoRSA *krPXp = 0;
   prc = XrdSslgsiX509CreateProxyReq(xPXp, &rPXp, &krPXp);
   if (prc == 0) {
      rPXp->Dump();
   } else {
      PRINT( ": problems creating request");
      exit(1);
   }

   //
   PRINT(": --------------------------------------------------- ");
   PRINT(": Testing request signature ");
   XrdCryptoX509 *xPXpp = 0;
   prc = XrdSslgsiX509SignProxyReq(xPXp, kPXp, rPXp, &xPXpp);
   if (prc == 0) {
      xPXpp->Dump();
      ext = (X509_EXTENSION *)xPXpp->GetExtension("1.3.6.1.4.1.3536.1.222");
   } else {
      PRINT( ": problems signing request");
      exit(1);
   }


   //
   // Test certificate and CRL loading
   XrdCryptoX509 *xCA1 = gCryptoFactory->X509(CAcer1);
   if (xCA1) {
      xCA1->Dump();
   } else {
      PRINT( ": problems loading CA1 cert");
   }
   XrdCryptoX509Crl *xCRL1 = gCryptoFactory->X509Crl(CRLcer1);
   if (xCRL1) {
      xCRL1->Dump();
      // Verify CRL signature
      bool crlsig = xCRL1->Verify(xCA1);
      PRINT( ": CRL signature OK? "<<crlsig);
      // Verify a serial number
      bool snrev = xCRL1->IsRevoked(25, 0);
      PRINT( ": SN: 25 revoked? "<<snrev);
      // Verify another serial number
      snrev = xCRL1->IsRevoked(0x20, 0);
      PRINT( ": SN: 32 revoked? "<<snrev);
   } else {
      PRINT( ": problems loading CA1 crl");
   }

   PRINT(": --------------------------------------------------- ");
   exit(0);
}
