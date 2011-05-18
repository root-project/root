// $Id$

const char *XrdSecgsitestCVSID = "$Id$";
//
//  Test program for XrdSecgsi
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <sys/types.h>
#include <pwd.h>

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
#include <openssl/x509.h>

//
// Globals 

// #define PRINT(x) {cerr <<x <<endl;}
XrdCryptoFactory *gCryptoFactory = 0;

XrdOucString EEcert = "";
XrdOucString EEkey = "";
XrdOucString PXcert = "";
XrdOucString PPXcert = "";
XrdOucString CAdir = "/etc/grid-security/certificates/";
int          CAnum = 0;
XrdOucString CAcert[5];

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
   // Find out the username and locate the relevant certificates and directories
   struct passwd *pw = getpwuid(geteuid());
   if (!pw) {
      PRINT(": could not resolve user info - exit");
      exit(1);
   }
   PRINT(": effective user is : "<<pw->pw_name<<", $HOME : "<<pw->pw_dir);

   //
   // User certificate
   EEcert = pw->pw_dir;
   EEcert += "/.globus/usercert.pem";
   if (getenv("X509_USER_CERT")) EEcert = getenv("X509_USER_CERT");
   PRINT(": user EE certificate: "<<EEcert);
   XrdCryptoX509 *xEE = gCryptoFactory->X509(EEcert.c_str());
   if (xEE) {
      xEE->Dump();
   } else {
      PRINT( ": problems loading user EE cert");
   }
   //
   // User key
   EEkey = pw->pw_dir;
   EEkey += "/.globus/userkey.pem";
   if (getenv("X509_USER_KEY")) EEkey = getenv("X509_USER_KEY");
   PRINT(": user EE key: "<<EEkey);
   //
   // User Proxy certificate
   PXcert = "/tmp/x509up_u";
   PXcert += (int) pw->pw_uid;
   if (getenv("X509_USER_PROXY")) PXcert = getenv("X509_USER_PROXY");
   PRINT(": user proxy certificate: "<<PXcert);
   XrdCryptoX509 *xPX = gCryptoFactory->X509(PXcert.c_str());
   if (xPX) {
      xPX->Dump();
   } else {
      PRINT( ": problems loading user proxy cert");
   }

   //
   PRINT(": --------------------------------------------------- ");
   PRINT(": recreate the proxy certificate ");
   XrdProxyOpt_t *pxopt = 0;   // defaults
   XrdCryptosslgsiX509Chain *cPXp = new XrdCryptosslgsiX509Chain();
   XrdCryptoRSA *kPXp = 0;
   XrdCryptoX509 *xPXp = 0;
   X509_EXTENSION *ext = 0;
   int prc = XrdSslgsiX509CreateProxy(EEcert.c_str(), EEkey.c_str(),
                                      pxopt, cPXp, &kPXp, PXcert.c_str());
   if (prc == 0) {
      cPXp->Dump();
      xPXp = (XrdCryptoX509 *)(cPXp->Begin());
      ext = (X509_EXTENSION *)(xPXp->GetExtension("1.3.6.1.4.1.3536.1.222"));
   } else {
      PRINT( ": problems creating proxy");
      exit(1);
   }

   //
   // Load CA certificates now
   XrdCryptoX509 *xCA[5], *xCAref = 0;
   if (getenv("X509_CERT_DIR")) CAdir = getenv("X509_CERT_DIR");
   if (!CAdir.endswith("/")) CAdir += "/";
   XrdCryptoX509 *xc = xEE;
   bool rCAfound = 0;
   int nCA = 0;
   while (!rCAfound && nCA < 5) {
      CAcert[nCA] = CAdir;
      CAcert[nCA] += xc->IssuerHash();
      PRINT(": issuer CA certificate path "<<CAcert[nCA]);
      xCA[nCA] = gCryptoFactory->X509(CAcert[nCA].c_str());
      if (xCA[nCA]) {
         xCA[nCA]->Dump();
      } else {
         PRINT( ": problems loading CA cert from : "<<CAcert[nCA]);
      }
      // Check if self-signed
      if (!strcmp(xCA[nCA]->IssuerHash(), xCA[nCA]->SubjectHash())) {
         rCAfound = 1;
         break;
      }
      // If not, parse the issuer ...
      xc = xCA[nCA];
      nCA++;
   }

   //
   PRINT(": --------------------------------------------------- ");
   PRINT(": Testing ParseFile ... ");
   XrdCryptoX509ParseFile_t ParseFile = gCryptoFactory->X509ParseFile();
   XrdCryptoRSA *key = 0;
   XrdCryptoX509Chain *chain = new XrdCryptoX509Chain();
   if (ParseFile) {
      int nci = (*ParseFile)(PXcert.c_str(), chain);
      key = chain->Begin()->PKI();
      PRINT(nci <<" certificates found parsing file");
      chain->Dump();
      int jCA = nCA + 1;
      while (jCA--) {
         chain->PushBack(xCA[jCA]);
      }
      chain->Dump();
      int rorc = chain->Reorder();
      if (rCAfound) {
         chain->Dump();
         PRINT(": form reorder: "<<rorc);
         XrdCryptoX509Chain::EX509ChainErr ecod = XrdCryptoX509Chain::kNone;
         int verc = chain->Verify(ecod);
         PRINT(": form verify: "<<verc);
      } else {
         PRINT(": full CA chain not available: verification not done ");
      }
   } else {
      PRINT( ": problems attaching to X509ParseFile");
      exit (1);
   }

   //
   PRINT(": Testing ExportChain ... ");
   XrdCryptoX509ExportChain_t ExportChain = gCryptoFactory->X509ExportChain();
   XrdSutBucket *chainbck = 0;
   if (ExportChain) {
      chainbck = (*ExportChain)(chain, 0);
   } else {
      PRINT( ": problems attaching to X509ExportChain");
      exit (1);
   }
   //
   PRINT(": Testing Chain import ... ");
   XrdCryptoX509ParseBucket_t ParseBucket = gCryptoFactory->X509ParseBucket();
   // Init new chain with CA certificate 
   int jCA = nCA;
   XrdCryptoX509Chain *CAchain = new XrdCryptoX509Chain(xCA[jCA]);
   while (jCA) { CAchain->PushBack(xCA[--jCA]); }
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
   } else {
      PRINT( ": problems creating new X509Chain" <<
                       " or attaching to X509ParseBucket");
      exit (1);
   }

   //
   PRINT(": Testing GSI chain import and verification ... ");
   // Init new GSI chain with CA certificate 
   jCA = nCA;
   XrdCryptosslgsiX509Chain *GSIchain = new XrdCryptosslgsiX509Chain(xCA[jCA]);
   while (jCA) { GSIchain->PushBack(xCA[--jCA]); }
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
      bool ok;
      jCA = nCA;
      while (jCA >= 0) {
         ok = xEE->Verify(xCA[jCA]);
         PRINT( ": verify cert: EE signed by CA? " <<ok<<" ("<<xCA[jCA]->Subject()<<")");
         if (ok) xCAref = xCA[jCA];
         jCA--;
      }
      ok = xPX->Verify(xEE);
      PRINT( ": verify cert: PX signed by EE? " <<ok);
      jCA = nCA;
      while (jCA >= 0) {
         ok = xPX->Verify(xCA[jCA]);
         PRINT( ": verify cert: PX signed by CA? " <<ok<<" ("<<xCA[jCA]->Subject()<<")");
         jCA--;
      }
   } else {
      PRINT( ": problems attaching to X509VerifyCert");
      exit (1);
   }


   //
   PRINT(": --------------------------------------------------- ");
   PRINT(": Testing request creation ");
   XrdCryptoX509Req *rPXp = 0;
   XrdCryptoRSA *krPXp = 0;
   prc = XrdSslgsiX509CreateProxyReq(xPX, &rPXp, &krPXp);
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
   prc = XrdSslgsiX509SignProxyReq(xPX, kPXp, rPXp, &xPXpp);
   if (prc == 0) {
      xPXpp->Dump();
      xPXpp->SetPKI((XrdCryptoX509data) krPXp->Opaque());
      ext = (X509_EXTENSION *)xPXpp->GetExtension("1.3.6.1.4.1.3536.1.222");
   } else {
      PRINT( ": problems signing request");
      exit(1);
   }

   //
   PRINT(": --------------------------------------------------- ");
   PRINT(": Testing export of signed proxy ");
   PPXcert = PXcert;
   PPXcert += "p";
   PRINT(": file for signed proxy chain: "<<PPXcert);
   XrdCryptoX509ChainToFile_t ChainToFile = gCryptoFactory->X509ChainToFile();
   // Init the proxy chain 
   XrdCryptoX509Chain *PXchain = new XrdCryptoX509Chain(xPXpp);
   PXchain->PushBack(xPX);
   PXchain->PushBack(xEE);
   if (ChainToFile && PXchain) {
      if ((*ChainToFile)(PXchain, PPXcert.c_str()) != 0) {
         PRINT(": problems saving signed proxy chain to file: "<<PPXcert);
      }
   } else {
      PRINT( ": problems creating new X509Chain" <<
                       " or attaching to X509ParseBucket");
      exit (1);
   }

   //
   PRINT(": --------------------------------------------------- ");
   PRINT(": Testing CRL identification ");
   X509_EXTENSION *crlext = 0;
   if (xCAref) {
      if ((crlext = (X509_EXTENSION *)xCAref->GetExtension("crlDistributionPoints"))) {
         PRINT( ": CRL distribution points extension OK");
      } else {
         PRINT( ": problems getting extension");
      }
   }

   //
   PRINT(": --------------------------------------------------- ");
   PRINT(": Testing CRL loading ");
   XrdCryptoX509Crl *xCRL1 = gCryptoFactory->X509Crl(xCAref);
   if (xCRL1) {
      xCRL1->Dump();
      // Verify CRL signature
      bool crlsig = 0;
      for (jCA = 0; jCA <= nCA; jCA++) {
         crlsig = xCRL1->Verify(xCA[jCA]);
         PRINT( ": CRL signature OK? "<<crlsig<<" ("<<xCA[jCA]->Subject()<<")");
      }
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
