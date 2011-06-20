/******************************************************************************/
/*                                                                            */
/*                 X r d S e c P r o t o c o l g s i . c c                    */
/*                                                                            */
/* (c) 2005 G. Ganis / CERN                                                   */
/*                                                                            */
/******************************************************************************/

#include <unistd.h>
#include <ctype.h>
#include <errno.h>
#include <stdlib.h>
#include <strings.h>
#include <stdio.h>
#include <sys/param.h>
#include <pwd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <dirent.h>

#include "XrdNet/XrdNetDNS.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include <XrdSys/XrdSysLogger.hh>
#include <XrdSys/XrdSysError.hh>
#include "XrdSys/XrdSysPlugin.hh"
#include "XrdSys/XrdSysPriv.hh"
#include <XrdOuc/XrdOucStream.hh>

#include <XrdSut/XrdSutCache.hh>

#include <XrdCrypto/XrdCryptoMsgDigest.hh>
#include <XrdCrypto/XrdCryptosslAux.hh>
#include <XrdCrypto/XrdCryptosslgsiAux.hh>

#include <XrdSecgsi/XrdSecProtocolgsi.hh>
#include <XrdSecgsi/XrdSecgsiTrace.hh>

/******************************************************************************/
/*                           S t a t i c   D a t a                            */
/******************************************************************************/
  
static String Prefix   = "xrd";
static String ProtoID  = XrdSecPROTOIDENT;
static const kXR_int32 Version = XrdSecgsiVERSION;

static const char *gsiClientSteps[] = {
   "kXGC_none",
   "kXGC_certreq",
   "kXGC_cert",
   "kXGC_reserved"
};

static const char *gsiServerSteps[] = {
   "kXGS_none",
   "kXGS_init",
   "kXGS_cert",
   "kXGS_reserved"
};

static const char *gGSErrStr[] = {
   "ErrParseBuffer",               // 10000
   "ErrDecodeBuffer",              // 10001
   "ErrLoadCrypto",                // 10002
   "ErrBadProtocol",               // 10003
   "ErrCreateBucket",              // 10004
   "ErrDuplicateBucket",           // 10005
   "ErrCreateBuffer",              // 10006
   "ErrSerialBuffer",              // 10007
   "ErrGenCipher",                 // 10008
   "ErrExportPuK",                 // 10009
   "ErrEncRndmTag",                // 10010
   "ErrBadRndmTag",                // 10011
   "ErrNoRndmTag",                 // 10012
   "ErrNoCipher",                  // 10013
   "ErrNoCreds",                   // 10014
   "ErrBadOpt",                    // 10015
   "ErrMarshal",                   // 10016
   "ErrUnmarshal",                 // 10017
   "ErrSaveCreds",                 // 10018
   "ErrNoBuffer",                  // 10019
   "ErrRefCipher",                 // 10020
   "ErrNoPublic",                  // 10021
   "ErrAddBucket",                 // 10022
   "ErrFinCipher",                 // 10023
   "ErrInit",                      // 10024
   "ErrBadCreds",                  // 10025
   "ErrError"                      // 10026  
};

// One day in secs
static const int kOneDay = 86400; 
static const char *gUsrPxyDef = "/tmp/x509up_u";

/******************************************************************************/
/*                     S t a t i c   C l a s s   D a t a                      */
/******************************************************************************/

XrdSysMutex XrdSecProtocolgsi::gsiContext;
String XrdSecProtocolgsi::CAdir    = "/etc/grid-security/certificates/";
String XrdSecProtocolgsi::CRLdir   = "/etc/grid-security/certificates/";
String XrdSecProtocolgsi::DefCRLext= ".r0";
String XrdSecProtocolgsi::GMAPFile = "/etc/grid-security/grid-mapfile";
String XrdSecProtocolgsi::SrvCert  = "/etc/grid-security/xrd/xrdcert.pem";
String XrdSecProtocolgsi::SrvKey   = "/etc/grid-security/xrd/xrdkey.pem";
String XrdSecProtocolgsi::UsrProxy;
String XrdSecProtocolgsi::UsrCert  = "/.globus/usercert.pem";
String XrdSecProtocolgsi::UsrKey   = "/.globus/userkey.pem";;
String XrdSecProtocolgsi::PxyValid = "12:00";
int    XrdSecProtocolgsi::DepLength= 0;
int    XrdSecProtocolgsi::DefBits  = 512;
int    XrdSecProtocolgsi::CACheck  = 1;
int    XrdSecProtocolgsi::CRLCheck = 1;
int    XrdSecProtocolgsi::GMAPOpt  = 1;
bool   XrdSecProtocolgsi::GMAPuseDNname = 0;
String XrdSecProtocolgsi::DefCrypto= "ssl";
String XrdSecProtocolgsi::DefCipher= "aes-128-cbc:bf-cbc:des-ede3-cbc";
String XrdSecProtocolgsi::DefMD    = "sha1:md5";
String XrdSecProtocolgsi::DefError = "invalid credentials ";
int    XrdSecProtocolgsi::PxyReqOpts = 0;
int    XrdSecProtocolgsi::AuthzPxyWhat = -1;
int    XrdSecProtocolgsi::AuthzPxyWhere = -1;
XrdSysPlugin   *XrdSecProtocolgsi::GMAPPlugin = 0;
XrdSecgsiGMAP_t XrdSecProtocolgsi::GMAPFun = 0;
XrdSysPlugin   *XrdSecProtocolgsi::AuthzPlugin = 0;
XrdSecgsiAuthz_t XrdSecProtocolgsi::AuthzFun = 0;
int    XrdSecProtocolgsi::GMAPCacheTimeOut = -1;
String XrdSecProtocolgsi::SrvAllowedNames;
//
// Crypto related info
int  XrdSecProtocolgsi::ncrypt    = 0;                 // Number of factories
XrdCryptoFactory *XrdSecProtocolgsi::cryptF[XrdCryptoMax] = {0};   // their hooks 
int  XrdSecProtocolgsi::cryptID[XrdCryptoMax] = {0};   // their IDs 
String XrdSecProtocolgsi::cryptName[XrdCryptoMax] = {0}; // their names 
XrdCryptoCipher *XrdSecProtocolgsi::refcip[XrdCryptoMax] = {0};    // ref for session ciphers 
//
// Caches
XrdSutCache XrdSecProtocolgsi::cacheCA;   // CA info
XrdSutCache XrdSecProtocolgsi::cacheCert; // Server certificates info
XrdSutCache XrdSecProtocolgsi::cachePxy;  // Client proxies
XrdSutCache XrdSecProtocolgsi::cacheGMAP; // Grid map entries
XrdSutCache XrdSecProtocolgsi::cacheGMAPFun; // Entries mapped by GMAPFun
//
// Running options / settings
int  XrdSecProtocolgsi::Debug       = 0; // [CS] Debug level
bool XrdSecProtocolgsi::Server      = 1; // [CS] If server mode 
int  XrdSecProtocolgsi::TimeSkew    = 300; // [CS] Allowed skew in secs for time stamps 
//
// Debug an tracing
XrdSysError    XrdSecProtocolgsi::eDest(0, "secgsi_");
XrdSysLogger   XrdSecProtocolgsi::Logger;
XrdOucTrace   *XrdSecProtocolgsi::GSITrace = 0;

XrdOucTrace *gsiTrace = 0;

/******************************************************************************/
/*                    S t a t i c   F u n c t i o n s                         */
/******************************************************************************/
//_____________________________________________________________________________
static const char *ClientStepStr(int kclt)
{
   // Return string with client step  
   static const char *ukn = "Unknown";

   kclt = (kclt < 0) ? 0 : kclt;
   kclt = (kclt > kXGC_reserved) ? 0 : kclt;
   kclt = (kclt >= kXGC_certreq) ? (kclt - kXGC_certreq + 1) : kclt;

   if (kclt < 0 || kclt > (kXGC_reserved - kXGC_certreq + 1))
      return ukn;  
   else
      return gsiClientSteps[kclt];
}

//_____________________________________________________________________________
static const char *ServerStepStr(int ksrv)
{
   // Return string with server step  
   static const char *ukn = "Unknown";

   ksrv = (ksrv < 0) ? 0 : ksrv;
   ksrv = (ksrv > kXGS_reserved) ? 0 : ksrv;
   ksrv = (ksrv >= kXGS_init) ? (ksrv - kXGS_init + 1) : ksrv;

   if (ksrv < 0 || ksrv > (kXGS_reserved - kXGS_init + 1))
      return ukn;  
   else
      return gsiServerSteps[ksrv];
}


/******************************************************************************/
/*       D u m p  o f   H a n d s h a k e   v a r i a b l e s                 */
/******************************************************************************/

//_____________________________________________________________________________
void gsiHSVars::Dump(XrdSecProtocolgsi *p)
{
   // Dump content
   EPNAME("HSVars::Dump");

   PRINT("----------------------------------------------------------------");
   PRINT("protocol instance:   "<<p);
   PRINT("this:                "<<this);
   PRINT(" ");
   PRINT("Time stamp:          "<<TimeStamp);
   PRINT("Crypto mod:          "<<CryptoMod);
   PRINT("Remote version:      "<<RemVers);
   PRINT("Ref cipher:          "<<Rcip);
   PRINT("Bucket for exp cert: "<<Cbck);
   PRINT("Handshake ID:        "<<ID);
   PRINT("Cache reference:     "<<Cref);
   PRINT("Relevant file entry: "<<Pent);
   PRINT("Chain pointer:       "<<Chain);
   PRINT("CRL pointer:         "<<Crl);
   PRINT("Proxy chain:         "<<PxyChain);
   PRINT("Rndm tag checked:    "<<RtagOK);
   PRINT("Last step:           "<<LastStep);
   PRINT("Options:             "<<Options);
   PRINT("----------------------------------------------------------------");
}

/******************************************************************************/
/*       P r o t o c o l   I n i t i a l i z a t i o n   M e t h o d s        */
/******************************************************************************/


//_____________________________________________________________________________
XrdSecProtocolgsi::XrdSecProtocolgsi(int opts, const char *hname,
                                     const struct sockaddr *ipadd,
                                     const char *parms) : XrdSecProtocol("gsi")
{
   // Default constructor
   EPNAME("XrdSecProtocolgsi");

   if (QTRACE(Authen)) { PRINT("constructing: "<<this); }

   // Create instance of the handshake vars
   if ((hs = new gsiHSVars())) {
      // Update time stamp
      hs->TimeStamp = time(0);
      // Local handshake variables
      hs->Tty = (isatty(0) == 0 || isatty(1) == 0) ? 0 : 1;
   } else {
      DEBUG("could not create handshake vars object");
   }

   // Set host name
   if (ipadd) {
      Entity.host = XrdNetDNS::getHostName((sockaddr&)*ipadd);
      // Set host addr
      memcpy(&hostaddr, ipadd, sizeof(hostaddr));
   } else {
      PRINT("WARNING: IP addr undefined: cannot determine host name: failure may follow");
   }

   // Init session variables
   sessionCF = 0;
   sessionKey = 0;
   bucketKey = 0;
   sessionMD = 0;
   sessionKsig = 0;
   sessionKver = 0;
   sessionKver = 0;
   proxyChain = 0;

   //
   // Notify, if required
   DEBUG("constructing: host: "<<hname);
   DEBUG("p: "<<XrdSecPROTOIDENT<<", plen: "<<XrdSecPROTOIDLEN);
   //
   // basic settings
   options  = opts;
   srvMode = 0;

   //
   // Mode specific initializations
   if (Server) {
      srvMode = 1;
      DEBUG("mode: server");
   } else {
      DEBUG("mode: client");
      //
      // Decode received buffer
      if (parms) {
         XrdOucString p("&P=gsi,");
         p += parms;
         hs->Parms = new XrdSutBuffer(p.c_str(), p.length());
      }
   }

   // We are done
   String vers = Version;
   vers.insert('.',vers.length()-2);
   vers.insert('.',vers.length()-5);
   DEBUG("object created: v"<<vers.c_str());
}

//_____________________________________________________________________________
char *XrdSecProtocolgsi::Init(gsiOptions opt, XrdOucErrInfo *erp)
{
   // Static method to the configure the static part of the protocol
   // Called once by XrdSecProtocolgsiInit
   EPNAME("Init");
   char *Parms = 0;

   //
   // Time stamp of initialization
   int timestamp = (int)time(0);
   //
   // Debug an tracing
   Debug = (opt.debug > -1) ? opt.debug : Debug;
   // Initiate error logging and tracing
   eDest.logger(&Logger);
   GSITrace    = new XrdOucTrace(&eDest);
   // Set debug mask ... also for auxilliary libs
   int trace = 0;
   if (Debug >= 3) {
      trace = cryptoTRACE_Dump;
      GSITrace->What |= TRACE_Authen;
      GSITrace->What |= TRACE_Debug;
   } else if (Debug >= 1) {
      trace = cryptoTRACE_Debug;
      GSITrace->What = TRACE_Debug;
   }
   gsiTrace = GSITrace;
   // ... also for auxilliary libs
   XrdSutSetTrace(trace);
   XrdCryptoSetTrace(trace);

   //
   // Operation mode
   Server = (opt.mode == 's');

   //
   // CA verification level
   //
   //    0   do not verify
   //    1   verify if self-signed; warn if not
   //    2   verify in all cases; fail if not possible
   //
   if (opt.ca >= 0 && opt.ca <= 2)
      CACheck = opt.ca;
   DEBUG("option CACheck: "<<CACheck);

   //
   // Check existence of CA directory
   struct stat st;
   if (opt.certdir) {
      DEBUG("testing CA dir(s): "<<opt.certdir);
      String CAtmp;
      String tmp = opt.certdir;
      String dp;
      int from = 0;
      while ((from = tmp.tokenize(dp, from, ',')) != -1) {
         if (dp.length() > 0) {
            if (XrdSutExpand(dp) == 0) {
               if (stat(dp.c_str(),&st) == -1) {
                  if (errno == ENOENT) {
                     ErrF(erp,kGSErrError,"CA directory non existing:",dp.c_str());
                     PRINT(erp->getErrText());
                  } else {
                     ErrF(erp,kGSErrError,"cannot stat CA directory:",dp.c_str());
                     PRINT(erp->getErrText());
                  }
               } else {
                  if (!(dp.endswith('/'))) dp += '/';
                  if (!(CAtmp.endswith(','))) CAtmp += ',';
                  CAtmp += dp;
               }
            } else {
               PRINT("Warning: could not expand: "<<dp);
            }
         }
      }
      if (CAtmp.length() > 0)
         CAdir = CAtmp;
   }
   DEBUG("using CA dir(s): "<<CAdir);

   //
   // CRL check level
   //
   //    0   do not care
   //    1   use if available
   //    2   require
   //    3   require not expired
   //
   if (opt.crl >= 0 && opt.crl <= 3)
      CRLCheck = opt.crl;
   DEBUG("option CRLCheck: "<<CRLCheck);

   //
   // Check existence of CRL directory
   if (opt.crldir) {

      DEBUG("testing CRL dir(s): "<<opt.crldir);
      String CRLtmp;
      String tmp = opt.crldir;
      String dp;
      int from = 0;
      while ((from = tmp.tokenize(dp, from, ',')) != -1) {
         if (dp.length() > 0) {
            if (XrdSutExpand(dp) == 0) {
               if (stat(dp.c_str(),&st) == -1) {
                  if (errno == ENOENT) {
                     ErrF(erp,kGSErrError,"CRL directory non existing:",dp.c_str());
                     PRINT(erp->getErrText());
                  } else {
                     ErrF(erp,kGSErrError,"cannot stat CRL directory:",dp.c_str());
                     PRINT(erp->getErrText());
                  }
               } else {
                  if (!(dp.endswith('/'))) dp += '/';
                  if (!(CRLtmp.endswith(','))) CRLtmp += ',';
                  CRLtmp += dp;
               }
            } else {
               PRINT("Warning: could not expand: "<<dp);
            }
         }
      }
      if (CRLtmp.length() > 0)
         CRLdir = CRLtmp;

   } else {
      // Use CAdir
      CRLdir = CAdir;
   }
   if (CRLCheck > 0)
      DEBUG("using CRL dir(s): "<<CRLdir);

   //
   // Default extension for CRL files
   if (opt.crlext)
      DefCRLext = opt.crlext;

   //
   // Server specific options
   if (Server) {
      //
      // List of supported / wanted crypto modules
      if (opt.clist)
         DefCrypto = opt.clist;
      //
      // List of crypto modules
      String cryptlist(DefCrypto,0,-1,64);
      // 
      // Load crypto modules
      XrdSutPFEntry ent;
      XrdCryptoFactory *cf = 0;
      if (cryptlist.length()) {
         String ncpt = "";
         int from = 0;
         while ((from = cryptlist.tokenize(ncpt, from, '|')) != -1) {
            if (ncpt.length() > 0 && ncpt[0] != '-') {
               // Try loading 
               if ((cf = XrdCryptoFactory::GetCryptoFactory(ncpt.c_str()))) {
                  // Add it to the list
                  cryptF[ncrypt] = cf;
                  cryptID[ncrypt] = cf->ID();
                  cryptName[ncrypt].insert(cf->Name(),0,strlen(cf->Name())+1);
                  cf->SetTrace(trace);
                  // Ref cipher
                  if (!(refcip[ncrypt] = cf->Cipher(0,0,0))) {
                     PRINT("ref cipher for module "<<ncpt<<
                           " cannot be instantiated : disable");
                     from -= ncpt.length();
                     cryptlist.erase(ncpt);
                  } else {
                     ncrypt++;
                     if (ncrypt >= XrdCryptoMax) {
                        PRINT("max number of crypto modules ("
                              << XrdCryptoMax <<") reached ");
                        break;
                     }
                  }
               } else {
                  PRINT("cannot instantiate crypto factory "<<ncpt<<
                           ": disable");
                  from -= ncpt.length();
                  cryptlist.erase(ncpt);
               }
            }
         }
      }
      //
      // We need at least one valid crypto module
      if (ncrypt <= 0) {
         ErrF(erp,kGSErrInit,"could not find any valid crypto module");
         PRINT(erp->getErrText());
         return Parms;
      }
      //
      // Load CA info into cache
      if (LoadCADir(timestamp) != 0) {
         ErrF(erp,kGSErrError,"problems loading CA info into cache");
         PRINT(erp->getErrText());
         return Parms;
      }
      if (QTRACE(Authen)) { cacheCA.Dump(); }

      //
      // List of supported / wanted ciphers
      if (opt.cipher)
         DefCipher = opt.cipher;
      // make sure we support all of them
      String cip = "";
      int from = 0;
      while ((from = DefCipher.tokenize(cip, from, ':')) != -1) {
         if (cip.length() > 0) {
            int i = 0;
            for (; i < ncrypt; i++) {
               if (!(cryptF[i]->SupportedCipher(cip.c_str()))) {
                  // Not supported: drop from the list
                  DEBUG("cipher type not supported ("<<cip<<") - disabling");
                  from -= cip.length();
                  DefCipher.erase(cip);
               }
            }
         }
      }

      //
      // List of supported / wanted Message Digest
      if (opt.md)
         DefMD = opt.md;
      // make sure we support all of them
      String md = "";
      from = 0;
      while ((from = DefMD.tokenize(md, from, ':')) != -1) {
         if (md.length() > 0) {
            int i = 0;
            for (; i < ncrypt; i++) {
               if (!(cryptF[i]->SupportedMsgDigest(md.c_str()))) {
                  // Not supported: drop from the list
                  PRINT("MD type not supported ("<<md<<") - disabling");
                  from -= md.length();
                  DefMD.erase(md);
               }
            }
         }
      }

      //
      // Load server certificate and key
      if (opt.cert) {
         String TmpCert = opt.cert;
         if (XrdSutExpand(TmpCert) == 0) {
            SrvCert = TmpCert;
         } else {
            PRINT("Could not expand: "<<opt.cert<<": use default");
         }
      }
      if (opt.key) {
         String TmpKey = opt.key;
         if (XrdSutExpand(TmpKey) == 0) {
            SrvKey = TmpKey;
         } else {
            PRINT("Could not expand: "<<opt.key<<": use default");
         }
      }
      //
      // Check if we can read the certificate key
      if (access(SrvKey.c_str(), R_OK)) {
         PRINT("WARNING: process has no permission to read the certificate key file: "<<SrvKey);
      }
      //
      // Get the IDs of the file: we need them to acquire the right privileges when opening
      // the certificate
      bool getpriv = 0;
      uid_t gsi_uid = geteuid();
      gid_t gsi_gid = getegid();
      if (!stat(SrvKey.c_str(), &st)) {
         if (st.st_uid != gsi_uid || st.st_gid != gsi_gid) {
            getpriv = 1;
            gsi_uid = st.st_uid;
            gsi_gid = st.st_gid;
         }
      }
      //
      // Init cache for certificates (we allow for more instances of
      // the same certificate, one for each different crypto module
      // available; this may evetually not be strictly needed.)
      if (cacheCert.Init(10) != 0) {
         ErrF(erp,kGSErrError,"problems init cache for certificates");
         PRINT(erp->getErrText());
         return Parms;
      }
      int i = 0;
      String certcalist = "";   // list of CA for server certificates
      for (; i<ncrypt; i++) {
         // Check normal certificates
         XrdCryptoX509 *xsrv = 0;
         {  XrdSysPrivGuard pGuard(gsi_uid, gsi_gid);
            if (pGuard.Valid())
               xsrv = cryptF[i]->X509(SrvCert.c_str(), SrvKey.c_str());
         }
         if (xsrv) {
            // Must be of EEC type
            if (xsrv->type != XrdCryptoX509::kEEC) {
               PRINT("problems loading srv cert: not EEC but: "<<xsrv->Type());
               continue;
            }
            // Must be valid
            if (!(xsrv->IsValid())) {
               PRINT("problems loading srv cert: invalid");
               continue;
            }
            // PKI must have been successfully initialized
            if (!xsrv->PKI() || xsrv->PKI()->status != XrdCryptoRSA::kComplete) {
               PRINT("problems loading srv cert: invalid PKI");
               continue;
            }
            // Must be exportable
            XrdSutBucket *xbck = xsrv->Export();
            if (!xbck) {
               PRINT("problems loading srv cert: cannot export into bucket");
               continue;
            }
            // Ok: save it into the cache
            String tag = cryptF[i]->Name();
            XrdSutPFEntry *cent = cacheCert.Add(tag.c_str());
            if (cent) {
               cent->status = kPFE_ok;
               cent->cnt = 0;
               cent->mtime = xsrv->NotAfter(); // expiration time
               // Save pointer to certificate
               cent->buf1.buf = (char *)xsrv;
               cent->buf1.len = 0;  // just a flag
               // Save pointer to key
               cent->buf2.buf = (char *)(xsrv->PKI());
               cent->buf2.len = 0;  // just a flag
               // Save pointer to bucket
               cent->buf3.buf = (char *)(xbck);
               cent->buf3.len = 0;  // just a flag
               // Save CA hash in list to communicate to clients
               if (certcalist.find(xsrv->IssuerHash()) == STR_NPOS) {
                  if (certcalist.length() > 0)
                     certcalist += "|";
                  certcalist += xsrv->IssuerHash();
               }
            }
         }
      }
      // Rehash cache
      cacheCert.Rehash(1);
      //
      // We must have got at least one valid certificate
      if (cacheCert.Empty()) {
         ErrF(erp,kGSErrError,"no valid server certificate found");
         PRINT(erp->getErrText());
         return Parms;
      }
      if (QTRACE(Authen)) { cacheCert.Dump(); }

      DEBUG("CA list: "<<certcalist);

      //
      // GRID map check option
      //
      //    0   do not use (DN hash will be used as identifier)
      //    1   use if available; otherwise as 0
      //    2   require
      //   10   do not use (DN name will be used as identifier)
      //   11   use if available; otherwise as 10
      const char *cogmap[] = { "do-not-use", "use-if-available", "require" };
      const char *codnnm[] = { "DN hash", "DN name"};
      if (opt.ogmap >= 10) {
         GMAPuseDNname = 1;
         opt.ogmap %= 10;
      }
      if (opt.ogmap >= 0 && opt.ogmap <= 2)
         GMAPOpt = opt.ogmap;
      DEBUG("user mapping file option: "<<cogmap[GMAPOpt]);
      if (GMAPOpt < 2)
         DEBUG("default option for entity name if no mapping available: "<<codnnm[(int)GMAPuseDNname]);

      //
      // Check existence of GRID map file
      if (opt.gridmap) {
         String GMAPTmp = opt.gridmap;
         if (XrdSutExpand(GMAPTmp) == 0) {
            GMAPFile = GMAPTmp;
         } else {
            PRINT("Could not expand: "<<opt.gridmap<<": use default");
         }
      }
      bool hasgmap = 0;
      if (GMAPOpt > 0) {
         if (access(GMAPFile.c_str(),R_OK) != 0) {
            if (GMAPOpt > 1) {
               if (errno == ENOENT) {
                  ErrF(erp,kGSErrError,"Grid map file non existing:",GMAPFile.c_str());
                  PRINT(erp->getErrText());
               } else {
                  ErrF(erp,kGSErrError,"'access' error on grid map file:",GMAPFile.c_str());
                  PRINT(erp->getErrText());
                  return Parms;
               }
            } else {
               DEBUG("Grid map file: "<<GMAPFile<<" cannot be 'access'ed: do not use");
            }
         } else {
            DEBUG("using grid map file: "<<GMAPFile);
            //
            // Init cache for gridmap entries
            if (LoadGMAP(timestamp) != 0) {
               ErrF(erp,kGSErrError,"problems initializing cache for gridmap entries");
               PRINT(erp->getErrText());
               return Parms;
            }
            if (QTRACE(Authen)) { cacheGMAP.Dump(); }
            hasgmap = 1;
         }
      }
      //
      // Load function be used to map DN to usernames, if specified
      bool hasauthzfun = 0;
      if (opt.authzfun && GMAPOpt > 0) {
         if (!(AuthzFun = LoadAuthzFun((const char *) opt.authzfun,
                                       (const char *) opt.authzfunparms))) {
            PRINT("Could not load plug-in: "<<opt.authzfun<<": ignore");
         } else {
            hasauthzfun = 1;
         }
      }
      bool hasgmapfun = 0;
      if (opt.gmapfun && GMAPOpt > 0) {
         if (!hasauthzfun) {
            if (!(GMAPFun = LoadGMAPFun((const char *) opt.gmapfun,
                                       (const char *) opt.gmapfunparms))) {
               PRINT("Could not load plug-in: "<<opt.gmapfun<<": ignore");
            } else {
               hasgmapfun = 1;
            }
         } else {
            PRINT("WARNING: ignoring 'gmapfun' directive since an authz plugin (specified"
                  " via 'authzfun') has already been successfully loaded");
         }
      }

      if (hasgmapfun || hasauthzfun) {
         // Init or reset the cache
         if (cacheGMAPFun.Empty()) {
            if (cacheGMAPFun.Init(100) != 0) {
               PRINT("Error initializing GMAPFun cache");
               return Parms;
            }
         } else {
            if (cacheGMAPFun.Reset() != 0) {
               PRINT("Error resetting GMAPFun cache");
               return Parms;
            }
         }
      }
      //
      // Disable GMAP if neither a grid mapfile nor a GMAP function are available
      if (!hasgmap && !hasgmapfun && !hasauthzfun) {
         if (GMAPOpt > 1) {
            ErrF(erp,kGSErrError,"User mapping required, but neither a grid mapfile"
                                 " nor a mapping function are available");
            PRINT(erp->getErrText());
            return Parms;
         }
         GMAPOpt = 0;
      }
      //
      // Expiration of GRIDMAP related cache entries
      if (GMAPOpt > 0 && !hasauthzfun && opt.gmapto > 0) {
         GMAPCacheTimeOut = opt.gmapto;
         DEBUG("grid-map cache entries expire after "<<GMAPCacheTimeOut<<" secs");
      }

      //
      // Request for delegated proxies
      if (opt.dlgpxy == 1 || opt.dlgpxy == 3)
         PxyReqOpts |= kOptsSrvReq;
      if (opt.dlgpxy == 2 || opt.dlgpxy == 3)
         PxyReqOpts |= kOptsPxFile;
      // Some notification
      DEBUG("Delegated proxies options: "<<PxyReqOpts);

      //
      // Request for proxy export for authorization
      // authzpxy = opt_what*10 + opt_where
      //        opt_what   = 0  full chain
      //                     1  last proxy only
      //        opt_where  = 1  Entity.creds
      //                     2  Entity.endorsements
      if (opt.authzpxy > 0) {
         AuthzPxyWhat = opt.authzpxy / 10;
         AuthzPxyWhere = opt.authzpxy % 10;
         // Some notification
         const char *capxy_what = (AuthzPxyWhat == 1) ? "'last proxy only'" : "'full proxy chain'";
         const char *capxy_where = (AuthzPxyWhere == 1) ? "XrdSecEntity.creds" : "XrdSecEntity.endorsements";
         DEBUG("Export proxy for authorization in '"<<capxy_where<<"': "<<capxy_what);
      }

      //
      // Template for the created proxy files
      if ((PxyReqOpts & kOptsPxFile)) {
         String TmpProxy = gUsrPxyDef;
         if (opt.exppxy) TmpProxy = opt.exppxy;
         if (XrdSutExpand(TmpProxy) == 0) {
            UsrProxy = TmpProxy;
         } else {
            UsrProxy = gUsrPxyDef;
            UsrProxy += "u<uid>";
         }
         DEBUG("Template for exported proxy files: "<<UsrProxy);
      }

      //
      // Parms in the form:
      //     &P=gsi,v:<version>,c:<cryptomod>,ca:<list_of_srv_cert_ca>
      Parms = new char[cryptlist.length()+3+12+certcalist.length()+5];
      if (Parms) {
         sprintf(Parms,"v:%d,c:%s,ca:%s",
                       Version,cryptlist.c_str(),certcalist.c_str());
      } else {
         ErrF(erp,kGSErrInit,"no system resources for 'Parms'");
         PRINT(erp->getErrText());
      }

      // Some notification
      DEBUG("available crypto modules: "<<cryptlist);
      DEBUG("issuer CAs of server certs (hashes): "<<certcalist);
   }

   //
   // Client specific options
   if (!Server) {
      //
      // Init cache for CA certificate info. Clients will fill it
      // upon need
      if (cacheCA.Init(100) != 0) {
         ErrF(erp,kGSErrError,"problems init cache for CA info");
         PRINT(erp->getErrText());
         return Parms;
      }
      //
      // Init cache for proxies (in the future we may allow
      // users to use more certificates, depending on the context)
      if (cachePxy.Init(2) != 0) {
         ErrF(erp,kGSErrError,"problems init cache for proxies");
         PRINT(erp->getErrText());
         return Parms;
      }
      // use default dir $(HOME)/.<prefix>
      struct passwd *pw = getpwuid(getuid());
      if (!pw) {
         DEBUG("WARNING: cannot get user information (uid:"<<getuid()<<")");
      }
      //
      // Define user proxy file
      UsrProxy = gUsrPxyDef;
      if (opt.proxy) {
         String TmpProxy = opt.proxy;
         if (XrdSutExpand(TmpProxy) == 0) {
            UsrProxy = TmpProxy;
         } else {
            PRINT("Could not expand: "<<opt.proxy<<": use default");
         }
      } else {
         if (pw)
            UsrProxy += (int)(pw->pw_uid);
      }
      // Define user certificate file
      if (opt.cert) {
         String TmpCert = opt.cert;
         if (XrdSutExpand(TmpCert) == 0) {
            UsrCert = TmpCert;
         } else {
            PRINT("Could not expand: "<<opt.cert<<": use default");
         }
      } else {
         if (pw)
            UsrCert.insert(XrdSutHome(),0);
      }
      // Define user private key file
      if (opt.key) {
         String TmpKey = opt.key;
         if (XrdSutExpand(TmpKey) == 0) {
            UsrKey = TmpKey;
         } else {
            PRINT("Could not expand: "<<opt.key<<": use default");
         }
      } else {
         if (pw)
            UsrKey.insert(XrdSutHome(),0);
      }
      // Define proxy validity at renewal
      if (opt.valid)
         PxyValid = opt.valid;
      // Set depth of signature path
      if (opt.deplen != DepLength)
         DepLength = opt.deplen;
      // Set number of bits for proxy key
      if (opt.bits > DefBits)
         DefBits = opt.bits;
      //
      // Delegate proxy options
      if (opt.dlgpxy == 1)
         PxyReqOpts |= kOptsDlgPxy;
      if (opt.dlgpxy == 2)
         PxyReqOpts |= kOptsFwdPxy;
      if (opt.sigpxy > 0 || opt.dlgpxy == 1)
         PxyReqOpts |= kOptsSigReq;
      //
      // Define valid CNs for the server certificates; default is null, which means that
      // the server CN must be in the form "*/<hostname>"
      if (opt.srvnames)
         SrvAllowedNames = opt.srvnames;
      //
      // Notify
      DEBUG("using certificate file:         "<<UsrCert);
      DEBUG("using private key file:         "<<UsrKey);
      DEBUG("proxy: file:                    "<<UsrProxy);
      DEBUG("proxy: validity:                "<<PxyValid);
      DEBUG("proxy: depth of signature path: "<<DepLength);
      DEBUG("proxy: bits in key:             "<<DefBits);
      DEBUG("server cert: allowed names:     "<<SrvAllowedNames);

      // We are done
      Parms = (char *)"";
   }

   // We are done
   return Parms;
}

/******************************************************************************/
/*                                D e l e t e                                 */
/******************************************************************************/
void XrdSecProtocolgsi::Delete()
{
   // Deletes the protocol
   SafeFree(Entity.name);
   SafeFree(Entity.host);
   SafeFree(Entity.vorg);
   SafeFree(Entity.role);
   SafeFree(Entity.grps);
   SafeFree(Entity.endorsements);
   SafeFree(Entity.creds);
   Entity.credslen = 0;
   // Cleanup the handshake variables, if still there
   SafeDelete(hs);
   // Cleanup any other instance specific to this protocol
   SafeDelete(sessionKey);    // Session Key (result of the handshake)
   SafeDelete(bucketKey);     // Bucket with the key in export form
   SafeDelete(sessionMD);     // Message Digest instance
   SafeDelete(sessionKsig);   // RSA key to sign
   SafeDelete(sessionKver);   // RSA key to verify
   SafeDelete(proxyChain);    // Chain with delegated proxies

   delete this;
}


/******************************************************************************/
/*       E n c r y p t i o n  R e l a t e d   M e t h o d s                   */
/******************************************************************************/

//_____________________________________________________________________________
int XrdSecProtocolgsi::Encrypt(const char *inbuf,  // Data to be encrypted
                               int inlen,          // Length of data in inbuff
                               XrdSecBuffer **outbuf)  // Returns encrypted data
{
   // Encrypt data in inbuff and place it in outbuff.
   //
   // Returns: < 0 Failed, the return value is -errno of the reason. Typically,
   //              -EINVAL    - one or more arguments are invalid.
   //              -ENOTSUP   - encryption not supported by the protocol
   //              -EOVERFLOW - outbuff is too small to hold result
   //              -ENOENT    - Context not initialized
   //          = 0 Success, outbuff contains a pointer to the encrypted data.
   //
   EPNAME("Encrypt");

   // We must have a key
   if (!sessionKey)
      return -ENOENT;

   // And something to encrypt
   if (!inbuf || inlen <= 0 || !outbuf)
      return -EINVAL;

   // Get output buffer
   char *buf = (char *)malloc(sessionKey->EncOutLength(inlen));
   if (!buf)
      return -ENOMEM;

   // Encrypt
   int len = sessionKey->Encrypt(inbuf, inlen, buf);
   if (len <= 0) {
      SafeFree(buf);
      return -EINVAL;
   }

   // Create and fill output buffer
   *outbuf = new XrdSecBuffer(buf, len);

   // We are done
   DEBUG("encrypted buffer has "<<len<<" bytes");
   return 0;
}

//_____________________________________________________________________________
int XrdSecProtocolgsi::Decrypt(const char *inbuf,  // Data to be decrypted
                               int inlen,          // Length of data in inbuff
                               XrdSecBuffer **outbuf)  // Returns decrypted data
{
   // Decrypt data in inbuff and place it in outbuff.
   //
   // Returns: < 0 Failed,the return value is -errno (see Encrypt).
   //          = 0 Success, outbuff contains a pointer to the encrypted data.
   EPNAME("Decrypt");

   // We must have a key
   if (!sessionKey)
      return -ENOENT;

   // And something to decrypt
   if (!inbuf || inlen <= 0 || !outbuf)
      return -EINVAL;

   // Get output buffer
   char *buf = (char *)malloc(sessionKey->DecOutLength(inlen));
   if (!buf)
      return -ENOMEM;

   // Decrypt
   int len = sessionKey->Decrypt(inbuf, inlen, buf);
   if (len <= 0) {
      SafeFree(buf);
      return -EINVAL;
   }

   // Create and fill output buffer
   *outbuf = new XrdSecBuffer(buf, len);

   // We are done
   DEBUG("decrypted buffer has "<<len<<" bytes");
   return 0;
}

//_____________________________________________________________________________
int XrdSecProtocolgsi::Sign(const char  *inbuf,   // Data to be signed
                            int inlen,    // Length of data to be signed
                            XrdSecBuffer **outbuf)   // Buffer for the signature
{
   // Sign data in inbuff and place the signature in outbuf.
   //
   // Returns: < 0 Failed, returned value is -errno (see Encrypt).
   //          = 0 Success, the return value is the length of the signature
   //              placed in outbuf.
   //
   EPNAME("Sign");

   // We must have a PKI and a digest
   if (!sessionKsig || !sessionMD)
      return -ENOENT;

   // And something to sign
   if (!inbuf || inlen <= 0 || !outbuf)
      return -EINVAL;

   // Reset digest
   sessionMD->Reset(0);

   // Calculate digest
   sessionMD->Update(inbuf, inlen);
   sessionMD->Final();

   // Output length
   int lmax = sessionKsig->GetOutlen(sessionMD->Length());
   char *buf = (char *)malloc(lmax);
   if (!buf)
      return -ENOMEM;

   // Sign
   int len = sessionKsig->EncryptPrivate(sessionMD->Buffer(),
                                         sessionMD->Length(),
                                         buf, lmax);
   if (len <= 0) {
      SafeFree(buf);
      return -EINVAL;
   }

   // Create and fill output buffer
   *outbuf = new XrdSecBuffer(buf, len);

   // We are done
   DEBUG("signature has "<<len<<" bytes");
   return 0;
}

//_____________________________________________________________________________
int XrdSecProtocolgsi::Verify(const char  *inbuf,   // Data to be verified
                              int  inlen,           // Length of data in inbuf
                              const char  *sigbuf,  // Buffer with signature
                              int  siglen)          // Length of signature
{
   // Verify a signature
   //
   // Returns: < 0 Failed, returned value is -errno (see Encrypt).
   //          = 0 Signature matches the value in inbuff.
   //          > 0 Failed to verify, signature does not match inbuff data.
   //
   EPNAME("Verify");

   // We must have a PKI and a digest
   if (!sessionKver || !sessionMD)
      return -ENOENT;

   // And something to verify
   if (!inbuf || inlen <= 0 || !sigbuf || siglen <= 0)
      return -EINVAL;

   // Reset digest
   sessionMD->Reset(0);

   // Calculate digest
   sessionMD->Update(inbuf, inlen);
   sessionMD->Final();

   // Output length
   int lmax = sessionKver->GetOutlen(siglen);
   char *buf = new char[lmax];
   if (!buf)
      return -ENOMEM;

   // Decrypt signature
   int len = sessionKver->DecryptPublic(sigbuf, siglen, buf, lmax);
   if (len <= 0) {
      delete[] buf;
      return -EINVAL;
   }

   // Verify signature
   bool bad = 1;
   if (len == sessionMD->Length()) {
      if (!strncmp(buf, sessionMD->Buffer(), len)) {
         // Signature matches
         bad = 0;
         DEBUG("signature successfully verified");
      }
   }

   // Cleanup
   if (buf) delete[] buf;

   // We are done
   return ((bad) ? 1 : 0);
}

//_____________________________________________________________________________
int XrdSecProtocolgsi::getKey(char *kbuf, int klen)
{
   // Get the current encryption key
   //
   // Returns: < 0 Failed, returned value if -errno (see Encrypt)
   //         >= 0 The size of the encyption key. The supplied buffer of length
   //              size hold the key. If the buffer address is 0, only the 
   //              size of the key is returned.
   //
   EPNAME("getKey");

   // Check if we have to serialize the key
   if (!bucketKey) {

      // We must have a key for that
      if (!sessionKey)
         // Invalid call
         return -ENOENT;
      // Create bucket
      bucketKey = sessionKey->AsBucket();
   }

   // Prepare output now, if we have any
   if (bucketKey) {
      // If are asked only the size, we are done
      if (kbuf == 0)
         return bucketKey->size;

      // Check the size of the buffer
      if (klen < bucketKey->size)
         // Too small
         return -EOVERFLOW;

      // Copy the buffer
      memcpy(kbuf, bucketKey->buffer, bucketKey->size);

      // We are done
      DEBUG("session key exported");
      return bucketKey->size;
   }

   // Key exists but we could export it in bucket format
   return -ENOMEM;
}

//_____________________________________________________________________________
int XrdSecProtocolgsi::setKey(char *kbuf, int klen)
{
   // Set the current encryption key
   //
   // Returns: < 0 Failed, returned value if -errno (see Encrypt)
   //            0 The new key has been set.
   //
   EPNAME("setKey");

   // Make sur that we can initialize the new key
   if (!kbuf || klen <= 0) 
      // Invalid inputs
      return -EINVAL;

   if (!sessionCF) 
      // Invalid context
      return -ENOENT;

   // Put the buffer key into a bucket
   XrdSutBucket *bck = new XrdSutBucket();
   if (!bck)
      // Cannot get buffer: out-of-resources?
      return -ENOMEM;
   // Set key buffer
   bck->SetBuf(kbuf, klen);

   // Init a new cipher from the bucket
   XrdCryptoCipher *newKey = sessionCF->Cipher(bck);
   if (!newKey) {
      SafeDelete(bck);
      return -ENOMEM;
   }

   // Delete current key
   SafeDelete(sessionKey);

   // Set the new key
   sessionKey = newKey;

   // Cleanup
   SafeDelete(bck);

   // Ok 
   DEBUG("session key update");
   return 0;
}

/******************************************************************************/
/*             C l i e n t   O r i e n t e d   F u n c t i o n s              */
/******************************************************************************/
/******************************************************************************/
/*                        g e t C r e d e n t i a l s                         */
/******************************************************************************/

XrdSecCredentials *XrdSecProtocolgsi::getCredentials(XrdSecParameters *parm,
                                                     XrdOucErrInfo    *ei)
{
   // Query client for the password; remote username and host
   // are specified in 'parm'. File '.rootnetrc' is checked. 
   EPNAME("getCredentials");

   // If we are a server the only reason to be here is to get the forwarded
   // or saved client credentials
   if (srvMode) {
      XrdSecCredentials *creds = 0;
      if (proxyChain) {
         // Export the proxy chain into a bucket
         XrdCryptoX509ExportChain_t ExportChain = sessionCF->X509ExportChain();
         if (ExportChain) {
            XrdSutBucket *bck = (*ExportChain)(proxyChain, 1);
            if (bck) {
               // We need to duplicate it because XrdSecCredentials uses
               // {malloc, free} instead of {new, delete}
               char *nbuf = (char *) malloc(bck->size);
               if (nbuf) {
                  memcpy(nbuf, bck->buffer, bck->size);
                  // Import the buffer in a XrdSecCredentials object
                  creds = new XrdSecCredentials(nbuf, bck->size);
               }
               delete bck;
            }
         }
      }
      return creds;
   }

   // Handshake vars container must be initialized at this point
   if (!hs)
      return ErrC(ei,0,0,0,kGSErrError,
                  "handshake var container missing","getCredentials");
   //
   // Nothing to do if buffer is empty
   if ((!parm && !hs->Parms) || (parm && (!(parm->buffer) || parm->size <= 0))) {
      if (hs->Iter == 0) 
         return ErrC(ei,0,0,0,kGSErrNoBuffer,"missing parameters","getCredentials");
      else
         return (XrdSecCredentials *)0;
   }

   // Count interations
   (hs->Iter)++;

   // Update time stamp
   hs->TimeStamp = time(0);

   // Local vars
   int step = 0;
   int nextstep = 0;
   const char *stepstr = 0;
   char *bpub = 0;
   int lpub = 0;
   String CryptList = "";
   String Host = "";
   String RemID = "";
   String Emsg;
   String specID = "";
   String issuerHash = "";
   // Buffer / Bucket related
   XrdSutBuffer *bpar   = 0;  // Global buffer
   XrdSutBuffer *bmai   = 0;  // Main buffer

   //
   // Decode received buffer
   bpar = hs->Parms;
   if (!bpar && !(bpar = new XrdSutBuffer((const char *)parm->buffer,parm->size)))
      return ErrC(ei,0,0,0,kGSErrDecodeBuffer,"global",stepstr);
   // Ownership has been transferred
   hs->Parms = 0;
   //
   // Check protocol ID name
   if (strcmp(bpar->GetProtocol(),XrdSecPROTOIDENT))
      return ErrC(ei,bpar,bmai,0,kGSErrBadProtocol,stepstr);
   //
   // The step indicates what we are supposed to do
   if (!(step = bpar->GetStep())) {
      // The first, fake, step
      step = kXGS_init;
      bpar->SetStep(step);
   }
   stepstr = ServerStepStr(step);
   // Dump, if requested
   if (QTRACE(Authen)) {
      XrdOucString msg("IN: ");
      msg += stepstr;
      bpar->Dump(msg.c_str());
   }
   //
   // Parse input buffer
   if (ParseClientInput(bpar, &bmai, Emsg) == -1) {
      DEBUG(Emsg<<" CF: "<<sessionCF);
      return ErrC(ei,bpar,bmai,0,kGSErrParseBuffer,Emsg.c_str(),stepstr);
   }
   // Dump, if requested
   if (QTRACE(Authen)) {
      if (bmai)
         bmai->Dump("IN: main");
    }
   //
   // Version
   DEBUG("version run by server: "<< hs->RemVers);
   //
   // Check random challenge
   if (!CheckRtag(bmai, Emsg))
      return ErrC(ei,bpar,bmai,0,kGSErrBadRndmTag,Emsg.c_str(),stepstr);
   //
   // Login name if any
   String user(Entity.name);
   if (user.length() <= 0) user = getenv("XrdSecUSER");
   //
   // Now action depens on the step
   nextstep = kXGC_none;

   XrdCryptoX509 *c = 0;

   switch (step) {

   case kXGS_init:
      //
      // Add bucket with cryptomod to the global list
      // (This must be always visible from now on)
      if (bpar->AddBucket(hs->CryptoMod,kXRS_cryptomod) != 0)
         return ErrC(ei,bpar,bmai,0,
              kGSErrCreateBucket,XrdSutBuckStr(kXRS_cryptomod),stepstr);
      //
      // Add bucket with our version to the main list
      if (bpar->MarshalBucket(kXRS_version,(kXR_int32)(Version)) != 0)
         return ErrC(ei,bpar,bmai,0, kGSErrCreateBucket,
                XrdSutBuckStr(kXRS_version),"global",stepstr);
      //
      // Add our issuer hash
      c = hs->PxyChain->Begin();
      if (c->type == XrdCryptoX509::kCA)
        issuerHash = c->SubjectHash();
      else
        issuerHash = c->IssuerHash();
      while ((c = hs->PxyChain->Next()) != 0) {
        if (c->type != XrdCryptoX509::kCA)
          break;
        issuerHash = c->SubjectHash();
      }
      DEBUG("Client issuer hash: " << issuerHash);
      if (bpar->AddBucket(issuerHash,kXRS_issuer_hash) != 0)
            return ErrC(ei,bpar,bmai,0, kGSErrCreateBucket,
                        XrdSutBuckStr(kXRS_issuer_hash),stepstr);
      //
      // Add bucket with our delegate proxy options
      if (hs->RemVers >= 10100) {
         if (bpar->MarshalBucket(kXRS_clnt_opts,(kXR_int32)(hs->Options)) != 0)
         return ErrC(ei,bpar,bmai,0, kGSErrCreateBucket,
                XrdSutBuckStr(kXRS_clnt_opts),"global",stepstr);
      }

      //
      nextstep = kXGC_certreq;
      break;

   case kXGS_cert:
      //
      // We must have a session cipher at this point
      if (!(sessionKey))
         return ErrC(ei,bpar,bmai,0,
              kGSErrNoCipher,"session cipher",stepstr);

      //
      // Extract buffer with public info for the cipher agreement
      if (!(bpub = sessionKey->Public(lpub))) 
         return ErrC(ei,bpar,bmai,0,
                     kGSErrNoPublic,"session",stepstr);
      //
      // Add it to the global list
      if (bpar->UpdateBucket(bpub,lpub,kXRS_puk) != 0)
         return ErrC(ei,bpar,bmai,0, kGSErrAddBucket,
                     XrdSutBuckStr(kXRS_puk),"global",stepstr);
      //
      // Add the proxy certificate
      bmai->AddBucket(hs->Cbck);
      //
      // Add login name if any, needed while chosing where to export the proxies
      if (user.length() > 0) {
         if (bmai->AddBucket(user, kXRS_user) != 0)
            return ErrC(ei,bpar,bmai,0, kGSErrCreateBucket,
                        XrdSutBuckStr(kXRS_user),stepstr);
      }
      //
      nextstep = kXGC_cert;
      break;

   case kXGS_pxyreq:
      //
      // If something went wrong, send explanation
      if (Emsg.length() > 0) {
         if (bmai->AddBucket(Emsg,kXRS_message) != 0)
            return ErrC(ei,bpar,bmai,0, kGSErrCreateBucket,
                        XrdSutBuckStr(kXRS_message),stepstr);
      }
      //
      // Add login name if any, needed while chosing where to export the proxies
      if (user.length() > 0) {
         if (bmai->AddBucket(user, kXRS_user) != 0)
            return ErrC(ei,bpar,bmai,0, kGSErrCreateBucket,
                        XrdSutBuckStr(kXRS_user),stepstr);
      }
      //
      // The relevant buckets should already be in the buffers
      nextstep = kXGC_sigpxy;
      break;

   default:
      return ErrC(ei,bpar,bmai,0, kGSErrBadOpt,stepstr);
   }

   //
   // Serialize and encrypt
   if (AddSerialized('c', nextstep, hs->ID,
                     bpar, bmai, kXRS_main, sessionKey) != 0) {
      // Remove to avoid destruction
      bmai->Remove(hs->Cbck);
      return ErrC(ei,bpar,bmai,0,
                  kGSErrSerialBuffer,"main",stepstr);
   }
   //
   // Serialize the global buffer
   char *bser = 0;
   int nser = bpar->Serialized(&bser,'f');

   if (QTRACE(Authen)) {
      XrdOucString msg("OUT: ");
      msg += ClientStepStr(bpar->GetStep());
      bpar->Dump(msg.c_str());
      msg.replace(ClientStepStr(bpar->GetStep()), "main");
      bmai->Dump(msg.c_str());
   }
   //
   // Remove to avoid destruction
   bmai->Remove(hs->Cbck);
   //
   // We may release the buffers now
   REL2(bpar,bmai);
   //
   // Return serialized buffer
   if (nser > 0) {
      DEBUG("returned " << nser <<" bytes of credentials");
      return new XrdSecCredentials(bser, nser);
   } else {
      DEBUG("problems with final serialization");
      return (XrdSecCredentials *)0;
   }
}

/******************************************************************************/
/*               S e r v e r   O r i e n t e d   M e t h o d s                */
/******************************************************************************/
/******************************************************************************/
/*                          A u t h e n t i c a t e                           */
/******************************************************************************/

int XrdSecProtocolgsi::Authenticate(XrdSecCredentials *cred,
                                    XrdSecParameters **parms,
                                    XrdOucErrInfo     *ei)
{
   //
   // Check if we have any credentials or if no credentials really needed.
   // In either case, use host name as client name
   EPNAME("Authenticate");

   //
   // If cred buffer is two small or empty assume host protocol
   if (cred->size <= (int)XrdSecPROTOIDLEN || !cred->buffer) {
      strncpy(Entity.prot, "host", sizeof(Entity.prot));
      return 0;
   }

   // Handshake vars conatiner must be initialized at this point
   if (!hs)
      return ErrS(Entity.tident,ei,0,0,0,kGSErrError,
                  "handshake var container missing",
                  "protocol initialization problems");

   // Update time stamp
   hs->TimeStamp = time(0);

   //
   // ID of this handshaking
   if (hs->ID.length() <= 0)
      hs->ID = Entity.tident;
   DEBUG("handshaking ID: " << hs->ID);

   // Local vars 
   int kS_rc = kgST_more;
   int step = 0;
   int nextstep = 0;
   char *bpub = 0;
   int lpub = 0;
   const char *stepstr = 0;
   String Message;
   String CryptList;
   String Host;
   String SrvPuKExp;
   String Salt;
   String RndmTag;
   String ClntMsg(256);
   // Buffer related
   XrdSutBuffer    *bpar = 0;  // Global buffer
   XrdSutBuffer    *bmai = 0;  // Main buffer

   //
   // Decode received buffer
   if (!(bpar = new XrdSutBuffer((const char *)cred->buffer,cred->size)))
      return ErrS(hs->ID,ei,0,0,0,kGSErrDecodeBuffer,"global",stepstr);
   //
   // Check protocol ID name
   if (strcmp(bpar->GetProtocol(),XrdSecPROTOIDENT))
      return ErrS(hs->ID,ei,bpar,bmai,0,kGSErrBadProtocol,stepstr);
   //
   // The step indicates what we are supposed to do
   step = bpar->GetStep();
   stepstr = ClientStepStr(step);
   // Dump, if requested
   if (QTRACE(Authen)) {
      XrdOucString msg("IN: ");
      msg += stepstr;
      bpar->Dump(msg.c_str());
   }
   //
   // Parse input buffer
   if (ParseServerInput(bpar, &bmai, ClntMsg) == -1) {
      DEBUG(ClntMsg);
      return ErrS(hs->ID,ei,bpar,bmai,0,kGSErrParseBuffer,ClntMsg.c_str(),stepstr);
   }
   //
   // Version
   DEBUG("version run by client: "<< hs->RemVers);
   DEBUG("options req by client: "<< hs->Options);
   //
   // Dump, if requested
   if (QTRACE(Authen)) {
      if (bmai)
         bmai->Dump("IN: main");
   }
   //
   // Check random challenge
   if (!CheckRtag(bmai, ClntMsg))
      return ErrS(hs->ID,ei,bpar,bmai,0,kGSErrBadRndmTag,stepstr,ClntMsg.c_str());
   //
   // Now action depens on the step
   switch (step) {

   case kXGC_certreq:
      //
      // Client required us to send our certificate and cipher public part:
      // add first this last one.
      // Extract buffer with public info for the cipher agreement
      if (!(bpub = hs->Rcip->Public(lpub))) 
         return ErrS(hs->ID,ei,bpar,bmai,0, kGSErrNoPublic,
                                         "session",stepstr);
      //
      // Add it to the global list
      if (bpar->AddBucket(bpub,lpub,kXRS_puk) != 0)
         return ErrS(hs->ID,ei,bpar,bmai,0, kGSErrAddBucket,
                                            "main",stepstr);
      //
      // Add bucket with list of supported ciphers
      if (bpar->AddBucket(DefCipher,kXRS_cipher_alg) != 0)
         return ErrS(hs->ID,ei,bpar,bmai,0,
              kGSErrAddBucket,XrdSutBuckStr(kXRS_cipher_alg),stepstr);
      //
      // Add bucket with list of supported MDs
      if (bpar->AddBucket(DefMD,kXRS_md_alg) != 0)
         return ErrS(hs->ID,ei,bpar,bmai,0,
              kGSErrAddBucket,XrdSutBuckStr(kXRS_md_alg),stepstr);
      //
      // Add the server certificate
      bpar->AddBucket(hs->Cbck);

      // We are done for the moment
      nextstep = kXGS_cert;
      break;

   case kXGC_cert:
      //
      // Client sent its own credentials: their are checked in
      // ParseServerInput, so if we are here they are OK
      kS_rc = kgST_ok;
      nextstep = kXGS_none;

      if (GMAPOpt > 0) {
         // Get name from gridmap
         String name;
         QueryGMAP(hs->Chain, hs->TimeStamp, name);
         DEBUG("username(s) associated with this DN: "<<name);
         if (name.length() <= 0) {
            // Grid map lookup failure
            if (GMAPOpt == 2) {
               // It was required, so we fail
               kS_rc = kgST_error;
               PRINT("ERROR: user mapping required, but lookup failed - failure");
               break;
            } else {
               DEBUG("WARNING: user mapping lookup failed - use DN or DN-hash as name");
            }
         } else {
            //
            // Extract user login name, if any
            XrdSutBucket *bck = 0;
            String user;
            if ((bck = bmai->GetBucket(kXRS_user))) {
               bck->ToString(user);
               bmai->Deactivate(kXRS_user);
            }
            DEBUG("target user: "<<user);
            if (user.length() > 0) {
               // Check if the wanted username is authorized
               String u;
               int from = 0;
               bool ok = 0;
               while ((from = name.tokenize(u, from, ',')) != -1) {
                  if (user == u) { ok = 1; break; }
               }
               if (ok) {
                  name = u;
                  DEBUG("DN mapping: requested user is authorized: name is '"<<name<<"'");
               } else {
                  // The requested username is not in the list; we warn and default to the first
                  // found (to be Globus compliant)
                  if (name.find(',') != STR_NPOS) name.erase(name.find(','));
                  PRINT("WARNING: user mapping lookup ok, but the requested user is not"
                        " authorized ("<<user<<"). Instead, mapped as " << name << ".");
               }
            } else {
               // No username requested: we default to the first found (to be Globus compliant)
               if (name.find(',') != STR_NPOS) name.erase(name.find(','));
               DEBUG("user mapping lookup successful: name is '"<<name<<"'");
            }
            Entity.name = strdup(name.c_str());
         }
      }
      // If not set, use DN
      if (!Entity.name || (strlen(Entity.name) <= 0)) {
         // No grid map: set the hash of the client DN as name
         if (!GMAPuseDNname && hs->Chain->EEChash()) {
            Entity.name = strdup(hs->Chain->EEChash());
         } else if (GMAPuseDNname && hs->Chain->EECname()) {
            Entity.name = strdup(hs->Chain->EECname());
         } else {
            PRINT("WARNING: DN missing: corruption? ");
         }
      }

      // Export proxy for authorization, if required
      if (AuthzPxyWhat >= 0) {
         XrdSutBucket *b = (AuthzPxyWhat == 1 && hs->Chain->End()) ? hs->Chain->End()->Export()
                                                                   : XrdCryptosslX509ExportChain(hs->Chain, true);
         XrdOucString s;
         b->ToString(s);
         if (AuthzPxyWhere == 1) {
            Entity.creds = strdup(s.c_str());
            Entity.credslen = s.length();
         } else {
            // This should be deprecated
            Entity.endorsements = strdup(s.c_str());
         }
         delete b;
         
         DEBUG("Entity.endorsements: "<<(void *)Entity.endorsements);
         DEBUG("Entity.creds:        "<<(void *)Entity.creds);
         DEBUG("Entity.credslen:     "<<Entity.credslen);
      }

      if (hs->RemVers >= 10100) {
         if (hs->PxyChain) {
            // The client is going to send over info for delegation
            kS_rc = kgST_more;
            nextstep = kXGS_pxyreq;
         }
      }

      break;

   case kXGC_sigpxy:
      //
      // Nothing to do after this
      kS_rc = kgST_ok;
      nextstep = kXGS_none;
      //
      // If something went wrong, print explanation
      if (ClntMsg.length() > 0) {
         PRINT(ClntMsg);
      }
      break;

   default:
      return ErrS(hs->ID,ei,bpar,bmai,0, kGSErrBadOpt, stepstr);
   }

   if (kS_rc == kgST_more) {
      //
      // Add message to client
      if (ClntMsg.length() > 0)
         if (bmai->AddBucket(ClntMsg,kXRS_message) != 0) {
            DEBUG("problems adding bucket with message for client");
         }
      //
      // Serialize, encrypt and add to the global list
      if (AddSerialized('s', nextstep, hs->ID,
                        bpar, bmai, kXRS_main, sessionKey) != 0) {
         // Remove to avoid destruction
         bpar->Remove(hs->Cbck);
         return ErrS(hs->ID,ei,bpar,bmai,0, kGSErrSerialBuffer,
                     "main / session cipher",stepstr);
      }
      //
      // Serialize the global buffer
      char *bser = 0;
      int nser = bpar->Serialized(&bser,'f');
      //
      // Dump, if requested
      if (QTRACE(Authen)) {
         XrdOucString msg("OUT: ");
         msg += ServerStepStr(bpar->GetStep());
         bpar->Dump(msg.c_str());
         msg.replace(ServerStepStr(bpar->GetStep()), "main");
         bmai->Dump(msg.c_str());
      }
      //
      // Create buffer for client
      *parms = new XrdSecParameters(bser,nser);
      //
      // Remove to avoid destruction
      bpar->Remove(hs->Cbck);

   } else {
      //
      // Remove to avoid destruction
      bpar->Remove(hs->Cbck);
      //
      // Cleanup handshake vars
      SafeDelete(hs);
   }
   //
   // We may release the buffers now
   REL2(bpar,bmai);
   //
   // All done
   return kS_rc;
}

/******************************************************************************/
/*              X r d S e c P r o t o c o l p w d I n i t                     */
/******************************************************************************/

extern "C"
{
char *XrdSecProtocolgsiInit(const char mode,
                            const char *parms, XrdOucErrInfo *erp)
{
   // One-time protocol initialization, filling the static flags and options
   // of the protocol.
   // For clients (mode == 'c') we use values in envs.
   // For servers (mode == 's') the command line options are passed through
   // parms.
   gsiOptions opts;
   char *rc = (char *)"";
   char *cenv = 0;

   //
   // Clients first
   if (mode == 'c') {
      //
      // Decode envs:
      //             "XrdSecDEBUG"           debug flag ("0","1","2","3")
      //             "XrdSecGSICADIR"        full path to an alternative path
      //                                     containing the CA info
      //                                     [/etc/grid-security/certificates]
      //             "XrdSecGSICRLDIR"       full path to an alternative path
      //                                     containing the CRL info
      //                                     [/etc/grid-security/certificates]
      //             "XrdSecGSICRLEXT"       default extension of CRL files [.r0]
      //             "XrdSecGSIUSERCERT"     full path to an alternative file
      //                                     containing the user certificate
      //                                     [$HOME/.globus/usercert.pem]
      //             "XrdSecGSIUSERKEY"      full path to an alternative file
      //                                     containing the user key
      //                                     [$HOME/.globus/userkey.pem]
      //             "XrdSecGSIUSERPROXY"    full path to an alternative file
      //                                     containing the user proxy
      //                                     [/tmp/x509up_u<uid>]
      //             "XrdSecGSIPROXYVALID"   validity of proxies in the 
      //                                     grid-proxy-init format
      //                                     ["12:00", i.e. 12 hours]
      //             "XrdSecGSIPROXYDEPLEN"  depth of signature path for proxies;
      //                                     use -1 for unlimited [0]
      //             "XrdSecGSIPROXYKEYBITS" bits in PKI for proxies [512]
      //             "XrdSecGSICACHECK"      CA check level [1]:
      //                                      0 do not verify;
      //                                      1 verify if self-signed, warn if not;
      //                                      2 verify in all cases, fail if not possible
      //             "XrdSecGSICRLCHECK"     CRL check level [2]:
      //                                      0 don't care;
      //                                      1 use if available;
      //                                      2 require,
      //                                      3 require non-expired CRL
      //             "XrdSecGSIDELEGPROXY"   Forwarding of credentials option:
      //                                     0 none; 1 sign request created
      //                                     by server; 2 forward local proxy
      //                                     (include private key) [0]
      //             "XrdSecGSISIGNPROXY"    permission to sign requests
      //                                     0 no, 1 yes [1]
      //             "XrdSecGSISRVNAMES"     Server names allowed: if the server CN
      //                                     does not match any of these, or it is
      //                                     explicitely denied by these, or it is
      //                                     not in the form "*/<hostname>", the
      //                                     handshake fails.

      //
      opts.mode = mode;
      // debug
      cenv = getenv("XrdSecDEBUG");
      if (cenv)
         if (cenv[0] >= 49 && cenv[0] <= 51) opts.debug = atoi(cenv);

      // directory with CA certificates
      cenv = (getenv("XrdSecGSICADIR") ? getenv("XrdSecGSICADIR")
                                       : getenv("X509_CERT_DIR"));
      if (cenv)
         opts.certdir = strdup(cenv);

      // directory with CRL info
      cenv = (getenv("XrdSecGSICRLDIR") ? getenv("XrdSecGSICRLDIR")
                                        : getenv("X509_CERT_DIR"));
      if (cenv)
         opts.crldir = strdup(cenv);

      // Default extension CRL files
      cenv = getenv("XrdSecGSICRLEXT");
      if (cenv)
         opts.crlext = strdup(cenv);

      // file with user cert
      cenv = (getenv("XrdSecGSIUSERCERT") ? getenv("XrdSecGSIUSERCERT")
                                          : getenv("X509_USER_CERT"));
      if (cenv)
         opts.cert = strdup(cenv);

      // file with user key
      cenv = (getenv("XrdSecGSIUSERKEY") ? getenv("XrdSecGSIUSERKEY")
                                         : getenv("X509_USER_KEY"));
      if (cenv)
         opts.key = strdup(cenv);

      // file with user proxy
      cenv = (getenv("XrdSecGSIUSERPROXY") ? getenv("XrdSecGSIUSERPROXY")
                                           : getenv("X509_USER_PROXY"));
      if (cenv)
         opts.proxy = strdup(cenv);

      // file with user proxy
      cenv = getenv("XrdSecGSIPROXYVALID");
      if (cenv)
         opts.valid = strdup(cenv);

      // Depth of signature path for proxies
      cenv = getenv("XrdSecGSIPROXYDEPLEN");
      if (cenv)
         opts.deplen = atoi(cenv);

      // Key Bit length 
      cenv = getenv("XrdSecGSIPROXYKEYBITS");
      if (cenv)
         opts.bits = atoi(cenv);

      // CA verification level 
      cenv = getenv("XrdSecGSICACHECK");
      if (cenv)
         opts.ca = atoi(cenv);

      // CRL check level 
      cenv = getenv("XrdSecGSICRLCHECK");
      if (cenv)
         opts.crl = atoi(cenv);

      // Delegate proxy
      cenv = getenv("XrdSecGSIDELEGPROXY");
      if (cenv)
         opts.dlgpxy = atoi(cenv);

      // Sign delegate proxy requests
      cenv = getenv("XrdSecGSISIGNPROXY");
      if (cenv)
         opts.sigpxy = atoi(cenv);

      // Allowed server name formats
      cenv = getenv("XrdSecGSISRVNAMES");
      if (cenv)
         opts.srvnames = strdup(cenv);

      //
      // Setup the object with the chosen options
      rc = XrdSecProtocolgsi::Init(opts,erp);

      // Some cleanup
      SafeFree(opts.certdir);
      SafeFree(opts.crldir);
      SafeFree(opts.crlext);
      SafeFree(opts.cert);
      SafeFree(opts.key);
      SafeFree(opts.proxy);
      SafeFree(opts.valid);
      SafeFree(opts.srvnames);

      // We are done
      return rc;
   }

   // Take into account xrootd debug flag
   cenv = getenv("XRDDEBUG");
   if (cenv && !strcmp(cenv,"1")) opts.debug = 1;

   //
   // Server initialization
   if (parms) {
      //
      // Duplicate the parms
      char parmbuff[1024];
      strlcpy(parmbuff, parms, sizeof(parmbuff));
      //
      // The tokenizer
      XrdOucTokenizer inParms(parmbuff);
      //
      // Decode parms:
      // for servers:
      //              [-d:<debug_level>]
      //              [-c:[-]ssl[:[-]<CryptoModuleName]]
      //              [-certdir:<dir_with_CA_info>]
      //              [-crldir:<dir_with_CRL_info>]
      //              [-crlext:<default_extension_CRL_files>]
      //              [-cert:<path_to_server_certificate>]
      //              [-key:<path_to_server_key>]
      //              [-cipher:<list_of_supported_ciphers>]
      //              [-md:<list_of_supported_digests>]
      //              [-ca:<crl_verification_level>]
      //              [-crl:<crl_check_level>]
      //              [-gridmap:<grid_map_file>]
      //              [-gmapfun:<grid_map_function>]
      //              [-gmapfunparms:<grid_map_function_init_parameters>]
      //              [-authzfun:<authz_function>]
      //              [-authzfunparms:<authz_function_init_parameters>]
      //              [-gmapto:<grid_map_cache_entry_validity_in_secs>]
      //              [-gmapopt:<grid_map_check_option>]
      //              [-dlgpxy:<proxy_req_option>]
      //              [-exppxy:<filetemplate>]
      //              [-authzpxy]
      //
      int debug = -1;
      String clist = "";
      String certdir = "";
      String crldir = "";
      String crlext = "";
      String cert = "";
      String key = "";
      String cipher = "";
      String md = "";
      String gridmap = "";
      String gmapfun = "";
      String gmapfunparms = "";
      String authzfun = "";
      String authzfunparms = "";
      String exppxy = "";
      int ca = 1;
      int crl = 1;
      int ogmap = 1;
      int gmapto = -1;
      int dlgpxy = 0;
      int authzpxy = 0;
      char *op = 0;
      while (inParms.GetLine()) { 
         while ((op = inParms.GetToken())) {
            if (!strncmp(op, "-d:",3)) {
               debug = atoi(op+3);
            } else if (!strncmp(op, "-c:",3)) {
               clist = (const char *)(op+3);
            } else if (!strncmp(op, "-certdir:",9)) {
               certdir = (const char *)(op+9);
            } else if (!strncmp(op, "-crldir:",8)) {
               crldir = (const char *)(op+8);
            } else if (!strncmp(op, "-crlext:",8)) {
               crlext = (const char *)(op+8);
            } else if (!strncmp(op, "-cert:",6)) {
               cert = (const char *)(op+6);
            } else if (!strncmp(op, "-key:",5)) {
               key = (const char *)(op+5);
            } else if (!strncmp(op, "-cipher:",8)) {
               cipher = (const char *)(op+8);
            } else if (!strncmp(op, "-md:",4)) {
               md = (const char *)(op+4);
            } else if (!strncmp(op, "-ca:",4)) {
               ca = atoi(op+4);
            } else if (!strncmp(op, "-crl:",5)) {
               crl = atoi(op+5);
            } else if (!strncmp(op, "-gmapopt:",9)) {
               ogmap = atoi(op+9);
            } else if (!strncmp(op, "-gridmap:",9)) {
               gridmap = (const char *)(op+9);
            } else if (!strncmp(op, "-gmapfun:",9)) {
               gmapfun = (const char *)(op+9);
            } else if (!strncmp(op, "-gmapfunparms:",14)) {
               gmapfunparms = (const char *)(op+14);
            } else if (!strncmp(op, "-authzfun:",10)) {
               authzfun = (const char *)(op+10);
            } else if (!strncmp(op, "-authzfunparms:",15)) {
               authzfunparms = (const char *)(op+15);
            } else if (!strncmp(op, "-gmapto:",8)) {
               gmapto = atoi(op+8);
            } else if (!strncmp(op, "-dlgpxy:",8)) {
               dlgpxy = atoi(op+8);
            } else if (!strncmp(op, "-exppxy:",8)) {
               exppxy = (const char *)(op+8);
            } else if (!strncmp(op, "-authzpxy:",10)) {
               authzpxy = atoi(op+10);
            } else if (!strncmp(op, "-authzpxy",9)) {
               authzpxy = 11;
            }
         }
      }

      //
      // Build the option object
      opts.debug = (debug > -1) ? debug : opts.debug;
      opts.mode = 's';
      opts.ca = ca;
      opts.crl = crl;
      opts.ogmap = ogmap;
      opts.gmapto = gmapto;
      opts.dlgpxy = dlgpxy;
      opts.authzpxy = authzpxy;
      if (clist.length() > 0)
         opts.clist = (char *)clist.c_str();
      if (certdir.length() > 0)
         opts.certdir = (char *)certdir.c_str();
      if (crldir.length() > 0)
         opts.crldir = (char *)crldir.c_str();
      if (crlext.length() > 0)
         opts.crlext = (char *)crlext.c_str();
      if (cert.length() > 0)
         opts.cert = (char *)cert.c_str();
      if (key.length() > 0)
         opts.key = (char *)key.c_str();
      if (cipher.length() > 0)
         opts.cipher = (char *)cipher.c_str();
      if (md.length() > 0)
         opts.md = (char *)md.c_str();
      if (gridmap.length() > 0)
         opts.gridmap = (char *)gridmap.c_str();
      if (gmapfun.length() > 0)
         opts.gmapfun = (char *)gmapfun.c_str();
      if (gmapfunparms.length() > 0)
         opts.gmapfunparms = (char *)gmapfunparms.c_str();
      if (authzfun.length() > 0)
         opts.authzfun = (char *)authzfun.c_str();
      if (authzfunparms.length() > 0)
         opts.authzfunparms = (char *)authzfunparms.c_str();
      if (exppxy.length() > 0)
         opts.exppxy = (char *)exppxy.c_str();
      //
      // Setup the plug-in with the chosen options
      return XrdSecProtocolgsi::Init(opts,erp);
   }
   //
   // Setup the plug-in with the defaults
   return XrdSecProtocolgsi::Init(opts,erp);
}}


/******************************************************************************/
/*              X r d S e c P r o t o c o l p w d O b j e c t                 */
/******************************************************************************/

extern "C"
{
XrdSecProtocol *XrdSecProtocolgsiObject(const char              mode,
                                        const char             *hostname,
                                        const struct sockaddr  &netaddr,
                                        const char             *parms,
                                        XrdOucErrInfo    *erp)
{
   XrdSecProtocolgsi *prot;
   int options = XrdSecNOIPCHK;

   //
   // Get a new protocol object
   if (!(prot = new XrdSecProtocolgsi(options, hostname, &netaddr, parms))) {
      char *msg = (char *)"Secgsi: Insufficient memory for protocol.";
      if (erp) 
         erp->setErrInfo(ENOMEM, msg);
      else 
         cerr <<msg <<endl;
      return (XrdSecProtocol *)0;
   }
   //
   // We are done
   if (!erp)
      cerr << "protocol object instantiated" << endl;
   return prot;
}}


/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/

//_________________________________________________________________________
int XrdSecProtocolgsi::AddSerialized(char opt, kXR_int32 step, String ID,
                                     XrdSutBuffer *bls, XrdSutBuffer *buf,
                                     kXR_int32 type,
                                     XrdCryptoCipher *cip)
{
   // Serialize buf, and add it encrypted to bls as bucket type
   // Cipher cip is used if defined; else PuK rsa .
   // If both are undefined the buffer is just serialized and added.
   EPNAME("AddSerialized");

   if (!bls || !buf || (opt != 0 && opt != 'c' && opt != 's')) {
      DEBUG("invalid inputs ("
            <<bls<<","<<buf<<","<<opt<<")"
            <<" - type: "<<XrdSutBuckStr(type));
      return -1;
   }

   //
   // Add step to indicate the counterpart what we send
   if (step > 0) {
      bls->SetStep(step);
      buf->SetStep(step);
      hs->LastStep = step;
   }

   //
   // If a random tag has been sent and we have a session cipher,
   // we sign it
   XrdSutBucket *brt = buf->GetBucket(kXRS_rtag);
   if (brt && sessionKsig) {
      //
      // Encrypt random tag with session cipher
      if (sessionKsig->EncryptPrivate(*brt) <= 0) {
         DEBUG("error encrypting random tag");
         return -1;
      }
      //
      // Update type
      brt->type = kXRS_signed_rtag;
   }
   //
   // Add an random challenge: if a next exchange is required this will
   // allow to prove authenticity of counter part
   //
   // Generate new random tag and create a bucket
   String RndmTag;
   XrdSutRndm::GetRndmTag(RndmTag);
   //
   // Get bucket
   brt = 0;
   if (!(brt = new XrdSutBucket(RndmTag,kXRS_rtag))) {
      DEBUG("error creating random tag bucket");
      return -1;
   }
   buf->AddBucket(brt);
   //
   // Get cache entry
   if (!hs->Cref) {
      DEBUG("cache entry not found: protocol error");
      return -1;
   }
   //
   // Add random tag to the cache and update timestamp
   hs->Cref->buf1.SetBuf(brt->buffer,brt->size);
   hs->Cref->mtime = (kXR_int32)hs->TimeStamp;
   //
   // Now serialize the buffer ...
   char *bser = 0;
   int nser = buf->Serialized(&bser);
   //
   // Update bucket with this content
   XrdSutBucket *bck = 0;;
   if (!(bck = bls->GetBucket(type))) {
      // or create new bucket, if not existing
      if (!(bck = new XrdSutBucket(bser,nser,type))) {
         DEBUG("error creating bucket "
               <<" - type: "<<XrdSutBuckStr(type));
         return -1;
      }
      //
      // Add the bucket to the list
      bls->AddBucket(bck);
   } else {
      bck->Update(bser,nser);
   }
   //
   // Encrypted the bucket
   if (cip) {
      if (cip->Encrypt(*bck) == 0) {
         DEBUG("error encrypting bucket - cipher "
               <<" - type: "<<XrdSutBuckStr(type));
         return -1;
      }
   }
   // We are done
   return 0;
}

//_________________________________________________________________________
int XrdSecProtocolgsi::ParseClientInput(XrdSutBuffer *br, XrdSutBuffer **bm,
                                        String &cmsg)
{
   // Parse received buffer b,
   // Result used to fill the handshake local variables
   EPNAME("ParseClientInput");

   // Space for pointer to main buffer must be already allocated
   if (!br || !bm) {
      DEBUG("invalid inputs ("<<br<<","<<bm<<")");
      cmsg = "invalid inputs";
      return -1;
   }

   //
   // Get the step
   int step = br->GetStep();

   // Do the right action
   switch (step) {
      case kXGS_init:
         // Process message
         if (ClientDoInit(br, bm, cmsg) != 0)
            return -1;
         break;
      case kXGS_cert:
         // Process message
         if (ClientDoCert(br, bm, cmsg) != 0)
            return -1;
         break;
      case kXGS_pxyreq:
         // Process message
         if (ClientDoPxyreq(br, bm, cmsg) != 0)
            return -1;
         break;
      default:
         cmsg = "protocol error: unknown action: "; cmsg += step;
         return -1;
         break;
   }

   // We are done
   return 0;
}

//_________________________________________________________________________
int XrdSecProtocolgsi::ClientDoInit(XrdSutBuffer *br, XrdSutBuffer **bm,
                                    String &emsg)
{
   // Client side: process a kXGS_init message.
   // Return 0 on success, -1 on error. If the case, a message is returned
   // in cmsg.
   EPNAME("ClientDoInit");

   //
   // Create the main buffer as a copy of the buffer received
   if (!((*bm) = new XrdSutBuffer(br->GetProtocol(),br->GetOptions()))) {
      emsg = "error instantiating main buffer";
      return -1;
   }
   //
   // Extract server version from options
   String opts = br->GetOptions();
   int ii = opts.find("v:");
   if (ii >= 0) {
      String sver(opts,ii+2);
      sver.erase(sver.find(','));
      hs->RemVers = atoi(sver.c_str());
   } else {
      hs->RemVers = Version;
      emsg = "server version information not found in options:"
             " assume same as local";
   }
   //
   // Create cache
   if (!(hs->Cref = new XrdSutPFEntry("c"))) {
      emsg = "error creating cache";
      return -1;
   }
   //
   // Save server version in cache
   hs->Cref->status = hs->RemVers;
   //
   // Set options
   hs->Options = PxyReqOpts;
   //
   // Extract list of crypto modules 
   String clist;
   ii = opts.find("c:");
   if (ii >= 0) {
      clist.assign(opts, ii+2);
      clist.erase(clist.find(','));
   } else {
      DEBUG("Crypto list missing: protocol error? (use defaults)");
      clist = DefCrypto;
   }
   // Parse the list loading the first we can
   if (ParseCrypto(clist) != 0) {
      emsg = "cannot find / load crypto requested modules :";
      emsg += clist;
      return -1;
   }
   //
   // Extract server certificate CA hashes
   String srvca;
   ii = opts.find("ca:");
   if (ii >= 0) {
      srvca.assign(opts, ii+3);
      srvca.erase(srvca.find(','));
   }
   // Parse the list loading the first we can
   if (ParseCAlist(srvca) != 0) {
      emsg = "unknown CA: cannot verify server certificate";
      hs->Chain = 0;
      return -1;
   }
   //
   // Resolve place-holders in cert, key and proxy file paths, if any
   if (XrdSutResolve(UsrCert, Entity.host, Entity.vorg, Entity.grps, Entity.name) != 0) {
      DEBUG("Problems resolving templates in "<<UsrCert);
      return -1;
   }
   if (XrdSutResolve(UsrKey, Entity.host, Entity.vorg, Entity.grps, Entity.name) != 0) {
      DEBUG("Problems resolving templates in "<<UsrKey);
      return -1;
   }
   if (XrdSutResolve(UsrProxy, Entity.host, Entity.vorg, Entity.grps, Entity.name) != 0) {
      DEBUG("Problems resolving templates in "<<UsrProxy);
      return -1;
   }
   //
   // Load / Attach-to user proxies
   ProxyIn_t pi = {UsrCert.c_str(), UsrKey.c_str(), CAdir.c_str(),
                   UsrProxy.c_str(), PxyValid.c_str(),
                   DepLength, DefBits};
   ProxyOut_t po = {hs->PxyChain, sessionKsig, hs->Cbck };
   if (QueryProxy(1, &cachePxy, "Proxy:0",
                  sessionCF, hs->TimeStamp, &pi, &po) != 0) {
      emsg = "error getting user proxies";
      hs->Chain = 0;
      return -1;
   }
   // Save the result
   hs->PxyChain = po.chain;
   hs->Cbck = po.cbck;
   if (!(sessionKsig = sessionCF->RSA(*(po.ksig)))) {
      emsg = "could not get a copy of the signing key:";
      hs->Chain = 0;
      return -1;
   }
   //
   // And we are done;
   return 0;
}

//_________________________________________________________________________
int XrdSecProtocolgsi::ClientDoCert(XrdSutBuffer *br, XrdSutBuffer **bm,
                                    String &emsg)
{
   // Client side: process a kXGS_cert message.
   // Return 0 on success, -1 on error. If the case, a message is returned
   // in cmsg.
   EPNAME("ClientDoCert");
   XrdSutBucket *bck = 0;

   //
   // make sure the cache is still there
   if (!hs->Cref) {
      emsg = "cache entry not found";
      hs->Chain = 0;
      return -1;
   }
   //
   // make sure is not too old
   int reftime = hs->TimeStamp - TimeSkew;
   if (hs->Cref->mtime < reftime) {
      emsg = "cache entry expired";
      // Remove: should not be checked a second time
      SafeDelete(hs->Cref);
      hs->Chain = 0;
      return -1;
   }
   //
   // Get from cache version run by server
   hs->RemVers = hs->Cref->status;

   //
   // Extract list of cipher algorithms supported by the server
   String cip = "";
   if ((bck = br->GetBucket(kXRS_cipher_alg))) {
      String ciplist;
      bck->ToString(ciplist);
      // Parse the list
      int from = 0;
      while ((from = ciplist.tokenize(cip, from, ':')) != -1) {
         if (cip.length() > 0) 
            if (sessionCF->SupportedCipher(cip.c_str()))
               break;
         cip = "";
      }
      if (cip.length() > 0)
         // COmmunicate to server
         br->UpdateBucket(cip, kXRS_cipher_alg);
   } else {
      DEBUG("WARNING: list of ciphers supported by server missing"
            " - using default");
   }

   //
   // Extract server public part for session cipher
   if (!(bck = br->GetBucket(kXRS_puk))) {
      emsg = "server public part for session cipher missing";
      hs->Chain = 0;
      return -1;
   }
   //
   // Initialize session cipher
   SafeDelete(sessionKey);
   if (!(sessionKey =
         sessionCF->Cipher(0,bck->buffer,bck->size,cip.c_str()))) {
            DEBUG("could not instantiate session cipher "
                  "using cipher public info from server");
            emsg = "could not instantiate session cipher ";
   }
   //
   // Extract server certificate
   if (!(bck = br->GetBucket(kXRS_x509))) {
      emsg = "server certificate missing";
      hs->Chain = 0;
      return -1;
   }

   //
   // Finalize chain: get a copy of it (we do not touch the reference)
   hs->Chain = new X509Chain(hs->Chain);
   if (!(hs->Chain)) {
      emsg = "cannot duplicate reference chain";
      return -1;
   }
   // The new chain must be deleted at destruction
   hs->Options |= kOptsDelChn;

   // Get hook to parsing function
   XrdCryptoX509ParseBucket_t ParseBucket = sessionCF->X509ParseBucket();
   if (!ParseBucket) {
      emsg = "cannot attach to ParseBucket function!";
      return -1;
   }
   // Parse bucket
   int nci = (*ParseBucket)(bck, hs->Chain);
   if (nci != 1) {
      emsg += nci;
      emsg += " vs 1 expected)";
      return -1;
   }
   //
   // Verify the chain
   x509ChainVerifyOpt_t vopt = { 0, hs->TimeStamp, -1, hs->Crl};
   XrdCryptoX509Chain::EX509ChainErr ecode = XrdCryptoX509Chain::kNone;
   if (!(hs->Chain->Verify(ecode, &vopt))) {
      emsg = "certificate chain verification failed: ";
      emsg += hs->Chain->LastError();
      return -1;
   }
   //
   // Verify server identity
   if (!ServerCertNameOK(hs->Chain->End()->Subject(), emsg)) {
      return -1;
   }
   //
   // Extract the server key
   sessionKver = sessionCF->RSA(*(hs->Chain->End()->PKI()));
   if (!sessionKver || !sessionKver->IsValid()) {
      emsg = "server certificate contains an invalid key";
      return -1;
   }

   // Deactivate what not needed any longer
   br->Deactivate(kXRS_puk);
   br->Deactivate(kXRS_x509);

   //
   // Extract list of MD algorithms supported by the server
   String md = "";
   if ((bck = br->GetBucket(kXRS_md_alg))) {
      String mdlist;
      bck->ToString(mdlist);
      // Parse the list
      int from = 0;
      while ((from = mdlist.tokenize(md, from, ':')) != -1) {
         if (md.length() > 0) 
            if (sessionCF->SupportedMsgDigest(md.c_str()))
               break;
         md = "";
      }
   } else {
      DEBUG("WARNING: list of digests supported by server missing"
            " - using default");
      md = "md5";
   }
   if (!(sessionMD = sessionCF->MsgDigest(md.c_str()))) {
      emsg = "could not instantiate digest object";
      return -1;
   }
   // Communicate choice to server
   br->UpdateBucket(md, kXRS_md_alg);

   //
   // Extract the main buffer (it contains the random challenge
   // and will contain our credentials encrypted)
   XrdSutBucket *bckm = 0;
   if (!(bckm = br->GetBucket(kXRS_main))) {
      emsg = "main buffer missing";
      return -1;
   }

   //
   // Deserialize main buffer
   if (!((*bm) = new XrdSutBuffer(bckm->buffer,bckm->size))) {
      emsg = "error deserializing main buffer";
      return -1;
   }

   //
   // And we are done;
   return 0;
}

//_________________________________________________________________________
int XrdSecProtocolgsi::ClientDoPxyreq(XrdSutBuffer *br, XrdSutBuffer **bm,
                                      String &emsg)
{
   // Client side: process a kXGS_pxyreq message.
   // Return 0 on success, -1 on error. If the case, a message is returned
   // in cmsg.
   XrdSutBucket *bck = 0;

   //
   // Extract the main buffer (it contains the random challenge
   // and will contain our credentials encrypted)
   XrdSutBucket *bckm = 0;
   if (!(bckm = br->GetBucket(kXRS_main))) {
      emsg = "main buffer missing";
      return -1;
   }
   //
   // Decrypt the main buffer with the session cipher, if available
   if (sessionKey) {
      if (!(sessionKey->Decrypt(*bckm))) {
         emsg = "error   with session cipher";
         return -1;
      }
   }

   //
   // Deserialize main buffer
   if (!((*bm) = new XrdSutBuffer(bckm->buffer,bckm->size))) {
      emsg = "error deserializing main buffer";
      return -1;
   }

   //
   // Check if we are ready to proces this
   if ((hs->Options & kOptsFwdPxy)) {
      // We have to send the private key of our proxy
      XrdCryptoX509 *pxy = 0;
      XrdCryptoRSA *kpxy = 0;
      if (!(hs->PxyChain) ||
          !(pxy = hs->PxyChain->End()) || !(kpxy = pxy->PKI())) {
         emsg = "local proxy info missing or corrupted";
         return 0;
      }
      // Send back the signed request as bucket
      String pri;
      if (kpxy->ExportPrivate(pri) != 0) {
         emsg = "problems exporting private key";
         return 0;
      }
      // Add it to the main list
      if ((*bm)->AddBucket(pri, kXRS_x509) != 0) {
         emsg = "problem adding bucket with private key to main buffer";
         return 0;
      }
   } else {
      // Proxy request: check if we are allowed to sign it
      if (!(hs->Options & kOptsSigReq)) {
         emsg = "Not allowed to sign proxy requests";
         return 0;
      }
      // Get the request
      if (!(bck = (*bm)->GetBucket(kXRS_x509_req))) {
         emsg = "bucket with proxy request missing";
         return 0;
      }
      XrdCryptoX509Req *req = sessionCF->X509Req(bck);
      if (!req) {
         emsg = "could not resolve proxy request";
         return 0;
      }
      req->SetVersion(hs->RemVers);
      // Get our proxy and its private key
      XrdCryptoX509 *pxy = 0;
      XrdCryptoRSA *kpxy = 0;
      if (!(hs->PxyChain) ||
          !(pxy = hs->PxyChain->End()) || !(kpxy = pxy->PKI())) {
         emsg = "local proxy info missing or corrupted";
         return 0;
      }
      // Sign the request
      XrdCryptoX509 *npxy = 0;
      if (XrdSslgsiX509SignProxyReq(pxy, kpxy, req, &npxy) != 0) {
         emsg = "problems signing the request";
         return 0;
      }
      // Send back the signed request as bucket
      if ((bck = npxy->Export())) {
         // Add it to the main list
         if ((*bm)->AddBucket(bck) != 0) {
            emsg = "problem adding signed request to main buffer";
            return 0;
         }
      }
   }

   //
   // And we are done;
   return 0;

}

//_________________________________________________________________________
int XrdSecProtocolgsi::ParseServerInput(XrdSutBuffer *br, XrdSutBuffer **bm,
                                        String &cmsg)
{
   // Parse received buffer b, extracting and decrypting the main 
   // buffer *bm and extracting the session 
   // cipher, random tag buckets and user name, if any.
   // Results used to fill the local handshake variables
   EPNAME("ParseServerInput");

   // Space for pointer to main buffer must be already allocated
   if (!br || !bm) {
      DEBUG("invalid inputs ("<<br<<","<<bm<<")");
      cmsg = "invalid inputs";
      return -1;
   }

   //
   // Get the step
   int step = br->GetStep();

   // Do the right action
   switch (step) {
      case kXGC_certreq:
         // Process message
         if (ServerDoCertreq(br, bm, cmsg) != 0)
            return -1;
         break;
      case kXGC_cert:
         // Process message
         if (ServerDoCert(br, bm, cmsg) != 0)
            return -1;
         break;
      case kXGC_sigpxy:
         // Process message
         if (ServerDoSigpxy(br, bm, cmsg) != 0)
            return -1;
         break;
      default:
         cmsg = "protocol error: unknown action: "; cmsg += step;
         return -1;
         break;
   }

   //
   // We are done
   return 0;
}

//_________________________________________________________________________
int XrdSecProtocolgsi::ServerDoCertreq(XrdSutBuffer *br, XrdSutBuffer **bm,
                                       String &cmsg)
{
   // Server side: process a kXGC_certreq message.
   // Return 0 on success, -1 on error. If the case, a message is returned
   // in cmsg.
   XrdSutBucket *bck = 0;
   XrdSutBucket *bckm = 0;

   //
   // Extract the main buffer 
   if (!(bckm = br->GetBucket(kXRS_main))) {
      cmsg = "main buffer missing";
      return -1;
   }
   //
   // Extract bucket with crypto module
   if (!(bck = br->GetBucket(kXRS_cryptomod))) {
      cmsg = "crypto module specification missing";
      return -1;
   }
   String cmod;
   bck->ToString(cmod);
   // Parse the list loading the first we can
   if (ParseCrypto(cmod) != 0) {
      cmsg = "cannot find / load crypto requested module :";
      cmsg += cmod;
      return -1;
   }
   //
   // Get version run by client, if there
   if (br->UnmarshalBucket(kXRS_version,hs->RemVers) != 0) {
      hs->RemVers = Version;
      cmsg = "client version information not found in options:"
             " assume same as local";
   } else {
      br->Deactivate(kXRS_version);
   }
   //
   // Extract bucket with client issuer hash
   if (!(bck = br->GetBucket(kXRS_issuer_hash))) {
      cmsg = "client issuer hash missing";
      return -1;
   }
   String cahash;
   bck->ToString(cahash);
   //
   // Check if we know it
   if (ParseCAlist(cahash) != 0) {
      cmsg = "unknown CA: cannot verify client credentials";
      return -1;
   }
   // Find our certificate in cache
   XrdSutPFEntry *cent = 0;
   if (!(cent = cacheCert.Get(sessionCF->Name()))) {
      cmsg = "cannot find certificate: corruption?";
      return -1;
   }
   // Check validity and run renewal for proxies or fail
   if (cent->mtime < hs->TimeStamp) {
      if (cent->status == kPFE_special) {
         // Try init proxies
         ProxyIn_t pi = {SrvCert.c_str(), SrvKey.c_str(), CAdir.c_str(),
                         UsrProxy.c_str(), PxyValid.c_str(), 0, 512};
         X509Chain *ch = 0;
         XrdCryptoRSA *k = 0;
         XrdSutBucket *b = 0;
         ProxyOut_t po = {ch, k, b };
         if (QueryProxy(0, &cacheCert, sessionCF->Name(),
                        sessionCF, hs->TimeStamp, &pi, &po) != 0) {
            cmsg = "proxy expired and cannot be renewed";
            return -1;
         }
      } else {
         cmsg = "certificate has expired - go and get a new one";
         return -1;
      }
   }

   // Fill some relevant handshake variables
   sessionKsig = sessionCF->RSA(*((XrdCryptoRSA *)(cent->buf2.buf)));
   hs->Cbck = (XrdSutBucket *)(cent->buf3.buf);

   // Create a handshake cache 
   if (!(hs->Cref = new XrdSutPFEntry(hs->ID.c_str()))) {
      cmsg = "cannot create cache entry";
      return -1;
   }
   //
   // Deserialize main buffer
   if (!((*bm) = new XrdSutBuffer(bckm->buffer,bckm->size))) {
      cmsg = "error deserializing main buffer";
      return -1;
   }

   // Deactivate what not need any longer
   br->Deactivate(kXRS_issuer_hash);

   //
   // Get options, if any
   if (br->UnmarshalBucket(kXRS_clnt_opts, hs->Options) == 0)
      br->Deactivate(kXRS_clnt_opts);

   //
   // Deserialize main buffer
   if (!((*bm) = new XrdSutBuffer(bckm->buffer,bckm->size))) {
      cmsg = "error deserializing main buffer";
      return -1;
   }

   // We are done
   return 0;
}

//_________________________________________________________________________
int XrdSecProtocolgsi::ServerDoCert(XrdSutBuffer *br,  XrdSutBuffer **bm,
                                    String &cmsg)
{
   // Server side: process a kXGC_cert message.
   // Return 0 on success, -1 on error. If the case, a message is returned
   // in cmsg.
   EPNAME("ServerDoCert");

   XrdSutBucket *bck = 0;
   XrdSutBucket *bckm = 0;

   //
   // Extract the main buffer 
   if (!(bckm = br->GetBucket(kXRS_main))) {
      cmsg = "main buffer missing";
      return -1;
   }
   //
   // Extract cipher algorithm chosen by the client
   String cip = "";
   if ((bck = br->GetBucket(kXRS_cipher_alg))) {
      bck->ToString(cip);
      // Parse the list
      if (DefCipher.find(cip) == -1) {
         cmsg = "unsupported cipher chosen by the client";
         hs->Chain = 0;
         return -1;
      }
      // Deactivate the bucket
      br->Deactivate(kXRS_cipher_alg);
   } else {
      DEBUG("WARNING: client choice for cipher missing"
            " - using default");
   }

   // First get the session cipher
   if ((bck = br->GetBucket(kXRS_puk))) {
      //
      // Cleanup
      SafeDelete(sessionKey);
      //
      // Prepare cipher agreement: make sure we have the reference cipher
      if (!hs->Rcip) {
         cmsg = "reference cipher missing";
         hs->Chain = 0;
         return -1;
      }
      // Prepare cipher agreement: get a copy of the reference cipher
      if (!(sessionKey = sessionCF->Cipher(*(hs->Rcip)))) {
         cmsg = "cannot get reference cipher";
         hs->Chain = 0;
         return -1;
      }
      //
      // Instantiate the session cipher 
      if (!(sessionKey->Finalize(bck->buffer,bck->size,cip.c_str()))) {
         cmsg = "cannot finalize session cipher";
         hs->Chain = 0;
         return -1;
      }
      //
      // We need it only once
      br->Deactivate(kXRS_puk);
   }
   //
   // Decrypt the main buffer with the session cipher, if available
   if (sessionKey) {
      if (!(sessionKey->Decrypt(*bckm))) {
         cmsg = "error decrypting main buffer with session cipher";
         hs->Chain = 0;
         return -1;
      }
   }
   //
   // Deserialize main buffer
   if (!((*bm) = new XrdSutBuffer(bckm->buffer,bckm->size))) {
      cmsg = "error deserializing main buffer";
      hs->Chain = 0;
      return -1;
   }
   //
   // Get version run by client, if there
   if (hs->RemVers == -1) {
      if ((*bm)->UnmarshalBucket(kXRS_version,hs->RemVers) != 0) {
         hs->RemVers = Version;
         cmsg = "client version information not found in options:"
                " assume same as local";
      } else {
        (*bm)->Deactivate(kXRS_version);
      }
   }

   //
   // Get cache entry 
   if (!hs->Cref) {
      cmsg = "session cache has gone";
      hs->Chain = 0;
      return -1;
   }
   //
   // make sure cache is not too old
   int reftime = hs->TimeStamp - TimeSkew;
   if (hs->Cref->mtime < reftime) {
      cmsg = "cache entry expired";
      SafeDelete(hs->Cref);
      hs->Chain = 0;
      return -1;
   }

   //
   // Extract the client certificate
   if (!(bck = (*bm)->GetBucket(kXRS_x509))) {
      cmsg = "client certificate missing";
      SafeDelete(hs->Cref);
      hs->Chain = 0;
      return -1;
   }
   //
   // Finalize chain: get a copy of it (we do not touch the reference)
   hs->Chain = new X509Chain(hs->Chain);
   if (!(hs->Chain)) {
      cmsg = "cannot suplicate reference chain";
      return -1;
   }
   // The new chain must be deleted at destruction
   hs->Options |= kOptsDelChn;

   // Get hook to parsing function
   XrdCryptoX509ParseBucket_t ParseBucket = sessionCF->X509ParseBucket();
   if (!ParseBucket) {
      cmsg = "cannot attach to ParseBucket function!";
      return -1;
   }
   // Parse bucket
   int nci = (*ParseBucket)(bck, hs->Chain);
   if (nci < 2) {
      cmsg = "wrong number of certificates in received bucket (";
      cmsg += nci;
      cmsg += " > 1 expected)";
      return -1;
   }
   //
   // Verify the chain
   x509ChainVerifyOpt_t vopt = { 0, hs->TimeStamp, -1, hs->Crl};
   XrdCryptoX509Chain::EX509ChainErr ecode = XrdCryptoX509Chain::kNone;
   if (!(hs->Chain->Verify(ecode, &vopt))) {
      cmsg = "certificate chain verification failed: ";
      cmsg += hs->Chain->LastError();
      return -1;
   }

   //
   // Check if there will be delegated proxies; these can be through
   // normal request+signature, or just forwarded by the client.
   // In both cases we need to save the proxy chain. If we need a 
   // request, we have to prepare it and send it back to the client.
   bool needReq =
      ((PxyReqOpts & kOptsSrvReq) && (hs->Options & kOptsSigReq)) ||
       (hs->Options & kOptsDlgPxy);
   if (needReq || (hs->Options & kOptsFwdPxy)) {
      // Create a new proxy chain
      hs->PxyChain = new X509Chain();
      // Add the current proxy
      if ((*ParseBucket)(bck, hs->PxyChain) > 1) {
         // Reorder it
         hs->PxyChain->Reorder();
         if (needReq) {
            // Create the request
            XrdCryptoX509Req *rPXp = (XrdCryptoX509Req *) &(hs->RemVers);
            XrdCryptoRSA *krPXp = 0;
            if (XrdSslgsiX509CreateProxyReq(hs->PxyChain->End(), &rPXp, &krPXp) == 0) {
               // Save key in the cache
               hs->Cref->buf4.buf = (char *)krPXp;
               // Prepare export bucket for request
               XrdSutBucket *bckr = rPXp->Export();
               // Add it to the main list
               if ((*bm)->AddBucket(bckr) != 0) {
                  SafeDelete(hs->PxyChain);
                  DEBUG("WARNING: proxy req: problem adding bucket to main buffer");
               }
            } else {
               SafeDelete(hs->PxyChain);
               DEBUG("WARNING: proxy req: problem creating request");
            }
         }
      } else {
         SafeDelete(hs->PxyChain);
         DEBUG("WARNING: proxy req: wrong number of certificates");
      }
   }

   //
   // Extract the client public key
   sessionKver = sessionCF->RSA(*(hs->Chain->End()->PKI()));
   if (!sessionKver || !sessionKver->IsValid()) {
      cmsg = "server certificate contains an invalid key";
      return -1;
   }
   // Deactivate certificate buffer 
   (*bm)->Deactivate(kXRS_x509);

   //
   // Extract the MD algorithm chosen by the client
   String md = "";
   if ((bck = br->GetBucket(kXRS_md_alg))) {
      String mdlist;
      bck->ToString(md);
      // Parse the list
      if (DefMD.find(md) == -1) {
         cmsg = "unsupported MD chosen by the client";
         return -1;
      }
      // Deactivate
      br->Deactivate(kXRS_md_alg);
   } else {
      DEBUG("WARNING: client choice for digests missing"
            " - using default");
      md = "md5";
   }
   if (!(sessionMD = sessionCF->MsgDigest(md.c_str()))) {
      cmsg = "could not instantiate digest object";
      return -1;
   }

   // We are done
   return 0;
}

//_________________________________________________________________________
int XrdSecProtocolgsi::ServerDoSigpxy(XrdSutBuffer *br,  XrdSutBuffer **bm,
                                      String &cmsg)
{
   // Server side: process a kXGC_sigpxy message.
   // Return 0 on success, -1 on error. If the case, a message is returned
   // in cmsg.
   EPNAME("ServerDoSigpxy");

   XrdSutBucket *bck = 0;
   XrdSutBucket *bckm = 0;

   //
   // Extract the main buffer 
   if (!(bckm = br->GetBucket(kXRS_main))) {
      cmsg = "main buffer missing";
      return 0;
   }
   //
   // Decrypt the main buffer with the session cipher, if available
   if (sessionKey) {
      if (!(sessionKey->Decrypt(*bckm))) {
         cmsg = "error decrypting main buffer with session cipher";
         return 0;
      }
   }
   //
   // Deserialize main buffer
   if (!((*bm) = new XrdSutBuffer(bckm->buffer,bckm->size))) {
      cmsg = "error deserializing main buffer";
      return 0;
   }

   // Get the bucket
   if (!(bck = (*bm)->GetBucket(kXRS_x509))) {
      cmsg = "buffer with requested info missing";
      // Is there a message from the client?
      if (!(bck = (*bm)->GetBucket(kXRS_message))) {
         // Yes: decode it and print it
         String m;
         bck->ToString(m);
         DEBUG("msg from client: "<<m);
         // Add it to the main message
         cmsg += " :"; cmsg += m;
      }
      return 0;
   }

   // Make sure we still have the chain
   X509Chain *pxyc = hs->PxyChain;
   if (!pxyc) {
      cmsg = "the proxy chain is gone";
      return 0;
   }

   // Action depend on the type of message
   if ((hs->Options & kOptsFwdPxy)) {
      // The bucket contains a private key to be added to the proxy
      // public key
      XrdCryptoRSA *kpx = pxyc->End()->PKI();
      if (kpx->ImportPrivate(bck->buffer, bck->size) != 0) {
         cmsg = "problems importing private key";
         return 0;
      }
   } else {
      // The bucket contains our request signed by the client
      // The full key is in the cache 
      if (!hs->Cref) {
         cmsg = "session cache has gone";
         return 0;
      }
      // Get the signed certificate
      XrdCryptoX509 *npx = sessionCF->X509(bck);
      if (!npx) {
         cmsg = "could not resolve signed request";
         return 0;
      }
      // Set full PKI
      XrdCryptoRSA *knpx = (XrdCryptoRSA *)(hs->Cref->buf4.buf);
      npx->SetPKI((XrdCryptoX509data)(knpx->Opaque()));
      // Add the new proxy ecert to the chain
      pxyc->PushBack(npx);
   }
   // Save the chain in the instance
   proxyChain = pxyc;
   hs->PxyChain = 0;
   // Notify
   if (QTRACE(Authen)) { proxyChain->Dump(); }

   //
   // Extract user login name, if any
   String user;
   if ((bck = (*bm)->GetBucket(kXRS_user))) {
      bck->ToString(user);
      (*bm)->Deactivate(kXRS_user);
   }
   if (user.length() <= 0) user = Entity.name;

   // Dump to file if required
   if ((PxyReqOpts & kOptsPxFile)) {
      if (user.length() > 0) {
         String pxfile = UsrProxy, name;
         struct passwd *pw = getpwnam(user.c_str());
         if (pw) {
            name = pw->pw_name;
         } else {
            // Get Hash of the subject
            XrdCryptoX509 *c = proxyChain->SearchBySubject(proxyChain->EECname());
            if (c) {
               name = c->SubjectHash();
            } else {
               cmsg = "proxy chain not dumped to file: could not find subject hash";
               return 0;
            }
         }
         if (XrdSutResolve(pxfile, Entity.host,
                           Entity.vorg, Entity.grps, name.c_str()) != 0) {
            DEBUG("Problems resolving templates in "<<pxfile);
            return 0;
         }
         // Replace <uid> placeholder
         if (pw && pxfile.find("<uid>") != STR_NPOS) {
            String suid; suid += (int) pw->pw_uid;
            pxfile.replace("<uid>", suid.c_str());
         }

         // Get the function
         XrdCryptoX509ChainToFile_t ctofile = sessionCF->X509ChainToFile();
         if ((*ctofile)(proxyChain,pxfile.c_str()) != 0) {
            cmsg = "problems dumping proxy chain to file ";
            cmsg += pxfile;
            return 0;
         }
      } else {
         cmsg = "proxy chain not dumped to file: entity name undefined";
         return 0;
      }
   }

   // We are done
   return 0;
}

//__________________________________________________________________
void XrdSecProtocolgsi::ErrF(XrdOucErrInfo *einfo, kXR_int32 ecode,
                             const char *msg1, const char *msg2,
                             const char *msg3)
{
   // Filling the error structure
   EPNAME("ErrF");

   char *msgv[12];
   int k, i = 0, sz = strlen("Secgsi");

   //
   // Code message, if any
   int cm = (ecode >= kGSErrParseBuffer && 
             ecode <= kGSErrError) ? (ecode-kGSErrParseBuffer) : -1;
   const char *cmsg = (cm > -1) ? gGSErrStr[cm] : 0;

   //
   // Build error message array
              msgv[i++] = (char *)"Secgsi";     //0
   if (cmsg) {msgv[i++] = (char *)": ";         //1
              msgv[i++] = (char *)cmsg;         //2
              sz += strlen(msgv[i-1]) + 2;
             }
   if (msg1) {msgv[i++] = (char *)": ";         //3
              msgv[i++] = (char *)msg1;         //4
              sz += strlen(msgv[i-1]) + 2;
             }
   if (msg2) {msgv[i++] = (char *)": ";         //5
              msgv[i++] = (char *)msg2;         //6
              sz += strlen(msgv[i-1]) + 2;
             }
   if (msg3) {msgv[i++] = (char *)": ";         //7
              msgv[i++] = (char *)msg3;         //8
              sz += strlen(msgv[i-1]) + 2;
             }

   // save it (or print it)
   if (einfo) {
      einfo->setErrInfo(ecode, (const char **)msgv, i);
   }
   if (QTRACE(Debug)) {
      char *bout = new char[sz+10];
      if (bout) {
         bout[0] = 0;
         for (k = 0; k < i; k++)
            sprintf(bout,"%s%s",bout,msgv[k]);
         DEBUG(bout);
      } else {
         for (k = 0; k < i; k++)
            DEBUG(msgv[k]);
      }
   }
}

//__________________________________________________________________
XrdSecCredentials *XrdSecProtocolgsi::ErrC(XrdOucErrInfo *einfo,
                                           XrdSutBuffer *b1,
                                           XrdSutBuffer *b2,
                                           XrdSutBuffer *b3,
                                           kXR_int32 ecode,
                                           const char *msg1,
                                           const char *msg2,
                                           const char *msg3)
{
   // Error logging client method

   // Fill the error structure
   ErrF(einfo, ecode, msg1, msg2, msg3);

   // Release buffers
   REL3(b1,b2,b3);

   // We are done
   return (XrdSecCredentials *)0;
}

//__________________________________________________________________
int XrdSecProtocolgsi::ErrS(String ID, XrdOucErrInfo *einfo,
                            XrdSutBuffer *b1, XrdSutBuffer *b2,
                            XrdSutBuffer *b3, kXR_int32 ecode,
                            const char *msg1, const char *msg2,
                            const char *msg3)
{
   // Error logging server method

   // Fill the error structure
   ErrF(einfo, ecode, msg1, msg2, msg3);

   // Release buffers
   REL3(b1,b2,b3);

   // We are done
   return kgST_error;
}

//______________________________________________________________________________
bool XrdSecProtocolgsi::CheckRtag(XrdSutBuffer *bm, String &emsg)
{
   // Check random tag signature if it was sent with previous packet
   EPNAME("CheckRtag");

   // Make sure we got a buffer
   if (!bm) {
      emsg = "Buffer not defined";
      return 0;
   }
   //
   // If we sent out a random tag check its signature
   if (hs->Cref && hs->Cref->buf1.len > 0) {
      XrdSutBucket *brt = 0;
      if ((brt = bm->GetBucket(kXRS_signed_rtag))) {
         // Make sure we got the right key to decrypt
         if (!(sessionKver)) {
            emsg = "Session cipher undefined";
            return 0;
         }
         // Decrypt it with the counter part public key
         if (sessionKver->DecryptPublic(*brt) <= 0) {
            emsg = "error decrypting random tag with public key";
            return 0;
         }
      } else {
         emsg = "random tag missing - protocol error";
         return 0;
      } 
      //
      // Random tag cross-check: content
      if (memcmp(brt->buffer,hs->Cref->buf1.buf,hs->Cref->buf1.len)) {
         emsg = "random tag content mismatch";
         SafeDelete(hs->Cref);
         // Remove: should not be checked a second time
         return 0;
      }
      //
      // Reset the cache entry but we will not use the info a second time
      memset(hs->Cref->buf1.buf,0,hs->Cref->buf1.len);
      hs->Cref->buf1.SetBuf();
      //
      // Flag successful check
      hs->RtagOK = 1;
      bm->Deactivate(kXRS_signed_rtag);
      DEBUG("Random tag successfully checked");
   } else {
      DEBUG("Nothing to check");
   }

   // We are done
   return 1;
}

//______________________________________________________________________________
int XrdSecProtocolgsi::LoadCADir(int timestamp)
{
   // Scan cadir for valid CA certificates and load them in memory
   // in cache ca.
   // Return 0 if ok, -1 if problems
   EPNAME("LoadCADir");

   // Init cache
   XrdSutCache *ca = &(XrdSecProtocolgsi::cacheCA);
   if (!ca || ca->Init(100) != 0) {
      DEBUG("problems init cache for CA info");
      return -1;
   }

   // Some global statics
   String cadir;
   int from = 0;
   while ((from = CAdir.tokenize(cadir, from, ',')) != -1) {
      if (cadir.length() <= 0) continue;

      // Open directory
      DIR *dd = opendir(cadir.c_str());
      if (!dd) {
         DEBUG("could not open directory: "<<cadir<<" (errno: "<<errno<<")");
         continue;
      }

      // Read the content
      int i = 0;
      XrdCryptoX509ParseFile_t ParseFile = 0;
      String enam(cadir.length()+100); 
      struct dirent *dent = 0;
      while ((dent = readdir(dd))) {
         // entry name
         enam = cadir + dent->d_name;
         DEBUG("analysing entry "<<enam);
         // Try to init a chain: for each crypto factory
         for (i = 0; i < ncrypt; i++) {
            X509Chain *chain = new X509Chain();
            // Get the parse function
            ParseFile = cryptF[i]->X509ParseFile();
            int nci = (*ParseFile)(enam.c_str(), chain);
            bool ok = 0;
            XrdCryptoX509Crl *crl = 0;
            // Check what we got
            if (chain && nci == 1) {
               // Verify the CA
               bool verified = VerifyCA(CACheck, chain, cryptF[i]);
               if (verified) {

                  // Get CRL, if required
                  if (CRLCheck > 0)
                     crl = LoadCRL(chain->Begin(), cryptF[i]);
                  // Apply requirements
                  if (CRLCheck < 2 || crl) {
                     if (CRLCheck < 3 ||
                        (CRLCheck == 3 && crl && !(crl->IsExpired(timestamp)))) {
                        // Good CA
                        ok = 1;
                     } else {
                        DEBUG("CRL is expired (CRLCheck: "<<CRLCheck<<")");
                     }
                  } else {
                     DEBUG("CRL is missing (CRLCheck: "<<CRLCheck<<")");
                  }
               }
            }
            //
            if (ok) {
               // Save the chain: create the tag first
               String tag(chain->Begin()->SubjectHash());
               tag += ':';
               tag += cryptID[i];
               // Add to the cache
               XrdSutPFEntry *cent = ca->Add(tag.c_str());
               if (cent) {
                  cent->buf1.buf = (char *)chain;
                  cent->buf1.len = 0;      // Just a flag
                  if (crl) {
                     cent->buf2.buf = (char *)crl;
                     cent->buf2.len = 0;      // Just a flag
                  }
                  cent->mtime = timestamp;
                  cent->status = kPFE_ok;
                  cent->cnt = 0;
               }
            } else {
               DEBUG("Entry "<<enam<<" does not contain a valid CA");
               if (chain)
                  chain->Cleanup();
               SafeDelete(chain);
               SafeDelete(crl);
            }
         }
      }
      // Close dir
      closedir(dd);
   }

   // Rehash cache
   ca->Rehash(1);

   // We are done
   return 0;
}

//______________________________________________________________________________
XrdCryptoX509Crl *XrdSecProtocolgsi::LoadCRL(XrdCryptoX509 *xca,
                                             XrdCryptoFactory *CF)
{
   // Scan crldir for a valid CRL certificate associated to CA whose
   // certificate is xca. If the CRL is found and is valid according
   // to the chosen option, return its content in a X509Crl object.
   // Return 0 in any other case
   EPNAME("LoadCRL");
   XrdCryptoX509Crl *crl = 0;

   // make sure we got what we need
   if (!xca || !CF) {
      DEBUG("Invalid inputs");
      return crl;
   }

   // Get the signing certificate
   bool verify = 1;
   XrdCryptoX509 *xcasig = xca;
   while (xcasig && strcmp(xcasig->Issuer(), xcasig->Subject())) {
      String crldir;
      int from = 0;
      while ((from = CRLdir.tokenize(crldir, from, ',')) != -1) {
         if (crldir.length() <= 0) continue;
         String casigfile = crldir + xcasig->IssuerHash();
         // Try to get the certificate
         if ((xcasig = CF->X509(casigfile.c_str()))) break;
      }
   }
   if (!xcasig) {
      verify = 0;
      if (CACheck == 2) {
         DEBUG("CA certificate to verify the signature could not be loaded - exit");
         return crl;
      } else if (CACheck == 1) {
         DEBUG("CA certificate to verify the signature could not be loaded - verification skipped");
      }
   }

   // Get the CA hash
   String cahash = xca->SubjectHash();
   // Drop the extension (".0")
   String caroot(cahash, 0, cahash.find(".0")-1);

   // The dir
   String crlext = XrdSecProtocolgsi::DefCRLext;

   String crldir;
   int from = 0;
   while ((from = CRLdir.tokenize(crldir, from, ',')) != -1) {
      if (crldir.length() <= 0) continue;
      // Add the default CRL extension and the dir
      String crlfile = crldir + caroot;
      crlfile += crlext;
      DEBUG("target file: "<<crlfile);
      // Try to init a crl
      if ((crl = CF->X509Crl(crlfile.c_str()))) {
         if (verify) {
            // Verify issuer
            if (!(strcmp(crl->Issuer(),xcasig->Subject()))) {
               // Verify signature
               if (crl->Verify(xcasig)) {
                  // Ok, we are done
                  return crl;
               }
            }
         } else {
            // Ok, we are done
            return crl;
         }
      }
      SafeDelete(crl);
   }

   // If not required, we are done
   if (CRLCheck < 2) {
      // Done
      return crl;
   }

   // If in 'required' mode, we will also try to load the CRL from the
   // information found in the CA certificate or in the certificate directory.
   // To avoid this overload, the CRL information should be installed offline, e.g. with
   // utils/getCRLcert

   // Try to retrieve it from the URI in the CA certificate, if any
   if ((crl = CF->X509Crl(xca))) {
      if (verify) {
         // Verify issuer
         if (!(strcmp(crl->Issuer(),xcasig->Subject()))) {
            // Verify signature
            if (crl->Verify(xcasig)) {
               // Ok, we are done
               return crl;
            }
         }
      } else {
         // Ok, we are done
         return crl;
      }
   }

   // Finally try the ".crl_url" file
   from = 0;
   while ((from = CRLdir.tokenize(crldir, from, ',')) != -1) {
      if (crldir.length() <= 0) continue;
      SafeDelete(crl);
      String crlurl = crldir + caroot;
      crlurl += ".crl_url";
      DEBUG("target file: "<<crlurl);
      FILE *furl = fopen(crlurl.c_str(), "r");
      if (!furl) {
         DEBUG("could not open file: "<<crlurl);
         continue;
      }
      char line[2048];
      while ((fgets(line, sizeof(line), furl))) {
         if (line[strlen(line) - 1] == '\n') line[strlen(line) - 1] = 0;
         if ((crl = CF->X509Crl(line, 1))) {
            if (verify) {
               // Verify issuer
               if (!(strcmp(crl->Issuer(),xcasig->Subject()))) {
                  // Verify signature
                  if (crl->Verify(xcasig)) {
                     // Ok, we are done
                     return crl;
                  }
               }
            } else {
               // Ok, we are done
               return crl;
            }
         }
      }
   }

   // We need to parse the full dirs: make some cleanup first
   from = 0;
   while ((from = CRLdir.tokenize(crldir, from, ',')) != -1) {
      if (crldir.length() <= 0) continue;
      SafeDelete(crl);
      // Open directory
      DIR *dd = opendir(crldir.c_str());
      if (!dd) {
         DEBUG("could not open directory: "<<crldir<<" (errno: "<<errno<<")");
         continue;
      }
      // Read the content
      struct dirent *dent = 0;
      while ((dent = readdir(dd))) {
         // Do not analyse the CA certificate
         if (!strcmp(cahash.c_str(),dent->d_name)) continue;
         // File name contain the root CA hash
         if (!strstr(dent->d_name,caroot.c_str())) continue;
         // candidate name
         String crlfile = crldir + dent->d_name;
         DEBUG("analysing entry "<<crlfile);
         // Try to init a crl
         crl = CF->X509Crl(crlfile.c_str());
         if (!crl) continue;
         if (verify) {
            // Verify issuer
            if (strcmp(crl->Issuer(),xca->Subject())) {
               SafeDelete(crl);
               continue;
            }
            // Verify signature
            if (!(crl->Verify(xca))) {
               SafeDelete(crl);
               continue;
            }
         }
         // Ok
         break;
      }
      // Close dir
      closedir(dd);
      // Are we done?
      if (crl) break;
   }

   // We are done
   return crl;
}
//______________________________________________________________________________
String XrdSecProtocolgsi::GetCApath(const char *cahash)
{
   // Look in the paths defined by CAdir for the certificate file related to
   // 'cahash', in the form <CAdir_entry>/<cahash>.0

   String path;
   String ent;
   int from = 0;
   while ((from = CAdir.tokenize(ent, from, ',')) != -1) {
      if (ent.length() > 0) {
         path = ent;
         if (!path.endswith('/'))
            path += "/";
         path += cahash;
         if (!path.endswith(".0"))
            path += ".0";
         if (!access(path.c_str(), R_OK))
            break;
      }
      path = "";
   }

   // Done
   return path;
}
//______________________________________________________________________________
bool XrdSecProtocolgsi::VerifyCA(int opt, X509Chain *cca, XrdCryptoFactory *CF)
{
   // Verify the CA in 'cca' according to 'opt':
   //   opt = 2    full check
   //         1    only if self-signed
   //         0    no check
   EPNAME("VerifyCA");

   bool verified = 0;
   XrdCryptoX509Chain::ECAStatus st = XrdCryptoX509Chain::kUnknown;
   cca->SetStatusCA(st);

   // We nust have got a chain
   if (!cca) {
      DEBUG("Invalid input ");
      return 0;
   }

   // Get the parse function
   XrdCryptoX509ParseFile_t ParseFile = CF->X509ParseFile();
   if (!ParseFile) {
      DEBUG("Cannot attach to the ParseFile function");
      return 0;
   }

   // Point to the certificate
   XrdCryptoX509 *xc = cca->Begin();
   // Is it self-signed ?
   bool self = (!strcmp(xc->IssuerHash(), xc->SubjectHash())) ? 1 : 0;
   if (!self) {
      String inam;
      if (opt == 2) {
         // We are requested to verify it
         bool notdone = 1;
         // We need to load the issuer(s) CA(s)
         XrdCryptoX509 *xd = xc;
         while (notdone) {
            inam = GetCApath(xd->IssuerHash());
            if (inam.length() <= 0) break;
            X509Chain *ch = new X509Chain();
            int ncis = (*ParseFile)(inam.c_str(), ch);
            if (ncis < 1) break;
            XrdCryptoX509 *xi = ch->Begin();
            while (xi) {
               if (!strcmp(xd->IssuerHash(), xi->SubjectHash()))
                  break;
               xi = ch->Next();
            }
            if (xi) {
               // Add the certificate to the requested CA chain
               ch->Remove(xi);
               cca->PutInFront(xi);
               SafeDelete(ch);
               // We may be over
               if (!strcmp(xi->IssuerHash(), xi->SubjectHash())) {
                  notdone = 0;
                  break;
               } else {
                  // This becomes the daughter
                  xd = xi;
               }
            } else {
               break;
            }
         }
         if (!notdone) {
            // Verify the chain
            X509Chain::EX509ChainErr e;
            verified = cca->Verify(e);
         } else {
            PRINT("CA certificate not self-signed: cannot verify integrity ("<<xc->SubjectHash()<<")");
         }
      } else {
         // Fill CA information
         cca->CheckCA(0);
         // Set OK in any case
         verified = 1;
         // Notify if some sort of check was required
         if (opt == 1) {
            DEBUG("Warning: CA certificate not self-signed:"
                  " integrity not checked, assuming OK ("<<xc->SubjectHash()<<")");
         }
      }
   } else if (CACheck > 0) {
      // Check self-signature
      verified = cca->CheckCA();
   }

   // Set the status in the chain
   st = (verified) ? XrdCryptoX509Chain::kValid : st;
   cca->SetStatusCA(st);

   // Done
   return verified;
}

//______________________________________________________________________________
int XrdSecProtocolgsi::GetCA(const char *cahash)
{
   // Gets entry for CA with hash cahash for crypt factory cryptF[ic].
   // If not found in cache, try loading from <CAdir>/<cahash>.0 .
   // Return 0 if ok, -1 if not available, -2 if CRL not ok
   EPNAME("GetCA");

   // We nust have got a CA hash
   if (!cahash) {
      DEBUG("Invalid input ");
      return -1;
   }

   // The tag
   String tag(cahash,20);
   tag += ':';
   tag += sessionCF->ID();
   DEBUG("Querying cache for tag: "<<tag);

   // Try first the cache
   XrdSutPFEntry *cent = cacheCA.Get(tag.c_str());

   // If found, we are done
   if (cent) {
      hs->Chain = (X509Chain *)(cent->buf1.buf);
      hs->Crl = (XrdCryptoX509Crl *)(cent->buf2.buf);
      return 0;
   }

   // If not, prepare the file name
   String fnam = GetCApath(cahash);
   DEBUG("trying to load CA certificate from "<<fnam);

   // Create chain
   hs->Chain = new X509Chain();
   if (!hs->Chain ) {
      DEBUG("could not create new GSI chain");
      return -1;
   }

   // Get the parse function
   XrdCryptoX509ParseFile_t ParseFile = sessionCF->X509ParseFile();
   if (ParseFile) {
      int nci = (*ParseFile)(fnam.c_str(), hs->Chain);
      bool ok = 0, verified = 0;
      if (nci == 1) {
         // Verify the CA
         verified = VerifyCA(CACheck, hs->Chain, sessionCF);

         if (verified) {
            // Get CRL, if required
            if (CRLCheck > 0)
               hs->Crl = LoadCRL(hs->Chain->Begin(), sessionCF);
            // Apply requirements
            if (CRLCheck < 2 || hs->Crl) {
               if (CRLCheck < 3 ||
                  (CRLCheck == 3 &&
                  hs->Crl && !(hs->Crl->IsExpired(hs->TimeStamp)))) {
                  // Good CA
                  ok = 1;
               } else {
                  DEBUG("CRL is expired (CRLCheck: "<<CRLCheck<<")");
               }
            } else {
               DEBUG("CRL is missing (CRLCheck: "<<CRLCheck<<")");
            }
         }
         //
         if (ok) {
            // Add to the cache
            cent = cacheCA.Add(tag.c_str());
            if (cent) {
               cent->buf1.buf = (char *)(hs->Chain);
               cent->buf1.len = 0;      // Just a flag
               if (hs->Crl) {
                  cent->buf2.buf = (char *)(hs->Crl);
                  cent->buf2.len = 0;      // Just a flag
               }
               cent->mtime = hs->TimeStamp;
               cent->status = kPFE_ok;
               cent->cnt = 0;
            }
         } else {
            return -2;
         }
      } else {
         DEBUG("certificate not found or invalid (nci: "<<nci<<", CA: "<<
               (int)(verified)<<")");
         return -1;
      }
   }

   // Rehash cache
   cacheCA.Rehash(1);

   // We are done
   return 0;
}

//______________________________________________________________________________
int XrdSecProtocolgsi::InitProxy(ProxyIn_t *pi, X509Chain *ch, XrdCryptoRSA **kp)
{
   // Invoke 'grid-proxy-init' via the shell to create a valid the proxy file
   // If the variable GLOBUS_LOCATION is defined it prepares the external shell
   // by sourcing $GLOBUS_LOCATION/etc/globus-user-env.sh .
   // Return 0 in cse of success, != 0 in any other case .
   EPNAME("InitProxy");
   int rc = 0;

   // We must be able to get an answer
   if (isatty(0) == 0 || isatty(1) == 0) {
      DEBUG("Not a tty: cannot prompt for proxies - do nothing ");
      return -1;
   }

#ifndef HASGRIDPROXYINIT
   //
   // Use internal function for proxy initialization
   //
   // Make sure we got a chain and a key to fill
   if (!ch || !kp) {
      DEBUG("chain or key container undefined");
      return -1;
   }
   //
   // Validity
   int valid = (pi->valid) ? XrdSutParseTime(pi->valid, 1) : -1;
   //
   // Options
   XrdProxyOpt_t pxopt = {pi->bits,    // bits in key
                          valid,       // duration validity in secs
                          pi->deplen}; // signature path depth
   //
   // Init now
   rc = XrdSslgsiX509CreateProxy(pi->cert, pi->key, &pxopt,
                                 ch, kp, pi->out);
#else
   // command string
   String cmd(kMAXBUFLEN);

   // Check if GLOBUS_LOCATION is defined
   if (getenv("GLOBUS_LOCATION"))
      cmd = "source $GLOBUS_LOCATION/etc/globus-user-env.sh;";

   // Add main command
   cmd += " grid-proxy-init";

   // Add user cert
   cmd += " -cert ";
   cmd += pi->cert;

   // Add user key
   cmd += " -key ";
   cmd += pi->key;

   // Add CA dir (no support for multi-dirs)
   String cdir(pi->certdir);
   cdir.erase(cdir.find(','));
   cmd += " -certdir ";
   cmd += cdir;

   // Add validity
   if (pi->valid) {
      cmd += " -valid ";
      cmd += pi->valid;
   }

   // Add number of bits in key
   if (pi->bits > 512) {
      cmd += " -bits ";
      cmd += pi->bits;
   }

   // Add depth of signature path
   if (pi->deplen > -1) {
      cmd += " -path-length ";
      cmd += pi->deplen;
   }

   // Add output proxy coordinates
   if (pi->out) {
      cmd += " -out ";
      cmd += pi->out;
   }
   // Notify
   DEBUG("executing: " << cmd);

   // Execute
   rc = system(cmd.c_str());
   DEBUG("return code: "<< rc << " (0x"<<(int *)rc<<")");
#endif

   // We are done
   return rc;
}

//__________________________________________________________________________
int XrdSecProtocolgsi::ParseCAlist(String calist)
{
   // Parse received ca list, find the first available CA in the list
   // and return a chain initialized with such a CA.
   // If nothing found return 0.
   EPNAME("ParseCAlist");

   // Check inputs
   if (calist.length() <= 0) {
      DEBUG("nothing to parse");
      return -1;
   }
   DEBUG("parsing list: "<<calist);
 
   // Load module and define relevant pointers
   hs->Chain = 0;
   String cahash = "";
   // Parse list
   if (calist.length()) {
      int from = 0;
      while ((from = calist.tokenize(cahash, from, '|')) != -1) {
         // Check this hash
         if (cahash.length()) {
            // Get the CA chain
            if (GetCA(cahash.c_str()) == 0)
               return 0;
         }
      }
   }

   // We did not find it
   return -1;
}

//__________________________________________________________________________
int XrdSecProtocolgsi::ParseCrypto(String clist)
{
   // Parse crypto list clist, extracting the first available module
   // and getting a related local cipher and a related reference
   // cipher to be used to agree the session cipher; the local lists
   // crypto info is updated, if needed
   // The results are used to fill the handshake part of the protocol
   // instance.
   EPNAME("ParseCrypto");

   // Check inputs
   if (clist.length() <= 0) {
      DEBUG("empty list: nothing to parse");
      return -1;
   }
   DEBUG("parsing list: "<<clist);
 
   // Load module and define relevant pointers
   hs->CryptoMod = "";

   // Parse list
   int from = 0;
   while ((from = clist.tokenize(hs->CryptoMod, from, '|')) != -1) {
      // Check this module
      if (hs->CryptoMod.length() > 0) {
         DEBUG("found module: "<<hs->CryptoMod);
         // Load the crypto factory
         if ((sessionCF = 
              XrdCryptoFactory::GetCryptoFactory(hs->CryptoMod.c_str()))) {
            sessionCF->SetTrace(GSITrace->What);
            int fid = sessionCF->ID();
            int i = 0;
            // Retrieve the index in local table
            while (i < ncrypt) {
               if (cryptID[i] == fid) break;
               i++;
            }
            if (i >= ncrypt) {
               if (ncrypt == XrdCryptoMax) {
                  DEBUG("max number of crypto slots reached - do nothing");
                  return 0;
               } else {
                  // Add new entry
                  cryptF[i] = sessionCF;
                  cryptID[i] = fid;
                  ncrypt++;
               }
            }
            // On servers the ref cipher should be defined at this point
            hs->Rcip = refcip[i];
            // we are done
            return 0;
         }
      }
   }

   // Nothing found
   return -1;
}

//__________________________________________________________________________
int XrdSecProtocolgsi::QueryProxy(bool checkcache, XrdSutCache *cache,
                                  const char *tag, XrdCryptoFactory *cf,
                                  int timestamp, ProxyIn_t *pi, ProxyOut_t *po)
{
   // Query users proxies, initializing if needed
   EPNAME("QueryProxy");

   bool hasproxy = 0;
   // We may already loaded valid proxies
   XrdSutPFEntry *cent = 0;
   if (checkcache) {
      cent = cache->Get(tag);
      if (cent && cent->buf1.buf) {
         //
         po->chain = (X509Chain *)(cent->buf1.buf);
         // Check validity of the entry found (it may have expired)
         if (po->chain->CheckValidity(1, timestamp) == 0) {
            po->ksig = (XrdCryptoRSA *)(cent->buf2.buf);
            po->cbck = (XrdSutBucket *)(cent->buf3.buf);
            hasproxy = 1;
            return 0;
         } else {
            // Cleanup the chain
            po->chain->Cleanup();
            // Cleanup cache entry
            cent->buf1.buf = 0;
            cent->buf1.len = 0;
            // The key is deleted by the certificate destructor
            // Just reset the buffer
            cent->buf2.buf = 0;
            cent->buf2.len = 0;
            // and the related bucket
            if (cent->buf3.buf)
               delete (XrdSutBucket *)(cent->buf3.buf);
            cent->buf3.buf = 0;
            cent->buf3.len = 0;
         }
      }
   }

   //
   // We do not have good proxies, try load (user may have initialized
   // them in the meanwhile)
   // Create a new chain first, if needed
   if (!(po->chain))
      po->chain = new X509Chain();
   if (!(po->chain)) {
      DEBUG("cannot create new chain!");
      return -1;
   }
   int ntry = 3;
   bool parsefile = 1;
   bool exportbucket = 0;
   XrdCryptoX509ParseFile_t ParseFile = 0;
   XrdCryptoX509ParseBucket_t ParseBucket = 0;
   while (!hasproxy && ntry > 0) {

      // Try init as last option
      if (ntry == 1) {

         // Cleanup the chain
         po->chain->Cleanup();

         if (InitProxy(pi, po->chain, &(po->ksig)) != 0) {
            DEBUG("problems initializing proxy via external shell");
            ntry--;
            continue;
         }
         // We need to explicitely export the proxy in a bucket
         exportbucket = 1;
#ifndef HASGRIDPROXYINIT
         // Chain is already loaded if we used the internal function
         // to initialize the proxies
         parsefile = 0;
         timestamp = (int)(time(0));
#endif
      }
      ntry--;

      //
      // A proxy chain may have been passed via XrdSecCREDS: check that first
      if (ntry == 2) {

         char *cbuf = getenv("XrdSecCREDS");
         if (cbuf) {
            // Import into a bucket
            po->cbck = new XrdSutBucket(0, 0, kXRS_x509);
            // Fill bucket
            po->cbck->SetBuf(cbuf, strlen(cbuf));
            // Parse the bucket
            if (!(ParseBucket = cf->X509ParseBucket())) {
               DEBUG("cannot attach to ParseBucket function!");
               continue;
            }
            int nci = (*ParseBucket)(po->cbck, po->chain);
            if (nci < 2) {
               DEBUG("proxy bucket must have at least two certificates"
                     " (found: "<<nci<<")");
               continue;
            }
         } else {
            // No env: parse the file
            ntry--;
         }
      }
      if (ntry == 1) {
         if (parsefile) {
            if (!ParseFile) {
               if (!(ParseFile = cf->X509ParseFile())) {
                  DEBUG("cannot attach to ParseFile function!");
                  continue;
               }
            }
            // Parse the proxy file
            int nci = (*ParseFile)(pi->out, po->chain);
            if (nci < 2) {
               DEBUG("proxy files must have at least two certificates"
                     " (found: "<<nci<<")");
               continue;
            }
            // Check if any CA was in the file
            bool checkselfsigned = (CACheck > 1) ? 1 : 0;
            po->chain->CheckCA(checkselfsigned);
            exportbucket = 1;
         }
      }

      // Check validity in time
      if (po->chain->CheckValidity(1, timestamp) != 0) {
         DEBUG("proxy files contains expired certificates");
         continue;
      }

      // Reorder chain
      if (po->chain->Reorder() != 0) {
         DEBUG("proxy files contains inconsistent certificates");
         continue;
      }

      // Check key
      po->ksig = po->chain->End()->PKI();
      if (po->ksig->status != XrdCryptoRSA::kComplete) {
         DEBUG("proxy files contain invalid key pair");
         continue;
      }

      XrdCryptoX509ExportChain_t ExportChain = cf->X509ExportChain();
      if (!ExportChain) {
         DEBUG("cannot attach to ExportChain function!");
         continue;
      }

      // Create bucket for export
      if (exportbucket) {
         po->cbck = (*ExportChain)(po->chain, 0);
         if (!(po->cbck)) {
            DEBUG("could not create bucket for export");
            continue;
         }
      }

      // Get attach an entry in cache
      if (!(cent = cache->Add(tag))) {
         DEBUG("could not create entry in cache");
         continue;
      }

      // Save info in cache
      cent->mtime = po->chain->End()->NotAfter(); // the expiring time
      cent->status = kPFE_special;  // distinguish from normal certs
      cent->cnt = 0;
      // The chain
      cent->buf1.buf = (char *)(po->chain);
      cent->buf1.len = 0;      // Just a flag
      // The key
      cent->buf2.buf = (char *)(po->chain->End()->PKI());
      cent->buf2.len = 0;      // Just a flag
      // The export bucket
      cent->buf3.buf = (char *)(po->cbck);
      cent->buf3.len = 0;      // Just a flag

      // Rehash cache
      cache->Rehash(1);

      // Set the positive flag
      hasproxy = 1;
   }

   // We are done
   if (!hasproxy) {
      // Some cleanup
      po->chain->Cleanup();
      SafeDelete(po->chain);
      SafeDelete(po->cbck);
      return -1;
   }
   return 0;
}

//__________________________________________________________________________
int XrdSecProtocolgsi::LoadGMAP(int now)
{
   // Load cache for gridmap entries with current content of the gridmap file.
   // The cache content is loaded only if the file was modified since last
   // access.
   // Returns 0 if successful, -1 if somethign went wrong
   EPNAME("LoadGMAP");

   // Time of last check
   static int lastCheck = -1;

   // We need a file to load
   if (GMAPFile.length() <= 0)
      return 0;

   // Get info about last time of modification
   struct stat st;
   if (stat(GMAPFile.c_str(), &st) != 0) {
      PRINT("error 'stat'-ing file "<<GMAPFile);
      return -1;
   }

   // Check against current time
   if (lastCheck > st.st_mtime)
      // Nothing to do
      return 0;

   // Init or reset the cache
   if (cacheGMAP.Empty()) {
      if (cacheGMAP.Init(100) != 0) {
         PRINT("error initializing cache");
         return -1;
      }
   } else {
      if (cacheGMAP.Reset() != 0) {
         PRINT("error resetting cache");
         return -1;
      }
   }

   // Open the file
   FILE *fm = fopen(GMAPFile.c_str(),"r");
   if (!fm) {
      PRINT("error opening file "<<GMAPFile);
      return -1;
   }

   // Read entries now
   char line[2048] = {0};
   while (fgets(line,sizeof(line),fm)) {
      // Skip comment line
      if (line[0] == '#') continue;
      // Get rid of \n
      if (line[strlen(line)-1] == '\n')
         line[strlen(line)-1] = 0;
      // Extract DN
      char *p0 = (line[0] == '"') ? &line[1] : &line[0];
      int l0 = 0;
      while (p0[l0] != '"')
          l0++;
      String udn(p0, l0);
      p0 = (p0 + l0 + 1);
      while (*p0 == ' ')
          p0++;

      // Extract username
      String usr(p0);

      // Notify
      DEBUG("Found: udn: "<<udn<<", usr: "<<usr);

      // Ok: save it into the cache
      XrdSutPFEntry *cent = cacheGMAP.Add(udn.c_str());
      if (cent) {
         cent->status = kPFE_ok;
         cent->cnt = 0;
         cent->mtime = now; // creation time
         // Add username
         SafeFree(cent->buf1.buf);
         cent->buf1.buf = strdup(usr.c_str());
         cent->buf1.len = usr.length();
      }
   }
   fclose(fm);

   // Rehash cache
   cacheGMAP.Rehash(1);

   // Save the time
   lastCheck = now;

   // We are done
   return 0;
}

//__________________________________________________________________________
void XrdSecProtocolgsi::QueryGMAP(XrdCryptoX509Chain *chain, int now, String &usrs)
{
   // Resolve usernames associated with this proxy. The lookup is typically
   // based on the 'dn' (either in the grid mapfile or via the 'GMAPFun' plugin) but
   // it can also be based on the full proxy via the AuthzFun plugin.
   // For 'grid mapfile' and 'GMAPFun' the result is kept valid for a certain amount
   // of time, hashed on the 'dn'.
   // On return, an empty string in 'usrs' indicates failure.
   // Note that 'usrs' can be a comma-separated list of usernames. 
   EPNAME("QueryGMAP");

   // List of user names attached to the entity
   usrs = "";

   // The chain must be defined
   if (!chain) {
      PRINT("input chain undefined!");
      return;
   }

   // Now we check the DN-mapping function and eventually the gridmap file.
   // The result can be cached for a while. 
   XrdSutPFEntry *cent = 0;
   const char *dn = chain->EECname();
   XrdOucString s;
   const char *key = dn;
   if (GMAPFun || AuthzFun) {
      if (AuthzFun) {
         // We export the full proxy and give it to the external plugin, so
         // that it can take the decision using whatever it needs (including, e.g.,
         // VOMS information).
         XrdSutBucket *bucket = XrdCryptosslX509ExportChain(chain, true);
         bucket->ToString(s);
         delete bucket;
         key = s.c_str();
      }
      // We may have it in the cache
      cent = cacheGMAPFun.Get(key);
      // Check expiration, if required
      if (GMAPCacheTimeOut > 0 &&
         (cent && (now - cent->mtime) > GMAPCacheTimeOut)) {
         // Invalidate the entry
         cacheGMAPFun.Remove(key);
         cent = 0;
      }
      // Run the search via the external function
      if (!cent) {
         char *name;
         if (GMAPFun) {
            name = (*GMAPFun)(key, now);
         } else {
            name = (*AuthzFun)(key, now);
         }
         if ((cent = cacheGMAPFun.Add(key))) {
            if (name) {
               cent->status = kPFE_ok;
               // Add username
               SafeFree(cent->buf1.buf);
               cent->buf1.buf = name;
               cent->buf1.len = strlen(name);
            } else {
               // We cache the resul to avoid repeating the search
               cent->status = kPFE_allowed;
            }
            // Fill up the rest
            cent->cnt = 0;
            cent->mtime = now; // creation time
            // Rehash cache
            cacheGMAPFun.Rehash(1);
         }
      }
      // This means that the previous search did not return success
      if (cent && (cent->status != kPFE_ok))
         cent = 0;
   }

   // Save the result, if any
   if (cent)
      usrs = (const char *)(cent->buf1.buf);

   // Try also the map file, if any
   if (LoadGMAP(now) != 0) {
      DEBUG("error loading/ refreshing grid map file");
      return;
   }

   // Lookup for 'dn' in the cache
   cent = cacheGMAP.Get(key);

   // Add / Save the result, if any
   if (cent) {
      if (usrs.length() > 0) usrs += ",";
      usrs += (const char *)(cent->buf1.buf);
   }

   // Done
   return;
}

//_____________________________________________________________________________
XrdSecgsiGMAP_t XrdSecProtocolgsi::LoadGMAPFun(const char *plugin,
                                               const char *parms)
{
   // Load the DN-Username mapping function from the specified plug-in
   EPNAME("LoadGMAPFun");

   // Make sure the input config file is defined
   if (!plugin || strlen(plugin) <= 0) {
      PRINT("plug-in file undefined");
      return (XrdSecgsiGMAP_t)0;
   }

   // Create the plug-in instance
   if (!(GMAPPlugin = new XrdSysPlugin(&XrdSecProtocolgsi::eDest, plugin))) {
      PRINT("could not create plugin instance for "<<plugin);
      return (XrdSecgsiGMAP_t)0;
   }

   // Use global symbols?
   bool useglobals = 0;
   XrdOucString params, ps(parms), p;
   int from = 0;
   while ((from = ps.tokenize(p, from, '|')) != -1) {
      if (p == "useglobals") {
         useglobals = 1;
      } else {
         if (params.length() > 0) params += " ";
         params += p;
      }
   }
   DEBUG("params: '"<< params<<"'; useglobals: "<<useglobals);

   // Get the function
   XrdSecgsiGMAP_t ep = 0;
   if (useglobals) {
      ep = (XrdSecgsiGMAP_t) GMAPPlugin->getPlugin("XrdSecgsiGMAPFun", 0, true);
   } else {
      ep = (XrdSecgsiGMAP_t) GMAPPlugin->getPlugin("XrdSecgsiGMAPFun");
   }
   if (!ep) {
      PRINT("could not find 'XrdSecgsiGMAPFun()' in "<<plugin);
      return (XrdSecgsiGMAP_t)0;
   }

   // Init it
   if ((*ep)(params.c_str(), 0) == (char *)-1) {
      PRINT("could not initialize 'XrdSecgsiGMAPFun()'");
      return (XrdSecgsiGMAP_t)0;
   }

   // Notify
   PRINT("using 'XrdSecgsiGMAPFun()' from "<<plugin);

   // Done
   return ep;
}

//_____________________________________________________________________________
XrdSecgsiAuthz_t XrdSecProtocolgsi::LoadAuthzFun(const char *plugin,
                                                 const char *parms)
{  
   // Load the authorization function from the specified plug-in
   EPNAME("LoadAuthzFun");
   
   // Make sure the input config file is defined
   if (!plugin || strlen(plugin) <= 0) {
      PRINT("plug-in file undefined");
      return (XrdSecgsiAuthz_t)0;
   }
   
   // Create the plug-in instance
   if (!(AuthzPlugin = new XrdSysPlugin(&XrdSecProtocolgsi::eDest, plugin))) {
      PRINT("could not create plugin instance for "<<plugin);
      return (XrdSecgsiAuthz_t)0;
   }

   // Use global symbols?
   bool useglobals = 0;
   XrdOucString params, ps(parms), p;
   int from = 0;
   while ((from = ps.tokenize(p, from, '|')) != -1) {
      if (p == "useglobals") {
         useglobals = 1;
      } else {
         if (params.length() > 0) params += " ";
         params += p;
      }
   }
   DEBUG("params: '"<< params<<"'; useglobals: "<<useglobals);

   // Get the function
   XrdSecgsiAuthz_t ep = 0;
   if (useglobals)
      ep = (XrdSecgsiAuthz_t) AuthzPlugin->getPlugin("XrdSecgsiAuthzFun", 0, true);
   else
      ep = (XrdSecgsiAuthz_t) AuthzPlugin->getPlugin("XrdSecgsiAuthzFun");
   if (!ep) {
      PRINT("could not find 'XrdSecgsiAuthzFun()' in "<<plugin);
      return (XrdSecgsiAuthz_t)0;
   }
   
   // Init it
   if ((*ep)(params.c_str(), 0) == (char *)-1) {
      PRINT("could not initialize 'XrdSecgsiAuthzFun()'");
      return (XrdSecgsiAuthz_t)0;
   }
   
   // Notify
   PRINT("using 'XrdSecgsiAuthzFun()' from "<<plugin);
   
   // Done
   return ep;
}

//_____________________________________________________________________________
bool XrdSecProtocolgsi::ServerCertNameOK(const char *subject, XrdOucString &emsg)
{
   // Check that the server certificate subject name is consistent with the
   // expectations defined by the static SrvAllowedNames

   // The subject must be defined
   if (!subject || strlen(subject) <= 0) return 0;

   bool allowed = 0;
   emsg = "";

   // The server subject and its CN
   String srvsubj(subject);
   String srvcn;
   int cnidx = srvsubj.find("CN=");
   if (cnidx != STR_NPOS) srvcn.assign(srvsubj, cnidx + 3);

   // Always check if the server CN is in the standard form "[*/]<target host name>[/*]"
   if (Entity.host) {
      if (srvcn != (const char *) Entity.host) {
         int ih = srvcn.find((const char *) Entity.host);
         if (ih == 0 || (ih > 0 && srvcn[ih-1] == '/')) {
            ih += strlen(Entity.host);
            if (ih >= srvcn.length() ||
                srvcn[ih] == '\0' || srvcn[ih] == '/') allowed = 1;
         }
      } else {
         allowed = 1;
      }
      // Update the error msg, if the case
      if (!allowed) {
         if (emsg.length() <= 0) {
            emsg = "server certificate CN '"; emsg += srvcn;
            emsg += "' does not match the expected format(s):";
         }
         String defcn("[*/]"); defcn += Entity.host; defcn += "[/*]";
         emsg += " '"; emsg += defcn; emsg += "' (default)";
      }
   }

   // Take into account specif requests, if any
   if (SrvAllowedNames.length() > 0) {
      // The SrvAllowedNames string contains the allowed formats separated by a '|'.
      // The specifications can contain the <host> or <fqdn> placeholders which
      // are replaced by Entity.host; they can also contain the '*' wildcard, in
      // which case XrdOucString::matches is used. A '-' before the specification
      // will deny the matching CN's; the last matching wins.
      String allowedfmts(SrvAllowedNames);
      allowedfmts.replace("<host>", (const char *) Entity.host);
      allowedfmts.replace("<fqdn>", (const char *) Entity.host);
      int from = 0;
      String fmt;
      while ((from = allowedfmts.tokenize(fmt, from, '|')) != -1) {
         // Check if this should be denied
         bool deny = 0;
         if (fmt.beginswith("-")) {
            deny = 1;
            fmt.erasefromstart(1);
         }
         if (srvcn.matches(fmt.c_str()) > 0) allowed = (deny) ? 0 : 1;
      }
      // Update the error msg, if the case
      if (!allowed) {
         if (emsg.length() <= 0) {
            emsg = "server certificate CN '"; emsg += srvcn;
            emsg += "' does not match the expected format:";
         }
         emsg += " '"; emsg += SrvAllowedNames; emsg += "' (exceptions)";
      }
   }
   // Reset error msg, if the match was successful
   if (allowed)
      emsg = "";
   else
      emsg += "; exceptions are controlled by the env XrdSecGSISRVNAMES";

   // Done
   return allowed;
}

