/******************************************************************************/
/*                                                                            */
/*                 X r d S e c P r o t o c o l p w d . c c                    */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
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
#include <sys/times.h>

// AFS support
#ifdef R__AFS
extern "C" {
#include <afs/stds.h>
#include <afs/kautils.h>
afs_int32 ka_Authenticate(char *name, char *instance, char *cell,
                          struct ubik_client *conn, int service,
                          struct ktc_encryptionKey *key, Date start,
                          Date end, struct ktc_token *token,
                          afs_int32 * pwexpires);
afs_int32 ka_ReadPassword(char *prompt, int verify, char *cell,
                          struct ktc_encryptionKey *key);
afs_int32 ka_AuthServerConn(char *cell, int service,
                            struct ktc_token *token,
                            struct ubik_client **conn);
char     *ka_LocalCell();
void      ka_StringToKey(char *str, char *cell,
                         struct ktc_encryptionKey *key);
}
#endif

#include "XrdSys/XrdSysHeaders.hh"
#include <XrdSys/XrdSysLogger.hh>
#include <XrdSys/XrdSysError.hh>
#include <XrdOuc/XrdOucStream.hh>

#include <XrdSys/XrdSysPriv.hh>

#include <XrdSut/XrdSutCache.hh>

#include <XrdSecpwd/XrdSecProtocolpwd.hh>
#include <XrdSecpwd/XrdSecpwdPlatform.hh>

/******************************************************************************/
/*                           S t a t i c   D a t a                            */
/******************************************************************************/
  
static String Prefix   = "xrd";
static String ProtoID  = XrdSecPROTOIDENT;
static const kXR_int32 Version = XrdSecpwdVERSION;
static String AdminRef = ProtoID + "admin";
static String SrvPukRef= ProtoID + "srvpuk";
static String UserRef  = ProtoID + "user";
static String NetRcRef = ProtoID + "netrc";

static const char *pwdClientSteps[] = {
   "kXPC_none",
   "kXPC_normal",
   "kXPC_verifysrv",
   "kXPC_signedrtag",
   "kXPC_creds",
   "kXPC_autoreg",
   "kXPC_failureack",
   "kXPC_reserved"
};

static const char *pwdServerSteps[] = {
   "kXPS_none",
   "kXPS_init",
   "kXPS_credsreq",
   "kXPS_rtag",
   "kXPS_signedrtag",
   "kXPS_newpuk",
   "kXPS_puk",
   "kXPS_failure",
   "kXPS_reserved"
};

static const char *gPWErrStr[] = {
   "parsing buffer",                     // 10000
   "decoding buffer",                    // 10001
   "loading crypto factory",             // 10002
   "protocol mismatch",                  // 10003
   "resolving user / host",              // 10004
   "user missing",                       // 10005
   "host missing",                       // 10006
   "unknown user",                       // 10007
   "creating bucket",                    // 10008
   "duplicating bucket",                 // 10009
   "creating buffer",                    // 10010
   "serializing buffer",                 // 10011
   "generating cipher",                  // 10012
   "exporting public key",               // 10013
   "encrypting random tag",              // 10014
   "random tag mismatch",                // 10015
   "random tag missing",                 // 10016
   "cipher missing",                     // 10017
   "getting credentials",                // 10018
   "credentials missing",                // 10019
   "wrong password for user",            // 10020
   "checking cache",                     // 10021
   "cache entry for link missing",       // 10022
   "session handshaking ID missing",     // 10023
   "session handshaking ID mismatch",    // 10024
   "unknown step option",                // 10025
   "marshaling integer",                 // 10026
   "unmarshaling integer",               // 10027
   "saving new credentials",             // 10028
   "salt missing",                       // 10029
   "buffer empty",                       // 10030
   "obtaining reference cipher",         // 10031
   "obtaining cipher public info",       // 10032
   "adding bucket to list",              // 10033
   "finalizing cipher from public info", // 10034
   "error during initialization",        // 10035
   "wrong credentials",                  // 10035
   "error"                               // 10036    
};

// Masks for options
static const short kOptsServer  = 0x0001;
static const short kOptsUserPwd = 0x0002;
static const short kOptsAutoReg = 0x0004;
static const short kOptsAregAll = 0x0008;
static const short kOptsVeriSrv = 0x0020;
static const short kOptsVeriClt = 0x0040;
static const short kOptsClntTty = 0x0080;
static const short kOptsExpCred = 0x0100;
static const short kOptsCrypPwd = 0x0200;
static const short kOptsChngPwd = 0x0400;
static const short kOptsAFSPwd  = 0x0800;
// One day in secs
static const int kOneDay = 86400; 

/******************************************************************************/
/*                     S t a t i c   C l a s s   D a t a                      */
/******************************************************************************/
XrdSysMutex XrdSecProtocolpwd::pwdContext;
String XrdSecProtocolpwd::FileAdmin= "";
String XrdSecProtocolpwd::FileExpCreds= "";
String XrdSecProtocolpwd::FileUser = "";
String XrdSecProtocolpwd::FileCrypt= "/.xrdpass";
String XrdSecProtocolpwd::FileSrvPuk= "";
String XrdSecProtocolpwd::SrvID    = "";
String XrdSecProtocolpwd::SrvEmail = "";
String XrdSecProtocolpwd::DefCrypto= "ssl";
String XrdSecProtocolpwd::DefError = "insufficient credentials - contact ";
XrdSutPFile XrdSecProtocolpwd::PFAdmin(0);   // Admin file (server)
XrdSutPFile XrdSecProtocolpwd::PFAlog(0);   // Autologin file (client)
XrdSutPFile XrdSecProtocolpwd::PFSrvPuk(0);  // File with server public keys (client)
//
// Crypto related info
int  XrdSecProtocolpwd::ncrypt    = 0;                 // Number of factories
int  XrdSecProtocolpwd::cryptID[XrdCryptoMax] = {0};   // their IDs 
String XrdSecProtocolpwd::cryptName[XrdCryptoMax] = {0}; // their names 
XrdCryptoCipher *XrdSecProtocolpwd::refcip[XrdCryptoMax] = {0};    // ref for session ciphers 
//
// Caches for info files
XrdSutCache XrdSecProtocolpwd::cacheAdmin;  // Admin file
XrdSutCache XrdSecProtocolpwd::cacheSrvPuk; // SrvPuk file
XrdSutCache XrdSecProtocolpwd::cacheUser;   // User files
XrdSutCache XrdSecProtocolpwd::cacheAlog;   // Autologin file
//
// Running options / settings
int  XrdSecProtocolpwd::Debug       = 0; // [CS] Debug level
bool XrdSecProtocolpwd::Server      = 1; // [CS] If server mode 
int  XrdSecProtocolpwd::UserPwd     = 0; // [S] Check passwd file in user's <xrdsecdir> 
bool XrdSecProtocolpwd::SysPwd      = 0; // [S] Check passwd file in user's <xrdsecdir> 
int  XrdSecProtocolpwd::VeriClnt    = 2; // [S] Client authenticity verification level:
                                         //     0  none, 1 timestamp, 2 random tag      
int  XrdSecProtocolpwd::VeriSrv     = 1; // [C] Server authenticity verification level:
                                         //     0  none, 1 random tag      
int  XrdSecProtocolpwd::AutoReg     = kpAR_none; // [S] Autoreg mode 
int  XrdSecProtocolpwd::LifeCreds   = 0; // [S] if > 0, time interval of validity for creds
int  XrdSecProtocolpwd::MaxPrompts  = 3; // [C] Repeating prompt
int  XrdSecProtocolpwd::MaxFailures = 10;// [S] Max passwd failures before blocking
int  XrdSecProtocolpwd::AutoLogin   = 0; // [C] do-not-check/check/update autologin info
int  XrdSecProtocolpwd::TimeSkew    = 300; // [CS] Allowed skew in secs for time stamps 
bool XrdSecProtocolpwd::KeepCreds   = 0; // [S] Keep / Do-Not-Keep client creds 
//
// Debug an tracing
XrdSysError    XrdSecProtocolpwd::eDest(0, "secpwd_");
XrdSysLogger   XrdSecProtocolpwd::Logger;
XrdOucTrace   *XrdSecProtocolpwd::SecTrace = 0;

/******************************************************************************/
/*                    S t a t i c   F u n c t i o n s                         */
/******************************************************************************/
//_____________________________________________________________________________
static const char *ClientStepStr(int kclt)
{
   // Return string with client step  
   static const char *ukn = "Unknown";

   kclt = (kclt < 0) ? 0 : kclt;
   kclt = (kclt > kXPC_reserved) ? 0 : kclt;
   kclt = (kclt >= kXPC_normal) ? (kclt - kXPC_normal + 1) : kclt;

   if (kclt < 0 || kclt > (kXPC_reserved - kXPC_normal + 1))
      return ukn;  
   else
      return pwdClientSteps[kclt];
}

//_____________________________________________________________________________
static const char *ServerStepStr(int ksrv)
{
   // Return string with server step  
   static const char *ukn = "Unknown";

   ksrv = (ksrv < 0) ? 0 : ksrv;
   ksrv = (ksrv > kXPS_reserved) ? 0 : ksrv;
   ksrv = (ksrv >= kXPS_init) ? (ksrv - kXPS_init + 1) : ksrv;

   if (ksrv < 0 || ksrv > (kXPS_reserved - kXPS_init + 1))
      return ukn;  
   else
      return pwdServerSteps[ksrv];
}

/******************************************************************************/
/*       P r o t o c o l   I n i t i a l i z a t i o n   M e t h o d s        */
/******************************************************************************/


//_____________________________________________________________________________
XrdSecProtocolpwd::XrdSecProtocolpwd(int opts, const char *hname,
                                     const struct sockaddr *ipadd,
                                     const char *parms) : XrdSecProtocol("pwd")
{
   // Default constructor
   EPNAME("XrdSecProtocolpwd");

   if (QTRACE(Authen)) { PRINT("constructing: "<<this); }

   // Create instance of the handshake vars
   if ((hs = new pwdHSVars())) {
      // Update time stamp
      hs->TimeStamp = time(0);
      // Local handshake variables
      hs->CryptoMod = "";       // crypto module in use
      hs->User = "";            // remote username
      hs->Tag.resize(256);      // tag for credentials
      hs->RemVers = -1;         // Version run by remote counterpart
      hs->CF = 0;               // crypto factory
      hs->Hcip = 0;             // handshake cipher
      hs->Rcip = 0;             // reference cipher
      hs->ID = "";              // Handshake ID (dummy for clients)
      hs->Cref = 0;             // Cache reference
      hs->Pent = 0;             // Pointer to relevant file entry
      hs->RtagOK = 0;           // Rndm tag checked / not checked
      hs->Tty = (isatty(0) == 0 || isatty(1) == 0) ? 0 : 1;
      hs->Step = 0;             // Current step
      hs->LastStep = 0;         // Step required at previous iteration
   } else {
      DEBUG("could not create handshake vars object");
   }

   // Used by servers to store forwarded credentials
   clientCreds = 0;

   // Save host name
   if (hname) {
      Entity.host = strdup(hname);
   } else {
      DEBUG("warning: host name undefined");
   }
   // Save host addr
   memcpy(&hostaddr, ipadd, sizeof(hostaddr));
   // Init client name
   CName[0] = '?'; CName[1] = '\0';

   //
   // Notify, if required
   DEBUG("constructing: host: "<<hname);
   DEBUG("p: "<<XrdSecPROTOIDENT<<", plen: "<<XrdSecPROTOIDLEN);
   //
   // basic settings
   options  = opts;

   //
   // Mode specific initializations
   if (Server) {
      srvMode = 1;
      DEBUG("mode: server");
   } else {
      srvMode = 0;
      DEBUG("mode: client");
      if (AutoLogin > 0) {
         DEBUG("using autologin file: "<<PFAlog.Name());
         if (AutoLogin > 1) {
            DEBUG("running in update-autologin mode");
         }
      }
      if (VeriSrv > 0) {
         DEBUG("server verification ON");
      } else {
         DEBUG("server verification OFF");
      }
      // Decode received buffer
      if (parms) {
         XrdOucString p("&P=pwd,");
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
char *XrdSecProtocolpwd::Init(pwdOptions opt, XrdOucErrInfo *erp)
{
   // Static method to the configure the static part of the protocol
   // Called once by XrdSecProtocolpwdInit
   EPNAME("Init");
   char *Parms = 0;
   //
   // Debug an tracing
   Debug = (opt.debug > -1) ? opt.debug : Debug;
   // Initiate error logging and tracing
   eDest.logger(&Logger);
   SecTrace    = new XrdOucTrace(&eDest);
   // Set debug mask ... also for auxilliary libs
   int trace = 0;
   if (Debug >= 3) {
      trace = cryptoTRACE_Dump;
      SecTrace->What |= TRACE_Authen;
      SecTrace->What |= TRACE_Debug;
   } else if (Debug >= 1) {
      trace = cryptoTRACE_Debug;
      SecTrace->What = TRACE_Debug;
   }
   // ... also for auxilliary libs
   XrdSutSetTrace(trace);
   XrdCryptoSetTrace(trace);

   // Get user info
   struct passwd *pw = getpwuid(getuid());
   if (!pw) {
      PRINT("no user info available - invalid ");
      ErrF(erp, kPWErrInit, "could not get user info from getpwuid");
      return Parms;
   }

   //
   // Operation mode
   Server = (opt.mode == 's');

   //
   // Directory with admin pwd files
   bool argdir = 0;
   String infodir(512);
   if (opt.dir) {
      infodir = opt.dir;
      // Expand
      if (XrdSutExpand(infodir) != 0) {
         PRINT("cannot expand "<<opt.dir);
         infodir = "";
      }
      argdir = 1;
   } else {
      // use default dir $(HOME)/.<prefix>
      infodir = XrdSutHome();
      infodir += ("/." + Prefix);
   }
   if (!infodir.endswith("/")) infodir += "/";
   //
   // If defined, check existence of the infodir and admin file
   if (infodir.length()) {
      // Acquire the privileges, if needed
      XrdSysPrivGuard priv(pw->pw_uid, pw->pw_gid);
      if (priv.Valid()) {
         struct stat st;
         if (stat(infodir.c_str(),&st) == -1) {
            if (errno == ENOENT) {
               if (argdir) {
                  DEBUG("infodir non existing: "<<infodir.c_str());
               } else {
                  DEBUG("creating infodir: "<<infodir.c_str());
                  if (XrdSutMkdir(infodir.c_str(),0777) != 0) {
                     DEBUG("cannot create infodir (errno: "<<errno<<")");
                     infodir = "";
                     argdir = 0;
                  }
               }
            } else {
               DEBUG("cannot stat infodir "<<infodir<<" (errno: "<<errno<<")");
               infodir = "";
               argdir = 0;
            }
         }
      }
   }
   DEBUG("using infodir: "<<infodir.c_str());

   //
   // Server specific options
   if (Server) {
      //
      // Auto registration
      AutoReg = (opt.areg > -1) ? opt.areg : AutoReg;
      //
      // Client verification level
      VeriClnt = (opt.vericlnt > -1) ? opt.vericlnt : VeriClnt;
      //
      // Whether to check pwd files in users' $HOME
      UserPwd = (opt.upwd > -1) ? opt.upwd : UserPwd;
      //
      // Whether to check system pwd files (if allowed)
      SysPwd = (opt.syspwd > -1) ? opt.syspwd : SysPwd;
      if (SysPwd) {
         // Make sure this setting makes sense
         if (pw) {
#ifndef R__AFS
#ifdef R__SHADOWPW
            // Acquire the privileges, if needed
            XrdSysPrivGuard priv((uid_t) 0, (gid_t) 0);
            if (priv.Valid()) {
               // System V Rel 4 style shadow passwords
               struct spwd *spw = getspnam(pw->pw_name);
               if (!spw) {
                  SysPwd = 0;
                  DEBUG("no privileges to access shadow passwd file");
               }
            } else {
               DEBUG("problems acquiring credentials"
                     " to access the system password file");
            }
#else
            // Normal passwd file
            if (!pw->pw_passwd &&
                (pw->pw_passwd && strlen(pw->pw_passwd) <= 1)) {
               SysPwd = 0;
               DEBUG("no privileges to access system passwd file");
            }
#endif
#else
            PRINT("configured with AFS support");
#endif
         } else
            SysPwd = 0;
      }
      //
      // Credential  lifetime
      LifeCreds = (opt.lifecreds > -1) ? opt.lifecreds : LifeCreds;
      //
      // Max number of failures
      MaxFailures = (opt.maxfailures > -1) ? opt.maxfailures : MaxFailures;

      //
      // If defined, check existence of the infodir and admin file
      if (infodir.length()) {
         // Acquire the privileges, if needed
         XrdSysPrivGuard priv(pw->pw_uid, pw->pw_gid);
         if (priv.Valid()) {
            struct stat st;
            //
            // Define admin file and check its existence
            FileAdmin = infodir + AdminRef;
            if (stat(FileAdmin.c_str(),&st) == -1) {
               if (errno == ENOENT) {
                  PRINT("FileAdmin non existing: "<<FileAdmin.c_str());
               } else {
                  PRINT("cannot stat FileAdmin (errno: "<<errno<<")");
               }
               FileAdmin = "";
               if (UserPwd == 0 && !SysPwd) {
                  PRINT("no passwd info available - invalid ");
                  ErrF(erp,kPWErrInit,"could not find a valid password file");
                  return Parms;
               }
            }
            if (FileAdmin.length() > 0) {
               //
               // Load server ID
               PFAdmin.Init(FileAdmin.c_str(),0);
               if (PFAdmin.IsValid()) {
                  //
                  // Init cache for admin file
                  if (cacheAdmin.Load(FileAdmin.c_str()) != 0) {
                     PRINT("problems init cache for file admin ");
                     ErrF(erp,kPWErrError,"initializing cache for file admin");
                     return Parms;
                  }
                  if (QTRACE(Authen)) { cacheAdmin.Dump(); }
                  XrdSutPFEntry *ent = cacheAdmin.Get("+++SrvID");
                  if (ent)
                     SrvID.insert(ent->buf1.buf, 0, ent->buf1.len);
                  ent = cacheAdmin.Get("+++SrvEmail");
                  if (ent)
                     SrvEmail.insert(ent->buf1.buf, 0, ent->buf1.len);
                  // Default error message
                  DefError += SrvEmail;
               }
               DEBUG("server ID: "<<SrvID);
               DEBUG("contact e-mail: "<<SrvEmail);
            }
         }
      } else if (UserPwd == 0 && !SysPwd) {
         PRINT("no passwd info available - invalid ");
         ErrF(erp,kPWErrError,"could not find a valid password file");
         return Parms;
      }
      //
      // Init cache for user pwd information
      if (UserPwd > 0 || SysPwd) {
         if (cacheUser.Init(100) != 0) {
            PRINT("problems init cache for user pwd info"
                  " - passwd files in user accounts will not be used");
            UserPwd = 0;
         }
      }

      //
      // List of crypto modules
      String cryptlist = opt.clist ? (const char *)(opt.clist) : DefCrypto;

      // 
      // Load crypto modules
      XrdSutPFEntry ent;
      XrdCryptoFactory *cf = 0;
      String clist = cryptlist;
      if (clist.length()) {
         String ncpt = "";
         int from = 0;
         while ((from = clist.tokenize(ncpt, from, '|')) != -1) {
            if (ncpt.length() > 0) {
               // Try loading 
               if ((cf = XrdCryptoFactory::GetCryptoFactory(ncpt.c_str()))) {
                  // Add it to the list
                  cryptID[ncrypt] = cf->ID();
                  cryptName[ncrypt].insert(cf->Name(),0,strlen(cf->Name())+1);
                  cf->SetTrace(trace);
                  // Ref cipher
                  String ptag("+++SrvPuk_");
                  ptag += cf->ID();
                  if (FileAdmin.length() > 0) {
                     // Acquire the privileges, if needed
                     XrdSysPrivGuard priv(pw->pw_uid, pw->pw_gid);
                     if (priv.Valid()) {
                        if (PFAdmin.ReadEntry(ptag.c_str(),ent) <= 0) {
                           PRINT("ref cipher for module "<<ncpt<<" missing: disable");
                           cryptlist.erase(ncpt);
                        } else {
                           XrdSutBucket bck;
                           bck.SetBuf(ent.buf1.buf,ent.buf1.len);
                           if (!(refcip[ncrypt] = cf->Cipher(&bck))) {
                              PRINT("ref cipher for module "<<ncpt<<
                                    " cannot be instantiated : disable");
                              cryptlist.erase(ncpt);
                           } else {
                              ncrypt++;
                              if (ncrypt >= XrdCryptoMax) {
                                 PRINT("max number of crypto modules ("
                                        << XrdCryptoMax <<") reached ");
                                 break;
                              }
                           }
                        }
                     }
                  }
               } else {
                  PRINT("cannot instantiate crypto factory "<<ncpt);
               }
            }
         }
      }

      //
      // We need at least one valid crypto module
      if (ncrypt <= 0) {
         PRINT("could not find any valid crypto module");
         ErrF(erp,kPWErrInit,"could not find any valid crypto module");
         return Parms;
      }

      //
      // users' pwd information 
      if (UserPwd > 0) {
         FileUser = ("/" + UserRef);
         if (opt.udir) {
            FileUser.insert(opt.udir,0);
            if (FileUser[0] != '/') FileUser.insert('/',0);
         } else {
            // Use default $(HOME)/.<Prefix>
            FileUser.insert(Prefix,0);
            FileUser.insert("/.",0);
         }
         //
         // Crypt-hash file name, if requested
         if (opt.cpass) {
            UserPwd = 2;
            FileCrypt = opt.cpass;
            if (FileCrypt[0] != '/') FileCrypt.insert('/',0);
         }
      }

      //
      // Whether to save client creds
      KeepCreds = (opt.keepcreds > -1) ? opt.keepcreds : KeepCreds;
      if (KeepCreds)
         PRINT("Exporting client creds to internal buffer");

      //
      // Whether to export client creds to a file
      FileExpCreds = (opt.expcreds) ? opt.expcreds : FileExpCreds;
      if (FileExpCreds.length() > 0)
         PRINT("Exporting client creds to files "<<FileExpCreds);

      //
      // Priority option field
      String popt = "";
      if (SysPwd) {
#ifndef R__AFS
         popt += "sys";
#else
         popt += "afs";
         popt += ka_LocalCell();
#endif
      }

      //
      // Parms in the form: &P=pwd,c:<cryptomod>,v:<version>,id:<srvid>
      Parms = new char[cryptlist.length()+3+12+SrvID.length()+5+popt.length()+3];
      if (Parms) {
         if (popt.length() > 0)
            sprintf(Parms,"v:%d,id:%s,c:%s,po:%s",
                          Version,SrvID.c_str(),cryptlist.c_str(),popt.c_str());
         else
            sprintf(Parms,"v:%d,id:%s,c:%s",
                          Version,SrvID.c_str(),cryptlist.c_str());
      } else {
         PRINT("no system resources for 'Parms'");
         ErrF(erp,kPWErrInit,"no system resources for 'Parms'");
      }

      // Some notification
      DEBUG("using FileAdmin: "<<FileAdmin);
      DEBUG("server ID: "<<SrvID);
      DEBUG("contact e-mail: "<<SrvEmail);
      DEBUG("auto-registration mode: "<<AutoReg);
      DEBUG("verify client mode: "<<VeriClnt);
      DEBUG("available crypto modules: "<<cryptlist);
      if (UserPwd > 0) {
         DEBUG("using private pwd files: $(HOME)"<<FileUser);
         if (UserPwd > 1) {
            DEBUG("using private crypt-hash files: $(HOME)"<<FileCrypt);
         }
      }
      if (SysPwd) {
#ifndef R__AFS
         DEBUG("using system pwd information");
#else
         DEBUG("using AFS information");
#endif
      }
      if (KeepCreds) {
         DEBUG("client credentials will be kept");
      }
   }

   //
   // Client specific options
   if (!Server) {
      //
      // Server verification level
      VeriSrv = (opt.verisrv > -1) ? opt.verisrv : VeriSrv;
      //
      // Server puks file
      FileSrvPuk = "";
      if (opt.srvpuk) {
         FileSrvPuk = opt.srvpuk;
         if (XrdSutExpand(FileSrvPuk) != 0) {
            PRINT("cannot expand "<<opt.srvpuk);
            FileSrvPuk = "";
         }
      }
      //
      // If not defined, use default
      if (FileSrvPuk.length() <= 0 && infodir.length() > 0)
         FileSrvPuk = infodir + SrvPukRef;

      if (FileSrvPuk.length() > 0) {
         kXR_int32 openmode = 0; 
         struct stat st;
         //
         if (stat(FileSrvPuk.c_str(),&st) == -1) {
            if (errno == ENOENT) {
               PRINT("server public key file "<<FileSrvPuk<<" non existing: creating");
               openmode = kPFEcreate;
               // Make sure that the dir path exists
               XrdOucString dir = FileSrvPuk;
               dir.erase(dir.rfind('/')+1);
               DEBUG("asserting dir: "<<dir);
               if (XrdSutMkdir(dir.c_str(),0777) != 0) {
                  DEBUG("cannot create dir for srvpuk(errno: "<<errno<<")");
                  ErrF(erp,kPWErrInit,"cannot create dir for server public key file- exit");
                  return Parms;
               }
            } else {
               PRINT("cannot stat server public key file (errno: "<<errno<<")");
               FileSrvPuk = "";
               PRINT("server public key file invalid - exit");
               ErrF(erp,kPWErrInit,"server public key file invalid - exit");
               return Parms;
            }
         }
         //
         // Load server ID
         PFSrvPuk.Init(FileSrvPuk.c_str(),openmode);
         if (PFSrvPuk.IsValid()) {
            //
            // Init cache for server puk file
            if (cacheSrvPuk.Load(FileSrvPuk.c_str()) != 0) {
               PRINT("problems init cache for server public key file ");
               ErrF(erp,kPWErrError,"initializing cache for server public key file ");
               return Parms;
            }
            if (QTRACE(Authen)) { cacheSrvPuk.Dump(); }
         } else {
            PRINT("server public key file invalid ");
            ErrF(erp,kPWErrInit,"server public key file invalid");
            return Parms;
         }
      } else {
         PRINT("server public key file undefined");
         ErrF(erp,kPWErrInit,"server public key file undefined");
         return Parms;
      }

      //
      // Whether to search for autologin information
      AutoLogin = (opt.alog > -1) ? opt.alog : AutoLogin;
      DEBUG("AutoLogin level: "<<AutoLogin);
      //
      // Max number of re-prompts (for inconsistent inputs)
      MaxPrompts = (opt.maxprompts > -1) ? opt.maxprompts : MaxPrompts;
      //
      // Attach autologin file name, if requested
      if (AutoLogin > 0) {
         bool filefound = 0;
         String fnrc(256);
         if (opt.alogfile) {
            fnrc = opt.alogfile;
            if (XrdSutExpand(fnrc) != 0) {
               PRINT("cannot expand "<<opt.alogfile);
               fnrc = "";
            }
         }
         //
         // If file name not specified ...
         if (fnrc.length() <= 0)
            // use default
            fnrc = infodir + NetRcRef;

         if (fnrc.length() > 0) {
            kXR_int32 openmode = 0; 
            struct stat st;
            if (stat(fnrc.c_str(),&st) == -1) {
               if (errno == ENOENT) {
                  PRINT("Autologin file "<<fnrc<<" non existing: creating");
                  openmode = kPFEcreate;
               } else {
                  PRINT("cannot stat autologin file (errno: "<<errno<<")");
                  PRINT("switching off auto-login");
                  AutoLogin = 0;
               }
            }

            if (AutoLogin > 0) {
               // Attach to file
               PFAlog.Init(fnrc.c_str(),openmode);
               if (PFAlog.IsValid()) {
                  // Init cache for autologin file
                  if (cacheAlog.Load(fnrc.c_str()) == 0) {
                     if (QTRACE(Authen)) { cacheAlog.Dump(); }
                     filefound =1;
                  } else {
                     PRINT("problems init cache for autologin file");
                  }
               } else {
                  PRINT("problems attaching-to / creating autologin file");
               }
            }
         } 
         //
         // Notify if not found
         if (!filefound) {
            DEBUG("could not init properly autologin - switch off ");
            AutoLogin = 0;
         }
      }
      //
      // Notify if not found
      if (AutoLogin <= 0) {
         // Init anyhow cache to cache information during session
         if (cacheAlog.Init(100) != 0) {
            PRINT("problems init cache for user temporary autolog");
         }
      }
      // We are done
      Parms = (char *)"";
   }

   // We are done
   return Parms;
}

/******************************************************************************/
/*                                D e l e t e                                 */
/******************************************************************************/
void XrdSecProtocolpwd::Delete()
{
   // Deletes the protocol
   if (Entity.host) free(Entity.host);
   // Cleanup the handshake variables, if still there
   SafeDelete(hs);
   delete this;
}

/******************************************************************************/
/*             C l i e n t   O r i e n t e d   F u n c t i o n s              */
/******************************************************************************/
/******************************************************************************/
/*                        g e t C r e d e n t i a l s                         */
/******************************************************************************/

XrdSecCredentials *XrdSecProtocolpwd::getCredentials(XrdSecParameters *parm,
                                                     XrdOucErrInfo    *ei)
{
   // Query client for the password; remote username and host
   // are specified in 'parm'. File '.rootnetrc' is checked. 
   EPNAME("getCredentials");

   // If we are a server the only reason to be here is to get the forwarded
   // or saved client credentials
   if (srvMode) {
      XrdSecCredentials *creds = 0;
      if (clientCreds) {
         // Duplicate the buffer (otherwise it will get deleted ...)
         int sz = clientCreds->size;
         char *nbuf = (char *) malloc(sz);
         if (nbuf) {
            memcpy(nbuf, clientCreds->buffer, sz);
            creds = new XrdSecCredentials(nbuf, sz);
         }
      }
      return creds;
   }

   // Handshake vars conatiner must be initialized at this point
   if (!hs)
      return ErrC(ei,0,0,0,kPWErrError,
                  "handshake var container missing","getCredentials");
   hs->ErrMsg = "";

   //
   // Nothing to do if buffer is empty and not filled during construction
   if ((!parm && !hs->Parms) || (parm && (!(parm->buffer) || parm->size <= 0)))
      return ErrC(ei,0,0,0,kPWErrNoBuffer,"missing parameters","getCredentials");

   // Count interations
   (hs->Iter)++;

   // Update time stamp
   hs->TimeStamp = time(0);

   // Local vars
   int nextstep = 0;
   const char *stepstr = 0;
   kXR_int32 status = 0;
   char *bpub = 0;
   int lpub = 0;
   String CryptList = "";
   String Host = "";
   String RemID = "";
   String Emsg;
   String specID = "";
   // Buffer / Bucket related
   XrdSutBucket *bck    = 0;
   XrdSutBuffer *bpar   = 0;  // Global buffer
   XrdSutBuffer *bmai   = 0;  // Main buffer
   // Session status
   pwdStatus_t   SessionSt;
   memset(&SessionSt,0,sizeof(SessionSt));

   //
   // Unlocks automatically returning
   XrdSysMutexHelper pwdGuard(&pwdContext);
   //
   // Decode received buffer
   bpar = hs->Parms;
   if (!bpar && !(bpar = new XrdSutBuffer((const char *)parm->buffer,parm->size)))
      return ErrC(ei,0,0,0,kPWErrDecodeBuffer,"global",stepstr);
   // Ownership has been transferred
   hs->Parms = 0;
   //
   // Check protocol ID name
   if (strcmp(bpar->GetProtocol(),XrdSecPROTOIDENT))
      return ErrC(ei,bpar,bmai,0,kPWErrBadProtocol,stepstr);
   //
   // The step indicates what we are supposed to do
   hs->Step = (bpar->GetStep()) ? bpar->GetStep() : kXPS_init;
   stepstr = ServerStepStr(hs->Step);
   // Dump, if requested
   if (QTRACE(Authen)) {
      bpar->Dump(stepstr);
   }
   //
   // Find first crypto module to be used
   if (ParseCrypto(bpar) != 0)
      return ErrC(ei,bpar,0,0,kPWErrLoadCrypto,stepstr);
   //
   // Parse input buffer
   if (ParseClientInput(bpar, &bmai, Emsg) == -1) {
      DEBUG(Emsg);
      return ErrC(ei,bpar,bmai,0,kPWErrParseBuffer,Emsg.c_str(),stepstr);
   }
   //
   // Version
   DEBUG("version run by server: "<< hs->RemVers);
   //
   // Dump what we got
   if (QTRACE(Authen)) {
      bmai->Dump("Main IN");
   }
   //
   // Print server messages, if any
   if (hs->Iter > 1) {
      bmai->Message();
      bmai->Deactivate(kXRS_message);
   }
   //
   // Check random challenge
   if (!CheckRtag(bmai, Emsg))
      return ErrC(ei,bpar,bmai,0,kPWErrBadRndmTag,Emsg.c_str(),stepstr);

   //
   // Get the status bucket, if any
   if ((bck = bmai->GetBucket(kXRS_status))) {
      int pst = 0;
      memcpy(&pst,bck->buffer,sizeof(pwdStatus_t));
      pst = ntohl(pst);
      memcpy(&SessionSt, &pst, sizeof(pwdStatus_t));
      bmai->Deactivate(kXRS_status);
   } else {
      SessionSt.ctype = kpCT_normal;
   }
   //
   // Now action depens on the step
   nextstep = kXPC_none;
   switch (hs->Step) {

   case kXPS_init:
      //
      // Add bucket with cryptomod to the global list
      // (This must be always visible from now on)
      if (bpar->AddBucket(hs->CryptoMod,kXRS_cryptomod) != 0)
         return ErrC(ei,bpar,bmai,0,
              kPWErrCreateBucket,XrdSutBuckStr(kXRS_cryptomod),stepstr);
      //
      // Add bucket with our version to the main list
      if (bmai->MarshalBucket(kXRS_version,(kXR_int32)(Version)) != 0)
         return ErrC(ei,bpar,bmai,0, kPWErrCreateBucket,
                XrdSutBuckStr(kXRS_version),"(main list)",stepstr);
      //
      // We set some options in the option field of a pwdStatus_t structure
      if (hs->Tty || (AutoLogin > 0))
         SessionSt.options = kOptsClntTty;

   case kXPS_puk:
      // After auto-reg request, server puk have been saved in ParseClientInput:
      // we need to start a full normal login now

      //
      // If we have a session cipher we extract the public part
      // and add to the main packet for transmission to server
      if (hs->Hcip) {
         //
         // Extract buffer with public info for the cipher agreement
         if (!(bpub = hs->Hcip->Public(lpub))) 
            return ErrC(ei,bpar,bmai,0,
                        kPWErrNoPublic,"session",stepstr);
         //
         // Add it to the global list
         if (bpar->UpdateBucket(bpub,lpub,kXRS_puk) != 0)
            return ErrC(ei,bpar,bmai,0, kPWErrAddBucket,
                        XrdSutBuckStr(kXRS_puk),"global",stepstr);
         SafeDelArray(bpub);
         //
         // If we are requiring server verification of puk ownership
         // we are done for this step
         if (VeriSrv == 1) {
            nextstep = kXPC_verifysrv;
            break;
         }
      }

   case kXPS_signedrtag:     // (after kXRC_verifysrv)
      //
      // Add the username
      if (hs->User.length()) {
         if (bmai->AddBucket(hs->User,kXRS_user) != 0)
            return ErrC(ei,bpar,bmai,0, kPWErrDuplicateBucket,
                        XrdSutBuckStr(kXRS_user),stepstr);
      } else
         return ErrC(ei,bpar,bmai,0, kPWErrNoUser,stepstr);

      //
      // If we do not have a session cipher, the only thing we can 
      // try is auto-registration
      if (!(hs->Hcip)) {
         nextstep = kXPC_autoreg;
         break;
      }

      //
      // Normal attempt: add credentials
      status = kpCT_normal;
      if (hs->SysPwd == 1)
         status = kpCT_crypt;
      if (hs->SysPwd == 2)
         status = kpCT_afs;
      if (!(bck = QueryCreds(bmai, (AutoLogin > 0), status)))
         return ErrC(ei,bpar,bmai,0, kPWErrQueryCreds,
                                     hs->Tag.c_str(),stepstr);
      bmai->AddBucket(bck);
      //
      // Tell the server we want to change the password, if so
      if (hs->Pent->status == kPFE_onetime)
         SessionSt.options |= kOptsChngPwd;
      //
      nextstep = kXPC_normal;
      break;

   case kXPS_credsreq:
      //
      // If this is not the first time, during the handshake, that
      // we query credentials, any save buffer must insufficient,
      // so invalidate it
      if (hs->Pent)
         hs->Pent->cnt = 1;
      //
      // Server requires additional credentials: the status bucket
      // tells us what she wants exactly
      status = SessionSt.ctype;
      if (!(bck = QueryCreds(bmai, 0, status)))
         return ErrC(ei,bpar,bmai,0, kPWErrQueryCreds,
                                     hs->Tag.c_str(),stepstr);
      bmai->AddBucket(bck);
      //
      nextstep = kXPC_creds;
      break;

   case kXPS_failure:
      //
      // Failure: invalidate cache
      hs->Pent->buf1.SetBuf();
      hs->Pent->buf2.SetBuf();
      //
      nextstep = kXPC_failureack;
      break;

   case kXPS_newpuk:
      //
      // New server puk have been saved in ParseClientInput: we
      // just need to sign the random tag
   case kXPS_rtag:
      //
      // Not much to do: the random tag is signed in AddSerialized 
      nextstep = kXPC_signedrtag;
      break;

   default:
      return ErrC(ei,bpar,bmai,0, kPWErrBadOpt,stepstr);
   }
   //
   // Add / Update status
   int *pst = (int *) new char[sizeof(pwdStatus_t)];
   memcpy(pst,&SessionSt,sizeof(pwdStatus_t));
   *pst = htonl(*pst);
   if (bmai->AddBucket((char *)pst,sizeof(pwdStatus_t), kXRS_status) != 0) {
      DEBUG("problems adding bucket kXRS_status");
   }
   //
   // Serialize and encrypt
   if (AddSerialized('c', nextstep, hs->ID,
                     bpar, bmai, kXRS_main, hs->Hcip) != 0)
      return ErrC(ei,bpar,bmai,0,
                  kPWErrSerialBuffer,"main",stepstr);
   //
   // Serialize the global buffer
   char *bser = 0;
   int nser = bpar->Serialized(&bser,'f');

   if (QTRACE(Authen)) {
      bpar->Dump(ClientStepStr(bpar->GetStep()));
      bmai->Dump("Main OUT");
   }
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

int XrdSecProtocolpwd::Authenticate(XrdSecCredentials *cred,
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

   // Handshake vars container must be initialized at this point
   if (!hs)
      return ErrS(String("none"),ei,0,0,0,kPWErrError,
                  "handshake var container missing",
                  "protocol initialization problems");
   hs->ErrMsg = "";
   //
   // Update time stamp
   hs->TimeStamp = time(0);

   //
   // ID of this handshaking
   hs->ID = Entity.tident;
   DEBUG("handshaking ID: " << hs->ID);

   // Local vars 
   int i = 0;
   int kS_rc = kpST_more;
   int rc = 0;
   int entst = 0;
   int nextstep = 0;
   int ctype = kpCT_normal;
   char *bpub = 0, *bpid = 0;
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
   XrdSutBucket    *bck  = 0;  // Generic bucket
   // The local status info
   pwdStatus_t      SessionSt = { 0, 0, 0};

   //
   // Unlocks automatically returning
   XrdSysMutexHelper pwdGuard(&pwdContext);
   //
   // Decode received buffer
   if (!(bpar = new XrdSutBuffer((const char *)cred->buffer,cred->size)))
      return ErrS(hs->ID,ei,0,0,0,kPWErrDecodeBuffer,"global",stepstr);
   //
   // Check protocol ID name
   if (strcmp(bpar->GetProtocol(),XrdSecPROTOIDENT))
      return ErrS(hs->ID,ei,bpar,bmai,0,kPWErrBadProtocol,stepstr);
   //
   // The step indicates what we are supposed to do
   hs->Step = bpar->GetStep();
   stepstr = ClientStepStr(hs->Step);
   // Dump, if requested
   if (QTRACE(Authen)) {
      bpar->Dump(stepstr);
   }

   //
   // Find first crypto module to be used
   if (ParseCrypto(bpar) != 0)
      return ErrS(hs->ID,ei,bpar,0,0,kPWErrLoadCrypto,stepstr);
   //
   // Parse input buffer
   if (ParseServerInput(bpar, &bmai, ClntMsg) == -1) {
      DEBUG(ClntMsg);
      return ErrS(hs->ID,ei,bpar,bmai,0,kPWErrParseBuffer,ClntMsg.c_str(),stepstr);
   }
   //
   // Get handshake status
   if ((bck = bmai->GetBucket(kXRS_status))) {
      int pst = 0;
      memcpy(&pst,bck->buffer,sizeof(pwdStatus_t));
      pst = ntohl(pst);
      memcpy(&SessionSt, &pst, sizeof(pwdStatus_t));
      bmai->Deactivate(kXRS_status);
   } else {
      DEBUG("no bucket kXRS_status found in main buffer");
   }   
   hs->Tty = SessionSt.options & kOptsClntTty;
   //
   // Client name
   unsigned int ulen = hs->User.length();
   ulen = (ulen > sizeof(CName)-1) ? sizeof(CName)-1 : ulen; 
   if (ulen)
      strcpy(CName, hs->User.c_str());
   // And set link to entity
   Entity.name = strdup(CName);

   //
   // Version
   DEBUG("version run by client: "<< hs->RemVers);
   //
   // Dump, if requested
   if (QTRACE(Authen)) {
      bmai->Dump("main IN");
   }
   //
   // Check random challenge
   if (!CheckRtag(bmai, ClntMsg))
      return ErrS(hs->ID,ei,bpar,bmai,0,kPWErrBadRndmTag,stepstr,ClntMsg.c_str());
   //
   // Check also host / time stamp (it will be done only if really neede)
   if (!CheckTimeStamp(bmai, TimeSkew, ClntMsg))
      return ErrS(hs->ID,ei,bpar,bmai,0,kPWErrBadRndmTag,stepstr,ClntMsg.c_str());
   //
   // Now action depens on the step
   bool savecreds = (SessionSt.options & kOptsExpCred);
   switch (hs->Step) {

   case kXPC_verifysrv:
      //
      // Client required us to sign a random challenge: this is done
      // in AddSerialized, so nothing to do here
      nextstep = kXPS_signedrtag;
      break;

   case kXPC_signedrtag:
      //
      // Client signed the random challenge we sent: if we are here,
      // everything was fine
      kS_rc = kpST_ok;
      nextstep = kXPS_none;
      break;

   case kXPC_failureack:
      //
      // Client acknowledged failure
      kS_rc = kpST_error;
      nextstep = kXPS_none;
      break;

   case kXPC_autoreg:
      //
      // Client has lost the key or requested auto-registration: we
      // check the username: if it has a good entry or it is allowed
      // to auto-register (the check is done in QueryUser) we send
      // the public part of the key; otherwise we fail
      rc = QueryUser(entst, ClntMsg);
      if (rc < 0 || (entst == kPFE_disabled))
         return ErrS(hs->ID,ei,bpar,bmai,0, kPWErrBadCreds,
                     DefError.c_str(),stepstr);
      //
      // We have to send the public key
      for (i = 0; i < ncrypt; i++) {
         if (refcip[i]) {
            //
            // Extract buffer with public info for the cipher agreement
            if (!(bpub = refcip[i]->Public(lpub))) 
               return ErrS(hs->ID,ei,bpar,bmai,0, kPWErrNoPublic,
                                               "session",stepstr);
            bpid = new char[lpub+5];
            if (bpid) {
               char cid[5] = {0};
               sprintf(cid,"%d",cryptID[i]);
               memcpy(bpid,cid,5);
               memcpy(bpid+5, bpub, lpub);
               //
               // Add it to the global list
               if (bmai->AddBucket(bpid,lpub+5,kXRS_puk) != 0)
                  return ErrS(hs->ID,ei,bpar,bmai,0, kPWErrAddBucket,
                                                  "main",stepstr);
            } else 
               return ErrS(hs->ID,ei,bpar,bmai,0, kPWErrError,
                                             "out-of-memory",stepstr);
            SafeDelArray(bpub); // bpid is taken by the bucket
         }
      }
      // client should now go through a complete login
      nextstep = kXPS_puk;
      break;

   case kXPC_normal:
      //
      // Complete login sequence: check user and creds
      if (QueryUser(entst,ClntMsg) != 0)
         return ErrS(hs->ID,ei,bpar,bmai,0, kPWErrBadCreds,
                     ": user ",hs->User.c_str(),stepstr);
      // Nothing to do, if disabled
      if (entst == kPFE_disabled)
         return ErrS(hs->ID,ei,bpar,bmai,0, kPWErrBadCreds,
                     ": user ",hs->User.c_str(),stepstr);

      if (entst == kPFE_expired || entst == kPFE_onetime) {
         // New credentials should asked upon success first check
         SessionSt.options |= kOptsExpCred;
      }
      if (entst == kPFE_crypt) {
         // User credentials are either in crypt form (private or
         // system ones) or of AFS type; in case of failure
         // this flag allows the client to send the right creds
         // at next iteration
         if (ClntMsg.beginswith("afs:")) {
            SessionSt.options |= kOptsAFSPwd;
         } else
            SessionSt.options |= kOptsCrypPwd;
         // Reset the message
         ClntMsg = "";
      }
      // Creds, if any, should be checked, unles we allow auto-registration
      savecreds = (entst != kPFE_allowed) ? 0 : 1;

   case kXPC_creds:
      //
      // Final login sequence: extract and check creds
      // Extract credentials from main buffer
      if (!(bck = bmai->GetBucket(kXRS_creds))) {
         //
         // If credentials are missing, require them
         kS_rc = kpST_more;
         nextstep = kXPS_credsreq;
         break;
      }
      //
      // If we required new credentials at previous step, just save them
      if (savecreds) {
         if (SaveCreds(bck) != 0) {
            ClntMsg = "Warning: could not correctly update credentials database"; 
         }
         kS_rc = kpST_ok;
         nextstep = kXPS_none;
         bmai->Deactivate(kXRS_creds);
         break;
      }
      //
      // Credential type
      ctype = kpCT_normal;
      if (SessionSt.options & kOptsCrypPwd)
         ctype = kpCT_crypt;
      else if (SessionSt.options & kOptsAFSPwd) {
         ctype = kpCT_afs;
         String afsInfo;
         XrdSutBucket *bafs = bmai->GetBucket(kXRS_afsinfo);
         if (bafs)
            bafs->ToString(afsInfo);
         if (afsInfo == "c")
            ctype = kpCT_afsenc;
      }
      //
      // Check credentials
      if (!CheckCreds(bck, ctype)) {
         //
         // Count temporary failures
         (hs->Cref->cnt)++;
         // Reset expired credentials flag
         SessionSt.options &= ~kOptsExpCred;
         // Repeat if not too many attempts
         ClntMsg = DefError;
         if (hs->Cref->cnt < MaxPrompts) {
            // Set next step to credential request
            nextstep = kXPS_credsreq;
            kS_rc = kpST_more;
            // request again creds
            if (hs->Pent->status == kPFE_crypt) {
               SessionSt.ctype = kpCT_crypt;
               if (ctype == kpCT_afs || ctype == kpCT_afsenc) {
                  SessionSt.ctype = kpCT_afs;
                  String afsinfo = hs->ErrMsg;
                  bmai->UpdateBucket(afsinfo, kXRS_afsinfo);
               }
               ClntMsg = "";
            } else {
               SessionSt.ctype = kpCT_normal;
               ClntMsg = "insufficient credentials";
            }
         } else {
            // We communicate failure
            kS_rc = kpST_more;
            nextstep = kXPS_failure;
            // Count failures
            (hs->Pent->cnt)++;
            // Count failures
            hs->Pent->mtime = (kXR_int32)time(0);
            // Flush cache content to source file
            XrdSysPrivGuard priv(getuid(), getgid());
            if (priv.Valid()) {
               if (cacheAdmin.Flush() != 0) {
                  DEBUG("WARNING: some problem flushing to admin"
                        " file after updating "<<hs->Pent->name);
               }
            }
         }
      } else {
         // Reset counter for temporary failures
         hs->Cref->cnt = 0;
         // Reset counter in file if needed
         if (hs->Pent->cnt > 0) {
            hs->Pent->cnt = 0;
            // Count failures
            hs->Pent->mtime = (kXR_int32)time(0);
            // Flush cache content to source file
            XrdSysPrivGuard priv(getuid(), getgid());
            if (priv.Valid()) {
               if (cacheAdmin.Flush() != 0) {
                  DEBUG("WARNING: some problem flushing to admin"
                        " file after updating "<<hs->Pent->name);
               }
            }
         }
         kS_rc = kpST_ok;
         nextstep = kXPS_none;
         if (SessionSt.options & kOptsExpCred ||
             // Client requested a pwd change
             SessionSt.options & kOptsChngPwd) {
            kS_rc = kpST_more;
            nextstep = kXPS_credsreq;
            if (SessionSt.options & kOptsExpCred) {
               ClntMsg = "Credentials expired";
            } else if (SessionSt.options & kOptsChngPwd) {
               ClntMsg = "Password change requested";
            }
            // request new creds
            SessionSt.ctype = kpCT_new;
            // So we can save at next round
            SessionSt.options |= kOptsExpCred;
         }
         // Create buffer to keep the credentials, if required
         if (KeepCreds) {
            int sz = bck->size+5;
            char *buf = (char *) malloc(sz);
            if (buf) {
               memcpy(buf, "&pwd", 4);
               buf[4] = 0;
               memcpy(buf+5, bck->buffer, bck->size);
               // Put in hex
               char *out = new char[2*sz+1];
               XrdSutToHex(buf, sz, out);
               // Cleanup any existing info
               SafeDelete(clientCreds);
               clientCreds = new XrdSecCredentials(out, 2*sz+1);
            }
         }
         // Export creds to a file, if required
         if (FileExpCreds.length() > 0) {
            if (ExportCreds(bck) != 0)
               DEBUG("WARNING: some problem exporting creds to file;"
                     " template is :"<<FileExpCreds);
         }
      }
      // We will not use again these credentials
      bmai->Deactivate(kXRS_creds);

      break;

   default:
      return ErrS(hs->ID,ei,bpar,bmai,0, kPWErrBadOpt, stepstr);
   }

   //
   // If strong signature checking is required add random tag
   if (kS_rc == kpST_ok) {
      if (VeriClnt == 2 && !(hs->RtagOK)) {
         // Send only the random tag to sign
         nextstep = kXPS_rtag;
         kS_rc = kpST_more;
      }
   }

   //
   // If we need additional info but the client caa not reply, just fail
   if (kS_rc == kpST_more && !(hs->Tty)) {
      DEBUG("client cannot reply to additional request: failure");
      // Deactivate everything
      bpar->Deactivate(-1);
      bmai->Deactivate(-1);
      kS_rc = kpST_error;
   }
   //
   if (kS_rc == kpST_more) {
      //
      // Add message to client
      if (ClntMsg.length() > 0)
         if (bmai->AddBucket(ClntMsg,kXRS_message) != 0) {
            DEBUG("problems adding bucket with message for client");
         }
      //
      // We set some options in the option field of a pwdStatus_t structure
      int *pst = (int *) new char[sizeof(pwdStatus_t)];
      memcpy(pst,&SessionSt,sizeof(pwdStatus_t));
      *pst = htonl(*pst);
      if (bmai->AddBucket((char *)pst,sizeof(pwdStatus_t), kXRS_status) != 0) {
         DEBUG("problems adding bucket kXRS_status");
      }
      //
      // Serialize, encrypt and add to the global list
      if (AddSerialized('s', nextstep, hs->ID,
                        bpar, bmai, kXRS_main, hs->Hcip) != 0)
         return ErrS(hs->ID,ei,bpar,bmai,0, kPWErrSerialBuffer,
                     "main / session cipher",stepstr);
      //
      // Serialize the global buffer
      char *bser = 0;
      int nser = bpar->Serialized(&bser,'f');
      //
      // Dump, if requested
      if (QTRACE(Authen)) {
         bpar->Dump(ServerStepStr(bpar->GetStep()));
         bmai->Dump("Main OUT");
      }
      //
      // Create buffer for client
      *parms = new XrdSecParameters(bser,nser);
   } else {
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
char *XrdSecProtocolpwdInit(const char mode,
                            const char *parms, XrdOucErrInfo *erp)
{
   // One-time protocol initialization, filling the static flags and options
   // of the protocol.
   // For clients (mode == 'c') we use values in envs.
   // For servers (mode == 's') the command line options are passed through
   // parms.
   pwdOptions opts;
   char *rc = (char *)"";
   char *cenv = 0;

   //
   // Clients first
   if (mode == 'c') {
      //
      // Decode envs:
      //              "XrdSecDEBUG"          debug flag ("0","1","2","3")
      //              "XrdSecPWDVERIFYSRV"   "1" server verification ON [default]
      //                                     "0" server verification OFF
      //              "XrdSecPWDSRVPUK"      full path to file with server puks
      //                                     [default: $HOME/.xrd/pwdsrvpuk]
      //              "XrdSecPWDAUTOLOG"     "1" autologin ON [default]
      //                                     "0" autologin OFF
      //              "XrdSecPWDALOGFILE"    full path to file with autologin
      //                                     info [default: $HOME/.xrd/pwdnetrc]
      //              "XrdSecPWDALOGUPDT"    update autologin file option:
      //                                     "0"  never [default]
      //                                     "1"  remove_obsolete_info
      //                                     "2"  "1" + register_new_valid_info
      //              "XrdSecPWDMAXPROMPT"   max number of attemts to get valid
      //                                     input info by prompting the client
      //
      opts.mode = mode;
      // debug
      cenv = getenv("XrdSecDEBUG");
      if (cenv)
         if (cenv[0] >= 49 && cenv[0] <= 51) opts.debug = atoi(cenv);  

      // server verification
      cenv = getenv("XrdSecPWDVERIFYSRV");
      if (cenv)
         if (cenv[0] >= 48 && cenv[0] <= 49) opts.verisrv = atoi(cenv);  
      // file with server public keys
      cenv = getenv("XrdSecPWDSRVPUK");
      if (cenv)
         opts.srvpuk = strdup(cenv);  
      // autologin
      cenv = getenv("XrdSecPWDAUTOLOG");
      if (cenv)
         if (cenv[0] >= 48 && cenv[0] <= 50) opts.alog = atoi(cenv);  
      // autologin file
      cenv = getenv("XrdSecPWDALOGFILE");
      if (cenv)
         opts.alogfile = strdup(cenv);  
      // max re-prompts
      cenv = getenv("XrdSecPWDMAXPROMPT");
      if (cenv) {
         opts.maxprompts = strtol(cenv, (char **)0, 10);
         if (errno == ERANGE) opts.maxprompts = -1;
      }
      //
      // Setup the object with the chosen options
      rc = XrdSecProtocolpwd::Init(opts,erp);

      // Some cleanup
      if (opts.srvpuk) free(opts.srvpuk);
      if (opts.alogfile) free(opts.alogfile);

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
      // for servers: [-upwd:<user_pwd_option>]
      //              [-a:<autoreg_level>]
      //              [-vc:<client_verification_level>]
      //              [-dir:<dir_with_pwd_info>]
      //              [-udir:<sub_dir_with_user_pwd_info>]
      //              [-c:[-]ssl[:[-]<CryptoModuleName]]
      //              [-d:<debug_level>]
      //              [-syspwd]
      //              [-lf:<credential_lifetime>]
      //              [-maxfail:<max_number_of_failures>]
      //              [-keepcreds]
      //              [-expcreds:<creds_file_name>]
      //
      // <user_pwd_opt> = 0 (do-not-use), 1 (use), 2 (also-crypt-hash)
      // <debug_level> = 0 (none), 1 (low), 2 (medium), 3 (high)   [0]
      // <autoreg_level> = 0 (none), 1 (local users + allowed tags), 2 (all) [0]
      // <credential_lifetime> = 1d, 5h:10m, ... (see XrdSutAux::ParseTime)
      // <client_verification_level> = 0 (none), 1 (timestamp), 2 (random tag) [2]
      // <creds_file_name> = can be a fully specified path or in the templated form
      //                     /path/<user>/file, with <user> expanded at the moment
      //                     of use with the login name.
      //
      int debug = -1;
      int areg = -1;
      int vc = -1;
      int upw = -1;
      int syspwd = -1;
      int lifetime = -1;
      int maxfail = -1;
      String dir = "";
      String udir = "";
      String clist = "";
      String cpass = "";
      int keepcreds = -1;
      String expcreds = "";
      char *op = 0;
      while (inParms.GetLine()) { 
         while ((op = inParms.GetToken())) {
            if (!strncmp(op, "-upwd:",6)) {
               upw = atoi(op+6);
            } else if (!strncmp(op, "-dir:",5)) {
               dir = (const char *)(op+5);
            } else if (!strncmp(op, "-udir:",6)) {
               udir = (const char *)(op+6);
            } else if (!strncmp(op, "-c:",3)) {
               clist = (const char *)(op+3);
            } else if (!strncmp(op, "-d:",3)) {
               debug = atoi(op+3);
            } else if (!strncmp(op, "-a:",3)) {
               areg = atoi(op+3);
            } else if (!strncmp(op, "-vc:",4)) {
               vc = atoi(op+4);
            } else if (!strncmp(op, "-syspwd",7)) {
               syspwd = 1;
            } else if (!strncmp(op, "-lf:",4)) {
               lifetime = XrdSutParseTime(op+4);
            } else if (!strncmp(op, "-maxfail:",9)) {
               maxfail =  atoi(op+9);
            } else if (!strncmp(op, "-cryptfile:",11)) {
               cpass = (const char *)(op+11);
            } else if (!strncmp(op, "-keepcreds",10)) {
               keepcreds = 1;
            } else if (!strncmp(op, "-expcreds:",10)) {
               expcreds = (const char *)(op+10);
            }
         }
         // Check inputs
         areg = (areg >= 0 && areg <= 2) ? areg : 0;
         vc = (vc >= 0 && vc <= 2) ? vc : 2;
      }

      //
      // Build the option object
      opts.debug = (debug > -1) ? debug : opts.debug;
      opts.mode = 's';
      opts.areg = areg;
      opts.vericlnt = vc;
      opts.upwd = upw;
      opts.syspwd = syspwd;
      opts.lifecreds = lifetime;
      opts.maxfailures = maxfail;
      if (dir.length() > 0)
         opts.dir = (char *)dir.c_str();
      if (udir.length() > 0)
         opts.udir = (char *)udir.c_str();
      if (clist.length() > 0)
         opts.clist = (char *)clist.c_str();
      if (cpass.length() > 0)
         opts.cpass = (char *)cpass.c_str();
      opts.keepcreds = keepcreds;
      if (expcreds.length() > 0)
         opts.expcreds = (char *)expcreds.c_str();
      //
      // Setup the plug-in with the chosen options
      return XrdSecProtocolpwd::Init(opts,erp);
   }
   //
   // Setup the plug-in with the defaults
   return XrdSecProtocolpwd::Init(opts,erp);
}}


/******************************************************************************/
/*              X r d S e c P r o t o c o l p w d O b j e c t                 */
/******************************************************************************/
  
extern "C"
{
XrdSecProtocol *XrdSecProtocolpwdObject(const char              mode,
                                        const char             *hostname,
                                        const struct sockaddr  &netaddr,
                                        const char             *parms,
                                        XrdOucErrInfo    *erp)
{
   XrdSecProtocolpwd *prot;
   int options = XrdSecNOIPCHK;

   //
   // Get a new protocol object
   if (!(prot = new XrdSecProtocolpwd(options, hostname, &netaddr, parms))) {
      char *msg = (char *)"Secpwd: Insufficient memory for protocol.";
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

//__________________________________________________________________________
int XrdSecProtocolpwd::ParseCrypto(XrdSutBuffer *buf)
{
   // Parse received buffer for the crypto module to be used.
   // Parse crypto list clist, extracting the first available module
   // and getting a related local cipher and a related reference
   // cipher to be used to agree the session cipher; the local lists
   // crypto info is updated, if needed
   // The results are used to fill the handshake part of the protocol
   // instance.
   EPNAME("ParseCrypto");

   // Check inputs
   if (!buf) {
      DEBUG("invalid input ("<<buf<<")");
      return -1;
   }

   String clist = "";
   XrdSutBucket *bck = 0;
   // Check type of buffer we got
   if (!buf->GetNBuckets()) {
      // If the bucket list is empty we assume this being the first iteration
      // step (the step is not defined at this point).
      // The option field should contain the relevant information
      String opts = buf->GetOptions();
      if (!(opts.length())) {
         DEBUG("missing options - bad format");
         return -1;
      }
      //
      // Extract crypto module list, if any
      int ii = opts.find("c:");
      if (ii >= 0) {
         clist.assign(opts, ii+2);
         clist.erase(clist.find(','));
      } else {
         DEBUG("crypto information not found in options");
         return -1;
      }
   } else {
      //
      // Extract crypto module name from the buffer
      if (!(bck = buf->GetBucket(kXRS_cryptomod))) {
         DEBUG("cryptomod buffer missing");
         return -1;
      }
      bck->ToString(clist);
   }
   DEBUG("parsing list: "<<clist.c_str());
 
   // Load module and define relevant pointers
   hs->CryptoMod = "";
   // Parse list
   if (clist.length()) {
      int from = 0;
      while ((from = clist.tokenize(hs->CryptoMod, from, '|')) != -1) {
         // Check this module
         if (hs->CryptoMod.length()) {
            // Load the crypto factory
            if ((hs->CF = XrdCryptoFactory::GetCryptoFactory(hs->CryptoMod.c_str()))) {
               int fid = hs->CF->ID();
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
   }

   return 1;
}

//____________________________________________________________________
bool XrdSecProtocolpwd::CheckCreds(XrdSutBucket *creds, int ctype)
{
   // Check credentials against information in password file
   EPNAME("CheckCreds");
   bool match = 0;
 
   // Check inputs
   if (!hs->CF || !creds || !hs->Pent) {
      DEBUG("Invalid inputs ("<<hs->CF<<","<<creds<<","<<hs->Pent<<")");
      return match;
   }
   // Make sure there is something to check against
   if (ctype != kpCT_afs && ctype != kpCT_afsenc &&
      (!(hs->Pent->buf1.buf) || hs->Pent->buf1.len <= 0)) {
      DEBUG("Cached information about creds missing");
      return match;
   }
   //
   // Create a buffer to store credentials, if required
   int len = creds->size+4;
   char *cbuf = (KeepCreds) ? new char[len] : (char *)0;

   //
   // Separate treatment for crypt-like creds
   if (ctype != kpCT_crypt && ctype != kpCT_afs && ctype != kpCT_afsenc) {
      //
      // Create a bucket for the salt to easy encryption
      XrdSutBucket *tmps = new XrdSutBucket();
      if (!tmps) {
         DEBUG("Could not allocate working buckets area for the salt");
         return match;
      }
      tmps->SetBuf(hs->Pent->buf1.buf, hs->Pent->buf1.len);
      //
      // Save input bucket if creds have to be kept
      if (KeepCreds) {
         memcpy(cbuf, "pwd:", 4);
         memcpy(cbuf+4, creds->buffer, creds->size);
      }
      //
      // Hash received buffer for the comparison
      DoubleHash(hs->CF,creds,tmps);
      // Compare
      if (hs->Pent->buf2.len == creds->size)
         if (!memcmp(creds->buffer, hs->Pent->buf2.buf, creds->size))
            match = 1;
      SafeDelete(tmps);
      //
      // recover input creds
      if (match && KeepCreds)
         creds->SetBuf(cbuf, len);

   } else {
#ifndef DONT_HAVE_CRYPT
#ifndef R__AFS
      // Crypt-like: get the pwhash
      String passwd(creds->buffer,creds->size+1);
      passwd.reset(0,creds->size,creds->size);
      // Get the crypt
      char *pass_crypt = crypt(passwd.c_str(), hs->Pent->buf1.buf);
      // Compare
      if (!strncmp(pass_crypt, hs->Pent->buf1.buf, hs->Pent->buf1.len + 1) != 0)
         match = 1;
      if (match && KeepCreds) {
         memcpy(cbuf, "cpt:", 4);
         memcpy(cbuf+4, creds->buffer, creds->size);
         creds->SetBuf(cbuf, len);
      }
#else
      // Check the AFS credentials
      match = CheckCredsAFS(creds, ctype);
#endif
#else
      DEBUG("Crypt-like passwords (via crypt(...)) not supported");
      match = 0;
#endif
   }

   // Cleanup
   if (cbuf)
      delete[] cbuf;

   // We are done
   return match;
}

#ifdef R__AFS
//________________________________________________________________________
bool XrdSecProtocolpwd::CheckCredsAFS(XrdSutBucket *creds, int ctype)
{
   // Check AFS credentials, either in plain (ctype==kpCT_afs) or
   // encrypted (ctype==kpCT_afsenc) form
   EPNAME("CheckCredsAFS");
   bool match = 0;
   int rc = 0;

   // Here we are interested to the minimal token length (5 min)
   int life = 60;

   // We need a link to the user name
   char *usr = (char *) hs->User.c_str();

   bool notify = ((hs->Step == kXPC_creds) || QTRACE(ALL)) ? 1 : 0;
   struct ktc_encryptionKey key;
   if (ctype == kpCT_afs) {
      char *errmsg;
      char *pwd = new char[creds->size + 1];
      memcpy(pwd, creds->buffer, creds->size);
      pwd[creds->size] = 0;
      rc = ka_UserAuthenticateGeneral(KA_USERAUTH_VERSION + KA_USERAUTH_DOSETPAG,
                                      usr, (char *)"", (char *)"", pwd,
                                      life, 0, 0, &errmsg);
      if (rc) {
         if (notify)
            PRINT("CheckAFS: failure: "<< errmsg);
         hs->ErrMsg += ka_LocalCell();
      } else {
         match = 1;
         if (KeepCreds)
            // We need to encrypt te plain passwd
            ka_StringToKey(pwd, 0, &key);
         if (QTRACE(ALL))
            PRINT("CheckAFS: success!");
      }
      if (pwd) delete [] pwd;

   } else if (ctype == kpCT_afsenc) {

      // Get the cell
      char *cell = 0;
      char cellname[MAXKTCREALMLEN];
      if (ka_ExpandCell(cell, cellname, 0) != 0) {
         PRINT("CheckAFS: failure expanding cell");
         return match;
      }
      cell = cellname;

      // Get an unauthenticated connection to desired cell 
      struct ubik_client *conn = 0;
      if (ka_AuthServerConn(cell, KA_AUTHENTICATION_SERVICE, 0, &conn) != 0) {
         PRINT("CheckAFS: failure getting an unauthenticated connection to the cell");
         return match;
      }

      // Authenticate now
      memcpy(key.data, creds->buffer, creds->size);
      struct ktc_token token;
      int pwexpires;
      int now = hs->TimeStamp;
      rc = ka_Authenticate(usr, (char *)"", cell, conn,
                           KA_TICKET_GRANTING_SERVICE,
                           &key, now, now + life,
                           &token, &pwexpires);
      if (rc) {
         if (notify)
            PRINT("CheckAFS: failure from ka_Authenticate");
         hs->ErrMsg += ka_LocalCell();
      } else {
         match = 1;
         if (QTRACE(ALL))
            PRINT("CheckAFS: success!");
      }
   } else {
      PRINT("CheckAFS: unknown credential type: "<< ctype);
   }

   // Save the creds, if requested
   if (match && KeepCreds) {
      // Create new buffer
      int len = strlen("afs:") + 8;
      char *buf = new char[len];
      memcpy(buf,"afs:",4);
      memcpy(buf+4,key.data,8);
      // Fill output
      creds->SetBuf(buf,len);
   }
   // We are done
   return match;
}
#else
//________________________________________________________________________
bool XrdSecProtocolpwd::CheckCredsAFS(XrdSutBucket *, int)
{
   // Check AFS credentials - not supported
   return 0;
}
#endif

//____________________________________________________________________
int XrdSecProtocolpwd::SaveCreds(XrdSutBucket *creds)
{
   // Save credentials in creds in the password file
   // Returns 0 if ok, -1 otherwise
   EPNAME("SaveCreds");

   // Check inputs
   if ((hs->User.length() <= 0) || !hs->CF || !creds) {
      DEBUG("Bad inputs ("<<hs->User.length()<<","<<hs->CF<<","
                          <<creds<<")");
      return -1;
   }
   // Build effective tag
   String wTag = hs->Tag + '_'; wTag += hs->CF->ID();
   //
   // Update entry in cache, if there, or add one
   XrdSutPFEntry *cent = cacheAdmin.Add(wTag.c_str());
   if (!cent) {
      DEBUG("Could not get entry in cache");
      return -1;
   }
   // Generate a salt and fill it in
   char *tmps = XrdSutRndm::GetBuffer(8,3);
   if (!tmps) {
      DEBUG("Could not generate salt: out-of-memory");
      return -1;
   }
   XrdSutBucket *salt = new XrdSutBucket(tmps,8);
   if (!salt) {
      DEBUG("Could not create salt bucket");
      return -1;
   }
   cent->buf1.SetBuf(salt->buffer,salt->size);
   //
   // Now we sign the creds with the salt
   DoubleHash(hs->CF,creds,salt);
   // and fill in the creds
   cent->buf2.SetBuf(creds->buffer,creds->size);
   //
   // Set entry status OK
   cent->status = kPFE_ok;
   //
   // Save entry
   cent->mtime = hs->TimeStamp;
   //
   DEBUG("Entry for tag: "<<wTag<<" updated in cache");
   //
   // Flush cache content to source file
   XrdSysPrivGuard priv(getuid(), getgid());
   if (priv.Valid()) {
      if (cacheAdmin.Flush() != 0) {
         DEBUG("WARNING: some problem flushing to admin file after updating "<<wTag);
      }
   }
   //
   // We are done
   return 0;
}

//____________________________________________________________________
int XrdSecProtocolpwd::ExportCreds(XrdSutBucket *creds)
{
   // Export client credentials to a PF file to be used as autologin
   // in a next step.
   // Returns 0 if ok, -1 otherwise
   EPNAME("ExportCreds");

   // Check inputs
   if ((hs->User.length() <= 0) || !hs->CF || !creds) {
      DEBUG("Bad inputs ("<<hs->User.length()<<","<<hs->CF<<","
                          <<creds<<")");
      return -1;
   }

   // Check inputs
   if (FileExpCreds.length() <= 0) {
      DEBUG("File (template) undefined - do nothing");
      return -1;
   }

   // Expand templated keywords, if needed
   String filecreds = FileExpCreds;
   // Resolve place-holders, if any
   if (XrdSutResolve(filecreds, Entity.host, Entity.vorg, Entity.grps, Entity.name) != 0) {
      DEBUG("Problems resolving templates in "<<filecreds);
      return -1;
   }
   DEBUG("Exporting client creds to: "<<filecreds);

   // Attach or create the file
   XrdSutPFile pfcreds(filecreds.c_str());
   if (!pfcreds.IsValid()) {
      DEBUG("Problem attaching / creating file "<<filecreds);
      return -1;
   }
   //
   // Build effective tag
   String wTag = hs->Tag + '_'; wTag += hs->CF->ID();
   //
   // Create and fill a new entry
   XrdSutPFEntry ent;
   ent.SetName(wTag.c_str());
   ent.status = kPFE_ok;
   ent.cnt    = 0;
   if (!strncmp(creds->buffer, "pwd:", 4)) {
      // Skip initial "pwd:"
      ent.buf1.SetBuf(creds->buffer+4, creds->size-4);
   } else {
      // For crypt and AFS we keep that to be able to distinguish
      // later on
      ent.buf1.SetBuf(creds->buffer,creds->size);
   }
   //
   // Write entry
   ent.mtime = time(0);
   pfcreds.WriteEntry(ent);
   DEBUG("New entry for "<<wTag<<" successfully written to file: "
                  <<filecreds);
   // We are done
   return 0;
}

//____________________________________________________________________
XrdSutBucket *XrdSecProtocolpwd::QueryCreds(XrdSutBuffer *bm,
                                            bool netrc, int &status)
{
   // Get credential information to be sent to the server
   EPNAME("QueryCreds");

   // Check inputs
   if (!bm || !hs->CF || hs->Tag.length() <= 0) {
      DEBUG("bad inputs ("<<bm<<","<<hs->CF<<","<<hs->Tag.length()<<")");
      return (XrdSutBucket *)0;
   }

   //
   // Type of creds (for the prompt)
   int ctype = (status > kpCT_undef) ? status : kpCT_normal;
   netrc = ((ctype == kpCT_normal || ctype == kpCT_onetime ||
             ctype == kpCT_old || ctype == kpCT_crypt)) ? netrc : 0;
   //
   // reset status
   status = kpCI_undef;
   // Output bucket
   XrdSutBucket *creds = new XrdSutBucket();
   if (!creds) {
      DEBUG("Could allocate bucket for creds");
      return (XrdSutBucket *)0;
   }
   creds->type = kXRS_creds;

   //
   // Build effective tag
   String wTag = hs->Tag + '_'; wTag += hs->CF->ID();

   //
   // If creds are available in the environment pick them up and use them
   char *cf = 0;
   char *cbuf = getenv("XrdSecCREDS");
   if (cbuf) {
      int len = strlen(cbuf);
      // From hex
      int sz = len;
      char *out = new char[sz/2+2];
      XrdSutFromHex((const char *)cbuf, out, len);
      if ((cf = strstr(out, "&pwd"))) {
         cf += 5;
         len -= 5;
         if (len > 0) {
            // Get prefix
            char pfx[5] = {0};
            memcpy(pfx, cf, 4);
            cf += 4;
            len -= 4;
            if (len > 0) {
               DEBUG("using "<<len<<" bytes of creds from the environment; pfx: "<<pfx);
               // Create or Fill entry in cache
               hs->Pent = cacheAlog.Add(wTag.c_str());
               if (hs->Pent) {
                 // Try only once
                  if (hs->Pent->cnt == 0) {
                     // Set buf
                     creds->SetBuf(cf,len);
                     // Fill entry
                     if (strncmp(pfx,"pwd",3))
                        hs->Pent->status = kPFE_crypt;
                     hs->Pent->mtime = hs->TimeStamp;
                     hs->Pent->buf1.SetBuf(cf, len);
                     // Just in case we need the passwd itself (like in crypt)
                     hs->Pent->buf2.SetBuf(cf, len);
                     // Tell the server
                     if (!strncmp(pfx,"afs",3)) {
                        String afsInfo = "c";
                        if (bm->UpdateBucket(afsInfo, kXRS_afsinfo) != 0)
                           PRINT("Warning: problems updating bucket with AFS info");
                     }
                     // Update status
                     status = kpCI_exact; 
                     // We are done
                     return creds;
                  } else {
                     // Cleanup
                     hs->Pent->buf1.SetBuf();
                     hs->Pent->buf2.SetBuf();
                  }
               } else {
                  PRINT("Could create new entry in cache");
                  return (XrdSutBucket *)0;
               }
            }
         }
      }
   }

   //
   // Extract AFS info (the cell), if any
   String afsInfo;
   if (ctype == kpCT_afs || ctype == kpCT_afsenc) {
      XrdSutBucket *bafs = bm->GetBucket(kXRS_afsinfo);
      if (bafs)
         bafs->ToString(afsInfo);
   }
   //
   // Search information in autolog file(s) first, if required
   if (netrc) {
      //
      // Make sure cache it is up-to-date
      if (PFAlog.IsValid()) {
         if (cacheAlog.Refresh() != 0) {
            DEBUG("problems assuring cache update for file alog ");
         }
      }
      //
      // We may already have an entry in the cache
      bool wild = 0;
      hs->Pent = cacheAlog.Get(wTag.c_str(),&wild);
      // Retrieve pwd information if ok 
      if (hs->Pent && hs->Pent->buf1.buf) {
         if (hs->Pent->cnt == 0) {
            cf = hs->Pent->buf1.buf;
            bool afspwd = strncmp(cf,"afs",3) ? 0 : 1;
            if (!strncmp(cf,"cpt",3) || afspwd) {
               int len = hs->Pent->buf1.len;
               cf += 4;
               len -= 4;
               hs->Pent->status = kPFE_crypt;
               hs->Pent->mtime = hs->TimeStamp;
               hs->Pent->buf1.SetBuf(cf, len);
               // Just in case we need the passwd itself (like in crypt)
               hs->Pent->buf2.SetBuf(cf, len);
               // Tell the server
               if (afspwd) {
                  afsInfo = "c";
                  if (bm->UpdateBucket(afsInfo, kXRS_afsinfo) != 0)
                     PRINT("Warning: problems updating bucket with AFS info");
               }
            }
            // Fill output with double hash
            creds->SetBuf(hs->Pent->buf1.buf,hs->Pent->buf1.len);
            // Update status
            status = wild ? kpCI_wildcard : kpCI_exact; 
            // We are done
            return creds;
         } else {
            // Entry not ok: probably previous attempt failed: discard
            hs->Pent->buf1.SetBuf();
         }
      }

      // for crypt-like, look also into a .netrc-like file, if any
      String passwd;
      String host(hs->Tag,hs->Tag.find("@",0)+1,hs->Tag.find(":",0)-1);
      if (QueryNetRc(host, passwd, status) == 0) {
         // Create or Fill entry in cache
         if ((hs->Pent = cacheAlog.Add(wTag.c_str()))) {
            // Fill entry
            hs->Pent->status = kPFE_crypt;
            hs->Pent->mtime = hs->TimeStamp;
            hs->Pent->buf1.SetBuf(passwd.c_str(),passwd.length());
            // Fill output
            creds->SetBuf(passwd.c_str(),passwd.length());
            // Update status
            status = kpCI_exact; 
            // We are done
            return creds;
         } else {
            DEBUG("Could create new entry in cache");
            return (XrdSutBucket *)0;
         }
      }
   }
   //
   // Create or Fill entry in cache
   if (!(hs->Pent) && !(hs->Pent = cacheAlog.Add(wTag.c_str()))) {
      DEBUG("Could create new entry in cache");
      return (XrdSutBucket *)0;
   }

   //
   // If a previous attempt was successful re-use same passwd
   if (hs->Pent && hs->Pent->buf1.buf && hs->Pent->cnt == 0) {
      // Fill output
      creds->SetBuf(hs->Pent->buf1.buf,hs->Pent->buf1.len);
      // Update status
      status = kpCI_exact; 
      // We are done
      return creds;
   }

   //
   // We are here because:
   //   1) autologin disabled or no autologin info found
   //       ==> hs->Pent empty ==> prompt for password
   //   2) we need to send a new password hash because it was wrong
   //       ==> hs->Pent->buf2 empty ==> prompt for password
   //   3) we need to send a new password hash because it has expired
   //      (either one-time or too old)
   //       ==> query hs->Pent->buf2 before prompting
   //   4) we need to send a real password because the server uses crypt()
   //      or AFS
   //       ==> query hs->Pent->buf2 from previous prompt

   //
   // If the previously cached entry has a second (final) passwd
   // use it. This is the case when the real passwd is required (like in
   // crypt), we may have it in cache from a previous prompt
   if (ctype == kpCT_crypt || ctype == kpCT_afs) {
      if (hs->Pent && hs->Pent->buf2.buf) {
         if (ctype == kpCT_afs) {
#ifdef R__AFS
            String passwd(hs->Pent->buf2.buf,hs->Pent->buf2.len);
            // We will send over and encrypted version
            struct ktc_encryptionKey key;
            ka_StringToKey((char *) passwd.c_str(),
                           (char *) afsInfo.c_str(), &key);
            // Fill output
            creds->SetBuf(key.data,8);
            // Tell the server
            afsInfo = "c";
            if (bm->UpdateBucket(afsInfo, kXRS_afsinfo) != 0)
               PRINT("Warning: problems updating bucket with AFS info");
#else
            // Fill output
            creds->SetBuf(hs->Pent->buf2.buf,hs->Pent->buf2.len);
            // Not needed anymore
            bm->Deactivate(kXRS_afsinfo);
#endif
         } else {
            // Fill output
            creds->SetBuf(hs->Pent->buf2.buf,hs->Pent->buf2.len);
         }
         // Save info in the first buffer and reset the second buffer
         hs->Pent->buf1.SetBuf(hs->Pent->buf2.buf,hs->Pent->buf2.len);
         hs->Pent->buf2.SetBuf();
         // Update status
         status = kpCI_exact; 
         // We are done
         return creds;
      }
   }

   //
   // From now we need to prompt the user: we can do this only if
   // connected to a terminal
   if (!(hs->Tty)) {
      DEBUG("Not connected to tty: cannot prompt user for credentials");
      return (XrdSutBucket *)0;
   }

   //
   // Prompt
   char prompt[XrdSutMAXPPT] = {0};
   if (ctype == kpCT_onetime)
      snprintf(prompt,XrdSutMAXPPT, "Password for %s not active: "
               "starting activation handshake.",hs->Tag.c_str());
   //
   // Prepare the prompt
   if (ctype == kpCT_new) {
      snprintf(prompt,XrdSutMAXPPT, "Enter new password: ");
   } else if (ctype == kpCT_crypt) {
      String host(hs->Tag,hs->Tag.find("@",0)+1,hs->Tag.find(":",0)-1);
      snprintf(prompt,XrdSutMAXPPT, "Password for %s@%s: ", 
                                    hs->User.c_str(), host.c_str());
   } else if (ctype == kpCT_afs || ctype == kpCT_afsenc) {
      snprintf(prompt,XrdSutMAXPPT, "AFS password for %s@%s: ", 
                                    hs->User.c_str(), hs->AFScell.c_str());
   } else {
      // Normal prompt
      snprintf(prompt,XrdSutMAXPPT,"Password for %s:",hs->Tag.c_str());
   }
   //
   // Inquire password
   int natt = MaxPrompts;
   String passwd = "";
   bool changepwd =0;
   while (natt-- && passwd.length() <= 0) {
      XrdSutGetPass(prompt, passwd);
      // If in the format $changepwd$<passwd> we are asking for
      // a password change
      if (passwd.beginswith("$changepwd$")) {
         PRINT("Requesting a password change");
         changepwd = 1;
         passwd.erase("$changepwd$",0,strlen("$changepwd$"));
      }
      if (passwd.length()) {
         // Fill in password
         creds->SetBuf(passwd.c_str(),passwd.length());
         if (ctype != kpCT_crypt && ctype != kpCT_afs) {
            // Self-Hash
            DoubleHash(hs->CF,creds,creds);
            // Update status
            status = kpCI_prompt;
         } else if (ctype == kpCT_afs) {
#ifdef R__AFS
            // We will send over and encrypted version
            struct ktc_encryptionKey key;
            ka_StringToKey((char *) passwd.c_str(),
                           (char *) afsInfo.c_str(), &key);
            creds->SetBuf(key.data,8);
            // Tell the server
            afsInfo = "c";
            if (bm->UpdateBucket(afsInfo, kXRS_afsinfo) != 0)
               PRINT("Warning: problems updating bucket with AFS info");
#endif
         }
         // Save creds to update auto-login file later
         // It will be flushed to file if required
         if (changepwd)
            hs->Pent->status = kPFE_onetime;
         else
            hs->Pent->status = kPFE_ok;
         hs->Pent->buf1.SetBuf(creds->buffer,creds->size);
         //
         // Just in case we need the passwd itself (like in crypt)
         hs->Pent->buf2.SetBuf(passwd.c_str(),passwd.length());
         // Update autologin, if required
         if (AutoLogin > 0)
            UpdateAlog();
      }
   }
   // Cleanup, if we did not get anything
   if (passwd.length() <= 0) {
      delete creds;
      creds = 0;
   }
   // We are done
   return creds;
}

//____________________________________________________________________
int XrdSecProtocolpwd::UpdateAlog()
{
   // Save pass hash in autologin file
   // Returns 0 if ok, -1 otherwise
   EPNAME("UpdateAlog");

   // Check inputs
   if (hs->Tag.length() <= 0) {
      DEBUG("Tag undefined - do nothing");
      return -1;
   }
   // Check inputs
   if (!(hs->Pent) || !(hs->Pent->buf1.buf)) {
      DEBUG("Nothing to do");
      return 0;
   }
   //
   // Build effective tag
   String wTag = hs->Tag + '_'; wTag += hs->CF->ID();
   //
   // Make sure the other buffers are reset
   hs->Pent->buf2.SetBuf();
   hs->Pent->buf3.SetBuf();
   hs->Pent->buf4.SetBuf();
   //
   // Set entry status OK
   hs->Pent->status = kPFE_ok;
   //
   // Reset count
   hs->Pent->cnt = 0;
   //
   // Save entry
   hs->Pent->mtime = hs->TimeStamp;
   //
   DEBUG("Entry for tag: "<<wTag<<" updated in cache");
   //
   // Flush cache content to source file
   if (cacheAlog.Flush() != 0) {
      DEBUG("WARNING: some problem flushing to alog file after updating "<<wTag);
   }
   //
   // We are done
   return 0;
}

//____________________________________________________________________
int XrdSecProtocolpwd::QueryUser(int &status, String &cmsg)
{
   // Check that info about the defined user is available
   EPNAME("QueryUser");

   DEBUG("Enter: " << hs->User);

   // Check inputs
   if (hs->User.length() <= 0 || !hs->CF || !hs->Cref) {
      DEBUG("Invalid inputs ("<<hs->User.length()<<","<<hs->CF<<","<<hs->Cref<<")");
      return -1;
   }
   //
   // Build effective tag
   String wTag = hs->Tag + '_'; wTag += hs->CF->ID();
   //
   // Default status
   status = kPFE_disabled;
   int bad = -1;
   cmsg = "";
   //
   // Check first info in user's home, if allowed
   if (UserPwd) {
      // Get userinfo
      struct passwd *pw = getpwnam(hs->User.c_str());
      int rcst = 0;
      kXR_int32 mtime = -1;
      bool fcrypt = 0;
      String File;
      if (pw) {
         File.resize(strlen(pw->pw_dir)+FileUser.length()+10);
         File.assign(pw->pw_dir, 0);
         File += FileUser;
         // Get status
         struct stat st;
         if ((rcst = stat(File.c_str(),&st)) != 0 && errno == ENOENT) {
            if (UserPwd > 1) {
               // Try special crypt like file
               File.replace(FileUser,FileCrypt);
               fcrypt = 1;
               rcst = 0;
            }
         }
         mtime = (rcst == 0) ? st.st_mtime : mtime;
      }

      if (rcst == 0) {
         //
         // Check cache first
         hs->Pent = cacheUser.Get(wTag.c_str());
         if (!hs->Pent || (hs->Pent->mtime < mtime)) {
            hs->Pent = (hs->Pent) ? hs->Pent : cacheUser.Add(wTag.c_str());
            if (hs->Pent) {
               //
               // Try the files
               if (!fcrypt) {
                  // Try to attach to File
                  XrdSutPFile ff(File.c_str(), kPFEopen,0,0);
                  if (ff.IsValid()) {
                     // Retrieve pwd information
                     if (ff.ReadEntry(wTag.c_str(),*(hs->Pent)) > 0) {
                        bad = 0;
                        status = hs->Pent->status;
                        ff.Close();
                        return 0;
                     }
                     ff.Close();
                  }
               } else if (UserPwd > 1) {
                  String pwhash;
                  if (QueryCrypt(FileCrypt, pwhash) > 0) {
                     bad = 0;
                     status = kPFE_crypt;
                     // Fill entry
                     hs->Pent->mtime = hs->TimeStamp;
                     hs->Pent->status = status;
                     hs->Pent->cnt = 0;
                     if (!FileCrypt.beginswith("afs:"))
                        hs->Pent->buf1.SetBuf(pwhash.c_str(),pwhash.length()+1);
                     // Trasmit the type of credentials we have found
                     cmsg = FileCrypt;
                     return 0;
                  }
               }
            }
         } else {
            // Fill entry
            bad = 0;
            status = hs->Pent->status;
            hs->Pent->mtime = hs->TimeStamp;
            if (status == kPFE_crypt)
               cmsg = FileCrypt;
            return 0;
         }
      }
   }

   //
   // Check system info, if enabled
   if (SysPwd) {
      String pwhash, fn;
      if (QueryCrypt(fn, pwhash) > 0) {
         bad = 0;
         status = kPFE_crypt;
         // Fill entry
         hs->Pent = cacheUser.Add(wTag.c_str());
         hs->Pent->mtime = hs->TimeStamp;
         hs->Pent->status = status;
         hs->Pent->cnt = 0;
         if (!fn.beginswith("afs:"))
            hs->Pent->buf1.SetBuf(pwhash.c_str(),pwhash.length()+1);
         // Trasmit the type of credentials we have found
         cmsg = fn;
         return 0;
      }
   }
   //
   // Check server admin files
   if (PFAdmin.IsValid()) {
      //
      // Make sure it is uptodate
      XrdSysPrivGuard priv(getuid(), getgid());
      if (priv.Valid()) {
         if (cacheAdmin.Refresh() != 0) {
            DEBUG("problems assuring cache update for file admin ");
            return -1;
         }
      }
      hs->Pent = cacheAdmin.Get(wTag.c_str());
      // Retrieve pwd information
      if (hs->Pent) {
         bad = 0;
         status = hs->Pent->status;
         if (status == kPFE_allowed) {
            if (AutoReg == kpAR_none) {
               // No auto-registration: disable
               status = kPFE_disabled;
               bad = 1;
            }
         } else if (status >= kPFE_ok) {
            // Check failure counter, if required
            if (MaxFailures > 0 && hs->Pent->cnt >= MaxFailures) {
               status = kPFE_disabled;
               bad = 2;
            }
            // Check expiration time, if required
            if (LifeCreds > 0) {
               int expt = hs->Pent->mtime + LifeCreds;
               int now = hs->TimeStamp;
               if (expt < now)
                  status = kPFE_expired;
            }
            if (status != kPFE_disabled)
               return 0;
         }
      }
   }

   //
   // If nothing found, auto-registration is enabled, and the tag 
   // corresponds to a local user, propose auto-registration
   if (bad == -1) {
      if (AutoReg != kpAR_none) {
         status = kPFE_allowed;
         if (AutoReg == kpAR_users) {
            struct passwd *pw = getpwnam(hs->User.c_str());
            if (!pw) {
               status = kPFE_disabled;
               bad = 1;
            }
         }
      } else
         bad = 1;
   }
   //
   // If disabled, fill salt string with message for the client
   if (status == kPFE_disabled) {
      char msg[XrdSutMAXPPT];
      switch (bad) {
      case 1:
         snprintf(msg,XrdSutMAXPPT,"user '%s' unknown: auto-registration"
                  " not allowed: contact %s to register",
                  hs->User.c_str(),SrvEmail.c_str());
         break;
      case 2:
         snprintf(msg,XrdSutMAXPPT,"max number of failures (%d) reached"
                  " for user '%s': contact %s to re-activate",
                  MaxFailures,hs->User.c_str(),SrvEmail.c_str());
         break;
      default:
         msg[0] = '\0';
      }
      cmsg.insert(msg,0,strlen(msg));
   }
   //
   // We are done
   return 0;
}

//_________________________________________________________________________
int XrdSecProtocolpwd::GetUserHost(String &user, String &host)
{
   // Resolve user and host
   EPNAME("GetUserHost");

   // Host
   host = Entity.host;
   if (host.length() <= 0) host = getenv("XrdSecHOST");

   // User
   user = Entity.name;
   if (user.length() <= 0) user = getenv("XrdSecUSER");

   // If user not given, prompt for it
   if (user.length() <= 0) {
      //
      // Make sure somebody can be prompted
      if (!(hs->Tty)) {
         DEBUG("user not defined:"
               "not tty: cannot prompt for user");
         return -1;
      }
      //
      // This is what we want
      String prompt = "Enter user or tag";
      if (host.length()) {
         prompt.append(" for host ");
         prompt.append(host);
      }
      prompt.append(":");
      XrdSutGetLine(user,prompt.c_str());
   }

   DEBUG(" user: "<<user<<", host: "<<host);

   // We are done
   return 0;
}

//_________________________________________________________________________
int XrdSecProtocolpwd::AddSerialized(char opt, kXR_int32 step, String ID,
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
   if (brt && cip) {
      //
      // Encrypt random tag with session cipher
      if (cip->Encrypt(*brt) == 0) {
         DEBUG("error encrypting random tag");
         return -1;
      }
      //
      // Update type
      brt->type = kXRS_signed_rtag;
   }
   // Clients send in any case something session dependent: the server
   // may optionally decide that's enough and save one exchange.
   if (opt == 'c') {
      //
      // Add bucket with our timestamp to the main list
      if (buf->MarshalBucket(kXRS_timestamp,(kXR_int32)(hs->TimeStamp)) != 0) {
         DEBUG("error adding bucket with time stamp");
         return -1;
      }
   }
   //
   // Add an random challenge: if a next exchange is required this will
   // allow to prove authenticity of counter part
   if (opt == 's' || step != kXPC_autoreg) {
      //
      // Generate new random tag and create/update bucket
      String RndmTag;
      XrdSutRndm::GetRndmTag(RndmTag);
      //
      // Get bucket
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
   }
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
int XrdSecProtocolpwd::ParseClientInput(XrdSutBuffer *br, XrdSutBuffer **bm,
                                        String &emsg)
{
   // Parse received buffer b, extracting and decrypting the main 
   // buffer *bm and extracting the session 
   // cipher and server public keys, if there
   // Result used to fill the handshake local variables
   EPNAME("ParseClientInput");

   // Space for pointer to main buffer must be already allocated
   if (!br || !bm) {
      DEBUG("invalid inputs ("<<br<<","<<bm<<")");
      emsg = "invalid inputs";
      return -1;
   }
   //
   // Get the step
   XrdSutBucket *bckm = 0;

   // If first call, not much to do
   if (!br->GetNBuckets()) {
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
      // Extract server ID
      String srvid;
      ii = opts.find("id:");
      if (ii >= 0) {
         srvid.assign(opts, ii+3);
         srvid.erase(srvid.find(','));
      }
      //
      // Extract priority options
      String popt;
      ii = opts.find("po:");
      if (ii >= 0) {
         popt.assign(opts, ii+3);
         popt.erase(popt.find(','));
         // Parse it
         if (popt.beginswith("sys")) {
            hs->SysPwd = 1;
         } else if (popt.beginswith("afs")) {
            hs->SysPwd = 2;
            hs->AFScell.assign(popt,3);
         }
      }
      //
      // Get user and host
      String host;
      if (GetUserHost(hs->User,host) != 0) {
         emsg = "error getting user and host";
         return -1;
      }
      //
      // Build tag and save it into the cache
      hs->Tag.resize(hs->User.length()+host.length()+srvid.length()+5);
      hs->Tag = hs->User;
      if (host.length() > 0)
         hs->Tag += ("@" + host);
      if (srvid.length() > 0)
         hs->Tag += (":" + srvid);
      //
      // Get server puk from cache and initialize handshake cipher
      if (!PFSrvPuk.IsValid()) {
         emsg = "file with server public keys invalid";
         return -1;
      }
      char *ptag = new char[host.length()+srvid.length()+10];
      if (ptag) {
         sprintf(ptag,"%s:%s_%d",host.c_str(),srvid.c_str(),hs->CF->ID());
         bool wild = 0;
         XrdSutPFEntry *ent = cacheSrvPuk.Get((const char *)ptag, &wild);
         if (ent) {
            // Initialize cipher
            SafeDelete(hs->Hcip);
            if (!(hs->Hcip =
                  hs->CF->Cipher(0,ent->buf1.buf,ent->buf1.len))) {
                     DEBUG("could not instantiate session cipher "
                           "using cipher public info from server");
                     emsg = "could not instantiate session cipher ";
            } else {
               DEBUG("hsHcip: 0x"<<hs->Hcip->AsHexString());
            }
         } else {
            // Autoreg is the only alternative at this point ...
            emsg = "server puk not found in cache - tag: ";
            emsg += ptag;
         }
         SafeDelArray(ptag);
      } else 
         emsg = "could not allocate buffer for server puk tag";
      //
      // And we are done;
      return 0;
   }
   //
   // make sure the cache is still there
   if (!hs->Cref) {
      emsg = "cache entry not found";
      return -1;
   }
   //
   // make sure is not too old
   int reftime = hs->TimeStamp - TimeSkew;
   if (hs->Cref->mtime < reftime) {
      emsg = "cache entry expired";
      // Remove: should not be checked a second time
      SafeDelete(hs->Cref);
      return -1;
   }
   //
   // Get from cache version run by server
   hs->RemVers = hs->Cref->status;
   //
   // Extract the main buffer 
   if (!(bckm = br->GetBucket(kXRS_main))) {
      emsg = "main buffer missing";
      return -1;
   }
   //
   // Decrypt, if it makes sense
   if (hs->LastStep != kXPC_autoreg) {
      //
      // make sure the cache is still there
      if (!hs->Hcip) {
         emsg = "session cipher not found";
         return -1;
      }
      //
      // Decrypt it 
      if (!(hs->Hcip->Decrypt(*bckm))) {
         emsg = "error decrypting main buffer with session cipher";
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
   // If (new) server public keys are there extract and save them
   bool newpuk = 0;
   XrdSutBuckList *bcklst = (*bm)->GetBuckList();
   XrdSutBucket *bp = bcklst->Begin();
   while (bp) {
      if (bp->type == kXRS_puk) {
         newpuk = 1;
         // ID is in the first 4 chars ( ....'\0'<puk>)
         char cid[5] = {0};
         memcpy(cid, bp->buffer, 5);
         int id = atoi(cid);
         // Build tag
         String ptag(hs->Tag);
         ptag.erase(0,ptag.find('@')+1);
         ptag += '_';
         ptag += cid;
         // Update or create new entry
         XrdSutPFEntry *ent = cacheSrvPuk.Add(ptag.c_str());
         if (ent) {
            // Set buffer
            ent->buf1.SetBuf((bp->buffer)+5,(bp->size)-5);
            ent->mtime = hs->TimeStamp;
            if (id == hs->CF->ID()) {
               // Initialize cipher
               SafeDelete(hs->Hcip);
               if (!(hs->Hcip =
                     hs->CF->Cipher(0,ent->buf1.buf,ent->buf1.len))) {
                        DEBUG("could not instantiate session cipher "
                              "using cipher public info from server");
                        emsg = "could not instantiate session cipher ";
               } else {
                  DEBUG("hsHcip: 0x"<<hs->Hcip->AsHexString());
               }
            }
        } else {
            // Autoreg is the only alternative at this point ...
            DEBUG("could not create entry in cache - tag: "<<ptag);
         }
      }
      // Get next
      bp = bcklst->Next();
   }
   (*bm)->Deactivate(kXRS_puk);   
   // Update the puk file (for the other sessions ...)
   if (newpuk)
      cacheSrvPuk.Flush();
   //
   // We are done
   return 0;
}

//_________________________________________________________________________
int XrdSecProtocolpwd::ParseServerInput(XrdSutBuffer *br, XrdSutBuffer **bm,
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
   XrdSutBucket *bck = 0;
   XrdSutBucket *bckm = 0;
   //
   // Extract the main buffer 
   if (!(bckm = br->GetBucket(kXRS_main))) {
      cmsg = "main buffer missing";
      return -1;
   }
   //
   // First get the session cipher
   if ((bck = br->GetBucket(kXRS_puk))) {
      //
      // Cleanup
      SafeDelete(hs->Hcip);
      //
      // Prepare cipher agreement: make sure we have the reference cipher
      if (!hs->Rcip) {
         cmsg = "reference cipher missing";
         return -1;
      }
      // Prepare cipher agreement: get a copy of the reference cipher
      if (!(hs->Hcip = hs->CF->Cipher(*hs->Rcip))) {
         cmsg = "cannot get reference cipher";
         return -1;
      }
      //
      // Instantiate the session cipher 
      if (!(hs->Hcip->Finalize(bck->buffer,bck->size,0))) {
         cmsg = "cannot finalize session cipher";
         return -1;
      }
      //
      // We need it only once
      br->Deactivate(kXRS_puk);
   }

   //
   // Decrypt the main buffer with the session cipher, if available
   if (hs->Hcip) {
      if (!(hs->Hcip->Decrypt(*bckm))) {
         cmsg = "error decrypting main buffer with session cipher";
         return -1;
      }
   }
   //
   // Deserialize main buffer
   if (!((*bm) = new XrdSutBuffer(bckm->buffer,bckm->size))) {
      cmsg = "error deserializing main buffer";
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
   // Get cache entry or create a new one
   if (!hs->Cref) {
      // Create it
      if (!(hs->Cref = new XrdSutPFEntry(hs->ID.c_str()))) {
         cmsg = "cannot create cache entry";
         return -1;
      }
   } else {
      //
      // make sure cache is not too old
      int reftime = hs->TimeStamp - TimeSkew;
      if (hs->Cref->mtime < reftime) {
         cmsg = "cache entry expired";
         SafeDelete(hs->Cref);
         return -1;
      }
   }

   //
   // Extract user name, if any
   if ((bck = (*bm)->GetBucket(kXRS_user))) {
      if (hs->User.length() <= 0) {
         bck->ToString(hs->User);
         // Build tag
         hs->Tag = hs->User;
      }
      (*bm)->Deactivate(kXRS_user);
   }
   //
   // We are done
   return 0;
}

//__________________________________________________________________
void XrdSecProtocolpwd::ErrF(XrdOucErrInfo *einfo, kXR_int32 ecode,
                             const char *msg1, const char *msg2,
                             const char *msg3)
{
   // Filling the error structure
   EPNAME("ErrF");

   char *msgv[12];
   int k, i = 0, sz = strlen("Secpwd");

   //
   // Code message, if any
   int cm = (ecode >= kPWErrParseBuffer && 
             ecode <= kPWErrError) ? (ecode-kPWErrParseBuffer) : -1;
   const char *cmsg = (cm > -1) ? gPWErrStr[cm] : 0;

   //
   // Build error message array
              msgv[i++] = (char *)"Secpwd";     //0
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
XrdSecCredentials *XrdSecProtocolpwd::ErrC(XrdOucErrInfo *einfo,
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
int XrdSecProtocolpwd::ErrS(String ID, XrdOucErrInfo *einfo,
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
   return kpST_error;
}

//_______________________________________________________________________
int XrdSecProtocolpwd::DoubleHash(XrdCryptoFactory *cf, XrdSutBucket *bck,
                                  XrdSutBucket *s1, XrdSutBucket *s2,
                                  const char *tag)
{
   // Apply single or double hash to bck using salts
   // in s1 and (if defined) s2.
   // Store result in *buf, with the new length in len.
   // Return 0 if ok or -1 otherwise
   EPNAME("DoubleHash");

   //
   // Check inputs
   if (!cf || !bck) {
      DEBUG("Bad inputs "<<cf<<","<<bck<<")");
      return -1;
   }
   //
   // At least one salt must be defined
   if ((!s1 || s1->size <= 0) && (!s2 || s2->size <= 0)) {
      DEBUG("Both salts undefined - do nothing");
      return 0;
   }
   //
   // Tag length, if there
   int ltag = (tag) ? strlen(tag) + 1 : 0;
   //
   // Get one-way hash function
   XrdCryptoKDFun_t KDFun = cf->KDFun();
   XrdCryptoKDFunLen_t KDFunLen = cf->KDFunLen();
   if (!KDFun || !KDFunLen) {
      DEBUG("Could not get hooks to one-way hash functions ("
            <<KDFun<<","<<KDFunLen<<")");
      return -1;
   }
   //
   // Apply first salt, if defined
   char *nhash = 0, *thash = bck->buffer;
   int nhlen = bck->size;
   if (s1 && s1->size > 0) {
      if (!(nhash = new char[(*KDFunLen)() + ltag])) {
         DEBUG("Could not allocate memory for hash - s1");
         return -1;
      }
      if ((nhlen = (*KDFun)(thash,nhlen,
                            s1->buffer,s1->size,nhash+ltag,0)) <= 0) {
         DEBUG("Problems hashing - s1");
         delete[] nhash;
         return -1;
      }
      thash = nhash;
   }
   //
   // Apply second salt, if defined
   if (s2 && s2->size > 0) {
      if (!(nhash = new char[(*KDFunLen)() + ltag])) {
         DEBUG("Could not allocate memory for hash - s2");
         return -1;
      }
      if (thash && thash != bck->buffer) thash += ltag;
      if ((nhlen = (*KDFun)(thash,nhlen,
                            s2->buffer,s2->size,nhash+ltag,0)) <= 0) {
         DEBUG("Problems hashing - s2");
         delete[] nhash;
         if (thash && thash != bck->buffer) delete[] thash;
         return -1;
      }
      if (thash && thash != bck->buffer) delete[] thash;
      thash = nhash;
   }
   //
   // Add tag if there
   if (tag)
      memcpy(thash,tag,ltag);
   //
   // Save result
   bck->SetBuf(thash,nhlen+ltag);
   //
   // We are done
   return 0;
}

//______________________________________________________________________________
int XrdSecProtocolpwd::QueryCrypt(String &fn, String &pwhash)
{
   // Retrieve crypt-like password-hash from $HOME/fn or from system password files,
   // if accessible.
   // To avoid problems with NFS-root-squashing, if 'root' changes temporarly the
   // uid/gid to those of the target user (usr).   
   // If OK, returns pass length and fill 'pass' with the password, null-terminated.
   // ('pass' is allocated externally to contain max lpwmax bytes).
   // If the file does not exists, return 0 and an empty pass.
   // If any problems with the file occurs, return a negative
   // code, -2 indicating wrong file permissions.
   // If any problem with changing ugid's occurs, prints a warning trying anyhow
   // to read the password hash.
   EPNAME("QueryCrypt");

   int rc = -1;
   int len = 0, n = 0, fid = -1;
   pwhash = "";
   DEBUG("analyzing file: "<<fn);

   //
   // Get the password structure
   struct passwd *pw = getpwnam(hs->User.c_str());
   if (!pw) {
      DEBUG("Cannot get pwnam structure for user "<<hs->User);
      return -1;
   }
   //
   // Check the user specific file first, if requested
   if (fn.length() > 0) {

      // target uid
      int uid = pw->pw_uid;

      // Acquire the privileges, if needed
      XrdSysPrivGuard priv(uid, pw->pw_gid);
      bool go = priv.Valid();
      if (!go) {
         DEBUG("problems acquiring temporarly identity: "<<hs->User);
      }

      // The file
      String fpw(pw->pw_dir, strlen(pw->pw_dir) + fn.length() + 5);
      if (go) {
         fpw += ("/" + fn);
         DEBUG("checking file "<<fpw<<" for user "<<hs->User);
      }

      // Check first the permissions: should be 0600
      struct stat st;
      if (go && stat(fpw.c_str(), &st) == -1) {
         if (errno != ENOENT) {
            DEBUG("cannot stat password file "<<fpw<<" (errno:"<<errno<<")");
            rc = -1;
         } else {
            DEBUG("file "<<fpw<<" does not exist");
            rc = 0;
         }
         go = 0;
      }
      if (go &&
         (!S_ISREG(st.st_mode) || S_ISDIR(st.st_mode) ||
          (st.st_mode & (S_IWGRP | S_IWOTH | S_IRGRP | S_IROTH)) != 0)) {
         DEBUG("pass file "<<fpw<<": wrong permissions "<<
               (st.st_mode & 0777) << " (should be 0600)");
         rc = -2;
         go = 0;
      }

      // Open the file
      if (go && (fid = open(fpw.c_str(), O_RDONLY)) == -1) {
         DEBUG("cannot open file "<<fpw<<" (errno:"<<errno<<")");
         rc = -1;
         go = 0;
      }

      // Read password-hash
      char pass[128];
      if (go && (n = read(fid, pass, sizeof(pass)-1)) <= 0) {
         close(fid);
         DEBUG("cannot read file "<<fpw<<" (errno:"<<errno<<")");
         rc = -1;
         go = 0;
      }
      if (fid > -1)
         close(fid);

      // Get rid of special trailing chars 
      if (go) {
         len = n;
         while (len-- && (pass[len] == '\n' || pass[len] == 32))
            pass[len] = 0;
         // Null-terminate
         pass[++len] = 0;
         rc = len;
         // Prepare for output
         pwhash = pass;
      }
   }
   //
   // If we go a pw-hash we are done
   if (pwhash.length() > 0)
      return rc;
   //
   // If not, we check the system files
#ifdef R__AFS
   // Send over the cell
   fn = "afs:";
   fn += ka_LocalCell();
   pwhash = "afs";
#else
#ifdef R__SHADOWPW
   {  // Acquire the privileges; needs to be 'superuser' to access the
      // shadow password file
      XrdSysPrivGuard priv((uid_t)0, (gid_t)0);
      if (priv.Valid()) {
         struct spwd *spw = 0;
         // System V Rel 4 style shadow passwords
         if ((spw = getspnam(hs->User.c_str())) == 0) {
            DEBUG("shadow passwd not accessible to this application");
         } else
            pwhash = spw->sp_pwdp;
      } else {
         DEBUG("problems acquiring temporarly superuser privileges");
      }
   }
#else
   pwhash = pw->pw_passwd;
#endif
   //
   // This is send back to the client to locate autologin info
   fn = "system";
#endif
   // Check if successful
   if ((rc = pwhash.length()) <= 2) {
      DEBUG("passwd hash not available for user "<<hs->User);
      pwhash = "";
      fn = "";
      rc = -1;
   }

   // We are done
   return rc;
}

//______________________________________________________________________________
int XrdSecProtocolpwd::QueryNetRc(String host, String &passwd, int &status)
{
   // Check netrc-like file defined by env 'XrdSecNETRC' for password information
   // matching ('user','host') and return the password in 'passwd'.
   // If found, 'status' is filled with 'kpCI_exact' or 'kpCI_wildcard' 
   // depending the type of match.
   // Same syntax as $HOME/.netrc is required; wild cards for hosts are 
   // supported: examples
   //
   // machine oplapro027.cern.ch login qwerty password Rt8dsAvV0
   // machine lxplus*.cern.ch login poiuyt password WtHAyD0iG
   //
   // Returns 0 is something found, -1 otherwise.
   // NB: file permissions must be: readable/writable by the owner only 
   EPNAME("QueryNetRc");
   passwd = ""; 
   //
   // Make sure a file name is defined
   String fnrc = getenv("XrdSecNETRC");
   if (fnrc.length() <= 0) {
      DEBUG("File name undefined");
      return -1;
   }
   // Resolve place-holders, if any
   if (XrdSutResolve(fnrc, Entity.host, Entity.vorg, Entity.grps, Entity.name) != 0) {
      DEBUG("Problems resolving templates in "<<fnrc);
      return -1;
   }
   DEBUG("checking file "<<fnrc<<" for user "<<hs->User);

   // Check first the permissions: should be 0600
   struct stat st;
   if (stat(fnrc.c_str(), &st) == -1) {
      if (errno != ENOENT) {
         DEBUG("cannot stat password file "<<fnrc<<" (errno:"<<errno<<")");
      } else {
         DEBUG("file "<<fnrc<<" does not exist");
      }
      return -1;
   }
   if (!S_ISREG(st.st_mode) || S_ISDIR(st.st_mode) ||
       (st.st_mode & (S_IWGRP | S_IWOTH | S_IRGRP | S_IROTH)) != 0) {
      DEBUG("pass file "<<fnrc<<": wrong permissions "<<
            (st.st_mode & 0777) << " (should be 0600)");
      return -2;
   }
   // Open the file
   FILE *fid = fopen(fnrc.c_str(), "r");
   if (!fid) {
      DEBUG("cannot open file "<<fnrc<<" (errno:"<<errno<<")");
      return -1;
   }
   char line[512];
   int nm = 0, nmmx = -1;
   while (fgets(line, sizeof(line), fid) != 0) {
      if (line[0] == '#')
         continue;
      char word[6][128];
      int nword = sscanf(line, "%s %s %s %s %s %s", word[0], word[1],
                         word[2], word[3], word[4], word[5]);
      if (nword != 6) continue;
      if (strcmp(word[0], "machine") || strcmp(word[2], "login") ||
          strcmp(word[4], "password"))
         continue;
      // Good entry format
      if ((nm = host.matches(word[1])) > 0) {
         // Host matches
         if (!strcmp(hs->User.c_str(),word[3])) {
            // User matches: if exact match we are done
            if (nm == host.length()) {
               passwd = word[5];
               status = kpCI_exact;
               break;
            } 
            // Else, we focalise on the best match
            if (nm > nmmx) {
               nmmx = nm;
               passwd = word[5];
               status = kpCI_wildcard;
            }
         }
      }
   }
   //
   // Close the file
   fclose(fid);
   //
   // We are done
   if (passwd.length() > 0)
      return 0;
   return -1;
}

//______________________________________________________________________________
bool XrdSecProtocolpwd::CheckTimeStamp(XrdSutBuffer *bm, int skew, String &emsg)
{
   // Check consistency of the time stamp in bucket kXRS_timestamp in bm;
   // skew is the allowed difference in times.
   // Return 1 if ok, 0 if not
   EPNAME("CheckTimeStamp");

   // Check inputs
   if (!bm || skew <= 0) { 
      if (!bm)
         emsg = "input buffer undefined ";
      else
         emsg = "negative skew: invalid ";
      return 0;
   }

   // We check only if requested and a stronger check has not been done
   // successfully already
   if (hs->RtagOK || VeriClnt != 1) {
      DEBUG("Nothing to do");
      // Deactivate the buffer, if there
      if (bm->GetBucket(kXRS_timestamp))
          bm->Deactivate(kXRS_timestamp);
      return 1;
   }

   //
   // Add bucket with our version to the main list
   kXR_int32 tstamp = 0;
   if (bm->UnmarshalBucket(kXRS_timestamp,tstamp) != 0) {
      emsg = "bucket with time stamp not found";
      return 0;
   }

   kXR_int32 dtim = hs->TimeStamp - tstamp;
   dtim = (dtim < 0) ? -dtim : dtim;
   if (dtim > skew) {
      emsg = "time difference too big: "; emsg += (int)dtim;
      emsg += " - allowed skew: "; emsg += skew;
      bm->Deactivate(kXRS_timestamp);
      return 0;
   }
   bm->Deactivate(kXRS_timestamp);

   DEBUG("Time stamp successfully checked");

   // Ok
   return 1;
}

//______________________________________________________________________________
bool XrdSecProtocolpwd::CheckRtag(XrdSutBuffer *bm, String &emsg)
{
   // Check random tag signature if it was sent with previous packet
   EPNAME("CheckRtag");

   // Make sure we got a buffer
   if (!bm) {
      emsg = "Buffer not defined";
      return 0;
   }
   //
   // If we sent out a random tag check it signature
   if (hs->Cref && hs->Cref->buf1.len > 0) {
      XrdSutBucket *brt = 0;
      if ((brt = bm->GetBucket(kXRS_signed_rtag))) {
         // Make suer we got a cipher
         if (!(hs->Hcip)) {
            emsg = "Session cipher undefined";
            return 0;
         }
         // Decrypt it with the session cipher
         if (!(hs->Hcip->Decrypt(*brt))) {
            emsg = "error decrypting random tag with session cipher";
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
