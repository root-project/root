// $Id$
/******************************************************************************/
/*                                                                            */
/*                 X r d S e c P r o t o c o l p w d . h h                    */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
#include <XrdOuc/XrdOucErrInfo.hh>
#include <XrdSys/XrdSysPthread.hh>
#include <XrdOuc/XrdOucString.hh>
#include <XrdOuc/XrdOucTokenizer.hh>

#include <XrdSec/XrdSecInterface.hh>
#include <XrdSec/XrdSecTrace.hh>

#include <XrdSut/XrdSutPFEntry.hh>
#include <XrdSut/XrdSutPFile.hh>
#include <XrdSut/XrdSutBuffer.hh>
#include <XrdSut/XrdSutRndm.hh>

#include <XrdCrypto/XrdCryptoAux.hh>
#include <XrdCrypto/XrdCryptoCipher.hh>
#include <XrdCrypto/XrdCryptoFactory.hh>

/******************************************************************************/
/*                               D e f i n e s                                */
/******************************************************************************/

typedef XrdOucString String;
  
#define XrdSecPROTOIDENT    "pwd"
#define XrdSecPROTOIDLEN    sizeof(XrdSecPROTOIDENT)
#define XrdSecpwdVERSION    10100
#define XrdSecNOIPCHK       0x0001
#define XrdSecDEBUG         0x1000
#define XrdCryptoMax        10

#define kMAXBUFLEN          1024
#define kMAXUSRLEN          9
#define kMAXPWDLEN          64

//
// Message codes either returned by server or included in buffers
enum kpwdStatus {
   kpST_error    = -1,      // error occured
   kpST_ok       =  0,      // ok
   kpST_more     =  1       // need more info
};

//
// Auto-reg modes
enum kpwdAutoreg {
   kpAR_none       =  0,      // autoreg disabled
   kpAR_users      =  1,      // only for tags in password files (local, system's) 
   kpAR_all        =  2       // for all tags
};

//
// Client update autologin modes
enum kpwdUpdate {
   kpUP_none       =  0,      // no update
   kpUP_remove     =  1,      // remove obsolete entries only 
   kpUP_all        =  2       // remove obsolete entries and register new valid info 
};

//
// Creds input type
enum kpwdCredsInput {
   kpCI_undef      = -1,      // undefined
   kpCI_prompt     =  0,      // from prompt
   kpCI_exact      =  1,      // from FileNetRc, exact tag 
   kpCI_wildcard   =  2       // from FileNetRc, wildcard tag
};

//
// Creds type (for prompt)
enum kpwdCredType {
   kpCT_undef      = -1,      // undefined
   kpCT_normal     =  0,      // confirmed credentials
   kpCT_onetime    =  1,      // one-time credentials
   kpCT_old        =  2,      // old credentials to be changed 
   kpCT_new        =  3,      // new credentials to be confirmed
   kpCT_newagain   =  4,      // new credentials again for confirmation
   kpCT_autoreg    =  5,      // autoreg: new creds to be confirmed
   kpCT_ar_again   =  6,      // autoreg: new creds again for confirmation
   kpCT_crypt      =  7,      // standard crypt hash
   kpCT_afs        =  8,      // AFS plain password
   kpCT_afsenc     =  9       // AFS encrypted password
};

//
// Creds actions
enum kpwdCredsActions {
   kpCA_undef      = -1,      // undefined
   kpCA_check      =  0,      // normal check of credentials
   kpCA_checkold   =  1,      // check current creds before asking for new ones 
   kpCA_cache      =  2,      // cache received (new) credentials
   kpCA_checkcache =  3       // check cached credentials and save them, if ok
};

// Client steps
enum kpwdClientSteps {
   kXPC_none = 0,
   kXPC_normal     = 1000, // 1000: standard packet
   kXPC_verifysrv,         // 1001: request for server verification 
   kXPC_signedrtag,        // 1002: signed rtag (after server request for verification)
   kXPC_creds,             // 1003: credentials packet
   kXPC_autoreg,           // 1004: query for autoregistration
   kXPC_failureack,        // 1005: failure acknowledgement
   kXPC_reserved           // 
};

// Server steps
enum kpwdServerSteps {
   kXPS_none = 0,
   kXPS_init       = 2000,   // 2000: fake code used the first time 
   kXPS_credsreq,            // 2001: request for credentials 
   kXPS_rtag,                // 2002: rndm tag to be signed (strong verification)
   kXPS_signedrtag,          // 2003: signed rtag (after client request for verification)
   kXPS_newpuk,              // 2004: new public part for session ciphers 
   kXPS_puk,                 // 2005: public part for session ciphers (after autoreg)
   kXPS_failure,             // 2006: signal failure to client to drop invalid cached info
   kXPS_reserved             //
};

// Error codes
enum kpwdErrors {
   kPWErrParseBuffer = 10000,       // 10000
   kPWErrDecodeBuffer,              // 10001
   kPWErrLoadCrypto,                // 10002
   kPWErrBadProtocol,               // 10003
   kPWErrNoUserHost,                // 10004
   kPWErrNoUser,                    // 10005
   kPWErrNoHost,                    // 10006
   kPWErrBadUser,                   // 10007
   kPWErrCreateBucket,              // 10008
   kPWErrDuplicateBucket,           // 10009
   kPWErrCreateBuffer,              // 10010
   kPWErrSerialBuffer,              // 10011
   kPWErrGenCipher,                 // 10012
   kPWErrExportPuK,                 // 10013
   kPWErrEncRndmTag,                // 10014
   kPWErrBadRndmTag,                // 10015
   kPWErrNoRndmTag,                 // 10016
   kPWErrNoCipher,                  // 10017
   kPWErrQueryCreds,                // 10018
   kPWErrNoCreds,                   // 10019
   kPWErrBadPasswd,                 // 10020
   kPWErrBadCache,                  // 10021
   kPWErrNoCache,                   // 10022
   kPWErrNoSessID,                  // 10023
   kPWErrBadSessID,                 // 10024
   kPWErrBadOpt,                    // 10025
   kPWErrMarshal,                   // 10026
   kPWErrUnmarshal,                 // 10027
   kPWErrSaveCreds,                 // 10028
   kPWErrNoSalt,                    // 10029
   kPWErrNoBuffer,                  // 10030
   kPWErrRefCipher,                 // 10031
   kPWErrNoPublic,                  // 10032
   kPWErrAddBucket,                 // 10033
   kPWErrFinCipher,                 // 10034
   kPWErrInit,                      // 10034
   kPWErrBadCreds,                  // 10035
   kPWErrError                      // 10036  
};

// Structuring the status word
typedef struct {
   char  ctype;
   char  action;
   short options; 
} pwdStatus_t;

#define REL1(x)     { if (x) delete x; }
#define REL2(x,y)   { if (x) delete x; if (y) delete y; }
#define REL3(x,y,z) { if (x) delete x; if (y) delete y; if (z) delete z; }

#ifndef NODEBUG
#define PRINT(y) {{SecTrace->Beg(epname); cerr <<y; SecTrace->End();}}
#else
#define PRINT(y) { }
#endif

#define SafeDelete(x) { if (x) delete x ; x = 0; }
#define SafeDelArray(x) { if (x) delete [] x ; x = 0; }

//
// This a small class to set the relevant options in one go
//
class pwdOptions {
public:
   short  debug;        // [cs] debug flag
   short  mode;         // [cs] 'c' or 's'
   short  areg;         // [cs] auto-registration opt (s); update-autolog-info opt (c)
   short  upwd;         // [s] check / do-not-check pwd file in user's $HOME
   short  alog;         // [c] check / do-not-check user's autologin info
   short  verisrv;      // [c] verify / do-not-verify server ownership of srvpuk
   short  vericlnt;     // [s] level of verification client ownership of clntpuk
   short  syspwd;       // [s] check / do-not-check system pwd (requires privileges) 
   int    lifecreds;    // [s] lifetime in seconds of credentials
   int    maxprompts;   // [c] max number of empty prompts
   int    maxfailures;  // [s] max passwd failures before blocking
   char  *clist;        // [s] list of crypto modules ["ssl"]
   char  *dir;          // [s] directory with admin pwd files [$HOME/.xrd]
   char  *udir;         // [s] users's sub-directory with pwd files [$HOME/.xrd]
   char  *cpass;        // [s] users's crypt hash pwd file [$HOME/.xrootdpass]
   char  *alogfile;     // [c] autologin file [$HOME/.xrd/pwdnetrc]
   char  *srvpuk;       // [c] file with server puks [$HOME/.xrd/pwdsrvpuk]
   short  keepcreds;    // [s] keep / do-not-keep client credentials 
   char  *expcreds;     // [s] (template for) file with exported creds

   pwdOptions() { debug = -1; mode = 's'; areg = -1; upwd = -1; alog = -1;
                  verisrv = -1; vericlnt = -1;
                  syspwd = -1; lifecreds = -1; maxprompts = -1; maxfailures = -1;
                  clist = 0; dir = 0; udir = 0; cpass = 0;
                  alogfile = 0; srvpuk = 0; keepcreds = 0; expcreds = 0;}
   virtual ~pwdOptions() { } // Cleanup inside XrdSecProtocolpwdInit
};

class pwdHSVars {
public:
   int               Iter;          // iteration number
   int               TimeStamp;     // Time of last call
   String            CryptoMod;     // crypto module in use
   String            User;          // remote username
   String            Tag;           // tag for credentials
   int               RemVers;       // Version run by remote counterpart
   XrdCryptoFactory *CF;            // crypto factory
   XrdCryptoCipher  *Hcip;          // handshake cipher
   XrdCryptoCipher  *Rcip;          // reference cipher
   String            ID;            // Handshake ID (dummy for clients)
   XrdSutPFEntry    *Cref;          // Cache reference
   XrdSutPFEntry    *Pent;          // Pointer to relevant file entry 
   bool              RtagOK;        // Rndm tag checked / not checked
   pwdStatus_t       Status;        // Some state flags
   bool              Tty;           // Terminal attached / not attached
   int               Step;          // Current step
   int               LastStep;      // Step required at previous iteration
   String            ErrMsg;        // Last error message
   int               SysPwd;        // 0 = no, 1 = Unix sys pwd, 2 = AFS pwd
   String            AFScell;       // AFS cell if it makes sense
   XrdSutBuffer     *Parms;         // Buffer with server parms on first iteration 

   pwdHSVars() { Iter = 0; TimeStamp = -1; CryptoMod = ""; User = ""; Tag = "";
                 RemVers = -1; CF = 0; Hcip = 0; Rcip = 0;
                 ID = ""; Cref = 0; Pent = 0; RtagOK = 0; Tty = 0;
                 Step = 0; LastStep = 0; ErrMsg = "";
                 SysPwd = 0; AFScell = "";
                 Status.ctype = 0; Status.action = 0; Status.options = 0; Parms = 0;}

   ~pwdHSVars() { SafeDelete(Cref); SafeDelete(Hcip); SafeDelete(Parms); }
};


/******************************************************************************/
/*              X r d S e c P r o t o c o l p w d   C l a s s                 */
/******************************************************************************/

class XrdSecProtocolpwd : public XrdSecProtocol
{
public:
        int                Authenticate  (XrdSecCredentials *cred,
                                          XrdSecParameters **parms,
                                          XrdOucErrInfo     *einfo=0);

        XrdSecCredentials *getCredentials(XrdSecParameters  *parm=0,
                                          XrdOucErrInfo     *einfo=0);

        XrdSecProtocolpwd(int opts, const char *hname,
                          const struct sockaddr *ipadd,
                          const char *parms = 0);
        virtual ~XrdSecProtocolpwd() {} // Delete() does it all

        // Initialization methods
        static char      *Init(pwdOptions o, XrdOucErrInfo *erp);

        void              Delete();

   static void       PrintTimeStat();

private:

   // Static members initialized at startup
   static XrdSysMutex      pwdContext;
   static String           FileAdmin;
   static String           FileExpCreds;     // (Template for) file with exported creds [S]
   static String           FileUser;
   static String           FileCrypt;
   static String           FileSrvPuk;
   static String           SrvID;
   static String           SrvEmail; 
   static String           DefCrypto;
   static String           DefError;
   static XrdSutPFile      PFAdmin;          // Admin file [S]
   static XrdSutPFile      PFAlog;           // Autologin file [CS]
   static XrdSutPFile      PFSrvPuk;         // File with server public keys [CS]
   //
   // Crypto related info
   static int              ncrypt;                  // Number of factories
   static int              cryptID[XrdCryptoMax];   // their IDs 
   static String           cryptName[XrdCryptoMax]; // their names 
   static XrdCryptoCipher *loccip[XrdCryptoMax];    // local ciphers
   static XrdCryptoCipher *refcip[XrdCryptoMax];    // ref for session ciphers 
   //
   // Caches for info files
   static XrdSutCache      cacheAdmin;  // Admin file
   static XrdSutCache      cacheSrvPuk; // SrvPuk file
   static XrdSutCache      cacheUser;   // User files
   static XrdSutCache      cacheAlog;   // Autologin file
   //
   // Running options / settings
   static int              Debug;          // [CS] Debug level
   static bool             Server;         // [CS] If server mode 
   static int              UserPwd;        // [S] Check passwd file in user's <xrdsecdir> 
   static bool             SysPwd;         // [S] Check system passwd file if allowed 
   static int              VeriClnt;       // [S] Client verification level
   static int              VeriSrv;        // [C] Server verification level
   static int              AutoReg;        // [S] Autoreg mode 
   static int              LifeCreds;      // [S] if > 0, credential lifetime in secs
   static int              MaxPrompts;     // [C] Repeating prompt
   static int              MaxFailures;    // [S] Max passwd failures before blocking
   static int              AutoLogin;      // [C] do-not-check/check/update autolog info
   static int              TimeSkew;       // [CS] Allowed skew in secs for time stamps 
   static bool             KeepCreds;      // [S] Keep / Do-Not-Keep client creds
   //
   // for error logging and tracing
   static XrdSysLogger     Logger;
   static XrdSysError      eDest;
   static XrdOucTrace     *SecTrace;

   // Information local to this instance
   int                     options;
   struct sockaddr         hostaddr;      // Client-side only
   char                    CName[256];    // Client-name
   bool                    srvMode;       // TRUE if server mode 

   // Handshake local info
   pwdHSVars              *hs;

   // Acquired credentials (server side)
   XrdSecCredentials      *clientCreds;

   // Parsing received buffers
   int            ParseClientInput(XrdSutBuffer *br, XrdSutBuffer **bm,
                                   String &emsg);
   int            ParseServerInput(XrdSutBuffer *br, XrdSutBuffer **bm,
                                   String &cmsg);
   int            ParseCrypto(XrdSutBuffer *buf);

   // Error functions
   static void    ErrF(XrdOucErrInfo *einfo, kXR_int32 ecode,
                       const char *msg1, const char *msg2 = 0,
                       const char *msg3 = 0);
   XrdSecCredentials *ErrC(XrdOucErrInfo *einfo, XrdSutBuffer *b1,
                           XrdSutBuffer *b2,XrdSutBuffer *b3,
                           kXR_int32 ecode, const char *msg1 = 0,
                           const char *msg2 = 0, const char *msg3 = 0);
   int            ErrS(String ID, XrdOucErrInfo *einfo, XrdSutBuffer *b1,
                       XrdSutBuffer *b2, XrdSutBuffer *b3,
                       kXR_int32 ecode, const char *msg1 = 0,
                       const char *msg2 = 0, const char *msg3 = 0);

   // Query methods
   XrdSutBucket  *QueryCreds(XrdSutBuffer *bm, bool netrc, int &status);
   int            QueryUser(int &status, String &cmsg);
   int            QueryCrypt(String &fn, String &pwhash);
   int            QueryNetRc(String host, String &passwd, int &status);

   // Check credentials
   bool           CheckCreds(XrdSutBucket *creds, int credtype);
   bool           CheckCredsAFS(XrdSutBucket *creds, int ctype);

   // Check Time stamp
   bool           CheckTimeStamp(XrdSutBuffer *b, int skew, String &emsg);

   // Check random challenge
   bool           CheckRtag(XrdSutBuffer *bm, String &emsg);

   // Saving / Updating
   int            ExportCreds(XrdSutBucket *creds);
   int            SaveCreds(XrdSutBucket *creds);
   int            UpdateAlog();

   // Auxilliary methods
   int            GetUserHost(String &usr, String &host);
   int            AddSerialized(char opt, kXR_int32 step, String ID, 
                                XrdSutBuffer *bls, XrdSutBuffer *buf,
                                kXR_int32 type, XrdCryptoCipher *cip);
   int            DoubleHash(XrdCryptoFactory *cf, XrdSutBucket *bck,
                             XrdSutBucket *s1, XrdSutBucket *s2 = 0,
                             const char *tag = 0);
};
