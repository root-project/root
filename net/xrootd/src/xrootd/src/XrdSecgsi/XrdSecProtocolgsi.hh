// $Id$
/******************************************************************************/
/*                                                                            */
/*                 X r d S e c P r o t o c o l g s i . h h                    */
/*                                                                            */
/* (c) 2005 G. Ganis / CERN                                                   */
/*                                                                            */
/******************************************************************************/
#include <XrdOuc/XrdOucErrInfo.hh>
#include <XrdSys/XrdSysPthread.hh>
#include <XrdOuc/XrdOucString.hh>
#include <XrdOuc/XrdOucTokenizer.hh>

#include <XrdSec/XrdSecInterface.hh>
#include <XrdSecgsi/XrdSecgsiTrace.hh>

#include <XrdSut/XrdSutPFEntry.hh>
#include <XrdSut/XrdSutPFile.hh>
#include <XrdSut/XrdSutBuffer.hh>
#include <XrdSut/XrdSutRndm.hh>

#include <XrdCrypto/XrdCryptoAux.hh>
#include <XrdCrypto/XrdCryptoCipher.hh>
#include <XrdCrypto/XrdCryptoFactory.hh>
#include <XrdCrypto/XrdCryptoX509Crl.hh>

#include <XrdCrypto/XrdCryptosslgsiX509Chain.hh>

/******************************************************************************/
/*                               D e f i n e s                                */
/******************************************************************************/

typedef XrdOucString String;
typedef XrdCryptosslgsiX509Chain X509Chain;
  
#define XrdSecPROTOIDENT    "gsi"
#define XrdSecPROTOIDLEN    sizeof(XrdSecPROTOIDENT)
#define XrdSecgsiVERSION    10200
#define XrdSecNOIPCHK       0x0001
#define XrdSecDEBUG         0x1000
#define XrdCryptoMax        10

#define kMAXBUFLEN          1024

//
// Message codes either returned by server or included in buffers
enum kgsiStatus {
   kgST_error    = -1,      // error occured
   kgST_ok       =  0,      // ok
   kgST_more     =  1       // need more info
};

// Client steps
enum kgsiClientSteps {
   kXGC_none = 0,
   kXGC_certreq     = 1000, // 1000: request server certificate
   kXGC_cert,               // 1001: packet with (proxy) certificate
   kXGC_sigpxy,             // 1002: packet with signed proxy certificate
   kXGC_reserved            // 
};

// Server steps
enum kgsiServerSteps {
   kXGS_none = 0,
   kXGS_init       = 2000,   // 2000: fake code used the first time 
   kXGS_cert,                // 2001: packet with certificate 
   kXGS_pxyreq,              // 2002: packet with proxy req to be signed 
   kXGS_reserved             //
};

// Handshake options
enum kgsiHandshakeOpts {
   kOptsDlgPxy     = 1,      // 0x0001: Ask for a delegated proxy
   kOptsFwdPxy     = 2,      // 0x0002: Forward local proxy
   kOptsSigReq     = 4,      // 0x0004: Accept to sign delegated proxy
   kOptsSrvReq     = 8,      // 0x0008: Server request for delegated proxy
   kOptsPxFile     = 16,     // 0x0010: Save delegated proxies in file
   kOptsDelChn     = 32      // 0x0020: Delete chain
};

// Error codes
enum kgsiErrors {
   kGSErrParseBuffer = 10000,       // 10000
   kGSErrDecodeBuffer,              // 10001
   kGSErrLoadCrypto,                // 10002
   kGSErrBadProtocol,               // 10003
   kGSErrCreateBucket,              // 10004
   kGSErrDuplicateBucket,           // 10005
   kGSErrCreateBuffer,              // 10006
   kGSErrSerialBuffer,              // 10007
   kGSErrGenCipher,                 // 10008
   kGSErrExportPuK,                 // 10009
   kGSErrEncRndmTag,                // 10010
   kGSErrBadRndmTag,                // 10011
   kGSErrNoRndmTag,                 // 10012
   kGSErrNoCipher,                  // 10013
   kGSErrNoCreds,                   // 10014
   kGSErrBadOpt,                    // 10015
   kGSErrMarshal,                   // 10016
   kGSErrUnmarshal,                 // 10017
   kGSErrSaveCreds,                 // 10018
   kGSErrNoBuffer,                  // 10019
   kGSErrRefCipher,                 // 10020
   kGSErrNoPublic,                  // 10021
   kGSErrAddBucket,                 // 10022
   kGSErrFinCipher,                 // 10023
   kGSErrInit,                      // 10024
   kGSErrBadCreds,                  // 10025
   kGSErrError                      // 10026  
};

#define REL1(x)     { if (x) delete x; }
#define REL2(x,y)   { if (x) delete x; if (y) delete y; }
#define REL3(x,y,z) { if (x) delete x; if (y) delete y; if (z) delete z; }

#define SafeDelete(x) { if (x) delete x ; x = 0; }
#define SafeDelArray(x) { if (x) delete [] x ; x = 0; }
#define SafeFree(x) { if (x) free(x) ; x = 0; }

// External function for DN-username mapping
typedef char *(*XrdSecgsiGMAP_t)(const char *, int);

//
// This a small class to set the relevant options in one go
//
class gsiOptions {
public:
   short  debug;  // [cs] debug flag
   short  mode;   // [cs] 'c' or 's'
   char  *clist;  // [s] list of crypto modules ["ssl" ]
   char  *certdir;// [cs] dir with CA info [/etc/grid-security/certificates]
   char  *crldir; // [cs] dir with CRL info [/etc/grid-security/certificates]
   char  *crlext; // [cs] extension of CRL files [.r0]
   char  *cert;   // [s] server certificate [/etc/grid-security/root/rootcert.pem]
                  // [c] user certificate [$HOME/.globus/usercert.pem]
   char  *key;    // [s] server private key [/etc/grid-security/root/rootkey.pem]
                  // [c] user private key [$HOME/.globus/userkey.pem]
   char  *cipher; // [s] list of ciphers [aes-128-cbc:bf-cbc:des-ede3-cbc]
   char  *md;     // [s] list of MDs [sha1:md5]
   int    crl;    // [cs] check level of CRL's [1] 
   int    ca;     // [cs] verification level of CA's [1] 
   char  *proxy;  // [c] user proxy  [/tmp/x509up_u<uid>]
   char  *valid;  // [c] proxy validity  [12:00]
   int    deplen; // [c] depth of signature path for proxies [0] 
   int    bits;   // [c] bits in PKI for proxies [512] 
   char  *gridmap;// [s] gridmap file [/etc/grid-security/gridmap]
   int    gmapto; // [s] validity in secs of grid-map cache entries [-1 => unlimited]
   char  *gmapfun;// [s] file with the function to map DN to usernames [0]
   char  *gmapfunparms;// [s] parameters for the function to map DN to usernames [0]
   int    ogmap;  // [s] gridmap file checking option 
   int    dlgpxy; // [c] explicitely ask the creation of a delegated proxy 
                  // [s] ask client for proxies
   int    sigpxy; // [c] accept delegated proxy requests 
   char  *srvnames;// [c] '|' separated list of allowed server names
   char  *exppxy; // [s] template for the exported file with proxies (dlgpxy == 3)

   gsiOptions() { debug = -1; mode = 's'; clist = 0; 
                  certdir = 0; crldir = 0; crlext = 0; cert = 0; key = 0;
                  cipher = 0; md = 0; ca = 1 ; crl = 1;
                  proxy = 0; valid = 0; deplen = 0; bits = 512;
                  gridmap = 0; gmapto = -1; gmapfun = 0; gmapfunparms = 0; ogmap = 1;
                  dlgpxy = 0; sigpxy = 1; srvnames = 0; exppxy = 0;}
   virtual ~gsiOptions() { } // Cleanup inside XrdSecProtocolgsiInit
};

class XrdSecProtocolgsi;
class gsiHSVars {
public:
   int               Iter;          // iteration number
   int               TimeStamp;     // Time of last call
   String            CryptoMod;     // crypto module in use
   int               RemVers;       // Version run by remote counterpart
   XrdCryptoCipher  *Rcip;          // reference cipher
   XrdSutBucket     *Cbck;          // Bucket with the certificate in export form
   String            ID;            // Handshake ID (dummy for clients)
   XrdSutPFEntry    *Cref;          // Cache reference
   XrdSutPFEntry    *Pent;          // Pointer to relevant file entry 
   X509Chain        *Chain;         // Chain to be eventually verified 
   XrdCryptoX509Crl *Crl;           // Pointer to CRL, if required 
   X509Chain        *PxyChain;      // Proxy Chain on clients
   bool              RtagOK;        // Rndm tag checked / not checked
   bool              Tty;           // Terminal attached / not attached
   int               LastStep;      // Step required at previous iteration
   int               Options;       // Handshake options;
   XrdSutBuffer     *Parms;         // Buffer with server parms on first iteration 

   gsiHSVars() { Iter = 0; TimeStamp = -1; CryptoMod = "";
                 RemVers = -1; Rcip = 0;
                 Cbck = 0;
                 ID = ""; Cref = 0; Pent = 0; Chain = 0; Crl = 0; PxyChain = 0;
                 RtagOK = 0; Tty = 0; LastStep = 0; Options = 0; Parms = 0;}

   ~gsiHSVars() { SafeDelete(Cref);
                  if (Options & kOptsDelChn) {
                     // Do not delete the CA certificate in the cached reference
                     if (Chain) Chain->Cleanup(1);
                     SafeDelete(Chain);
                  }
                  // The proxy chain is owned by the proxy cache; invalid proxies are
                  // detected (and eventually removed) by QueryProxy
                  PxyChain = 0;
                  SafeDelete(Parms); }
   void Dump(XrdSecProtocolgsi *p = 0);
};

// From a proxy query
typedef struct {
   X509Chain        *chain;
   XrdCryptoRSA     *ksig;
   XrdSutBucket     *cbck;
} ProxyOut_t;

// To query proxies
typedef struct {
   const char *cert;
   const char *key;
   const char *certdir;
   const char *out;
   const char *valid;
   int         deplen;
   int         bits;
} ProxyIn_t;

/******************************************************************************/
/*              X r d S e c P r o t o c o l g s i   C l a s s                 */
/******************************************************************************/

class XrdSecProtocolgsi : public XrdSecProtocol
{
public:
        int                Authenticate  (XrdSecCredentials *cred,
                                          XrdSecParameters **parms,
                                          XrdOucErrInfo     *einfo=0);

        XrdSecCredentials *getCredentials(XrdSecParameters  *parm=0,
                                          XrdOucErrInfo     *einfo=0);

        XrdSecProtocolgsi(int opts, const char *hname,
                          const struct sockaddr *ipadd, const char *parms = 0);
        virtual ~XrdSecProtocolgsi() {} // Delete() does it all

        // Initialization methods
        static char      *Init(gsiOptions o, XrdOucErrInfo *erp);

        void              Delete();

        // Encrypt / Decrypt methods
        int               Encrypt(const char *inbuf, int inlen,
                                  XrdSecBuffer **outbuf);
        int               Decrypt(const char *inbuf, int inlen,
                                  XrdSecBuffer **outbuf);
        // Sign / Verify methods
        int               Sign(const char *inbuf, int inlen,
                               XrdSecBuffer **outbuf);
        int               Verify(const char *inbuf, int inlen,
                                 const char *sigbuf, int siglen);

        // Export session key
        int               getKey(char *kbuf=0, int klen=0);
        // Import a key
        int               setKey(char *kbuf, int klen);

private:

   // Static members initialized at startup
   static XrdSysMutex      gsiContext;
   static String           CAdir;
   static String           CRLdir;
   static String           DefCRLext;
   static String           SrvCert;
   static String           SrvKey;
   static String           UsrProxy;
   static String           UsrCert;
   static String           UsrKey;
   static String           PxyValid;
   static int              DepLength;
   static int              DefBits;
   static int              CACheck;
   static int              CRLCheck;
   static String           DefCrypto;
   static String           DefCipher;
   static String           DefMD;
   static String           DefError;
   static String           GMAPFile;
   static int              GMAPOpt;
   static int              GMAPCacheTimeOut;
   static XrdSysPlugin    *GMAPPlugin;
   static XrdSecgsiGMAP_t  GMAPFun;
   static int              PxyReqOpts;
   static String           SrvAllowedNames;
   //
   // Crypto related info
   static int              ncrypt;                  // Number of factories
   static XrdCryptoFactory *cryptF[XrdCryptoMax];   // their hooks 
   static int              cryptID[XrdCryptoMax];   // their IDs 
   static String           cryptName[XrdCryptoMax]; // their names 
   static XrdCryptoCipher *refcip[XrdCryptoMax];    // ref for session ciphers 
   //
   // Caches 
   static XrdSutCache      cacheCA;   // Info about trusted CA's
   static XrdSutCache      cacheCert; // Cache for available server certs
   static XrdSutCache      cachePxy;  // Cache for client proxies
   static XrdSutCache      cacheGMAP; // Cache for gridmap entries
   static XrdSutCache      cacheGMAPFun; // Cache for entries mapped by GMAPFun
   //
   // Running options / settings
   static int              Debug;          // [CS] Debug level
   static bool             Server;         // [CS] If server mode 
   static int              TimeSkew;       // [CS] Allowed skew in secs for time stamps 
   //
   // for error logging and tracing
   static XrdSysLogger     Logger;
   static XrdSysError      eDest;
   static XrdOucTrace     *GSITrace;

   // Information local to this instance
   int              options;
   struct sockaddr  hostaddr;      // Client-side only
   XrdCryptoFactory *sessionCF;    // Chosen crypto factory
   XrdCryptoCipher *sessionKey;    // Session Key (result of the handshake)
   XrdSutBucket    *bucketKey;     // Bucket with the key in export form
   XrdCryptoMsgDigest *sessionMD;  // Message Digest instance
   XrdCryptoRSA    *sessionKsig;   // RSA key to sign
   XrdCryptoRSA    *sessionKver;   // RSA key to verify
   X509Chain       *proxyChain;    // Chain with the delegated proxy on servers
   bool             srvMode;       // TRUE if server mode 

   // Temporary Handshake local info
   gsiHSVars     *hs;

   // Parsing received buffers: client
   int            ParseClientInput(XrdSutBuffer *br, XrdSutBuffer **bm,
                                   String &emsg);
   int            ClientDoInit(XrdSutBuffer *br, XrdSutBuffer **bm,
                               String &cmsg);
   int            ClientDoCert(XrdSutBuffer *br,  XrdSutBuffer **bm,
                               String &cmsg);
   int            ClientDoPxyreq(XrdSutBuffer *br,  XrdSutBuffer **bm,
                                 String &cmsg);

   // Parsing received buffers: server
   int            ParseServerInput(XrdSutBuffer *br, XrdSutBuffer **bm,
                                   String &cmsg);
   int            ServerDoCertreq(XrdSutBuffer *br, XrdSutBuffer **bm,
                                  String &cmsg);
   int            ServerDoCert(XrdSutBuffer *br,  XrdSutBuffer **bm,
                               String &cmsg);
   int            ServerDoSigpxy(XrdSutBuffer *br,  XrdSutBuffer **bm,
                                 String &cmsg);

   // Auxilliary functions
   int            ParseCrypto(String cryptlist);
   int            ParseCAlist(String calist);

   // Load CA certificates
   static int     LoadCADir(int timestamp);
   int            GetCA(const char *cahash);
   static String  GetCApath(const char *cahash);
   static bool    VerifyCA(int opt, X509Chain *cca, XrdCryptoFactory *cf);
   bool           ServerCertNameOK(const char *subject, String &e);

   // Load CRLs
   static XrdCryptoX509Crl *LoadCRL(XrdCryptoX509 *xca,
                                    XrdCryptoFactory *CF);

   // Updating proxies
   static int     QueryProxy(bool checkcache, XrdSutCache *cache, const char *tag,
                             XrdCryptoFactory *cf, int timestamp,
                             ProxyIn_t *pi, ProxyOut_t *po);
   static int     InitProxy(ProxyIn_t *pi,
                            X509Chain *ch = 0, XrdCryptoRSA **key = 0);

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

   // Check Time stamp
   bool           CheckTimeStamp(XrdSutBuffer *b, int skew, String &emsg);

   // Check random challenge
   bool           CheckRtag(XrdSutBuffer *bm, String &emsg);

   // Auxilliary methods
   int            AddSerialized(char opt, kXR_int32 step, String ID, 
                                XrdSutBuffer *bls, XrdSutBuffer *buf,
                                kXR_int32 type, XrdCryptoCipher *cip);
   // Grid map cache handling
   static int     LoadGMAP(int now); // Init or refresh the cache
   static XrdSecgsiGMAP_t            // Load alternative function for mapping
                  LoadGMAPFun(const char *plugin, const char *parms);
   static void    QueryGMAP(const char *dn, int now, String &name); //Lookup info for DN
};
