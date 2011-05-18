/******************************************************************************/
/*                                                                            */
/*                 X r d S e c P r o t o c o l s s l . h h                    */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <unistd.h>
#include <ctype.h>
#include <errno.h>
#include <iostream>
#include <stdlib.h>
#include <strings.h>
#include <grp.h>
#include <pwd.h>

#define OPENSSL_THREAD_DEFINES
#include <openssl/opensslconf.h>

#include <openssl/crypto.h>
#include <openssl/x509v3.h>
#include <openssl/ssl.h>
#include <openssl/pem.h>
#include <openssl/err.h>
#include <openssl/rand.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/file.h>
#include <fcntl.h>
#include <pwd.h>
#include <grp.h>

#include "XrdNet/XrdNetDNS.hh"
#include "XrdOuc/XrdOucErrInfo.hh"
#include "XrdOuc/XrdOucHash.hh"
#include "XrdOuc/XrdOucString.hh"
#include "XrdOuc/XrdOucTrace.hh"
#include "XrdOuc/XrdOucTokenizer.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdSec/XrdSecInterface.hh"
#include "XrdSec/XrdSecTLayer.hh"
#include "XrdSecssl/XrdSecProtocolsslTrace.hh"
#include "XrdSecssl/XrdSecProtocolsslProc.hh"
#include "libsslGridSite/grst_verifycallback.h"
#include "libsslGridSite/gridsite.h"

#define EXPORTKEYSTRENGTH 10

#define PROTOCOLSSL_MAX_CRYPTO_MUTEX 256


// fix for SSL 098 stuff and g++ 

#ifdef R__SSL_GE_098
#undef PEM_read_SSL_SESSION
#undef PEM_write_SSL_SESSION

#define PEM_read_SSL_SESSION(fp,x,cb,u) (SSL_SESSION *)PEM_ASN1_read( (void *(*)(void **, const unsigned char **, long int))d2i_SSL_SESSION,PEM_STRING_SSL_SESSION,fp,(void **)x,cb,u)

#define PEM_write_SSL_SESSION(fp,x) PEM_ASN1_write((int (*)(void*, unsigned char**))i2d_SSL_SESSION, PEM_STRING_SSL_SESSION,fp, (char *)x,  NULL,NULL,0,NULL,NULL)
#else
#if defined(__APPLE__) && !defined(MAC_OS_X_VERSION_10_5)
#undef PEM_read_SSL_SESSION
#undef PEM_write_SSL_SESSION

#define PEM_read_SSL_SESSION(fp,x,cb,u) (SSL_SESSION *)PEM_ASN1_read( (char *(*)(...))d2i_SSL_SESSION,PEM_STRING_SSL_SESSION,fp,(char **)x  ,cb,u)
#define PEM_write_SSL_SESSION(fp,x) PEM_ASN1_write((int(*)(...))i2d_SSL_SESSION, PEM_STRING_SSL_SESSION,fp, (char *)x,NULL,NULL,0,NULL,NULL)
#endif
#endif

#define l2n(l,c)        (*((c)++)=(unsigned char)(((l)>>24)&0xff), \
                         *((c)++)=(unsigned char)(((l)>>16)&0xff), \
                         *((c)++)=(unsigned char)(((l)>> 8)&0xff), \
                         *((c)++)=(unsigned char)(((l)    )&0xff))

#ifdef SUNCC
#define __FUNCTION__ "-unknown-"
#endif

static XrdOucTrace        *SSLxTrace=0;

class XrdSecProtocolssl;

#define MAX_SESSION_ID_ATTEMPTS 10

/******************************************************************************/
/*              X r d S e c P r o t o c o l s s l C l a s s                   */
/******************************************************************************/

class XrdSecsslSessionLock {
private:
static  XrdSysMutex sessionmutex;
  int sessionfd;
  
public:
  XrdSecsslSessionLock() {sessionfd=0;}
  bool SoftLock() { sessionmutex.Lock();return true;}
  bool SoftUnLock() {sessionmutex.UnLock();return true;}
#ifdef SUNCC
  bool HardLock(const char* path) {return true;}
  bool HardUnLock() {return true;}
  ~XrdSecsslSessionLock() {sessionmutex.UnLock();}
#else
  bool HardLock(const char* path) {sessionfd = open(path,O_RDWR); if ( (sessionfd>0) && (!flock(sessionfd,LOCK_EX)))return true;return false;}
  bool HardUnLock() {if (sessionfd>0) {flock(sessionfd,LOCK_UN);close(sessionfd);sessionfd=0;}return true;}
  ~XrdSecsslSessionLock() {if (sessionfd>0) {flock(sessionfd,LOCK_UN);close(sessionfd);};}
#endif

};



class XrdSecProtocolssl : public XrdSecTLayer
{
public:
  friend class XrdSecProtocolDummy; // Avoid stupid gcc warnings about destructor

  XrdSecProtocolssl(const char* hostname, const struct sockaddr  *ipaddr) : XrdSecTLayer("ssl",XrdSecTLayer::isClient) {
    credBuff    = 0;
    ssl         = 0;
    Entity.name = 0;
    Entity.grps = 0;
    Entity.endorsements = 0;
    strncpy(Entity.prot,"ssl", sizeof(Entity.prot));
    host        = hostname;
    if (ipaddr)
      Entity.host = (XrdNetDNS::getHostName((sockaddr&)*ipaddr));
    else 
      Entity.host = strdup("");
    proxyBuff[0]=0;
    client_cert=0;
    server_cert=0;
    ssl = 0 ;
    clientctx = 0;
    terminate = 0;
  }
  
  virtual void   secClient(int theFD, XrdOucErrInfo      *einfo);
  virtual void   secServer(int theFD, XrdOucErrInfo      *einfo=0);

  // triggers purging of expired SecTLayer threads
  static  int    dummy(const char* key, XrdSecProtocolssl *ssl, void* Arg) { return 0;}

  // delayed garbage collection
  virtual void              Delete() {
    terminate = true;
    if (secTid) XrdSysThread::Join(secTid,NULL);
    secTid=0;
    SSLMutex.Lock();
    if (credBuff)    free(credBuff);
    if (Entity.name) free(Entity.name);
    if (Entity.grps) free(Entity.grps);
    if (Entity.role) free(Entity.role);
    if (Entity.host) free(Entity.host);
    if (ssl) SSL_free(ssl);
    if (client_cert) X509_free(client_cert);
    if (server_cert) X509_free(server_cert);
    credBuff = 0;
    Entity.name = 0;
    Entity.grps = 0;
    Entity.role = 0;
    Entity.host = 0;
    client_cert = 0;
    server_cert = 0;
    ssl=0;
    secTid=0;
    SSLMutex.UnLock();
    delete this;
  }


  static int GenerateSession(const SSL* ssl, unsigned char *id, unsigned int *id_len);
  static int NewSession(SSL* ssl, SSL_SESSION *pNew);
  static int GetSession(SSL* ssl, SSL_SESSION *pNew);

  static char*              SessionIdContext ;
  static char*              sslcadir; 
  static char*              sslvomsdir;
  static char*              sslserverkeyfile; 
  static char*              sslkeyfile;
  static char*              sslcertfile;
  static char*              sslproxyexportdir;
  static bool               sslproxyexportplain;
  static char               sslserverexportpassword[EXPORTKEYSTRENGTH+1];
  static int                threadsinuse;
  static char*              gridmapfile;
  static char*              vomsmapfile;
  static bool               mapuser;
  static bool               mapnobody;
  static bool               mapgroup;
  static bool               mapcerncertificates;
  static int                debug;
  static time_t             sslsessionlifetime;
  static int                sslselecttimeout;
  static int                sslsessioncachesize;
  static char*              procdir;
  static XrdSecProtocolsslProc* proc;

  static int                errortimeout;
  static int                errorverify;
  static int                errorqueue;
  static int                erroraccept;
  static int                errorabort;
  static int                errorread;
  static int                forwardedproxies;

  static bool               isServer;
  static bool               forwardProxy;
  static bool               allowSessions;
  static X509_STORE*        store;  
  static X509_LOOKUP*       lookup;
  static int                verifydepth;
  static int                verifyindex;
  int                       sessionfd;
  X509*    client_cert; 
  X509*    server_cert;
  XrdOucString              host;

  // User/Group mapping
  static void ReloadGridMapFile();
  static void ReloadVomsMapFile();
  static bool VomsMapGroups(const char* groups, XrdOucString& allgroups, XrdOucString& defaultgroup);

  static void GetEnvironment();
  static  XrdOucHash<XrdOucString>  gridmapstore;
  static  XrdOucHash<XrdOucString>  vomsmapstore;
  static  XrdOucHash<XrdOucString>  stringstore;
  static  XrdSysMutex               StoreMutex;
  static  XrdSysMutex               VomsMapMutex;
  static  XrdSysMutex               GridMapMutex;
  static  XrdSysMutex*              CryptoMutexPool[PROTOCOLSSL_MAX_CRYPTO_MUTEX];
  static  XrdSysMutex               ThreadsInUseMutex;
  static  XrdSysMutex               ErrorMutex;

  // for error logging and tracing
  static XrdSysLogger       Logger;
  static XrdSysError        ssleDest;
  static time_t             storeLoadTime;
  
  typedef struct {
    int verbose_mode;
    int verify_depth;
    int always_continue;
  } sslverify_t;
  
  char proxyBuff[16384];
  static SSL_CTX* ctx;
  SSL_CTX* clientctx;

  XrdSysMutex SSLMutex;
  bool terminate;
  ~XrdSecProtocolssl() {
  }

  static int Fatal(XrdOucErrInfo *erp, const char* msg, int rc);
  
  
  struct sockaddr           hostaddr;      // Client-side only
  char                     *credBuff;      // Credentials buffer (server)
  int                       Step;          // Indicates step in authentication
  
  int sd;
  int listen_sd;
  struct sockaddr_in sa_serv;
  struct sockaddr_in sa_cli;
  SSL*     ssl;
};

extern "C"
{
  char  *XrdSecProtocolsslInit(const char     mode,
			       const char    *parms,
			       XrdOucErrInfo *erp);
}


class XrdSecsslThreadInUse {
public:
  XrdSecsslThreadInUse() {XrdSecProtocolssl::ThreadsInUseMutex.Lock();XrdSecProtocolssl::threadsinuse++;XrdSecProtocolssl::ThreadsInUseMutex.UnLock();}
  ~XrdSecsslThreadInUse() {XrdSecProtocolssl::ThreadsInUseMutex.Lock();XrdSecProtocolssl::threadsinuse--;XrdSecProtocolssl::ThreadsInUseMutex.UnLock();}
};


