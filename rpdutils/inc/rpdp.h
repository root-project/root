// @(#)root/rpdutils:$Name:  $:$Id: rpdp.h,v 1.7 2003/10/22 18:48:36 rdm Exp $
// Author: Gerardo Ganis   7/4/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_rpdp
#define ROOT_rpdp


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// rpdp                                                                 //
//                                                                      //
// This header file contains private definitions and declarations       //
// used by modules in rpdutils/src.                                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_VarArgs
#include "Varargs.h"
#endif
#ifndef ROOT_MessageTypes
#include "MessageTypes.h"
#endif
#ifndef ROOT_rpderr
#include "rpderr.h"
#endif

typedef void (*SigPipe_t)(int);

// Global consts
const int  kMAXSEC           = 6;
const int  kMAXSECBUF        = 2048;
const int  kAUTH_REUSE_MSK   = 0x1;
const int  kAUTH_CRYPT_MSK   = 0x2;
const int  kAUTH_SSALT_MSK   = 0x4;
const int  kMAXPATHLEN       = 1024;
const int  kMAXTABSIZE       = 1000000000;
const int  kMAXRSATRIES      = 100;

// type of authentication method
enum  ESecurity { kClear, kSRP, kKrb5, kGlobus, kSSH, kRfio };

// type of service
enum  EService  { kROOTD = 1, kPROOFD };

// Extern globals
namespace ROOT {

extern int  gAltSRP;
extern int  gAnon;
extern int  gAuth;
extern int  gClientProtocol;
extern int  gGlobus;
extern int  gNumAllow;
extern int  gNumLeft;
extern int  gOffSet;
extern int  gParallel;
extern int  gPort;
extern int  gRemPid;
extern int  gReUseAllow;
extern int  gReUseRequired;
extern int  gSaltRequired;
extern int  gSockFd;
extern int  gSshdPort;

extern char gAltSRPPass[kMAXPATHLEN];
extern char gAnonUser[64];
extern char gSystemDaemonRc[kMAXPATHLEN];
extern char gExecDir[kMAXPATHLEN];        // for use in rootd ...
extern char gFile[kMAXPATHLEN];
extern char gFileLog[kMAXPATHLEN];
extern char gHostCertConf[kMAXPATHLEN];   // defines certificate location for globus authentication
extern char gOpenHost[256];
extern char gService[10];                 // "rootd" or "proofd", defined in proofd/rootd.cxx ...
extern char gTmpDir[kMAXPATHLEN];         // directory for temporary files (RW)

extern char gUser[64];
extern char gPasswd[64];                  // only used for anonymous access

extern const char *kAuthMeth[kMAXSEC];    // authentication method list
extern const char kDaemonRc[];        // file containing daemon access rules

extern SigPipe_t gSigPipeHook;

extern double gBytesSent;
extern double gBytesRecv;

// Error handlers prototypes ...
extern ErrorHandler_t gErrSys;
extern ErrorHandler_t gErrFatal;
extern ErrorHandler_t gErr;

} // namespace ROOT


namespace ROOT {

// error.cxx
int  GetErrno();
void ResetErrno();
void ErrorInit(const char *ident);
void ErrorInfo(const char *va_(fmt), ...);
void Perror(char *buf);
void Error(ErrorHandler_t ErrHand,int code,const char *va_(fmt), ...);

// net.cxx
int NetSendRaw(const void *buf, int len);
int NetRecvRaw(void *buf, int len);
int NetRecvRaw(int sock, void *buf, int len);
int NetSend(const void *buf, int len, EMessageTypes kind);
int NetSend(int code, EMessageTypes kind);
int NetSend(const char *msg, EMessageTypes kind = kMESS_STRING);
int NetSendAck();
int NetSendError(ERootdErrors err);
int NetRecv(void *&buf, int &len, EMessageTypes &kind);
int NetRecv(char *msg, int len, EMessageTypes &kind);
int NetRecv(char *msg, int max);
int NetOpen(int inetdflag, EService service);
void NetClose();
const char *NetRemoteHost();
int  NetInit(const char *service, int &port1, int port2, int tcpwindowsize);
void NetInit(const char *service, int port, int tcpwindowsize);
void NetSetOptions(EService service, int sock, int tcpwindowsize);

// netpar.cxx
int  NetParOpen(int port, int size);
void NetParClose();
int  NetParSend(const void *buf, int len);
int  NetParRecv(void *buf, int len);

// daemon.cxx
void DaemonStart(int ignsigcld, int fdkeep, EService service);

// rpdutils.cxx
int  RpdGetAuthMethod(int kind);
int  RpdUpdateAuthTab(int opt, char *line, char **token);
int  RpdCleanupAuthTab(char *Host, int RemId);
int  RpdCheckAuthTab(int Sec, char *User, char *Host,int RemId, int *OffSet);
bool RpdReUseAuth(const char *sstr, int kind);
int  RpdCheckAuthAllow(int Sec, char *Host);
int  RpdCheckHostWild(const char *Host, const char *host);
char *RpdGetIP(const char *host);
void RpdSendAuthList();
void RpdCheckSession();

void RpdUser(const char *sstr);
void RpdSshAuth(const char *sstr);
void RpdKrb5Auth(const char *sstr);
void RpdSRPUser(const char *user);
int  RpdCheckSpecialPass(const char *passwd);
void RpdPass(const char *pass);
void RpdGlobusAuth(const char *sstr);
void RpdRfioAuth(const char *sstr);
void RpdCleanup(const char *sstr);

void RpdDefaultAuthAllow();
int  RpdCheckDaemon(const char *daemon);
int  RpdCheckSshd();
int  RpdGuessClientProt(const char *buf, EMessageTypes kind);
char *RpdGetRandString(int Opt, int Len);
bool RpdCheckToken(char *tknin, char *tknref);

void RpdSetAuthTabFile(char *AuthTabFile);
void RpdSetDebugFlag(int Debug);
void RpdSetRootLogFlag(int RootLog);

void RpdInitRand();
int  RpdGenRSAKeys();
int  RpdGetRSAKeys(char *PubKey, int Opt);
int  RpdRecvClientRSAKey();
void RpdSavePubKey(char *PubKey, int OffSet, char *User);
int  RpdSecureSend(char *Str);
int  RpdSecureRecv(char **Str);


} // namespace ROOT

// Globus stuff ...
#ifdef R__GLBS
extern "C" {
#ifdef R__GLBS22
   #include <globus_gsi_credential.h>
#else
   #include <sslutils.h>
#endif
   #include <globus_gss_assist.h>
   #include <openssl/x509.h>
   #include <openssl/pem.h>
   #include <sys/ipc.h>
   #include <sys/shm.h>
}
// Globus Utility Function prototypes ...
namespace ROOT {

void  GlbsToolError(char *, int, int, int);
int   GlbsToolCheckCert(char *, char **);
int   GlbsToolCheckContext(int);
int   GlbsToolCheckProxy(char *, char **);
int   GlbsToolStoreContext(gss_ctx_id_t, char *);
int   GlbsToolStoreToShm(gss_buffer_t, int *);
char *GlbsToolExpand(char *);

} // namespace ROOT

#endif  // Globus ...

namespace ROOT {

// Ssh Utility Function prototypes ...
int   SshToolAllocateSocket(unsigned int, unsigned int, char **);
void  SshToolDiscardSocket(const char *, int);
int   SshToolNotifyFailure(const char *);
int   SshToolGetAuth(int);

} // namespace ROOT

#endif
