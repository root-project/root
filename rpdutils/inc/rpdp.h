// @(#)root/rpdutils:$Name:  $:$Id: rpdp.h,v 1.18 2004/05/30 16:13:05 rdm Exp $
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
#include <string>
#ifdef R__GLOBALSTL
namespace std { using ::string; }
#endif

/////////////////////////////////////////////////////////////////////
//                                                                 //
// Definition and prototypes used by rootd and proofd              //
//                                                                 //
/////////////////////////////////////////////////////////////////////

//
// Typedefs
typedef void (*SigPipe_t)(int);

//
// Global consts
#include "AuthConst.h"
const int  kMAXRECVBUF       = 1024;
const int  kMAXPATHLEN       = kMAXSECBUF;
const int  kMAXUSERLEN       = 128;

// Masks for initialization options
const unsigned int kDMN_RQAUTH = 0x1;  // Require authentication
const unsigned int kDMN_INCTKN = 0x2;  // Create inclusive tokens
const unsigned int kDMN_HOSTEQ = 0x4;  // Allow host equivalence 
const unsigned int kDMN_SYSLOG = 0x8;  // Log messages to syslog i.o. stderr 

//
// type of service
enum  EService  { kSOCKD = 0, kROOTD, kPROOFD };

//
// rpdutils globals
namespace ROOT {

// Utility functions
int SPrintf(char *buf, size_t size, const char *va_(fmt), ...);

// error handling functions
void Error(ErrorHandler_t ErrHand,int code,const char *va_(fmt), ...);
void ErrorInfo(const char *va_(fmt), ...);
void ErrorInit(const char *ident);
int  GetErrno();
void Perror(char *buf, int size = kMAXPATHLEN);
void ResetErrno();

// network functions
void   NetClose();
double NetGetBytesRecv();
double NetGetBytesSent();
void   NetGetRemoteHost(std::string &host);
int    NetGetSockFd();
int    NetInit(EService service, int port1, int port2, int tcpwindowsize);
int    NetParOpen(int port, int size);
int    NetOpen(int inetdflag, EService service);
int    NetRecv(char *msg, int len, EMessageTypes &kind);
int    NetRecvRaw(void *buf, int len);
void   NetResetByteCount();
int    NetSend(int code, EMessageTypes kind);
int    NetSend(const char *msg, EMessageTypes kind = kMESS_STRING);
int    NetSendError(ERootdErrors err);
int    NetSendRaw(const void *buf, int len);
void   NetSetSigPipeHook(SigPipe_t hook);

// fork functionality
void DaemonStart(int ignsigcld, int fdkeep, EService service);

// rpdutils.cxx
void RpdAuthCleanup(const char *sstr, int opt);
int  RpdGenRSAKeys(int);
#ifdef R__GLBS
int  RpdGetShmIdCred();
#endif
int  RpdInitSession(int, std::string &, int &);
int  RpdInitSession(int, std::string &, int &, int &, std::string &);
void RpdInit(EService serv, int pid, int sproto, 
             unsigned int opts, int rumsk, int sshp, 
             const char *tmpd, const char *asrpp);
void RpdSetErrorHandler(ErrorHandler_t Err, ErrorHandler_t Sys,
                        ErrorHandler_t Fatal);
#ifdef R__KRB5
void RpdSetKeytabFile(const char *keytabfile);
#endif
void RpdSetSysLogFlag(int syslog);
int  RpdUpdateAuthTab(int opt, const char *line, char **token, int ilck = 0);

} // namespace ROOT


/////////////////////////////////////////////////////////////////////
//                                                                 //
// Internal Definition and prototypes used by modules in rpdutils  //
//                                                                 //
/////////////////////////////////////////////////////////////////////

//
// type of authentication method
enum  ESecurity { kClear, kSRP, kKrb5, kGlobus, kSSH, kRfio };

//
// Prototypes
//
namespace ROOT {

//
// Utility functions
char *ItoA(int i);

//
// net.cxx
int NetRecv(char *msg, int max);
int NetRecvRaw(int sock, void *buf, int len);
int NetSend(const void *buf, int len, EMessageTypes kind);
int NetSendAck();
void NetSetOptions(EService service, int sock, int tcpwindowsize);

//
// netpar.cxx
void NetParClose();
int  NetParRecv(void *buf, int len);
int  NetParSend(const void *buf, int len);

//
// rpdutils.cxx
void RpdAuthenticate();
int  RpdCheckAuthAllow(int Sec, const char *Host);
int  RpdCheckAuthTab(int Sec, const char *User, const char *Host,
                     int RemId, int *OffSet);
int  RpdCheckDaemon(const char *daemon);
int  RpdCheckHost(const char *Host, const char *host);
int  RpdCheckOffSet(int Sec, const char *User, const char *Host, int RemId,
                    int *OffSet, char **tkn, int *shmid, char **glbsuser);
int  RpdCheckSpecialPass(const char *passwd);
int  RpdCheckSshd(int opt);
bool RpdCheckToken(char *tknin, char *tknref);
int  RpdCleanupAuthTab(const char *Host, int RemId, int OffSet);
void RpdDefaultAuthAllow();
int  RpdDeleteKeyFile(int ofs);
void RpdFreeKeys();
int  RpdGetAuthMethod(int kind);
char *RpdGetIP(const char *host);
char *RpdGetRandString(int Opt, int Len);
int  RpdGetRSAKeys(const char *PubKey, int Opt);
void RpdGlobusAuth(const char *sstr);
int  RpdGuessClientProt(const char *buf, EMessageTypes kind);
void RpdInitAuth();
void RpdInitRand();
void RpdKrb5Auth(const char *sstr);
void RpdLogin(int);
void RpdNoAuth(int);
void RpdPass(const char *pass);
void RpdProtocol(int);
int  RpdRecvClientRSAKey();
int  RpdRenameKeyFile(int oofs, int nofs);
bool RpdReUseAuth(const char *sstr, int kind);
void RpdRfioAuth(const char *sstr);
void RpdSavePubKey(const char *PubKey, int OffSet, char *User);
int  RpdSecureRecv(char **Str);
int  RpdSecureSend(char *Str);
void RpdSendAuthList();
void RpdSetUid(int uid);
void RpdSRPUser(const char *user);
void RpdSshAuth(const char *sstr);
void RpdUser(const char *sstr);

//
// Ssh Utility Function prototypes ...
int  SshToolAllocateSocket(unsigned int, unsigned int, char **);
void SshToolDiscardSocket(const char *, int);
int  SshToolGetAuth(int);
int  SshToolGetAuth(int, const char *);
int  SshToolNotifyFailure(const char *);

} // namespace ROOT

//
// Globus stuff ...
//
#ifdef R__GLBS
#define HAVE_MEMMOVE 1
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

#endif
