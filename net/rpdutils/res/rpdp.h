// @(#)root/rpdutils:$Id$
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

#include "Varargs.h"
#include "MessageTypes.h"
#include "rpderr.h"
#include <string>

/////////////////////////////////////////////////////////////////////
//                                                                 //
// Definition used by daemons                                      //
//                                                                 //
/////////////////////////////////////////////////////////////////////

#include "rpddefs.h"

#include "Rtypes.h" //const int  kMAXPATHLEN = kMAXSECBUF;

/////////////////////////////////////////////////////////////////////
//                                                                 //
// Prototypes used by daemons                                      //
//                                                                 //
/////////////////////////////////////////////////////////////////////

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
int    NetRecvAllocate(void *&buf, int &len, EMessageTypes &kind);
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
int  RpdGetAuthProtocol();
const char *RpdGetKeyRoot();
int  RpdGetClientProtocol();
int  RpdGetOffSet();
int  RpdInitSession(int, std::string &, int &);
int  RpdInitSession(int, std::string &, int &, int &, std::string &);
int  RpdInitSession(int, std::string &, int &, int &, int &, std::string &);
void RpdInit(EService serv, int pid, int sproto,
             unsigned int opts, int rumsk, int sshp,
             const char *tmpd, const char *asrpp, int login = 0);
void RpdSetErrorHandler(ErrorHandler_t Err, ErrorHandler_t Sys,
                        ErrorHandler_t Fatal);
#ifdef R__KRB5
void RpdSetKeytabFile(const char *keytabfile);
#endif
void RpdSetSysLogFlag(int syslog);
void RpdSetMethInitFlag(int methinit);
int  RpdUpdateAuthTab(int opt, const char *line, char **token, int ilck = 0);

} // namespace ROOT


/////////////////////////////////////////////////////////////////////
//                                                                 //
// Internal Definition and prototypes used by modules in rpdutils  //
//                                                                 //
/////////////////////////////////////////////////////////////////////

//
// Prototypes
//
namespace ROOT {

//
// Utility functions
char *ItoA(int i);

//
// net.cxx
int  NetRecv(char *msg, int max);
int  NetRecvRaw(int sock, void *buf, int len);
int  NetSend(const void *buf, int len, EMessageTypes kind);
int  NetSendAck();
void NetSetOptions(EService service, int sock, int tcpwindowsize);

//
// netpar.cxx
void NetParClose();
int  NetParRecv(void *buf, int len);
int  NetParSend(const void *buf, int len);

//
// rpdutils.cxx
int  RpdAuthenticate();
int  RpdCheckAuthAllow(int Sec, const char *Host);
int  RpdCheckAuthTab(int Sec, const char *User, const char *Host,
                     int RemId, int *OffSet);
int  RpdCheckDaemon(const char *daemon);
int  RpdCheckHost(const char *Host, const char *host);
int  RpdCheckOffSet(int Sec, const char *User, const char *Host, int RemId,
                    int *OffSet, char **tkn, int *shmid, char **glbsuser);
int  RpdCheckSpecialPass(const char *passwd);
 int RpdRetrieveSpecialPass(const char *usr, const char *fpw, char *pwd, int lmx);
int  RpdCheckSshd(int opt);
bool RpdCheckToken(char *tknin, char *tknref);
int  RpdCleanupAuthTab(const char *crypttoken);
int  RpdCleanupAuthTab(const char *Host, int RemId, int OffSet);
void RpdDefaultAuthAllow();
int  RpdDeleteKeyFile(int ofs);
void RpdFreeKeys();
int  RpdGetAuthMethod(int kind);
char *RpdGetIP(const char *host);
char *RpdGetRandString(int Opt, int Len);
int  RpdGetRSAKeys(const char *PubKey, int Opt);
int  RpdGlobusAuth(const char *sstr);
int  RpdGuessClientProt(const char *buf, EMessageTypes kind);
void RpdInitAuth();
void RpdInitRand();
int  RpdKrb5Auth(const char *sstr);
int  RpdLogin(int,int);
int  RpdNoAuth(int);
int  RpdPass(const char *pass, int errheq = 0);
int  RpdProtocol(int);
int  RpdRecvClientRSAKey();
int  RpdRenameKeyFile(int oofs, int nofs);
int  RpdReUseAuth(const char *sstr, int kind);
int  RpdRfioAuth(const char *sstr);
int  RpdSavePubKey(const char *PubKey, int OffSet, char *User);
int  RpdSecureRecv(char **Str);
int  RpdSecureSend(char *Str);
void RpdSendAuthList();
int  RpdSetUid(int uid);
int  RpdSRPUser(const char *user);
int  RpdSshAuth(const char *sstr);
int  RpdUser(const char *sstr);

//
// Ssh Utility Function prototypes ...
int  SshToolAllocateSocket(unsigned int, unsigned int, char **);
void SshToolDiscardSocket(const char *, int);
int  SshToolGetAuth(int);
int  SshToolGetAuth(int, const char *);
int  SshToolNotifyFailure(const char *);

} // namespace ROOT

#endif
