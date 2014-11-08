// @(#)root/auth:$Id$
// Author: Gerri Ganis  19/1/2004

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_DaemonUtils
#define ROOT_DaemonUtils


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// DaemonUtils                                                          //
//                                                                      //
// This file defines wrappers to client utils calls used by server      //
// authentication daemons.                                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <string>

#ifndef ROOT_TSocket
#include "TSocket.h"
#endif
#ifndef ROOT_TSeqCollection
#include "TSeqCollection.h"
#endif
#ifndef ROOT_NetErrors
#include "NetErrors.h"
#endif
#ifndef ROOT_rpddefs
#include "rpddefs.h"
#endif


extern Int_t SrvAuthImpl(TSocket *socket, const char *, const char *,
                         std::string &user, Int_t &meth,
                         Int_t &type, std::string &ctoken, TSeqCollection *);
extern Int_t SrvClupImpl(TSeqCollection *);

typedef void (*ErrorHandler_t)(int level, const char *msg, int size);


namespace ROOT {

// Error handlers prototypes ...
extern ErrorHandler_t gErrSys;
extern ErrorHandler_t gErrFatal;
extern ErrorHandler_t gErr;

int  GetErrno();
void ResetErrno();
void ErrorInit(const char *ident);
void ErrorInfo(const char *fmt, ...);
void Perror(char *buf, int size);
void Error(ErrorHandler_t ErrHand,int code,const char *fmt, ...);

void RpdAuthCleanup(const char *sstr, int opt);
int  RpdCleanupAuthTab(const char *crypttoken);
int  RpdGenRSAKeys(int);
void RpdSetErrorHandler(ErrorHandler_t Err, ErrorHandler_t Sys,
                        ErrorHandler_t Fatal);
void RpdSetMethInitFlag(int methinit);
int  RpdInitSession(int, std::string &, int &, int &, int &, std::string &);
void RpdInit(EService serv, int pid, int sproto,
             unsigned int opts, int rumsk, int sshp,
             const char *tmpd, const char *asrpp, int login = 0);

void SrvSetSocket(TSocket *socket);

void NetClose();
int NetParOpen(int port, int size);
int NetRecv(char *msg, int max);
int NetRecv(char *msg, int len, EMessageTypes &kind);
int NetRecv(void *&buf, int &len, EMessageTypes &kind);
int NetRecvRaw(void *buf, int len);
int NetRecvRaw(int sock, void *buf, int len);
int NetSend(int code, EMessageTypes kind);
int NetSend(const char *msg, EMessageTypes kind);
int NetSend(const void *buf, int len, EMessageTypes kind);
int NetSendAck();
int NetSendError(ERootdErrors err);
int NetSendRaw(const void *buf, int len);
void NetGetRemoteHost(std::string &openhost);
int NetGetSockFd();

}

#endif
