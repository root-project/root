// @(#)root/proofd:$Name:  $:$Id: proofdp.h,v 1.1 2000/12/15 19:38:35 rdm Exp $
// Author: Fons Rademakers   15/12/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_proofdp
#define ROOT_proofdp


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// proofdp                                                              //
//                                                                      //
// This header file contains private definitions and declarations       //
// used by proofd.                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Varargs.h"
#include "MessageTypes.h"

extern int  gInetdFlag;
extern int  gPort;
extern int  gDebug;
extern int  gSockFd;
extern int  gAuth;

void DaemonStart(int ignsigcld);

void ErrorInit(const char *ident);
void ErrorInfo(const char *va_(fmt), ...);
void ErrorSys(const char *va_(fmt), ...);
void ErrorFatal(const char *va_(fmt), ...);
int  GetErrno();
void ResetErrno();

void NetInit(const char *service, int port);
int  NetOpen(int inetdflag);
void NetClose();
int  NetSendRaw(const void *buf, int len);
int  NetSend(const void *buf, int len, EMessageTypes kind);
int  NetSend(int code, EMessageTypes kind);
int  NetSend(const char *msg, EMessageTypes kind = kMESS_STRING);
int  NetRecvRaw(void *buf, int len);
int  NetRecv(void *&buf, int &len, EMessageTypes &kind);
int  NetRecv(char *msg, int len, EMessageTypes &kind);
int  NetRecv(char *msg, int len);

#endif
