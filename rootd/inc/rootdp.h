/* @(#)root/rootd:$Name:  $:$Id: rootdp.h,v 1.2 2001/01/26 16:44:35 rdm Exp $ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_rootdp
#define ROOT_rootdp


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// rootdp                                                               //
//                                                                      //
// This header file contains private definitions and declarations       //
// used by rootd.                                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Varargs.h"
#include "MessageTypes.h"
#include "rootd.h"

extern int  gInetdFlag;
extern int  gPort;
extern int  gDebug;
extern int  gSockFd;
extern int  gAuth;
extern int  gFd;
extern int  gParallel;
extern char gFile[];
extern char gOption[];
extern char gRemoteHost[];

extern double  gBytesRead;
extern double  gBytesWritten;
extern double  gBytesSent;
extern double  gBytesRecv;


void  DaemonStart(int ignsigcld);

void  ErrorInit(const char *ident);
void  ErrorInfo(const char *va_(fmt), ...);
void  ErrorSys(ERootdErrors code, const char *va_(fmt), ...);
void  ErrorFatal(ERootdErrors code, const char *va_(fmt), ...);
int   GetErrno();
void  ResetErrno();

void  NetInit(const char *service, int port, int tcpwindowsize);
int   NetOpen(int inetdflag);
void  NetClose();
char *NetRemoteHost();
void  NetSetOptions(int sock, int tcpwindowsize);
int   NetSendRaw(const void *buf, int len);
int   NetSend(const void *buf, int len, EMessageTypes kind);
int   NetSend(int code, EMessageTypes kind);
int   NetSend(const char *msg, EMessageTypes kind);
int   NetSendAck();
int   NetSendError(ERootdErrors err);
int   NetRecvRaw(void *buf, int len);
int   NetRecv(void *&buf, int &len, EMessageTypes &kind);
int   NetRecv(char *msg, int len, EMessageTypes &kind);

int   NetParOpen(int port, int size);
void  NetParClose();
int   NetParSend(const void *buf, int len);
int   NetParRecv(void *buf, int len);

void  RootdClose();
int   RootdIsOpen();

#endif
