// @(#)root/proofd:$Name:$:$Id:$
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
void NetSend(const char *msg);
int  NetRecv(char *msg, int len);

#endif
