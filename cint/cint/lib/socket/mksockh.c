/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include "cintsock.h"

void createLongDefine(FILE *fp,char *name,long value) {
  fprintf(fp,"#define %s %ld\n",name,value);
}

void createULongDefine(FILE *fp,char *name,unsigned long value) {
  fprintf(fp,"#define %s %lu\n",name,value);
}

void createDoubleDefine(FILE *fp,char *name,double value) {
  fprintf(fp,"#define %s %g\n",name,value);
}

void createStringDefine(FILE *fp,char *name,char* value) {
  fprintf(fp,"#define %s \"%s\"\n",name,value);
}

int main() {
  FILE *fp = fopen("../../include/socket.h","w");

  fprintf(fp,"/***********************************************************\n");
  fprintf(fp,"* socket.h \n");
  fprintf(fp,"***********************************************************/\n");
  fprintf(fp,"#ifndef G__SOCKET_H\n");
  fprintf(fp,"#define G__SOCKET_H\n");
  fprintf(fp,"\n");

  fprintf(fp,"\n");
  fprintf(fp,"#pragma include_noerr \"cintsock.dll\"\n");
  fprintf(fp,"\n");
  fprintf(fp,"#ifndef G__CINTSOCK_H /* G__CINTSOCK_H is defined in socket.dll */\n");
  fprintf(fp,"#error cintsock.dll is not ready. Run setup script in $CINTSYSDIR/lib/socket directory\n");
  fprintf(fp,"#endif\n");
  fprintf(fp,"\n");
  fprintf(fp,"\n");


#ifdef AF_INET
  createLongDefine(fp,"AF_INET",AF_INET);
#endif

#ifdef AF_UNIX
  createLongDefine(fp,"AF_UNIX",AF_UNIX);
#endif

#ifdef AF_ISO
  createLongDefine(fp,"AF_ISO",AF_ISO);
#endif

#ifdef AF_NS
  createLongDefine(fp,"AF_NS",AF_NS);
#endif

#ifdef AF_IMPLINK
  createLongDefine(fp,"AF_IMPLINK",AF_IMPLINK);
#endif

#ifdef INADDR_ANY
  createULongDefine(fp,"INADDR_ANY",INADDR_ANY);
#endif

#ifdef SOCK_STREAM
  createLongDefine(fp,"SOCK_STREAM",SOCK_STREAM);
#endif

#ifdef SOCK_DGRAM
  createLongDefine(fp,"SOCK_DGRAM",SOCK_DGRAM);
#endif

#ifdef SOCK_RAW
  createLongDefine(fp,"SOCK_RAW",SOCK_RAW);
#endif

#ifdef SOCK_SEQPACKET
  createLongDefine(fp,"SOCK_SEQPACKET",SOCK_SEQPACKET);
#endif

#ifdef SOCK_RDM
  createLongDefine(fp,"SOCK_RDM",SOCK_RDM);
#endif

#ifdef MSG_OOB
  createLongDefine(fp,"MSG_OOB",MSG_OOB);
#endif

#ifdef MSG_DONTROUTE
  createLongDefine(fp,"MSG_DONTROUTE",MSG_DONTROUTE);
#endif

#ifdef INVALID_SOCKET
  createLongDefine(fp,"INVALID_SOCKET",INVALID_SOCKET);
#else
  createLongDefine(fp,"INVALID_SOCKET",-1);
#endif

#ifdef SOCKET_ERROR
  createLongDefine(fp,"SOCKET_ERROR",SOCKET_ERROR);
#else
  createLongDefine(fp,"SOCKET_ERROR",-1);
#endif

  fprintf(fp,"\n");

  fprintf(fp,"\n");
  fprintf(fp,"#endif\n");

  fclose(fp);

  return(0);
}

