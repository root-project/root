// @(#)root/main:$Name:  $:$Id: rmain.cxx,v 1.6 2002/05/10 08:30:06 brun Exp $
// Author: Gerardo Ganis    1/7/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// ssh2rpd                                                              //
//                                                                      //
// Small program to communicate successful result of sshd auth to the   //
// relevant rootd and proofd daemons                                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <syslog.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include "Varargs.h"


const int  kMAXPATHLEN    = 1024;
char      *gFileLog       = 0;
int        gDebug         = 0;

//______________________________________________________________________________
void Info(const char *va_(fmt), ...)
{
   // Write info message to syslog.

   char    buf[1024];
   va_list ap;

   va_start(ap,va_(fmt));
   vsprintf(buf, fmt, ap);
   va_end(ap);

   if (gFileLog != 0 && strlen(gFileLog) > 0) {
      FILE *fl= fopen(gFileLog,"a");
      fprintf(fl, "%s",buf);
      fclose(fl);
   } else {
      syslog(LOG_INFO, buf);
   }
}

//______________________________________________________________________________
int main(int argc, char **argv)
{
   char *Pipe = 0; int ProId=-1, RemId=-1;

   if (argc < 2) {
      Info("ssh2rpd: argc=%d : at least one additional argument required - exit\n", argc);
      exit(1);
   }

   // Parse Arguments
   gDebug = atoi(argv[1]);
   if (argc > 2) Pipe     = strdup(argv[2]);
   if (argc > 3) ProId    = atoi(argv[3]);
   if (argc > 4) RemId    = atoi(argv[4]);
   if (argc > 5) gFileLog = strdup(argv[5]);


   if (gDebug > 0)
      Info("ssh2rpd: forked with args: %d %s %d %d '%s'\n",gDebug,Pipe,ProId,RemId,gFileLog);

   // Preparing socket connection
   struct sockaddr_un servAddr;
   servAddr.sun_family = AF_UNIX;
   strcpy(servAddr.sun_path,Pipe);
   int sd;
   if ((sd = socket(AF_UNIX, SOCK_STREAM, 0)) < 0) {
      Info("ssh2rpd: cannot open socket: exiting ");
      exit(1);
   }
   // Connecting to socket
   int rc;
   if ((rc = connect(sd, (struct sockaddr *) &servAddr, sizeof(servAddr))) < 0) {
      Info("ssh2rpd: cannot connect socket: exiting (errno= %d)",errno);
      exit(1);
   }

   // Sending "OK" ...
   char *okbuf = "OK";
   rc = send(sd, okbuf, strlen(okbuf), 0);
   if (rc != (int)strlen(okbuf)) {
      Info("ssh2rpd: sending might have been unsuccessful (bytes send: %d)",rc);
   }

   if (Pipe != 0) free(Pipe);
   if (gFileLog != 0) free(gFileLog);

   exit(0);
}
