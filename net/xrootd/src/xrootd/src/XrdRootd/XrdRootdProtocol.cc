/******************************************************************************/
/*                                                                            */
/*                   X r d R o o t d P r o t o c o l . c c                    */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//       $Id$ 

const char *XrdRootdProtocolCVSID = "$Id$";

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <netinet/in.h>

#include "XrdSys/XrdSysError.hh"
#include "XrdRootd/XrdRootdProtocol.hh"
#include "Xrd/XrdLink.hh"
#include "Xrd/XrdScheduler.hh"
#undef   XRD_TRACE
#define  XRD_TRACE XrdTrace->
#include "Xrd/XrdTrace.hh"

/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/
  
      int   XrdRootdProtocol::Count = 0;

const char *XrdRootdProtocol::TraceID = "Rootd: ";

/******************************************************************************/
/*                         L o c a l   D e f i n e s                          */
/******************************************************************************/
  
#define MAX_ARGS 128

/******************************************************************************/
/*                       P r o t o c o l   L o a d e r                        */
/******************************************************************************/
  
// This protocol is meant to live in a shared library. The interface below is
// used by the server to obtain a copy of the protocol object that can be used
// to decide whether or not a link is talking a particular protocol.
//
extern "C"
{
XrdProtocol *XrdgetProtocol(const char *pname, char *parms,
                     XrdProtocol_Config *pi)
{
   char *pc, *pgm, *fn;
   int acnt = 0;
   char *parg[MAX_ARGS], **pap;

// If a command is available, then this protocol can be used
//
   if (!(pc = parms))
      {pi->eDest->Say(0,"rootd: Protocol handler command not specified");
       return (XrdProtocol *)0;
      }
   if (*pc != '/')
      {pi->eDest->Say(0,"rootd: Protocol handler command is not absolute");
       return (XrdProtocol *)0;
      }

// Make sure the command is actually an executable program
//
   while(*pc && *pc == ' ') pc++;
   pgm = pc;
   while(*pc && *pc != ' ') pc++;
   if (*pc) {*pc = '\0'; pc++;}
   if (access(pgm, F_OK) || access(pgm, X_OK))
      {pi->eDest->Emsg("rootd" ,errno, "find rootd program", pgm);
       return (XrdProtocol *)0;
      }

// Establish the first argument (filename)
//
   fn = pc-1;
   while(*fn != '/' && fn != pgm) fn--;
   parg[0] = strdup(++fn);

// Force first argument to the program be '-i', so the user does not
// need to specify it in the config file; and if it does, the second
// instance will be ignored
//
   parg[1] = strdup("-i");

// Tokenize the remaining arguments (we do not support quotes)
//
   for (acnt = 2; acnt < MAX_ARGS-1 && *pc; acnt++)
       {while(*pc && *pc == ' ') pc++;
        parg[acnt] = pc;
        while(*pc && *pc != ' ') pc++;
        if (*pc) {*pc = '\0'; pc++;}
        parg[acnt] = strdup(parg[acnt]);
       }

// Check if we execeeded the arg count (yeah, we leak if an error occurs but
// this program will be shortly terminated if we do).
//
   if (*pc)
      {pi->eDest->Say("rootd: Too many arguments to program ", pgm);
       return (XrdProtocol *)0;
      }

// Copy the arglist
//
   parg[acnt++] = 0;
   pap = (char **)malloc(acnt*sizeof(char *));
   memcpy((void *)pap, (const void *)parg, acnt*sizeof(char *));

// Issue herald
//
  pi->eDest->Say(0, "rootd protocol interface V 1.1 successfully loaded.");

// Return the protocol object to be used
//
   return (XrdProtocol *)new XrdRootdProtocol(pi,strdup(pgm),(const char **)pap);
}
}

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/

XrdRootdProtocol::XrdRootdProtocol(XrdProtocol_Config *pi,
                               const char *pgm, const char **pap) :
                               XrdProtocol("rootd protocol")
{
    Program   = pgm;
    ProgArg   = pap;

    eDest     = pi->eDest;
    Scheduler = pi->Sched;
    XrdTrace = pi->Trace;
    stderrFD  = eDest->baseFD();
    ReadWait  = pi->readWait;
}

/******************************************************************************/
/*                                 M a t c h                                  */
/******************************************************************************/

#define TRACELINK lp
  
XrdProtocol *XrdRootdProtocol::Match(XrdLink *lp)
{
     struct handshake
           {int eight;
           } hsdata;

     char  *hsbuff = (char *)&hsdata;
     pid_t pid;
     int  dlen;

// Peek at the first 4 bytes of data
//
   if ((dlen = lp->Peek(hsbuff, sizeof(hsdata), ReadWait)) != sizeof(hsdata))
      {lp->setEtext("rootd handshake not received");
       return (XrdProtocol *)0;
      }

// Verify that this is our protocol
//
   hsdata.eight = ntohl(hsdata.eight);
   if (dlen != sizeof(hsdata) || hsdata.eight != 8) return (XrdProtocol *)0;
   Count++;
   TRACEI(PROT, "Matched rootd protocol on link");

// Trace this now since we won't be able to do much after the fork()
//
   TRACEI(PROT, "executing " <<Program);

// Fork a process to handle this protocol
//
   if ((pid = Scheduler->Fork(lp->Name())))
      {if (pid < 0) lp->setEtext("rootd fork failed");
          else lp->setEtext("link transfered");
       return (XrdProtocol *)0;
      }

// Restablish standard error for the program we will exec
//
   dup2(stderrFD, STDERR_FILENO);
   close(stderrFD);

// Force stdin/out to point to the socket FD (this will also bypass the
// close on exec setting for the socket)
//
   dup2(lp->FDnum(), STDIN_FILENO);
   dup2(lp->FDnum(), STDOUT_FILENO);

// Do the exec
//
   execv((const char *)Program, (char * const *)ProgArg);
   cerr <<"Xrdrootd: Oops! Exec(" <<Program <<") failed; errno=" <<errno <<endl;
   _exit(17);
}
 
/******************************************************************************/
/*                                 S t a t s                                  */
/******************************************************************************/
  
int XrdRootdProtocol::Stats(char *buff, int blen, int do_sync)
{

   static char statfmt[] = "<stats id=\"rootd\"><num>%ld</num></stats>";

// If caller wants only size, give it to him
//
   if (!buff) return sizeof(statfmt)+16;

// We have only one statistic -- number of successful matches, ignore do_sync
//
   return snprintf(buff, blen, statfmt, Count);
}
