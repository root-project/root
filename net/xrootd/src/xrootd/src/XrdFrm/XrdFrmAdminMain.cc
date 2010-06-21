/******************************************************************************/
/*                                                                            */
/*                    X r d F r m A d m i n M a i n . c c                     */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//           $Id$

const char *XrdFrmAdminMainCVSID = "$Id$";

/* This is the "main" part of the frm_PreStage command. Syntax is:
*/
static const char *XrdFrmOpts  = "c:dhn:v";
static const char *XrdFrmUsage =

  " [-c <cfgfile>] [-d] [-h] [-n name] [-v] [help | cmd & opts]\n";
/*
Where:

   -c     The configuration file. The default is '/opt/xrootd/etc/xrootd.cf'

   -d     Turns on debugging mode.

   -h     Print helpful information (other options ignored).

   -n     The instance name.

   cmd    Specific commands, see the help information.

   opts   Options specific to the command.
*/

/******************************************************************************/
/*                         i n c l u d e   f i l e s                          */
/******************************************************************************/
  
#include <unistd.h>
#include <ctype.h>
#include <errno.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>
#include <sys/param.h>

#ifdef HAVE_READLINE
#include <readline/readline.h>
#include <readline/history.h>
#endif

#include "XrdFrm/XrdFrmAdmin.hh"
#include "XrdFrm/XrdFrmConfig.hh"
#include "XrdFrm/XrdFrmTrace.hh"
#include "XrdNet/XrdNetOpts.hh"
#include "XrdNet/XrdNetSocket.hh"
#include "XrdOuc/XrdOucTokenizer.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysLogger.hh"

using namespace XrdFrm;
  
/******************************************************************************/
/*                      G l o b a l   V a r i a b l e s                       */
/******************************************************************************/

       XrdFrmConfig       XrdFrm::Config(XrdFrmConfig::ssAdmin,
                                         XrdFrmOpts, XrdFrmUsage);

       XrdFrmAdmin        XrdFrm::Admin;

// The following is needed to resolve symbols for objects included from xrootd
//
       XrdOucTrace       *XrdXrootdTrace;
       XrdSysError        XrdLog(0, "");
       XrdOucTrace        XrdTrace(&Say);

/******************************************************************************/
/*                              r e a d l i n e                               */
/******************************************************************************/

#ifndef HAVE_READLINE

// replacement function for GNU readline
//
char *readline(const char *prompt)
{
   char buff[4096];
  
   cout << prompt;
   if (!fgets(buff, 4096, stdin) || *buff == '\n' || !strlen(buff)) return 0;
   return strdup(buff);
}

void    add_history(const char *cLine) {}
void stifle_history(int hnum) {}
#endif

/******************************************************************************/
/*                                  m a i n                                   */
/******************************************************************************/
  
int main(int argc, char *argv[])
{
   XrdSysLogger Logger;
   sigset_t myset;
   XrdOucTokenizer Request(0);
   char *cLine = 0, *pLine = 0, *Cmd = 0, *CmdArgs;
   int IMode;

// Turn off sigpipe and host a variety of others before we start any threads
//
   signal(SIGPIPE, SIG_IGN);  // Solaris optimization
   sigemptyset(&myset);
   sigaddset(&myset, SIGPIPE);
   sigaddset(&myset, SIGCHLD);
   pthread_sigmask(SIG_BLOCK, &myset, NULL);

// Perform configuration
//
   Say.logger(&Logger);
   XrdLog.logger(&Logger);
   if (!Config.Configure(argc, argv, 0)) exit(4);

// Fill out the dummy symbol to avoid crashes
//
   XrdXrootdTrace = new XrdOucTrace(&Say);

// We either have a command line or need to enter interactive mode
//
   if (Config.nextArg >= argc) IMode = 1;
      else {Cmd = argv[Config.nextArg++];
            Admin.setArgs(argc-Config.nextArg, &argv[Config.nextArg]);
            IMode = 0;
           }

// Set readline history list (keep only 256 lines, max)
//
   if (IMode) stifle_history(256);

// Process the request(s)
//
   do {if (IMode)
          {if (!(cLine = readline("frm_admin> "))) Admin.Quit();
           if (!pLine || strcmp(pLine, cLine))
              {add_history(cLine);
               if (pLine) free(pLine);
               pLine = strdup(cLine);
              }
           Request.Attach(cLine);
           if (!Request.GetLine() || !(Cmd=Request.GetToken(&CmdArgs)))
              Admin.Quit();
           Admin.setArgs(CmdArgs);
          }
       Admin.xeqArgs(Cmd);
       if (cLine) free(cLine);
      } while(IMode);

// All done
//
   Admin.Quit();
}
