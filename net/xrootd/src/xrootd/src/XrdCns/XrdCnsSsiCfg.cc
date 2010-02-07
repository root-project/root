/******************************************************************************/
/*                                                                            */
/*                       X r d C n s S s i C f g . c c                        */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdCnsSsiCfgCVSID = "$Id$";

#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>

#include "XrdOuc/XrdOuca2x.hh"
#include "XrdOuc/XrdOucArgs.hh"
#include "XrdOuc/XrdOucTList.hh"
#include "XrdOuc/XrdOucUtils.hh"

#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysLogger.hh"

#include "XrdCns/XrdCnsLog.hh"
#include "XrdCns/XrdCnsSsiCfg.hh"
#include "XrdCns/XrdCnsSsiSay.hh"

/******************************************************************************/
/*           G l o b a l   C o n f i g u r a t i o n   O b j e c t            */
/******************************************************************************/

namespace XrdCns
{
       XrdCnsSsiCfg      Config;

extern XrdSysError       MLog;

extern XrdCnsSsiSay      Say;
}

using namespace XrdCns;
  
/******************************************************************************/
/*                             C o n f i g u r e                              */
/******************************************************************************/

int XrdCnsSsiCfg::Configure(int argc, char **argv)
{
   const char *Opts = 0;

// Determine the function and configure for that
//
        if (argc <= 1) Usage("Function not specified.");
   else if (!strcmp("diff", argv[1])) {Xeq = 'd'; Opts = "fFhmnps";
           Say.M("The diff function is not yet implemented.");
           return 0;
          }
   else if (!strcmp("list", argv[1])) {Xeq = 'l'; Opts = "fhlmnpsS";}
   else if (!strcmp("updt", argv[1])) {Xeq = 'u'; Opts = "v";}
   else Usage("Invalid function - ", argv[1]);

   Func = strdup(argv[1]);
   return Configure(argc-1, argv+1, Opts);
}

/******************************************************************************/

int XrdCnsSsiCfg::Configure(int argc, char **argv, const char *Opts)
{
/*
  Function: Establish configuration at start up time via arglist.

  Input:    None.

  Output:   1 upon success or 0 otherwise.
*/

   XrdOucArgs Spec(&MLog, "cns_ssi: ", Opts);
   struct stat Stat;
   char buff[2048], *dP;
   char theOpt, *theArg;
   int n, NoGo = 0;

// Setup the arguments
//
   Spec.Set(argc-1, argv+1);

// Parse the options
//
   while((theOpt = Spec.getopt()) != -1) 
     {switch(theOpt)
       {
       case 'h': Lopt |= Lhost;
                 break;
       case 'l': if (Xeq == 'l') Lopt |= Lfull;
                    else logFN = Spec.argval;
                 break;
       case 'm': Lopt |= Lmode;
                 break;
       case 'n': Lopt |= Lname;
                 break;
       case 'p': Lopt |= Lmount;
                 break;
       case 's': Lopt |= Lsize;
                 break;
       case 'S': Lopt |= Lsize | Lfmts;
                 break;
       case 'v': Verbose = 1;
                 break;
       default:  NoGo = 1;
       }
     }

// Get the destination for the name space
//
   if ((theArg = Spec.getarg())) bPath = strdup(theArg);
      else Usage("Inventory directory not specified.");

// Verify that the target directory exists and is a directory
//
        if (stat(bPath, &Stat)) n = errno;
   else if ((Stat.st_mode & S_IFMT) != S_IFDIR) n = ENOTDIR;
   else n = 0;
   if (n) {Say.M("Unable to process '",bPath,"'; ",
           XrdOucUtils::eText(n, buff, sizeof(buff)));
           return 0;
          }

// Strip trailing slashes from the path
//
   dP = bPath + strlen(bPath) - 1;
   while(*dP == '/') {*dP = '\0'; dP--;}

// Make sure we have some host entries
//
   if (!(dirList = XrdCnsLog::Dirs(bPath, n)))
      {Say.M("No server inventories found in '",bPath,"'."); return 0;}

// Issue warning if errors occurred
//
   if (n) Say.M("Errors encountered; ",Func, " may be incomplete!");

// All done here
//
   return !NoGo;
}

/******************************************************************************/
/*                                 U s a g e                                  */
/******************************************************************************/
  
void XrdCnsSsiCfg::Usage(const char *T1, const char *T2)
{
   const char *What=
   "cns_ssi {diff npath | list [lopts] | updt [uopts]} <dirpath>\n\n"
   "lopts: [-h] [-l] [-m] [-n] [-p] [-s] [-S]\n"
   "uopts: [-l <lfile>] [-v]";

   MLog.Say("cns_ssi: ", T1, T2);
   MLog.Say("\nUsage: ", What);
   exit(4);
}
