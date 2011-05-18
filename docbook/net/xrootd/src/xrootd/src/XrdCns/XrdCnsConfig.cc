/******************************************************************************/
/*                                                                            */
/*                       X r d C n s C o n f i g . c c                        */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdCnsConfigCVSID = "$Id$";

#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <sys/types.h>

#include "Xrd/XrdTrace.hh"

#include "XrdClient/XrdClientConst.hh"
#include "XrdClient/XrdClientEnv.hh"

#include "XrdNet/XrdNetDNS.hh"
#include "XrdNet/XrdNetOpts.hh"
#include "XrdNet/XrdNetSocket.hh"

#include "XrdOuc/XrdOuca2x.hh"
#include "XrdOuc/XrdOucArgs.hh"
#include "XrdOuc/XrdOucName2Name.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdOuc/XrdOucTList.hh"
#include "XrdOuc/XrdOucTokenizer.hh"
#include "XrdOuc/XrdOucUtils.hh"

#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdSys/XrdSysPlugin.hh"
#include "XrdSys/XrdSysTimer.hh"

#include "XrdCns/XrdCnsConfig.hh"
#include "XrdCns/XrdCnsDaemon.hh"
#include "XrdCns/XrdCnsLogRec.hh"
#include "XrdCns/XrdCnsLogServer.hh"
#include "XrdCns/XrdCnsXref.hh"

/******************************************************************************/
/*           G l o b a l   C o n f i g u r a t i o n   O b j e c t            */
/******************************************************************************/

namespace XrdCns
{
       XrdCnsConfig      Config;

extern XrdCnsDaemon      XrdCnsd;

extern XrdSysError       MLog;

extern XrdOucTrace       XrdTrace;
}

using namespace XrdCns;

/******************************************************************************/
/*            E x t e r n a l   T h r e a d   I n t e r f a c e s             */
/******************************************************************************/
  
namespace XrdCns
{
void *CnsEvents(void *parg)
{
   XrdOucStream fifoEvents;    // FIFO fed events
   int eFD = *static_cast<int *>(parg);
   fifoEvents.Attach(eFD, 32*1024);
   XrdCnsd.getEvents(fifoEvents, "fifo");
   return (void *)0;
}

void *CnsInt(void *parg)
{
   XrdCnsLogRec *lrP;

// Just blab out the midnight herald
//
   while(1)
        {XrdSysTimer::Snooze(Config.cInt);
         lrP = XrdCnsLogRec::Alloc();
         lrP->setType('\0');
         lrP->Queue();
        }
   return (void *)0;
}
}
  
/******************************************************************************/
/*                             C o n f i g u r e                              */
/******************************************************************************/

int XrdCnsConfig::Configure(int argc, char **argv, char *argt)
{
/*
  Function: Establish configuration at start up time via arglist.

  Input:    None.

  Output:   1 upon success or 0 otherwise.
*/

   const char *TraceID = "Config";
   XrdOucArgs Spec(&MLog,(argt ? "Cns_Config: ":"XrdCnsd: "),
                          "a:b:B:c:dD:e:E:i:I:l:L:N:p:q:R:");
   char buff[2048], *dP, *tP, *dnsEtxt = 0, *n2n = 0, *lroot = 0, *xpl = 0;
   char theOpt, *theArg;
   long long llval;
   int n, bPort = 0, haveArk = 0, NoGo = 0;

// Setup the logger
//
   if (argt) Spec.Set(argt);
      else   Spec.Set(argc-1, argv+1);

// Parse the options
//
   while((theOpt = Spec.getopt()) != -1) 
     {switch(theOpt)
       {
       case 'a': if (*aPath == '/') aPath = Spec.argval;
                    else NoGo = NAPath("'-a'", Spec.argval);
                 break;
       case 'B': Opts |= optNoCns;
       case 'b': bPath = Spec.argval;
                 break;
       case 'c': cPath = Spec.argval;
                 break;
       case 'D': NoGo |= XrdOuca2x::a2i(MLog,"-D value",Spec.argval,&n,0,4);
                 if (!NoGo) EnvPutInt("DebugLevel", n);
                 break;
       case 'd': XrdTrace.What = TRACE_ALL;
                 XrdSysThread::setDebug(&MLog);
                 break;
       case 'e': if (*ePath == '/') ePath = Spec.argval;
                    else NoGo = NAPath("'-e'", Spec.argval);
                 break;
       case 'k': n = strlen(Spec.argval)-1;
                 NoGo |= (isalpha(Spec.argval[n])
                      ? XrdOuca2x::a2sz(MLog,"keep size", Spec.argval,&llval)
                      : XrdOuca2x::a2ll(MLog,"keep count",Spec.argval,&llval));
                 if (!isalpha(Spec.argval[n])) llval = -llval;
                 logKeep = static_cast<int>(llval);
                 break;
       case 'i': NoGo |= XrdOuca2x::a2tm(MLog,"-i value",Spec.argval,&cInt,1);
                 break;
       case 'I': NoGo |= XrdOuca2x::a2tm(MLog,"-I value",Spec.argval,&mInt,1);
                 break;
       case 'l': logfn = Spec.argval;
                 break;
       case 'L': lroot = Spec.argval;
                 break;
       case 'N': n2n   = Spec.argval;
                 break;
       case 'p': NoGo |= XrdOuca2x::a2i(MLog,"-p value",Spec.argval,&Port,1,65535);
                 bPort = Port;
                 break;
       case 'q': NoGo |= XrdOuca2x::a2i(MLog,"-q value",Spec.argval,&qLim,1,1024);
                 break;
       case 'R': Opts |= optRecr;
                 xpl   = Spec.argval;
                 break;
       default:  NoGo = 1;
       }
     }

// The recreate option is only valid if we are not running under an xrootd
//
   if (Opts & optRecr)
      {if (getenv("XRDINSTANCE") || getenv("XRDPROG"))
          {MLog.Emsg("Config","'-R' is valid only for a stand-alone command.");
           return 0;
          }
       if (bPath) {free(bPath); bPath = 0;}
       if (lroot)
          {sprintf(buff, "XRDLCLROOT=%s", lroot); putenv(strdup(buff));}
       if (n2n)
          {if ((tP=index(n2n, ' '))) {*tP++ = '\0'; while(*tP == ' ') tP++;}
           sprintf(buff, "XRDN2NLIB=%s", n2n); putenv(strdup(buff));
           if (tP && *tP)
              {sprintf(buff, "XRDN2NPARMS=%s", tP); putenv(strdup(buff));}
          }
       if (xpl && *xpl)
          {char *Colon = xpl;
           while((Colon = index(Colon, ':'))) *Colon++ = ' ';
           sprintf(buff, "XRDEXPORTS=%s", xpl); putenv(strdup(buff));
          } else {MLog.Emsg("Config","'-R' requires exports to be specified.");
                  return 0;
                 }
       Space = new XrdCnsXref("public",0);
      } else {
       *buff = '\0'; tP = buff;
       if (lroot) {*tP++ = ' '; *tP++ = '-'; *tP++ = 'L';}
       if (n2n)   {*tP++ = ' '; *tP++ = '-'; *tP++ = 'N';}
       if (*buff)
          MLog.Emsg("Config", buff+1, "options ignored; valid only with -R.");
      }

// Handle config
//
   if (!cPath) cPath = getenv("XRDCONFIGFN");
   cPath = (cPath ? strdup(cPath) : (char *)"");

// Handle the backup directory now. If there is one then we will create a
// thread that periodically closes and backs up the log files.
//
   if (bPath)
      {char *bHost = 0;
       if (!bPort) bPort = Port;
            if (*bPath == '/') strcpy(buff, bPath);
       else if (!(dP = index(bPath, '/')) || *(dP-1) != ':') *buff = 0;
       else {char hBuff[1024], *cP = dP-1;
             strncpy(hBuff+1, bPath, cP-bPath); hBuff[cP-bPath+1] = '\0';
             if ((cP = index(hBuff+1, ':'))
             &&  XrdOuca2x::a2i(MLog,"-b port",cP+1,&bPort,1,65535)) *buff = 0;
             if (cP) *cP = '\0';
             bHost = XrdNetDNS::getHostName(hBuff+1, &dnsEtxt);
             if (dnsEtxt)
                {*hBuff = '\''; strcat(hBuff+1, "\'"); *buff = 0;
                 MLog.Emsg("Config", hBuff, dnsEtxt);
                } else strcpy(buff, dP);
            }
       if (!*buff)
          {MLog.Emsg("Config","Backup path cannot be determined."); NoGo=1;}
          else {if (buff[strlen(buff)-1] == '/') strcat(buff, "cns/");
                   else strcat(buff, "/cns/");
                bPath = strdup(buff);
                if (bHost)
                   {sprintf(buff, "%s:%d", bHost, bPort); free(bHost);
                    bDest = new XrdOucTList(buff, -bPort);
                    TRACE(DEBUG, "Bkp host =" <<bDest->text);
                   }
                TRACE(DEBUG, "Bkp path =" <<bPath);
               }
      }

// Get the destination for the name space and log files. In the process if we
// create a client who will not be archiving but archive-only mode is in
// effect; then delete that newly created client. Yes, Amelia, this is an odd
// way of doing this but is much less complicated given the logic choices.
//
   while((theArg = Spec.getarg()))
        {strcpy(buff, theArg);
         if (!strncmp("xroot://", buff, 8)) dP = buff+8;
            else if (!strncmp( "root://", buff, 7)) dP = buff+7;
                    else dP = buff;
         if ( (tP = index(dP, '/'))) *tP = '\0';
         if (!(tP = index(dP, ':')))   n = Port;
            else if ((n = atoi(tP+1)) <= 0)
                    {MLog.Emsg("Config", "Invalid port number in", dP);
                     NoGo = 1; continue;
                    } else *tP = '\0';
         dnsEtxt = 0;
         tP = XrdNetDNS::getHostName(dP, &dnsEtxt);
         if (dnsEtxt)
            {buff[0] = '\''; buff[1] = ' '; strcpy(buff+2, dnsEtxt);
             MLog.Emsg("Config", "'", dP, buff);
             NoGo = 1; delete tP; continue;
            }
         sprintf(buff, "%s:%d", tP, n); delete tP;
              if (!bDest)  Dest = new XrdOucTList(buff, (bPath ? -n : n), Dest);
         else if (haveArk) Dest = new XrdOucTList(buff, n, Dest);
         else if (strcmp(buff, bDest->text))
                           Dest = new XrdOucTList(buff, n, Dest);
         else {bDest->next = Dest; Dest = bDest; haveArk = 1;}

         if (Opts & optNoCns && Dest->val >= 0)
            {XrdOucTList *xP = Dest; Dest = xP->next; delete xP;}
        }

// Chain in backup host if we have not done so
//
   if (bDest && !haveArk) {bDest->next = Dest; Dest = bDest;}

// All done here
//
   return !NoGo;
}

/******************************************************************************/

int XrdCnsConfig::Configure()
{
/*
  Function: Establish configuration at start up time.

  Input:    None.

  Output:   1 upon success or 0 otherwise.
*/
   const char *TraceID = "Config";
   static int eFD;
   XrdOucTokenizer mToks(0);
   XrdNetSocket   *EventSock;
   pthread_t tid;
   int n, retc, NoGo = 0;
   const char *iP;
   char buff[2048], *dP, *tP, *eVar;

// Put out the herald
//
   if (!(Opts & optRecr)) MLog.Emsg("Config", "Cns initialization started.");

// Set current working directory for core files
//
   if ((iP = XrdOucUtils::InstName(-1))) {strcpy(buff,"./"); strcat(buff, iP);}
      else strcpy(buff, ".");
   strcat(buff, "/cns/");
   if (!XrdOucUtils::makePath(buff,0770) && chdir(buff)) {}

// Do some XrdClient specific optimizations
//
   EnvPutInt(NAME_DATASERVERCONN_TTL, 2147483647); // Prevent timeouts

// Get the directory where the meta information is to go
//
   if (!aPath && !(aPath = getenv("XRDADMINPATH"))) aPath = (char *)"/tmp/";
   strcpy(buff, aPath);
   if (buff[strlen(buff)-1] == '/') strcat(buff, "cns/");
      else strcat(buff, "/cns/");
   aPath = strdup(buff);
   TRACE(DEBUG, "Admin path=" <<aPath);

// Create the admin path if it is not there
//
   if ((retc = XrdOucUtils::makePath(aPath,0770)))
      {MLog.Emsg("Config", retc, "create admin directory", aPath);
       NoGo = 1;
      }

// Establish the event directory path
//
   if (!ePath) ePath = aPath;
      else {strcpy(buff, ePath);
            if (buff[strlen(buff)-1] != '/') strcat(buff, "/");
            ePath = strdup(buff);
           }
   TRACE(DEBUG, "Event path=" <<ePath);

// Create the event path if it is not there (forget it for 1-time recreates)
//
   if (!(Opts & optRecr))
      if (aPath != ePath && (retc = XrdOucUtils::makePath(ePath,0770)))
         {MLog.Emsg("Config", retc, "create event directory", ePath);
          NoGo = 1;
         }

// Handle the name2name library
//
   NoGo |= ConfigN2N();

// Handle the exports list
//
   if ((eVar = getenv("XRDEXPORTS")) && *eVar)
      {eVar = strdup(eVar); mToks.Attach(eVar); mToks.GetLine();
       n = 9999;
       while((dP = mToks.GetToken()))
            {if (!LocalPath(dP, buff, sizeof(buff))) NoGo = 1;
                else {Exports =  new XrdOucTList(buff, strlen(buff), Exports);
                      TRACE(DEBUG, "Exported physical path=" <<buff);
                     }
            }
       free(eVar);
      }

// Check if we have any exported paths
//
   if (!Exports)
      {MLog.Emsg("Config", "No paths have been exported!");
       NoGo = 1;
      }

// Get the destination for the name space
//
   if (!Dest)
      {if ((eVar = getenv("XRDCMSMAN")) && *eVar)
          {eVar = strdup(eVar); mToks.Attach(eVar); mToks.GetLine();
           while((dP = mToks.GetToken()))
                {if ((tP = index(dP, ':'))) *tP = '\0';
                 sprintf(buff, "%s:%d", tP, Port);
                 if (*tP) *tP = ':';
                 Dest = new XrdOucTList(dP, Port, Dest);
                 TRACE(DEBUG, "CNS dest=" <<dP <<':' <<Port);
                }
           free(eVar);
          }
       if (!Dest) 
          {MLog.Emsg("Config","Name space routing not specified."); NoGo=1;}
      }

// If we have an archiver, create a thread that periodically closes and 
// backs up the log files.
//
   if (bPath)
      {if ((retc = XrdSysThread::Run(&tid,CnsInt,0,XRDSYSTHREAD_BIND,
                                                  "Interval logging")))
          {MLog.Emsg("Config",retc,"create interval logging thread"); NoGo=1;}
      } else {
       if (!(Opts & optRecr))
          MLog.Emsg("Config","Backup path not specified; inventory disabled!");
      }

// Check if we should continue
//
   if (NoGo)
      {MLog.Emsg("Config", "Cns initialization failed.");
       return 0;
      }

// Initialize event handling and return for 1-time recreates
//
   XrdCnsLog = new XrdCnsLogServer();
   NoGo = !XrdCnsLog->Init(Dest);
   if (Opts & optRecr) exit(NoGo ? 4 : 0);

// Create our notification path (r/w for us and our group) and start it
//
   if ((EventSock = XrdNetSocket::Create(&MLog, aPath, "XrdCnsd.events",
                                  0660, XRDNET_FIFO)))
      {eFD = EventSock->Detach();
       delete EventSock;
       if ((retc = XrdSysThread::Run(&tid, CnsEvents, (void *)&eFD,
                                 XRDSYSTHREAD_BIND, "FIFO event handler")))
          {MLog.Emsg("Config", retc, "create FIFO event thread"); NoGo = 1;}
      } else NoGo = 1;

// All done
//
   MLog.Emsg("Config", "Cns initialization",(NoGo ? "failed.":"completed."));
   return !NoGo;
}

/******************************************************************************/
/* Private:                    C o n f i g N 2 N                              */
/******************************************************************************/

int XrdCnsConfig::ConfigN2N()
{
   XrdSysPlugin    *myLib;
   XrdOucName2Name *(*ep)(XrdOucgetName2NameArgs);
   char *N2NLib, *N2NParms = 0;

// Get local root
//
   if ((LCLRoot = getenv("XRDLCLROOT")) && !*LCLRoot) LCLRoot = 0;

// If we have no library path then use the default method (this will always
// succeed).
//
   if (!(N2NLib = getenv("XRDN2NLIB")) || !*N2NLib)
      {if (LCLRoot) N2N = XrdOucgetName2Name(&MLog, 0, "", LCLRoot, 0);
       return 0;
      }

// Get the N2N parameters
//
   if ((N2NParms = getenv("XRDN2NPARMS"))) N2NParms = strdup(N2NParms);

// Create a pluin object (we will throw this away without deletion because
// the library must stay open but we never want to reference it again).
//
   if (!(myLib = new XrdSysPlugin(&MLog, N2NLib))) return 1;

// Now get the entry point of the object creator
//
   ep = (XrdOucName2Name *(*)(XrdOucgetName2NameArgs))(myLib->getPlugin("XrdOucgetName2Name"));
   if (!ep) return 1;

// Get the Object now
//
   N2N = ep(&MLog, cPath,(N2NParms ? N2NParms:""),LCLRoot,0);
   if (N2NParms) free(N2NParms);
   return N2N == 0;
}

/******************************************************************************/
/* Public:                     L o c a l P a t h                              */
/******************************************************************************/
  
int XrdCnsConfig::LocalPath(const char *oldp, char *newp, int newpsz)
{
    int rc = 0;

    if (N2N) rc = N2N->lfn2pfn(oldp, newp, newpsz);
       else if (((int)strlen(oldp)) >= newpsz) rc = ENAMETOOLONG;
               else strcpy(newp, oldp);
    if (rc) {MLog.Emsg("Config", rc, "generate local path from", oldp);
             return 0;
            }
    return 1;
}

/******************************************************************************/
/* Public:                     L o g i c P a t h                              */
/******************************************************************************/
  
int XrdCnsConfig::LogicPath(const char *oldp, char *newp, int newpsz)
{
    int rc = 0;

    if (N2N) rc = N2N->pfn2lfn(oldp, newp, newpsz);
       else if (((int)strlen(oldp)) >= newpsz) rc = ENAMETOOLONG;
               else strcpy(newp, oldp);
    if (rc) {MLog.Emsg("Config", rc, "generate logical path from", oldp);
             return 0;
            }
    return 1;
}

/******************************************************************************/
/* Public:                     M o u n t P a t h                              */
/******************************************************************************/
  
int XrdCnsConfig::MountPath(const char *lfnP, char *newp, int newpsz)
{
   XrdOucTList *xP = Exports;
   int n = strlen(lfnP);

// Find the export path for this incomming path
//
   while(xP)
        {if (n >= xP->val && !strncmp(xP->text, lfnP, xP->val)) break;
         xP = xP->next;
        }

// Enter the mount path
//
   if (!xP)
      {strcpy(newp, LCLRoot ? LCLRoot : "/");
       return 0;
      }

// Convert export to a physical path and use that
//
   Config.LocalPath(xP->text, newp, newpsz);
   return 1;
}

/******************************************************************************/
/* Private:                       N A P a t h                                 */
/******************************************************************************/
  
int XrdCnsConfig::NAPath(const char *What, const char *Path)
{
   MLog.Emsg("Config", "Absolute path required in", What, Path);
   return 1;
}
