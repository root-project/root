/******************************************************************************/
/*                                                                            */
/*                    X r d X r o o t d C o n f i g . c c                     */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*                DE-AC03-76-SFO0515 with the Deprtment of Energy             */
/******************************************************************************/
 
#include <unistd.h>
#include <ctype.h>
#include <fcntl.h>
#include <string.h>
#include <stdio.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/un.h>

#include "XrdVersion.hh"

#include "XrdSfs/XrdSfsInterface.hh"
#include "XrdNet/XrdNetDNS.hh"
#include "XrdNet/XrdNetOpts.hh"
#include "XrdNet/XrdNetSocket.hh"
#include "XrdOuc/XrdOuca2x.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOuc/XrdOucProg.hh"
#include "XrdOuc/XrdOucReqID.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdOuc/XrdOucTrace.hh"
#include "XrdOuc/XrdOucUtils.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysLogger.hh"

#include "XrdXrootd/XrdXrootdAdmin.hh"
#include "XrdXrootd/XrdXrootdAio.hh"
#include "XrdXrootd/XrdXrootdCallBack.hh"
#include "XrdXrootd/XrdXrootdFile.hh"
#include "XrdXrootd/XrdXrootdFileLock.hh"
#include "XrdXrootd/XrdXrootdFileLock1.hh"
#include "XrdXrootd/XrdXrootdJob.hh"
#include "XrdXrootd/XrdXrootdMonitor.hh"
#include "XrdXrootd/XrdXrootdPrepare.hh"
#include "XrdXrootd/XrdXrootdProtocol.hh"
#include "XrdXrootd/XrdXrootdStats.hh"
#include "XrdXrootd/XrdXrootdTrace.hh"
#include "XrdXrootd/XrdXrootdXPath.hh"

#include "Xrd/XrdBuffer.hh"

/******************************************************************************/
/*         P r o t o c o l   C o m m a n d   L i n e   O p t i o n s          */
/******************************************************************************/
  
/* This is the XRootd server. The syntax is:

   xrootd [options]

   options: [<xopt>] [-r] [-t] [-y] [path]

Where:
   xopt   are xrd specified options that are screened out.

   -r     This is a redirecting server.

   -t     This server is a redirection target.

   -y     This server is a proxy server.

    path  Export path. Any number of paths may be specified.
          By default, only '/tmp' is exported.

*/
/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/
  
extern          XrdOucTrace       *XrdXrootdTrace;

                XrdXrootdPrepare  *XrdXrootdPrepQ;

                XrdOucReqID       *XrdXrootdReqID;

                const char        *XrdXrootdInstance;

                XrdInet           *XrdXrootdNetwork;

                int                XrdXrootdPort;

/******************************************************************************/
/*                             C o n f i g u r e                              */
/******************************************************************************/
  
int XrdXrootdProtocol::Configure(char *parms, XrdProtocol_Config *pi)
{
/*
  Function: Establish configuration at load time.

  Input:    None.

  Output:   0 upon success or !0 otherwise.
*/
   extern XrdSfsFileSystem *XrdSfsGetDefaultFileSystem
                            (XrdSfsFileSystem *nativeFS,
                             XrdSysLogger     *Logger,
                             const char       *configFn);
   extern XrdSecService    *XrdXrootdloadSecurity(XrdSysError *, char *, char *);
   extern XrdSfsFileSystem *XrdXrootdloadFileSystem(XrdSysError *, char *, 
                                                    const char *);
   extern int optind, opterr;

   XrdXrootdXPath *xp;
   char *adminp, *fsver, *rdf, *bP, *tmp, c, buff[1024];
   int i, n, deper = 0;

// Copy out the special info we want to use at top level
//
   eDest.logger(pi->eDest->logger());
   XrdXrootdTrace = new XrdOucTrace(&eDest);
   SI           = new XrdXrootdStats(pi->Stats);
   Sched        = pi->Sched;
   BPool        = pi->BPool;
   hailWait     = pi->hailWait;
   readWait     = pi->readWait;
   Port         = pi->Port;
   myInst       = pi->myInst;
   Window       = pi->WSize;
   WANPort      = pi->WANPort;
   WANWindow    = pi->WANWSize;

// Record globally accessible values
//
   XrdXrootdInstance = pi->myInst;
   XrdXrootdNetwork  = pi->NetTCP;
   XrdXrootdPort     = pi->Port;

// Set the callback object static areas now!
//
   XrdXrootdCallBack::setVals(&eDest, SI, Sched, Port);

// Prohibit this program from executing as superuser
//
   if (geteuid() == 0)
      {eDest.Emsg("Config", "Security reasons prohibit xrootd running as "
                  "superuser; xrootd is terminating.");
       _exit(8);
      }

// Process any command line options
//
   opterr = 0; optind = 1;
   if (pi->argc > 1 && '-' == *(pi->argv[1]))
      while ((c=getopt(pi->argc,pi->argv,"mrst")) && ((unsigned char)c != 0xff))
     { switch(c)
       {
       case 'r': deper = 1;
       case 'm': XrdOucEnv::Export("XRDREDIRECT", "R");
                 break;
       case 't': deper = 1;
       case 's': XrdOucEnv::Export("XRDRETARGET", "1");
                 break;
       case 'y': XrdOucEnv::Export("XRDREDPROXY", "1");
                 break;
       default:  eDest.Say("Config warning: ignoring invalid option '",pi->argv[optind-1],"'.");
       }
     }

// Check for deprecated options
//
   if (deper) eDest.Say("Config warning: '-r -t' are deprecated; use '-m -s' instead.");

// Pick up exported paths
//
   for ( ; optind < pi->argc; optind++) xexpdo(pi->argv[optind]);

// Pre-initialize some i/o values. Note that we now set maximum readv element
// transfer size to the buffer size (before it was a reasonable 256K).
//
   if (!(as_miniosz = as_segsize/2)) as_miniosz = as_segsize;
   maxTransz = maxBuffsz = BPool->MaxSize();

// Now process and configuration parameters
//
   rdf = (parms && *parms ? parms : pi->ConfigFN);
   if (rdf && Config(rdf)) return 0;
   if (pi->DebugON) XrdXrootdTrace->What = TRACE_ALL;

// Check if we are exporting anything
//
   if (!(xp = XPList.Next()))
      {XPList.Insert("/tmp"); n = 8;
       eDest.Say("Config warning: only '/tmp' will be exported.");
      } else {
       n = 0;
       while(xp) {eDest.Say("Config exporting ", xp->Path(i));
                  n += i+2; xp = xp->Next();
                 }
      }

// Export the exports
//
   bP = tmp = (char *)malloc(n);
   xp = XPList.Next();
   while(xp) {strcpy(bP, xp->Path(i)); bP += i; *bP++ = ' '; xp = xp->Next();}
   *(bP-1) = '\0';
   XrdOucEnv::Export("XRDEXPORTS", tmp); free(tmp);

// Initialiaze for AIO
//
   if (!as_noaio) XrdXrootdAioReq::Init(as_segsize, as_maxperreq, as_maxpersrv);
      else eDest.Say("Config warning: asynchronous I/O has been disabled!");

// Initialize the security system if this is wanted
//
   if (!SecLib) eDest.Say("Config warning: 'xrootd.seclib' not specified;"
                          " strong authentication disabled!");
      else {TRACE(DEBUG, "Loading security library " <<SecLib);
            if (!(CIA = XrdXrootdloadSecurity(&eDest, SecLib, pi->ConfigFN)))
               {eDest.Emsg("Config", "Unable to load security system.");
                return 0;
               }
           }

// Get the filesystem to be used
//
   if (FSLib)
      {TRACE(DEBUG, "Loading filesystem library " <<FSLib);
       osFS = XrdXrootdloadFileSystem(&eDest, FSLib, pi->ConfigFN);
      } else osFS = XrdSfsGetDefaultFileSystem(0, eDest.logger(), pi->ConfigFN);
   if (!osFS)
      {eDest.Emsg("Config", "Unable to load file system.");
       return 0;
      } else SI->setFS(osFS);

// Check if the file system version matches our version
//
   if (chkfsV)
      {fsver = (char *)osFS->getVersion();
       if (strcmp(XrdVERSION, fsver))
          eDest.Emsg("Config", "Warning! xrootd build version " XrdVERSION
                               "differs from file system version ", fsver);
      }

// Create the file lock manager
//
   Locker = (XrdXrootdFileLock *)new XrdXrootdFileLock1();
   XrdXrootdFile::Init(Locker, as_nosf == 0);
   if (as_nosf) eDest.Say("Config warning: sendfile I/O has been disabled!");

// Schedule protocol object cleanup
//
   ProtStack.Set(pi->Sched, XrdXrootdTrace, TRACE_MEM);
   ProtStack.Set((pi->ConnMax/3 ? pi->ConnMax/3 : 30), 60*60);

// Initialize the request ID generation object
//
   XrdXrootdReqID = new XrdOucReqID((int)Port, pi->myName,
                                    XrdNetDNS::IPAddr(pi->myAddr));

// Initialize for prepare processing
//
   XrdXrootdPrepQ = new XrdXrootdPrepare(&eDest, pi->Sched);
   sprintf(buff, "udp://%s:%d/&L=%%d&U=%%s", pi->myName, pi->Port);
   Notify = strdup(buff);

// Check if we are redirecting anything
//
   if ((xp = RPList.Next()))
      {int k;
       char buff[512];
       do {k = xp->Opts();
           sprintf(buff, " to %s:%d", Route[k].Host, Route[k].Port);
           eDest.Say("Config redirecting ", xp->Path(), buff);
           xp = xp->Next();
          } while(xp);
      }

// Set the redirect flag if we are a pure redirector
//
   if ((rdf = getenv("XRDREDIRECT"))
   && (!strcmp(rdf, "R") || !strcmp(rdf, "M"))) isRedir = *rdf;

// Check if monitoring should be enabled
//
   if (!isRedir && !XrdXrootdMonitor::Init(Sched,&eDest)) return 0;

// Add all jobs that we can run to the admin object
//
   if (JobCKS) XrdXrootdAdmin::addJob("chksum", JobCKS);

// Establish the path to be used for admin functions. We will loose this
// storage upon an error but we don't care because we'll just exit.
//
   adminp = XrdOucUtils::genPath(pi->AdmPath, 0, ".xrootd");

// Setup the admin path (used in all roles).
//
   if (!(AdminSock = XrdNetSocket::Create(&eDest, adminp, "admin", pi->AdmMode))
   ||  !XrdXrootdAdmin::Init(&eDest, AdminSock)) return 0;

// Setup pid file
//
   PidFile();

// Return success
//
   free(adminp);
   return 1;
}

/******************************************************************************/
/*                                C o n f i g                                 */
/******************************************************************************/
  
#define TS_Xeq(x,m) (!strcmp(x,var)) GoNo = m(Config)

int XrdXrootdProtocol::Config(const char *ConfigFN)
{
   XrdOucEnv myEnv;
   XrdOucStream Config(&eDest, getenv("XRDINSTANCE"), &myEnv, "=====> ");
   char *var;
   int cfgFD, GoNo, NoGo = 0, ismine;

   // Open and attach the config file
   //
   if ((cfgFD = open(ConfigFN, O_RDONLY, 0)) < 0)
       return eDest.Emsg("Config", errno, "open config file", ConfigFN);
   Config.Attach(cfgFD);

   // Process items
   //
   while((var = Config.GetMyFirstWord()))
        {     if ((ismine = !strncmp("xrootd.", var, 7)) && var[7]) var += 7;
         else if ((ismine = !strcmp("all.export", var)))    var += 4;
         else if ((ismine = !strcmp("all.pidpath",var)))    var += 4;
         else if ((ismine = !strcmp("all.seclib", var)))    var += 4;

         if (ismine)
            {     if TS_Xeq("async",         xasync);
             else if TS_Xeq("chksum",        xcksum);
             else if TS_Xeq("export",        xexp);
             else if TS_Xeq("fslib",         xfsl);
             else if TS_Xeq("log",           xlog);
             else if TS_Xeq("monitor",       xmon);
             else if TS_Xeq("pidpath",       xpidf);
             else if TS_Xeq("prep",          xprep);
             else if TS_Xeq("redirect",      xred);
             else if TS_Xeq("seclib",        xsecl);
             else if TS_Xeq("trace",         xtrace);
             else {eDest.Say("Config warning: ignoring unknown directive '",var,"'.");
                   Config.Echo();
                   continue;
                  }
             if (GoNo) {Config.Echo(); NoGo = 1;}
            }
        }
   return NoGo;
}

/******************************************************************************/
/*                     P r i v a t e   F u n c t i o n s                      */
/******************************************************************************/
/******************************************************************************/
/*                               P i d F i l e                                */
/******************************************************************************/
  
void XrdXrootdProtocol::PidFile()
{
    int rc, xfd;
    char buff[32], pidFN[1200];
    char *ppath=XrdOucUtils::genPath(pidPath,XrdOucUtils::InstName(-1));
    const char *xop = 0;

    if ((rc = XrdOucUtils::makePath(ppath,XrdOucUtils::pathMode)))
       {xop = "create"; errno = rc;}
       else {snprintf(pidFN, sizeof(pidFN), "%s/xrootd.pid", ppath);

            if ((xfd = open(pidFN, O_WRONLY|O_CREAT|O_TRUNC,0644)) < 0)
               xop = "open";
               else {if (write(xfd,buff,snprintf(buff,sizeof(buff),"%d",
                         static_cast<int>(getpid()))) < 0) xop = "write";
                     close(xfd);
                    }
            }

    free(ppath);
    if (xop) eDest.Emsg("Config", errno, xop, pidFN);
}

/******************************************************************************/
/*                                x a s y n c                                 */
/******************************************************************************/

/* Function: xasync

   Purpose:  To parse directive: async [limit <aiopl>] [maxsegs <msegs>]
                                       [maxtot <mtot>] [segsize <segsz>]
                                       [minsize <iosz>] [maxstalls <cnt>]
                                       [force] [syncw] [off] [nosf]

             <aiopl>  maximum number of async ops per link. Default 8.
             <msegs>  maximum number of async ops per request. Default 8.
             <mtot>   maximum number of async ops per server. Default is 20%
                      of maximum connection times aiopl divided by two.
             <segsz>  The aio segment size. This is the maximum size that data
                      will be read or written. The defaults to 64K but is
                      adjusted for each request to minimize latency.
             <iosz>   the minimum number of bytes that must be read or written
                      to allow async processing to occur (default is maxbsz/2
                      typically 1M).
             <cnt>    Maximum number of client stalls before synchronous i/o is
                      used. Async mode is tried after <cnt> requests.
             force    Uses async i/o for all requests, even when not explicitly
                      requested (this is compatible with synchronous clients).
             syncw    Use synchronous i/o for write requests.
             off      Disables async i/o
             nosf     Disables use of sendfile to send data to the client.

   Output: 0 upon success or 1 upon failure.
*/

int XrdXrootdProtocol::xasync(XrdOucStream &Config)
{
    char *val;
    int  i, ppp;
    int  V_force=-1, V_syncw = -1, V_off = -1, V_mstall = -1, V_nosf = -1;
    int  V_limit=-1, V_msegs=-1, V_mtot=-1, V_minsz=-1, V_segsz=-1;
    int  V_minsf=-1;
    long long llp;
    static struct asyncopts {const char *opname; int minv; int *oploc;
                             const char *opmsg;} asopts[] =
       {
        {"force",     -1, &V_force, ""},
        {"off",       -1, &V_off,   ""},
        {"nosf",      -1, &V_nosf,  ""},
        {"syncw",     -1, &V_syncw, ""},
        {"limit",      0, &V_limit, "async limit"},
        {"segsize", 4096, &V_segsz, "async segsize"},
        {"maxsegs",    0, &V_msegs, "async maxsegs"},
        {"maxstalls",  0, &V_mstall,"async maxstalls"},
        {"maxtot",     0, &V_mtot,  "async maxtot"},
        {"minsfsz",    1, &V_minsf, "async minsfsz"},
        {"minsize", 4096, &V_minsz, "async minsize"}};
    int numopts = sizeof(asopts)/sizeof(struct asyncopts);

    if (!(val = Config.GetWord()))
       {eDest.Emsg("Config", "async option not specified"); return 1;}

    while (val)
         {for (i = 0; i < numopts; i++)
              if (!strcmp(val, asopts[i].opname))
                 {if (asopts[i].minv >=  0 && !(val = Config.GetWord()))
                     {eDest.Emsg("Config","async",(char *)asopts[i].opname,
                                 "value not specified");
                      return 1;
                     }
                       if (asopts[i].minv >  0)
                          if (XrdOuca2x::a2sz(eDest,asopts[i].opmsg, val, &llp,
                                         (long long)asopts[i].minv)) return 1;
                             else *asopts[i].oploc = (int)llp;
                  else if (asopts[i].minv == 0)
                          if (XrdOuca2x::a2i(eDest,asopts[i].opmsg,val,&ppp,1))
                                            return 1;
                             else *asopts[i].oploc = ppp;
                  else *asopts[i].oploc = 1;
                  break;
                 }
          if (i >= numopts)
             eDest.Emsg("Config", "Warning, invalid async option", val);
          val = Config.GetWord();
         }

// Make sure max values are consistent
//
   if (V_limit > 0 && V_mtot > 0 && V_limit > V_mtot)
           {eDest.Emsg("Config", "async limit may not be greater than maxtot");
            return 1;
           }

// Calculate the actual segment size
//
   if (V_segsz > 0)
      {i = BPool->Recalc(V_segsz);
       if (!i) {eDest.Emsg("Config", "async segsize is too large"); return 1;}
       if (i != V_segsz)
          {char buff[64];
           sprintf(buff, "%d readjusted to %d", V_segsz, i);
           eDest.Emsg("Config", "async segsize", buff);
           V_segsz = i;
          }
      }

// Establish async options
//
   if (V_limit > 0) as_maxperlnk = V_limit;
   if (V_msegs > 0) as_maxperreq = V_msegs;
   if (V_mtot  > 0) as_maxpersrv = V_mtot;
   if (V_minsz > 0) as_miniosz   = V_minsz;
   if (V_segsz > 0) as_segsize   = V_segsz;
   if (V_mstall> 0) as_maxstalls = V_mstall;
   if (V_force > 0) as_force     = 1;
   if (V_off   > 0) as_noaio     = 1;
   if (V_syncw > 0) as_syncw     = 1;
   if (V_nosf  > 0) as_nosf      = 1;
   if (V_minsf > 0) as_minsfsz   = V_minsf;

   return 0;
}

/******************************************************************************/
/*                                x c k s u m                                 */
/******************************************************************************/

/* Function: xcksum

   Purpose:  To parse the directive: chksum [max <n>] <type> <path>

             max       maximum number of simultaneous jobs
             <type>    algorithm of checksum (e.g., md5)
             <path>    the path of the program performing the checksum

  Output: 0 upon success or !0 upon failure.
*/

int XrdXrootdProtocol::xcksum(XrdOucStream &Config)
{
   static XrdOucProg *theProg = 0;
   char *palg, prog[2048];
   int jmax = 4;

// Get the algorithm name and the program implementing it
//
   while ((palg = Config.GetWord()) && *palg != '/')
         {if (strcmp(palg, "max")) break;
          if (!(palg = Config.GetWord()))
             {eDest.Emsg("Config", "chksum max not specified"); return 1;}
          if (XrdOuca2x::a2i(eDest, "chksum max", palg, &jmax, 1)) return 1;
         }

// Verify we have an algoritm
//
   if (!palg || *palg == '/')
      {eDest.Emsg("Config", "chksum algorithm not specified"); return 1;}
   if (JobCKT) free(JobCKT);
   JobCKT = strdup(palg);

// Verify that we have a program
//
   if (!Config.GetRest(prog, sizeof(prog)))
      {eDest.Emsg("Config", "cksum parameters too long"); return 1;}
   if (*prog == '\0')
      {eDest.Emsg("Config", "chksum program not specified"); return 1;}

// Set up the program and job
//
   if (!theProg) theProg = new XrdOucProg(0);
   if (theProg->Setup(prog, &eDest)) return 1;
   if (JobCKS) delete JobCKS;
   JobCKS = new XrdXrootdJob(Sched, theProg, "chksum", jmax);
   return 0;
}
  
/******************************************************************************/
/*                                  x e x p                                   */
/******************************************************************************/

/* Function: xexp

   Purpose:  To parse the directive: export <path> [lock|nolock]

             <path>    the path to be exported.

  Output: 0 upon success or !0 upon failure.
*/

int XrdXrootdProtocol::xexp(XrdOucStream &Config)
{
    char *val, pbuff[1024];
    int   popt = 0;

// Get the path
//
   val = Config.GetWord();
   if (!val || !val[0])
      {eDest.Emsg("Config", "export path not specified"); return 1;}
   strlcpy(pbuff, val, sizeof(pbuff));

// Get export lock option
//
   if ((val = Config.GetWord()))
      {if (!strcmp("nolock", val)) popt = XROOTDXP_NOLK;
          else if (strcmp("lock", val)) Config.RetToken();
      }

// Add path to configuration
//
   return xexpdo(pbuff, popt);
}

/******************************************************************************/

int XrdXrootdProtocol::xexpdo(char *path, int popt)
{
   const char *opaque;

// Make sure path start with a slash
//
   if (rpCheck(path, &opaque))
      {eDest.Emsg("Config", "non-absolute export path -", path); return 1;}

// Record the path
//
   if (!Squash(path)) XPList.Insert(path, popt);
   return 0;
}
  
/******************************************************************************/
/*                                  x f s l                                   */
/******************************************************************************/

/* Function: xfsl

   Purpose:  To parse the directive: fslib [?] <path>

             ?         check if fslib build version matches our version
             <path>    the path of the filesystem library to be used.

  Output: 0 upon success or !0 upon failure.
*/

int XrdXrootdProtocol::xfsl(XrdOucStream &Config)
{
    char *val, *Slash;

// Get the path
//
   chkfsV = 0;
   if ((val = Config.GetWord()) && *val == '?' && !val[1])
      {chkfsV = '?'; val = Config.GetWord();}

   if (!val || !val[0])
      {eDest.Emsg("Config", "fslib not specified"); return 1;}

// If this is the "standard" name tell the user that we are ignoring this lib.
// Otherwise, record the path and return.
//
   if (!(Slash = rindex(val, '/'))) Slash = val;
   if (!strcmp(Slash, "/libXrdOfs.so"))
      eDest.Say("Config warning: ignoring fslib; libXrdOfs.so is built-in.");
      else {if (FSLib) free(FSLib);
            FSLib = strdup(val);
           }
   return 0;
}

/******************************************************************************/
/*                                  x l o g                                   */
/******************************************************************************/

/* Function: xlog

   Purpose:  To parse the directive: log <events>

             <events> the blank separated list of events to log.

   Output: 0 upon success or 1 upon failure.
*/

int XrdXrootdProtocol::xlog(XrdOucStream &Config)
{
    char *val;
    static struct logopts {const char *opname; int opval;} lgopts[] =
       {
        {"all",     -1},
        {"disc",    SYS_LOG_02},
        {"login",   SYS_LOG_01}
       };
    int i, neg, lgval = -1, numopts = sizeof(lgopts)/sizeof(struct logopts);

    if (!(val = Config.GetWord()))
       {eDest.Emsg("config", "log option not specified"); return 1;}
    while (val)
          {if ((neg = (val[0] == '-' && val[1]))) val++;
           for (i = 0; i < numopts; i++)
               {if (!strcmp(val, lgopts[i].opname))
                   {if (neg) lgval &= ~lgopts[i].opval;
                       else  lgval |=  lgopts[i].opval;
                    break;
                   }
               }
           if (i >= numopts) eDest.Emsg("config","invalid log option",val);
           val = Config.GetWord();
          }
    eDest.setMsgMask(lgval);
    return 0;
}

/******************************************************************************/
/*                                  x m o n                                   */
/******************************************************************************/

/* Function: xmon

   Purpose:  Parse directive: monitor [all] [mbuff <sz>] 
                                      [flush <sec>] [window <sec>]
                                      dest [Events] <host:port>

   Events: [files] [info] [io] [stage] [user] <host:port>

         all                enables monitoring for all connections.
         mbuff  <sz>        size of message buffer.
         flush  <sec>       time (seconds, M, H) between auto flushes.
         window <sec>       time (seconds, M, H) between timing marks.
         dest               specified routing information. Up to two dests
                            may be specified.
         files              only monitors file open/close events.
         info               monitors client appid and info requests.
         io                 monitors I/O requests, and files open/close events.
         stage              monitors file stage operations
         user               monitors user login and disconnect events.
         <host:port>        where monitor records are to be sentvia UDP.

   Output: 0 upon success or !0 upon failure. Ignored by master.
*/
int XrdXrootdProtocol::xmon(XrdOucStream &Config)
{   char  *val, *cp, *monDest[2] = {0, 0};
    long long tempval;
    int i, monFlush=0, monMBval=0, monWWval=0, xmode=0, monMode[2] = {0, 0};

    while((val = Config.GetWord()))

         {     if (!strcmp("all",  val)) xmode = XROOTD_MON_ALL;
          else if (!strcmp("flush", val))
                {if (!(val = Config.GetWord()))
                    {eDest.Emsg("Config", "monitor flush value not specified");
                     return 1;
                    }
                 if (XrdOuca2x::a2tm(eDest,"monitor flush",val,
                                           &monFlush,1)) return 1;
                }
          else if (!strcmp("mbuff",val))
                  {if (!(val = Config.GetWord()))
                      {eDest.Emsg("Config", "monitor mbuff value not specified");
                       return 1;
                      }
                   if (XrdOuca2x::a2sz(eDest,"monitor mbuff", val,
                                             &tempval, 1024, 65536)) return 1;
                    monMBval = static_cast<int>(tempval);
                  }
          else if (!strcmp("window", val))
                {if (!(val = Config.GetWord()))
                    {eDest.Emsg("Config", "monitor window value not specified");
                     return 1;
                    }
                 if (XrdOuca2x::a2tm(eDest,"monitor window",val,
                                           &monWWval,1)) return 1;
                }
          else break;
         }

    if (!val) {eDest.Emsg("Config", "monitor dest not specified"); return 1;}

    for (i = 0; i < 2; i++)
        {if (strcmp("dest", val)) break;
         while((val = Config.GetWord()))
                   if (!strcmp("files",val)) monMode[i] |=  XROOTD_MON_FILE;
              else if (!strcmp("info", val)) monMode[i] |=  XROOTD_MON_INFO;
              else if (!strcmp("io",   val)) monMode[i] |=  XROOTD_MON_IO;
              else if (!strcmp("stage",val)) monMode[i] |=  XROOTD_MON_STAGE;
              else if (!strcmp("user", val)) monMode[i] |=  XROOTD_MON_USER;
              else break;
          if (!val) {eDest.Emsg("Config","monitor dest value not specified");
                     return 1;
                    }
          if (!(cp = index(val, (int)':')) || !atoi(cp+1))
             {eDest.Emsg("Config","monitor dest port missing or invalid in",val);
              return 1;
             }
          monDest[i] = strdup(val);
         if (!(val = Config.GetWord())) break;
        }

    if (val)
       {if (!strcmp("dest", val))
           eDest.Emsg("Config", "Warning, a maximum of two dest values allowed.");
           else eDest.Emsg("Config", "Warning, invalid monitor option", val);
       }

// Make sure dests differ
//
   if (monDest[0] && monDest[1] && !strcmp(monDest[0], monDest[1]))
      {eDest.Emsg("Config", "Warning, monitor dests are identical.");
       monMode[0] |= monMode[1]; monMode[1] = 0;
       free(monDest[1]); monDest[1] = 0;
      }

// Screen out requests that only want to stage
//
   if (monDest[0] && monMode[0] == XROOTD_MON_STAGE)
      {monMode[0] = 0; free(monDest[0]); monDest[0] = 0;}
   if (monDest[1] && monMode[1] == XROOTD_MON_STAGE)
      {monMode[1] = 0; free(monDest[1]); monDest[1] = 0;}

// Add files option if I/O is enabled
//
   if (monMode[0] & XROOTD_MON_IO) monMode[0] |= XROOTD_MON_FILE;
   if (monMode[1] & XROOTD_MON_IO) monMode[1] |= XROOTD_MON_FILE;

// Set the monitor defaults
//
   XrdXrootdMonitor::Defaults(monMBval, monWWval, monFlush);
   if (monDest[0]) monMode[0] |= (monMode[0] ? xmode : XROOTD_MON_FILE|xmode);
   if (monDest[1]) monMode[1] |= (monMode[1] ? xmode : XROOTD_MON_FILE|xmode);
   XrdXrootdMonitor::Defaults(monDest[0],monMode[0],monDest[1],monMode[1]);
   return 0;
}
  
/******************************************************************************/
/*                                 x p i d f                                  */
/******************************************************************************/

/* Function: xpidf

   Purpose:  To parse the directive: pidpath <path>

             <path>    the path where the pid file is to be created.

  Output: 0 upon success or !0 upon failure.
*/

int XrdXrootdProtocol::xpidf(XrdOucStream &Config)
{
    char *val;

// Get the path
//
   val = Config.GetWord();
   if (!val || !val[0])
      {eDest.Emsg("Config", "pidpath not specified"); return 1;}

// Record the path
//
   if (pidPath) free(pidPath);
   pidPath = strdup(val);
   return 0;
}

/******************************************************************************/
/*                                 x p r e p                                  */
/******************************************************************************/

/* Function: xprep

   Purpose:  To parse the directive: prep [keep <sec>] [scrub <sec>]
                                          [logdir <path>]
         keep   <sec>  time (seconds, M, H) to keep logdir entries.
         scrub  <sec>  time (seconds, M, H) between logdir scrubs.
         logdir <path> the absolute path to the prepare log directory.

   Output: 0 upon success or !0 upon failure. Ignored by master.
*/
int XrdXrootdProtocol::xprep(XrdOucStream &Config)
{   int   rc, keep = 0, scrub=0;
    char  *ldir=0,*val,buff[1024];

    if (!(val = Config.GetWord()))
       {eDest.Emsg("Config", "prep options not specified"); return 1;}

        do { if (!strcmp("keep", val))
                {if (!(val = Config.GetWord()))
                    {eDest.Emsg("Config", "prep keep value not specified");
                     return 1;
                    }
                 if (XrdOuca2x::a2tm(eDest,"prep keep int",val,&keep,1)) return 1;
                }
        else if (!strcmp("scrub", val))
                {if (!(val = Config.GetWord()))
                    {eDest.Emsg("Config", "prep scrub value not specified");
                     return 1;
                    }
                 if (XrdOuca2x::a2tm(eDest,"prep scrub",val,&scrub,0)) return 1;
                }
        else if (!strcmp("logdir", val))
                {if (!(ldir = Config.GetWord()))
                   {eDest.Emsg("Config", "prep logdir value not specified");
                    return 1;
                   }
                }
        else eDest.Emsg("Config", "Warning, invalid prep option", val);
       } while((val = Config.GetWord()));

// Set the values
//
   if (scrub || keep) XrdXrootdPrepare::setParms(scrub, keep);
   if (ldir) 
       if ((rc = XrdOucUtils::genPath(buff, sizeof(buff), ldir, myInst)) < 0
       ||  (rc = XrdOucUtils::makePath(buff, XrdOucUtils::pathMode)) < 0
       ||  (rc = XrdXrootdPrepare::setParms(buff)) < 0)
          {eDest.Emsg("Config", rc, "process logdir", ldir);
           return 1;
          }
   return 0;
}

/******************************************************************************/
/*                                  x r e d                                   */
/******************************************************************************/
  
/* Function: xred

   Purpose:  To parse the directive: redirect <host>:<port> {<funcs>|<path>}

             <funcs>   are one or more of the following functions that will
                       be immediately redirected to <host>:<port>. Each function
                       may be prefixed by a minus sign to disable redirection.

                       chmod dirlist locate mkdir mv prepare rm rmdir stat

             <paths>   redirects the client when an attempt is made to open
                       one of absolute <paths>. Up to 4 different redirect
                       combinations may be specified.

  Output: 0 upon success or !0 upon failure.
*/

int XrdXrootdProtocol::xred(XrdOucStream &Config)
{
    static struct rediropts {const char *opname; RD_func opval;} rdopts[] =
       {
        {"chmod",    RD_chmod},
        {"dirlist",  RD_dirlist},
        {"locate",   RD_locate},
        {"mkdir",    RD_mkdir},
        {"mv",       RD_mv},
        {"prepare",  RD_prepare},
        {"prepstage",RD_prepstg},
        {"rm",       RD_rm},
        {"rmdir",    RD_rmdir},
        {"stat",     RD_stat}
       };
    char rHost[512], *val, *pp;
    int i, k, neg, rPort, numopts = sizeof(rdopts)/sizeof(struct rediropts);

// Get the host and port
//
   val = Config.GetWord();
   if (!val || !val[0] || val[0] == ':')
      {eDest.Emsg("Config", "redirect host not specified"); return 1;}
   if (!(pp = index(val, ':')))
      {eDest.Emsg("Config", "redirect port not specified"); return 1;}
   if (!(rPort = atoi(pp+1)))
      {eDest.Emsg("Config", "redirect port is invalid");    return 1;}
   *pp = '\0';
   strlcpy(rHost, val, sizeof(rHost));

// Set all redirect target functions
//
    if (!(val = Config.GetWord()))
       {eDest.Emsg("config", "redirect option not specified"); return 1;}

    if (*val == '/')
       {for (k = static_cast<int>(RD_open1); k < RD_Num; k++)
            if (!Route[k].Host
            || (strcmp(Route[k].Host, rHost) && Route[k].Port == rPort)) break;
        if (k >= RD_Num)
           {eDest.Emsg("Config", "too many diffrent path redirects"); return 1;}
        xred_set(RD_func(k), rHost, rPort);
        do {RPList.Insert(val, k, 0);
            if ((val = Config.GetWord()) && *val != '/')
               {eDest.Emsg("Config", "non-absolute redirect path -", val);
                return 1;
               }
           } while(val);
        return 0;
       }

    while (val)
          {if (!strcmp(val, "all"))
              {for (i = 0; i < numopts; i++)
                   xred_set(rdopts[i].opval, rHost, rPort);
              }
              else {if ((neg = (val[0] == '-' && val[1]))) val++;
                    for (i = 0; i < numopts; i++)
                       {if (!strcmp(val, rdopts[i].opname))
                           {if (neg) xred_set(rdopts[i].opval, 0, 0);
                               else  xred_set(rdopts[i].opval, rHost, rPort);
                            break;
                           }
                       }
                   if (i >= numopts)
                      eDest.Emsg("config", "invalid redirect option", val);
                  }
          val = Config.GetWord();
         }
   return 0;
}

void XrdXrootdProtocol::xred_set(RD_func func, const char *rHost, int rPort)
{
// Reset static redirection
//
   if (Route[func].Host) free(Route[func].Host);
   Route[func].Host = (rHost ? strdup(rHost) : 0);
   Route[func].Port = rPort;
}

/******************************************************************************/
/*                                 x s e c l                                  */
/******************************************************************************/

/* Function: xsecl

   Purpose:  To parse the directive: seclib <path>

             <path>    the path of the security library to be used.

  Output: 0 upon success or !0 upon failure.
*/

int XrdXrootdProtocol::xsecl(XrdOucStream &Config)
{
    char *val;

// Get the path
//
   val = Config.GetWord();
   if (!val || !val[0])
      {eDest.Emsg("Config", "XRootd seclib not specified"); return 1;}

// Record the path
//
   if (SecLib) free(SecLib);
   SecLib = strdup(val);
   return 0;
}
  
/******************************************************************************/
/*                                x t r a c e                                 */
/******************************************************************************/

/* Function: xtrace

   Purpose:  To parse the directive: trace <events>

             <events> the blank separated list of events to trace. Trace
                      directives are cummalative.

   Output: 0 upon success or 1 upon failure.
*/

int XrdXrootdProtocol::xtrace(XrdOucStream &Config)
{
    char *val;
    static struct traceopts {const char *opname; int opval;} tropts[] =
       {
        {"all",      TRACE_ALL},
        {"emsg",     TRACE_EMSG},
        {"debug",    TRACE_DEBUG},
        {"fs",       TRACE_FS},
        {"login",    TRACE_LOGIN},
        {"mem",      TRACE_MEM},
        {"stall",    TRACE_STALL},
        {"redirect", TRACE_REDIR},
        {"request",  TRACE_REQ},
        {"response", TRACE_RSP}
       };
    int i, neg, trval = 0, numopts = sizeof(tropts)/sizeof(struct traceopts);

    if (!(val = Config.GetWord()))
       {eDest.Emsg("config", "trace option not specified"); return 1;}
    while (val)
         {if (!strcmp(val, "off")) trval = 0;
             else {if ((neg = (val[0] == '-' && val[1]))) val++;
                   for (i = 0; i < numopts; i++)
                       {if (!strcmp(val, tropts[i].opname))
                           {if (neg) trval &= ~tropts[i].opval;
                               else  trval |=  tropts[i].opval;
                            break;
                           }
                       }
                   if (i >= numopts)
                      eDest.Emsg("config", "invalid trace option", val);
                  }
          val = Config.GetWord();
         }
    XrdXrootdTrace->What = trval;
    return 0;
}
