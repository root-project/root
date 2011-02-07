/*******************************************************************************/
/*                                                                            */
/*                          X r d C o n f i g . c c                           */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*                DE-AC03-76-SFO0515 with the Deprtment of Energy             */
/******************************************************************************/

//         $Id$

const char *XrdConfigCVSID = "$Id$";

/*
   The default port number comes from:
   1) The command line option,
   2) The config file,
   3) The /etc/services file for service corresponding to the program name.
*/
  
#include <unistd.h>
#include <ctype.h>
#include <pwd.h>
#include <string.h>
#include <stdio.h>
#include <sys/param.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "Xrd/XrdBuffer.hh"
#include "Xrd/XrdConfig.hh"
#include "Xrd/XrdInet.hh"
#include "Xrd/XrdLink.hh"
#include "Xrd/XrdPoll.hh"
#include "Xrd/XrdProtLoad.hh"
#include "Xrd/XrdScheduler.hh"
#include "Xrd/XrdStats.hh"
#include "Xrd/XrdTrace.hh"
#include "Xrd/XrdInfo.hh"

#include "XrdNet/XrdNetDNS.hh"
#include "XrdNet/XrdNetSecurity.hh"

#include "XrdOuc/XrdOuca2x.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdOuc/XrdOucUtils.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdSys/XrdSysTimer.hh"

#ifdef __linux__
#include <netinet/tcp.h>
#endif
#ifdef __macos__
#include <AvailabilityMacros.h>
#endif

/******************************************************************************/
/*           G l o b a l   C o n f i g u r a t i o n   O b j e c t            */
/******************************************************************************/

       XrdBuffManager    XrdBuffPool;

       int               XrdNetTCPlep = -1;
       XrdInet          *XrdNetTCP[XrdProtLoad::ProtoMax+1] = {0};
extern XrdInet          *XrdNetADM;

extern XrdScheduler      XrdSched;

extern XrdSysError       XrdLog;

extern XrdSysLogger      XrdLogger;

extern XrdSysThread     *XrdThread;

extern XrdOucTrace       XrdTrace;

       const char       *XrdConfig::TraceID = "Config";

/******************************************************************************/
/*                               d e f i n e s                                */
/******************************************************************************/

#define TS_Xeq(x,m)    if (!strcmp(x,var)) return m(eDest, Config);

#ifndef S_IAMB
#define S_IAMB  0x1FF
#endif

/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/

class XrdConfigProt
{
public:

XrdConfigProt  *Next;
char           *proname;
char           *libpath;
char           *parms;
int             port;
int             wanopt;

                XrdConfigProt(char *pn, char *ln, char *pp, int np=-1, int wo=0)
                    {Next = 0; proname = pn; libpath = ln; parms = pp; 
                     port=np; wanopt = wo;
                    }
               ~XrdConfigProt()
                    {free(proname);
                     if (libpath) free(libpath);
                     if (parms)   free(parms);
                    }
};

class XrdLogWorker : XrdJob
{
public:

     void DoIt() {XrdLog.Say(0, XrdBANNER);
                  XrdLog.Say(0, mememe, " running.");
                  midnite += 86400;
                  XrdSched.Schedule((XrdJob *)this, midnite);
                 }

          XrdLogWorker(char *who) : XrdJob("midnight runner")
                         {midnite = XrdSysTimer::Midnight() + 86400;
                          mememe = strdup(who);
                          XrdSched.Schedule((XrdJob *)this, midnite);
                         }
         ~XrdLogWorker() {}
private:
time_t midnite;
const char *mememe;
};

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdConfig::XrdConfig(void)
{

// Preset all variables with common defaults
//
   PortTCP  = -1;
   PortUDP  = -1;
   PortWAN  = 0;
   ConfigFN = 0;
   myInsName= 0;
   AdminPath= strdup("/tmp");
   AdminMode= 0700;
   Police   = 0;
   Net_Blen = 0;  // Accept OS default (leave Linux autotune in effect)
   Net_Opts = 0;
   Wan_Blen = 1024*1024; // Default window size 1M
   Wan_Opts = 0;
   setSched = 1;
   repDest[0] = 0;
   repDest[1] = 0;
   repInt     = 600;
   repOpts    = 0;

   Firstcp = Lastcp = 0;

   ProtInfo.eDest   = &XrdLog;          // Stable -> Error Message/Logging Handler
   ProtInfo.NetTCP  = 0;                // Stable -> Network Object
   ProtInfo.BPool   = &XrdBuffPool;     // Stable -> Buffer Pool Manager
   ProtInfo.Sched   = &XrdSched;        // Stable -> System Scheduler
   ProtInfo.ConfigFN= 0;                // We will fill this in later
   ProtInfo.Stats   = 0;                // We will fill this in later
   ProtInfo.Trace   = &XrdTrace;        // Stable -> Trace Information
   ProtInfo.Threads = 0;                // Stable -> The thread manager (later)
   ProtInfo.AdmPath = AdminPath;        // Stable -> The admin path
   ProtInfo.AdmMode = AdminMode;        // Stable -> The admin path mode

   ProtInfo.Format   = XrdFORMATB;
   ProtInfo.WANPort  = 0;
   ProtInfo.WANWSize = 0;
   ProtInfo.WSize    = 0;
   ProtInfo.ConnMax  = -1;     // Max       connections (fd limit)
   ProtInfo.readWait = 3*1000; // Wait time for data before we reschedule
   ProtInfo.idleWait = 0;      // Seconds connection may remain idle (0=off)
   ProtInfo.hailWait =30*1000; // Wait time for data before we drop connection
   ProtInfo.DebugON  = 0;      // 1 if started with -d
   ProtInfo.argc     = 0;
   ProtInfo.argv     = 0;
}
  
/******************************************************************************/
/*                             C o n f i g u r e                              */
/******************************************************************************/
  
int XrdConfig::Configure(int argc, char **argv)
{
/*
  Function: Establish configuration at start up time.

  Input:    None.

  Output:   0 upon success or !0 otherwise.
*/
   const char *xrdInst="XRDINSTANCE=";

   static sockaddr myIPAddr;
   int n, retc, dotrim = 1, NoGo = 0, aP = 1, clPort = -1, optbg = 0;
   const char *temp;
   char c, buff[512], *dfltProt, *logfn = 0;
   long long logkeep = 0;
   uid_t myUid = 0;
   gid_t myGid = 0;
   extern char *optarg;
   extern int optind, opterr;

// Obtain the protocol name we will be using
//
    retc = strlen(argv[0]);
    while(retc--) if (argv[0][retc] == '/') break;
    myProg = dfltProt = &argv[0][retc+1];

// Process the options
//
   opterr = 0;
   if (argc > 1 && '-' == *argv[1]) 
      while ((c = getopt(argc,argv,"bc:dhHk:l:n:p:P:R:"))
             && ((unsigned char)c != 0xff))
     { switch(c)
       {
       case 'b': optbg = 1;
                 break;
       case 'c': if (ConfigFN) free(ConfigFN);
                 ConfigFN = strdup(optarg);
                 break;
       case 'd': XrdTrace.What |= TRACE_ALL;
                 ProtInfo.DebugON = 1;
                 putenv((char *)"XRDDEBUG=1"); // XrdOucEnv::Export()
                 break;
       case 'h': Usage(0);
                 break;
       case 'H': Usage(-1);
                 break;
       case 'k': n = strlen(optarg)-1;
                 retc = (isalpha(optarg[n])
                        ? XrdOuca2x::a2sz(XrdLog,"keep size", optarg,&logkeep)
                        : XrdOuca2x::a2ll(XrdLog,"keep count",optarg,&logkeep));
                 if (retc) Usage(1);
                 if (!isalpha(optarg[n])) logkeep = -logkeep;
                 break;
       case 'l': if (logfn) free(logfn);
                 logfn = strdup(optarg);
                 break;
       case 'n': myInsName = optarg;
                 break;
       case 'p': if ((clPort = yport(&XrdLog, "tcp", optarg)) < 0) Usage(1);
                 break;
       case 'P': dfltProt = optarg; dotrim = 0;
                 break;
       case 'R': if (!(getUG(optarg, myUid, myGid))) Usage(1);
                 break;
       default:  if (index("clpP", (int)(*(argv[optind-1]+1))))
                    {XrdLog.Emsg("Config", argv[optind-1],
                                 "parameter not specified.");
                     Usage(1);
                    }
                 argv[aP++] = argv[optind-1];
                 if (argv[optind] && *argv[optind] != '-') 
                    argv[aP++] = argv[optind++];
       }
     }

// Drop into non-privileged state if so requested
//
   if (myGid && setegid(myGid))
      {XrdLog.Emsg("Config", errno, "set effective gid"); exit(17);}
   if (myUid && seteuid(myUid))
      {XrdLog.Emsg("Config", errno, "set effective uid"); exit(17);}

// Pass over any parameters
//
   if (aP != optind)
      {for ( ; optind < argc; optind++) argv[aP++] = argv[optind];
       argv[aP] = 0;
       ProtInfo.argc = aP;
      } else ProtInfo.argc = argc;
   ProtInfo.argv = argv;

// Resolve background/foreground issues
//
   if (optbg) XrdOucUtils::Undercover(XrdLog, !logfn);

// Bind the log file if we have one
//
   if (logfn)
      {char *lP;
       if (!(logfn = XrdOucUtils::subLogfn(XrdLog, myInsName, logfn))) _exit(16);
       if (logkeep) XrdLogger.setKeep(logkeep);
       XrdLogger.Bind(logfn, 24*60*60);
       if ((lP = rindex(logfn,'/'))) {*(lP+1) = '\0'; lP = logfn;}
          else lP = (char *)"./";
       XrdOucEnv::Export("XRDLOGDIR", lP);
       free(logfn);
      }

// Get the full host name. In theory, we should always get some kind of name.
//
   if (!(myName = XrdNetDNS::getHostName()))
      {XrdLog.Emsg("Config", "Unable to determine host name; "
                             "execution terminated.");
       _exit(16);
      }

// Verify that we have a real name. We've had problems with people setting up
// bad /etc/hosts files that can cause connection failures if "allow" is used.
// Otherwise, determine our domain name.
//
   if (isdigit(*myName) && (isdigit(*(myName+1)) || *(myName+1) == '.'))
      {XrdLog.Emsg("Config", myName, "is not the true host name of this machine.");
       XrdLog.Emsg("Config", "Verify that the '/etc/hosts' file is correct and "
                             "this machine is registered in DNS.");
       XrdLog.Emsg("Config", "Execution continues but connection failures may occur.");
       myDomain = 0;
      } else if (!(myDomain = index(myName, '.')))
                XrdLog.Say("Config warning: this hostname, ", myName,
                            ", is registered without a domain qualification.");

// Get our IP address
//
   XrdNetDNS::getHostAddr(myName, &myIPAddr);
   ProtInfo.myName = myName;
   ProtInfo.myAddr = &myIPAddr;
   ProtInfo.myInst = XrdOucUtils::InstName(myInsName);
   ProtInfo.myProg = myProg;

// Set the Environmental variable to hold the instance name
// XRDINSTANCE=<pgm> <instance name>@<host name>
//                 XrdOucEnv::Export("XRDINSTANCE")
//
   sprintf(buff,"%s%s %s@%s", xrdInst, myProg, ProtInfo.myInst, myName);
   myInstance = strdup(buff);
   putenv(myInstance);   // XrdOucEnv::Export("XRDINSTANCE",...)
   myInstance += strlen(xrdInst);
   XrdOucEnv::Export("XRDHOST", myName);
   XrdOucEnv::Export("XRDNAME", ProtInfo.myInst);
   XrdOucEnv::Export("XRDPROG", myProg);

// Put out the herald
//
   XrdLog.Say(0, "Scalla is starting. . .");
   XrdLog.Say(XrdBANNER);

// Setup the initial required protocol
//
   if (dotrim && *dfltProt != '.' )
      {char *p = dfltProt;
       while (*p && *p != '.') p++;
       if (*p == '.') *p = '\0';
      }
   Firstcp = Lastcp = new XrdConfigProt(strdup(dfltProt), 0, 0);

// Process the configuration file, if one is present
//
   XrdLog.Say("++++++ ", myInstance, " initialization started.");
   if (ConfigFN && *ConfigFN)
      {XrdLog.Say("Config using configuration file ", ConfigFN);
       ProtInfo.ConfigFN = ConfigFN;
       XrdOucEnv::Export("XRDCONFIGFN", ConfigFN);
       NoGo = ConfigProc();
      }
   if (clPort >= 0) PortTCP = clPort;
   if (ProtInfo.DebugON) 
      {XrdTrace.What = TRACE_ALL;
       XrdSysThread::setDebug(&XrdLog);
      }
   if (!NoGo) NoGo = Setup(dfltProt);
   ProtInfo.Threads = XrdThread;

// If we hae a net name change the working directory
//
   if (myInsName) XrdOucUtils::makeHome(XrdLog, myInsName);

// All done, close the stream and return the return code.
//
   temp = (NoGo ? " initialization failed." : " initialization completed.");
   sprintf(buff, "%s:%d", myInstance, PortTCP);
   XrdLog.Say("------ ", buff, temp);
   if (logfn) new XrdLogWorker(buff);
   return NoGo;
}

/******************************************************************************/
/*                             C o n f i g X e q                              */
/******************************************************************************/

int XrdConfig::ConfigXeq(char *var, XrdOucStream &Config, XrdSysError *eDest)
{
   int dynamic;

   // Determine whether is is dynamic or not
   //
   if (eDest) dynamic = 1;
      else   {dynamic = 0; eDest = &XrdLog;}

   // Process common items
   //
   TS_Xeq("buffers",       xbuf);
   TS_Xeq("network",       xnet);
   TS_Xeq("sched",         xsched);
   TS_Xeq("trace",         xtrace);

   // Process items that can only be processed once
   //
   if (!dynamic)
   {
   TS_Xeq("adminpath",     xapath);
   TS_Xeq("allow",         xallow);
   TS_Xeq("port",          xport);
   TS_Xeq("protocol",      xprot);
   TS_Xeq("report",        xrep);
   TS_Xeq("timeout",       xtmo);
   }

   // No match found, complain.
   //
   eDest->Say("Config warning: ignoring unknown xrd directive '",var,"'.");
   Config.Echo();
   return 0;
}

/******************************************************************************/
/*                     P r i v a t e   F u n c t i o n s                      */
/******************************************************************************/
/******************************************************************************/
/*                               A S o c k e t                                */
/******************************************************************************/
  
int XrdConfig::ASocket(const char *path, const char *fname, mode_t mode)
{
   char xpath[MAXPATHLEN+8], sokpath[108];
   int  plen = strlen(path), flen = strlen(fname);
   int rc;

// Make sure we can fit everything in our buffer
//
   if ((plen + flen + 3) > (int)sizeof(sokpath))
      {XrdLog.Emsg("Config", "admin path", path, "too long");
       return 1;
      }

// Create the directory path
//
   strcpy(xpath, path);
   if ((rc = XrdOucUtils::makePath(xpath, mode)))
       {XrdLog.Emsg("Config", rc, "create admin path", xpath);
        return 1;
       }

// *!*!* At this point we do not yet support the admin path for xrd.
// sp we comment out all of the following code.

/*
// Construct the actual socket name
//
  if (sokpath[plen-1] != '/') sokpath[plen++] = '/';
  strcpy(&sokpath[plen], fname);

// Create an admin network
//
   XrdNetADM = new XrdInet(&XrdLog);
   if (myDomain) XrdNetADM->setDomain(myDomain);

// Bind the netwok to the named socket
//
   if (!XrdNetADM->Bind(sokpath)) return 1;

// Set the mode and return
//
   chmod(sokpath, mode); // This may fail on some platforms
*/
   return 0;
}

/******************************************************************************/
/*                            C o n f i g P r o c                             */
/******************************************************************************/
  
int XrdConfig::ConfigProc()
{
  char *var;
  int  cfgFD, retc, NoGo = 0;
  XrdOucEnv myEnv;
  XrdOucStream Config(&XrdLog, myInstance, &myEnv, "=====> ");

// Try to open the configuration file.
//
   if ( (cfgFD = open(ConfigFN, O_RDONLY, 0)) < 0)
      {XrdLog.Emsg("Config", errno, "open config file", ConfigFN);
       return 1;
      }
   Config.Attach(cfgFD);

// Now start reading records until eof.
//
   while((var = Config.GetMyFirstWord()))
        if (!strncmp(var, "xrd.", 4)
        ||  !strcmp (var, "all.adminpath"))
           if (ConfigXeq(var+4, Config)) {Config.Echo(); NoGo = 1;}

// Now check if any errors occured during file i/o
//
   if ((retc = Config.LastError()))
      NoGo = XrdLog.Emsg("Config", retc, "read config file", ConfigFN);
   Config.Close();

// Return final return code
//
   return NoGo;
}

/******************************************************************************/
/*                                 g e t U G                                  */
/******************************************************************************/
  
int XrdConfig::getUG(char *parm, uid_t &newUid, gid_t &newGid)
{
   struct passwd *pp;

// Get the userid entry
//
   if (!(*parm))
      {XrdLog.Emsg("Config", "-R user not specified."); return 0;}

   if (isdigit(*parm))
      {if (!(newUid = atol(parm)))
          {XrdLog.Emsg("Config", "-R", parm, "is invalid"); return 0;}
       pp = getpwuid(newUid);
      }
      else pp = getpwnam(parm);

// Make sure it is valid and acceptable
//
   if (!pp) 
      {XrdLog.Emsg("Config", errno, "retrieve -R user password entry");
       return 0;
      }
   if (!(newUid = pp->pw_uid))
      {XrdLog.Emsg("Config", "-R", parm, "is still unacceptably a superuser!");
       return 0;
      }
   newGid = pp->pw_gid;
   return 1;
}

/******************************************************************************/
/*                                s e t F D L                                 */
/******************************************************************************/
  
int XrdConfig::setFDL()
{
   struct rlimit rlim;
   char buff[100];

// Get the resource limit
//
   if (getrlimit(RLIMIT_NOFILE, &rlim) < 0)
      return XrdLog.Emsg("Config", errno, "get FD limit");

// Set the limit to the maximum allowed
//
   rlim.rlim_cur = rlim.rlim_max;
#if (defined(__macos__) && defined(MAC_OS_X_VERSION_10_5))
   if (rlim.rlim_cur == RLIM_INFINITY || rlim.rlim_cur > OPEN_MAX)
     rlim.rlim_cur = OPEN_MAX;
#endif
   if (setrlimit(RLIMIT_NOFILE, &rlim) < 0)
      return XrdLog.Emsg("Config", errno,"set FD limit");

// Obtain the actual limit now
//
   if (getrlimit(RLIMIT_NOFILE, &rlim) < 0)
      return XrdLog.Emsg("Config", errno, "get FD limit");

// Establish operating limit
//
   ProtInfo.ConnMax = rlim.rlim_cur;
   sprintf(buff, "%d", ProtInfo.ConnMax);
   XrdLog.Say("Config maximum number of connections restricted to ", buff);

   return 0;
}

/******************************************************************************/
/*                                 S e t u p                                  */
/******************************************************************************/
  
int XrdConfig::Setup(char *dfltp)
{
   XrdInet *NetWAN;
   XrdConfigProt *cp, *pp, *po, *POrder = 0;
   int wsz, lastPort = -17;

// Establish the FD limit
//
   if (setFDL()) return 1;

// Special handling for Linux sendfile()
//
#if defined(__linux__) && defined(TCP_CORK)
{  int sokFD, setON = 1;
   if ((sokFD = socket(PF_INET, SOCK_STREAM, 0)) >= 0)
      {setsockopt(sokFD, XrdNetDNS::getProtoID("tcp"), TCP_NODELAY,
                  &setON, sizeof(setON));
       if (setsockopt(sokFD, SOL_TCP, TCP_CORK, &setON, sizeof(setON)) < 0)
          XrdLink::sfOK = 0;
       close(sokFD);
      }
}
#endif

// Indicate how sendfile is being handled
//
   TRACE(NET,"sendfile " <<(XrdLink::sfOK ? "enabled." : "disabled!"));

// Initialize the buffer manager
//
   XrdBuffPool.Init();

// Start the scheduler
//
   XrdSched.Start();

// Setup the link and socket polling infrastructure
//
   if (!XrdLink::Setup(ProtInfo.ConnMax, ProtInfo.idleWait)
   ||  !XrdPoll::Setup(ProtInfo.ConnMax)) return 1;

// Modify the AdminPath to account for any instance name. Note that there is
// a negligible memory leak under ceratin path combinations. Not enough to
// warrant a lot of logic to get around.
//
   if (myInsName) ProtInfo.AdmPath = XrdOucUtils::genPath(AdminPath,myInsName);
      else ProtInfo.AdmPath = AdminPath;
   XrdOucEnv::Export("XRDADMINPATH", ProtInfo.AdmPath);
   AdminPath = XrdOucUtils::genPath(AdminPath, myInsName, ".xrd");

// Setup admin connection now
//
   if (ASocket(AdminPath, "admin", (mode_t)AdminMode)) return 1;

// Determine the default port number (only for xrootd) if not specified.
//
   if (PortTCP < 0)  
      {if ((PortTCP = XrdNetDNS::getPort(dfltp, "tcp"))) PortUDP = PortTCP;
          else PortTCP = -1;
      }

// We now go through all of the protocols and get each respective port
// number and arrange them in descending port number order.
// XrdOucEnv::Export(XRDPORT
//
   while((cp = Firstcp))
        {ProtInfo.Port = (cp->port < 0 ? PortTCP : cp->port);
         XrdOucEnv::Export("XRDPORT", ProtInfo.Port);
         if ((cp->port = XrdProtLoad::Port(cp->libpath, cp->proname,
                                           cp->parms, &ProtInfo)) < 0) return 1;
         pp = 0; po = POrder; Firstcp = cp->Next;
         while(po && po->port > cp->port) {pp = po; po = po->Next;}
         if (pp) {pp->Next = cp;   cp->Next = po;}
            else {cp->Next = POrder; POrder = cp;}
        }

// Allocate the statistics object. This is akward since we only know part
// of the current configuration. The object will figure this out later.
//
   ProtInfo.Stats = new XrdStats(ProtInfo.myName, POrder->port,
                                 ProtInfo.myInst, ProtInfo.myProg);

// Allocate a WAN port number of we need to
//
   if (PortWAN &&  (NetWAN = new XrdInet(&XrdLog, Police)))
      {if (Wan_Opts || Wan_Blen) NetWAN->setDefaults(Wan_Opts, Wan_Blen);
       if (myDomain) NetWAN->setDomain(myDomain);
       if (NetWAN->Bind((PortWAN > 0 ? PortWAN : 0), "tcp")) return 1;
       PortWAN  = NetWAN->Port();
       wsz      = NetWAN->WSize();
       Wan_Blen = (wsz < Wan_Blen || !Wan_Blen ? wsz : Wan_Blen);
       TRACE(NET,"WAN port " <<PortWAN <<" wsz=" <<Wan_Blen <<" (" <<wsz <<')');
       XrdNetTCP[XrdProtLoad::ProtoMax] = NetWAN;
      } else {PortWAN = 0; Wan_Blen = 0;}

// Load the protocols. For each new protocol port number, create a new
// network object to handle the port dependent communications part. All
// port issues will have been resolved at this point.
//
   while((cp= POrder))
        {if (cp->port != lastPort)
            {XrdNetTCP[++XrdNetTCPlep] = new XrdInet(&XrdLog, Police);
             if (Net_Opts || Net_Blen)
                XrdNetTCP[XrdNetTCPlep]->setDefaults(Net_Opts, Net_Blen);
             if (myDomain) XrdNetTCP[XrdNetTCPlep]->setDomain(myDomain);
             if (XrdNetTCP[XrdNetTCPlep]->Bind(cp->port, "tcp")) return 1;
             ProtInfo.Port   = XrdNetTCP[XrdNetTCPlep]->Port();
             ProtInfo.NetTCP = XrdNetTCP[XrdNetTCPlep];
             wsz             = XrdNetTCP[XrdNetTCPlep]->WSize();
             ProtInfo.WSize  = (wsz < Net_Blen || !Net_Blen ? wsz : Net_Blen);
             TRACE(NET,"LCL port " <<ProtInfo.Port <<" wsz=" <<ProtInfo.WSize
                       <<" (" <<wsz <<')');
             if (cp->wanopt)
                {ProtInfo.WANPort = PortWAN;
                 ProtInfo.WANWSize= Wan_Blen;
                } else ProtInfo.WANPort = ProtInfo.WANWSize = 0;
             XrdOucEnv::Export("XRDPORT", ProtInfo.Port);
             lastPort = cp->port;
            }
         if (!XrdProtLoad::Load(cp->libpath,cp->proname,cp->parms,&ProtInfo))
            return 1;
         POrder = cp->Next;
         delete cp;
        }

// Leave the env port number to be the first used port number. This may
// or may not be the same as the default port number.
//
   ProtInfo.Port = XrdNetTCP[0]->Port();
   PortTCP = ProtInfo.Port;
   XrdOucEnv::Export("XRDPORT", PortTCP);

// Now check if we have to setup automatic reporting
//
   if (repDest[0] != 0 && repOpts) 
      ProtInfo.Stats->Report(repDest, repInt, repOpts);

// All done
//
   return 0;
}

/******************************************************************************/
/*                                 U s a g e                                  */
/******************************************************************************/
  
void XrdConfig::Usage(int rc)
{
  extern const char *XrdLicense;

  if (rc < 0) cerr <<XrdLicense;
     else
     cerr <<"\nUsage: " <<myProg <<" [-b] [-c <cfn>] [-d] [-k {n|sz}] [-l <fn>] "
            "[-L] [-n name] [-p <port>] [-P <prot>] [<prot_options>]" <<endl;
     _exit(rc > 0 ? rc : 0);
}

/******************************************************************************/
/*                                x a p a t h                                 */
/******************************************************************************/

/* Function: xapath

   Purpose:  To parse the directive: adminpath <path> [group]

             <path>    the path of the FIFO to use for admin requests.

             group     allows group access to the admin path

   Note: A named socket is created <path>/<name>/.xrd/admin

   Output: 0 upon success or !0 upon failure.
*/

int XrdConfig::xapath(XrdSysError *eDest, XrdOucStream &Config)
{
    char *pval, *val;
    mode_t mode = S_IRWXU;

// Get the path
//
   pval = Config.GetWord();
   if (!pval || !pval[0])
      {eDest->Emsg("Config", "adminpath not specified"); return 1;}

// Make sure it's an absolute path
//
   if (*pval != '/')
      {eDest->Emsg("Config", "adminpath not absolute"); return 1;}

// Record the path
//
   if (AdminPath) free(AdminPath);
   AdminPath = strdup(pval);

// Get the optional access rights
//
   if ((val = Config.GetWord()) && val[0])
      {if (!strcmp("group", val)) mode |= S_IRWXG;
          else {eDest->Emsg("Config", "invalid admin path modifier -", val);
                return 1;
               }
      }
   AdminMode = ProtInfo.AdmMode = mode;
   return 0;
}
  
/******************************************************************************/
/*                                x a l l o w                                 */
/******************************************************************************/

/* Function: xallow

   Purpose:  To parse the directive: allow {host | netgroup} <name>

             <name> The dns name of the host that is allowed to connect or the
                    netgroup name the host must be a member of. For DNS names,
                    a single asterisk may be specified anywhere in the name.

   Output: 0 upon success or !0 upon failure.
*/

int XrdConfig::xallow(XrdSysError *eDest, XrdOucStream &Config)
{
    char *val;
    int ishost;

    if (!(val = Config.GetWord()))
       {eDest->Emsg("Config", "allow type not specified"); return 1;}

    if (!strcmp(val, "host")) ishost = 1;
       else if (!strcmp(val, "netgroup")) ishost = 0;
               else {eDest->Emsg("Config", "invalid allow type -", val);
                     return 1;
                    }

    if (!(val = Config.GetWord()))
       {eDest->Emsg("Config", "allow target name not specified"); return 1;}

    if (!Police) Police = new XrdNetSecurity();
    if (ishost)  Police->AddHost(val);
       else      Police->AddNetGroup(val);

    return 0;
}

/******************************************************************************/
/*                                  x b u f                                   */
/******************************************************************************/

/* Function: xbuf

   Purpose:  To parse the directive: buffers <memsz> [<rint>]

             <memsz>    maximum amount of memory devoted to buffers
             <rint>     minimum buffer reshape interval in seconds

   Output: 0 upon success or !0 upon failure.
*/
int XrdConfig::xbuf(XrdSysError *eDest, XrdOucStream &Config)
{
    int bint = -1;
    long long blim;
    char *val;

    if (!(val = Config.GetWord()))
       {eDest->Emsg("Config", "buffer memory limit not specified"); return 1;}
    if (XrdOuca2x::a2sz(*eDest,"buffer limit value",val,&blim,
                       (long long)1024*1024)) return 1;

    if ((val = Config.GetWord()))
       if (XrdOuca2x::a2tm(*eDest,"reshape interval", val, &bint, 300))
          return 1;

    XrdBuffPool.Set((int)blim, bint);
    return 0;
}

/******************************************************************************/
/*                                  x n e t                                   */
/******************************************************************************/

/* Function: xnet

   Purpose:  To parse directive: network [wan] [keepalive] [buffsz <blen>]
                                         [[no]dnr]

             wan       parameters apply only to the wan port
             keepalive sets the socket keepalive option.
             <blen>    is the socket's send/rcv buffer size.
             [no]dnr   do [not] perform a reverse DNS lookup if not needed.

   Output: 0 upon success or !0 upon failure.
*/

int XrdConfig::xnet(XrdSysError *eDest, XrdOucStream &Config)
{
    char *val;
    int  i, V_keep = 0, V_nodnr = 0, V_iswan = 0, V_blen = -1;
    long long llp;
    static struct netopts {const char *opname; int hasarg; int opval;
                           int  *oploc;  const char *etxt;}
           ntopts[] =
       {
        {"keepalive",  0, 1, &V_keep,   "option"},
        {"buffsz",     1, 0, &V_blen,   "network buffsz"},
        {"dnr",        0, 0, &V_nodnr,  "option"},
        {"nodnr",      0, 1, &V_nodnr,  "option"},
        {"wan",        0, 1, &V_iswan,  "option"}
       };
    int numopts = sizeof(ntopts)/sizeof(struct netopts);

    if (!(val = Config.GetWord()))
       {eDest->Emsg("Config", "net option not specified"); return 1;}

    while (val)
    {for (i = 0; i < numopts; i++)
         if (!strcmp(val, ntopts[i].opname))
            {if (!ntopts[i].hasarg) llp=static_cast<long long>(ntopts[i].opval);
                else {if (!(val = Config.GetWord()))
                         {eDest->Emsg("Config", "network",
                              ntopts[i].opname, ntopts[i].etxt);
                          return 1;
                         }
                      if (XrdOuca2x::a2sz(*eDest,ntopts[i].etxt,val,&llp,0))
                         return 1;
                     }
             *ntopts[i].oploc = (int)llp;
              break;
            }
      if (i >= numopts)
         eDest->Say("Config warning: ignoring invalid net option '",val,"'.");
      val = Config.GetWord();
     }

     if (V_iswan)
        {if (V_blen >= 0) Wan_Blen = V_blen;
         Wan_Opts  = (V_keep  ? XRDNET_KEEPALIVE : 0)
                   | (V_nodnr ? XRDNET_NORLKUP   : 0);
         if (!PortWAN) PortWAN = -1;
        } else {
         if (V_blen >= 0) Net_Blen = V_blen;
         Net_Opts  = (V_keep  ? XRDNET_KEEPALIVE : 0)
                   | (V_nodnr ? XRDNET_NORLKUP   : 0);
        }
     return 0;
}

/******************************************************************************/
/*                                 x p o r t                                  */
/******************************************************************************/

/* Function: xport

   Purpose:  To parse the directive: port [wan] <tcpnum>
                                               [if [<hlst>] [named <nlst>]]

             wan        apply this to the wan port
             <tcpnum>   number of the tcp port for incomming requests
             <hlst>     list of applicable host patterns
             <nlst>     list of applicable instance names.

   Output: 0 upon success or !0 upon failure.
*/
int XrdConfig::xport(XrdSysError *eDest, XrdOucStream &Config)
{   int rc, iswan = 0, pnum = 0;
    char *val, cport[32];

    do {if (!(val = Config.GetWord()))
           {eDest->Emsg("Config", "tcp port not specified"); return 1;}
        if (strcmp("wan", val) || iswan) break;
        iswan = 1;
       } while(1);

    strncpy(cport, val, sizeof(cport)-1); cport[sizeof(cport)-1] = '\0';

    if ((val = Config.GetWord()) && !strcmp("if", val))
       if ((rc = XrdOucUtils::doIf(eDest,Config, "port directive", myName,
                              ProtInfo.myInst, myProg)) <= 0) return (rc < 0);

    if ((pnum = yport(eDest, "tcp", cport)) < 0) return 1;
    if (iswan) PortWAN = pnum;
       else PortTCP = PortUDP = pnum;

    return 0;
}

/******************************************************************************/

int XrdConfig::yport(XrdSysError *eDest, const char *ptype, const char *val)
{
    int pnum;
    if (!strcmp("any", val)) return 0;

    const char *invp = (*ptype == 't' ? "tcp port" : "udp port" );
    const char *invs = (*ptype == 't' ? "Unable to find tcp service" :
                                        "Unable to find udp service" );

    if (isdigit(*val))
       {if (XrdOuca2x::a2i(*eDest,invp,val,&pnum,1,65535)) return 0;}
       else if (!(pnum = XrdNetDNS::getPort(val, "tcp")))
               {eDest->Emsg("Config", invs, val);
                return -1;
               }
    return pnum;
}
  
/******************************************************************************/
/*                                 x p r o t                                  */
/******************************************************************************/

/* Function: xprot

   Purpose:  To parse the directive: protocol [wan] <name>[:<port>] <loc> [<parm>]

             wan    The protocol is WAN optimized
             <name> The name of the protocol (e.g., rootd)
             <port> Port binding for the protocol, if not the default.
             <loc>  The shared library in which it is located.
             <parm> A one line parameter to be passed to the protocol.

   Output: 0 upon success or !0 upon failure.
*/

int XrdConfig::xprot(XrdSysError *eDest, XrdOucStream &Config)
{
    XrdConfigProt *cpp;
    char *val, *parms, *lib, proname[64], buff[1024];
    int vlen, bleft = sizeof(buff), portnum = -1, wanopt = 0;

    do {if (!(val = Config.GetWord()))
           {eDest->Emsg("Config", "protocol name not specified"); return 1;}
        if (wanopt || strcmp("wan", val)) break;
        wanopt = 1;
       } while(1);

    if (strlen(val) > sizeof(proname)-1)
       {eDest->Emsg("Config", "protocol name is too long"); return 1;}
    strcpy(proname, val);

    if (!(val = Config.GetWord()))
       {eDest->Emsg("Config", "protocol library not specified"); return 1;}
    if (strcmp("*", val)) lib = strdup(val);
       else lib = 0;

    parms = buff;
    while((val = Config.GetWord()))
         {vlen = strlen(val); bleft -= (vlen+1);
          if (bleft <= 0)
             {eDest->Emsg("Config", "Too many parms for protocol", proname);
              return 1;
             }
          *parms = ' '; parms++; strcpy(parms, val); parms += vlen;
         }
    if (parms != buff) parms = strdup(buff+1);
       else parms = 0;

    if ((val = index(proname, ':')))
       {if ((portnum = yport(&XrdLog, "tcp", val+1)) < 0) return 1;
           else *val = '\0';
       }

    if (wanopt && !PortWAN) PortWAN = 1;

    if ((cpp = Firstcp))
       do {if (!strcmp(proname, cpp->proname))
              {if (cpp->libpath) free(cpp->libpath);
               if (cpp->parms)   free(cpp->parms);
               cpp->libpath = lib;
               cpp->parms   = parms;
               cpp->wanopt  = wanopt;
               return 0;
              }
          } while((cpp = cpp->Next));

    if (lib)
       {cpp = new XrdConfigProt(strdup(proname), lib, parms, portnum, wanopt);
        if (Lastcp) Lastcp->Next = cpp;
           else    Firstcp = cpp;
        Lastcp = cpp;
       }

    return 0;
}

/******************************************************************************/
/*                                  x r e p                                   */
/******************************************************************************/
  
/* Function: xrep

   Purpose:  To parse the directive: report <dest1>[,<dest2>]
                                            [every <sec>] <opts>

             <dest1>   where a UDP based report is to be sent. It may be a
                       <host:port> or a local named UDP pipe (i.e., "/...").

             <dest2>   A secondary destination.

             <sec>     the reporting interval. The default is 10 minutes.

             <opts>    What to report. "all" is the default.

  Output: 0 upon success or !0 upon failure.
*/

int XrdConfig::xrep(XrdSysError *eDest, XrdOucStream &Config)
{
   static struct repopts {const char *opname; int opval;} rpopts[] =
       {
        {"all",      XRD_STATS_ALL},
        {"buff",     XRD_STATS_BUFF},
        {"info",     XRD_STATS_INFO},
        {"link",     XRD_STATS_LINK},
        {"poll",     XRD_STATS_POLL},
        {"process",  XRD_STATS_PROC},
        {"protocols",XRD_STATS_PROT},
        {"prot",     XRD_STATS_PROT},
        {"sched",    XRD_STATS_SCHD},
        {"sgen",     XRD_STATS_SGEN},
        {"sync",     XRD_STATS_SYNC},
        {"syncwp",   XRD_STATS_SYNCA}
       };
   int i, neg, numopts = sizeof(rpopts)/sizeof(struct repopts);
   char  *val, *cp;

   if (!(val = Config.GetWord()))
      {eDest->Emsg("Config", "report parameters not specified"); return 1;}

// Cleanup to start anew
//
   if (repDest[0]) {free(repDest[0]); repDest[0] = 0;}
   if (repDest[1]) {free(repDest[1]); repDest[1] = 0;}
   repOpts = 0;
   repInt  = 600;

// Decode the destination
//
   if ((cp = (char *)index(val, ',')))
      {if (!*(cp+1))
          {eDest->Emsg("Config","malformed report destination -",val); return 1;}
          else { repDest[1] = cp+1; *cp = '\0';}
      }
   repDest[0] = val;
   for (i = 0; i < 2; i++)
       {if (!(val = repDest[i])) break;
        if (*val != '/' && (!(cp = index(val, (int)':')) || !atoi(cp+1)))
           {eDest->Emsg("Config","report dest port missing or invalid in",val);
            return 1;
           }
        repDest[i] = strdup(val);
       }

// Make sure dests differ
//
   if (repDest[0] && repDest[1] && !strcmp(repDest[0], repDest[1]))
      {eDest->Emsg("Config", "Warning, report dests are identical.");
       free(repDest[1]); repDest[1] = 0;
      }

// Get optional "every"
//
   if (!(val = Config.GetWord())) {repOpts = XRD_STATS_ALL; return 0;}
   if (!strcmp("every", val))
      {if (!(val = Config.GetWord()))
          {eDest->Emsg("Config", "report every value not specified"); return 1;}
       if (XrdOuca2x::a2tm(*eDest,"report every",val,&repInt,1)) return 1;
       val = Config.GetWord();
      }

// Get reporting options
//
   while(val)
        {if (!strcmp(val, "off")) repOpts = 0;
            else {if ((neg = (val[0] == '-' && val[1]))) val++;
                  for (i = 0; i < numopts; i++)
                      {if (!strcmp(val, rpopts[i].opname))
                          {if (neg) repOpts &= ~rpopts[i].opval;
                              else  repOpts |=  rpopts[i].opval;
                           break;
                          }
                      }
                  if (i >= numopts)
                     eDest->Say("Config warning: ignoring invalid report option '",val,"'.");
                 }
         val = Config.GetWord();
        }

// All done
//
   if (!(repOpts & XRD_STATS_ALL)) repOpts = XRD_STATS_ALL & ~XRD_STATS_INFO;
   return 0;
}

/******************************************************************************/
/*                                x s c h e d                                 */
/******************************************************************************/

/* Function: xsched

   Purpose:  To parse directive: sched [mint <mint>] [maxt <maxt>] [avlt <at>]
                                       [idle <idle>] [stksz <qnt>]

             <mint>   is the minimum number of threads that we need. Once
                      this number of threads is created, it does not decrease.
             <maxt>   maximum number of threads that may be created. The
                      actual number of threads will vary between <mint> and
                      <maxt>.
             <avlt>   Are the number of threads that must be available for
                      immediate dispatch. These threads are never bound to a
                      connection (i.e., made stickied). Any available threads
                      above <ft> will be allowed to stick to a connection.
             <idle>   The time (in time spec) between checks for underused
                      threads. Those found will be terminated. Default is 780.
             <qnt>    The thread stack size in bytes or K, M, or G.

   Output: 0 upon success or 1 upon failure.
*/

int XrdConfig::xsched(XrdSysError *eDest, XrdOucStream &Config)
{
    char *val;
    long long lpp;
    int  i, ppp;
    int  V_mint = -1, V_maxt = -1, V_idle = -1, V_avlt = -1;
    static struct schedopts {const char *opname; int minv; int *oploc;
                             const char *opmsg;} scopts[] =
       {
        {"stksz",      0,       0, "sched stksz"},
        {"mint",       1, &V_mint, "sched mint"},
        {"maxt",       1, &V_maxt, "sched maxt"},
        {"avlt",       1, &V_avlt, "sched avlt"},
        {"idle",       0, &V_idle, "sched idle"}
       };
    int numopts = sizeof(scopts)/sizeof(struct schedopts);

    if (!(val = Config.GetWord()))
       {eDest->Emsg("Config", "sched option not specified"); return 1;}

    while (val)
          {for (i = 0; i < numopts; i++)
               if (!strcmp(val, scopts[i].opname))
                  {if (!(val = Config.GetWord()))
                      {eDest->Emsg("Config", "sched", scopts[i].opname,
                                  "value not specified");
                       return 1;
                      }
                        if (*scopts[i].opname == 'i')
                           {if (XrdOuca2x::a2tm(*eDest, scopts[i].opmsg, val,
                                                &ppp, scopts[i].minv)) return 1;
                           }
                   else if (*scopts[i].opname == 's')
                           {if (XrdOuca2x::a2sz(*eDest, scopts[i].opmsg, val,
                                                &lpp, scopts[i].minv)) return 1;
                            XrdSysThread::setStackSize((size_t)lpp);
                            break;
                           }
                   else if (XrdOuca2x::a2i(*eDest, scopts[i].opmsg, val,
                                     &ppp,scopts[i].minv)) return 1;
                   *scopts[i].oploc = ppp;
                   break;
                  }
           if (i >= numopts)
              eDest->Say("Config warning: ignoring invalid sched option '",val,"'.");
           val = Config.GetWord();
          }

// Make sure specified quantities are consistent
//
  if (V_maxt > 0)
     {if (V_mint > 0 && V_mint > V_maxt)
         {eDest->Emsg("Config", "sched mint must be less than maxt");
          return 1;
         }
      if (V_avlt > 0 && V_avlt > V_maxt)
         {eDest->Emsg("Config", "sched avlt must be less than maxt");
          return 1;
         }
     }

// Establish scheduler options
//
   if (V_mint > 0 || V_maxt > 0 || V_avlt > 0) setSched = 0;
   XrdSched.setParms(V_mint, V_maxt, V_avlt, V_idle);
   return 0;
}

/******************************************************************************/
/*                                  x t m o                                   */
/******************************************************************************/

/* Function: xtmo

   Purpose:  To parse directive: timeout [read <msd>] [hail <msh>]
                                         [idle <msi>] [kill <msk>]

             <msd>    is the maximum number of seconds to wait for pending
                      data to arrive before we reschedule the link
                      (default is 5 seconds).
             <msh>    is the maximum number of seconds to wait for the initial
                      data after a connection  (default is 30 seconds)
             <msi>    is the minimum number of seconds a connection may remain
                      idle before it is closed (default is 5400 = 90 minutes)
             <msk>    is the minimum number of seconds to wait after killing a
                      connection for it to end (default is 3 seconds)

   Output: 0 upon success or 1 upon failure.
*/

int XrdConfig::xtmo(XrdSysError *eDest, XrdOucStream &Config)
{
    char *val;
    int  i, ppp, rc;
    int  V_read = -1, V_idle = -1, V_hail = -1, V_kill = -1;
    static struct tmoopts { const char *opname; int istime; int minv;
                            int  *oploc;  const char *etxt;}
           tmopts[] =
       {
        {"read",       1, 1, &V_read, "timeout read"},
        {"hail",       1, 1, &V_hail, "timeout hail"},
        {"idle",       1, 0, &V_idle, "timeout idle"},
        {"kill",       1, 0, &V_kill, "timeout kill"}
       };
    int numopts = sizeof(tmopts)/sizeof(struct tmoopts);

    if (!(val = Config.GetWord()))
       {eDest->Emsg("Config", "timeout option not specified"); return 1;}

    while (val)
          {for (i = 0; i < numopts; i++)
               if (!strcmp(val, tmopts[i].opname))
                   {if (!(val = Config.GetWord()))
                       {eDest->Emsg("Config","timeout", tmopts[i].opname,
                                   "value not specified");
                        return 1;
                       }
                    rc = (tmopts[i].istime ?
                          XrdOuca2x::a2tm(*eDest,tmopts[i].etxt,val,&ppp,
                                                 tmopts[i].minv) :
                          XrdOuca2x::a2i (*eDest,tmopts[i].etxt,val,&ppp,
                                                 tmopts[i].minv));
                    if (rc) return 1;
                    *tmopts[i].oploc = ppp;
                    break;
                   }
           if (i >= numopts)
              eDest->Say("Config warning: ignoring invalid timeout option '",val,"'.");
           val = Config.GetWord();
          }

// Set values and return
//
   if (V_read >  0) ProtInfo.readWait = V_read*1000;
   if (V_hail >= 0) ProtInfo.hailWait = V_hail*1000;
   if (V_idle >= 0) ProtInfo.idleWait = V_idle;
   XrdLink::setKWT(V_read, V_kill);
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

int XrdConfig::xtrace(XrdSysError *eDest, XrdOucStream &Config)
{
    char *val;
    static struct traceopts {const char *opname; int opval;} tropts[] =
       {
        {"all",      TRACE_ALL},
        {"off",      TRACE_NONE},
        {"none",     TRACE_NONE},
        {"conn",     TRACE_CONN},
        {"debug",    TRACE_DEBUG},
        {"mem",      TRACE_MEM},
        {"net",      TRACE_NET},
        {"poll",     TRACE_POLL},
        {"protocol", TRACE_PROT},
        {"sched",    TRACE_SCHED}
       };
    int i, neg, trval = 0, numopts = sizeof(tropts)/sizeof(struct traceopts);

    if (!(val = Config.GetWord()))
       {eDest->Emsg("Config", "trace option not specified"); return 1;}
    while (val)
         {if (!strcmp(val, "off")) trval = 0;
             else {if ((neg = (val[0] == '-' && val[1]))) val++;
                   for (i = 0; i < numopts; i++)
                       {if (!strcmp(val, tropts[i].opname))
                           {if (neg)
                               if (tropts[i].opval) trval &= ~tropts[i].opval;
                                  else trval = TRACE_ALL;
                               else if (tropts[i].opval) trval |= tropts[i].opval;
                                       else trval = TRACE_NONE;
                            break;
                           }
                       }
                   if (i >= numopts)
                      eDest->Say("Config warning: ignoring invalid trace option '",val,"'.");
                  }
          val = Config.GetWord();
         }
    XrdTrace.What = trval;
    return 0;
}
