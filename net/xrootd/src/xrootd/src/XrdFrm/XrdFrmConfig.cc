/******************************************************************************/
/*                                                                            */
/*                       X r d F r m C o n f i g . c c                        */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*                DE-AC02-76-SFO0515 with the Deprtment of Energy             */
/******************************************************************************/

//         $Id$

const char *XrdFrmConfigCVSID = "$Id$";
  
#include <unistd.h>
#include <ctype.h>
#include <dirent.h>
#include <string.h>
#include <stdio.h>
#include <fcntl.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "Xrd/XrdInfo.hh"
#include "XrdCms/XrdCmsNotify.hh"
#include "XrdFrm/XrdFrmConfig.hh"
#include "XrdFrm/XrdFrmTrace.hh"
#include "XrdNet/XrdNetDNS.hh"
#include "XrdOss/XrdOss.hh"
#include "XrdOss/XrdOssSpace.hh"
#include "XrdOuc/XrdOuca2x.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOuc/XrdOucMsubs.hh"
#include "XrdOuc/XrdOucName2Name.hh"
#include "XrdOuc/XrdOucProg.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdOuc/XrdOucTList.hh"
#include "XrdOuc/XrdOucUtils.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysLogger.hh"
#include "XrdSys/XrdSysPlugin.hh"
#include "XrdSys/XrdSysTimer.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdXrootd/XrdXrootdMonitor.hh"

using namespace XrdFrm;

/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/
  
class XrdFrmConfigSE
{
public:

XrdSysSemaphore mySem;
int             myFD;
int             seFD;
int             BLen;
char            Buff[32000];

                XrdFrmConfigSE() : mySem(0), myFD(-1), seFD(-1), BLen(0) {}
               ~XrdFrmConfigSE() {}
};

/******************************************************************************/
/*                     T h r e a d   I n t e r f a c e s                      */
/******************************************************************************/

void *XrdFrmConfigMum(void *parg)
{
   XrdFrmConfigSE *theSE = (XrdFrmConfigSE *)parg;
   char *bp = theSE->Buff;
   int  n, bleft = sizeof(theSE->Buff)-2;

// Let the calling thread continue at this point
//
   theSE->mySem.Post();

// Read everything we can
//
   do {if ((n = read(theSE->myFD, bp, bleft)) <= 0)
          {if (!n || (n < 0 && errno != EINTR)) break;}
       bp += n;
      } while ((bleft -= n));

// Refalgomize everything
//
   dup2(theSE->seFD, STDERR_FILENO);
   close(theSE->seFD);

// Check if we should add a newline character
//
   if (theSE->Buff[bp-(theSE->Buff)-1] != '\n') *bp++ = '\n';
   theSE->BLen = bp-(theSE->Buff);

// All done
//
   theSE->mySem.Post();
   pthread_exit((void *)0);
   return (void *)0;
}

void *XrdLogWorker(void *parg)
{
   time_t midnite = XrdSysTimer::Midnight() + 86400;
   char *mememe = strdup((char *)parg);

   while(1)
        {XrdSysTimer::Snooze(midnite-time(0));
         Say.Say(0, XrdBANNER);
         Say.Say(0, mememe, " running.");
         midnite += 86400;
        }
   return (void *)0;
}

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdFrmConfig::XrdFrmConfig(SubSys ss, const char *vopts, const char *uinfo)
{
   char *sP, buff[128];

// Preset all variables with common defaults
//
   vOpts    = vopts;
   uInfo    = uinfo;
   ssID     = ss;
   AdminPath= 0;
   AdminMode= 0740;
   xfrMax   = 1;
   WaitTime = 300;
   MSSCmd   = 0;
   xfrCmd   = strdup("/opt/xrootd/utils/frm_xfr -p $OFLAG $RFN $PFN");
   xfrVec   = 0;
   qPath    = 0;
   isAgent  = (getenv("XRDADMINPATH") ? 1 : 0);
   ossLib   = 0;
   cmsPath  = 0;
   monStage = 0;
   sSpec    = 0;
   Solitary = 0;
   lockFN   = "DIR_LOCK";  // May be ".DIR_LOCK" if hidden

   myUid    = geteuid();
   myGid    = getegid();

   LocalRoot= RemoteRoot = 0;
   lcl_N2N  = rmt_N2N = the_N2N = 0;
   N2N_Lib  = N2N_Parms         = 0;

// Establish our instance name
//
   if ((sP = getenv("XRDNAME")) && *sP) myInsName = sP;
      else myInsName = 0;

// Establish default config file
//
   if (!(sP = getenv("XRDCONFIGFN")) || !*sP) 
            ConfigFN = 0;
      else {ConfigFN = strdup(sP); isAgent = 1;}

// Establish directive postfix
//
        if (ss == ssAdmin) {myFrmid = "admin"; myFrmID = "ADMIN";}
   else if (ss == ssMigr)  {myFrmid = "migr";  myFrmID = "MIGR";}
   else if (ss == ssPstg)  {myFrmid = "pstg";  myFrmID = "PSTG";}
   else if (ss == ssPurg)  {myFrmid = "purg";  myFrmID = "PURG";}
   else                    {myFrmid = "frm";   myFrmID = "FRM";}

// Set correct error prefix
//
   strcpy(buff, myFrmid);
   strcat(buff, "_");
   Say.SetPrefix(strdup(buff));

// Set correct oss type
//
   sprintf(buff, "XRDOSSTYPE=%s", myFrmid);
   putenv(strdup(buff));

// Set correct option prefix
//
   strcpy(buff, "frm.");
   strcat(buff, myFrmid);
   strcat(buff, ".");
   pfxDTS = strdup(buff); plnDTS = strlen(buff);
}
  
/******************************************************************************/
/* Public:                     C o n f i g u r e                              */
/******************************************************************************/
  
int XrdFrmConfig::Configure(int argc, char **argv, int (*ppf)())
{
   extern XrdOss *XrdOssGetSS(XrdSysLogger *, const char *, const char *);
   XrdFrmConfigSE theSE;
   int n, retc, isMum = 0, myXfrMax = -1, NoGo = 0, Verbose = 0;
   const char *temp;
   char c, buff[1024], *logfn = 0;
   long long logkeep = 0;
   extern char *optarg;
   extern int opterr, optopt;

// Obtain the program name (used for logging)
//
    retc = strlen(argv[0]);
    while(retc--) if (argv[0][retc] == '/') break;
    myProg = &argv[0][retc+1];

// Process the options
//
   opterr = 0; nextArg = 1;
   if (argc > 1 && '-' == *argv[1]) 
      while ((c = getopt(argc,argv,vOpts)) && ((unsigned char)c != 0xff))
     { switch(c)
       {
       case 'c': if (ConfigFN) free(ConfigFN);
                 ConfigFN = strdup(optarg);
                 break;
       case 'd': Trace.What |= TRACE_ALL;
                 putenv((char *)"XRDDEBUG=1");
                 break;
       case 'h': Usage(0);
       case 'k': n = strlen(optarg)-1;
                 retc = (isalpha(optarg[n])
                        ? XrdOuca2x::a2sz(Say,"keep size", optarg,&logkeep)
                        : XrdOuca2x::a2ll(Say,"keep count",optarg,&logkeep));
                 if (retc) Usage(1);
                 if (!isalpha(optarg[n])) logkeep = -logkeep;
                 break;
       case 'l': if (logfn) free(logfn);
                 logfn = strdup(optarg);
                 break;
       case 'm': if (XrdOuca2x::a2i(Say,"max number",optarg,&myXfrMax))
                    Usage(1);
                 break;
       case 'n': myInsName = optarg;
                 break;
       case 's': sSpec = 1;
                 break;
       case 'v': Verbose = 1;
                 break;
       case 'w': if (XrdOuca2x::a2tm(Say,"wait time",optarg,&WaitTime))
                    Usage(1);
                 break;
       default:  sprintf(buff,"'%c'", optopt);
                 if (c == ':') Say.Emsg("Config", buff, "value not specified.");
                    else Say.Emsg("Config", buff, "option is invalid");
                 Usage(1);
       }
     nextArg = optind;
     }

// If we are an agent without a logfile and one is actually defined for the
// underlying system, use the directory of the underlying system.
//
   if (ssID != ssAdmin)
      {if (!logfn)
          {if (isAgent && (logfn = getenv("XRDLOGDIR")))
              {sprintf(buff, "%s%s%clog", logfn, myFrmid, (isAgent ? 'a' : 'd'));
               logfn = strdup(buff);
              }
          } else if (!(logfn=XrdOucUtils::subLogfn(Say,myInsName,logfn))) _exit(16);

   // Bind the log file if we have one
   //
       if (logfn)
          {if (logkeep) Logger.setKeep(logkeep);
           Logger.Bind(logfn, 24*60*60);
          }
       }

// Get the full host name. In theory, we should always get some kind of name.
//
   if (!(myName = XrdNetDNS::getHostName()))
      {Say.Emsg("Config","Unable to determine host name; execution terminated.");
       _exit(16);
      }

// Set the Environmental variables to hold some config information
// XRDINSTANCE=<pgm> <instance name>@<host name>
//
   sprintf(buff,"XRDINSTANCE=%s %s@%s",myProg,(myInst ? myInst:"anon"),myName);
   putenv(strdup(buff)); // XRDINSTANCE
   myInstance = strdup(index(buff,'=')+1);
   sprintf(buff,"XRDHOST=%s", myName); putenv(strdup(buff));
   sprintf(buff,"XRDPROG=%s", myProg); putenv(strdup(buff));
   if (myInsName)
      {sprintf(buff, "XRDNAME=%s", myInsName); putenv(strdup(buff));}

// We need to divert the output if we are in admin mode with no logfile
//
   if (!logfn && ssID == ssAdmin && !Verbose) isMum = ConfigMum(theSE);

// Put out the herald
//
   sprintf(buff, "Scalla %s is starting. . .", myProg);
   Say.Say(0, buff);
   Say.Say(XrdBANNER);

// Process the configuration file.
//
   Say.Say("++++++ ", myInstance, " initialization started.");
   if (!ConfigFN || !*ConfigFN) ConfigFN = strdup("/opt/xrootd/etc/xrootd.cf");
   Say.Say("Config using configuration file ", ConfigFN);
   NoGo = ConfigProc();

// Create the correct admin path
//
   if (!NoGo) NoGo = ConfigPaths();

// Obtain and configure the oss (lightweight option only)
//
   if (!isAgent)
      {putenv(strdup("XRDREDIRECT=Q"));
       Solitary = 1;
       if (!NoGo && !(ossFS=XrdOssGetSS(Say.logger(),ConfigFN,ossLib))) NoGo=1;
      }

// Configure the admin component
//
   if (!NoGo && ssID == ssAdmin
   && (ConfigN2N() || ConfigMss())) NoGo = 1;

// Configure the pstg  component
//
   if (!NoGo && ssID == ssPstg && !isAgent
   && (ConfigN2N() || !XrdXrootdMonitor::Init(0,&Say)
      || !(xfrVec = ConfigCmd("xfrcmd", xfrCmd)))) NoGo = 1;

// If we have a post-processing routine, invoke it
//
   if (!NoGo && ppf) NoGo = ppf();

// Start the log turn-over thread
//
   if (!NoGo && logfn)
      {pthread_t tid;
       if ((retc = XrdSysThread::Run(&tid, XrdLogWorker, (void *)myInstance,
                                     XRDSYSTHREAD_BIND, "midnight runner")))
          {Say.Emsg("Config", retc, "create logger thread"); NoGo = 1;}
      }

// Print ending message
//
   temp = (NoGo ? " initialization failed." : " initialization completed.");
   Say.Say("------ ", myInstance, temp);

// Finish up mum processing
//
   if (isMum)
      {close(STDERR_FILENO);
       theSE.mySem.Wait();
       if (NoGo) write(STDERR_FILENO, theSE.Buff, theSE.BLen);
      }

// All done
//
   return !NoGo;
}

/******************************************************************************/
/* Public:                     L o c a l P a t h                              */
/******************************************************************************/
  
int XrdFrmConfig::LocalPath(const char *oldp, char *newp, int newpsz)
{
    int rc = 0;

    if (lcl_N2N) rc = lcl_N2N->lfn2pfn(oldp, newp, newpsz);
       else if (((int)strlen(oldp)) >= newpsz) rc = ENAMETOOLONG;
               else strcpy(newp, oldp);
    if (rc) {Say.Emsg("Config", rc, "generate local path from", oldp);
             return 0;
            }
    return 1;
}

/******************************************************************************/
/* Public:                    R e m o t e P a t h                             */
/******************************************************************************/
  
int XrdFrmConfig::RemotePath(const char *oldp, char *newp, int newpsz)
{
    int rc = 0;

    if (rmt_N2N) rc = rmt_N2N->lfn2rfn(oldp, newp, newpsz);
       else if (((int)strlen(oldp)) >= newpsz) rc = ENAMETOOLONG;
               else strcpy(newp, oldp);
    if (rc) {Say.Emsg("Config", rc, "generate rmote path from", oldp);
             return 0;
            }
    return 1;
}
  
/******************************************************************************/
/*                                 S p a c e                                  */
/******************************************************************************/
  
XrdOucTList *XrdFrmConfig::Space(const char *Name, const char *Path)
{
   static XrdOucTList nullEnt;
   struct VPInfo *vP = VPList;
          XrdOucTList *tP;
   char buff[1032];
   int n;

// First find the space entry
//
   while(vP && strcmp(vP->Name, Name)) vP = vP->Next;
   if (!vP) return 0;

// Check if we should find a particular path
//
   if (!Path) return vP->Dir;

// Make sure it nds with a slash (it usually does not)
//
   n = strlen(Path)-1;
   if (Path[n] != '/')
      {if (n >= (int)sizeof(buff)-2) return &nullEnt;
       strcpy(buff, Path); buff[n+1] = '/'; buff[n+2] = '\0';
       Path = buff;
      }

// Find the path
//
   tP = vP->Dir;
   while(tP && strcmp(Path, tP->text)) tP = tP->next;
   return (tP ? tP : &nullEnt);
}

/******************************************************************************/
/*                     P r i v a t e   F u n c t i o n s                      */
/******************************************************************************/
/******************************************************************************/
/* Private:                    C o n f i g C m d                              */
/******************************************************************************/
  
XrdOucMsubs *XrdFrmConfig::ConfigCmd(const char *cname, char *cdata)
{
   XrdOucMsubs *msubs;
   char *cP;

   if (!cdata) {Say.Emsg("Config", cname, "not specified."); return 0;}

   if ((cP = index(cdata, ' '))) *cP = '\0';

   if (access(cdata, X_OK))
      {Say.Emsg("Config", errno, "set up", cdata);
       return 0;
      }
   *cP = ' ';

   msubs = new XrdOucMsubs(&Say);
   if (msubs->Parse(cname, cdata)) return msubs;

   return 0;  // We will exit no need to delete msubs
}

/******************************************************************************/
/* Private:                    C o n f i g M s s                              */
/******************************************************************************/
  
int XrdFrmConfig::ConfigMss()
{
   if (MSSCmd)
      {MSSProg = new XrdOucProg(&Say);
       if (MSSProg->Setup(MSSCmd)) return 1;
      }
   return 0;
}

/******************************************************************************/
/* Private:                    C o n f i g N 2 N                              */
/******************************************************************************/

int XrdFrmConfig::ConfigN2N()
{
   XrdSysPlugin    *myLib;
   XrdOucName2Name *(*ep)(XrdOucgetName2NameArgs);

// If we have no library path then use the default method (this will always
// succeed).
//
   if (!N2N_Lib)
      {the_N2N = XrdOucgetName2Name(&Say, ConfigFN, "", LocalRoot, RemoteRoot);
       if (LocalRoot)  lcl_N2N = the_N2N;
       if (RemoteRoot) rmt_N2N = the_N2N;
       return 0;
      }

// Create a pluin object (we will throw this away without deletion because
// the library must stay open but we never want to reference it again).
//
   if (!(myLib = new XrdSysPlugin(&Say, N2N_Lib))) return 1;

// Now get the entry point of the object creator
//
   ep = (XrdOucName2Name *(*)(XrdOucgetName2NameArgs))(myLib->getPlugin("XrdOucgetName2Name"));
   if (!ep) return 1;


// Get the Object now
//
   lcl_N2N = rmt_N2N = the_N2N = ep(&Say, ConfigFN, 
                                   (N2N_Parms ? N2N_Parms : ""),
                                   LocalRoot, RemoteRoot);
   return lcl_N2N == 0;
}

/******************************************************************************/
/*                             C o n f i g M u m                              */
/******************************************************************************/

int XrdFrmConfig::ConfigMum(XrdFrmConfigSE &theSE)
{
   class Recover
        {public:
         int fdvec[2];
         int stdErr;
             Recover() : stdErr(-1) {fdvec[0] = -1; fdvec[1] = -1;}
            ~Recover() {if (fdvec[0] >= 0) close(fdvec[0]);
                        if (fdvec[1] >= 0) close(fdvec[1]);
                        if (stdErr >= 0)   {dup2(stdErr, STDERR_FILENO);
                                            close(stdErr);
                                           }
                       }
        };
   Recover FD;
   pthread_t tid;
   int rc;

// Create a pipe
//
   if (pipe(FD.fdvec) < 0) return 0;
   fcntl(FD.fdvec[0], F_SETFD, FD_CLOEXEC);

// Save the current standard error FD
//
   if ((FD.stdErr = dup(STDERR_FILENO)) < 0) return 0;

// Now hook-up the pipe to standard error
//
   if (dup2(FD.fdvec[1], STDERR_FILENO) < 0) return 0;
   close(FD.fdvec[1]); FD.fdvec[1] = -1;

// Prepare arguments to the thread that will suck up the output
//
   theSE.myFD = FD.fdvec[0];
   theSE.seFD = FD.stdErr;

// Start a thread to read all of the output
//
    if ((rc = XrdSysThread::Run(&tid, XrdFrmConfigMum, (void *)&theSE,
                                XRDSYSTHREAD_BIND, "Mumify"))) return 0;

// Now fixup to return correctly
//
   theSE.mySem.Wait();
   FD.fdvec[0] = -1;
   FD.stdErr = -1;
   return 1;
}
  
/******************************************************************************/
/*                           C o n f i g P a t h s                            */
/******************************************************************************/
  
int XrdFrmConfig::ConfigPaths()
{
   char *xPath, *yPath, buff[MAXPATHLEN];
   int retc;

// Establish the cmsd notification path
//
   if (!(xPath = AdminPath) && !(xPath = getenv("XRDADMINPATH")))
      xPath = (char *)"/tmp/";
   cmsPath = new XrdCmsNotify(&Say, xPath, myInsName, XrdCmsNotify::isServ);

// Set the directory where the meta information is to go
//
   yPath = XrdOucUtils::genPath(xPath, myInsName, "frm");
   if (AdminPath) free(AdminPath); AdminPath = yPath;

// Create the admin directory if it does not exists
//
   if ((retc = XrdOucUtils::makePath(AdminPath, AdminMode)))
      {Say.Emsg("Config", retc, "create admin directory", AdminPath);
       return 0;
      }

// Now we should create a home directory for core files
//
   if (myInsName) XrdOucUtils::makeHome(Say, myInsName);

// Set up the stop file path
//
   if (!StopFile)
      {sprintf(buff,"%sSTOP%s", AdminPath, myFrmID); StopFile = strdup(buff);}

// If a qpath was specified, differentiate it by the instance name
//
   if (qPath)
      {xPath = XrdOucUtils::genPath(qPath, myInsName, "frm");
       free(qPath); qPath = xPath;
      }

// All done
//
   return 0;
}

/******************************************************************************/
/* Private:                   C o n f i g P r o c                             */
/******************************************************************************/
  
int XrdFrmConfig::ConfigProc()
{
  char *var;
  int  cfgFD, retc, mbok, NoGo = 0;
  XrdOucEnv myEnv;
  XrdOucStream cfgFile(&Say, myInstance, &myEnv, "=====> ");

// Try to open the configuration file.
//
   if ( (cfgFD = open(ConfigFN, O_RDONLY, 0)) < 0)
      {Say.Emsg("Config", errno, "open config file", ConfigFN);
       return 1;
      }
   cfgFile.Attach(cfgFD); cFile = &cfgFile;

// Now start reading records until eof.
//
   while((var = cFile->GetMyFirstWord()))
        {mbok = 0;
         if (!strncmp(var, pfxDTS, plnDTS)) {var += plnDTS; mbok = 1;}
         if(ConfigXeq(var, mbok)) {cfgFile.Echo(); NoGo = 1;}
        }

// Now check if any errors occured during file i/o
//
   if ((retc = cfgFile.LastError()))
      NoGo = Say.Emsg("Config", retc, "read config file", ConfigFN);
   cfgFile.Close(); cFile = 0;

// Return final return code
//
   return NoGo;
}

/******************************************************************************/
/* Prvate:                     C o n f i g X e q                              */
/******************************************************************************/

int XrdFrmConfig::ConfigXeq(char *var, int mbok)
{

// Process common items to all subsystems
//
   if (!strcmp(var, "all.adminpath" )) return xapath();

// Process directives specific to each subsystem
//
// if (ssID == ssAdmin)
//    {if (!strcmp(var, "oss.mssgwcmd"  )) return Grab(var, &MSSCmd,    0);
//     if (!strcmp(var, "oss.msscmd"    )) return Grab(var, &MSSCmd,    0);
//    }

   if (ssID == ssPstg || ssID == ssAdmin)
      {if (!strcmp(var, "ofs.osslib"    )) return Grab(var, &ossLib,    0);
       if (!strcmp(var, "oss.cache"     )) return xcache();
       if (!strcmp(var, "oss.localroot" )) return Grab(var, &LocalRoot, 0);
       if (!strcmp(var, "oss.namelib"   )) return xnml();
       if (!strcmp(var, "oss.remoteroot")) return Grab(var, &LocalRoot, 0);
      }

   if (ssID == ssPstg)
      {
       if (!strcmp(var, "xrootd.monitor")) return xmon();
       if (!strcmp(var, "waittime"      )) return xwtm();
       if (!strcmp(var, "xfrmax"        )) return xmaxx();
       if (!strcmp(var, "xfrcmd"        )) return Grab(var, &xfrCmd,    1);
       if (!strcmp(var, "stopfile"      )) return Grab(var, &StopFile,  0);
       if (!strcmp(var, "queuepath"     )) return Grab(var, &qPath,     0);
      }

   // No match found, complain.
   //
   if (!mbok) cFile->noEcho();
      else {Say.Say("Config warning: ignoring unknown frm directive '",var,"'.");
            cFile->Echo();
           }
   return 0;
}

/******************************************************************************/
/* Private:                         G r a b                                   */
/******************************************************************************/
  
int XrdFrmConfig::Grab(const char *var, char **Dest, int nosubs)
{
    char  myVar[80], buff[2048], *val;
    XrdOucEnv *myEnv = 0;

// Copy the variable name as this may change because it points to an
// internal buffer in Config. The vagaries of effeciency.
//
   strlcpy(myVar, var, sizeof(myVar));
   var = myVar;

// If substitutions allowed then we need to grab a single token else grab
// the remainder of the line but suppress substitutions.
//
   if (!nosubs) val = cFile->GetWord();
      else {myEnv = cFile->SetEnv(0);
            if (!cFile->GetRest(buff, sizeof(buff)))
               {Say.Emsg("Config", "arguments too long for", var);
                cFile->SetEnv(myEnv);
                return 1;
               }
            val = buff;
            cFile->SetEnv(myEnv);
           }

// At this point, make sure we have a value
//
   if (!val || !(*val))
      {Say.Emsg("Config", "no value for directive", var);
       return 1;
      }

// Set the value
//
   if (*Dest) free(*Dest);
   *Dest = strdup(val);
   return 0;
}

/******************************************************************************/
/* Private:                        U s a g e                                  */
/******************************************************************************/
  
void XrdFrmConfig::Usage(int rc)
{
     cerr <<"\nUsage: " <<myProg <<" " <<uInfo <<endl;
     _exit(rc);
}


/******************************************************************************/
/*                                x c a c h e                                 */
/******************************************************************************/

/* Function: xcache

   Purpose:  To parse the directive: cache <group> <path> [xa]

             <group>  logical group name for the cache filesystem.
             <path>   path to the cache.
             xa       support extended attributes

   Output: 0 upon success or !0 upon failure.
*/

int XrdFrmConfig::xcache()
{
   char *val, *pfxdir, *sfxdir;
   char grp[XrdOssSpace::minSNbsz], fn[MAXPATHLEN], dn[MAXNAMLEN];
   int i, k, rc, pfxln, isxa = 0, cnum = 0;
   struct dirent *dp;
   struct stat buff;
   DIR *DFD;

   if (!(val = cFile->GetWord()))
      {Say.Emsg("Config", "cache group not specified"); return 1;}
   if (strlen(val) >= (int)sizeof(grp))
      {Say.Emsg("Config","excessively long cache name - ",val); return 1;}
   strcpy(grp, val);

   if (!(val = cFile->GetWord()))
      {Say.Emsg("Config", "cache path not specified"); return 1;}

   k = strlen(val);
   if (k >= (int)(sizeof(fn)-1) || val[0] != '/' || k < 2)
      {Say.Emsg("Config", "invalid cache path - ", val); return 1;}
   strcpy(fn, val);

   if ((val = cFile->GetWord()))
      {if (strcmp("xa", val))
          {Say.Emsg("Config","invalid cache option - ",val); return 1;}
          else isxa = 1;
      }

   if (fn[k-1] != '*')
      {for (i = k-1; i; i--) if (fn[i] != '/') break;
       fn[i+1] = '/'; fn[i+2] = '\0';
       xcacheBuild(grp, fn, isxa);
       return 0;
      }

   for (i = k-1; i; i--) if (fn[i] == '/') break;
   i++; strcpy(dn, &fn[i]); fn[i] = '\0';
   sfxdir = &fn[i]; pfxdir = dn; pfxln = strlen(dn)-1;
   if (!(DFD = opendir(fn)))
      {Say.Emsg("Config", errno, "open cache directory", fn); return 1;}

   errno = 0;
   while((dp = readdir(DFD)))
        {if (!strcmp(dp->d_name, ".") || !strcmp(dp->d_name, "..")
         || (pfxln && strncmp(dp->d_name, pfxdir, pfxln)))
            continue;
         strcpy(sfxdir, dp->d_name);
         if (stat(fn, &buff)) break;
         if ((buff.st_mode & S_IFMT) == S_IFDIR)
            {val = sfxdir + strlen(sfxdir) - 1;
            if (*val++ != '/') {*val++ = '/'; *val = '\0';}
            xcacheBuild(grp, fn, isxa);
            cnum++;
            }
         errno = 0;
        }

   if ((rc = errno))
      Say.Emsg("Config", errno, "process cache directory", fn);
      else if (!cnum) Say.Say("Config warning: no cache directories found in ",val);

   closedir(DFD);
   return rc != 0;
}

void XrdFrmConfig::xcacheBuild(char *grp, char *fn, int isxa)
{
   struct VPInfo *nP = VPList;
   XrdOucTList *tP;

   while(nP && strcmp(nP->Name, grp)) nP = nP->Next;

   if (!nP) VPList = nP = new VPInfo(grp, VPList);

   tP = nP->Dir;
   while(tP && strcmp(tP->text, fn)) tP = tP->next;
   if (!tP) nP->Dir = new XrdOucTList(fn, isxa, nP->Dir);
}

/******************************************************************************/
/* Private:                       x a p a t h                                 */
/******************************************************************************/

/* Function: xapath

   Purpose:  To parse the directive: adminpath <path> [group]

             <path>    the path of the FIFO to use for admin requests.

             group     allows group access to the admin path

   Output: 0 upon success or !0 upon failure.
*/

int XrdFrmConfig::xapath()
{
    char *pval, *val;
    mode_t mode = S_IRWXU;

// Get the path
//
   pval = cFile->GetWord();
   if (!pval || !pval[0])
      {Say.Emsg("Config", "adminpath not specified"); return 1;}

// Make sure it's an absolute path
//
   if (*pval != '/')
      {Say.Emsg("Config", "adminpath not absolute"); return 1;}

// Record the path
//
   if (AdminPath) free(AdminPath);
   AdminPath = strdup(pval);

// Get the optional access rights
//
   if ((val = cFile->GetWord()) && val[0])
      {if (!strcmp("group", val)) mode |= S_IRWXG;
          else {Say.Emsg("Config", "invalid admin path modifier -", val);
                return 1;
               }
      }
   AdminMode = mode;
   return 0;
}
  
/******************************************************************************/
/* Private:                        x m a x x                                  */
/******************************************************************************/

/* Function: xmaxx

   Purpose:  To parse the directive: xfrmax <num>

             <num>     maximum number of simultaneous transfers

   Output: 0 upon success or !0 upon failure.
*/
int XrdFrmConfig::xmaxx()
{   int xmax = 1;
    char *val;

    if (!(val = cFile->GetWord()))
       {Say.Emsg("Config", "xfrmax value not specified"); return 1;}
    if (XrdOuca2x::a2tm(Say, "xfrmax", val, &xmax, 1)) return 1;
    xfrMax = xmax;
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
int XrdFrmConfig::xmon()
{   char  *val, *cp, *monDest[2] = {0, 0};
    long long tempval;
    int i, monFlush=0, monMBval=0, monWWval=0, xmode=0, monMode[2] = {0, 0};

    while((val = cFile->GetWord()))

         {     if (!strcmp("all",  val)) xmode = XROOTD_MON_ALL;
          else if (!strcmp("flush", val))
                {if (!(val = cFile->GetWord()))
                    {Say.Emsg("Config", "monitor flush value not specified");
                     return 1;
                    }
                 if (XrdOuca2x::a2tm(Say,"monitor flush",val,
                                         &monFlush,1)) return 1;
                }
          else if (!strcmp("mbuff",val))
                  {if (!(val = cFile->GetWord()))
                      {Say.Emsg("Config", "monitor mbuff value not specified");
                       return 1;
                      }
                   if (XrdOuca2x::a2sz(Say,"monitor mbuff", val,
                                           &tempval, 1024, 65536)) return 1;
                    monMBval = static_cast<int>(tempval);
                  }
          else if (!strcmp("window", val))
                {if (!(val = cFile->GetWord()))
                    {Say.Emsg("Config", "monitor window value not specified");
                     return 1;
                    }
                 if (XrdOuca2x::a2tm(Say,"monitor window",val,
                                         &monWWval,1)) return 1;
                }
          else break;
         }

    if (!val) {Say.Emsg("Config", "monitor dest not specified"); return 1;}

    for (i = 0; i < 2; i++)
        {if (strcmp("dest", val)) break;
         while((val = cFile->GetWord()))
                   if (!strcmp("files",val)
                   ||  !strcmp("info", val)
                   ||  !strcmp("io",   val)
                   ||  !strcmp("user", val)) {}
              else if (!strcmp("stage",val)) monMode[i] |=  XROOTD_MON_STAGE;
              else break;
          if (!val) {Say.Emsg("Config","monitor dest value not specified");
                     return 1;
                    }
          if (!(cp = index(val, (int)':')) || !atoi(cp+1))
             {Say.Emsg("Config","monitor dest port missing or invalid in",val);
              return 1;
             }
          monDest[i] = strdup(val);
         if (!(val = cFile->GetWord())) break;
        }

    if (val)
       {if (!strcmp("dest", val))
           Say.Emsg("Config", "Warning, a maximum of two dest values allowed.");
           else Say.Emsg("Config", "Warning, invalid monitor option", val);
       }

// Make sure dests differ
//
   if (monDest[0] && monDest[1] && !strcmp(monDest[0], monDest[1]))
      {Say.Emsg("Config", "Warning, monitor dests are identical.");
       monMode[0] |= monMode[1]; monMode[1] = 0;
       free(monDest[1]); monDest[1] = 0;
      }

// Don't bother doing any more if staging is not enabled
//
   if (!monMode[0] && !monMode[1]) return 0;
   monStage = 1;

// Set the monitor defaults
//
   XrdXrootdMonitor::Defaults(monMBval, monWWval, monFlush);
   if (monDest[0]) monMode[0] |= (monMode[0] ? XROOTD_MON_FILE|xmode : xmode);
   if (monDest[1]) monMode[1] |= (monMode[1] ? XROOTD_MON_FILE|xmode : xmode);
   XrdXrootdMonitor::Defaults(monDest[0],monMode[0],monDest[1],monMode[1]);
   return 0;
}

/******************************************************************************/
/* Private:                         x n m l                                   */
/******************************************************************************/

/* Function: xnml

   Purpose:  To parse the directive: namelib <path> [<parms>]

             <path>    the path of the filesystem library to be used.
             <parms>   optional parms to be passed

  Output: 0 upon success or !0 upon failure.
*/

int XrdFrmConfig::xnml()
{
    char *val, parms[1024];

// Get the path
//
   if (!(val = cFile->GetWord()) || !val[0])
      {Say.Emsg("Config", "namelib not specified"); return 1;}

// Record the path
//
   if (N2N_Lib) free(N2N_Lib);
   N2N_Lib = strdup(val);

// Record any parms
//
   if (!cFile->GetRest(parms, sizeof(parms)))
      {Say.Emsg("Config", "namelib parameters too long"); return 1;}
   if (N2N_Parms) free(N2N_Parms);
   N2N_Parms = (*parms ? strdup(parms) : 0);
   return 0;
}

/******************************************************************************/
/* Private:                         x w t m                                   */
/******************************************************************************/

/* Function: xwtm

   Purpose:  To parse the directive: waittime <sec>

             <sec>     number of seconds between scans.

   Output: 0 upon success or !0 upon failure.
*/
int XrdFrmConfig::xwtm()
{   int wscan = 0;
    char *val;

    if (!(val = cFile->GetWord()))
       {Say.Emsg("Config", "wait time not specified"); return 1;}
    if (XrdOuca2x::a2tm(Say, "wait time", val, &wscan, 30)) return 1;
    WaitTime = wscan;
    return 0;
}
