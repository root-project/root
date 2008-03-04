/******************************************************************************/
/*                                                                            */
/*                       X r d O s s C o n f i g . c c                        */
/*                                                                            */
/*                                                                            */
/* (C) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*                DE-AC03-76-SFO0515 with the Deprtment of Energy             */
/******************************************************************************/

//         $Id$

const char *XrdOssConfigCVSID = "$Id$";

/*
   The routines in this file handle initialization. They get the
   configuration values either from configuration file or XrdOssconfig.h (in that
   order of precedence).

   These routines are thread-safe if compiled with:
   AIX: -D_THREAD_SAFE
   SUN: -D_REENTRANT
*/
  
#include <unistd.h>
#include <ctype.h>
#include <dirent.h>
#include <fcntl.h>
#include <iostream.h>
#include <strings.h>
#include <stdio.h>
#include <sys/param.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "XrdOss/XrdOssApi.hh"
#include "XrdOss/XrdOssConfig.hh"
#include "XrdOss/XrdOssError.hh"
#include "XrdOss/XrdOssMio.hh"
#include "XrdOss/XrdOssOpaque.hh"
#include "XrdOss/XrdOssTrace.hh"
#include "XrdOuc/XrdOuca2x.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdOuc/XrdOucExport.hh"
#include "XrdOuc/XrdOucMsubs.hh"
#include "XrdOuc/XrdOucName2Name.hh"
#include "XrdOuc/XrdOucProg.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdSys/XrdSysPlugin.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdSys/XrdSysPlatform.hh"

/******************************************************************************/
/*                 S t o r a g e   S y s t e m   O b j e c t                  */
/******************************************************************************/
  
extern XrdOssSys   *XrdOssSS;

extern XrdOucTrace  OssTrace;

/******************************************************************************/
/*                            E r r o r   T e x t                             */
/******************************************************************************/
  
const char *XrdOssErrorText[] =
      {XRDOSS_T8001,
       XRDOSS_T8002,
       XRDOSS_T8003,
       XRDOSS_T8004,
       XRDOSS_T8005,
       XRDOSS_T8006,
       XRDOSS_T8007,
       XRDOSS_T8008,
       XRDOSS_T8009,
       XRDOSS_T8010,
       XRDOSS_T8011,
       XRDOSS_T8012,
       XRDOSS_T8013,
       XRDOSS_T8014,
       XRDOSS_T8015,
       XRDOSS_T8016,
       XRDOSS_T8017,
       XRDOSS_T8018,
       XRDOSS_T8019,
       XRDOSS_T8020,
       XRDOSS_T8021,
       XRDOSS_T8022,
       XRDOSS_T8023,
       XRDOSS_T8024,
       XRDOSS_T8025
      };

/******************************************************************************/
/*                               d e f i n e s                                */
/******************************************************************************/

#define Duplicate(x,y) if (y) free(y); y = strdup(x)

#define TS_Xeq(x,m)    if (!strcmp(x,var)) return m(Config, Eroute);

#define TS_String(x,m) if (!strcmp(x,var)) {free(m); m = strdup(val); return 0;}

#define TS_List(x,m,v) if (!strcmp(x,var)) \
                          {m.Insert(new XrdOucPList(val, v); return 0;}

#define TS_Char(x,m)   if (!strcmp(x,var)) {m = val[0]; return 0;}

#define TS_Add(x,m,v,s) if (!strcmp(x,var)) {m |= (v|s); return 0;}
#define TS_Ade(x,m,v,s) if (!strcmp(x,var)) {m |= (v|s); Config.Echo(); return 0;}
#define TS_Rem(x,m,v,s) if (!strcmp(x,var)) {m = (m & ~v) | s; return 0;}

#define TS_Set(x,m,v)  if (!strcmp(x,var)) {m = v; Config.Echo(); return 0;}

#define xrdmax(a,b)       (a < b ? b : a)

/******************************************************************************/
/*            E x t e r n a l   T h r e a d   I n t e r f a c e s             */
/******************************************************************************/
  
void *XrdOssxfr(void *carg)       {return XrdOssSS->Stage_In(carg);}

void *XrdOssCacheScan(void *carg) {return XrdOssSS->CacheScan(carg);}

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/

XrdOssSys::XrdOssSys()
{
   static char *syssfx[] = {XRDOSS_SFX_LIST, 0};

   sfx           = syssfx;
   xfrtcount     = 0;
   fsdata        = 0;
   fsfirst       = 0;
   fslast        = 0;
   fscurr        = 0;
   fsgroups      = 0;
   xsdata        = 0;
   xsfirst       = 0;
   xslast        = 0;
   xscurr        = 0;
   xsgroups      = 0;
   pndbytes      = 0;
   stgbytes      = 0;
   totbytes      = 0;
   totreqs       = 0;
   badreqs       = 0;
   CompSuffix    = 0;
   CompSuflen    = 0;
   MaxTwiddle    = 3;
   tryMmap       = 0;
   chkMmap       = 0;
   lcl_N2N = rmt_N2N = the_N2N = 0; 
   N2N_Lib = N2N_Parms         = 0;
   StageQ.pendList.setItem(0);
   StageQ.fullList.setItem(0);
   StageCmd      = 0;
   StageMsg      = 0; 
   StageSnd      = 0;
   StageRealTime = 1;
   StageAsync    = 0;
   StageCreate   = 0;
   StageEvents   = (char *)"-";
   StageEvSize   = 1;
   StageAction   = (char *)"wq "; 
   StageActLen   = 3;
   MSSgwCmd      = 0;
   DirFlags      = 0; 
   OptFlags      = 0;
   LocalRoot     = 0;
   RemoteRoot    = 0;
   cscanint      = XrdOssCSCANINT;
   FDFence       = -1;
   FDLimit       = XrdOssFDLIMIT;
   MaxDBsize     = XrdOssMAXDBSIZE;
   minalloc      = XrdOssMINALLOC;
   ovhalloc      = XrdOssOVRALLOC;
   fuzalloc      = XrdOssFUZALLOC;
   xfrspeed      = XrdOssXFRSPEED;
   xfrovhd       = XrdOssXFROVHD;
   xfrhold       = XrdOssXFRHOLD;
   xfrkeep       = 20*60;
   xfrthreads    = XrdOssXFRTHREADS;
   ConfigFN      = 0;
   DeprLine      = 0;
}
  
/******************************************************************************/
/*                             C o n f i g u r e                              */
/******************************************************************************/
  
int XrdOssSys::Configure(const char *configfn, XrdSysError &Eroute)
{
/*
  Function: Establish default values using a configuration file.

  Input:    None.

  Output:   0 upon success or !0 otherwise.
*/
   XrdSysError_Table *ETab = new XrdSysError_Table(XRDOSS_EBASE, XRDOSS_ELAST,
                                                   XrdOssErrorText);
   char *val;
   int  retc, NoGo = XrdOssOK;
   pthread_t tid;

// Do the herald thing
//
   Eroute.Say("++++++ Storage system initialization started.");
   Eroute.addTable(ETab);
   if (getenv("XRDDEBUG")) OssTrace.What = TRACE_ALL;

// Preset all variables with common defaults
//
   ConfigFN = (configfn && *configfn ? strdup(configfn) : 0);

// Process the configuration file
//
   NoGo = ConfigProc(Eroute);

// Establish the FD limit
//
   {struct rlimit rlim;
    if (getrlimit(RLIMIT_NOFILE, &rlim) < 0)
       Eroute.Emsg("Config", errno, "get resource limits");
       else Hard_FD_Limit = rlim.rlim_max;

    if (FDLimit <= 0) FDLimit = rlim.rlim_cur;
       else {rlim.rlim_cur = FDLimit;
            if (setrlimit(RLIMIT_NOFILE, &rlim) < 0)
               NoGo = Eroute.Emsg("Config", errno,"set FD limit");
            }
    if (FDFence < 0 || FDFence >= FDLimit) FDFence = FDLimit >> 1;
   }

// Establish cached filesystems
//
   ReCache();

// Configure the MSS interface including staging
//
   if (!NoGo) NoGo = ConfigStage(Eroute);

// Configure async I/O
//
   if (!NoGo) NoGo = !AioInit();

// Initialize memory mapping setting to speed execution
//
   if (!NoGo) ConfigMio(Eroute);

// Establish the actual default path settings (modified by the above)
//
   RPList.Set(DirFlags);

// Start up the cache scan thread
//
   if ((retc = XrdSysThread::Run(&tid, XrdOssCacheScan, (void *)0,
                                 0, "cache scan")))
      Eroute.Emsg("Config", retc, "create cache scan thread");

// Display the final config if we can continue
//
   if (!NoGo) Config_Display(Eroute);

// All done, close the stream and return the return code.
//
   val = (NoGo ? (char *)"failed." : (char *)"completed.");
   Eroute.Say("------ Storage system initialization ", val);
   return NoGo;
}
  
/******************************************************************************/
/*                   o o s s _ C o n f i g _ D i s p l a y                    */
/******************************************************************************/
  
#define XrdOssConfig_Val(base, opt) \
             (Have ## base  ? "       oss." #opt " " : ""), \
             (Have ## base  ? base     : ""), \
             (Have ## base  ? "\n"     : "")
  
#define XrdOssConfig_Vop(base, opt, optchk0, opt1, opt2, optchk1, opt3, opt4) \
             (Have ## base  ? "       oss." #opt " " : ""), \
             (Have ## base  ? (optchk0 ? opt1 : opt2) : ""), \
             (Have ## base  ? (optchk1 ? opt3 : opt4) : ""), \
             (Have ## base  ? base     : ""), \
             (Have ## base  ? "\n"     : "")

void XrdOssSys::Config_Display(XrdSysError &Eroute)
{
     char buff[4096], *cloc;
     XrdOucPList *fp;

     // Preset some tests
     //
     int HaveMSSgwCmd   = (MSSgwCmd   && MSSgwCmd[0]);
     int HaveStageCmd   = (StageCmd   && StageCmd[0]);
     int HaveRemoteRoot = (RemoteRoot && RemoteRoot[0]);
     int HaveLocalRoot  = (LocalRoot  && LocalRoot[0]);
     int HaveStageMsg   = (StageMsg   && StageMsg[0]);
     int HaveN2N_Lib    = (N2N_Lib != 0);

     if (!ConfigFN || !ConfigFN[0]) cloc = (char *)"Default";
        else cloc = ConfigFN;

     snprintf(buff, sizeof(buff), "Config effective %s oss configuration:\n"
                                  "       oss.alloc        %lld %d %d\n"
                                  "       oss.cachescan    %d\n"
                                  "       oss.compdetect   %s\n"
                                  "       oss.fdlimit      %d %d\n"
                                  "       oss.maxdbsize    %lld\n"
                                  "%s%s%s"
                                  "%s%s%s"
                                  "%s%s%s"
                                  "%s%s%s%s%s"
                                  "%s%s%s"
                                  "%s%s%s"
                                  "       oss.trace        %x\n"
                                  "       oss.xfr          %d %d %d %d",
             cloc,
             minalloc, ovhalloc, fuzalloc,
             cscanint,
             (CompSuffix ? CompSuffix : "*"),
             FDFence, FDLimit, MaxDBsize,
             XrdOssConfig_Val(N2N_Lib,    namelib),
             XrdOssConfig_Val(LocalRoot,  localroot),
             XrdOssConfig_Val(RemoteRoot, remoteroot),
             XrdOssConfig_Vop(StageCmd,   stagecmd, StageAsync,  "async ","sync ",
                                                    StageCreate, "creates ", ""),
             XrdOssConfig_Val(StageMsg,   stagemsg),
             XrdOssConfig_Val(MSSgwCmd,   mssgwcmd),
             OssTrace.What,
             xfrthreads, xfrspeed, xfrovhd, xfrhold);

     Eroute.Say(buff);

     XrdOssMio::Display(Eroute);

     List_Cache(  (char *)"       oss.cache ", 0, Eroute);
     if (!(OptFlags & XrdOss_ROOTDIR)) 
           List_Path("       oss.defaults ", (char *)"", DirFlags, Eroute);
     fp = RPList.First();
     while(fp)
          {List_Path("       oss.path ", fp->Path(), fp->Flag(), Eroute);
           fp = fp->Next();
          }
}

/******************************************************************************/
/*                     P r i v a t e   F u n c t i o n s                      */
/******************************************************************************/
/******************************************************************************/
/*                             C o n f i g M i o                              */
/******************************************************************************/
  
void XrdOssSys::ConfigMio(XrdSysError &Eroute)
{
     XrdOucPList *fp;
     unsigned long long flags = 0;
     int setoff = 0;

// Initialize memory mapping setting to speed execution
//
   if (!(tryMmap = XrdOssMio::isOn())) return;
   chkMmap = XrdOssMio::isAuto();

// Run through all the paths and get the composite flags
//
   fp = RPList.First();
   while(fp)
        {flags |= fp->Flag();
         fp = fp->Next();
        }

// Handle default settings
//
   if (DirFlags & XRDEXP_MEMAP && !(DirFlags & XRDEXP_NOTRW))
      DirFlags |= XRDEXP_FORCERO;
   if (!(OptFlags & XrdOss_ROOTDIR)) flags |= DirFlags;
   if (DirFlags & (XRDEXP_MLOK | XRDEXP_MKEEP)) DirFlags |= XRDEXP_MMAP;

// Produce warnings if unsupported features have been selected
//
#if !defined(_POSIX_MAPPED_FILES)
   if (flags & XRDEXP_MEMAP)
      {Eroute.Say("Config warning: memory mapped files not supported; "
                             "feature disabled.");
       setoff = 1;
       fp = RPList.First();
       while(fp)
            {fp->Set(fp->Flag() & ~XRDEXP_MEMAP);
             fp = fp->Next();
            }
       DirFlags = DirFlags & ~XRDEXP_MEMAP;
      }
#elif !defined(_POSIX_MEMLOCK)
   if (flags & XRDEXP_MLOK)
      {Eroute.Say("Config warning: memory locked files not supported; "
                             "feature disabled.");
       fp = RPList.First();
       while(fp)
            {fp->Set(fp->Flag() & ~XRDEXP_MLOK);
             fp = fp->Next();
            }
       DirFlags = DirFlags & ~XRDEXP_MLOK;
      }
#endif

// If no memory flags are set, turn off memory mapped files
//
   if (!(flags & XRDEXP_MEMAP) || setoff)
     {XrdOssMio::Set(0, 0, 0, 0, 0);
      tryMmap = 0; chkMmap = 0;
     }
}
  
/******************************************************************************/
/*                             C o n f i g N 2 N                              */
/******************************************************************************/

int XrdOssSys::ConfigN2N(XrdSysError &Eroute)
{
   XrdSysPlugin    *myLib;
   XrdOucName2Name *(*ep)(XrdOucgetName2NameArgs);

// If we have no library path then use the default method (this will always
// succeed).
//
   if (!N2N_Lib)
      {the_N2N = XrdOucgetName2Name(&Eroute, ConfigFN, "", LocalRoot, RemoteRoot);
       if (LocalRoot)  lcl_N2N = the_N2N;
       if (RemoteRoot) rmt_N2N = the_N2N;
       return 0;
      }

// Create a pluin object (we will throw this away without deletion because
// the library must stay open but we never want to reference it again).
//
   if (!(myLib = new XrdSysPlugin(&Eroute, N2N_Lib))) return 1;

// Now get the entry point of the object creator
//
   ep = (XrdOucName2Name *(*)(XrdOucgetName2NameArgs))(myLib->getPlugin("XrdOucgetName2Name"));
   if (!ep) return 1;


// Get the Object now
//
   lcl_N2N = rmt_N2N = the_N2N = ep(&Eroute, ConfigFN, 
                                   (N2N_Parms ? N2N_Parms : ""),
                                   LocalRoot, RemoteRoot);
   return lcl_N2N == 0;
}
  
/******************************************************************************/
/*                            C o n f i g P r o c                             */
/******************************************************************************/
  
int XrdOssSys::ConfigProc(XrdSysError &Eroute)
{
  char *var;
  int  cfgFD, retc, NoGo = XrdOssOK;
  XrdOucEnv myEnv;
  XrdOucStream Config(&Eroute, getenv("XRDINSTANCE"), &myEnv, "=====> ");

// If there is no config file, return with the defaults sets.
//
   if( !ConfigFN || !*ConfigFN)
     {Eroute.Say("Config warning: config file not specified; defaults assumed.");
      return XrdOssOK;
     }

// Try to open the configuration file.
//
   if ( (cfgFD = open(ConfigFN, O_RDONLY, 0)) < 0)
      {Eroute.Emsg("Config", errno, "open config file", ConfigFN);
       return 1;
      }
   Config.Attach(cfgFD);

// Now start reading records until eof.
//
   while((var = Config.GetMyFirstWord()))
        {if (!strncmp(var, "oss.", 4))
            {if (ConfigXeq(var+4, Config, Eroute)) {Config.Echo(); NoGo = 1;}}
            else if (!strcmp(var,"all.export"))
                    {OptFlags |= XrdOss_EXPORT;
                     if (xpath(Config, Eroute)) {Config.Echo(); NoGo = 1;}
                    }
        }

// All done scanning the file, set dependent parameters.
//
   if (N2N_Lib || LocalRoot || RemoteRoot) NoGo |= ConfigN2N(Eroute);

// Now check if any errors occured during file i/o
//
   if ((retc = Config.LastError()))
      NoGo = Eroute.Emsg("Config", retc, "read config file", ConfigFN);
   Config.Close();

// Check if we have any conflicts using new and old options
//
   if ((OptFlags & XrdOss_EXPORT) && DeprLine)
      {Eroute.Emsg("Config", "'all.export' conflicts with deprecated",DeprLine);
       Eroute.Emsg("Config", "'oss.defaults' must be used instead!");
       NoGo = 1;
      }

// Return final return code
//
   return NoGo;
}

/******************************************************************************/
/*                           C o n f i g S t a g e                            */
/******************************************************************************/

int XrdOssSys::ConfigStage(XrdSysError &Eroute)
{
   char *tp, *gwp = 0, *stgp = 0;
   unsigned long long dflags, flags;
   int retc, numt, NoGo = 0;
   pthread_t tid;
   XrdOucPList *fp;

// A mssgwcmd implies mig and a stagecmd implies stage as defaults
//
   dflags = (MSSgwCmd ? XRDEXP_MIG : XRDEXP_NOCHECK|XRDEXP_NODREAD);
   if (!StageCmd) dflags |= XRDEXP_NOSTAGE;
   DirFlags = DirFlags | (dflags & (~(DirFlags >> XRDEXP_MASKSHIFT)));
   if (MSSgwCmd && (DirFlags & XRDEXP_MIG)) DirFlags |= XRDEXP_REMOTE;
   RPList.Default(DirFlags);

// Reprocess the paths to set correct defaults
//
   fp = RPList.First();
   while(fp) 
        {flags = fp->Flag();
         flags = flags | (DirFlags & (~(flags >> XRDEXP_MASKSHIFT)));
         if (!(flags & XRDEXP_NOSTAGE)) gwp = stgp = fp->Path();
            else if (!(flags & XRDEXP_NOCHECK) || !(flags & XRDEXP_NODREAD) ||
                    (flags & XRDEXP_RCREATE))  gwp = fp->Path();
         if (MSSgwCmd && (flags & XRDEXP_MIG)) flags |= XRDEXP_REMOTE;
         fp->Set(flags);
         fp = fp->Next();
        }

// Include the defaults if a root directory was not specified
//
   if (!(OptFlags & XrdOss_ROOTDIR))
      {if (!(DirFlags & XRDEXP_NOSTAGE)) gwp = stgp = (char *)"/";
          else if (!(DirFlags & XRDEXP_NOCHECK) || !(DirFlags & XRDEXP_NODREAD) ||
                  (DirFlags & XRDEXP_RCREATE))  gwp = (char *)"/";
      }

// Check if we need or don't need the stagecmd
//
   if (stgp && !StageCmd)
      {Eroute.Emsg("Config","Stageable path", stgp,
                            "present but stagecmd not specified.");
       NoGo = 1;
      }
      else if (StageCmd && !stgp)
              {Eroute.Say("Config warning: 'stagecmd' ignored; no stageable paths present.");
               free(StageCmd); StageCmd = 0;
              }

// Check if we need or don't need the gateway
//
   if (gwp && !MSSgwCmd)
      {Eroute.Emsg("Config","MSS path", gwp,
                            "present but mssgwcmd not specified.");
       NoGo = 1;
      }
      else if (MSSgwCmd && !gwp)
              {Eroute.Say("Config warning: 'msscmd' ignored; no path has "
                           "check, dread, rcreate, or stage attributes.");
               free(MSSgwCmd); MSSgwCmd = 0;
              }

// If we have any errors at this point, just return failure
//
   if (NoGo) return 1;
   if (!MSSgwCmd && !StageCmd) return 0;
   Eroute.Say("++++++ Mass Storage System interface initialization started.");

// Allocate a prgram object for the gateway command
//
   if (MSSgwCmd)
      {MSSgwProg = new XrdOucProg(&Eroute);
       if (MSSgwProg->Setup(MSSgwCmd)) NoGo = 1;
      }

// Initialize staging if we need to
//
   if (!NoGo && StageCmd)
      {
       // The stage command is interactive if it starts with an | (i.e., pipe in)
       //
          tp = StageCmd;
          while(*tp && *tp == ' ') tp++;
          if (*tp == '|') {StageRealTime = 0; StageCmd = tp+1;}

      // Set up a program object for the command
      //
         StageProg = new XrdOucProg(&Eroute);
         if (StageProg->Setup(StageCmd)) NoGo = 1;

      // For old-style real-time staging, create threads to handle the staging
      // For queue-style staging, start the program that handles the queue
      //
         if (!NoGo)
            if (StageRealTime)
               {if ((numt = xfrthreads - xfrtcount) > 0) while(numt--)
                    if ((retc = XrdSysThread::Run(&tid,XrdOssxfr,(void *)0,0,"staging")))
                       Eroute.Emsg("Config", retc, "create staging thread");
                       else xfrtcount++;
               } else NoGo = StageProg->Start();

      // Set up the event path
      //
         StageAction = (char *)"wfn "; StageActLen = 4;
         if ((tp = getenv("XRDOFSEVENTS")))
            {char sebuff[1024];
             StageEvSize = sprintf(sebuff, "file:///%s", tp);
             StageEvents = strdup(sebuff);
            } else {StageEvents = (char *)"-"; StageEvSize = 1;}
     }

// Setup the additional stage information vector. Variable substitution:
// <data>$var;<data>.... (max of MaxArgs substitutions)
//
   if (!NoGo && !StageRealTime && StageMsg)
      {XrdOucMsubs *msubs = new XrdOucMsubs(&Eroute);
       if (msubs->Parse("stagemsg", StageMsg)) StageSnd = msubs;
          else NoGo = 1;  // We will exit no need to delete msubs
      }

// All done
//
   tp = (NoGo ? (char *)"failed." : (char *)"completed.");
   Eroute.Say("------ Mass Storage System interface initialization ", tp);
   return NoGo;
}
  
/******************************************************************************/
/*                             C o n f i g X e q                              */
/******************************************************************************/

int XrdOssSys::ConfigXeq(char *var, XrdOucStream &Config, XrdSysError &Eroute)
{
    char  myVar[64], buff[2048], *val;
    int nosubs;
    XrdOucEnv *myEnv = 0;

   // Check for deprecated options
   //
   if (!chkDep(var))
      {strcpy(buff, "oss."); strcat(buff, var);
       Eroute.Say("Config warning: '", buff,
                  "' is deprecated; use 'oss.defaults' instead!");
       Config.Echo();
       if (DeprLine)
          {strcpy(buff, DeprLine); strcat(buff," oss."); strcat(buff, var);
           free(DeprLine);
          }
       DeprLine = strdup(buff);
       return 0;
      }

   TS_Ade("userprty",      OptFlags, XrdOss_USRPRTY, 0);

   TS_Xeq("alloc",         xalloc);
   TS_Xeq("cache",         xcache);
   TS_Xeq("cachescan",     xcachescan);
   TS_Xeq("compdetect",    xcompdct);
   TS_Xeq("defaults",      xdefault);
   TS_Xeq("fdlimit",       xfdlimit);
   TS_Xeq("maxsize",       xmaxdbsz);
   TS_Xeq("memfile",       xmemf);
   TS_Xeq("namelib",       xnml);
   TS_Xeq("path",          xpath);
   TS_Xeq("stagecmd",      xstg);
   TS_Xeq("trace",         xtrace);
   TS_Xeq("xfr",           xxfr);

   // Accepts options that used to be valid but otherwise ignore them
   //
   if (!strcmp("mssgwpath", var)) return 0;
   if (!strcmp("gwbacklog", var)) return 0;

   // Check if var substitutions are prohibited (e.g., stagemsg). Note that
   // TS_String() returns upon success so be careful when adding new opts.
   //
   if ((nosubs = !strcmp(var, "stagemsg"))) myEnv = Config.SetEnv(0);

   // Copy the variable name as this may change because it points to an
   // internal buffer in Config. The vagaries of effeciency.
   //
   strlcpy(myVar, var, sizeof(myVar));
   var = myVar;

   // We need to suck all the tokens to the end of the line for remaining
   // options. Do so, until we run out of space in the buffer.
   //
   if (!Config.GetRest(buff, sizeof(buff)))
      {Eroute.Emsg("Config", "arguments too long for", var);
       if (nosubs) Config.SetEnv(myEnv);
       return 1;
      }
   val = buff;

   // Restore substititions at this point if need be
   //
   if (nosubs) Config.SetEnv(myEnv);

   // At this point, make sure we have a value
   //
   if (!(*val))
      {Eroute.Emsg("Config", "no value for directive", var);
       return 1;
      }

   // Check for tokens taking a variable number of parameters
   //
   TS_String("localroot",  LocalRoot);
   TS_String("remoteroot", RemoteRoot);
   TS_String("stagemsg",   StageMsg);
   TS_String("mssgwcmd",   MSSgwCmd);  // Deprecated
   TS_String("msscmd",     MSSgwCmd);

   // No match found, complain.
   //
   Eroute.Say("Config warning: ignoring unknown directive '",var,"'.");
   Config.Echo();
   return 0;
}
  
/******************************************************************************/
/*                                c h k D e p                                 */
/******************************************************************************/

int XrdOssSys::chkDep(const char *var)
{
   // Process items that don't need a vlaue
   //
   TS_Add("compchk",       DirFlags, XRDEXP_COMPCHK, 0);
   TS_Add("forcero",       DirFlags, XRDEXP_FORCERO, XRDEXP_ROW_X);
   TS_Add("readonly",      DirFlags, XRDEXP_READONLY,XRDEXP_ROW_X);
   TS_Add("notwritable",   DirFlags, XRDEXP_READONLY,XRDEXP_ROW_X);
   TS_Rem("writable",      DirFlags, XRDEXP_NOTRW,   XRDEXP_ROW_X);

   TS_Add("mig",           DirFlags, XRDEXP_MIG,     XRDEXP_MIG_X);
   TS_Rem("nomig",         DirFlags, XRDEXP_MIG,     XRDEXP_MIG_X);
   TS_Add("migratable",    DirFlags, XRDEXP_MIG,     XRDEXP_MIG_X);
   TS_Rem("notmigratable", DirFlags, XRDEXP_MIG,     XRDEXP_MIG_X);

   TS_Add("mkeep",         DirFlags, XRDEXP_MKEEP,   XRDEXP_MKEEP_X);
   TS_Rem("nomkeep",       DirFlags, XRDEXP_MKEEP,   XRDEXP_MKEEP_X);

   TS_Add("mlock",         DirFlags, XRDEXP_MLOK,    XRDEXP_MLOK_X);
   TS_Rem("nomlock",       DirFlags, XRDEXP_MLOK,    XRDEXP_MLOK_X);

   TS_Add("mmap",          DirFlags, XRDEXP_MMAP,    XRDEXP_MMAP_X);
   TS_Rem("nommap",        DirFlags, XRDEXP_MMAP,    XRDEXP_MMAP_X);

   TS_Rem("check",         DirFlags, XRDEXP_NOCHECK, XRDEXP_CHECK_X);
   TS_Add("nocheck",       DirFlags, XRDEXP_NOCHECK, XRDEXP_CHECK_X);

   TS_Rem("dread",         DirFlags, XRDEXP_NODREAD, XRDEXP_DREAD_X);
   TS_Add("nodread",       DirFlags, XRDEXP_NODREAD, XRDEXP_DREAD_X);

   TS_Rem("ssdec",         DirFlags, XRDEXP_NOSSDEC, 0);
   TS_Add("nossdec",       DirFlags, XRDEXP_NOSSDEC, 0);

   TS_Rem("stage",         DirFlags, XRDEXP_NOSTAGE, XRDEXP_STAGE_X);
   TS_Add("nostage",       DirFlags, XRDEXP_NOSTAGE, XRDEXP_STAGE_X);

   TS_Add("rcreate",       DirFlags, XRDEXP_RCREATE, XRDEXP_RCREATE_X);
   TS_Rem("norcreate",     DirFlags, XRDEXP_RCREATE, XRDEXP_RCREATE_X);

   return 1;
}

/******************************************************************************/
/*                                x a l l o c                                 */
/******************************************************************************/

/* Function: aalloc

   Purpose:  To parse the directive: alloc <min> [<headroom> [<fuzz>]]

             <min>       minimum amount of free space needed in a partition.
                         (asterisk uses default).
             <headroom>  percentage of requested space to be added to the
                         free space amount (asterisk uses default).
             <fuzz>      the percentage difference between two free space
                         quantities that may be ignored when selecting a cache
                           0 - reduces to finding the largest free space
                         100 - reduces to simple round-robin allocation

   Output: 0 upon success or !0 upon failure.
*/

int XrdOssSys::xalloc(XrdOucStream &Config, XrdSysError &Eroute)
{
    char *val;
    long long mina = XrdOssMINALLOC;
    int       fuzz = XrdOssFUZALLOC;
    int       hdrm = XrdOssOVRALLOC;

    if (!(val = Config.GetWord()))
       {Eroute.Emsg("Config", "alloc minfree not specified"); return 1;}
    if (strcmp(val, "*") &&
        XrdOuca2x::a2sz(Eroute, "alloc minfree", val, &mina, 0)) return 1;

    if ((val = Config.GetWord()))
       {if (strcmp(val, "*") &&
            XrdOuca2x::a2i(Eroute,"alloc headroom",val,&hdrm,0,100)) return 1;

        if ((val = Config.GetWord()))
           {if (strcmp(val, "*") &&
            XrdOuca2x::a2i(Eroute, "alloc fuzz", val, &fuzz, 0, 100)) return 1;
           }
       }

    minalloc = mina;
    ovhalloc = hdrm;
    fuzalloc = fuzz;
    return 0;
}

/******************************************************************************/
/*                                x c a c h e                                 */
/******************************************************************************/

/* Function: xcache

   Purpose:  To parse the directive: cache <group> <path>

             <group>  logical group name for the cache filesystem.
             <path>   path to the cache.

   Output: 0 upon success or !0 upon failure.
*/

int XrdOssSys::xcache(XrdOucStream &Config, XrdSysError &Eroute)
{
    char *val, *pfxdir, *sfxdir, grp[17], fn[XrdOssMAX_PATH_LEN+1];
    int i, k, rc, pfxln, cnum = 0;
    struct dirent *dp;
    struct stat buff;
    DIR *DFD;

    if (!(val = Config.GetWord()))
       {Eroute.Emsg("Config", "cache group not specified"); return 1;}
    if (strlen(val) >= sizeof(grp))
       {Eroute.Emsg("Config", "invalid cache group - ", val); return 1;}
    strcpy(grp, val);

    if (!(val = Config.GetWord()))
       {Eroute.Emsg("Config", "cache path not specified"); return 1;}

    k = strlen(val);
    if (k >= (int)(sizeof(fn)-1) || val[0] != '/' || k < 2)
       {Eroute.Emsg("Config", "invalid cache path - ", val); return 1;}

    if (val[k-1] != '*')
       {for (i = k-1; i; i--) if (val[i] != '/') break;
        fn[i+1] = '/'; fn[i+2] = '\0';
        while (i >= 0) {fn[i] = val[i]; i--;}
        return !xcacheBuild(grp, fn, Eroute);
       }

    for (i = k-1; i; i--) if (val[i] == '/') break;
    i++; strncpy(fn, val, i); fn[i] = '\0';
    sfxdir = &fn[i]; pfxdir = &val[i]; pfxln = strlen(pfxdir)-1;
    if (!(DFD = opendir(fn)))
       {Eroute.Emsg("Config", errno, "open cache directory", fn); return 1;}

    errno = 0;
    while((dp = readdir(DFD)))
         {if (!strcmp(dp->d_name, ".") || !strcmp(dp->d_name, "..")
          || (pfxln && strncmp(dp->d_name, pfxdir, pfxln)))
             continue;
          strcpy(sfxdir, dp->d_name);
          if (stat(fn, &buff)) break;
          if (buff.st_mode & S_IFDIR)
             {val = sfxdir + strlen(sfxdir) - 1;
             if (*val++ != '/') {*val++ = '/'; *val = '\0';}
             if (xcacheBuild(grp, fn, Eroute)) cnum++;
                else {closedir(DFD); return 1;}
             }
          errno = 0;
         }

    if ((rc = errno))
       Eroute.Emsg("Config", errno, "process cache directory", fn);
       else if (!cnum) Eroute.Say("Config warning: no cache directories found in ",val);

    closedir(DFD);
    return rc != 0;
}

int XrdOssSys::xcacheBuild(char *grp, char *fn, XrdSysError &Eroute)
{
    XrdOssCache_FS *fsp;
    int rc;
    if (!(fsp = new XrdOssCache_FS(rc, grp, fn)))
       {Eroute.Emsg("Config", ENOMEM, "create cache", fn); return 0;}
    if (rc)
       {Eroute.Emsg("Config", rc, "create cache", fn);
        delete fsp;
        return 0;
       }
    return 1;
}

/******************************************************************************/
/*                              x c o m p d c t                               */
/******************************************************************************/

/* Function: xcompdct

   Purpose:  To parse the directive: compdetect { * | <sfx>}

             *        perform autodetect for compression
             <sfx>    path suffix to indicate that file is compressed

   Output: 0 upon success or !0 upon failure.
*/

int XrdOssSys::xcompdct(XrdOucStream &Config, XrdSysError &Eroute)
{
    char *val;

    if (!(val = Config.GetWord()))
       {Eroute.Emsg("Config", "compdetect suffix not specified"); return 1;}

    if (CompSuffix) free(CompSuffix);
    CompSuffix = 0; CompSuflen = 0;

    if (!strcmp("*", val))
       {CompSuffix = strdup(val); CompSuflen = strlen(val);}

    return 0;
}

/******************************************************************************/
/*                            x c a c h e s c a n                             */
/******************************************************************************/

/* Function: xcachescan

   Purpose:  To parse the directive: cachescan <num>

             <num>     number of seconds between cache scans.

   Output: 0 upon success or !0 upon failure.
*/
int XrdOssSys::xcachescan(XrdOucStream &Config, XrdSysError &Eroute)
{   int cscan = 0;
    char *val;

    if (!(val = Config.GetWord()))
       {Eroute.Emsg("Config", "cachescan not specified"); return 1;}
    if (XrdOuca2x::a2tm(Eroute, "cachescan", val, &cscan, 30)) return 1;
    cscanint = cscan;
    return 0;
}

/******************************************************************************/
/*                              x d e f a u l t                               */
/******************************************************************************/

/* Function: xdefault

   Purpose:  Parse: defaults <default options>
                              
   Notes: See the oss configuration manual for the meaning of each option.
          The actual implementation is defined in XrdOucExport.

   Output: 0 upon success or !0 upon failure.
*/

int XrdOssSys::xdefault(XrdOucStream &Config, XrdSysError &Eroute)
{
   DirFlags = XrdOucExport::ParseDefs(Config, Eroute, DirFlags);
   return 0;
}
  
/******************************************************************************/
/*                              x f d l i m i t                               */
/******************************************************************************/

/* Function: xfdlimit

   Purpose:  To parse the directive: fdlimit <fence> [ <max> ]

             <fence>  lowest number to use for file fd's (0 -> max). If
                      specified as * then max/2 is used.
             <max>    highest number that can be used. The soft rlimit is set
                      to this value. If not supplied, the limit is not changed.
                      If supplied as 'max' then the hard limit is used.

   Output: 0 upon success or !0 upon failure.
*/

int XrdOssSys::xfdlimit(XrdOucStream &Config, XrdSysError &Eroute)
{
    char *val;
    int fence = 0, fdmax = XrdOssFDLIMIT;

      if (!(val = Config.GetWord()))
         {Eroute.Emsg("Config", "fdlimit fence not specified"); return 1;}

      if (!strcmp(val, "*")) fence = -1;
         else if (XrdOuca2x::a2i(Eroute,"fdlimit fence",val,&fence,0)) return 1;

      if (!(val = Config.GetWord())) fdmax = -1;
         else if (!strcmp(val, "max")) fdmax = Hard_FD_Limit;
                 else if (XrdOuca2x::a2i(Eroute, "fdlimit value", val, &fdmax,
                              xrdmax(fence,XrdOssFDMINLIM))) return -EINVAL;
                         else if (fdmax > Hard_FD_Limit)
                                 {fdmax = Hard_FD_Limit;
                                  Eroute.Say("Config warning: ",
                                              "'fdlimit' forced to hard max");
                                 }
      FDFence = fence;
      FDLimit = fdmax;
      return 0;
}
  
/******************************************************************************/
/*                              x m a x d b s z                               */
/******************************************************************************/

/* Function: xmaxdbsz

   Purpose:  Parse the directive:  maxdbsize <num>

             <num> Maximum number of bytes in a database file.

   Output: 0 upon success or !0 upon failure.
*/

int XrdOssSys::xmaxdbsz(XrdOucStream &Config, XrdSysError &Eroute)
{   long long mdbsz;
    char *val;

    if (!(val = Config.GetWord()))
       {Eroute.Emsg("Config", "maxdbsize value not specified"); return 1;}
    if (XrdOuca2x::a2sz(Eroute, "maxdbsize", val, &mdbsz, 1024*1024)) return 1;
    MaxDBsize = mdbsz;
    return 0;
}

/******************************************************************************/
/*                                 x m e m f                                  */
/******************************************************************************/
  
/* Function: xmemf

   Purpose:  Parse the directive: memfile [off] [max <msz>]
                                          [check {keep | lock | map}] [preload]

             check keep Maps files that have ".mkeep" shadow file, premanently.
             check lock Maps and locks files that have ".mlock" shadow file.
             check map  Maps files that have ".mmap" shadow file.
             all        Preloads the complete file into memory.
             off        Disables memory mapping regardless of other options.
             on         Enables memory mapping
             preload    Preloads the file after every opn reference.
             <msz>      Maximum amount of memory to use (can be n% or real mem).

   Output: 0 upon success or !0 upon failure.
*/

int XrdOssSys::xmemf(XrdOucStream &Config, XrdSysError &Eroute)
{
    char *val;
    int i, j, V_autolok=-1, V_automap=-1, V_autokeep=-1, V_preld = -1, V_on=-1;
    long long V_max = 0;

    static struct mmapopts {const char *opname; int otyp;
                            const char *opmsg;} mmopts[] =
       {
        {"off",        0, ""},
        {"preload",    1, "memfile preload"},
        {"check",      2, "memfile check"},
        {"max",        3, "memfile max"}};
    int numopts = sizeof(mmopts)/sizeof(struct mmapopts);

    if (!(val = Config.GetWord()))
       {Eroute.Emsg("Config", "memfile option not specified"); return 1;}

    while (val)
         {for (i = 0; i < numopts; i++)
              if (!strcmp(val, mmopts[i].opname)) break;
          if (i >= numopts)
             Eroute.Say("Config warning: ignoring invalid memfile option '",val,"'.");
             else {if (mmopts[i].otyp >  1 && !(val = Config.GetWord()))
                      {Eroute.Emsg("Config","memfile",mmopts[i].opname,
                                   "value not specified");
                       return 1;
                      }
                   switch(mmopts[i].otyp)
                         {case 1: V_preld = 1;
                                  break;
                          case 2:     if (!strcmp("lock", val)) V_autolok=1;
                                 else if (!strcmp("map",  val)) V_automap=1;
                                 else if (!strcmp("keep", val)) V_autokeep=1;
                                 else {Eroute.Emsg("Config",
                                       "mmap auto neither keep, lock, nor map");
                                       return 1;
                                      }
                                  break;
                          case 3: j = strlen(val);
                                  if (val[j-1] == '%')
                                     {val[j-1] = '\0';
                                      if (XrdOuca2x::a2i(Eroute,mmopts[i].opmsg,
                                                     val, &j, 1, 1000)) return 1;
                                      V_max = -j;
                                     } else if (XrdOuca2x::a2sz(Eroute,
                                                mmopts[i].opmsg, val, &V_max,
                                                10*1024*1024)) return 1;
                                  break;
                          default: V_on = 0; break;
                         }
                  val = Config.GetWord();
                 }
         }

// Set the values
//
   XrdOssMio::Set(V_on, V_preld, V_autolok, V_automap, V_autokeep);
   XrdOssMio::Set(V_max);
   return 0;
}

/******************************************************************************/
/*                                  x n m l                                   */
/******************************************************************************/

/* Function: xnml

   Purpose:  To parse the directive: namelib <path> [<parms>]

             <path>    the path of the filesystem library to be used.
             <parms>   optional parms to be passed

  Output: 0 upon success or !0 upon failure.
*/

int XrdOssSys::xnml(XrdOucStream &Config, XrdSysError &Eroute)
{
    char *val, parms[1024];

// Get the path
//
   if (!(val = Config.GetWord()) || !val[0])
      {Eroute.Emsg("Config", "namelib not specified"); return 1;}

// Record the path
//
   if (N2N_Lib) free(N2N_Lib);
   N2N_Lib = strdup(val);

// Record any parms
//
   if (!Config.GetRest(parms, sizeof(parms)))
      {Eroute.Emsg("Config", "namelib parameters too long"); return 1;}
   if (N2N_Parms) free(N2N_Parms);
   N2N_Parms = (*parms ? strdup(parms) : 0);
   return 0;
}

/******************************************************************************/
/*                                 x p a t h                                  */
/******************************************************************************/

/* Function: xpath

   Purpose:  To parse the directive: {export | path} <path> [<options>]

             <path>    the full path that resides in a remote system.
             <options> a blank separated list of options (see XrdOucExport)

   Output: 0 upon success or !0 upon failure.
*/

int XrdOssSys::xpath(XrdOucStream &Config, XrdSysError &Eroute)
{
    XrdOucPList *plp, *olp;
    unsigned long long Opts;

// Parse the arguments
//
   if (!(plp = XrdOucExport::ParsePath(Config, Eroute, DirFlags))) return 1;

// Check if this path is being modified or added. For modifications, turn off
// all bitsin the old path specified in the new path and then set the new bits.
//
   if (!(olp = RPList.Match(plp->Path()))) 
      {RPList.Insert(plp);
       if (!strcmp(plp->Path(), "/")) OptFlags |= XrdOss_ROOTDIR;
      }
      else {Opts = plp->Flag() >> XRDEXP_MASKSHIFT;
            Opts = olp->Flag() & ~Opts;
            olp->Set(Opts | plp->Flag());
            delete plp;
           }
   return 0;
}

/******************************************************************************/
/*                                  x s t g                                   */
/******************************************************************************/

/* Function: xstg

   Purpose:  To parse the directive: 
                stagecmd [async | sync] [creates] [|]<cmd>]

             async     Client is to be notified when <cmd> sends an event
             sync      Client is to poll for <cmd> completion.
             creates   Route file creation requests to <cmd>.
             <cmd>     The command and args to stage in the file. If the
                       <cmd> is prefixed ny '|' then pipe in the requests.

  Output: 0 upon success or !0 upon failure.
*/

int XrdOssSys::xstg(XrdOucStream &Config, XrdSysError &Eroute)
{
    char *val, buff[2048], *bp = buff;
    int vlen, blen = sizeof(buff)-1, isAsync = 0, isCreate = 0;

// Get the aync or async option
//
    if ((val = Config.GetWord()))
       if ((isAsync = !strcmp(val, "async")) || !strcmp(val, "sync"))
          val = Config.GetWord();

// Get the create option
//
   if (val)
       if ((isCreate = !strcmp(val, "creates"))) val = Config.GetWord();

// Get the command
//
   if (!val) {Eroute.Emsg("Config", "stagecmd not specified"); return 1;}

// Copy the command and all of it's arguments
//
   do {if ((vlen = strlen(val)) >= blen)
          {Eroute.Emsg("Config", "stagecmd arguments too long"); break;}
       *bp = ' '; bp++; strcpy(bp, val); bp += vlen; blen -= vlen;
      } while((val = Config.GetWord()));

    if (val) return 1;
    *bp = '\0'; val = buff+1;

// Record the command and operating mode
//
   StageAsync = (isAsync ? 1 : 0);
   StageCreate= isCreate;
   if (StageCmd) free(StageCmd);
   StageCmd = strdup(val);
   return 0;
}

  
/******************************************************************************/
/*                                x t r a c e                                 */
/******************************************************************************/

/* Function: xtrace

   Purpose:  To parse the directive: trace <events>

             <events> the blank separated list of events to trace. Trace
                      directives are cummalative.

   Output: retc upon success or -EINVAL upon failure.
*/

int XrdOssSys::xtrace(XrdOucStream &Config, XrdSysError &Eroute)
{
    char *val;
    static struct traceopts {const char *opname; int opval;} tropts[] =
       {
        {"all",      TRACE_ALL},
        {"debug",    TRACE_Debug},
        {"open",     TRACE_Open},
        {"opendir",  TRACE_Opendir}
       };
    int i, neg, trval = 0, numopts = sizeof(tropts)/sizeof(struct traceopts);

    if (!(val = Config.GetWord()))
       {Eroute.Emsg("Config", "trace option not specified"); return 1;}
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
                      Eroute.Say("Config warning: ignoring invalid trace option '",val,"'.");
                  }
          val = Config.GetWord();
         }
    OssTrace.What = trval;
    return 0;
}

/******************************************************************************/
/*                                  x x f r                                   */
/******************************************************************************/
  
/* Function: xxfr

   Purpose:  To parse the directive: xfr [keep <sec>] 
                                         [<threads> [<speed> [<ovhd> [<hold>]]]]

             keep      number of seconds to keep queued requests
             <threads> number of threads for staging (* uses default).
             <speed>   average speed in bytes/second (* uses default).
             <ovhd>    minimum seconds of overhead (* uses default).
             <hold>    seconds to hold failing requests (* uses default).

   Output: 0 upon success or !0 upon failure.
*/

int XrdOssSys::xxfr(XrdOucStream &Config, XrdSysError &Eroute)
{
    char *val;
    int       thrds = XrdOssXFRTHREADS;
    long long speed = XrdOssXFRSPEED;
    int       ovhd  = XrdOssXFROVHD;
    int       htime = XrdOssXFRHOLD;
    int       ktime;
    int       haveparm = 0;

    while((val = Config.GetWord()))        // <threads> | keep
         {if (!strcmp("keep", val))
             {if ((val = Config.GetWord()))     // keep time
                 if (XrdOuca2x::a2tm(Eroute,"xfr keep",val,&ktime,0)) return 1;
                    else {xfrkeep=ktime; haveparm=1;}
             }
             else break;
         };

    if (!val)
       if (haveparm) return 0;
          else {Eroute.Emsg("Config", "xfr parameter not specified");
                return 1;
               }

      if (strcmp(val, "*") && XrdOuca2x::a2i(Eroute,"xfr threads",val,&thrds,1))
         return 1;

      if ((val = Config.GetWord()))         // <speed>
         {if (strcmp(val, "*") && 
              XrdOuca2x::a2sz(Eroute,"xfr speed",val,&speed,1024)) return 1;

          if ((val = Config.GetWord()))     // <ovhd>
             {if (strcmp(val, "*") && 
                  XrdOuca2x::a2tm(Eroute,"xfr overhead",val,&ovhd,0)) return 1;

              if ((val = Config.GetWord())) // <hold>
                 if (strcmp(val, "*") && 
                    XrdOuca2x::a2tm(Eroute,"xfr hold",val,&htime,0)) return 1;
             }
         }

      xfrthreads = thrds;
      xfrspeed   = speed;
      xfrovhd    = ovhd;
      xfrhold    = htime;
      return 0;
}

/******************************************************************************/
/*                            L i s t _ P a t h                               */
/******************************************************************************/
  
void XrdOssSys::List_Path(const char *pfx, char *pname, 
                          unsigned long long flags, XrdSysError &Eroute)
{
     char buff[4096], *rwmode;

     if (flags & XRDEXP_FORCERO) rwmode = (char *)" forcero";
        else if (flags & XRDEXP_READONLY) rwmode = (char *)" r/o ";
                else rwmode = (char *)" r/w ";
                                 // 0 1 2 3 4 5 6 7 8 9 0 1 2 3
     snprintf(buff, sizeof(buff), "%s%s%s%s%s%s%s%s%s%s%s%s%s%s%s",
              pfx, pname,                                           // 0
              (flags & XRDEXP_COMPCHK  ?  " compchk" : ""),         // 1
              rwmode,                                               // 2
              (flags & XRDEXP_INPLACE  ? " inplace" : ""),          // 3
              (flags & XRDEXP_LOCAL    ? " local"   : ""),          // 4
              (flags & XRDEXP_GLBLRO   ? " globalro": ""),          // 5
              (flags & XRDEXP_NOCHECK  ? " nocheck" : " check"),    // 6
              (flags & XRDEXP_NODREAD  ? " nodread" : " dread"),    // 7
              (flags & XRDEXP_MIG      ? " mig"     : " nomig"),    // 8
     (!(flags & XRDEXP_MMAP)           ? ""         :               // 9
              (flags & XRDEXP_MKEEP    ? " mkeep"   : " nomkeep")),
     (!(flags & XRDEXP_MMAP)           ? ""         :               // 10
              (flags & XRDEXP_MLOK     ? " mlock"   : " nomlock")),
              (flags & XRDEXP_MMAP     ? " mmap"    : ""),          // 11
              (flags & XRDEXP_RCREATE  ? " rcreate" : " norcreate"),// 12
              (flags & XRDEXP_NOSTAGE  ? " nostage" : " stage")     // 13
              );
     Eroute.Say(buff); 
}
