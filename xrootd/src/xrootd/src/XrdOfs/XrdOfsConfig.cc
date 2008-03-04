/******************************************************************************/
/*                                                                            */
/*                       X r d O f s C o n f i g . c c                        */
/*                                                                            */
/* (C) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*               DE-AC03-76-SFO0515 with the Deprtment of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdOfsConfigCVSID = "$Id$";

/*
   The routines in this file handle ofs() initialization. They get the
   configuration values either from configuration file or XrdOfsconfig.h (in that
   order of precedence).

   These routines are thread-safe if compiled with:
   AIX: -D_THREAD_SAFE
   SUN: -D_REENTRANT
*/
  
#include <unistd.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <iostream.h>
#include <netdb.h>
#include <stdlib.h>
#include <strings.h>
#include <stdio.h>
#include <netinet/in.h>
#include <sys/param.h>

#include "XrdOfs/XrdOfs.hh"
#include "XrdOfs/XrdOfsConfig.hh"
#include "XrdOfs/XrdOfsEvs.hh"
#include "XrdOfs/XrdOfsTrace.hh"

#include "XrdNet/XrdNetDNS.hh"

#include "XrdOuc/XrdOuca2x.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPlugin.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdOuc/XrdOucTrace.hh"
#include "XrdOuc/XrdOucUtils.hh"

#include "XrdOdc/XrdOdcFinder.hh"
#include "XrdAcc/XrdAccAuthorize.hh"

/******************************************************************************/
/*                        G l o b a l   O b j e c t s                         */
/******************************************************************************/

extern XrdOucTrace OfsTrace;

/******************************************************************************/
/*                               d e f i n e s                                */
/******************************************************************************/

#define TS_Xeq(x,m)   if (!strcmp(x,var)) return m(Config,Eroute);

#define TS_Str(x,m)   if (!strcmp(x,var)) {free(m); m = strdup(val); return 0;}

#define TS_PList(x,m)  if (!strcmp(x,var)) \
                          {m.Insert(new XrdOucPList(val,1)); return 0;}

#define TS_Chr(x,m)   if (!strcmp(x,var)) {m = val[0]; return 0;}

#define TS_Bit(x,m,v) if (!strcmp(x,var)) {m |= v; Config.Echo(); return 0;}

#define Max(x,y) (x > y ? x : y)

/******************************************************************************/
/*                             C o n f i g u r e                              */
/******************************************************************************/
  
int XrdOfs::Configure(XrdSysError &Eroute) {
/*
  Function: Establish default values using a configuration file.

  Input:    None.

  Output:   0 upon success or !0 otherwise.
*/
   char *var;
   const char *tmp;
   int  i, j, cfgFD, retc, NoGo = 0;
   XrdOucEnv myEnv;
   XrdOucStream Config(&Eroute, getenv("XRDINSTANCE"), &myEnv, "=====> ");

// Print warm-up message
//
   Eroute.Say("++++++ File system initialization started.");

// Preset all variables with common defaults
//
   Options            = 0;
   if (getenv("XRDDEBUG")) OfsTrace.What = TRACE_MOST | TRACE_debug;

// If there is no config file, return with the defaults sets.
//
   if( !ConfigFN || !*ConfigFN)
     Eroute.Emsg("Config", "Configuration file not specified.");
     else {
           // Try to open the configuration file.
           //
           if ( (cfgFD = open(ConfigFN, O_RDONLY, 0)) < 0)
              return Eroute.Emsg("Config", errno, "open config file",
                                 ConfigFN);
           Config.Attach(cfgFD);

           // Now start reading records until eof.
           //
           while((var = Config.GetMyFirstWord()))
                {if (!strncmp(var, "ofs.", 4)
                 ||  !strcmp(var, "all.role"))
                    if (ConfigXeq(var+4,Config,Eroute)) {Config.Echo();NoGo=1;}
                }

           // Now check if any errors occured during file i/o
           //
           if ((retc = Config.LastError()))
           NoGo = Eroute.Emsg("Config", -retc, "read config file",
                              ConfigFN);
           Config.Close();
          }

// Determine whether we should initialize authorization
//
   if (Options & XrdOfsAUTHORIZE) NoGo |= setupAuth(Eroute);

// Check if redirection wanted
//
   if (getenv("XRDREDIRECT")) i  = XrdOfsREDIRRMT;
      else i = 0;
   if (getenv("XRDRETARGET")) i |= XrdOfsREDIRTRG;
   if (getenv("XRDREDPROXY")) i |= XrdOfsREDIROXY;
   if (i)
      {if ((j = Options & XrdOfsREDIRECT) && (i ^ j))
          {free(myRole); myRole = strdup(theRole(i));
           Eroute.Say("Config warning: command line role options override "
                       "config file; 'ofs.role", myRole, "' in effect.");
          }
       Options &= ~(XrdOfsREDIRECT);
       Options |= i;
      }

// Set the redirect option for upper layers
//
   if (Options & XrdOfsREDIRRMT)
           putenv((char *)"XRDREDIRECT=R");
      else putenv((char *)"XRDREDIRECT=0");

// Initialize redirection.  We type te herald here to minimize confusion
//
   if (Options & XrdOfsREDIRECT)
      {Eroute.Say("++++++ Configuring ", myRole, " role. . .");
       NoGo |= ConfigRedir(Eroute);
      }

// Turn off forwarding if we are not a pure remote redirector or a peer
//
   if (Options & XrdOfsFWD)
      if (!(Options & XrdOfsREDIREER)
      && (Options & (XrdOfsREDIRTRG | XrdOfsREDIROXY)))
         {Eroute.Say("Config warning: forwarding turned off; not a pure manager");
          Options &= ~(XrdOfsFWD);
          fwdCHMOD      = 0; fwdMKDIR      = 0; fwdMKPATH     = 0;
          fwdMV         = 0; fwdRM         = 0; fwdRMDIR      = 0;
         }

// Initialize th Evr object if we are an actual server
//
   if (!(Options & XrdOfsREDIRRMT) 
   && !evrObject.Init(&Eroute, Balancer)) NoGo = 1;

// If we need to send notifications, initialize the interface
//
   if (!NoGo && evsObject) NoGo = evsObject->Start(&Eroute);

// Display final configuration
//
   if (!NoGo) Config_Display(Eroute);

// All done
//
   tmp = (NoGo ? " initialization failed." : " initialization completed.");
   Eroute.Say("------ File system ", myRole, tmp);
   return NoGo;
}

/******************************************************************************/
/*                        C o n f i g _ D i s p l a y                         */
/******************************************************************************/

#define setBuff(x,y) {strcpy(bp, x); bp += y;}
  
void XrdOfs::Config_Display(XrdSysError &Eroute)
{
     const char *cloc;
     char buff[8192], fwbuff[256], *bp;
     int i;

     if (!(Options &  XrdOfsFWD)) fwbuff[0] = '\0';
        else {bp = fwbuff;
              setBuff("       ofs.forward", 11);
              if (fwdCHMOD) setBuff(" chmod", 6);
              if (fwdMKDIR) setBuff(" mkdir", 6);
              if (fwdMV   ) setBuff(" mv"   , 3);
              if (fwdRM   ) setBuff(" rm"   , 3);
              if (fwdRMDIR) setBuff(" rmdir", 6);
              setBuff("\n", 1);
             }

     if (!ConfigFN || !ConfigFN[0]) cloc = "default";
        else cloc = ConfigFN;
     snprintf(buff, sizeof(buff), "Config effective %s ofs configuration:\n"
                                  "       ofs.role %s\n"
                                  "%s"
                                  "%s%s%s"
                                  "%s"
                                  "       ofs.fdscan     %d %d %d\n"
                                  "%s"
                                  "       ofs.maxdelay   %d\n"
                                  "%s%s%s"
                                  "       ofs.trace      %x",
              cloc, myRole,
              (Options & XrdOfsAUTHORIZE ? "       ofs.authorize\n" : ""),
              (AuthLib                   ? "       ofs.authlib " : ""),
              (AuthLib ? AuthLib : ""), (AuthLib ? "\n" : ""),
              (Options & XrdOfsFDNOSHARE ? "       ofs.fdnoshare\n" : ""),
              FDOpenMax, FDMinIdle, FDMaxIdle, fwbuff, MaxDelay,
              (OssLib                    ? "       ofs.osslib " : ""),
              (OssLib ? OssLib : ""), (OssLib ? "\n" : ""),
              OfsTrace.What);

     Eroute.Say(buff);

     if (evsObject)
        {bp = buff;
         setBuff("       ofs.notify ", 11);              //  1234567890
         if (evsObject->Enabled(XrdOfsEvs::Chmod))  setBuff("chmod ",  6);
         if (evsObject->Enabled(XrdOfsEvs::Closer)) setBuff("closer ", 7);
         if (evsObject->Enabled(XrdOfsEvs::Closew)) setBuff("closew ", 7);
         if (evsObject->Enabled(XrdOfsEvs::Create)) setBuff("create ", 7);
         if (evsObject->Enabled(XrdOfsEvs::Mkdir))  setBuff("mkdir ",  6);
         if (evsObject->Enabled(XrdOfsEvs::Mv))     setBuff("mv ",     3);
         if (evsObject->Enabled(XrdOfsEvs::Openr))  setBuff("openr ",  6);
         if (evsObject->Enabled(XrdOfsEvs::Openw))  setBuff("openw ",  6);
         if (evsObject->Enabled(XrdOfsEvs::Rm))     setBuff("rm ",     3);
         if (evsObject->Enabled(XrdOfsEvs::Rmdir))  setBuff("rmdir ",  6);
         if (evsObject->Enabled(XrdOfsEvs::Fwrite)) setBuff("fwrite ", 7);
         setBuff("msgs ", 5);
         i=sprintf(fwbuff,"%d %d ",evsObject->maxSmsg(),evsObject->maxLmsg());
         setBuff(fwbuff, i);
         cloc = evsObject->Prog();
         if (*cloc != '>') setBuff("|",1);
         setBuff(cloc, strlen(cloc));
         setBuff("\n", 1);
         Eroute.Say(buff);
        }

     List_VPlist((char *)"       ofs.validpath  ", VPlist, Eroute);
}

/******************************************************************************/
/*                     p r i v a t e   f u n c t i o n s                      */
/******************************************************************************/
/******************************************************************************/
/*                           C o n f i g R e d i r                            */
/******************************************************************************/
  
int XrdOfs::ConfigRedir(XrdSysError &Eroute) 
{
   int isRedir = Options & XrdOfsREDIRRMT;

// For manager roles, we simply do a standard config
//
   if (isRedir) {Finder=(XrdOdcFinder *)new XrdOdcFinderRMT(Eroute.logger(),
                           (Options & XrdOfsREDIRTRG  ? XrdOdcIsTarget : 0));
       if (!Finder->Configure(ConfigFN))
          {delete Finder; Finder = 0; return 1;}
      }

// For proxy roles, we specify the proxy oss library if possible
//
   if (Options & XrdOfsREDIROXY)
      {char buff[2048], *bp, *libofs = getenv("XRDOFSLIB");
       if (OssLib) Eroute.Say("Config warning: ",
                   "specified osslib overrides default proxy lib.");
          else {if (!libofs) bp = buff;
                   else {strcpy(buff, libofs); bp = buff+strlen(buff)-1;
                         while(bp != buff && *(bp-1) != '/') bp--;
                        }
                strcpy(bp, "libXrdProxy.so");
                OssLib = strdup(buff);
               }
      }

// For server roles find the port number and create the object
//
   if (Options & (XrdOfsREDIRTRG | (XrdOfsREDIREER & ~ XrdOfsREDIRRMT)))
      {if (!myPort)
          {Eroute.Emsg("Config", "Unable to determine server's port number.");
           return 1;
          }
       Balancer = new XrdOdcFinderTRG(Eroute.logger(), 
                         (isRedir ? XrdOdcIsRedir : 0), myPort);
       if (!Balancer->Configure(ConfigFN)) 
          {delete Balancer; Balancer = 0; return 1;}
       if (Options & XrdOfsREDIROXY) Balancer = 0; // No chatting for proxies
      }

// All done
//
   return 0;
}
  
/******************************************************************************/
/*                             C o n f i g X e q                              */
/******************************************************************************/
  
int XrdOfs::ConfigXeq(char *var, XrdOucStream &Config,
                                 XrdSysError &Eroute)
{
    char *val, vBuff[64];

    // Now assign the appropriate global variable
    //
    TS_Bit("authorize",     Options, XrdOfsAUTHORIZE);
    TS_Xeq("authlib",       xalib);
    TS_Bit("fdnoshare",     Options, XrdOfsFDNOSHARE);
    TS_Xeq("fdscan",        xfdscan);
    TS_Xeq("forward",       xforward);
    TS_Xeq("locktry",       xlocktry); // Deprecated
    TS_Xeq("maxdelay",      xmaxd);
    TS_Xeq("notify",        xnot);
    TS_Xeq("osslib",        xolib);
    TS_Xeq("redirect",      xred);     // Deprecated
    TS_Xeq("role",          xrole);
    TS_Xeq("trace",         xtrace);

    // Get the actual value for simple directives
    //
    strlcpy(vBuff, var, sizeof(vBuff)); var = vBuff;
    if (!(val = Config.GetWord()))
       {Eroute.Emsg("Config", "value not specified for", var); return 1;}

    // Process simple directives
    //
    TS_PList("validpath",   VPlist);

    // No match found, complain.
    //
    Eroute.Say("Config warning: ignoring unknown directive '",var,"'.");
    Config.Echo();
    return 0;
}

/******************************************************************************/
/*                                 x a l i b                                  */
/******************************************************************************/
  
/* Function: xalib

   Purpose:  To parse the directive: authlib <path> [<parms>]

             <path>    the path of the authorization library to be used.
             <parms>   optional parms to be passed

  Output: 0 upon success or !0 upon failure.
*/

int XrdOfs::xalib(XrdOucStream &Config, XrdSysError &Eroute)
{
    char *val, parms[1024];

// Get the path
//
   if (!(val = Config.GetWord()) || !val[0])
      {Eroute.Emsg("Config", "authlib not specified"); return 1;}

// Record the path
//
   if (AuthLib) free(AuthLib);
   AuthLib = strdup(val);

// Record any parms
//
   if (!Config.GetRest(parms, sizeof(parms)))
      {Eroute.Emsg("Config", "authlib parameters too long"); return 1;}
   if (AuthParm) free(AuthParm);
   AuthParm = (*parms ? strdup(parms) : 0);
   return 0;
}

/******************************************************************************/
/*                               x f d s c a n                                */
/******************************************************************************/

/* Function: xfdscan

   Purpose:  To parse the directive: fdscan <numopen> <minidle> <maxidle>

             <numopen> number of fd's that must be open for scan to commence.
             <minidle> minimum number of seconds between scans.
             <maxidle> maximum number of seconds a file can be idle before
                       it is closed.

   Output: 0 upon success or !0 upon failure.
*/

int XrdOfs::xfdscan(XrdOucStream &Config, XrdSysError &Eroute)
{
    char *val;
    int numf, minidle, maxidle;

      if (!(val = Config.GetWord()))
         {Eroute.Emsg("Config","fdscan numfiles value not specified");return 1;}
      if (XrdOuca2x::a2i(Eroute, "fdscan numfiles", val, &numf, 0)) return 1;

      if (!(val = Config.GetWord()))
         {Eroute.Emsg("Config","fdscan minidle value not specified"); return 1;}
      if (XrdOuca2x::a2tm(Eroute, "fdscan minidle",val, &minidle,0)) return 1;

      if (!(val = Config.GetWord()))
         {Eroute.Emsg("Config","fdscan maxidle value not specified"); return 1;}
      if (XrdOuca2x::a2tm(Eroute,"fdscan maxidle", val, &maxidle, minidle))
         return 1;

      FDOpenMax = numf;
      FDMinIdle = minidle;
      FDMaxIdle = maxidle;
      return 0;
}

/******************************************************************************/
/*                              x f o r w a r d                               */
/******************************************************************************/
  
/* Function: xforward

   Purpose:  To parse the directive: forward [1way | 2way] <metaops>

             1way      forward does not respond (the default)
             2way      forward responds; relay response back.
             <metaops> list of meta-file operations to forward to manager

   Output: 0 upon success or !0 upon failure.
*/

int XrdOfs::xforward(XrdOucStream &Config, XrdSysError &Eroute)
{
    enum fwdType {OfsFWDALL = 0x1f, OfsFWDCHMOD = 0x01, OfsFWDMKDIR = 0x02,
                  OfsFWDMV  = 0x04, OfsFWDRM    = 0x08, OfsFWDRMDIR = 0x10,
                  OfsFWDREM = 0x18, OfsFWDNONE  = 0};

    static struct fwdopts {const char *opname; fwdType opval;} fwopts[] =
       {
        {"all",      OfsFWDALL},
        {"chmod",    OfsFWDCHMOD},
        {"mkdir",    OfsFWDMKDIR},
        {"mv",       OfsFWDMV},
        {"rm",       OfsFWDRM},
        {"rmdir",    OfsFWDRMDIR},
        {"remove",   OfsFWDREM}
       };
    int fwval = OfsFWDNONE, fwspec = OfsFWDNONE;
    int numopts = sizeof(fwopts)/sizeof(struct fwdopts);
    int i, neg, is2way = 0;
    char *val;

    if (!(val = Config.GetWord()))
       {Eroute.Emsg("Config", "foward option not specified"); return 1;}
    if ((is2way = !strcmp("2way", val)) || !strcmp("1way", val))
       if (!(val = Config.GetWord()))
          {Eroute.Emsg("Config", "foward operation not specified"); return 1;}

    while (val)
         {if (!strcmp(val, "off")) {fwval = OfsFWDNONE; fwspec = OfsFWDALL;}
             else {if ((neg = (val[0] == '-' && val[1]))) val++;
                   for (i = 0; i < numopts; i++)
                       {if (!strcmp(val, fwopts[i].opname))
                           {if (neg) fwval &= ~fwopts[i].opval;
                               else  fwval |=  fwopts[i].opval;
                            fwspec |= fwopts[i].opval;
                            break;
                           }
                       }
                   if (i >= numopts)
                      Eroute.Say("Config warning: ignoring invalid foward option '",val,"'.");
                  }
          val = Config.GetWord();
         }

    if (fwspec & OfsFWDCHMOD) 
        fwdCHMOD = (fwval & OfsFWDCHMOD ? (is2way ? "+chmod"  : "chmod")  : 0);
    if (fwspec & OfsFWDMKDIR) 
        fwdMKDIR = (fwval & OfsFWDMKDIR ? (is2way ? "+mkdir"  : "mkdir")  : 0);
        fwdMKPATH= (fwval & OfsFWDMKDIR ? (is2way ? "+mkpath" : "mkpath") : 0);
    if (fwspec & OfsFWDMV)    
        fwdMV    = (fwval & OfsFWDMV    ? (is2way ? "+mv"     : "mv")     : 0);
    if (fwspec & OfsFWDRM)    
        fwdRM    = (fwval & OfsFWDRM    ? (is2way ? "+rm"     : "rm")     : 0);
    if (fwspec & OfsFWDRMDIR) 
        fwdRMDIR = (fwval & OfsFWDRMDIR ? (is2way ? "+rmdir"  : "rmdir")  : 0);

// All done
//
   if (fwdCHMOD || fwdMKDIR || fwdMV || fwdRM || fwdRMDIR)
           Options |=   XrdOfsFWD;
      else Options &= ~(XrdOfsFWD);
   return 0;
}

/******************************************************************************/
/*                              x l o c k t r y                               */
/******************************************************************************/
  
/* Function: locktry

   Purpose:  To parse the directive: locktry <times> <wait>

             <times>   number of times to try to get a lock.
             <wait>    number of milliseconds to wait between tries.

   Output: 0 upon success or !0 upon failure.
*/

int XrdOfs::xlocktry(XrdOucStream &Config, XrdSysError &Eroute)
{
    char *val;
    int numt, mswt;

      if (!(val = Config.GetWord()))
         {Eroute.Emsg("Config","locktry count not specified"); return 1;}
      if (XrdOuca2x::a2i(Eroute, "locktry count", val, &numt, 0)) return 1;

      if (!(val = Config.GetWord()))
         {Eroute.Emsg("Config","locktry wait interval not specified");return 1;}
      if (XrdOuca2x::a2i(Eroute, "locktry wait",val, &mswt,0)) return 1;

      LockTries = numt;
      LockWait  = mswt;
      return 0;
}
  
/******************************************************************************/
/*                                 x m a x d                                  */
/******************************************************************************/

/* Function: xmaxd

   Purpose:  To parse the directive: maxdelay <secs>

             <secs>    maximum delay imposed for staging

   Output: 0 upon success or !0 upon failure.
*/

int XrdOfs::xmaxd(XrdOucStream &Config, XrdSysError &Eroute)
{
    char *val;
    int maxd;

      if (!(val = Config.GetWord()))
         {Eroute.Emsg("Config","maxdelay value not specified");return 1;}
      if (XrdOuca2x::a2i(Eroute, "maxdelay", val, &maxd, 30)) return 1;

      MaxDelay = maxd;
      return 0;
}

/******************************************************************************/
/*                                  x n o t                                   */
/* Based on code developed by Derek Feichtinger, CERN.                        */
/******************************************************************************/

/* Function: xnot

   Purpose:  Parse directive: notify <events> [msgs <min> [<max>]] 
                                     {|<prog> | ><path>}

   Args:     <events> - one or more of: all chmod closer closew close mkdir mv
                                        openr openw open rm rmdir fwrite
             msgs     - Maximum number of messages to keep and queue. The
                        <min> if for small messages (default 90) and <max> is
                        for big messages (default 10).
             <prog>   - is the program to execute and dynamically feed messages
                        about the indicated events. Messages are piped to prog.
             <path>   - is the udp named socket to receive the message. The
                        server creates the path if it's not present.

   Output: 0 upon success or !0 upon failure.
*/
int XrdOfs::xnot(XrdOucStream &Config, XrdSysError &Eroute)
{
    static struct notopts {const char *opname; XrdOfsEvs::Event opval;} 
        noopts[] = {
        {"all",      XrdOfsEvs::All},
        {"chmod",    XrdOfsEvs::Chmod},
        {"close",    XrdOfsEvs::Close},
        {"closer",   XrdOfsEvs::Closer},
        {"closew",   XrdOfsEvs::Closew},
        {"create",   XrdOfsEvs::Create},
        {"mkdir",    XrdOfsEvs::Mkdir},
        {"mv",       XrdOfsEvs::Mv},
        {"open",     XrdOfsEvs::Open},
        {"openr",    XrdOfsEvs::Openr},
        {"openw",    XrdOfsEvs::Openw},
        {"rm",       XrdOfsEvs::Rm},
        {"rmdir",    XrdOfsEvs::Rmdir},
        {"fwrite",   XrdOfsEvs::Fwrite}
       };
    XrdOfsEvs::Event noval = XrdOfsEvs::None;
    int numopts = sizeof(noopts)/sizeof(struct notopts);
    int i, neg, msgL = 90, msgB = 10;
    char *val, parms[1024];

    if (!(val = Config.GetWord()))
       {Eroute.Emsg("Config", "notify parameters not specified"); return 1;}
    while (val && *val != '|' && *val != '>')
         {if (!strcmp(val, "msgs"))
             {if (!(val = Config.GetWord()))
                 {Eroute.Emsg("Config", "notify msgs value not specified");
                  return 1;
                 }
              if (XrdOuca2x::a2i(Eroute, "msg count", val, &msgL, 0)) return 1;
              if (!(val = Config.GetWord())) break;
              if (isdigit(*val)
              && XrdOuca2x::a2i(Eroute, "msg count", val, &msgB, 0)) return 1;
              if (!(val = Config.GetWord())) break;
              continue;
             }
          if ((neg = (val[0] == '-' && val[1]))) val++;
          for (i = 0; i < numopts; i++)
              {if (!strcmp(val, noopts[i].opname))
                  {if (neg) noval = static_cast<XrdOfsEvs::Event>(~noopts[i].opval & noval);
                      else  noval = static_cast<XrdOfsEvs::Event>(noopts[i].opval|noval);
                   break;
                  }
              }
          if (i >= numopts)
             Eroute.Say("Config warning: ignoring invalid notify event '",val,"'.");
          val = Config.GetWord();
         }

// Check if we have a program here and some events
//
   if (!val)   {Eroute.Emsg("Config","notify program not specified");return 1;}
   if (!noval) {Eroute.Emsg("Config","notify events not specified"); return 1;}

// Get the remaining parameters
//
   Config.RetToken();
   if (!Config.GetRest(parms, sizeof(parms)))
      {Eroute.Emsg("Config", "authlib parameters too long"); return 1;}
   val = (*parms == '|' ? parms+1 : parms);

// Get the remaining

// Create an notification object
//
   if (evsObject) delete evsObject;
   evsObject = new XrdOfsEvs(noval, val, msgL, msgB);

// All done
//
   return 0;
}
  

/******************************************************************************/
/*                                 x o l i b                                  */
/******************************************************************************/
  
/* Function: xolib

   Purpose:  To parse the directive: osslib <path> [<parms>]

             <path>    the path of the oss library to be used.
             <parms>   optional parms to be passed

  Output: 0 upon success or !0 upon failure.
*/

int XrdOfs::xolib(XrdOucStream &Config, XrdSysError &Eroute)
{
    char *val, parms[2048];
    int pl;

// Get the path and parms
//
   if (!(val = Config.GetWord()) || !val[0])
      {Eroute.Emsg("Config", "osslib not specified"); return 1;}

// Combine the path and parameters
//
   strcpy(parms, val);
   pl = strlen(val);
   *(parms+pl) = ' ';
   if (!Config.GetRest(parms+pl+1, sizeof(parms)-pl-1))
      {Eroute.Emsg("Config", "osslib parameters too long"); return 1;}

// Record the path
//
   if (OssLib) free(OssLib);
   OssLib = strdup(parms);
   return 0;
}

/******************************************************************************/
/*                                  x r e d                                   */
/******************************************************************************/

/* Function: xred

   Purpose:  Parse directive: redirect [proxy|remote|target] [if ...]

   Args:     proxy    - enables this server for proxy   load balancing
             remote   - enables this server for dynamic load balancing
             target   - enables this server as a redirection target
             if       - applies directive if "if" is true. 
                        See XrdOucUtils::doIf() for syntax.

   Output: 0 upon success or !0 upon failure.
*/

int XrdOfs::xred(XrdOucStream &Config, XrdSysError &Eroute)
{
    const char *mode = "remote";
    char *val;
    int rc, ropt = 0;

    Eroute.Say("Config warning: redirect directive is deprecated; use 'all.role'.");

    if ((val = Config.GetWord()))
       {     if (!strcmp("proxy",  val)) {ropt = XrdOfsREDIROXY;
                                          mode = "proxy";
                                         }
        else if (!strcmp("remote", val))  ropt = XrdOfsREDIRRMT;
        else if (!strcmp("target", val)) {ropt = XrdOfsREDIRTRG;
                                          mode = "target";
                                         }
       }

    if (!ropt) ropt = XrdOfsREDIRRMT;
       else if (val) val = Config.GetWord();

    if (val)
       {if (strcmp("if", val)) Config.RetToken();
        if ((rc = XrdOucUtils::doIf(&Eroute, Config, "redirect directive",
                                   getenv("XRDHOST"), getenv("XRDNAME"),
                                   getenv("XRDPROG"))) <= 0)
           return (rc < 0);
       }
    Options |= ropt;
    return 0;
}

/******************************************************************************/
/*                                 x r o l e                                  */
/******************************************************************************/

/* Function: xrole

   Purpose:  Parse: role {[peer] [proxy] manager | peer | proxy | [proxy] server
                          | [proxy] supervisor} [if ...]

             manager    xrootd: act as a manager (redirecting server). Prefix
                                modifications are ignored.
                        olbd:   accept server subscribes and redirectors. Prefix
                                modifiers do the following:
                                peer  - subscribe to other managers as a peer
                                proxy - manage a cluster of proxy servers

             peer       xrootd: same as "peer manager"
                        olbd:   same as "peer manager" but no server subscribers
                                are required to function (i.e., run stand-alone).

             proxy      xrootd: act as a server but supply data from another 
                                server. No local olbd is present or required.
                        olbd:   Generates an error as this makes no sense.

             server     xrootd: act as a server (supply local data). Prefix
                                modifications do the following:
                                proxy - server is part of a cluster. A local
                                        olbd is required.
                        olbd:   subscribe to a manager, possibly as a proxy.

             supervisor xrootd: equivalent to manager. The prefix modification
                                is ignored.
                        olbd:   equivalent to manager but also subscribe to a
                                manager. When proxy is specified, then subscribe
                                as a proxy and only accept proxies.

             if         Apply the manager directive if "if" is true. See
                        XrdOucUtils:doIf() for "if" syntax.

   Notes: 1. This directive superceeds the redirect directive. There is no
             equivalent for "peer" designation. For other possibilities:
             manager    -> redirect remote
             proxy      -> redirect proxy
             server     -> redirect target
             supervisor -> redirect remote + target

          2. The peer designation only affects how the olbd communicates.

   Type: Server only, non-dynamic.

   Output: 0 upon success or !0 upon failure.
*/

int XrdOfs::xrole(XrdOucStream &Config, XrdSysError &Eroute)
{
   const int resetit = ~XrdOfsREDIRECT;
   char role[64];
   char *val;
   int rc, qopt = 0, ropt = 0, sopt = 0;

   *role = '\0';
   if (!(val = Config.GetWord()))
      {Eroute.Emsg("Config", "role not specified"); return 1;}

// First screen for "peer"
//
   if (!strcmp("peer", val))
      {qopt = XrdOfsREDIREER;
       strcpy(role, val);
       val = Config.GetWord();
      }

// Now scan for "proxy"
//
   if (val && !strcmp("proxy", val))
      {ropt = XrdOfsREDIROXY;
       if (qopt) strcat(role, " ");
       strcat(role, val);
       val = Config.GetWord();
      }

// Scan for other possible alternatives
//
   if (val && strcmp("if", val))
      {     if (!strcmp("manager",    val)) sopt = XrdOfsREDIRRMT;
       else if (!strcmp("server",     val)) sopt = XrdOfsREDIRTRG;
       else if (!strcmp("supervisor", val)) sopt = XrdOfsREDIRVER;
       else    {Eroute.Emsg("Config", "invalid role -", val); return 1;}

       if (qopt || ropt) strcat(role, " ");
       strcat(role, val);
       val = Config.GetWord();
      }

// Scan for invalid roles: peer proxy | peer server | {peer} supervisor
//
   if ((qopt && ropt && !sopt) 
   ||  (qopt && sopt == XrdOfsREDIRTRG)
   ||  (qopt && sopt == XrdOfsREDIRVER))
      {Eroute.Emsg("Config", "invalid role -", role); return 1;}

// Make sure a role was specified
//
    if (!(ropt = qopt | ropt | sopt))
       {Eroute.Emsg("Config", "role not specified"); return 1;}

// Pick up optional "if"
//
    if (val && !strcmp("if", val))
       if ((rc = XrdOucUtils::doIf(&Eroute,Config,"role directive",
                                   getenv("XRDHOST"), getenv("XRDNAME"),
                                   getenv("XRDPROG"))) <= 0)
           return (rc < 0);

// Set values
//
    free(myRole);
    myRole = strdup(role);
    Options &= resetit;
    Options |= ropt;
    return 0;
}

/******************************************************************************/
/*                                x t r a c e                                 */
/******************************************************************************/

/* Function: xtrace

   Purpose:  To parse the directive: trace <events>

             <events> the blank separated list of events to trace. Trace
                      directives are cummalative.

   Output: 0 upon success or !0 upon failure.
*/

int XrdOfs::xtrace(XrdOucStream &Config, XrdSysError &Eroute)
{
    static struct traceopts {const char *opname; int opval;} tropts[] =
       {
        {"aio",      TRACE_aio},
        {"all",      TRACE_ALL},
        {"chmod",    TRACE_chmod},
        {"close",    TRACE_close},
        {"closedir", TRACE_closedir},
        {"debug",    TRACE_debug},
        {"delay",    TRACE_delay},
        {"dir",      TRACE_dir},
        {"exists",   TRACE_exists},
        {"getstats", TRACE_getstats},
        {"fsctl",    TRACE_fsctl},
        {"io",       TRACE_IO},
        {"mkdir",    TRACE_mkdir},
        {"most",     TRACE_MOST},
        {"open",     TRACE_open},
        {"opendir",  TRACE_opendir},
        {"qscan",    TRACE_qscan},
        {"read",     TRACE_read},
        {"readdir",  TRACE_readdir},
        {"redirect", TRACE_redirect},
        {"remove",   TRACE_remove},
        {"rename",   TRACE_rename},
        {"sync",     TRACE_sync},
        {"truncate", TRACE_truncate},
        {"write",    TRACE_write}
       };
    int i, neg, trval = 0, numopts = sizeof(tropts)/sizeof(struct traceopts);
    char *val;

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
    OfsTrace.What = trval;

// All done
//
   return 0;
}

/******************************************************************************/
/*                             s e t u p A u t h                              */
/******************************************************************************/

int XrdOfs::setupAuth(XrdSysError &Eroute)
{
   XrdSysPlugin    *myLib;
   XrdAccAuthorize *(*ep)(XrdSysLogger *, const char *, const char *);

// Authorization comes from the library or we use the default
//
   if (!AuthLib) return 0 == (Authorization =
                 XrdAccAuthorizeObject(Eroute.logger(),ConfigFN,AuthParm));

// Create a pluin object (we will throw this away without deletion because
// the library must stay open but we never want to reference it again).
//
   if (!(myLib = new XrdSysPlugin(&Eroute, AuthLib))) return 1;

// Now get the entry point of the object creator
//
   ep = (XrdAccAuthorize *(*)(XrdSysLogger *, const char *, const char *))
                             (myLib->getPlugin("XrdAccAuthorizeObject"));
   if (!ep) return 1;

// Get the Object now
//
   return 0 == (Authorization = ep(Eroute.logger(), ConfigFN, AuthParm));
}
  
/******************************************************************************/
/*                               t h e R o l e                                */
/******************************************************************************/
  
const char *XrdOfs::theRole(int opts)
{
          if (opts & XrdOfsREDIREER) return "peer";
     else if (opts & XrdOfsREDIRRMT
          &&  opts & XrdOfsREDIRTRG) return "supervisor";
     else if (opts & XrdOfsREDIRRMT) return "manager";
     else if (opts & XrdOfsREDIROXY) return "proxy";
                                     return "server";
}

/******************************************************************************/
/*                           L i s t _ V P l i s t                            */
/******************************************************************************/
  
void XrdOfs::List_VPlist(char *lname, 
                      XrdOucPListAnchor &plist, XrdSysError &Eroute)
{
     XrdOucPList *fp;

     fp = plist.Next();
     while(fp) {Eroute.Say(lname, fp->Path()); fp = fp->Next();}
}
