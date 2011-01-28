/******************************************************************************/
/*                                                                            */
/*                       X r d B w m C o n f i g . c c                        */
/*                                                                            */
/* (C) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*               DE-AC02-76-SFO0515 with the Deprtment of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdBwmConfigCVSID = "$Id$";
  
#include <unistd.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <strings.h>
#include <stdio.h>
#include <sys/param.h>

#include "XrdBwm/XrdBwm.hh"
#include "XrdBwm/XrdBwmLogger.hh"
#include "XrdBwm/XrdBwmPolicy.hh"
#include "XrdBwm/XrdBwmPolicy1.hh"
#include "XrdBwm/XrdBwmTrace.hh"

#include "XrdOuc/XrdOuca2x.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysPlugin.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdOuc/XrdOucTrace.hh"

#include "XrdAcc/XrdAccAuthorize.hh"
  
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
  
int XrdBwm::Configure(XrdSysError &Eroute) {
/*
  Function: Establish default values using a configuration file.

  Input:    None.

  Output:   0 upon success or !0 otherwise.
*/
   char *var;
   int  cfgFD, retc, NoGo = 0;
   XrdOucEnv myEnv;
   XrdOucStream Config(&Eroute, getenv("XRDINSTANCE"), &myEnv, "=====> ");

// Print warm-up message
//
   Eroute.Say("++++++ Bwm initialization started.");

// Get the debug level from the command line
//
   if (getenv("XRDDEBUG")) BwmTrace.What = TRACE_ALL;

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
                {if (!strncmp(var, "bwm.", 4))
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
   if (Authorize) NoGo |= setupAuth(Eroute);

// Establish scheduling policy
//
   if (PolLib) NoGo |= setupPolicy(Eroute);
      else Policy = new XrdBwmPolicy1(PolSlotsIn, PolSlotsOut);

// Start logger object
//
   if (!NoGo && Logger) NoGo = Logger->Start(&Eroute);

// Inform the handle of the policy and logger
//
   if (!NoGo) XrdBwmHandle::setPolicy(Policy, Logger);

// All done
//
   Eroute.Say("------ Bwm initialization ", (NoGo ? "failed." : "completed."));
   return NoGo;
}

/******************************************************************************/
/*                     p r i v a t e   f u n c t i o n s                      */
/******************************************************************************/
/******************************************************************************/
/*                             C o n f i g X e q                              */
/******************************************************************************/
  
int XrdBwm::ConfigXeq(char *var, XrdOucStream &Config,
                                 XrdSysError &Eroute)
{
    TS_Bit("authorize",     Authorize, 1);
    TS_Xeq("authlib",       xalib);
    TS_Xeq("log",           xlog);
    TS_Xeq("policy",        xpol);
    TS_Xeq("trace",         xtrace);

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

int XrdBwm::xalib(XrdOucStream &Config, XrdSysError &Eroute)
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
/*                                  x l o g                                   */
/******************************************************************************/
  
/* Function: xlog

   Purpose:  Parse directive: log {* | <|prog> | <>path>}

             <prog>   - is the program to execute and dynamically feed messages
                        about the indicated events. Messages are piped to prog.
             <path>   - is the udp named socket to receive the message. The
                        server creates the path if it's not present. If <path>
                        is an asterisk, then messages are written to standard
                        log file.

   Output: 0 upon success or !0 upon failure.
*/
int XrdBwm::xlog(XrdOucStream &Config, XrdSysError &Eroute)
{
    char *val, parms[1024];

    if (!(val = Config.GetWord()))
       {Eroute.Emsg("Config", "log parameters not specified"); return 1;}

// Get the remaining parameters
//
   Config.RetToken();
   if (!Config.GetRest(parms, sizeof(parms)))
      {Eroute.Emsg("Config", "log parameters too long"); return 1;}
   val = (*parms == '|' ? parms+1 : parms);

// Create a log object
//
   if (Logger) delete Logger;
   Logger = new XrdBwmLogger(val);

// All done
//
   return 0;
}

/******************************************************************************/
/*                                  x p o l                                   */
/******************************************************************************/
  
/* Function: xpol

   Purpose:  To parse the directive: policy args

             Args: {maxslots <innum> <outnum> | lib <path> [<parms>]}

             <num>     maximum number of slots available.
             <path>    if preceeded by lib, the path of the policy library to 
                       be used; otherwise, the file that describes policy.
             <parms>   optional parms to be passed

  Output: 0 upon success or !0 upon failure.
*/

int XrdBwm::xpol(XrdOucStream &Config, XrdSysError &Eroute)
{
    char *val, parms[2048];
    int pl;

// Get next token
//
   if (!(val = Config.GetWord()) || !val[0])
      {Eroute.Emsg("Config", "policy  not specified"); return 1;}

// Start afresh
//
   if (PolLib)  {free(PolLib);  PolLib  = 0;}
   if (PolParm) {free(PolParm); PolParm = 0;}
   PolSlotsIn = PolSlotsOut = 0;

// If the word maxslots then this is a simple policy
//
   if (!strcmp("maxslots", val))
      {if (!(val = Config.GetWord()) || !val[0])
          {Eroute.Emsg("Config", "policy in slots not specified"); return 1;}
       if (XrdOuca2x::a2i(Eroute,"policy in slots",val,&pl,0,32767)) return 1;
       PolSlotsIn = pl;
       if (!(val = Config.GetWord()) || !val[0])
          {Eroute.Emsg("Config", "policy out slots not specified"); return 1;}
       if (XrdOuca2x::a2i(Eroute,"policy out slots",val,&pl,0,32767)) return 1;
       PolSlotsOut = pl;
       return 0;
      }

// Make sure the word is lib
//
   if (strcmp("lib", val))
      {Eroute.Emsg("Config", "invalid policy keyword -", val); return 1;}
   if (!(val = Config.GetWord()) || !val[0])
      {Eroute.Emsg("Config", "policy library not specified"); return 1;}

// Set the library
//
   PolLib = strdup(val);

// Get any parameters
//
   if (!Config.GetRest(parms, sizeof(parms)))
      {Eroute.Emsg("Config", "policy lib parameters too long"); return 1;}
   PolParm = (*parms ? strdup(parms) : 0);

// All done
//
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

int XrdBwm::xtrace(XrdOucStream &Config, XrdSysError &Eroute)
{
    static struct traceopts {const char *opname; int opval;} tropts[] =
       {
        {"all",      TRACE_ALL},
        {"calls",    TRACE_calls},
        {"debug",    TRACE_debug},
        {"delay",    TRACE_delay},
        {"sched",    TRACE_sched},
        {"tokens",   TRACE_tokens}
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
    BwmTrace.What = trval;

// All done
//
   return 0;
}

/******************************************************************************/
/*                             s e t u p A u t h                              */
/******************************************************************************/

int XrdBwm::setupAuth(XrdSysError &Eroute)
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
/*                           s e t u p P o l i c y                            */
/******************************************************************************/

int XrdBwm::setupPolicy(XrdSysError &Eroute)
{
   XrdSysPlugin    *myLib;
   XrdBwmPolicy    *(*ep)(XrdSysLogger *, const char *, const char *);

// Create a plugin object (we will throw this away without deletion because
// the library must stay open but we never want to reference it again).
//
   if (!(myLib = new XrdSysPlugin(&Eroute, PolLib))) return 1;

// Now get the entry point of the object creator
//
   ep = (XrdBwmPolicy *(*)(XrdSysLogger *, const char *, const char *))
                          (myLib->getPlugin("XrdBwmPolicyObject"));
   if (!ep) return 1;

// Get the Object now
//
   return 0 == (Policy = ep(Eroute.logger(), ConfigFN, PolParm));
}
