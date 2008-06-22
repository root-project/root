/******************************************************************************/
/*                                                                            */
/*                       X r d O d c C o n f i g . c c                        */
/*                                                                            */
/* (C) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*                DE-AC03-76-SFO0515 with the Deprtment of Energy             */
/******************************************************************************/

//          $Id$

const char *XrdOdcConfigCVSID = "$Id$";

#include <unistd.h>
#include <ctype.h>
#include <strings.h>
#include <stdio.h>
#include <sys/param.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <fcntl.h>

#include "XrdSys/XrdSysHeaders.hh"
#include "XrdOdc/XrdOdcConfig.hh"
#include "XrdOdc/XrdOdcMsg.hh"
#include "XrdOdc/XrdOdcTrace.hh"
#include "XrdOuc/XrdOuca2x.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdOuc/XrdOucTList.hh"
#include "XrdOuc/XrdOucUtils.hh"
#include "XrdNet/XrdNetDNS.hh"

/******************************************************************************/
/*                               d e f i n e s                                */
/******************************************************************************/

#define TS_Xeq(x,m)    if (!strcmp(x,var)) return m(eDest, Config);

/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/
  
XrdOdcConfig::~XrdOdcConfig()
{
  XrdOucTList *tp, *tpp;

  tpp = ManList;
  while((tp = tpp)) {tpp = tp->next; delete tp;}
  tpp = PanList;
  while((tp = tpp)) {tpp = tp->next; delete tp;}
}

/******************************************************************************/
/*                             C o n f i g u r e                              */
/******************************************************************************/
  
int XrdOdcConfig::Configure(char *cfn, const char *mode, int isBoth)
{
/*
  Function: Establish configuration at start up time.

  Input:    None.

  Output:   0 upon success or !0 otherwise.
*/
   extern XrdOucTrace OdcTrace;
   int i, NoGo = 0;
   char buff[256], *slash, *temp;

// Preset tracing options
//
   if (getenv("XRDDEBUG")) OdcTrace.What = TRACE_ALL;
   myHost = getenv("XRDHOST");
   myName = getenv("XRDNAME");
   if (!myName || !*myName) myName = "anon";
   OLBPath= strdup("/tmp/");

// Process the configuration file
//
   if (!(NoGo = ConfigProc(cfn)))
           {if (*mode == 'P')
               {if (!PanList)
                   {eDest->Emsg("Config", "Proxy manager not specified.");
                    NoGo=1;
                   }
               }
           }
      else if (*mode == 'R' && !isBoth)
              {if (!ManList)
                  {eDest->Emsg("Config", "Manager not specified.");
                   NoGo=1;
                  }
              }

// Set proper local socket path
//
   temp=XrdOucUtils::genPath(OLBPath,(strcmp("anon",myName)?myName:0), ".olb");
   free(OLBPath); 
   OLBPath = temp;
   sprintf(buff, "XRDOLBPATH=%s", temp);
   putenv(strdup(buff));
   i = strlen(OLBPath);

// Construct proper olb communications path for a supervisor node
//
   if (*mode == 'R' && isBoth)
      {XrdOucTList *tpl;
       while((tpl = ManList)) {ManList = tpl->next; delete tpl;}
       slash = (OLBPath[i-1] == '/' ? (char *)"" : (char *)"/");
       sprintf(buff, "%s%solbd.super", OLBPath, slash);
       ManList = new XrdOucTList(buff, -1, 0);
       SMode = SModeP = ODC_FAILOVER;
      }

// Construct proper old communication path for a target node
//
   temp = (isBoth ? (char *)"nimda" : (char *)"admin");
   slash = (OLBPath[i-1] == '/' ? (char *)"" : (char *)"/");
   sprintf(buff, "%s%solbd.%s", OLBPath, slash, temp);
   free(OLBPath);
   OLBPath = strdup(buff);

   RepWaitMS = RepWait * 1000;

// Initialize the msg queue
//
   if (XrdOdcMsg::Init())
      {eDest->Emsg("Config", ENOMEM, "allocate initial msg objects");
       NoGo = 1;
      }

   return NoGo;
}

/******************************************************************************/
/*                     P r i v a t e   F u n c t i o n s                      */
/******************************************************************************/
/******************************************************************************/
/*                            C o n f i g P r o c                             */
/******************************************************************************/
  
int XrdOdcConfig::ConfigProc(char *ConfigFN)
{
  static int DoneOnce = 0;
  char *var;
  int  cfgFD, retc, NoGo = 0;
  XrdOucEnv myEnv;
  XrdOucStream Config((DoneOnce ? 0 : eDest), getenv("XRDINSTANCE"), 
                      &myEnv, "=====> ");

// Make sure we have a config file
//
   if (!ConfigFN || !*ConfigFN)
      {eDest->Emsg("Config", "odc configuration file not specified.");
       return 1;
      }

// Try to open the configuration file.
//
   if ( (cfgFD = open(ConfigFN, O_RDONLY, 0)) < 0)
      {eDest->Emsg("Config", errno, "open config file", ConfigFN);
       return 1;
      }
   Config.Attach(cfgFD);

// Now start reading records until eof.
//
   while((var = Config.GetMyFirstWord()))
        {if (!strncmp(var, "odc.", 4)
         ||  !strcmp(var, "all.manager")
         ||  !strcmp(var, "all.adminpath")
         ||  !strcmp(var, "olb.adminpath"))
            if (ConfigXeq(var+4, Config)) {Config.Echo(); NoGo = 1;}
        }

// Now check if any errors occured during file i/o
//
   if ((retc = Config.LastError()))
      NoGo = eDest->Emsg("Config", retc, "read config file", ConfigFN);
   Config.Close();

// Return final return code
//
   DoneOnce = 1;
   return NoGo;
}

/******************************************************************************/
/*                             C o n f i g X e q                              */
/******************************************************************************/

int XrdOdcConfig::ConfigXeq(char *var, XrdOucStream &Config)
{

   // Process items. for either a local or a remote configuration
   //
   TS_Xeq("conwait",       xconw);
   TS_Xeq("manager",       xmang);
   TS_Xeq("adminpath",     xapath);
   TS_Xeq("olbapath",      xapath);
   TS_Xeq("request",       xreqs);
   TS_Xeq("trace",         xtrac);

   // Directives for backward comapatibility
   //
   if (!strcmp(var, "msgkeep")) return 0;

   // No match found, complain.
   //
   eDest->Say("Config warning: ignoring unknown directive '",var,"'.");
   Config.Echo();
   return 0;
}

/******************************************************************************/
/*                                x a p a t h                                 */
/******************************************************************************/

/* Function: xapath

   Purpose:  To parse the directive: olbapath <path> [ group ]

             <path>    the path of the named socket to use for admin requests.
                       Only the path may be specified, not the filename.
             group     allow group access to the path.

   Type: Manager only, non-dynamic.

   Output: 0 upon success or !0 upon failure.
*/
  
int XrdOdcConfig::xapath(XrdSysError *errp, XrdOucStream &Config)
{
    struct sockaddr_un USock;
    char *pval;

// Get the path
//
   pval = Config.GetWord();
   if (!pval || !pval[0])
      {errp->Emsg("Config", "olb admin path not specified"); return 1;}

// Make sure it's an absolute path
//
   if (*pval != '/')
      {errp->Emsg("Config", "olb admin path not absolute"); return 1;}

// Make sure path is not too long (account for "/olbd.admin")
//                                              12345678901
   if (strlen(pval) > sizeof(USock.sun_path) - 11)
      {errp->Emsg("Config", "olb admin path is too long.");
       return 1;
      }

// Record the path
//
   if (OLBPath) free(OLBPath);
   OLBPath = strdup(pval);
   return 0;
}

/******************************************************************************/
/*                                 x c o n w                                  */
/******************************************************************************/

/* Function: xconw

   Purpose:  To parse the directive: conwait <sec>

             <sec>   number of seconds to wait for a manager connection

   Type: Remote server only, dynamic.

   Output: 0 upon success or !0 upon failure.
*/

int XrdOdcConfig::xconw(XrdSysError *errp, XrdOucStream &Config)
{
    char *val;
    int cw;

    if (!(val = Config.GetWord()))
       {errp->Emsg("Config", "conwait value not specified."); return 1;}

    if (XrdOuca2x::a2tm(*errp,"conwait value",val,&cw,1)) return 1;

    ConWait = cw;
    return 0;
}
  
/******************************************************************************/
/*                                 x m a n g                                  */
/******************************************************************************/

/* Function: xmang

   Purpose:  Parse: manager [peer | proxy] [all|any] <host>[+][:<port>|<port>] 
                                                     [if ...]

             peer   For olbd:   Specified the manager when running as a peer
                    For xrootd: The directive is ignored.
             proxy  For olbd:   This directive is ignored.
                    For xrootd: Specifies the odc-proxy service manager
             all    Distribute requests across all managers.
             any    Choose different manager only when necessary (default).
             <host> The dns name of the host that is the cache manager.
                    If the host name ends with a plus, all addresses that are
                    associated with the host are treated as managers.
             <port> The port number to use for this host.
             if     Apply the manager directive if "if" is true. See
                    XrdOucUtils:doIf() for "if" syntax.

   Notes:   Any number of manager directives can be given. When niether peer nor
            proxy is specified, then regardless of role the following occurs:
            olbd:   Subscribes to each manager whens role is not peer.
            xrootd: Logins in as a redirector to each manager when role is not 
                    proxy or server.

   Type: Remote server only, non-dynamic.

   Output: 0 upon success or !0 upon failure.
*/

int XrdOdcConfig::xmang(XrdSysError *errp, XrdOucStream &Config)
{
    struct sockaddr InetAddr[8];
    XrdOucTList *tp = 0;
    char *val, *bval = 0, *mval = 0;
    int rc, i, port, isProxy = 0, smode = ODC_FAILOVER;

//  Process the optional "peer" or "proxy"
//
    if ((val = Config.GetWord()))
       {if (!strcmp("peer", val)) return 0;
        if ((isProxy = !strcmp("proxy", val))) val = Config.GetWord();
       }

//  We can accept this manager. Skip the optional "all" or "any"
//
    if (val)
       {     if (!strcmp("any", val)) smode = ODC_FAILOVER;
        else if (!strcmp("all", val)) smode = ODC_ROUNDROB;
        else                          smode = 0;
        if (smode)
           {if (isProxy) SModeP = smode;
               else      SMode  = smode;
            val = Config.GetWord();
           }
       }

//  Get the actual manager
//
    if (!val)
       {errp->Emsg("Config","manager host name not specified"); return 1;}
       else mval = strdup(val);

    if (!(val = index(mval,':'))) val = Config.GetWord();
       else {*val = '\0'; val++;}

    if (val)
       {if (isdigit(*val))
            {if (XrdOuca2x::a2i(*errp,"manager port",val,&port,1,65535))
                port = 0;
            }
            else if (!(port = XrdNetDNS::getPort(val, "tcp")))
                    {errp->Emsg("Config", "unable to find tcp service", val);
                     port = 0;
                    }
       } else errp->Emsg("Config","manager port not specified for",mval);

    if (!port) {free(mval); return 1;}

    if (myHost && (val = Config.GetWord()) && !strcmp("if", val))
       if ((rc = XrdOucUtils::doIf(eDest,Config,"role directive",myHost, myName,
                                   getenv("XRDPROG"))) <= 0)
          {free(mval);
           return (rc < 0);
          }

    i = strlen(mval);
    if (mval[i-1] != '+') i = 0;
        else {bval = strdup(mval); mval[i-1] = '\0';
              if (!(i = XrdNetDNS::getHostAddr(mval, InetAddr, 8)))
                 {errp->Emsg("Config","Manager host", mval, "not found");
                  free(bval); free(mval); return 1;
                 }
             }

    do {if (i)
           {i--; free(mval);
            mval = XrdNetDNS::getHostName(InetAddr[i]);
            errp->Emsg("Config", bval, "-> odc.manager", mval);
           }
        tp = (isProxy ? PanList : ManList);
        while(tp) 
             if (strcmp(tp->text, mval) || tp->val != port) tp = tp->next;
                else {errp->Emsg("Config","Duplicate manager",mval);
                      break;
                     }
        if (tp) break;
        if (isProxy) PanList = new XrdOucTList(mval, port, PanList);
           else      ManList = new XrdOucTList(mval, port, ManList);
       } while(i);

    if (bval) free(bval);
    free(mval);
    return tp != 0;
}
  
/******************************************************************************/
/*                                 x r e q s                                  */
/******************************************************************************/

/* Function: xreqs

   Purpose:  To parse the directive: request [repwait <sec1>] [delay <sec2>]
                                             [noresp <cnt>] [prep <ms>]

             <sec1>  number of seconds to wait for a locate reply
             <sec2>  number of seconds to delay a retry upon failure
             <cnt>   number of no-responses before olb fault declared.
             <ms>    milliseconds between prepare requests

   Type: Remote server only, dynamic.

   Output: 0 upon success or !0 upon failure.
*/

int XrdOdcConfig::xreqs(XrdSysError *errp, XrdOucStream &Config)
{
    char *val;
    static struct reqsopts {const char *opname; int istime; int *oploc;}
           rqopts[] =
       {
        {"delay",    1, &RepDelay},
        {"noresp",   0, &RepNone},
        {"prep",     1, &PrepWait},
        {"repwait",  1, &RepWait}
       };
    int i, ppp, numopts = sizeof(rqopts)/sizeof(struct reqsopts);

    if (!(val = Config.GetWord()))
       {errp->Emsg("Config", "request arguments not specified"); return 1;}

    while (val)
    do {for (i = 0; i < numopts; i++)
            if (!strcmp(val, rqopts[i].opname))
               { if (!(val = Config.GetWord()))
                  {errp->Emsg("Config", 
                      "request argument value not specified"); 
                   return 1;}
                   if (rqopts[i].istime ?
                       XrdOuca2x::a2tm(*errp,"request value",val,&ppp,1) :
                       XrdOuca2x::a2i( *errp,"request value",val,&ppp,1))
                      return 1;
                      else *rqopts[i].oploc = ppp;
                break;
               }
        if (i >= numopts) errp->Say("Config warning: ignoring invalid request option '",val,"'.");
       } while((val = Config.GetWord()));
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

int XrdOdcConfig::xtrac(XrdSysError *Eroute, XrdOucStream &Config)
{
    extern XrdOucTrace OdcTrace;
    char  *val;
    static struct traceopts {const char *opname; int opval;} tropts[] =
       {
        {"all",      TRACE_ALL},
        {"debug",    TRACE_Debug},
        {"forward",  TRACE_Forward},
        {"redirect", TRACE_Redirect}
       };
    int i, neg, trval = 0, numopts = sizeof(tropts)/sizeof(struct traceopts);

    if (!(val = Config.GetWord()))
       {Eroute->Emsg("config", "trace option not specified"); return 1;}
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
                      Eroute->Say("Config warning: ignoring invalid trace option '",val,"'.");
                  }
          val = Config.GetWord();
         }
    OdcTrace.What = trval;
    return 0;
}
