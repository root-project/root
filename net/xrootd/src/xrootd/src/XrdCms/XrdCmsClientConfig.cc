/******************************************************************************/
/*                                                                            */
/*                 X r d C m s C l i e n t C o n f i g . c c                  */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

const char *XrdCmsClientConfigCVSID = "$Id$";

// Based on: XrdCmsClientConfig.cc,v 1.24 2007/07/31 02:24:52 abh

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

#include "XrdCms/XrdCmsClientConfig.hh"
#include "XrdCms/XrdCmsClientMsg.hh"
#include "XrdCms/XrdCmsSecurity.hh"
#include "XrdCms/XrdCmsTrace.hh"

#include "XrdSys/XrdSysHeaders.hh"
#include "XrdOuc/XrdOuca2x.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdOuc/XrdOucTList.hh"
#include "XrdOuc/XrdOucUtils.hh"
#include "XrdNet/XrdNetDNS.hh"

using namespace XrdCms;

/******************************************************************************/
/*                               d e f i n e s                                */
/******************************************************************************/

#define TS_Xeq(x,m)    if (!strcmp(x,var)) return m(Config);
  
/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/
  
XrdCmsClientConfig::~XrdCmsClientConfig()
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
  
int XrdCmsClientConfig::Configure(char *cfn, configWhat What, configHow How)
{
/*
  Function: Establish configuration at start up time.

  Input:    None.

  Output:   0 upon success or !0 otherwise.
*/
   EPNAME("Configure");
   static const char *mySid = 0;
   XrdOucTList *tpe, *tpl;
   int i, NoGo = 0;
   const char *eText = 0;
   char buff[256], *slash, *temp, *bP, sfx;

// Preset some values
//
   myHost = getenv("XRDHOST");
   myName = XrdOucUtils::InstName(1);
   CMSPath= strdup("/tmp/");
   isMeta = How & configMeta;
   isMan  = What& configMan;

// Process the configuration file
//
   if (!(NoGo = ConfigProc(cfn)) && isMan)
      {if (How & configProxy) eText = (PanList ? 0 : "Proxy manager");
          else if (!ManList)
                  eText = (How & configMeta ? "Meta manager" : "Manager");
       if (eText) {Say.Emsg("Config", eText, "not specified."); NoGo=1;}
      }

// Reset tracing options
//
   if (getenv("XRDDEBUG")) Trace.What = TRACE_ALL;

// Set proper local socket path
//
   temp=XrdOucUtils::genPath(CMSPath, XrdOucUtils::InstName(-1), ".olb");
   free(CMSPath); CMSPath = temp;
   XrdOucEnv::Export("XRDCMSPATH", temp);
   XrdOucEnv::Export("XRDOLBPATH", temp); //Compatability

// Generate the system ID for this configuration.
//
   tpl = (How & configProxy ? PanList : ManList);
   if (!mySid)
      {     if (What & configServer) sfx = 's';
       else if (What & configSuper)  sfx = 'u';
       else                          sfx = 'm';
       if (!(mySid = XrdCmsSecurity::setSystemID(tpl, myName, myHost, sfx)))
          {Say.Emsg("xrootd","Unable to generate system ID; too many managers.");
           NoGo = 1;
          } else {DEBUG("Global System Identification: " <<mySid);}
      }

// Export the manager list
//
   if (tpl)
      {i = 0; tpe = tpl;
       while(tpe) {i += strlen(tpe->text) + 9; tpe = tpe->next;}
       bP = temp = (char *)malloc(i);
       while(tpl)
            {bP += sprintf(bP, "%s:%d ", tpl->text, tpl->val);
             tpl = tpl->next;
            }
       *(bP-1) = '\0';
       XrdOucEnv::Export("XRDCMSMAN", temp); free(temp);
      }

// Construct proper communications path for a supervisor node
//
   i = strlen(CMSPath);
   if (What & configSuper)
      {while((tpl = ManList)) {ManList = tpl->next; delete tpl;}
       slash = (CMSPath[i-1] == '/' ? (char *)"" : (char *)"/");
       sprintf(buff, "%s%solbd.super", CMSPath, slash);
       ManList = new XrdOucTList(buff, -1, 0);
       SMode = SModeP = FailOver;
      }

// Construct proper old communication path for a target node
//
   temp = (What & (configMan|configSuper) ? (char *)"nimda" : (char *)"admin");
   slash = (CMSPath[i-1] == '/' ? (char *)"" : (char *)"/");
   sprintf(buff, "%s%solbd.%s", CMSPath, slash, temp);
   free(CMSPath); CMSPath = strdup(buff);

   RepWaitMS = RepWait * 1000;

// Initialize the msg queue
//
   if (XrdCmsClientMsg::Init())
      {Say.Emsg("Config", ENOMEM, "allocate initial msg objects");
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
  
int XrdCmsClientConfig::ConfigProc(char *ConfigFN)
{
  static int DoneOnce = 0;
  char *var;
  int  cfgFD, retc, NoGo = 0;
  XrdOucEnv myEnv;
  XrdOucStream Config((DoneOnce ? 0 : &Say), getenv("XRDINSTANCE"),
                      &myEnv, "=====> ");

// Make sure we have a config file
//
   if (!ConfigFN || !*ConfigFN)
      {Say.Emsg("Config", "cms configuration file not specified.");
       return 1;
      }

// Try to open the configuration file.
//
   if ( (cfgFD = open(ConfigFN, O_RDONLY, 0)) < 0)
      {Say.Emsg("Config", errno, "open config file", ConfigFN);
       return 1;
      }
   Config.Attach(cfgFD);

// Now start reading records until eof.
//
   while((var = Config.GetMyFirstWord()))
        {if (!strncmp(var, "cms.", 4)
         ||  !strncmp(var, "odc.", 4)      // Compatability
         ||  !strcmp(var, "all.manager")
         ||  !strcmp(var, "all.adminpath")
         ||  !strcmp(var, "olb.adminpath"))
            if (ConfigXeq(var+4, Config)) {Config.Echo(); NoGo = 1;}
        }

// Now check if any errors occured during file i/o
//
   if ((retc = Config.LastError()))
      NoGo = Say.Emsg("Config", retc, "read config file", ConfigFN);
   Config.Close();

// Return final return code
//
   DoneOnce = 1;
   return NoGo;
}

/******************************************************************************/
/*                             C o n f i g X e q                              */
/******************************************************************************/

int XrdCmsClientConfig::ConfigXeq(char *var, XrdOucStream &Config)
{

   // Process items. for either a local or a remote configuration
   //
   TS_Xeq("conwait",       xconw);
   TS_Xeq("manager",       xmang);
   TS_Xeq("adminpath",     xapath);
   TS_Xeq("request",       xreqs);
   TS_Xeq("trace",         xtrac);
   return 0;
}

/******************************************************************************/
/*                                x a p a t h                                 */
/******************************************************************************/

/* Function: xapath

   Purpose:  To parse the directive: adminpath <path> [ group ]

             <path>    the path of the named socket to use for admin requests.
                       Only the path may be specified, not the filename.
             group     allow group access to the path.

   Type: Manager only, non-dynamic.

   Output: 0 upon success or !0 upon failure.
*/
  
int XrdCmsClientConfig::xapath(XrdOucStream &Config)
{
    struct sockaddr_un USock;
    char *pval;

// Get the path
//
   pval = Config.GetWord();
   if (!pval || !pval[0])
      {Say.Emsg("Config", "cms admin path not specified"); return 1;}

// Make sure it's an absolute path
//
   if (*pval != '/')
      {Say.Emsg("Config", "cms admin path not absolute"); return 1;}

// Make sure path is not too long (account for "/olbd.admin")
//                                              12345678901
   if (strlen(pval) > sizeof(USock.sun_path) - 11)
      {Say.Emsg("Config", "cms admin path is too long.");
       return 1;
      }

// Record the path
//
   if (CMSPath) free(CMSPath);
   CMSPath = strdup(pval);
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

int XrdCmsClientConfig::xconw(XrdOucStream &Config)
{
    char *val;
    int cw;

    if (!(val = Config.GetWord()))
       {Say.Emsg("Config", "conwait value not specified."); return 1;}

    if (XrdOuca2x::a2tm(Say,"conwait value",val,&cw,1)) return 1;

    ConWait = cw;
    return 0;
}
  
/******************************************************************************/
/*                                 x m a n g                                  */
/******************************************************************************/

/* Function: xmang

   Purpose:  Parse: manager [meta | peer | proxy] [all|any]
                            <host>[+][:<port>|<port>] [if ...]

             meta   For cmsd:   Specifies the manager when running as a manager
                    For xrootd: Specifies the manager when running as a meta
             peer   For cmsd:   Specifies the manager when running as a peer
                    For xrootd: The directive is ignored.
             proxy  For cmsd:   This directive is ignored.
                    For xrootd: Specifies the cms-proxy service manager
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
            cmsd:   Subscribes to each manager whens role is not peer.
            xrootd: Logins in as a redirector to each manager when role is not 
                    proxy or server.

   Type: Remote server only, non-dynamic.

   Output: 0 upon success or !0 upon failure.
*/

int XrdCmsClientConfig::xmang(XrdOucStream &Config)
{
    struct sockaddr InetAddr[8];
    XrdOucTList *tp = 0, *tpp = 0, *tpnew;
    char *val, *bval = 0, *mval = 0;
    int rc, i, j, port, xMeta = 0, isProxy = 0, smode = FailOver;

//  Process the optional "peer" or "proxy"
//
    if ((val = Config.GetWord()))
       {if (!strcmp("peer", val)) return Config.noEcho();
        if ((isProxy = !strcmp("proxy", val))) val = Config.GetWord();
           else if ((xMeta = !strcmp("meta", val)))
                   if (isMeta || isMan) val = Config.GetWord();
                      else return Config.noEcho();
                   else if (isMeta) return Config.noEcho();
       }

//  We can accept this manager. Skip the optional "all" or "any"
//
    if (val)
       {     if (!strcmp("any", val)) smode = FailOver;
        else if (!strcmp("all", val)) smode = RoundRob;
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
       {Say.Emsg("Config","manager host name not specified"); return 1;}
       else mval = strdup(val);

    if (!(val = index(mval,':'))) val = Config.GetWord();
       else {*val = '\0'; val++;}

    if (val)
       {if (isdigit(*val))
           {if (XrdOuca2x::a2i(Say,"manager port",val,&port,1,65535))
               port = 0;
           }
           else if (!(port = XrdNetDNS::getPort(val, "tcp")))
                   {Say.Emsg("Config", "unable to find tcp service", val);
                    port = 0;
                   }
       } else Say.Emsg("Config","manager port not specified for",mval);

    if (!port) {free(mval); return 1;}

    if (myHost && (val = Config.GetWord()) && !strcmp("if", val))
       if ((rc = XrdOucUtils::doIf(&Say,Config,"role directive",myHost, myName,
                                   getenv("XRDPROG"))) <= 0)
          {free(mval);
           return (rc < 0);
          }

    i = strlen(mval);
    if (mval[i-1] != '+')
       {i = 0; val = mval; mval = XrdNetDNS::getHostName(mval); free(val);}
        else {bval = strdup(mval); mval[i-1] = '\0';
              if (!(i = XrdNetDNS::getHostAddr(mval, InetAddr, 8)))
                 {Say.Emsg("Config","Manager host", mval, "not found");
                  free(bval); free(mval); return 1;
                 }
             }

    if (xMeta && !isMeta) 
       {haveMeta = 1; free(bval); free(mval); return 0;}

    do {if (i)
           {i--; free(mval);
            mval = XrdNetDNS::getHostName(InetAddr[i]);
            Say.Emsg("Config", bval, "-> all.manager", mval);
           }
        tp = (isProxy ? PanList : ManList); tpp = 0; j = 1;
        while(tp) 
             if ((j = strcmp(tp->text, mval)) < 0 || tp->val != port)
                {tpp = tp; tp = tp->next;}
                else {if (!j) Say.Emsg("Config","Duplicate manager",mval);
                      break;
                     }
        if (j) {tpnew = new XrdOucTList(mval, port, tp);
                if (tpp) tpp->next = tpnew;
                   else if (isProxy) PanList = tpnew;
                           else      ManList = tpnew;
               }
       } while(i);

    if (bval) free(bval);
    free(mval);
    return 0;
}
  
/******************************************************************************/
/*                                 x r e q s                                  */
/******************************************************************************/

/* Function: xreqs

   Purpose:  To parse the directive: request [repwait <sec1>] [delay <sec2>]
                                             [noresp <cnt>] [prep <ms>]

             <sec1>  number of seconds to wait for a locate reply
             <sec2>  number of seconds to delay a retry upon failure
             <cnt>   number of no-responses before cms fault declared.
             <ms>    milliseconds between prepare requests

   Type: Remote server only, dynamic.

   Output: 0 upon success or !0 upon failure.
*/

int XrdCmsClientConfig::xreqs(XrdOucStream &Config)
{
    char *val;
    static struct reqsopts {const char *opname; int istime; int *oploc;}
           rqopts[] =
       {
        {"delay",    1, &RepDelay},
        {"fwd",      1, &FwdWait},
        {"noresp",   0, &RepNone},
        {"prep",     1, &PrepWait},
        {"repwait",  1, &RepWait}
       };
    int i, ppp, numopts = sizeof(rqopts)/sizeof(struct reqsopts);

    if (!(val = Config.GetWord()))
       {Say.Emsg("Config", "request arguments not specified"); return 1;}

    while (val)
    do {for (i = 0; i < numopts; i++)
            if (!strcmp(val, rqopts[i].opname))
               { if (!(val = Config.GetWord()))
                  {Say.Emsg("Config","request argument value not specified");
                   return 1;}
                   if (rqopts[i].istime ?
                       XrdOuca2x::a2tm(Say,"request value",val,&ppp,1) :
                       XrdOuca2x::a2i( Say,"request value",val,&ppp,1))
                      return 1;
                      else *rqopts[i].oploc = ppp;
                break;
               }
        if (i >= numopts) Say.Say("Config warning: ignoring invalid request option '",val,"'.");
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

int XrdCmsClientConfig::xtrac(XrdOucStream &Config)
{
    char  *val;
    static struct traceopts {const char *opname; int opval;} tropts[] =
       {
        {"all",      TRACE_ALL},
        {"debug",    TRACE_Debug},
        {"forward",  TRACE_Forward},
        {"redirect", TRACE_Redirect},
        {"defer",    TRACE_Defer},
        {"stage",    TRACE_Stage}
       };
    int i, neg, trval = 0, numopts = sizeof(tropts)/sizeof(struct traceopts);

    if (!(val = Config.GetWord()))
       {Say.Emsg("config", "trace option not specified"); return 1;}
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
                      Say.Say("Config warning: ignoring invalid trace option '",val,"'.");
                  }
          val = Config.GetWord();
         }
    Trace.What = trval;
    return 0;
}
