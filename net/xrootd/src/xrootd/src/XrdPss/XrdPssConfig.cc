/******************************************************************************/
/*                                                                            */
/*                       X r d P s s C o n f i g . c c                        */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

const char *XrdPssConfigCVSID = "$Id$";

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

#include "XrdPss/XrdPss.hh"
#include "XrdOuc/XrdOuca2x.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdOuc/XrdOucTList.hh"
#include "XrdOuc/XrdOucUtils.hh"
#include "XrdNet/XrdNetDNS.hh"
#include "XrdPosix/XrdPosixXrootd.hh"

/******************************************************************************/
/*                               d e f i n e s                                */
/******************************************************************************/

#define TS_Xeq(x,m)    if (!strcmp(x,var)) return m(&eDest, Config);

/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/

const char  *XrdPssSys::ConfigFN;       // -> Pointer to the config file name
const char  *XrdPssSys::myHost;
const char  *XrdPssSys::myName;
XrdOucTList *XrdPssSys::PanList = 0;
char        *XrdPssSys::hdrData;
char         XrdPssSys::hdrLen;
long         XrdPssSys::rdAheadSz =  0;
long         XrdPssSys::rdCacheSz =  0;
long         XrdPssSys::numStream =  8;

namespace XrdProxy
{
static XrdPosixXrootd  *Xroot;
  
extern XrdSysError      eDest;
}

using namespace XrdProxy;

/******************************************************************************/
/*                             C o n f i g u r e                              */
/******************************************************************************/
  
int XrdPssSys::Configure(const char *cfn)
{
/*
  Function: Establish configuration at start up time.

  Input:    None.

  Output:   0 upon success or !0 otherwise.
*/
   int NoGo = 0;

// Preset tracing options
//
   if (getenv("XRDDEBUG")) XrdPosixXrootd::setDebug(1);
   myHost = getenv("XRDHOST");
   myName = XrdOucUtils::InstName(1);

// Set the default read cache size value and parallel streams
//
   if (rdAheadSz >= 0) XrdPosixXrootd::setEnv("ReadAheadSize",       rdAheadSz);
   if (rdCacheSz >= 0) XrdPosixXrootd::setEnv("ReadCacheSize",       rdCacheSz);
   if (numStream >= 0) XrdPosixXrootd::setEnv("ParStreamsPerPhyConn",numStream);

// Process the configuration file
//
   if ((NoGo = ConfigProc(cfn))) return NoGo;

// Build the URL header
//
   if (!PanList)
      {eDest.Emsg("Config", "Manager for proxy service not specified.");
       return 1;
      }
   if (buildHdr()) return 1;

// Allocate an Xroot proxy object (only one needed here)
//
   Xroot = new XrdPosixXrootd(32768, 16384);
   return 0;
}

/******************************************************************************/
/*                     P r i v a t e   F u n c t i o n s                      */
/******************************************************************************/
/******************************************************************************/
/*                              b u i l d H d r                               */
/******************************************************************************/
  
int XrdPssSys::buildHdr()
{
   XrdOucTList *tp = PanList;
   char port[16], buff[1024], *pb;
   int n, bleft = sizeof(buff);

   strcpy(buff, "root://"); pb = buff+strlen(buff); bleft -= strlen(buff);

   while(tp)
        {if ((n = strlcpy(pb, tp->text, bleft)) >= bleft) break;
         pb += n; bleft -= n;
         if (bleft <= 0) break;
         sprintf(port, ":%d", tp->val);
         if ((n = strlcpy(pb, port, bleft)) >= bleft) break;
         pb += n; bleft -= n;
         if (bleft <= 1) break;
         if (tp->next) *pb++ = ',';
            else       *pb++ = '/';
         bleft--; *pb = '\0';
         tp = tp->next;
        }

   if (tp)
      {eDest.Emsg("Config", "Too many proxy service managers specified.");
       return 1;
      }

   hdrData = strdup(buff);
   hdrLen  = strlen(buff);
   return 0;
}

/******************************************************************************/
/*                            C o n f i g P r o c                             */
/******************************************************************************/
  
int XrdPssSys::ConfigProc(const char *Cfn)
{
  char *var;
  int  cfgFD, retc, NoGo = 0;
  XrdOucEnv myEnv;
  XrdOucStream Config(&eDest, getenv("XRDINSTANCE"), &myEnv, "=====> ");

// Make sure we have a config file
//
   if (!Cfn || !*Cfn)
      {eDest.Emsg("Config", "pss configuration file not specified.");
       return 1;
      }

// Try to open the configuration file.
//
   if ( (cfgFD = open(Cfn, O_RDONLY, 0)) < 0)
      {eDest.Emsg("Config", errno, "open config file", Cfn);
       return 1;
      }
   Config.Attach(cfgFD);

// Now start reading records until eof.
//
   while((var = Config.GetMyFirstWord()))
        {if (!strncmp(var, "pss.", 4)
         ||  !strcmp(var, "all.manager")
         ||  !strcmp(var, "all.adminpath"))
            if (ConfigXeq(var+4, Config)) {Config.Echo(); NoGo = 1;}
        }

// Now check if any errors occured during file i/o
//
   if ((retc = Config.LastError()))
      NoGo = eDest.Emsg("Config", retc, "read config file", Cfn);
   Config.Close();

// Return final return code
//
   return NoGo;
}

/******************************************************************************/
/*                             C o n f i g X e q                              */
/******************************************************************************/

int XrdPssSys::ConfigXeq(char *var, XrdOucStream &Config)
{

   // Process items. for either a local or a remote configuration
   //
   TS_Xeq("manager",       xmang);
   TS_Xeq("setopt",        xsopt);
   TS_Xeq("trace",         xtrac);

   // No match found, complain.
   //
   eDest.Say("Config warning: ignoring unknown directive '",var,"'.");
   Config.Echo();
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
                    For xrootd: Specifies the pss-proxy service manager
             all    Distribute requests across all managers.
                    This is the default for proxy servers.
             any    Choose different manager only when necessary.
                    This is ignored for proxy servers.
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

int XrdPssSys::xmang(XrdSysError *errp, XrdOucStream &Config)
{
    struct sockaddr InetAddr[8];
    XrdOucTList *tp = 0;
    char *val, *bval = 0, *mval = 0;
    int rc, i, port;

//  Only accept "manager proxy"
//
    if ((val = Config.GetWord()))
       {if (strcmp("proxy", val)) return 0;
        val = Config.GetWord();
       }

//  We can accept this manager. Skip the optional "all" or "any"
//
    if (val)
       {     if (!strcmp("any", val)
             ||  !strcmp("all", val)) val = Config.GetWord();
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
       if ((rc = XrdOucUtils::doIf(errp,Config,"role directive",myHost, myName,
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
        tp = PanList;
        while(tp) 
             if (strcmp(tp->text, mval) || tp->val != port) tp = tp->next;
                else {errp->Emsg("Config","Duplicate manager",mval);
                      break;
                     }
        if (tp) break;
        PanList = new XrdOucTList(mval, port, PanList);
       } while(i);

    if (bval) free(bval);
    free(mval);
    return tp != 0;
}
  
/******************************************************************************/
/*                                 x s o p t                                  */
/******************************************************************************/

/* Function: xsopt

   Purpose:  To parse the directive: setopt <keyword> <value>

             <keyword> is an XrdClient option keyword.
             <value>   is the value the option is to have.

   Output: retc upon success or -EINVAL upon failure.
*/

int XrdPssSys::xsopt(XrdSysError *Eroute, XrdOucStream &Config)
{
    char  kword[256], *val, *kvp;
    long  kval;
    static const char *Sopts[] =
       {
         "DataServerConn_ttl",
         "DebugLevel",
         "DfltTcpWindowSize",
         "LBServerConn_ttl"
         "ParStreamsPerPhyConn",
         "ParStreamsPerPhyConn",
         "ReadAheadSize",
         "ReadCacheBlk",
         "ReadCacheSize",
         "RemoveUsedCacheBlocks"
       };
    int i, numopts = sizeof(Sopts)/sizeof(const char *);

    if (!(val = Config.GetWord()))
       {Eroute->Emsg("config", "setopt keyword not specified"); return 1;}
    strlcpy(kword, val, sizeof(kword));
    if (!(val = Config.GetWord()))
       {Eroute->Emsg("config", "setopt", kword, "value not specified"); 
        return 1;
       }

    kval = strtol(val, &kvp, 10);
    if (*kvp)
       {Eroute->Emsg("config", kword, "setopt keyword value is invalid -", val);
        return 1;
       }

    for (i = 0; i < numopts; i++)
        if (!strcmp(Sopts[i], kword))
           {XrdPosixXrootd::setEnv(kword, kval);
            return 0;
           }

    Eroute->Say("Config warning: ignoring unknown setopt '",kword,"'.");
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

int XrdPssSys::xtrac(XrdSysError *Eroute, XrdOucStream &Config)
{
    char  *val;
    static struct traceopts {const char *opname; int opval;} tropts[] =
       {
        {"all",      3},
        {"debug",    2},
        {"on",       1}
       };
    int i, trval = 0, numopts = sizeof(tropts)/sizeof(struct traceopts);

    if (!(val = Config.GetWord()))
       {Eroute->Emsg("config", "trace option not specified"); return 1;}
    while (val)
         {if (!strcmp(val, "off")) trval = 0;
             else {for (i = 0; i < numopts; i++)
                       {if (!strcmp(val, tropts[i].opname))
                           {trval |=  tropts[i].opval;
                            break;
                           }
                       }
                   if (i >= numopts)
                      Eroute->Say("Config warning: ignoring invalid trace option '",val,"'.");
                  }
          val = Config.GetWord();
         }
    XrdPosixXrootd::setDebug(trval);
    return 0;
}
