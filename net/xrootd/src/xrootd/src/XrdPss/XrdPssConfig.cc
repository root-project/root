/******************************************************************************/
/*                                                                            */
/*                       X r d P s s C o n f i g . c c                        */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <unistd.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <sys/param.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <fcntl.h>

#include "XrdFfs/XrdFfsDent.hh"
#include "XrdFfs/XrdFfsMisc.hh"
#include "XrdFfs/XrdFfsWcache.hh"
#include "XrdFfs/XrdFfsQueue.hh"

#include "XrdPss/XrdPss.hh"

#include "XrdOuc/XrdOuca2x.hh"
#include "XrdOuc/XrdOucEnv.hh"

#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdSys/XrdSysPlugin.hh"

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
uid_t        XrdPssSys::myUid     =  geteuid();
gid_t        XrdPssSys::myGid     =  getegid();

XrdOucTList *XrdPssSys::ManList   =  0;
const char  *XrdPssSys::urlPlain  =  0;
int          XrdPssSys::urlPlen   =  0;
int          XrdPssSys::hdrLen    =  0;
const char  *XrdPssSys::hdrData   =  0;
int          XrdPssSys::Workers   = 16;

char         XrdPssSys::allChmod  =  0;
char         XrdPssSys::allMkdir  =  0;
char         XrdPssSys::allMv     =  0;
char         XrdPssSys::allRm     =  0;
char         XrdPssSys::allRmdir  =  0;
char         XrdPssSys::allTrunc  =  0;

namespace XrdProxy
{
static XrdPosixXrootd  *Xroot;
  
extern XrdSysError      eDest;

static const int maxHLen = 1024;
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
   struct {const char *Typ; char *Loc;} Fwd[] = {{" ch", &allChmod},
                                                 {" mk", &allMkdir},
                                                 {" mv", &allMv   },
                                                 {" rd", &allRmdir},
                                                 {" rm", &allRm   },
                                                 {" tr", &allTrunc},
                                                 {0,     0        }
                                                };
   const char *xP;
   char *eP, theRdr[maxHLen+1024];
   int i, NoGo = 0;

   N2NLib = NULL;
   theN2N = NULL;

// Preset tracing options
//
   if (getenv("XRDDEBUG")) XrdPosixXrootd::setDebug(1);
   myHost = getenv("XRDHOST");
   myName = XrdOucUtils::InstName(1);

// Set the default values for the client
//
   XrdPosixXrootd::setEnv("ReadAheadSize",           1024*1024);
   XrdPosixXrootd::setEnv("ReadCacheSize",       512*1024*1024);
   XrdPosixXrootd::setEnv("ParStreamsPerPhyConn",      long(0)); // Temp!
   XrdPosixXrootd::setEnv("PurgeWrittenBlocks",              1);
   XrdPosixXrootd::setEnv("DataServerConn_ttl",          20*60);
   XrdPosixXrootd::setEnv("LBServerConn_ttl",            60*60);

// Process the configuration file
//
   if ((NoGo = ConfigProc(cfn))) return NoGo;

// Build the URL header
//
   if (!ManList)
      {eDest.Emsg("Config", "Origin for proxy service not specified.");
       return 1;
      }
   if (buildHdr()) return 1;

// Copy out the forwarding that might be happening via the ofs
//
   i = 0;
   if ((eP = getenv("XRDOFS_FWD")))
      while(Fwd[i].Typ)
           {if (!strstr(eP, Fwd[i].Typ)) *(Fwd[i].Loc) = 1; i++;}

// Create a plain url for future use
//
   urlPlen = sprintf(theRdr, hdrData, "", "", "", "", "", "", "", "");
   urlPlain= strdup(theRdr);

// Configure the N2N library:
   if (N2NLib && (NoGo = ConfigN2N())) return NoGo;

// We would really like that the Ffs interface use the generic method of
// keeping track of data servers. It does not and it even can't handle more
// than one export (really). But it does mean we need to give it a valid one.
//
   if (!(eP = getenv("XRDEXPORTS")) || *eP != '/') xP = "/tmp";
      else if ((xP = rindex(eP, ' '))) xP++;
              else xP = eP;

// Initialize the Ffs (we don't use xrd_init() as it messes up the settings
// We also do not initialize secsss as we don't know how to effectively use it.
//
   strcpy(&theRdr[urlPlen], xP);
// XrdFfsMisc_xrd_secsss_init();
   XrdFfsMisc_refresh_url_cache(theRdr);
   XrdFfsDent_cache_init();
   XrdFfsWcache_init();
   XrdFfsQueue_create_workers(Workers);

// Allocate an Xroot proxy object (only one needed here). Tell it to not
// shadow open files with real file descriptors (we will be honest).
//
   Xroot = new XrdPosixXrootd(-32768, 16384);
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
   XrdOucTList *tp = ManList;
   char buff[maxHLen], *pb;
   int n, bleft = sizeof(buff);

// Fill in start of header
//
   strcpy(buff, "root://"); pb = buff+strlen(buff); bleft -= strlen(buff);

// The redirector list must fit into 1K bytes (along with header)
//
   while(tp)
        {n = snprintf(pb, bleft, "%%s%s:%d%c", tp->text, tp->val,
                                              (tp->next ? ',':'/'));
         if (n >= bleft) break;
         pb += n; bleft -= n;
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
        {if (!strncmp(var, "pss.", 4))
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
/*                             C o n f i g N 2 N                              */
/******************************************************************************/

int XrdPssSys::ConfigN2N()
{  
   XrdSysPlugin    *myLib;
   XrdOucName2Name *(*ep)(XrdOucgetName2NameArgs);

// Create a plugin object (we will throw this away without deletion because
// the library must stay open but we never want to reference it again).
// 
   if (!(myLib = new XrdSysPlugin(&eDest, N2NLib))) return 1;

// Now get the entry point of the object creator
// 
   ep = (XrdOucName2Name *(*)(XrdOucgetName2NameArgs))(myLib->getPlugin("XrdOucgetName2Name"));
   if (!ep) return 1;


// Get the Object now
// 
   theN2N = ep(&eDest, ConfigFN, (N2NParms ? N2NParms : ""), NULL, NULL);
   return theN2N == 0;
}

/******************************************************************************/
/*                             C o n f i g X e q                              */
/******************************************************************************/

int XrdPssSys::ConfigXeq(char *var, XrdOucStream &Config)
{

   // Process items. for either a local or a remote configuration
   //
   TS_Xeq("config",        xconf);
   TS_Xeq("origin",        xorig);
   TS_Xeq("setopt",        xsopt);
   TS_Xeq("trace",         xtrac);
   TS_Xeq("namelib",       xnml);

   // No match found, complain.
   //
   eDest.Say("Config warning: ignoring unknown directive '",var,"'.");
   Config.Echo();
   return 0;
}
  
/******************************************************************************/
/*                                 x c o n f                                  */
/******************************************************************************/

/* Function: xconf

   Purpose:  To parse the directive: config <keyword> <value>

             <keyword> is one of the following:
             workers   number of queue workers > 0

   Output: 0 upon success or 1 upon failure.
*/

int XrdPssSys::xconf(XrdSysError *Eroute, XrdOucStream &Config)
{
   char  *val, *kvp;
   int    kval;
   struct Xtab {const char *Key; int *Val;} Xopts[] =
               {{"workers", &Workers}};
   int i, numopts = sizeof(Xopts)/sizeof(struct Xtab);

   if (!(val = Config.GetWord()))
      {Eroute->Emsg("Config", "options argument not specified."); return 1;}

do{for (i = 0; i < numopts; i++) if (!strcmp(Xopts[i].Key, val)) break;

   if (i > numopts)
      Eroute->Say("Config warning: ignoring unknown config option '",val,"'.");
      else {if (!(val = Config.GetWord()))
               {Eroute->Emsg("Config", "config", val, "value not specified.");
                return 1;
               }

            kval = strtol(val, &kvp, 10);
            if (*kvp || !kval)
               {Eroute->Emsg("Config", Xopts[i].Key, 
                             "config value is invalid -", val);
                return 1;
               }
            *(Xopts[i].Val) = kval;
           }
   val = Config.GetWord();
  } while(val && *val);

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

int XrdPssSys::xnml(XrdSysError *Eroute, XrdOucStream &Config)
{
    char *val, parms[1024];

// Get the path
//
   if (!(val = Config.GetWord()) || !val[0])
      {Eroute->Emsg("Config", "namelib not specified"); return 1;}

// Record the path
//
   if (N2NLib) free(N2NLib);
   N2NLib = strdup(val);

// Record any parms
//
   if (!Config.GetRest(parms, sizeof(parms)))
      {Eroute->Emsg("Config", "namelib parameters too long"); return 1;}
   if (N2NParms) free(N2NParms);
   N2NParms = (*parms ? strdup(parms) : 0);
   return 0;
}

/******************************************************************************/
/*                                 x o r i g                                  */
/******************************************************************************/

/* Function: xorig

   Purpose:  Parse: origin <host>[+][:<port>|<port>]

   Output: 0 upon success or !0 upon failure.
*/

int XrdPssSys::xorig(XrdSysError *errp, XrdOucStream &Config)
{
    XrdOucTList *tp = 0;
    char *val, *mval = 0;
    int  i, port;

//  We are looking for regular managers. These are our points of contact
//
    if (!(val = Config.GetWord()))
       {errp->Emsg("Config","origin host name not specified"); return 1;}
       else mval = strdup(val);

// Check if there is a port number. This could be as ':port' or ' port'.
//
    if (!(val = index(mval,':'))) val = Config.GetWord();
       else {*val = '\0'; val++;}

// Validate the port number
//
    if (val)
       {if (isdigit(*val))
            {if (XrdOuca2x::a2i(*errp,"origin port",val,&port,1,65535))
                port = 0;
            }
            else if (!(port = XrdNetDNS::getPort(val, "tcp")))
                    {errp->Emsg("Config", "unable to find tcp service", val);
                     port = 0;
                    }
       } else errp->Emsg("Config","origin port not specified for",mval);

// If port is invalid or missing, fail this
//
    if (!port) {free(mval); return 1;}

// For proxies we need not expand 'host+' spec but need to supress the plus
//
    if ((i = strlen(mval)) > 1 && mval[i-1] == '+') mval[i-1] = 0;

// Check if this is a duplicate, if its new, add to the list
//
   tp = ManList;
   while(tp && (strcmp(tp->text, mval) || tp->val != port)) tp = tp->next;
   if (tp) errp->Emsg("Config","Duplicate origin",mval);
      else ManList = new XrdOucTList(mval, port, ManList);

// All done
//
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

   Output: 0 upon success or !0 upon failure.
*/

int XrdPssSys::xsopt(XrdSysError *Eroute, XrdOucStream &Config)
{
    char  kword[256], *val, *kvp;
    long  kval;
    static const char *Sopts[] =
       {
         "ConnectTimeout",
         "DataServerConn_ttl",
         "DebugLevel",
         "DfltTcpWindowSize",
         "LBServerConn_ttl",
         "ParStreamsPerPhyConn",
         "ParStreamsPerPhyConn",
         "RedirCntTimeout",
         "ReadAheadSize",
         "ReadAheadStrategy",
         "ReadCacheBlkRemPolicy",
         "ReadCacheSize",
         "ReadTrimBlockSize",
         "ReconnectWait",
         "RemoveUsedCacheBlocks",
         "RequestTimeout",
         "TransactionTimeout"
       };
    int i, numopts = sizeof(Sopts)/sizeof(const char *);

    if (!(val = Config.GetWord()))
       {Eroute->Emsg("Config", "setopt keyword not specified"); return 1;}
    strlcpy(kword, val, sizeof(kword));
    if (!(val = Config.GetWord()))
       {Eroute->Emsg("Config", "setopt", kword, "value not specified");
        return 1;
       }

    kval = strtol(val, &kvp, 10);
    if (*kvp)
       {Eroute->Emsg("Config", kword, "setopt keyword value is invalid -", val);
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
       {Eroute->Emsg("Config", "trace option not specified"); return 1;}
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
