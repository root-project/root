/******************************************************************************/
/*                                                                            */
/*                        X r d P r o t L o a d . c c                         */
/*                                                                            */
/* (c) 2006 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//        $Id$  

const char *XrdProtLoadCVSID = "$Id$";

#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPlugin.hh"

#include "Xrd/XrdLink.hh"
#include "Xrd/XrdPoll.hh"
#include "Xrd/XrdProtLoad.hh"
#include "Xrd/XrdTrace.hh"
 
/******************************************************************************/
/*                        G l o b a l   O b j e c t s                         */
/******************************************************************************/

extern XrdSysError XrdLog;

extern XrdOucTrace XrdTrace;

XrdProtocol *XrdProtLoad::ProtoWAN[ProtoMax] = {0};
XrdProtocol *XrdProtLoad::Protocol[ProtoMax] = {0};
char        *XrdProtLoad::ProtName[ProtoMax] = {0};
int          XrdProtLoad::ProtPort[ProtoMax] = {0};

int          XrdProtLoad::ProtoCnt = 0;
int          XrdProtLoad::ProtWCnt = 0;

char         *XrdProtLoad::liblist[ProtoMax];
XrdSysPlugin *XrdProtLoad::libhndl[ProtoMax];

int           XrdProtLoad::libcnt = 0;

/******************************************************************************/
/*            C o n s t r u c t o r   a n d   D e s t r u c t o r             */
/******************************************************************************/
  
 XrdProtLoad::XrdProtLoad(int port) :
              XrdProtocol("protocol loader") {myPort = port;}

 XrdProtLoad::~XrdProtLoad() {}
 
/******************************************************************************/
/*                                  L o a d                                   */
/******************************************************************************/

int XrdProtLoad::Load(const char *lname, const char *pname,
                      char *parms, XrdProtocol_Config *pi)
{
   XrdProtocol *xp;
   int i, j, port = pi->Port;
   int wanopt = pi->WANPort;

// Trace this load if so wanted
//
   if (TRACING(TRACE_DEBUG))
      {XrdTrace.Beg("Protocol");
       cerr <<"getting protocol object " <<pname;
       XrdTrace.End();
      }

// First check to see that we haven't exceeded our protocol count
//
   if (ProtoCnt >= ProtoMax)
      {XrdLog.Emsg("Protocol", "Too many protocols have been defined.");
       return 0;
      }

// Obtain an instance of this protocol
//
   if (lname)  xp =    getProtocol(lname, pname, parms, pi);
      else     xp = XrdgetProtocol(pname, parms, pi);
   if (!xp) {XrdLog.Emsg("Protocol","Protocol", pname, "could not be loaded");
             return 0;
            }

// If this is a WAN enabled protocol then add it to the WAN table
//
   if (wanopt) ProtoWAN[ProtWCnt++] = xp;

// Find a port associated slot in the table
//
   for (i = ProtoCnt-1; i >= 0; i--) if (port == ProtPort[i]) break;
   for (j = ProtoCnt-1; j > i; j--)
       {ProtName[j+1] = ProtName[j];
        ProtPort[j+1] = ProtPort[j];
        Protocol[j+1] = Protocol[j];
       }

// Add protocol to our table of protocols
//
   ProtName[j+1] = strdup(pname);
   ProtPort[j+1] = port;
   Protocol[j+1] = xp;
   ProtoCnt++;
   return 1;
}
  
/******************************************************************************/
/*                                  P o r t                                   */
/******************************************************************************/

int XrdProtLoad::Port(const char *lname, const char *pname,
                      char *parms, XrdProtocol_Config *pi)
{
   int port;

// Trace this load if so wanted
//
   if (TRACING(TRACE_DEBUG))
      {XrdTrace.Beg("Protocol");
       cerr <<"getting port from protocol " <<pname;
       XrdTrace.End();
      }

// Obtain the port number to be used by this protocol
//
   if (lname)  port =    getProtocolPort(lname, pname, parms, pi);
      else     port = XrdgetProtocolPort(pname, parms, pi);
   if (port < 0) XrdLog.Emsg("Protocol","Protocol", pname,
                             "port number could not be determined");
   return port;
}
  
/******************************************************************************/
/*                               P r o c e s s                                */
/******************************************************************************/
  
int XrdProtLoad::Process(XrdLink *lp)
{
     XrdProtocol *pp = 0;
     int i;

// Check if this is a WAN lookup or standard lookup
//
   if (myPort < 0)
      {for (i = 0; i < ProtWCnt; i++)
           if ((pp = ProtoWAN[i]->Match(lp))) break;
              else if (lp->isFlawed()) return -1;
      } else {
       for (i = 0; i < ProtoCnt; i++)
           if (myPort == ProtPort[i] && (pp = Protocol[i]->Match(lp))) break;
               else if (lp->isFlawed()) return -1;
      }
   if (!pp) {lp->setEtext("matching protocol not found"); return -1;}

// Now attach the new protocol object to the link
//
   lp->setProtocol(pp);

// Trace this load if so wanted
//                                                x
   if (TRACING(TRACE_DEBUG))
      {XrdTrace.Beg("Protocol");
       cerr <<"matched protocol " <<ProtName[i];
       XrdTrace.End();
      }

// Attach this link to the appropriate poller
//
   if (!XrdPoll::Attach(lp)) {lp->setEtext("attach failed"); return -1;}

// Take a short-cut and process the initial request as a sticky request
//
   return pp->Process(lp);
}
 
/******************************************************************************/
/*                               R e c y c l e                                */
/******************************************************************************/
  
void XrdProtLoad::Recycle(XrdLink *lp, int ctime, const char *reason)
{

// Document non-protocol errors
//
   if (lp && reason)
      XrdLog.Emsg("Protocol", lp->ID, "terminated", reason);
}

/******************************************************************************/
/*                                 S t a t s                                  */
/******************************************************************************/

int XrdProtLoad::Stats(char *buff, int blen, int do_sync)
{
    int i, k, totlen = 0;

    for (i = 0; i < ProtoCnt && (blen > 0 || !buff); i++)
        {k = Protocol[i]->Stats(buff, blen, do_sync);
         totlen += k; buff += k; blen -= k;
        }

    return totlen;
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                           g e t P r o t o c o l                            */
/******************************************************************************/
  
XrdProtocol *XrdProtLoad::getProtocol(const char *lname,
                                      const char *pname,
                                            char *parms,
                              XrdProtocol_Config *pi)
{
   XrdProtocol *(*ep)(const char *, char *, XrdProtocol_Config *);
   void *epvoid;
   int i;

// Find the matching library. It must be here because getPort was already called
//
   for (i = 0; i < libcnt; i++) if (!strcmp(lname, liblist[i])) break;
   if (i >= libcnt)
      {XrdLog.Emsg("Protocol", pname, "was lost during loading", lname);
       return 0;
      }

// Obtain an instance of the protocol object and return it
//
   if (!(epvoid = libhndl[i]->getPlugin("XrdgetProtocol"))) return 0;
   ep = (XrdProtocol *(*)(const char*,char*,XrdProtocol_Config*))epvoid;
   return ep(pname, parms, pi);
}

/******************************************************************************/
/*                       g e t P r o t o c o l P o r t                        */
/******************************************************************************/
  
int XrdProtLoad::getProtocolPort(const char *lname,
                                 const char *pname,
                                       char *parms,
                         XrdProtocol_Config *pi)
{
   int (*ep)(const char *, char *, XrdProtocol_Config *);
   void *epvoid;
   int i;

// See if the library is already opened, if not open it
//
   for (i = 0; i < libcnt; i++) if (!strcmp(lname, liblist[i])) break;
   if (i >= libcnt)
      {if (libcnt >= ProtoMax)
          {XrdLog.Emsg("Protocol", "Too many protocols have been defined.");
           return -1;
          }
       if (!(libhndl[i] = new XrdSysPlugin(&XrdLog, lname))) return -1;
       liblist[i] = strdup(lname);
       libcnt++;
      }

// Get the port number to be used
//
   if (!(epvoid = libhndl[i]->getPlugin("XrdgetProtocolPort", 1))) 
      return (pi->Port < 0 ? 0 : pi->Port);
   ep = (int (*)(const char*,char*,XrdProtocol_Config*))epvoid;
   return ep(pname, parms, pi);
}
