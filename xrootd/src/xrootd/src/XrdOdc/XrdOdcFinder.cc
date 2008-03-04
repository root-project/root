/******************************************************************************/
/*                                                                            */
/*                       X r d O d c F i n d e r . c c                        */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

const char *XrdOdcFinderCVSID = "$Id$";

#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>
#include <sys/shm.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/times.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <sys/wait.h>
  
#include "XrdOdc/XrdOdcConfig.hh"
#include "XrdOdc/XrdOdcFinder.hh"
#include "XrdOdc/XrdOdcManager.hh"
#include "XrdOdc/XrdOdcMsg.hh"
#include "XrdOdc/XrdOdcTrace.hh"
#include "XrdOuc/XrdOucEnv.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdOuc/XrdOucErrInfo.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdOuc/XrdOucReqID.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdSys/XrdSysTimer.hh"
#include "XrdNet/XrdNetDNS.hh"
#include "XrdNet/XrdNetOpts.hh"
#include "XrdNet/XrdNetSocket.hh"
#include "XrdSfs/XrdSfsInterface.hh"

/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/

XrdSysError  OdcEDest(0, "odc_");
  
XrdOucTrace  OdcTrace(&OdcEDest);

char        *XrdOdcFinder::OLBPath = 0;

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdOdcFinder::XrdOdcFinder(XrdSysLogger *lp, Persona acting)
{
   OdcEDest.logger(lp);
   myPersona = acting;
}

/******************************************************************************/
/*                         R e m o t e   F i n d e r                          */
/******************************************************************************/
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdOdcFinderRMT::XrdOdcFinderRMT(XrdSysLogger *lp, int whoami)
               : XrdOdcFinder(lp, (whoami & XrdOdcIsProxy 
                                          ? XrdOdcFinder::amProxy
                                          : XrdOdcFinder::amRemote))
{
     myManagers  = 0;
     myManCount  = 0;
     SMode       = 0;
     isTarget    = whoami & XrdOdcIsTarget;
}
 
/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/

XrdOdcFinderRMT::~XrdOdcFinderRMT()
{
    XrdOdcManager *mp, *nmp = myManagers;

    while((mp = nmp)) {nmp = mp->nextManager(); delete mp;}
}

/******************************************************************************/
/*                             C o n f i g u r e                              */
/******************************************************************************/
  
int XrdOdcFinderRMT::Configure(char *cfn)
{
   XrdOdcConfig config(&OdcEDest);

// Set the error dest and simply call the configration object
//
   if (config.Configure(cfn, (myPersona == XrdOdcFinder::amProxy ?
                             "Proxy" : "Remote"), isTarget)) return 0;

// Set configured values and start the managers
//
   OLBPath    = config.OLBPath;
   RepDelay   = config.RepDelay;
   RepNone    = config.RepNone;
   RepWait    = config.RepWait;
   ConWait    = config.ConWait;
   PrepWait   = config.PrepWait;
   if (myPersona == XrdOdcFinder::amProxy)
           {SMode = config.SModeP; StartManagers(config.PanList);}
      else {SMode = config.SMode;  StartManagers(config.ManList);}

// All done
//
   return 1;
}

/******************************************************************************/
/*                               F o r w a r d                                */
/******************************************************************************/

int XrdOdcFinderRMT::Forward(XrdOucErrInfo &Resp, const char *cmd, 
                             const char *arg1, const char *arg2)
{
   int  i;
   XrdOdcManager *Manp;
   struct iovec xmsg[8];

// Make sure we are configured
//
   if (!myManagers)
      {OdcEDest.Emsg("Finder", "Forward() called prior to Configure().");
       Resp.setErrInfo(EINVAL, "Internal error locating file.");
       return -EINVAL;
      }

// Construct a message to be sent to the manager
//
              xmsg[0].iov_base = (char *)"0 "; xmsg[0].iov_len = 2;
              xmsg[1].iov_base = (char *)cmd;  xmsg[1].iov_len = strlen(cmd);
              i = 2;
   if (arg1) {xmsg[i].iov_base = (char *)" ";  xmsg[i++].iov_len = 1;
              xmsg[i].iov_base = (char *)arg1; xmsg[i++].iov_len = strlen(arg1);
             }
   if (arg2) {xmsg[i].iov_base = (char *)" ";  xmsg[i++].iov_len = 1;
              xmsg[i].iov_base = (char *)arg2; xmsg[i++].iov_len = strlen(arg2);
             }
              xmsg[i].iov_base = (char *)"\n"; xmsg[i++].iov_len = 1;

// This may be a 2way message. If so, use the longer path.
//
   if (*cmd == '+') 
      {xmsg[1].iov_base = (char *)cmd+1; xmsg[1].iov_len--;
       return send2Man(Resp, (arg1 ? arg1 : "/"), xmsg, i);
      }

// Select the right manager for this request
//
   if (!(Manp = SelectManager(Resp, (arg1 ? arg1 : "/")))) return 1;

// Send message and simply wait for the reply
//
   if (Manp->Send(xmsg, i)) return 0;

// Indicate client should retry later
//
   Resp.setErrInfo(RepDelay, "");
   return RepDelay;
}
  
/******************************************************************************/
/*                                L o c a t e                                 */
/******************************************************************************/
  
int XrdOdcFinderRMT::Locate(XrdOucErrInfo &Resp, const char *path, int flags,
                            XrdOucEnv *Env)
{
   const char *ptype;
   char *Avoid;
   int   ioveol = 3;
   struct iovec xmsg[8];

// Make sure we are configured
//
   if (!myManagers)
      {OdcEDest.Emsg("Finder", "Locate() called prior to Configure().");
       Resp.setErrInfo(EINVAL, "Internal error locating file.");
       return -EINVAL;
      }

// Check if there is a server we need to avoid (we wish to tell the olb)
//
   if (Env) Avoid = Env->Get("tried");
      else  Avoid = 0;

// Compute command and mode mode:
// selects - requests a cache refresh for <path>
//       c - file will be created
//       d - file will be created or truncated
//       r - file will only be read
//       w - file will be read and writen
//       s - only stat information will be obtained
//       x - only stat information will be obtained (file must be resident)
//       y - locate file at currently know locations (do not wait)
//       z - locate file.
//
        if (flags & SFS_O_CREAT)
           ptype = (flags & (SFS_O_WRONLY | SFS_O_RDWR) && flags & SFS_O_TRUNC
                 ? "d " : "c ");
   else if (flags & (SFS_O_WRONLY | SFS_O_RDWR))
           ptype = (flags & SFS_O_TRUNC ? "t " : "w ");
   else if (flags & SFS_O_LOCATE)
           ptype = (flags & SFS_O_NOWAIT ? "y " : "z ");
   else if (flags & SFS_O_STAT)   ptype = "s ";
   else if (flags & SFS_O_NOWAIT) ptype = "x ";
   else    ptype = "r ";

// Construct a message to be sent to the manager. The first element is filled
// in by send2Man() and is the requestid.
//
   if (flags & SFS_O_RESET)
      {xmsg[1].iov_base = (char *)"selects "; xmsg[1].iov_len = 8;}
      else
      {xmsg[1].iov_base = (char *)"select " ; xmsg[1].iov_len = 7;}
       xmsg[2].iov_base = (char *)ptype;      xmsg[2].iov_len = 2;
   if (Avoid)
      {xmsg[3].iov_base = (char *)"-";        xmsg[3].iov_len = 1;
       xmsg[4].iov_base = Avoid;              xmsg[4].iov_len = strlen(Avoid);
       xmsg[5].iov_base = (char *)" ";        xmsg[5].iov_len = 1;
       ioveol = 6;
      }

   xmsg[ioveol].iov_base = (char *)path;  xmsg[ioveol++].iov_len = strlen(path);
   xmsg[ioveol].iov_base = (char *)"\n";  xmsg[ioveol  ].iov_len = 1;

// Send the 2way message
//
   return send2Man(Resp, path, xmsg, ioveol+1);
}
  
/******************************************************************************/
/*                               P r e p a r e                                */
/******************************************************************************/
  
int XrdOdcFinderRMT::Prepare(XrdOucErrInfo &Resp, XrdSfsPrep &pargs)
{
   EPNAME("Prepare")
   static XrdSysMutex prepMutex;
   char mbuff1[32], mbuff2[32], *mode;
   XrdOucTList *tp;
   int pathloc, plenloc = 0;
   XrdOdcManager *Manp = 0;
   struct iovec iodata[8];

// Make sure we are configured
//
   if (!myManagers)
      {OdcEDest.Emsg("Finder", "Prepare() called prior to Configure().");
       Resp.setErrInfo(EINVAL, "Internal error preparing files.");
       return -EINVAL;
      }

// Check for a cancel request
//
   if (!(tp = pargs.paths))
      {if (!(Manp = SelectManager(Resp, 0))) return ConWait;
       iodata[0].iov_base = (char *)"0 prepdel ";
       iodata[0].iov_len  = 10;    //1234567890
       iodata[1].iov_base = pargs.reqid;
       iodata[1].iov_len  = strlen(pargs.reqid);
       iodata[2].iov_base = (char *)"\n";
       iodata[2].iov_len  = 1;
       if (Manp->Send((const struct iovec *)&iodata, 3)) return 0;
          else {Resp.setErrInfo(RepDelay, "");
                DEBUG("Finder: Failed to send prepare cancel to " <<Manp->Name()
                      <<" reqid=" <<pargs.reqid);
                return RepDelay;
               }
      }

// Decode the options and preset iovec. The format of the message is:
// 0 prepsel <reqid> <notify>-n <prty> <mode> <path>\n
//
   iodata[0].iov_base = (char *)"0 prepadd ";
   iodata[0].iov_len  = 10;       //1234567890
   iodata[1].iov_base = pargs.reqid;
   iodata[1].iov_len  = strlen(pargs.reqid);
   iodata[2].iov_base = (char *)" ";
   iodata[2].iov_len  = 1;
   if (!pargs.notify || !(pargs.opts & Prep_SENDACK))
      {iodata[3].iov_base = (char *)"*";
       iodata[3].iov_len  = 1;
       mode = (char *)" %d %cq ";
      } else {
       iodata[3].iov_base = pargs.notify;
       iodata[3].iov_len  = strlen(pargs.notify);
       plenloc = 4;         // Where the msg is in iodata
       mode = (pargs.opts & Prep_SENDERR ? (char *)"-%%d %d %cn "
                                         : (char *)"-%%d %d %cnq ");
      }
   iodata[4].iov_len  = sprintf(mbuff1, mode, (pargs.opts & Prep_PMASK),
                                (pargs.opts & Prep_WMODE ? 'w' : 'r'));
   iodata[4].iov_base = (plenloc ? mbuff2 : mbuff1);
   pathloc = 5;
   iodata[6].iov_base = (char *)"\n";
   iodata[6].iov_len  = 1;

// Distribute out paths to the various managers
//
   while(tp)
        {if (!(Manp = SelectManager(Resp, tp->text))) break;
         iodata[pathloc].iov_base = tp->text;
         iodata[pathloc].iov_len  = strlen(tp->text);
         if (plenloc) iodata[plenloc].iov_len = 
                      sprintf(mbuff2, mbuff1, tp->val);

         DEBUG("Finder: Sending " <<Manp->Name() <<' ' <<iodata[0].iov_base
                      <<' ' <<iodata[1].iov_base <<' ' <<iodata[3].iov_base
                      <<' ' <<iodata[5].iov_base);

         if (!Manp->Send((const struct iovec *)&iodata, 7)) break;
         if ((tp = tp->next))
            {prepMutex.Lock();
             XrdSysTimer::Wait(PrepWait);
             prepMutex.UnLock();
            }
        }

// Check if all went well
//
   if (!tp) return 0;
   Resp.setErrInfo(RepDelay, "");
   DEBUG("Finder: Failed to send prepare to " <<(Manp ? Manp->Name() : "?")
                  <<" reqid=" <<pargs.reqid);
   return RepDelay;
}

/******************************************************************************/
/*                         S e l e c t M a n a g e r                          */
/******************************************************************************/
  
XrdOdcManager *XrdOdcFinderRMT::SelectManager(XrdOucErrInfo &Resp, 
                                              const char *path)
{
   XrdOdcManager *Womp, *Manp;

// Get where to start
//
   if (SMode != ODC_ROUNDROB || !path) Womp = Manp = myManagers;
      else Womp = Manp = myManTable[XrdOucReqID::Index(myManCount, path)];

// Find the next active server
//
   do {if (Manp->isActive()) return Manp;
      } while((Manp = Manp->nextManager()) != Womp);

// All managers are dead
//
   SelectManFail(Resp);
   return (XrdOdcManager *)0;
}
  
/******************************************************************************/
/*                         S e l e c t M a n F a i l                          */
/******************************************************************************/
  
void XrdOdcFinderRMT::SelectManFail(XrdOucErrInfo &Resp)
{
   EPNAME("SelectManFail")
   static time_t nextMsg = 0;
   time_t now;

// All servers are dead, indicate so every minute
//
   now = time(0);
   myData.Lock();
   if (nextMsg < now)
      {nextMsg = now + 60;
       myData.UnLock();
       OdcEDest.Emsg("Finder", "All managers are disfunctional.");
      } else myData.UnLock();
   Resp.setErrInfo(ConWait, "");
   TRACE(Redirect, "user=" <<Resp.getErrUser() <<" No managers available; wait " <<ConWait);
}
  
/******************************************************************************/
/*                              s e n d 2 M a n                               */
/******************************************************************************/
  
int XrdOdcFinderRMT::send2Man(XrdOucErrInfo &Resp, const char *path,
                              struct iovec *xmsg, int xnum)
{
   EPNAME("send2Man")
   int  val, retc;
   char *cgi, *colon, *msg, idbuff[16];
   XrdOdcMsg *mp;
   XrdOdcManager *Manp;

// Select the right manager for this request
//
   if (!(Manp = SelectManager(Resp, path))) return ConWait;

// Allocate a message object. There is only a fixed number of these and if
// all of them are in use, th client has to wait to prevent over-runs.
//
   if (!(mp = XrdOdcMsg::Alloc(&Resp)))
      {Resp.setErrInfo(RepDelay, "");
       TRACE(Redirect, Resp.getErrUser() <<" no more msg objects; path=" <<path);
       return RepDelay;
      }

// Insert the response ID into the message
//
   xmsg[0].iov_len  = sprintf(idbuff, "%d ", mp->ID());
   xmsg[0].iov_base = idbuff;

// Send message and simply wait for the reply (msg object is locked via Alloc)
//
   if (!Manp->Send(xmsg, xnum) || (mp->Wait4Reply(RepWait)))
      {mp->Recycle();
       Resp.setErrInfo(RepDelay, "");
       Manp->whatsUp();
       TRACE(Redirect, Resp.getErrUser() <<" got no response from "
                       <<Manp->NPfx() <<" path=" <<path);
       return RepDelay;
      }

// A reply was received; process as appropriate
//
   msg = (char *)Resp.getErrText(retc);
   if (retc == -EINPROGRESS) retc = Manp->delayResp(Resp);

        if (retc == -EREMOTE)
           {TRACE(Redirect, Resp.getErrUser() <<" redirected to " <<msg
                  <<" by " << Manp->NPfx() <<" path=" <<path);
            if ( (cgi   = index(msg, (int)'?'))) *cgi = '\0';
            if (!(colon = index(msg, (int)':'))) 
               {val = 0;
                if (cgi) *cgi ='?';
               } else {
                *colon = '\0';
                val = atoi(colon+1);
                if (cgi) {*cgi = '?'; strcpy(colon, cgi);}
               }
            Resp.setErrCode(val);
           }
   else if (retc == -EAGAIN)
           {if (!(retc = atoi(msg))) retc = RepDelay;
            Resp.setErrInfo(retc, "");
            TRACE(Redirect, Resp.getErrUser() <<" asked to wait "
                  <<retc <<" by " << Manp->NPfx() <<" path=" <<path);
           }
   else if (retc == -EINPROGRESS)
           {TRACE(Redirect, Resp.getErrUser() <<" in reply wait by "
                  << Manp->NPfx() <<" path=" <<path);
           }
   else if (retc == -EALREADY)
           {TRACE(Redirect, Resp.getErrUser() <<" given text data '"
                  <<msg <<"' by " << Manp->NPfx() <<" path=" <<path);
            Resp.setErrCode(*msg ? strlen(msg)+1 : 0);
           }
   else if (retc == -EINVAL)
           {TRACE(Redirect, Resp.getErrUser() <<" given error msg '"
                  <<msg <<"' by " << Manp->NPfx() <<" path=" <<path);
           }
   else    {TRACE(Redirect, Resp.getErrUser() <<" given error "
                  <<retc <<" by " << Manp->NPfx() <<" path=" <<path);
           }

// All done
//
   mp->Recycle();
   return retc;
}

/******************************************************************************/
/*                         S t a r t M a n a g e r s                          */
/******************************************************************************/
  
void *XrdOdcStartManager(void *carg)
      {XrdOdcManager *mp = (XrdOdcManager *)carg;
       return mp->Start();
      }

int XrdOdcFinderRMT::StartManagers(XrdOucTList *myManList)
{
   XrdOucTList *tp;
   XrdOdcManager *mp, *firstone = 0;
   int i = 0;
   pthread_t tid;
   char buff[128];

// Clear manager table
//
   memset((void *)myManTable, 0, sizeof(myManTable));

// For each manager, start a thread to handle it
//
   tp = myManList;
   while(tp && i < XRDODCMAXMAN)
        {mp = new XrdOdcManager(&OdcEDest, tp->text, tp->val, ConWait, RepNone);
         myManTable[i] = mp;
         if (myManagers) mp->setNext(myManagers);
            else firstone = mp;
         myManagers = mp;
         if (XrdSysThread::Run(&tid,XrdOdcStartManager,(void *)mp,0,tp->text))
            OdcEDest.Emsg("Config", errno, "start manager");
            else mp->setTID(tid);
         tp = tp->next; i++;
        }

// Check if we exceeded maximum manager count
//
   if (tp) while(tp)
                {OdcEDest.Emsg("Config warning: too many managers; ",tp->text,
                               " ignored.");
                 tp = tp->next;
                }

// Make this a circular chain
//
   if (firstone) firstone->setNext(myManagers);

// Indicate how many managers have been started
//
   sprintf(buff, "%d manager(s) started.", i);
   OdcEDest.Say("Config ", buff);
   myManCount = i;

// All done
//
   return 0;
}
 
/******************************************************************************/
/*                         T a r g e t   F i n d e r                          */
/******************************************************************************/
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdOdcFinderTRG::XrdOdcFinderTRG(XrdSysLogger *lp, int whoami, int port)
               : XrdOdcFinder(lp, XrdOdcFinder::amTarget)
{
   char buff [256];
   isRedir = whoami & XrdOdcIsRedir;
   isProxy = whoami & XrdOdcIsProxy;
   OLBPath = 0;
   OLBp    = new XrdOucStream(&OdcEDest);
   Active  = 0;
   myPort  = port;
   sprintf(buff, "login %c %d port %d\n",(isProxy ? 'P' : 'p'),getpid(),port);
   Login = strdup(buff);
}
 
/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/

XrdOdcFinderTRG::~XrdOdcFinderTRG()
{
  if (OLBp)  delete OLBp;
  if (Login) free(Login);
}
  
/******************************************************************************/
/*                                 A d d e d                                  */
/******************************************************************************/
  
void XrdOdcFinderTRG::Added(const char *path)
{
   char *data[4];
   int   dlen[4];

// Set up to notify the olb domain that a file has been removed
//
   data[0] = (char *)"newfn ";   dlen[0] = 6;
   data[1] = (char *)path;       dlen[1] = strlen(path);
   data[2] = (char *)"\n";       dlen[2] = 1;
   data[3] = 0;                  dlen[3] = 0;

// Now send the notification
//
   myData.Lock();
   if (Active && OLBp->Put((const char **)data, (const int *)dlen))
      {OLBp->Close(); Active = 0;}
   myData.UnLock();
}

/******************************************************************************/
/*                             C o n f i g u r e                              */
/******************************************************************************/
  
void *XrdOdcStartOlb(void *carg)
      {XrdOdcFinderTRG *mp = (XrdOdcFinderTRG *)carg;
       return mp->Start();
      }
  
int XrdOdcFinderTRG::Configure(char *cfn)
{
   XrdOdcConfig config(&OdcEDest);
   pthread_t tid;

// Set the error dest and simply call the configration object
//
   if (config.Configure(cfn, "Target", isRedir)) return 0;
   if (!(OLBPath = config.OLBPath))
      {OdcEDest.Emsg("Config", "Unable to determine olb admin path"); return 0;}

// Start a thread to connect with the local olb
//
   if (XrdSysThread::Run(&tid, XrdOdcStartOlb, (void *)this, 0, "olb i/f"))
      OdcEDest.Emsg("Config", errno, "start olb interface");

// All done
//
   return 1;
}
  
/******************************************************************************/
/*                               R e m o v e d                                */
/******************************************************************************/
  
void XrdOdcFinderTRG::Removed(const char *path)
{
   char *data[4];
   int   dlen[4];

// Set up to notify the olb domain that a file has been removed
//
   data[0] = (char *)"rmdid ";   dlen[0] = 6;
   data[1] = (char *)path;       dlen[1] = strlen(path);
   data[2] = (char *)"\n";       dlen[2] = 1;
   data[3] = 0;                  dlen[3] = 0;

// Now send the notification
//
   myData.Lock();
   if (Active && OLBp->Put((const char **)data, (const int *)dlen))
      {OLBp->Close(); Active = 0;}
   myData.UnLock();
}

/******************************************************************************/
/*                                 S t a r t                                  */
/******************************************************************************/
  
void *XrdOdcFinderTRG::Start()
{
   int   retc;

// First step is to connect to the local server olb
//
   while(1)
        {do {Hookup();

             // Login to the olb
             //
             myData.Lock();
             retc = OLBp->Put(Login);
             myData.UnLock();

             // Put up a read. We don't expect any responses at this point but
             // should the olb die, we will notice and try to reconnect.
             //
             while(OLBp->GetLine()) {}
             break;
            } while(1);
         // The olb went away
         //
         myData.Lock();
         OLBp->Close();
         Active = 0;
         myData.UnLock();
         OdcEDest.Emsg("olb", "Lost contact with olb via", OLBPath);
         XrdSysTimer::Wait(10*1000);
        }

// We should never get here
//
   return (void *)0;
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                                H o o k u p                                 */
/******************************************************************************/
  
void XrdOdcFinderTRG::Hookup()
{
   struct stat buf;
   XrdNetSocket Sock(&OdcEDest);
   int opts = 0, tries = 6;

// Wait for the olb path to be created
//
   while(stat(OLBPath, &buf)) 
        {if (!tries--)
            {OdcEDest.Emsg("olb", "Waiting for olb path", OLBPath); tries=6;}
         XrdSysTimer::Wait(10*1000);
        }

// We can now try to connect
//
   tries = 0;
   while(Sock.Open(OLBPath, -1, opts) < 0)
        {if (!tries--)
            {opts = XRDNET_NOEMSG;
             tries = 6;
            } else if (!tries) opts = 0;
         XrdSysTimer::Wait(10*1000);
        };

// Transfer the socket FD to a stream
//
   myData.Lock();
   Active = 1;
   OLBp->Attach(Sock.Detach());
   myData.UnLock();

// Tell the world
//
   OdcEDest.Emsg("olb", "Connected to olb via", OLBPath);
}
