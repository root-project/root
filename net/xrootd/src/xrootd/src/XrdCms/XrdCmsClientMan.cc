/******************************************************************************/
/*                                                                            */
/*                    X r d C m s C l i e n t M a n . c c                     */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//          $Id$

// Based on: XrdOdcManager.cc,v 1.13 2006/09/26 07:49:14 abh

const char *XrdCmsClientManCVSID = "$Id$";

#include <time.h>

#include "XrdCms/XrdCmsClientMan.hh"
#include "XrdCms/XrdCmsClientMsg.hh"
#include "XrdCms/XrdCmsLogin.hh"
#include "XrdCms/XrdCmsTrace.hh"

#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysTimer.hh"

#include "Xrd/XrdInet.hh"
#include "Xrd/XrdLink.hh"

using namespace XrdCms;
 
/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/
  
extern        XrdInet *XrdXrootdNetwork;

XrdNetBufferQ XrdCmsClientMan::BuffQ(2048,64);

char          XrdCmsClientMan::doDebug   = 0;

char         *XrdCmsClientMan::ConfigFN  = 0;

XrdSysMutex   XrdCmsClientMan::manMutex;

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdCmsClientMan::XrdCmsClientMan(char *host, int port, 
                                 int cw, int nr, int rw, int rd)
                : syncResp(0)
{
   static XrdSysMutex initMutex;
   static int Instance = 0;
   char  *dot;

   Host    = strdup(host);
   if ((dot = index(Host, '.')))
      {*dot = '\0'; HPfx = strdup(Host); *dot = '.';}
      else HPfx = strdup(Host);
   Port    = port;
   Link    = 0;
   Active  = 0;
   Silent  = 0;
   Suspend = 1;
   RecvCnt = 0;
   nrMax   = nr;
   NetBuff = BuffQ.Alloc();
   repWMax = rw;
   repWait = 0;
   minDelay= rd;
   maxDelay= rd*3;
   chkCount= chkVal;
   lastUpdt= lastTOut = time(0);

// Compute dally value
//
   dally = cw / 2 - 1;
   if (dally < 3) dally = 3;
      else if (dally > 10) dally = 10;

// Provide a unique mask number for this manager
//
   initMutex.Lock();
   manMask = 1<<Instance++;
   initMutex.UnLock();
}

/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/

XrdCmsClientMan::~XrdCmsClientMan()
{
  if (Link)    Link->Close();
  if (Host)    free(Host);
  if (HPfx)    free(HPfx);
  if (NetBuff) NetBuff->Recycle();
}
  
/******************************************************************************/
/*                             d e l a y R e s p                              */
/******************************************************************************/
  
int XrdCmsClientMan::delayResp(XrdOucErrInfo &Resp)
{
   XrdCmsResp *rp;
   int msgid;

// Obtain the message ID
//
   if (!(msgid = Resp.getErrInfo()))
      {Say.Emsg("Manager", Host, "supplied invalid waitr msgid");
       Resp.setErrInfo(0, "redirector protocol error");
       syncResp.Post();
       return -EINVAL;
      }

// Allocate a delayed response object
//
   if (!(rp = XrdCmsResp::Alloc(&Resp, msgid)))
      {Say.Emsg("Manager",ENOMEM,"allocate resp object for",Resp.getErrUser());
       Resp.setErrInfo(0, "0");
       syncResp.Post();
       return -EAGAIN;
      }

// Add this object to our delayed response queue. If the manager bounced then
// purge all of the pending repsonses to avoid sending wrong ones.
//
   if (msgid < maxMsgID) RespQ.Purge();
   maxMsgID = msgid;
   RespQ.Add(rp);

// Tell client to wait for response. The semaphore post allows the manager
// to get the next message from the cmsd. This prevents us from getting the
// delayed response before the response object is added to the queue.
//
   Resp.setErrInfo(0, "");
   syncResp.Post();
   return -EINPROGRESS;
}

/******************************************************************************/
/*                                  S e n d                                   */
/******************************************************************************/
  
int XrdCmsClientMan::Send(char *msg, int mlen)
{
   int allok = 0;

// Determine message length
//
   if (!mlen) mlen = strlen(msg);

// Send the request
//
   if (Active)
      {myData.Lock();
       if (Link)
          {if (!(allok = Link->Send(msg, mlen) > 0))
              {Active = 0;
               Link->Close(1);
              } else SendCnt++;
          }
       myData.UnLock();
      }

// All done
//
   return allok;
}

/******************************************************************************/
  
int XrdCmsClientMan::Send(const struct iovec *iov, int iovcnt, int iotot)
{
   int allok = 0;

// Send the request
//
   if (Active)
      {myData.Lock();
       if (Link)
          {if (!(allok = Link->Send(iov, iovcnt, iotot) > 0))
              {Active = 0;
               Link->Close(1);
              } else SendCnt++;
          }
       myData.UnLock();
      }

// All done
//
   return allok;
}

/******************************************************************************/
/*                                 S t a r t                                  */
/******************************************************************************/
  
void *XrdCmsClientMan::Start()
{

// First step is to connect to the manager
//
   do {Hookup();
       // Now simply start receiving messages on the stream. When we get a
       // respwait reply then we must be assured that the object representing
       // the request is added to the queue before the actual reply arrives.
       // We do this by waiting on syncResp which is posted once the request
       // object is fully processed. The actual response associated with the
       // respwait is synchronized during the callback phase since the client
       // must receive the respwait before the subsequent response.
       //
       while(Receive())
                 if (Response.modifier & CmsResponse::kYR_async) relayResp();
            else if (Response.rrCode == kYR_status) setStatus();
            else if (XrdCmsClientMsg::Reply(HPfx, Response, NetBuff))
                    {if (Response.rrCode == kYR_waitresp) syncResp.Wait();}

       // Tear down the connection
       //
       myData.Lock();
       if (Link) {Link->Close(); Link = 0;}
       Active = 0; Suspend = 1;
       myData.UnLock();

       // Indicate the problem
       //
       Say.Emsg("ClientMan", "Disconnected from", Host);
       XrdSysTimer::Snooze(dally);
      } while(1);

// We should never get here
//
   return (void *)0;
}

/******************************************************************************/
/*                               w h a t s U p                                */
/******************************************************************************/
  
int  XrdCmsClientMan::whatsUp(const char *user, const char *path)
{
   EPNAME("whatsUp");
   int theDelay, inQ;

// The cmsd did not respond. Increase silent count and see if restart is needed
// Otherwise, increase the wait interval just in case things are just slow.
//
   myData.Lock();
   if (Active)
      {if (Active == RecvCnt)
          {if ((time(0)-lastTOut) >= repWait)
              {Silent++;
               if (Silent > nrMax)
                  {Active = 0; Silent = 0; Suspend = 1;
                   if (Link) Link->Close(1);
                  } else if (Silent & 0x02 && repWait < repWMax) repWait++;
              }
          } else {Active = RecvCnt; Silent = 0; lastTOut = time(0);}
      }

// Calclulate how long to delay the client. This will be based on the number
// of outstanding requests bounded by the config delay value.
//
   inQ = SendCnt - RecvCnt;
   theDelay = inQ * qTime;
   myData.UnLock();
   theDelay = theDelay/1000 + (theDelay % 1000 ? 1 : 0);
   if (theDelay < minDelay) return minDelay;
   if (theDelay > maxDelay) return maxDelay;

// Do Some tracing here
//
   TRACE(Redirect, user <<" no resp from " <<HPfx  <<"; inQ " <<inQ <<" wait " <<theDelay <<" path=" <<path);
   return theDelay;
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                                H o o k u p                                 */
/******************************************************************************/
  
int XrdCmsClientMan::Hookup()
{
   EPNAME("Hookup");
   CmsLoginData Data;
   XrdLink *lp;
   char buff[256];
   int rc, oldWait, tries = 12, opts = 0;

// Turn off our debugging and version flags
//
   manMutex.Lock();
   doDebug    &= ~manMask;
   manMutex.UnLock();

// Keep trying to connect to the manager
//
   do {while(!(lp = XrdXrootdNetwork->Connect(Host, Port, opts)))
            {XrdSysTimer::Snooze(dally);
             if (tries--) opts = XRDNET_NOEMSG;
                else     {opts = 0; tries = 12;}
             continue;
            }
       memset(&Data, 0, sizeof(Data));
       Data.Mode = CmsLoginData::kYR_director;
       Data.HoldTime = static_cast<int>(getpid());
       if (!(rc = XrdCmsLogin::Login(lp, Data))) break;
       lp->Close();
       XrdSysTimer::Snooze(dally);
      } while(1);

// Establish global state
//
   manMutex.Lock();
   doDebug |= (Data.Mode & CmsLoginData::kYR_debug ? manMask : 0);
   manMutex.UnLock();

// All went well, finally
//
   myData.Lock();
   Link     = lp;
   Active   = 1;
   Silent   = 0;
   RecvCnt  = 1;
   SendCnt  = 1;
   Suspend  = (Data.Mode & CmsLoginData::kYR_suspend);

// Calculate how long we will wait for replies before delaying the client.
// This is computed dynamically based on the expected response window.
//
   if ((oldWait = (repWait*20/100)) < 2) oldWait = 2;
   if (Data.HoldTime > repWMax*1000) repWait = repWMax;
      else if (Data.HoldTime <= 0)   repWait = repWMax;
              else {repWait = Data.HoldTime*3;
                    repWait = (repWait/1000) + (repWait % 1000 ? 1 : 0);
                    if (repWait > repWMax) repWait = repWMax;
                       else if (repWait < oldWait) repWait = oldWait;
                   }
   qTime = (Data.HoldTime < 100 ? 100 : Data.HoldTime);
   lastTOut = time(0);
   myData.UnLock();

// Tell the world
//
   sprintf(buff, "v %d", Data.Version);
   Say.Emsg("ClientMan", (Suspend ? "Connected to suspended" : "Connected to"),
                         Host, buff);
   DEBUG(Host <<" qt=" <<qTime <<"ms rw=" <<repWait);
   return 1;
}

/******************************************************************************/
/*                               R e c e i v e                                */
/******************************************************************************/
  
int XrdCmsClientMan::Receive()
{
// This method is always run out of the object's main thread. Other threads
// may call methods that initiate a link reset via a deferred close. We will
// notice that here because the file descriptor will be closed. This will
// cause us to return an error and precipitate a connection teardown.
//
   EPNAME("Receive")
   if (Link->RecvAll((char *)&Response, sizeof(Response)) > 0)
      {int dlen = static_cast<int>(ntohs(Response.datalen));
       RecvCnt++; NetBuff->dlen = dlen;
       DEBUG(Link->Name() <<' ' <<dlen <<" bytes on " <<Response.streamid);
       if (!dlen) return 1;
       if (dlen > NetBuff->BuffSize())
          Say.Emsg("ClientMan", "Excessive msg length from", Host);
          else return Link->RecvAll(NetBuff->data, dlen);
      }
   return 0;
}

/******************************************************************************/
/*                             r e l a y R e s p                              */
/******************************************************************************/
  
void XrdCmsClientMan::relayResp()
{
   EPNAME("relayResp");
   XrdCmsResp *rp;

// Remove the response object from our queue.
//
   if (!(rp = RespQ.Rem(Response.streamid)))
      {DEBUG(Host <<" replied to non-existent request; id=" <<Response.streamid);
       return;
      }

// Queue the request for reply (this transfers the network buffer)
//
   rp->Reply(HPfx, Response, NetBuff);

// Obtain a new network buffer
//
   NetBuff = BuffQ.Alloc();
}

/******************************************************************************/
/*                             c h k S t a t u s                              */
/******************************************************************************/

void XrdCmsClientMan::chkStatus()
{
   static CmsUpdateRequest Updt = {{0, kYR_update, 0, 0}};
   time_t nowTime;

// Count down the query count and ask again every 30 seconds
//
   myData.Lock();
   if (!chkCount--)
      {chkCount = chkVal;
       nowTime = time(0);
       if ((nowTime - lastUpdt) >= 30)
          {lastUpdt = nowTime;
           if (Active) Link->Send((char *)&Updt, sizeof(Updt));
          }
      }
   myData.UnLock();
}
  
/******************************************************************************/
/*                             s e t S t a t u s                              */
/******************************************************************************/

void XrdCmsClientMan::setStatus()
{
   EPNAME("setStatus");
   const char *State = 0, *Event = "?";


   myData.Lock();
   if (Response.modifier & CmsStatusRequest::kYR_Suspend)
      {Event = "suspend";
       if (!Suspend) {Suspend = 1; State = "suspended";}
      }
      else if (Response.modifier & CmsStatusRequest::kYR_Resume)
              {Event = "resume";
               if (Suspend) {Suspend = 0; State = "resumed";}
              }
   myData.UnLock();

   DEBUG(Host <<" sent " <<Event <<" event");
   if (State) Say.Emsg("setStatus", "Manager", Host, State);
}
