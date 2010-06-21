/******************************************************************************/
/*                                                                            */
/*                       X r d S e c T L a y e r . c c                        */
/*                                                                            */
/*                                                                            */
/* (c) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//       $Id$

#include <sys/types.h>
#include <sys/socket.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <stdio.h>
#include <unistd.h>

#include "XrdOuc/XrdOucErrInfo.hh"
#include "XrdSec/XrdSecTLayer.hh"
#include "XrdSys/XrdSysHeaders.hh"

/******************************************************************************/
/*                         S t a t i c   V a l u e s                          */
/******************************************************************************/

// Some compilers are incapable of optimizing away inline static const char's.
//
const char  XrdSecTLayer::TLayerRR::endData;
const char  XrdSecTLayer::TLayerRR::xfrData;
  
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/

XrdSecTLayer::XrdSecTLayer(const char *pName, Initiator who1st)
             : secTid(0), mySem(0), Starter(who1st), myFD(-1), urFD(-1),
               Tmax(275), Tcur(0), eCode(0), eText(0)
{

// Do the standard stuff
//
   memset((void *)&Hdr, 0, sizeof(Hdr));
   strncpy(Hdr.protName,pName,sizeof(Hdr.protName)-1);
}
  
/******************************************************************************/
/*               C l i e n t   O r i e n t e d   M e t h o d s                */
/******************************************************************************/
/******************************************************************************/
/*                        g e t C r e d e n t i a l s                         */
/******************************************************************************/

XrdSecCredentials *XrdSecTLayer::getCredentials(XrdSecParameters *parm,
                                                XrdOucErrInfo    *einfo)
{
   char Buff[dataSz];
   int Blen = 0, wrLen = 0;
   char *bP, Req = TLayerRR::xfrData;

// If this is the first time call, perform boot-up sequence and start the flow
//
   eDest = einfo;
   if (!parm)
      {if (!bootUp(isClient)) return 0;
       if (Starter == isServer)
          {Hdr.protCode = TLayerRR::xfrData;
           bP = (char *)malloc(hdrSz);
           memcpy(bP, (char *)&Hdr, hdrSz);
           return new XrdSecCredentials(bP, hdrSz);
          }
      } else {
       if (parm->size < hdrSz) 
          {secError("Invalid parms length", EPROTO);
           return 0;
          }
       Req  = ((TLayerRR *)parm->buffer)->protCode;
       wrLen= parm->size - hdrSz;
      }

// Perform required action
// xfrData -> xfrData | endData if socket gets closed
// endData -> endData           if socket still open else protocol error
//
   switch(Req)
         {case TLayerRR::xfrData:
               if (wrLen > 0 && write(myFD, parm->buffer+hdrSz, wrLen) < 0)
                  {secError("Socket write failed", errno); return 0;}
               Blen = Read(myFD, Buff, dataSz);
               if (Blen < 0 && (Blen != -EPIPE) && (Blen != -ECONNRESET))
                  {secError("Socket read failed", -Blen); return 0;}
               break;
          case TLayerRR::endData:
               if (myFD < 0) {secError("Protocol violation", EPROTO); return 0;}
               Blen = -1;
               break;
          default: secError("Unknown parms request", EINVAL); return 0;
         }

// Set correct protocol code based on value in Blen. On the client side we
// check for proper completion upon socket close or when we get endData.
// Note that we apply self-pacing here as well since either side can pace,
//
        if (Blen < 0)      {if (!secDone()) return 0;
                            Blen = 0; Hdr.protCode = TLayerRR::endData;}
   else if (Blen || wrLen) {Tcur = 0; Hdr.protCode = TLayerRR::xfrData;}
   else if (++Tcur <= Tmax)           Hdr.protCode = TLayerRR::xfrData;
   else                    {Tcur = 0; Hdr.protCode = TLayerRR::endData;}

// Return the credentials
//
   bP = (char *)malloc(hdrSz+Blen);
   memcpy(bP, (char *)&Hdr, hdrSz);
   if (Blen) memcpy(bP+hdrSz, Buff, Blen);
   return new XrdSecCredentials(bP, hdrSz+Blen);
}

/******************************************************************************/
/*               S e r v e r   O r i e n t e d   M e t h o d s                */
/******************************************************************************/
  
int XrdSecTLayer::Authenticate  (XrdSecCredentials  *cred,
                                 XrdSecParameters  **parms,
                                 XrdOucErrInfo      *einfo)
{
   char Buff[dataSz];
   int Blen = 0, wrLen;
   char *bP, Req;

// If this is the first time call, perform boot-up sequence and start the flow
//
   eDest = einfo;
   if (myFD < 0 && !bootUp(isServer)) return -1;

// Get the request code
//
   if (cred->size < hdrSz) {secError("Invalid credentials",EBADMSG); return -1;}
   Req  = ((TLayerRR *)cred->buffer)->protCode;
   wrLen= cred->size - hdrSz;

// Perform required action
// xfrData -> xfrData | endData if socket gets closed
// endData -> noresponse
//
   switch(Req)
         {case TLayerRR::xfrData:
               if (wrLen > 0 && write(myFD, cred->buffer+hdrSz, wrLen) < 0)
                  {secError("Socket write failed", errno); return -1;}
               Blen = Read(myFD, Buff, dataSz);
               if (Blen < 0 && (Blen != -EPIPE) && (Blen != -ECONNRESET))
                  {secError("Socket read failed", -Blen); return 0;}
               break;
          case TLayerRR::endData: return (secDone() ? 0 : -1);
          default: secError("Unknown parms request", EINVAL); return -1;
         }

// Set correct protocol code based on value in Blen and wrLen. Note that if
// both are zero then we decrease the pace count and bail if it reaches zero.
// Otherwise, we reset the pace count to it initial value. On the server side,
// we defer the socket drain until we receive a endData notification.
//
        if (Blen < 0)      {Blen = 0; Hdr.protCode = TLayerRR::endData;}
   else if (Blen || wrLen) {Tcur = 0; Hdr.protCode = TLayerRR::xfrData;}
   else if (++Tcur <= Tmax)           Hdr.protCode = TLayerRR::xfrData;
   else                    {Tcur = 0; Hdr.protCode = TLayerRR::endData;}

// Return the credentials
//
   bP = (char *)malloc(hdrSz+Blen);
   memcpy(bP, (char *)&Hdr, hdrSz);
   if (Blen) memcpy(bP+hdrSz, Buff, Blen);
   *parms = new XrdSecParameters(bP, hdrSz+Blen);

   return 1;
}


/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                                b o o t U p                                 */
/******************************************************************************/

void *XrdSecTLayerBootUp(void *carg)
      {XrdSecTLayer *tP = (XrdSecTLayer *)carg;
       tP->secXeq();
       return (void *)0;
      }

/******************************************************************************/
  
int XrdSecTLayer::bootUp(Initiator whoami)
{
   int sv[2];

// Create a socket pair
//
   if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv))
      {secError("Unable to create socket pair", errno); return 0;}
   myFD = sv[0]; urFD = sv[1];
   Responder = whoami;

// Make sure these sockets don't persist
//
   fcntl(myFD, F_SETFD, FD_CLOEXEC);
   fcntl(urFD, F_SETFD, FD_CLOEXEC);

// Start a thread to handle the socket interaction
//
   if (XrdSysThread::Run(&secTid,XrdSecTLayerBootUp,(void *)this, XRDSYSTHREAD_HOLD))
      {int rc = errno;
       close(myFD); myFD = -1;
       close(urFD); urFD = -1;
       secError("Unable to create thread", rc);
       return 0;
      }

// All done
//
   return 1;
}

/******************************************************************************/
/*                                  R e a d                                   */
/******************************************************************************/
  
int XrdSecTLayer::Read(int FD, char *Buff, int rdLen)
{
   struct pollfd polltab = {FD, POLLIN|POLLRDNORM|POLLHUP, 0};
   int retc, xWt, Tlen = 0;

// Read the data. We will employ a self-pacing read schedule where the
// assumptioon is that should a read produce zero bytes after a small amount
// of time then the data supplier needs additional data (i.e., writes) before
// it can supply data to be read. This occurs because certain transport layer
// protocols issue several writes in a row to complete the client/server
// interaction. We cannot use a fixed schedule for this because streams may
// coalesce adjacent writes, sigh.

// Compute the actual poll wait time
//
   if (Tcur) xWt = (Tcur+10)/10;
      else   xWt = 1;

// Now do the interaction
//
   do {do {retc = poll(&polltab, 1, xWt);} while(retc < 0 && errno == EINTR);
       if (retc <= 0) return (retc ? -errno : Tlen);
       do {retc = read(FD, Buff, rdLen);}  while(retc < 0 && errno == EINTR);
       if (retc <= 0) return (retc ? -errno : (Tlen ? Tlen : -EPIPE));
       Tlen += retc; Buff += retc; rdLen -= retc; xWt = 1;
      } while(rdLen > 0);

   return Tlen;
}

/******************************************************************************/
/*                               s e c D o n e                                */
/******************************************************************************/

int XrdSecTLayer::secDone()
{

// First close the socket and wait for thread completion
//
   secDrain();

// Next, check if everything went well
//
   if (!eCode) return 1;

// Diagnose the problem and return failure
//
   secError((eText ? eText : "?"), eCode, 0);
   return 0;
}
  
/******************************************************************************/
/*                              s e c D r a i n                               */
/******************************************************************************/
  
void XrdSecTLayer::secDrain()
{
   if (myFD >= 0)
      {close(myFD); myFD = -1;
       mySem.Wait();
      }
}

/******************************************************************************/
/*                              s e c E r r n o                               */
/******************************************************************************/
  
const char *XrdSecTLayer::secErrno(int rc, char *buff)
{
   sprintf(buff, "err %d", rc);
   return buff;
}

/******************************************************************************/
/*                              s e c E r r o r                               */
/******************************************************************************/
  
void XrdSecTLayer::secError(const char *Msg, int rc, int iserrno)
{
   char buff[32];
   const char *tlist[] = {"XrdSecProtocol", Hdr.protName, ": ", Msg, "; ", 
                          (iserrno ? strerror(rc) : secErrno(rc,buff))
                         };
   int i, n = sizeof(tlist)/sizeof(const char *);

   if (eDest) eDest->setErrInfo(rc, tlist, n);
      else {for (i = 0; i < n; i++) cerr <<tlist[i]; cerr <<endl;}

   secDrain();
}

/******************************************************************************/
/*                                s e c X e q                                 */
/******************************************************************************/

void XrdSecTLayer::secXeq()
{
   XrdOucErrInfo einfo;
   const char *Msg;

// Initiate the protocol
//
   if (Responder == XrdSecTLayer::isClient) secClient(urFD, &einfo);
      else secServer(urFD, &einfo);
// Extract out the completion code
//
   Msg = einfo.getErrText(eCode);
   if (eText) {free(eText); eText = 0;}
   if (eCode) eText = strdup(Msg ? Msg : "Authentication failed");

// Indicate we are done
//
   if (urFD>0) close(urFD);
   urFD = -1;
   mySem.Post();
}
