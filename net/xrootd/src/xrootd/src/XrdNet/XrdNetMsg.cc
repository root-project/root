/******************************************************************************/
/*                                                                            */
/*                          X r d N e t M s g . c c                           */
/*                                                                            */
/* (c) 2007 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

#include <sys/poll.h>

#include "XrdNet/XrdNet.hh"
#include "XrdNet/XrdNetDNS.hh"
#include "XrdNet/XrdNetMsg.hh"
#include "XrdNet/XrdNetOpts.hh"
#include "XrdNet/XrdNetPeer.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPlatform.hh"

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdNetMsg::XrdNetMsg(XrdSysError *erp, const char *dest)
{
   XrdNet     myNet(erp);
   XrdNetPeer Peer;

   eDest = erp; DestHN = 0; FD = -1;
   if (dest)
      {if (!XrdNetDNS::Host2Dest(dest, DestIP))
          eDest->Emsg("Msg", "Default", dest, "is unreachable");
          else DestHN = strdup(dest);
      }

    if (!myNet.Relay(Peer, 0, XRDNET_SENDONLY))
       eDest->Emsg("Msg", "Unable top create UDP msg socket.");
       else FD = Peer.fd;
}

/******************************************************************************/
/*                                  S e n d                                   */
/******************************************************************************/
  
int XrdNetMsg::Send(const char *Buff, int Blen, const char *dest, int tmo)
{
   int retc;
   struct sockaddr destip, *dP;

   if (!Blen && !(Blen = strlen(Buff))) return  0;

   if (!dest)
       {if (!DestHN)
           {eDest->Emsg("Msg", "Destination not specified."); return -1;}
        dP = &DestIP; dest = DestHN;
       }
      else if (!XrdNetDNS::Host2Dest(dest, destip))
              {eDest->Emsg("Msg", dest, "is unreachable");    return -1;}
              else dP = &destip;

   if (tmo >= 0 && !OK2Send(tmo, dest)) return 1;

   do {retc = sendto(FD, (Sokdata_t)Buff, Blen, 0, dP, destSZ);}
       while (retc < 0 && errno == EINTR);

   if (retc < 0) return retErr(errno, dest);
   return 0;
}

/******************************************************************************/

int XrdNetMsg::Send(const struct iovec iov[], int iovcnt, 
                    const char  *dest,        int tmo)
{
   char buff[4096], *bp = buff;
   int i, retc, dsz = sizeof(buff);
   struct sockaddr destip, *dP;

   if (!dest)
       {if (!DestHN)
           {eDest->Emsg("Msg", "Destination not specified."); return -1;}
        dP = &DestIP; dest = DestHN;
       }
      else if (!XrdNetDNS::Host2Dest(dest, destip))
              {eDest->Emsg("Msg", dest, "is unreachable");    return -1;}
              else dP = &destip;

   if (tmo >= 0 && !OK2Send(tmo, dest)) return 1;

   for (i = 0; i < iovcnt; i++)
       {dsz -= iov[i].iov_len;
        if (dsz < 0) return retErr(EMSGSIZE, dest);
        memcpy((void *)bp,(const void *)iov[i].iov_base,iov[i].iov_len);
        bp += iov[i].iov_len;
       }

   do {retc = sendto(FD, (Sokdata_t)buff, (int)(bp-buff), 0, dP, destSZ);}
       while (retc < 0 && errno == EINTR);

   if (retc < 0) return retErr(errno, dest);
   return 0;
}
  
/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                               O K 2 S e n d                                */
/******************************************************************************/
  
int XrdNetMsg::OK2Send(int timeout, const char *dest)
{
   struct pollfd polltab = {FD, POLLOUT|POLLWRNORM, 0};
   int retc;

   do {retc = poll(&polltab, 1, timeout);} while(retc < 0 && errno == EINTR);

   if (retc == 0 || !(polltab.revents & (POLLOUT | POLLWRNORM)))
      eDest->Emsg("Msg", "UDP link to", dest, "is blocked.");
      else if (retc < 0)
              eDest->Emsg("Msg",errno,"poll", dest);
              else return 1;
   return 0;
}
  
/******************************************************************************/
/*                                r e t E r r                                 */
/******************************************************************************/
  
int XrdNetMsg::retErr(int ecode, const char *dest)
{
   eDest->Emsg("Msg", ecode, "send to", dest);
   return (EWOULDBLOCK == ecode || EAGAIN == ecode ? 1 : -1);
}
