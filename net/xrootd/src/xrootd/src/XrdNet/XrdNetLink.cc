/******************************************************************************/
/*                                                                            */
/*                         X r d N e t L i n k . c c                          */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//          $Id$

const char *XrdNetLinkCVSID = "$Id$";
  
#include <fcntl.h>
#ifndef WIN32
#include <poll.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#else
#include "XrdSys/XrdWin32.hh"
#endif

#include "XrdNet/XrdNet.hh"
#include "XrdNet/XrdNetBuffer.hh"
#include "XrdNet/XrdNetDNS.hh"
#include "XrdNet/XrdNetLink.hh"
#include "XrdNet/XrdNetPeer.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdOuc/XrdOucStream.hh"
#include "XrdOuc/XrdOucTokenizer.hh"
 
/******************************************************************************/
/*                 S t a t i c   I n i t i a l i z a t i o n                  */
/******************************************************************************/

XrdSysMutex             XrdNetLink::LinkList;
XrdOucStack<XrdNetLink> XrdNetLink::LinkStack;
int                     XrdNetLink::maxlink = 16;
int                     XrdNetLink::numlink = 0;
int                     XrdNetLink::devNull = open("/dev/null", O_RDONLY);
  
/******************************************************************************/
/*                                 A l l o c                                  */
/******************************************************************************/
  
XrdNetLink *XrdNetLink::Alloc(XrdSysError *erp, XrdNet *Net, XrdNetPeer &Peer,
                              XrdNetBufferQ *bq, int opts)
{
   XrdNetLink *lp;

// Lock the data area
//
   LinkList.Lock();
   if ((lp = LinkStack.Pop())) numlink--;
   LinkList.UnLock();

// Either return a new link or an old one
//
   if (!lp) lp = new XrdNetLink(erp, bq);
      else if (bq != lp->BuffQ)
              {if (lp->recvbuff) {lp->recvbuff->Recycle(); lp->recvbuff = 0;}
               if (lp->sendbuff) {lp->sendbuff->Recycle(); lp->sendbuff = 0;}
               lp->BuffQ = bq;
              }

// Unlock the data area return if no link was allocated
//
   if (!lp) return lp;
   lp->noclose = (opts & XRDNETLINK_NOCLOSE);
   lp->isReset = 0;

// Set the buffer pointer for this link
//
  if (Peer.InetBuff) 
     { lp->recvbuff = Peer.InetBuff;
       if (!(lp->Bucket = new XrdOucTokenizer(Peer.InetBuff->data)))
          {lp->Recycle(); lp = 0;}
     } else if (!(opts & XRDNETLINK_NOSTREAM))
               {if (!(lp->Stream = new XrdOucStream(erp)))
                   {lp->Recycle(); lp = 0;}
                   else lp->Stream->Attach(Peer.fd);
               }

// Establish the address and connection type of this link
//
   if (lp) 
      {memcpy((void *)&(lp->InetAddr), (const void *)&Peer.InetAddr,
                                       sizeof(Peer.InetAddr));
       if (Peer.InetName) lp->Lname = strdup(Peer.InetName);
          else lp->Lname = XrdNetDNS::getHostName(Peer.InetAddr);
       lp->Sname = strdup(lp->Lname);
       Net->Trim(lp->Sname);
       lp->FD = Peer.fd;
       }

// Return the link
//
   return lp;
}
  
/******************************************************************************/
/*                                 C l o s e                                  */
/******************************************************************************/
  
int XrdNetLink::Close(int defer)
{

// Make sure no output activity is occuring. Sync on input only if this is not
// a deferred close. Otherwise, we will likely deadlock.
//
   if(!defer) rdMutex.Lock();
              wrMutex.Lock();

// If close is not to be defered until a recycle, then delete appendages and
// close the file descriptor unless it is being shared. Otherwise, substitute
// '/dev/null' for the file descriptor target to scuttle future reads & writes.
//
    if (!defer)
       {if (Stream)   {Stream->Detach(); delete Stream; Stream = 0;}
        if (Bucket)   {delete Bucket; Bucket = 0;}
        if (recvbuff) {recvbuff->Recycle(); recvbuff = 0;}
        if (sendbuff) {sendbuff->Recycle(); sendbuff = 0;}
        if (Lname)    {free(Lname); Lname = 0;}
        if (FD >= 0 && !noclose) close(FD);
        FD = -1;
       } else if (FD >= 0 && !isReset)
                 {dup2(devNull, FD); isReset = 1;}

// All done
//
                wrMutex.UnLock();
    if (!defer) rdMutex.UnLock();
    return 0;
}

/******************************************************************************/
/*                               G e t L i n e                                */
/******************************************************************************/
  
char *XrdNetLink::GetLine()
{
     if (Stream) return Stream->GetLine();
        else if (Bucket) return Bucket->GetLine();
                else if (recvbuff && recvbuff->dlen) return recvbuff->data;
     return 0;
}

/******************************************************************************/
/*                              G e t T o k e n                               */
/******************************************************************************/
  
char *XrdNetLink::GetToken(char **rest)
{
     if (Stream)   return Stream->GetToken(rest);
        else if (Bucket) return Bucket->GetToken(rest);
     return 0;
}
  
char *XrdNetLink::GetToken()
{
     if (Stream)   return Stream->GetToken();
        else if (Bucket) return Bucket->GetToken();
     return 0;
}

/******************************************************************************/
/*                             L a s t E r r o r                              */
/******************************************************************************/
  
int XrdNetLink::LastError()
{
    return (Stream ? Stream->LastError() : 0);
}

/******************************************************************************/
/*                               O K 2 R e c v                                */
/******************************************************************************/
  
int XrdNetLink::OK2Recv(int timeout)
{
   struct pollfd polltab = {FD, POLLIN|POLLRDNORM, 0};
   int retc;

   do {retc = poll(&polltab, 1, timeout);} while(retc < 0 && errno == EINTR);

   return (retc == 1 && (polltab.revents & (POLLIN | POLLRDNORM)));
}

/******************************************************************************/
/*                               R e c y c l e                                */
/******************************************************************************/

void XrdNetLink::Recycle()
{

// Check if we have enough objects, if so, delete ourselves and return
//
   if (numlink >= maxlink) {delete this; return;}
   Close();

// Add the link to the recycle list
//
   LinkList.Lock();
   LinkStack.Push(&LinkLink);
   numlink++;
   LinkList.UnLock();
   return;
}

/******************************************************************************/
/*                                  R e c v                                   */
/******************************************************************************/

int XrdNetLink::Recv(char *Buff, int Blen)
{
   ssize_t rlen;

// Note that we will read only as much as is queued.
//
   rdMutex.Lock();
   do {rlen = read(FD, Buff, Blen);} while(rlen < 0 && errno == EINTR);
   rdMutex.UnLock();

   if (rlen >= 0) return int(rlen);
   eDest->Emsg("Link", errno, "recieve from", Lname);
   return -1;
}

/******************************************************************************/
/*                              R e t T o k e n                               */
/******************************************************************************/
  
void XrdNetLink::RetToken()
{
     if (Stream) Stream->RetToken();
        else if (Bucket) Bucket->RetToken();
}

/******************************************************************************/
/*                                  S e n d                                   */
/******************************************************************************/

int XrdNetLink::Send(const char *Buff, int Blen, int tmo)
{
   int retc;

   if (!Blen && !(Blen = strlen(Buff))) return 0;
   if ('\n' != Buff[Blen-1])
      {const struct iovec iodata[2] = {{IOV_INIT((char *)Buff, Blen)},
                                       {IOV_INIT((char *)"\n", 1)}};
       return Send(iodata, 2, tmo);
      }

   wrMutex.Lock();

   if (tmo >= 0 && !OK2Send(tmo))
      {wrMutex.UnLock(); return -2;}

   if (Stream)
      do {retc = write(FD, Buff, Blen);}
         while (retc < 0 && errno == EINTR);
      else
      do {retc = sendto(FD, (Sokdata_t)Buff, Blen, 0,
                       (struct sockaddr *)&InetAddr, sizeof(InetAddr));}
         while (retc < 0 && errno == EINTR);
   if (retc < 0) return retErr(errno);
   wrMutex.UnLock();
   return 0;
}

int XrdNetLink::Send(const void *Buff, int Blen, int tmo)
{
   int retc;

   wrMutex.Lock();

   if (tmo >= 0 && !OK2Send(tmo))
      {wrMutex.UnLock(); return -2;}

   if (Stream)
      do {retc = write(FD, Buff, Blen);}
         while (retc < 0 && errno == EINTR);
      else
      do {retc = sendto(FD, (Sokdata_t)Buff, Blen, 0,
                       (struct sockaddr *)&InetAddr, sizeof(InetAddr));}
         while (retc < 0 && errno == EINTR);
   if (retc < 0) return retErr(errno);
   wrMutex.UnLock();
   return 0; 
}
  
int XrdNetLink::Send(const char *dest, const char *Buff, int Blen, int tmo)
{
   int retc;
   struct sockaddr destip;

   if (!Blen && !(Blen = strlen(Buff))) return 0;
   if ('\n' != Buff[Blen-1])
      {const struct iovec iodata[2] = {{IOV_INIT((char *)Buff, Blen)},
                                       {IOV_INIT((char *)"\n", 1)}};
       return Send(dest, iodata, 2, tmo);
      }

   if (!XrdNetDNS::Host2Dest(dest, destip))
      {eDest->Emsg("Link", dest, "is unreachable");
       return -1;
      }

   if (Stream)
      {eDest->Emsg("Link", "Unable to send msg to", dest, "on a stream socket");
       return -1;
      }

   wrMutex.Lock();

   if (tmo >= 0 && !OK2Send(tmo, dest))
      {wrMutex.UnLock(); return -2;}

   do {retc = sendto(FD, (Sokdata_t)Buff, Blen, 0,
                    (struct sockaddr *)&destip, sizeof(destip));}
       while (retc < 0 && errno == EINTR);

   if (retc < 0) return retErr(errno, dest);
   wrMutex.UnLock();
   return 0;
}

int XrdNetLink::Send(const struct iovec iov[], int iovcnt, int tmo)
{
   int i, dsz, retc;
   char *bp;

   wrMutex.Lock();

   if (tmo >= 0 && !OK2Send(tmo))
      {wrMutex.UnLock(); return -2;}

   if (Stream)
      do {retc = writev(FD, iov, iovcnt);}
         while (retc < 0 && errno == EINTR);
      else
      {if (!sendbuff && !(sendbuff = BuffQ->Alloc())) return retErr(ENOMEM);
       dsz = sendbuff->BuffSize(); bp = sendbuff->data;
       for (i = 0; i < iovcnt; i++)
           {dsz -= iov[i].iov_len;
            if (dsz < 0) return retErr(EMSGSIZE);
            memcpy((void *)bp,(const void *)iov[i].iov_base,iov[i].iov_len);
            bp += iov[i].iov_len;
           }
       do {retc = sendto(FD, (Sokdata_t)sendbuff->data, (int)(bp-sendbuff->data), 0,
                       (struct sockaddr *)&InetAddr, sizeof(InetAddr));}
           while (retc < 0 && errno == EINTR);
       }

   if (retc < 0) return retErr(errno);
   wrMutex.UnLock();
   return 0;
}

int XrdNetLink::Send(const char *dest,const struct iovec iov[],int iovcnt,int tmo)
{
   int i, dsz, retc;
   char *bp;
   struct sockaddr destip;

   if (!XrdNetDNS::Host2Dest(dest, destip))
      {eDest->Emsg("Link", dest, "is unreachable");
       return -1;
      }

   if (Stream)
      {eDest->Emsg("Link", "Unable to send msg to", dest, "on a stream socket");
       return -1;
      }

   wrMutex.Lock();

   if (tmo >= 0 && !OK2Send(tmo, dest))
      {wrMutex.UnLock(); return -2;}

   if (!sendbuff && !(sendbuff = BuffQ->Alloc())) return retErr(ENOMEM);
   dsz = sendbuff->BuffSize(); bp = sendbuff->data;
   for (i = 0; i < iovcnt; i++)
       {dsz -= iov[i].iov_len;
        if (dsz < 0) return retErr(EMSGSIZE);
        memcpy((void *)bp,(const void *)iov[i].iov_base,iov[i].iov_len);
        bp += iov[i].iov_len;
       }
   do {retc = sendto(FD, (Sokdata_t)sendbuff->data, (int)(bp-sendbuff->data), 0,
                     &destip, sizeof(destip));}
      while (retc < 0 && errno == EINTR);

   if (retc < 0) return retErr(errno, dest);
   wrMutex.UnLock();
   return 0;
}

/******************************************************************************/
/*                                   S e t                                    */
/******************************************************************************/
  
void XrdNetLink::Set(int maxl)
{

// Lock the data area, set max links, unlock and return
//
   LinkList.Lock();
   maxlink = maxl;
   LinkList.UnLock();
   return;
}
 
/******************************************************************************/
/*                               S e t O p t s                                */
/******************************************************************************/
  
void XrdNetLink::SetOpts(int opts)
{

// Set options we care about
//
   if (opts & XRDNETLINK_NOBLOCK) fcntl(FD, F_SETFL, O_NONBLOCK);
}
  
/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                               O K 2 S e n d                                */
/******************************************************************************/
  
int XrdNetLink::OK2Send(int timeout, const char *dest)
{
   struct pollfd polltab = {FD, POLLOUT|POLLWRNORM, 0};
   int retc;

   do {retc = poll(&polltab, 1, timeout);} while(retc < 0 && errno == EINTR);

   if (retc == 0 || !(polltab.revents & (POLLOUT | POLLWRNORM)))
      eDest->Emsg("Link",(dest ? dest : Lname), "is blocked.");
      else if (retc < 0)
              eDest->Emsg("Link",errno,"poll",(dest ? dest : Lname));
              else return 1;
   return 0;
}
  
/******************************************************************************/
/*                                r e t E r r                                 */
/******************************************************************************/
  
int XrdNetLink::retErr(int ecode, const char *dest)
{
   wrMutex.UnLock();
   eDest->Emsg("Link", ecode, "send to", (dest ? dest : Lname));
   return (EWOULDBLOCK == ecode || EAGAIN == ecode ? -2 : -1);
}
