/******************************************************************************/
/*                                                                            */
/*                            X r d L i n k . c c                             */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//           $Id$

const char *XrdLinkCVSID = "$Id$";

#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/uio.h>

#ifdef __linux__
#include <netinet/tcp.h>
#if !defined(TCP_CORK)
#undef HAVE_SENDFILE
#endif
#endif

#ifdef HAVE_SENDFILE

#ifndef __macos__
#include <sys/sendfile.h>
#endif

#endif

#include "XrdNet/XrdNetDNS.hh"
#include "XrdNet/XrdNetPeer.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPlatform.hh"

#include "Xrd/XrdBuffer.hh"
#include "Xrd/XrdLink.hh"
#include "Xrd/XrdInet.hh"
#include "Xrd/XrdPoll.hh"
#include "Xrd/XrdScheduler.hh"
#define  TRACELINK this
#include "Xrd/XrdTrace.hh"
 
/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/
/******************************************************************************/
/*                    C l a s s   x r d _ L i n k S c a n                     */
/******************************************************************************/
  
class XrdLinkScan : XrdJob
{
public:

void          DoIt() {idleScan();}

              XrdLinkScan(int im, int it, const char *lt="idle link scan") :
                                           XrdJob(lt)
                          {idleCheck = im; idleTicks = it;}
             ~XrdLinkScan() {}

private:

             void   idleScan();

             int    idleCheck;
             int    idleTicks;

static const char *TraceID;
};
  
/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/
  
extern XrdSysError     XrdLog;

extern XrdScheduler    XrdSched;

extern XrdInet        *XrdNetTCP;
extern XrdOucTrace     XrdTrace;

#if defined(HAVE_SENDFILE)
       int             XrdLink::sfOK = 1;
#else
       int             XrdLink::sfOK = 0;
#endif

       XrdLink       **XrdLink::LinkTab;
       char           *XrdLink::LinkBat;
       unsigned int    XrdLink::LinkAlloc;
       int             XrdLink::LTLast = -1;
       XrdSysMutex     XrdLink::LTMutex;

       const char     *XrdLink::TraceID = "Link";

       long long       XrdLink::LinkBytesIn   = 0;
       long long       XrdLink::LinkBytesOut  = 0;
       long long       XrdLink::LinkConTime   = 0;
       long long       XrdLink::LinkCountTot  = 0;
       int             XrdLink::LinkCount     = 0;
       int             XrdLink::LinkCountMax  = 0;
       int             XrdLink::LinkTimeOuts  = 0;
       int             XrdLink::LinkStalls    = 0;
       int             XrdLink::LinkSfIntr    = 0;
       XrdSysMutex     XrdLink::statsMutex;

       const char     *XrdLinkScan::TraceID = "LinkScan";
       int             XrdLink::devNull = open("/dev/null", O_RDONLY);
       short           XrdLink::killWait= 3;  // Kill then wait
       short           XrdLink::waitKill= 4;  // Wait then kill

// The following values are defined for LinkBat[]. We assume that FREE is 0
//
#define XRDLINK_FREE 0x00
#define XRDLINK_USED 0x01
#define XRDLINK_IDLE 0x02
  
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdLink::XrdLink() : XrdJob("connection"), IOSemaphore(0, "link i/o")
{
  Etext = 0;
  HostName = 0;
  Reset();
}

void XrdLink::Reset()
{
  FD    = -1;
  if (Etext)    {free(Etext); Etext = 0;}
  if (HostName) {free(HostName); HostName = 0;}
  Uname[sizeof(Uname)-1] = '@';
  Uname[sizeof(Uname)-2] = '?';
  Lname[0] = '?';
  Lname[1] = '\0';
  ID       = &Uname[sizeof(Uname)-2];
  Comment  = ID;
  Next     = 0;
  Protocol = 0; 
  ProtoAlt = 0;
  conTime  = time(0);
  stallCnt = stallCntTot = 0;
  tardyCnt = tardyCntTot = 0;
  InUse    = 1;
  Poller   = 0; 
  PollEnt  = 0;
  isEnabled= 0;
  isIdle   = 0;
  inQ      = 0;
  tBound   = 0;
  BytesOut = BytesIn = BytesOutTot = BytesInTot = 0;
  doPost   = 0;
  LockReads= 0;
  KeepFD   = 0;
  udpbuff  = 0;
  Instance = 0;
  KillcvP  = 0;
  KillCnt  = 0;
}

/******************************************************************************/
/*                                 A l l o c                                  */
/******************************************************************************/
  
XrdLink *XrdLink::Alloc(XrdNetPeer &Peer, int opts)
{
   static XrdSysMutex  instMutex;
   static unsigned int myInstance = 1;
   XrdLink *lp;
   char *unp, buff[16];
   int bl;

// Make sure that the link slot is available
//
   LTMutex.Lock();
   if (LinkBat[Peer.fd])
      {LTMutex.UnLock();
       XrdLog.Emsg("Link", "attempt to reuse active link");
       return (XrdLink *)0;
      }

// Check if we already have a link object in this slot. If not, allocate
// a quantum of link objects and put them in the table.
//
   if (!(lp = LinkTab[Peer.fd]))
      {unsigned int i;
       XrdLink **blp, *nlp = new XrdLink[LinkAlloc]();
       if (!nlp)
          {LTMutex.UnLock();
           XrdLog.Emsg("Link", ENOMEM, "create link"); 
           return (XrdLink *)0;
          }
       blp = &LinkTab[Peer.fd/LinkAlloc*LinkAlloc];
       for (i = 0; i < LinkAlloc; i++, blp++) *blp = &nlp[i];
       lp = LinkTab[Peer.fd];
      }
      else lp->Reset();
   LinkBat[Peer.fd] = XRDLINK_USED;
   if (Peer.fd > LTLast) LTLast = Peer.fd;
   LTMutex.UnLock();

// Establish the instance number of this link. This is will prevent us from
// sending asynchronous responses to the wrong client when the file descriptor
// gets reused for connections to the same host.
//
   instMutex.Lock();
   lp->Instance = myInstance++;
   instMutex.UnLock();

// Establish the address and connection type of this link
//
   memcpy((void *)&(lp->InetAddr), (const void *)&Peer.InetAddr,
          sizeof(struct sockaddr));
   if (Peer.InetName) strlcpy(lp->Lname, Peer.InetName, sizeof(lp->Lname));
      else {char *host = XrdNetDNS::getHostName(Peer.InetAddr);
            strlcpy(lp->Lname, host, sizeof(lp->Lname));
            free(host);
           }
   lp->HostName = strdup(lp->Lname);
   lp->HNlen = strlen(lp->HostName);
   XrdNetTCP->Trim(lp->Lname);
   bl = sprintf(buff, "?:%d", Peer.fd);
   unp = lp->Lname - bl - 1;
   strncpy(unp, buff, bl);
   lp->ID = unp;
   lp->FD = Peer.fd;
   lp->udpbuff = Peer.InetBuff;
   lp->Comment = (const char *)unp;

// Set options as needed
//
   lp->LockReads = (0 != (opts & XRDLINK_RDLOCK));
   lp->KeepFD    = (0 != (opts & XRDLINK_NOCLOSE));

// Return the link
//
   statsMutex.Lock();
   LinkCountTot++;
   if (LinkCountMax == LinkCount++) LinkCountMax = LinkCount;
   statsMutex.UnLock();
   return lp;
}
  
/******************************************************************************/
/*                                  B i n d                                   */
/******************************************************************************/
  
void XrdLink::Bind(pthread_t tid)
{

// For bind operations, it's quite simple
//
   TID = tid; 
   tBound = 1; 
}

/******************************************************************************/

void XrdLink::Bind()
{
#ifdef __linux__
   pthread_t curTID = (tBound ? TID : XrdSysThread::ID());
#endif

// For unbind operations, we need to do some additional work. This is specific
// to Linux. See the discussion under defered close in the Close() method.
//
   if (tBound)
      {tBound = 0;
#ifdef __linux__
       if (!XrdSysThread::Same(curTID, XrdSysThread::ID()))
          {XrdSysThread::Signal(curTID, SIGSTOP);
           XrdSysThread::Signal(curTID, SIGCONT);
          }
#endif
      }
}

/******************************************************************************/
/*                                C l i e n t                                 */
/******************************************************************************/
  
int XrdLink::Client(char *nbuf, int nbsz)
{
   int ulen;

// Generate full client name
//
   if (nbsz <= 0) return 0;
   ulen = (Lname - ID);
   if ((ulen + HNlen) >= nbsz) ulen = 0;
      else {strncpy(nbuf, ID, ulen);
            strcpy(nbuf+ulen, HostName);
            ulen += HNlen;
           }
   return ulen;
}

/******************************************************************************/
/*                                 C l o s e                                  */
/******************************************************************************/
  
int XrdLink::Close(int defer)
{   int csec, fd, rc = 0;

// If a defer close is requested, we can close the descriptor but we must
// keep the slot number to prevent a new client getting the same fd number.
// Linux is peculiar in that any in-progress operations will remain in that
// state even after the FD is closed unless there is some activity either on
// the connection or an event occurs that causes an operation restart. We
// accomplish this in Linux by stopping and then starting the thread that may
// be bound to this link (see Bind()). Ugly, but that's what happens in Linux.
// We also add a bit of portability by issuing a shutdown() on the socket prior
// closing it. On most platforms, this informs readers that the connection is
// gone (though not on Linux, sigh).
//
   opMutex.Lock();
   if (defer)
      {TRACEI(DEBUG, "Closing FD only");
       if (FD > 1)
          {fd = FD; FD = -FD; csec = Instance; Instance = 0;
           if (!KeepFD)
              {shutdown(fd, SHUT_RDWR);
               if (dup2(devNull, fd) < 0)
                  {FD = fd; Instance = csec;
                   XrdLog.Emsg("Link",errno,"close FD for",ID);
                  } else Bind();
              }
          }
       opMutex.UnLock();
       return 0;
      }

// Multiple protocols may be bound to this link. If it is in use, defer the
// actual close until the use count drops to one.
//
   while(InUse > 1)
      {opMutex.UnLock();
       TRACEI(DEBUG, "Close defered, use count=" <<InUse);
       Serialize();
       opMutex.Lock();
      }
   InUse--;
   Instance = 0;

// Add up the statistic for this link
//
   syncStats(&csec);

// Clean this link up
//
   if (Protocol) {Protocol->Recycle(this, csec, Etext); Protocol = 0;}
   if (ProtoAlt) {ProtoAlt->Recycle(this, csec, Etext); ProtoAlt = 0;}
   if (udpbuff)  {udpbuff->Recycle();  udpbuff  = 0;}
   if (Etext) {free(Etext); Etext = 0;}
   InUse    = 0;

// At this point we can have no lock conflicts, so if someone is waiting for
// us to terminate let them know about it. Note that we will get the condvar
// mutex while we hold the opMutex. This is the required order! We will also
// zero out the pointer to the condvar while holding the opmutex.
//
   if (KillcvP) {KillcvP->  Lock(); KillcvP->Signal();
                 KillcvP->UnLock(); KillcvP = 0;
                }

// Remove ourselves from the poll table and then from the Link table. We may
// not hold on to the opMutex when we acquire the LTMutex. However, the link
// table needs to be cleaned up prior to actually closing the socket. So, we
// do some fancy footwork to prevent multiple closes of this link.
//
   fd = (FD < 0 ? -FD : FD);
   if (FD != -1)
      {if (Poller) {XrdPoll::Detach(this); Poller = 0;}
       FD = -1;
       opMutex.UnLock();
       LTMutex.Lock();
       LinkBat[fd] = XRDLINK_FREE;
       if (fd == LTLast) while(LTLast && !(LinkBat[LTLast])) LTLast--;
       LTMutex.UnLock();
      } else opMutex.UnLock();

// Close the file descriptor if it isn't being shared. Do it as the last
// thing because closes and accepts and not interlocked.
//
   if (fd >= 2) {if (KeepFD) rc = 0;
                    else rc = (close(fd) < 0 ? errno : 0);
                }
   if (rc) XrdLog.Emsg("Link", rc, "close", ID);
   return rc;
}

/******************************************************************************/
/*                                  D o I t                                   */
/******************************************************************************/
 
void XrdLink::DoIt()
{
   int rc;

// The Process() return code tells us what to do:
// < 0 -> Stop getting requests, 
//        -EINPROGRESS leave link disabled but otherwise all is well
//        -n           Error, disable and close the link
// = 0 -> OK, get next request, if allowed, o/w enable the link
// > 0 -> Slow link, stop getting requests  and enable the link
//
   if (Protocol)
      do {rc = Protocol->Process(this);} while (!rc && XrdSched.canStick());
      else {XrdLog.Emsg("Link", "Dispatch on closed link", ID);
            return;
           }

// Either re-enable the link and cycle back waiting for a new request, leave
// disabled, or terminate the connection.
//
   if (rc >= 0) {if (Poller) Poller->Enable(this);}
      else if (rc != -EINPROGRESS) Close();
}
  
/******************************************************************************/
/*                                  F i n d                                   */
/******************************************************************************/

// Warning: curr must be set to a value of 0 or less on the initial call and
//          not touched therafter unless a null pointer is returned. When an
//          actual link object pointer is returned, it's refcount is increased.
//          The count is automatically decreased on the next call to Find().
//
XrdLink *XrdLink::Find(int &curr, XrdLinkMatch *who)
{
   XrdLink *lp;
   const int MaxSeek = 16;
   unsigned int myINS;
   int i, seeklim = MaxSeek;

// Do initialization
//
   LTMutex.Lock();
   if (curr >= 0 && LinkTab[curr]) LinkTab[curr]->setRef(-1);
      else curr = -1;

// Find next matching link. Since this may take some time, we periodically
// release the LTMutex lock which drives up overhead but will still allow
// other critical operations to occur.
//
   for (i = curr+1; i <= LTLast; i++)
       {if ((lp = LinkTab[i]) && LinkBat[i] && lp->HostName)
           if (!who 
           ||   who->Match(lp->ID,lp->Lname-lp->ID-1,lp->HostName,lp->HNlen))
              {myINS = lp->Instance;
               LTMutex.UnLock();
               lp->setRef(1);
               curr = i;
               if (myINS == lp->Instance) return lp;
               LTMutex.Lock();
              }
        if (!seeklim--) {LTMutex.UnLock(); seeklim = MaxSeek; LTMutex.Lock();}
       }

// Done scanning the table
//
    LTMutex.UnLock();
    curr = -1;
    return 0;
}

/******************************************************************************/
/*                               g e t N a m e                                */
/******************************************************************************/

// Warning: curr must be set to a value of 0 or less on the initial call and
//          not touched therafter unless null is returned. Returns the length
//          the name in nbuf.
//
int XrdLink::getName(int &curr, char *nbuf, int nbsz, XrdLinkMatch *who)
{
   XrdLink *lp;
   const int MaxSeek = 16;
   int i, ulen = 0, seeklim = MaxSeek;

// Find next matching link. Since this may take some time, we periodically
// release the LTMutex lock which drives up overhead but will still allow
// other critical operations to occur.
//
   LTMutex.Lock();
   for (i = curr+1; i <= LTLast; i++)
       {if ((lp = LinkTab[i]) && LinkBat[i] && lp->HostName)
           if (!who 
           ||   who->Match(lp->ID,lp->Lname-lp->ID-1,lp->HostName,lp->HNlen))
              {ulen = lp->Client(nbuf, nbsz);
               LTMutex.UnLock();
               curr = i;
               return ulen;
              }
        if (!seeklim--) {LTMutex.UnLock(); seeklim = MaxSeek; LTMutex.Lock();}
       }
   LTMutex.UnLock();

// Done scanning the table
//
   curr = -1;
   return 0;
}

/******************************************************************************/
/*                                  P e e k                                   */
/******************************************************************************/
  
int XrdLink::Peek(char *Buff, int Blen, int timeout)
{
   XrdSysMutexHelper theMutex;
   struct pollfd polltab = {FD, POLLIN|POLLRDNORM, 0};
   ssize_t mlen;
   int retc;

// Lock the read mutex if we need to, the helper will unlock it upon exit
//
   if (LockReads) theMutex.Lock(&rdMutex);

// Wait until we can actually read something
//
   isIdle = 0;
   do {retc = poll(&polltab, 1, timeout);} while(retc < 0 && errno == EINTR);
   if (retc != 1)
      {if (retc == 0) return 0;
       return XrdLog.Emsg("Link", -errno, "poll", ID);
      }

// Verify it is safe to read now
//
   if (!(polltab.revents & (POLLIN|POLLRDNORM)))
      {XrdLog.Emsg("Link", XrdPoll::Poll2Text(polltab.revents),
                           "polling", ID);
       return -1;
      }

// Do the peek.
//
   do {mlen = recv(FD, Buff, Blen, MSG_PEEK);}
      while(mlen < 0 && errno == EINTR);

// Return the result
//
   if (mlen >= 0) return int(mlen);
   XrdLog.Emsg("Link", errno, "peek on", ID);
   return -1;
}
  
/******************************************************************************/
/*                                  R e c v                                   */
/******************************************************************************/
  
int XrdLink::Recv(char *Buff, int Blen)
{
   ssize_t rlen;

// Note that we will read only as much as is queued. Use Recv() with a
// timeout to receive as much data as possible.
//
   if (LockReads) rdMutex.Lock();
   isIdle = 0;
   do {rlen = read(FD, Buff, Blen);} while(rlen < 0 && errno == EINTR);
   if (LockReads) rdMutex.UnLock();

   if (rlen >= 0) return int(rlen);
   if (FD >= 0) XrdLog.Emsg("Link", errno, "receive from", ID);
   return -1;
}

/******************************************************************************/

int XrdLink::Recv(char *Buff, int Blen, int timeout)
{
   XrdSysMutexHelper theMutex;
   struct pollfd polltab = {FD, POLLIN|POLLRDNORM, 0};
   ssize_t rlen, totlen = 0;
   int retc;

// Lock the read mutex if we need to, the helper will unlock it upon exit
//
   if (LockReads) theMutex.Lock(&rdMutex);

// Wait up to timeout milliseconds for data to arrive
//
   isIdle = 0;
   while(Blen > 0)
        {do {retc = poll(&polltab,1,timeout);} while(retc < 0 && errno == EINTR);
         if (retc != 1)
            {if (retc == 0)
                {tardyCnt++;
                 if (totlen  && (++stallCnt & 0xff) == 1)
                    TRACEI(DEBUG, "read timed out");
                 return int(totlen);
                }
             return (FD >= 0 ? XrdLog.Emsg("Link", -errno, "poll", ID) : -1);
            }

         // Verify it is safe to read now
         //
         if (!(polltab.revents & (POLLIN|POLLRDNORM)))
            {XrdLog.Emsg("Link", XrdPoll::Poll2Text(polltab.revents),
                                 "polling", ID);
             return -1;
            }

         // Read as much data as you can. Note that we will force an error
         // if we get a zero-length read after poll said it was OK.
         //
         do {rlen = recv(FD, Buff, Blen, 0);} while(rlen < 0 && errno == EINTR);
         if (rlen <= 0)
            {if (!rlen) return -ENOMSG;
             return (FD<0 ? -1 : XrdLog.Emsg("Link",-errno,"receive from",ID));
            }
         BytesIn += rlen; totlen += rlen; Blen -= rlen; Buff += rlen;
        }

   return int(totlen);
}


/******************************************************************************/
/*                               R e c v A l l                                */
/******************************************************************************/
  
int XrdLink::RecvAll(char *Buff, int Blen, int timeout)
{
   struct pollfd polltab = {FD, POLLIN|POLLRDNORM, 0};
   ssize_t rlen;
   int     retc;

// Check if timeout specified. Notice that the timeout is the max we will
// for some data. We will wait forever for all the data. Yeah, it's weird.
//
   if (timeout >= 0)
      {do {retc = poll(&polltab,1,timeout);} while(retc < 0 && errno == EINTR);
       if (retc != 1)
          {if (!retc) return -ETIMEDOUT;
           XrdLog.Emsg("Link",errno,"poll",ID);
           return -1;
          }
       if (!(polltab.revents & (POLLIN|POLLRDNORM)))
          {XrdLog.Emsg("Link",XrdPoll::Poll2Text(polltab.revents),"polling",ID);
           return -1;
          }
      }

// Note that we will block until we receive all he bytes.
//
   if (LockReads) rdMutex.Lock();
   isIdle = 0;
   do {rlen = recv(FD,Buff,Blen,MSG_WAITALL);} while(rlen < 0 && errno == EINTR);
   if (LockReads) rdMutex.UnLock();

   if (int(rlen) == Blen) return Blen;
   if (!rlen) {TRACEI(DEBUG, "No RecvAll() data; errno=" <<errno);}
      else if (rlen > 0) XrdLog.Emsg("RecvAll","Premature end from", ID);
              else if (FD >= 0) XrdLog.Emsg("Link",errno,"recieve from",ID);
   return -1;
}

/******************************************************************************/
/*                                  S e n d                                   */
/******************************************************************************/
  
int XrdLink::Send(const char *Buff, int Blen)
{
   ssize_t retc = 0, bytesleft = Blen;

// Get a lock
//
   wrMutex.Lock();
   isIdle = 0;

// Write the data out
//
   while(bytesleft)
        {if ((retc = write(FD, Buff, bytesleft)) < 0)
            {if (errno == EINTR) continue;
                else break;
            }
         BytesOut += retc; bytesleft -= retc; Buff += retc;
        }

// All done
//
   wrMutex.UnLock();
   if (retc >= 0) return Blen;
   XrdLog.Emsg("Link", errno, "send to", ID);
   return -1;
}

/******************************************************************************/
  
int XrdLink::Send(const struct iovec *iov, int iocnt, int bytes)
{
   ssize_t bytesleft, n, retc = 0;
   const char *Buff;
   int i;

// Add up bytes if they were not given to us
//
   if (!bytes) for (i = 0; i < iocnt; i++) bytes += iov[i].iov_len;
   bytesleft = static_cast<ssize_t>(bytes);

// Get a lock and assume we will be successful (statistically we are)
//
   wrMutex.Lock();
   isIdle = 0;
   BytesOut += bytes;

// Write the data out. On some version of Unix (e.g., Linux) a writev() may
// end at any time without writing all the bytes when directed to a socket.
// So, we attempt to resume the writev() using a combination of write() and
// a writev() continuation. This approach slowly converts a writev() to a
// series of writes if need be. We must do this inline because we must hold
// the lock until all the bytes are written or an error occurs.
//
   while(bytesleft)
        {do {retc = writev(FD, iov, iocnt);} while(retc < 0 && errno == EINTR);
         if (retc >= bytesleft || retc < 0) break;
         bytesleft -= retc;
         while(retc >= (n = static_cast<ssize_t>(iov->iov_len)))
              {retc -= n; iov++; iocnt--;}
         Buff = (const char *)iov->iov_base + retc; n -= retc; iov++; iocnt--;
         while(n) {if ((retc = write(FD, Buff, n)) < 0)
                      {if (errno == EINTR) continue;
                          else break;
                      }
                   n -= retc; Buff += retc;
                  }
         if (retc < 0 || iocnt < 1) break;
        }

// All done
//
   wrMutex.UnLock();
   if (retc >= 0) return bytes;
   XrdLog.Emsg("Link", errno, "send to", ID);
   return -1;
}
 
/******************************************************************************/
int XrdLink::Send(const struct sfVec *sfP, int sfN)
{
#if !defined(HAVE_SENDFILE)
   return -1;
#else
// Make sure we have valid vector count
//
   if (sfN < 1 || sfN > sfMax)
      {XrdLog.Emsg("Link", EINVAL, "send file to", ID);
       return -1;
      }

#ifdef __solaris__
    sendfilevec_t vecSF[sfMax], *vecSFP = vecSF;
    size_t xframt, totamt, bytes = 0;
    ssize_t retc;
    int i = 0;

// Construct the sendfilev() vector
//
   for (i = 0; i < sfN; sfP++, i++)
       {if (sfP->fdnum < 0)
           {vecSF[i].sfv_fd  = SFV_FD_SELF;
            vecSF[i].sfv_off = (off_t)sfP->buffer;
           } else {
            vecSF[i].sfv_fd  = sfP->fdnum;
            vecSF[i].sfv_off = sfP->offset;
           }
        vecSF[i].sfv_flag = 0;
        vecSF[i].sfv_len  = sfP->sendsz;
        bytes += sfP->sendsz;
       }
   totamt = bytes;

// Lock the link, issue sendfilev(), and unlock the link. The documentation
// is very spotty and inconsistent. We can only retry this operation under
// very limited conditions.
//
   wrMutex.Lock();
   isIdle = 0;
do{retc = sendfilev(FD, vecSFP, sfN, &xframt);

// Check if all went well and return if so (usual case)
//
   if (xframt == bytes)
      {BytesOut += bytes;
       wrMutex.UnLock();
       return totamt;
      }

// The only one we will recover from is EINTR. We cannot legally get EAGAIN.
//
   if (retc < 0 && errno != EINTR) break;

// Try to resume the transfer
//
   if (xframt > 0)
      {BytesOut += xframt; bytes -= xframt; SfIntr++;
       while(xframt > 0 && sfN)
            {if ((ssize_t)xframt < (ssize_t)vecSFP->sfv_len)
                {vecSFP->sfv_off += xframt; vecSFP->sfv_len -= xframt; break;}
             xframt -= vecSFP->sfv_len; vecSFP++; sfN--;
            }
      }
  } while(sfN > 0);

// See if we can recover without destroying the connection
//
   retc = (retc < 0 ? errno : ECANCELED);
   wrMutex.UnLock();
   XrdLog.Emsg("Link", retc, "send file to", ID);
   return -1;

#elif defined(__linux__)

   static const int setON = 1, setOFF = 0;
   ssize_t retc = 0, bytesleft;
   off_t myOffset;
   int i, xfrbytes = 0, uncork = 1, xIntr = 0;

// lock the link
//
   wrMutex.Lock();
   isIdle = 0;

// In linux we need to cork the socket. On permanent errors we do not uncork
// the socket because it will be closed in short order.
//
   if (setsockopt(FD, SOL_TCP, TCP_CORK, &setON, sizeof(setON)) < 0)
      {XrdLog.Emsg("Link", errno, "cork socket for", ID); 
       uncork = 0; sfOK = 0;
      }

// Send the header first
//
   for (i = 0; i < sfN; sfP++, i++)
       {if (sfP->fdnum < 0) retc = sendData(sfP->buffer, sfP->sendsz);
           else {myOffset = sfP->offset; bytesleft = sfP->sendsz;
                 while(bytesleft
                    && (retc=sendfile(FD,sfP->fdnum,&myOffset,bytesleft)) > 0)
                      {myOffset += retc; bytesleft -= retc; xIntr++;}
                }
        if (retc <  0 && errno == EINTR) continue;
        if (retc <= 0) break;
        xfrbytes += sfP->sendsz;
       }

// Diagnose any sendfile errors
//
   if (retc <= 0)
      {if (retc == 0) errno = ECANCELED;
       wrMutex.UnLock();
       XrdLog.Emsg("Link", errno, "send file to", ID);
       return -1;
      }

// Now uncork the socket
//
   if (uncork && setsockopt(FD, SOL_TCP, TCP_CORK, &setOFF, sizeof(setOFF)) < 0)
      XrdLog.Emsg("Link", errno, "uncork socket for", ID);

// All done
//
   if (xIntr > sfN) SfIntr += (xIntr - sfN);
   BytesOut += xfrbytes;
   wrMutex.UnLock();
   return xfrbytes;
#endif
#endif
}

/******************************************************************************/
/* private                      s e n d D a t a                               */
/******************************************************************************/
  
int XrdLink::sendData(const char *Buff, int Blen)
{
   ssize_t retc = 0, bytesleft = Blen;

// Write the data out
//
   while(bytesleft)
        {if ((retc = write(FD, Buff, bytesleft)) < 0)
            {if (errno == EINTR) continue;
                else break;
            }
         bytesleft -= retc; Buff += retc;
        }

// All done
//
   return retc;
}

/******************************************************************************/
/*                              s e t E t e x t                               */
/******************************************************************************/

int XrdLink::setEtext(const char *text)
{
     opMutex.Lock();
     if (Etext) free(Etext);
     Etext = (text ? strdup(text) : 0);
     opMutex.UnLock();
     return -1;
}
  
/******************************************************************************/
/*                                 s e t I D                                  */
/******************************************************************************/
  
void XrdLink::setID(const char *userid, int procid)
{
   char buff[sizeof(Uname)], *bp, *sp;
   int ulen;

   snprintf(buff, sizeof(buff), "%s.%d:%d", userid, procid, FD);
   ulen = strlen(buff);
   sp = buff + ulen - 1;
   bp = &Uname[sizeof(Uname)-1];
   if (ulen > (int)sizeof(Uname)) ulen = sizeof(Uname);
   *bp = '@'; bp--;
   while(ulen--) {*bp = *sp; bp--; sp--;}
   ID = bp+1;
   Comment = (const char *)ID;
}
 
/******************************************************************************/
/*                                 S e t u p                                  */
/******************************************************************************/

int XrdLink::Setup(int maxfds, int idlewait)
{
   int numalloc, iticks, ichk;

// Make sure our static /dev/null fd is closed whn we exec
//
   fcntl(devNull, F_SETFD, FD_CLOEXEC);

// Compute the number of link objects we should allocate at a time. Generally,
// we like to allocate 8k of them at a time but always as a power of two.
//
   numalloc = 8192 / sizeof(XrdLink);
   LinkAlloc = 1;
   while((numalloc = numalloc/2)) LinkAlloc = LinkAlloc*2;
   TRACE(DEBUG, "Allocating " <<LinkAlloc <<" link objects at a time");

// Create the link table
//
   if (!(LinkTab = (XrdLink **)malloc(maxfds*sizeof(XrdLink *)+LinkAlloc)))
      {XrdLog.Emsg("Link", ENOMEM, "create LinkTab"); return 0;}
   memset((void *)LinkTab, 0, maxfds*sizeof(XrdLink *));

// Create the slot status table
//
   if (!(LinkBat = (char *)malloc(maxfds*sizeof(char)+LinkAlloc)))
      {XrdLog.Emsg("Link", ENOMEM, "create LinkBat"); return 0;}
   memset((void *)LinkBat, XRDLINK_FREE, maxfds*sizeof(char));

// Create an idle connection scan job
//
   if (idlewait)
      {if (!(ichk = idlewait/3)) {iticks = 1; ichk = idlewait;}
          else iticks = 3;
       XrdLinkScan *ls = new XrdLinkScan(ichk, iticks);
       XrdSched.Schedule((XrdJob *)ls, ichk+time(0));
      }

   return 1;
}
  
/******************************************************************************/
/*                             S e r i a l i z e                              */
/******************************************************************************/
  
void XrdLink::Serialize()
{

// This is meant to make sure that no protocol objects are refering to this
// link so that we can safely run in psuedo single thread mode for critical
// functions.
//
   opMutex.Lock();
   if (InUse <= 1) opMutex.UnLock();
      else {doPost++;
            opMutex.UnLock();
            TRACEI(DEBUG, "Waiting for link serialization; use=" <<InUse);
            IOSemaphore.Wait();
           }
}

/******************************************************************************/
/*                                s e t K W T                                 */
/******************************************************************************/
  
void XrdLink::setKWT(int wkSec, int kwSec)
{
   if (wkSec > 0) waitKill = static_cast<short>(wkSec);
   if (kwSec > 0) killWait = static_cast<short>(kwSec);
}
  
/******************************************************************************/
/*                           s e t P r o t o c o l                            */
/******************************************************************************/
  
XrdProtocol *XrdLink::setProtocol(XrdProtocol *pp)
{

// Set new protocol.
//
   opMutex.Lock();
   XrdProtocol *op = Protocol;
   Protocol = pp; 
   opMutex.UnLock();
   return op;
}

/******************************************************************************/
/*                                s e t R e f                                 */
/******************************************************************************/
  
void XrdLink::setRef(int use)
{
   opMutex.Lock();
   TRACEI(DEBUG,"Setting ref to " <<InUse <<'+' <<use <<" post=" <<doPost);
   InUse += use;

         if (!InUse)
            {InUse = 1; opMutex.UnLock();
             XrdLog.Emsg("Link", "Zero use count for", ID);
            }
    else if (InUse == 1 && doPost)
            {doPost--;
             IOSemaphore.Post();
             TRACEI(CONN, "setRef posted link");
             opMutex.UnLock();
            }
    else if (InUse < 0)
            {InUse = 1;
             opMutex.UnLock();
             XrdLog.Emsg("Link", "Negative use count for", ID);
            }
    else opMutex.UnLock();
}
 
/******************************************************************************/
/*                                 S t a t s                                  */
/******************************************************************************/

int XrdLink::Stats(char *buff, int blen, int do_sync)
{
   static const char statfmt[] = "<stats id=\"link\"><num>%d</num>"
          "<maxn>%d</maxn><tot>%lld</tot><in>%lld</in><out>%lld</out>"
          "<ctime>%lld</ctime><tmo>%d</tmo><stall>%d</stall>"
          "<sfps>%d</sfps></stats>";
   int i, myLTLast;

// Check if actual length wanted
//
   if (!buff) return sizeof(statfmt)+17*6;

// We must synchronize the statistical counters
//
   if (do_sync)
      {LTMutex.Lock(); myLTLast = LTLast; LTMutex.UnLock();
       for (i = 0; i <= myLTLast; i++) 
           if (LinkBat[i] == XRDLINK_USED && LinkTab[i]) 
              LinkTab[i]->syncStats();
      }

// Obtain lock on the stats area and format it
//
   statsMutex.Lock();
   i = snprintf(buff, blen, statfmt, LinkCount,   LinkCountMax, LinkCountTot,
                                     LinkBytesIn, LinkBytesOut, LinkConTime,
                                     LinkTimeOuts,LinkStalls,   LinkSfIntr);
   statsMutex.UnLock();
   return i;
}
  
/******************************************************************************/
/*                             s y n c S t a t s                              */
/******************************************************************************/
  
void XrdLink::syncStats(int *ctime)
{

// If this is dynamic, get the opMutex lock
//
   if (!ctime) opMutex.Lock();

// Either the caller has the opMutex or this is called out of close. In either
// case, we need to get the read
//
   statsMutex.Lock();
   rdMutex.Lock();
   LinkBytesIn  += BytesIn; BytesInTot   += BytesIn;   BytesIn = 0;
   LinkTimeOuts += tardyCnt; tardyCntTot += tardyCnt; tardyCnt = 0;
   LinkStalls   += stallCnt; stallCntTot += stallCnt; stallCnt = 0;
   rdMutex.UnLock();
   wrMutex.Lock();
   LinkBytesOut += BytesOut; BytesOutTot += BytesOut;BytesOut = 0;
   LinkSfIntr   += SfIntr;   SfIntr = 0;
   wrMutex.UnLock();
   if (ctime)
      {*ctime = time(0) - conTime;
       LinkConTime += *ctime;
       if (!(LinkCount--)) LinkCount = 0;
      }
   statsMutex.UnLock();

// Make sure the protocol updates it's statistics as well
//
   if (Protocol) Protocol->Stats(0, 0, 1);

// Clear our local counters
//
   if (!ctime) opMutex.UnLock();
}
 
/******************************************************************************/
/*                             T e r m i n a t e                              */
/******************************************************************************/
  
int XrdLink::Terminate(const XrdLink *owner, int fdnum, unsigned int inst)
{
   XrdSysCondVar killDone(0);
   XrdLink *lp;
   char buff[1024], *cp;
   int wTime, didKW = KillCnt & KillXwt;

// Find the correspodning link
//
   KillCnt = KillCnt & KillMsk;
   if (!(lp = fd2link(fdnum, inst))) return (didKW ? -EPIPE : -ESRCH);

// If this is self termination, then indicate that to the caller
//
   if (lp == owner) return 0;

// Serialize the target link
//
   lp->Serialize();
   lp->opMutex.Lock();

// If this link is now dead, simply ignore the request. Typically, this
// indicates a race condition that the server won.
//
   if ( lp->FD != fdnum ||   lp->Instance != inst
   || !(lp->Poller)     || !(lp->Protocol))
      {lp->opMutex.UnLock();
       return -EPIPE;
      }

// Verify that the owner of this link is making the request
//
   if (owner 
   && (!(cp = index(owner->ID, ':')) 
      || strncmp(lp->ID, owner->ID, cp-(owner->ID))
      || strcmp(owner->Lname, lp->Lname)))
      {lp->opMutex.UnLock();
       return -EACCES;
      }

// Check if we have too many tries here
//
   if (lp->KillCnt > KillMax)
      {lp->opMutex.UnLock();
       return -ETIME;
      }
   wTime = lp->KillCnt++;

// Make sure we can disable this link. Of not, then force the caller to wait
// a tad more than the read timeout interval.
//
   if (!(lp->isEnabled) || lp->InUse > 1 || lp->KillcvP)
      {wTime = wTime*2+waitKill;
       KillCnt |= KillXwt;
       lp->opMutex.UnLock();
       return (wTime > 60 ? 60: wTime);
      }

// Set the pointer to our condvar. We are holding the opMutex to prevent a race.
//
   lp->KillcvP = &killDone;
   killDone.Lock();

// We can now disable the link and schedule a close
//
   snprintf(buff, sizeof(buff), "ended by %s", ID);
   buff[sizeof(buff)-1] = '\0';
   lp->Poller->Disable(lp, buff);
   lp->opMutex.UnLock();

// Now wait for the link to shutdown. This avoids lock problems.
//
   if (killDone.Wait(int(killWait))) {wTime += killWait; KillCnt |= KillXwt;}
      else wTime = -EPIPE;
   killDone.UnLock();

// Reobtain the opmutex so that we can zero out the pointer the condvar pntr
// This is really stupid code but because we don't have a way of associating
// an arbitrary mutex with a condvar. But since this code is rarely executed
// the ugliness is sort of tolerable.
//
   lp->opMutex.Lock(); lp->KillcvP = 0; lp->opMutex.UnLock();

// Do some tracing
//
   TRACEI(DEBUG,"Terminate " << (wTime <= 0 ? "complete ":"timeout ") <<wTime);
   return wTime;
}

/******************************************************************************/
/*                              i d l e S c a n                               */
/******************************************************************************/
  
#undef   TRACELINK
#define  TRACELINK lp

void XrdLinkScan::idleScan()
{
   XrdLink *lp;
   int i, ltlast, lnum = 0, tmo = 0, tmod = 0;

// Get the current link high watermark
//
   XrdLink::LTMutex.Lock();
   ltlast = XrdLink::LTLast;
   XrdLink::LTMutex.UnLock();

// Scan across all links looking for idle links. Links are never deallocated
// so we don't need any special kind of lock for these
//
   for (i = 0; i <= ltlast; i++)
       {if (XrdLink::LinkBat[i] != XRDLINK_USED 
        || !(lp = XrdLink::LinkTab[i])) continue;
        lnum++;
        lp->opMutex.Lock();
        if (lp->isIdle) tmo++;
        lp->isIdle++;
        if ((int(lp->isIdle)) < idleTicks) {lp->opMutex.UnLock(); continue;}
        lp->isIdle = 0;
        if (!(lp->Poller) || !(lp->isEnabled))
           XrdLog.Emsg("LinkScan","Link",lp->ID,"is disabled and idle.");
           else if (lp->InUse == 1)
                   {lp->Poller->Disable(lp, "idle timeout");
                    tmod++;
                   }
        lp->opMutex.UnLock();
       }

// Trace what we did
//
   TRACE(CONN, lnum <<" links; " <<tmo <<" idle; " <<tmod <<" force closed");

// Reschedule ourselves
//
   XrdSched.Schedule((XrdJob *)this, idleCheck+time(0));
}
