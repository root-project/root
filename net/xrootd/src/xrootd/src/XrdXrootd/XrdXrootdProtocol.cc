/******************************************************************************/
/*                                                                            */
/*                  X r d X r o o t d P r o t o c o l . c c                   */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
 
#include "XrdVersion.hh"

#include "XrdSfs/XrdSfsInterface.hh"
#include "Xrd/XrdBuffer.hh"
#include "Xrd/XrdLink.hh"
#include "XProtocol/XProtocol.hh"
#include "XrdSys/XrdSysTimer.hh"
#include "XrdXrootd/XrdXrootdAio.hh"
#include "XrdXrootd/XrdXrootdFile.hh"
#include "XrdXrootd/XrdXrootdFileLock.hh"
#include "XrdXrootd/XrdXrootdFileLock1.hh"
#include "XrdXrootd/XrdXrootdMonitor.hh"
#include "XrdXrootd/XrdXrootdPio.hh"
#include "XrdXrootd/XrdXrootdProtocol.hh"
#include "XrdXrootd/XrdXrootdStats.hh"
#include "XrdXrootd/XrdXrootdTrace.hh"
#include "XrdXrootd/XrdXrootdXPath.hh"

/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/

XrdOucTrace          *XrdXrootdTrace;

XrdXrootdXPath        XrdXrootdProtocol::RPList;
XrdXrootdXPath        XrdXrootdProtocol::XPList;
XrdSfsFileSystem     *XrdXrootdProtocol::osFS;
char                 *XrdXrootdProtocol::FSLib    = 0;
XrdXrootdFileLock    *XrdXrootdProtocol::Locker;
XrdSecService        *XrdXrootdProtocol::CIA      = 0;
char                 *XrdXrootdProtocol::SecLib   = 0;
char                 *XrdXrootdProtocol::pidPath  = strdup("/tmp");
XrdScheduler         *XrdXrootdProtocol::Sched;
XrdBuffManager       *XrdXrootdProtocol::BPool;
XrdSysError           XrdXrootdProtocol::eDest(0, "Xrootd");
XrdXrootdStats       *XrdXrootdProtocol::SI;
XrdXrootdJob         *XrdXrootdProtocol::JobCKS   = 0;
char                 *XrdXrootdProtocol::JobCKT   = 0;

char                 *XrdXrootdProtocol::Notify = 0;
int                   XrdXrootdProtocol::hailWait;
int                   XrdXrootdProtocol::readWait;
int                   XrdXrootdProtocol::Port;
int                   XrdXrootdProtocol::Window;
int                   XrdXrootdProtocol::WANPort;
int                   XrdXrootdProtocol::WANWindow;
char                  XrdXrootdProtocol::isRedir = 0;
char                  XrdXrootdProtocol::chkfsV  = 0;
XrdNetSocket         *XrdXrootdProtocol::AdminSock= 0;

int                   XrdXrootdProtocol::hcMax        = 28657; // const for now
int                   XrdXrootdProtocol::maxBuffsz;
int                   XrdXrootdProtocol::maxTransz    = 262144; // 256KB
int                   XrdXrootdProtocol::as_maxperlnk = 8;   // Max ops per link
int                   XrdXrootdProtocol::as_maxperreq = 8;   // Max ops per request
int                   XrdXrootdProtocol::as_maxpersrv = 4096;// Max ops per server
int                   XrdXrootdProtocol::as_segsize   = 131072;
int                   XrdXrootdProtocol::as_miniosz   = 32768;
#ifdef __solaris__
int                   XrdXrootdProtocol::as_minsfsz   = 1;
#else
int                   XrdXrootdProtocol::as_minsfsz   = 8192;
#endif
int                   XrdXrootdProtocol::as_maxstalls = 5;
int                   XrdXrootdProtocol::as_force     = 0;
int                   XrdXrootdProtocol::as_noaio     = 0;
int                   XrdXrootdProtocol::as_nosf      = 0;
int                   XrdXrootdProtocol::as_syncw     = 0;

const char           *XrdXrootdProtocol::myInst  = 0;
const char           *XrdXrootdProtocol::TraceID = "Protocol";
int                   XrdXrootdProtocol::myPID = static_cast<int>(getpid());

struct XrdXrootdProtocol::RD_Table XrdXrootdProtocol::Route[RD_Num] = {{0,0}};

/******************************************************************************/
/*            P r o t o c o l   M a n a g e m e n t   S t a c k s             */
/******************************************************************************/
  
XrdObjectQ<XrdXrootdProtocol>
            XrdXrootdProtocol::ProtStack("ProtStack",
                                       "xrootd protocol anchor");

/******************************************************************************/
/*                         L o c a l   D e f i n e s                          */
/******************************************************************************/
  
#define UPSTATS(x) SI->statsMutex.Lock(); SI->x++; SI->statsMutex.UnLock()

/******************************************************************************/
/*                       P r o t o c o l   L o a d e r                        */
/*                        X r d g e t P r o t o c o l                         */
/******************************************************************************/
  
// This protocol can live in a shared library. The interface below is used by
// the protocol driver to obtain a copy of the protocol object that can be used
// to decide whether or not a link is talking a particular protocol.
//
extern "C"
{
XrdProtocol *XrdgetProtocol(const char *pname, char *parms,
                              XrdProtocol_Config *pi)
{
   XrdProtocol *pp = 0;
   const char *txt = "completed.";

// Put up the banner
//
   pi->eDest->Say("Copr.  2007 Stanford University, xrootd version "
                   XROOTD_VERSION " build "  XrdVERSION);
   pi->eDest->Say("++++++ xrootd protocol initialization started.");

// Return the protocol object to be used if static init succeeds
//
   if (XrdXrootdProtocol::Configure(parms, pi))
      pp = (XrdProtocol *)new XrdXrootdProtocol();
      else txt = "failed.";
    pi->eDest->Say("------ xrootd protocol initialization ", txt);
   return pp;
}
}

/******************************************************************************/
/*                                                                            */
/*           P r o t o c o l   P o r t   D e t e r m i n a t i o n            */
/*                    X r d g e t P r o t o c o l P o r t                     */
/******************************************************************************/

// This function is called early on to determine the port we need to use. The
// default is ostensibly 1094 but can be overidden; which we allow.
//
extern "C"
{
int XrdgetProtocolPort(const char *pname, char *parms, XrdProtocol_Config *pi)
{

// Figure out what port number we should return. In practice only one port
// number is allowed. However, we could potentially have a clustered port
// and several unclustered ports. So, we let this practicality slide.
//
   if (pi->Port < 0) return 1094;
   return pi->Port;
}
}
  
/******************************************************************************/
/*               X r d P r o t o c o l X r o o t d   C l a s s                */
/******************************************************************************/
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/

XrdXrootdProtocol::XrdXrootdProtocol() 
                    : XrdProtocol("xrootd protocol handler"), ProtLink(this),
                      Entity("")
{
   Reset();
}

/******************************************************************************/
/*                   A s s i g n m e n t   O p e r a t o r                    */
/******************************************************************************/

XrdXrootdProtocol XrdXrootdProtocol::operator =(const XrdXrootdProtocol &rhs)
{
// Reset all common fields
//
   Reset();

// Now copy the relevant fields only
//
   Link          = rhs.Link;
   Link->setRef(1);      // Keep the link stable until we dereference it
   Status        = rhs.Status;
   myFile        = rhs.myFile;
   myIOLen       = rhs.myIOLen;
   myOffset      = rhs.myOffset;
   Response      = rhs.Response;
   memcpy((void *)&Request,(const void *)&rhs.Request, sizeof(Request));
   Client        = rhs.Client;
   AuthProt      = rhs.AuthProt;
   return *this;
}
  
/******************************************************************************/
/*                                 M a t c h                                  */
/******************************************************************************/

#define TRACELINK lp
  
XrdProtocol *XrdXrootdProtocol::Match(XrdLink *lp)
{
        struct ClientInitHandShake hsdata;
        char  *hsbuff = (char *)&hsdata;

static  struct hs_response
               {kXR_unt16 streamid;
                kXR_unt16 status;
                kXR_int32 rlen;
                kXR_int32 pval;
                kXR_int32 styp;
               } hsresp={0, 0, htonl(8), // isRedir == 'M' -> MetaManager
                         htonl(XROOTD_VERSBIN),
                         (isRedir ? htonl(kXR_LBalServer)
                                  : htonl(kXR_DataServer))};

XrdXrootdProtocol *xp;
int dlen;

// Peek at the first 20 bytes of data
//
   if ((dlen = lp->Peek(hsbuff,sizeof(hsdata), hailWait)) != sizeof(hsdata))
      {if (dlen <= 0) lp->setEtext("handshake not received");
       return (XrdProtocol *)0;
      }

// Trace the data
//
// TRACEI(REQ, "received: " <<Trace->bin2hex(hsbuff,dlen));

// Verify that this is our protocol
//
   hsdata.fourth  = ntohl(hsdata.fourth);
   hsdata.fifth   = ntohl(hsdata.fifth);
   if (dlen != sizeof(hsdata) ||  hsdata.first || hsdata.second
   || hsdata.third || hsdata.fourth != 4 || hsdata.fifth != ROOTD_PQ) return 0;

// Respond to this request with the handshake response
//
   if (!lp->Send((char *)&hsresp, sizeof(hsresp)))
      {lp->setEtext("handshake failed");
       return (XrdProtocol *)0;
      }

// We can now read all 20 bytes and discard them (no need to wait for it)
//
   if (lp->Recv(hsbuff, sizeof(hsdata)) != sizeof(hsdata))
      {lp->setEtext("reread failed");
       return (XrdProtocol *)0;
      }

// Get a protocol object off the stack (if none, allocate a new one)
//
   if (!(xp = ProtStack.Pop())) xp = new XrdXrootdProtocol();

// Bind the protocol to the link and return the protocol
//
   UPSTATS(Count);
   xp->Link = lp;
   xp->Response.Set(lp);
   strcpy(xp->Entity.prot, "host");
   xp->Entity.host = (char *)lp->Host();
   return (XrdProtocol *)xp;
}
 
/******************************************************************************/
/*                               P r o c e s s                                */
/******************************************************************************/

#undef  TRACELINK
#define TRACELINK Link
  
int XrdXrootdProtocol::Process(XrdLink *lp) // We ignore the argument here
{
   int rc;

// Check if we are servicing a slow link
//
   if (Resume)
      {if (myBlen && (rc = getData("data", myBuff, myBlen)) != 0)
          {if (rc < 0 && myAioReq) myAioReq->Recycle(-1);
           return rc;
          }
          else if ((rc = (*this.*Resume)()) != 0) return rc;
                  else {Resume = 0; return 0;}
      }

// Read the next request header
//
   if ((rc=getData("request",(char *)&Request,sizeof(Request))) != 0) return rc;

// Deserialize the data
//
   Request.header.requestid = ntohs(Request.header.requestid);
   Request.header.dlen      = ntohl(Request.header.dlen);
   Response.Set(Request.header.streamid);
   TRACEP(REQ, "req=" <<Request.header.requestid <<" dlen=" <<Request.header.dlen);

// Every request has an associated data length. It better be >= 0 or we won't
// be able to know how much data to read.
//
   if (Request.header.dlen < 0)
      {Response.Send(kXR_ArgInvalid, "Invalid request data length");
       return Link->setEtext("protocol data length error");
      }

// Read any argument data at this point, except when the request is a write.
// The argument may have to be segmented and we're not prepared to do that here.
//
   if (Request.header.requestid != kXR_write && Request.header.dlen)
      {if (!argp || Request.header.dlen+1 > argp->bsize)
          {if (argp) BPool->Release(argp);
           if (!(argp = BPool->Obtain(Request.header.dlen+1)))
              {Response.Send(kXR_ArgTooLong, "Request argument is too long");
               return 0;
              }
           hcNow = hcPrev; halfBSize = argp->bsize >> 1;
          }
       if ((rc = getData("arg", argp->buff, Request.header.dlen)))
          {Resume = &XrdXrootdProtocol::Process2; return rc;}
       argp->buff[Request.header.dlen] = '\0';
      }

// Continue with request processing at the resume point
//
   return Process2();
}

/******************************************************************************/
/*                      p r i v a t e   P r o c e s s 2                       */
/******************************************************************************/
  
int XrdXrootdProtocol::Process2()
{

// If the user is not yet logged in, restrict what the user can do
//
   if (!Status)
      switch(Request.header.requestid)
            {case kXR_login:    return do_Login();
             case kXR_protocol: return do_Protocol();
             case kXR_bind:     return do_Bind();
             default:           Response.Send(kXR_InvalidRequest,
                                "Invalid request; user not logged in");
                                return Link->setEtext("protocol sequence error 1");
            }

// Help the compiler, select the the high activity requests (the ones with
// file handles) in a separate switch statement
//
   switch(Request.header.requestid)   // First, the ones with file handles
         {case kXR_read:     return do_Read();
          case kXR_readv:    return do_ReadV();
          case kXR_write:    return do_Write();
          case kXR_sync:     return do_Sync();
          case kXR_close:    return do_Close();
          case kXR_truncate: if (!Request.header.dlen) return do_Truncate();
                             break;
          case kXR_query:    if (!Request.header.dlen) return do_Qfh();
          default:           break;
         }

// Now select the requests that do not need authentication
//
   switch(Request.header.requestid)
         {case kXR_protocol: return do_Protocol();   // dlen ignored
          case kXR_ping:     return do_Ping();       // dlen ignored
          default:           break;
         }

// Force authentication at this point, if need be
//
   if (Status & XRD_NEED_AUTH)
      {if (Request.header.requestid == kXR_auth) return do_Auth();
          else {Response.Send(kXR_InvalidRequest,
                              "Invalid request; user not authenticated");
                return -1;
               }
      }

// Process items that don't need arguments but may have them
//
   switch(Request.header.requestid)
         {case kXR_endsess:   return do_Endsess();
          default:            break;
         }

// All remaining requests require an argument. Make sure we have one
//
   if (!argp || !Request.header.dlen)
      {Response.Send(kXR_ArgMissing, "Required argument not present");
       return 0;
      }

// Construct request ID as the following functions are async eligible
//
   ReqID.setID(Request.header.streamid);

// Process items that keep own statistics
//
   switch(Request.header.requestid)
         {case kXR_open:      return do_Open();
          case kXR_getfile:   return do_Getfile();
          case kXR_putfile:   return do_Putfile();
          default:            break;
         }

// Update misc stats count
//
   UPSTATS(miscCnt);

// Now process whatever we have
//
   switch(Request.header.requestid)
         {case kXR_admin:     if (Status & XRD_ADMINUSER) return do_Admin();
                                 else break;
          case kXR_chmod:     return do_Chmod();
          case kXR_dirlist:   return do_Dirlist();
          case kXR_locate:    return do_Locate();
          case kXR_mkdir:     return do_Mkdir();
          case kXR_mv:        return do_Mv();
          case kXR_query:     return do_Query();
          case kXR_prepare:   return do_Prepare();
          case kXR_rm:        return do_Rm();
          case kXR_rmdir:     return do_Rmdir();
          case kXR_set:       return do_Set();
          case kXR_stat:      return do_Stat();
          case kXR_statx:     return do_Statx();
          case kXR_truncate:  return do_Truncate();
          default:            break;
         }

// Whatever we have, it's not valid
//
   Response.Send(kXR_InvalidRequest, "Invalid request code");
   return 0;
}

/******************************************************************************/
/*                               R e c y c l e                                */
/******************************************************************************/

#undef  TRACELINK
#define TRACELINK Link
  
void XrdXrootdProtocol::Recycle(XrdLink *lp, int csec, const char *reason)
{
   char *sfxp, ctbuff[24], buff[128];

// Document the disconnect or undind
//
   if (lp)
      {XrdSysTimer::s2hms(csec, ctbuff, sizeof(ctbuff));
       if (reason) {snprintf(buff, sizeof(buff), "%s (%s)", ctbuff, reason);
                    sfxp = buff;
                   } else sfxp = ctbuff;

       eDest.Log(SYS_LOG_02, "Xeq", lp->ID,
             (Status == XRD_BOUNDPATH ? (char *)"unbind":(char *)"disc"), sfxp);
      }

// If this is a bound stream then we cannot release the resources until
// the main stream closes this stream (i.e., lp == 0). On the other hand, the
// main stream will not be trying to do this if we are still tagged as active.
// So, we need to redrive the main stream to complete the full shutdown.
//
   if (Status == XRD_BOUNDPATH && Stream[0])
      {Stream[0]->streamMutex.Lock();
       isDead = 1;
       if (isActive)
          {isActive = 0;
           Stream[0]->Link->setRef(-1);
          }
       Stream[0]->streamMutex.UnLock();
       if (lp) return;  // Async close
      }

// Check if we should monitor disconnects
//
   if (XrdXrootdMonitor::monUSER && Monitor) Monitor->Disc(monUID, csec);

// Release all appendages
//
   Cleanup();

// Set fields to starting point (debugging mostly)
//
   Reset();

// Push ourselves on the stack
//
   ProtStack.Push(&ProtLink);
}

/******************************************************************************/
/*                                 S t a t s                                  */
/******************************************************************************/
  
int XrdXrootdProtocol::Stats(char *buff, int blen, int do_sync)
{
// Synchronize statistics if need be
//
   if (do_sync)
      {SI->statsMutex.Lock();
       SI->readCnt += numReads;
       cumReads += numReads; numReads  = 0;
       SI->prerCnt += numReadP;
       cumReadP += numReadP; numReadP = 0;
       SI->writeCnt += numWrites;
       cumWrites+= numWrites;numWrites = 0;
       SI->statsMutex.UnLock();
      }

// Now return the statistics
//
   return SI->Stats(buff, blen, do_sync);
}
  
/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                               C l e a n u p                                */
/******************************************************************************/
  
void XrdXrootdProtocol::Cleanup()
{
   XrdXrootdPio *pioP;
   int i;

// If we have a buffer, release it
//
   if (argp) {BPool->Release(argp); argp = 0;}

// Delete the FTab if we have it
//
   if (FTab) {delete FTab; FTab = 0;}

// Handle parallel stream cleanup. The session stream cannot be closed if
// there is any queued activity on subordinate streams. A subordinate
// can either be closed from the session stream or asynchronously only if
// it is active. Which means they could be running while we are running.
//
   if (isBound && Status != XRD_BOUNDPATH)
      {streamMutex.Lock();
       for (i = 1; i < maxStreams; i++)
           if (Stream[i])
              {Stream[i]->isBound = 0; Stream[i]->Stream[0] = 0;
               if (Stream[i]->isDead) Stream[i]->Recycle(0, 0, 0);
                  else Stream[i]->Link->Close();
               Stream[i] = 0;
              }
       streamMutex.UnLock();
      }

// Handle statistics
//
   SI->statsMutex.Lock();
   SI->readCnt += numReads; SI->writeCnt += numWrites;
   SI->statsMutex.UnLock();

// Handle Monitor
//
   if (Monitor) {Monitor->unAlloc(Monitor); Monitor = 0;}

// Handle authentication protocol
//
   if (AuthProt) {AuthProt->Delete(); AuthProt = 0;}

// Handle parallel I/O appendages
//
   while((pioP = pioFirst)) {pioFirst = pioP->Next; pioP->Recycle();}
   while((pioP = pioFree )) {pioFree  = pioP->Next; pioP->Recycle();}
}
  
/******************************************************************************/
/*                               g e t D a t a                                */
/******************************************************************************/
  
int XrdXrootdProtocol::getData(const char *dtype, char *buff, int blen)
{
   int rlen;

// Read the data but reschedule he link if we have not received all of the
// data within the timeout interval.
//
   rlen = Link->Recv(buff, blen, readWait);
   if (rlen  < 0)
      {if (rlen != -ENOMSG) return Link->setEtext("link read error");
          else return -1;
      }
   if (rlen < blen)
      {myBuff = buff+rlen; myBlen = blen-rlen;
       TRACEP(REQ, dtype <<" timeout; read " <<rlen <<" of " <<blen <<" bytes");
       return 1;
      }
   return 0;
}

/******************************************************************************/
/*                                 R e s e t                                  */
/******************************************************************************/
  
void XrdXrootdProtocol::Reset()
{
   Status             = 0;
   argp               = 0;
   Link               = 0;
   FTab               = 0;
   Resume             = 0;
   myBuff             = (char *)&Request;
   myBlen             = sizeof(Request);
   myBlast            = 0;
   myOffset           = 0;
   myIOLen            = 0;
   myStalls           = 0;
   myAioReq           = 0;
   numReads           = 0;
   numReadP           = 0;
   numWrites          = 0;
   numFiles           = 0;
   cumReads           = 0;
   cumReadP           = 0;
   cumWrites          = 0;
   totReadP           = 0;
   hcPrev             =13;
   hcNext             =21;
   hcNow              =13;
   Monitor            = 0;
   monUID             = 0;
   monFILE            = 0;
   monIO              = 0;
   Client             = 0;
   AuthProt           = 0;
   mySID              = 0;
   CapVer             = 0;
   reTry              = 0;
   PathID             = 0;
   pioFree = pioFirst = pioLast = 0;
   isActive = isDead  = isNOP = isBound = 0;
   memset(&Entity, 0, sizeof(Entity));
   memset(Stream,  0, sizeof(Stream));
}
