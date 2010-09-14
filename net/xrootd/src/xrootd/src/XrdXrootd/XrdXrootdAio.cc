/******************************************************************************/
/*                                                                            */
/*                       X r d X r o o t d A i o . c c                        */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//        $Id$

const char *XrdXrootdAioCVSID = "$Id$";
  
#include <unistd.h>

#include "Xrd/XrdBuffer.hh"
#include "Xrd/XrdLink.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdSfs/XrdSfsInterface.hh"
#include "XrdXrootd/XrdXrootdAio.hh"
#include "XrdXrootd/XrdXrootdFile.hh"
#include "XrdXrootd/XrdXrootdProtocol.hh"
#include "XrdXrootd/XrdXrootdStats.hh"
#include "XrdXrootd/XrdXrootdTrace.hh"
 
/******************************************************************************/
/*                        S t a t i c   O b j e c t s                         */
/******************************************************************************/
  
XrdBuffManager           *XrdXrootdAio::BPool;
XrdScheduler             *XrdXrootdAio::Sched;
XrdXrootdStats           *XrdXrootdAio::SI;

XrdSysMutex               XrdXrootdAio::fqMutex;
XrdXrootdAio             *XrdXrootdAio::fqFirst = 0;
const char               *XrdXrootdAio::TraceID = "Aio";

int                       XrdXrootdAio::maxAio;

XrdSysError              *XrdXrootdAioReq::eDest;
XrdSysMutex               XrdXrootdAioReq::rqMutex;
XrdXrootdAioReq          *XrdXrootdAioReq::rqFirst = 0;
const char               *XrdXrootdAioReq::TraceID = "AioReq";

int                       XrdXrootdAioReq::QuantumMin;
int                       XrdXrootdAioReq::Quantum;
int                       XrdXrootdAioReq::QuantumMax;
int                       XrdXrootdAioReq::maxAioPR  = 8;
int                       XrdXrootdAioReq::maxAioPR2 =16;

extern XrdOucTrace       *XrdXrootdTrace;
 
/******************************************************************************/
/*                   X r d X r o o t d A i o : : A l l o c                    */
/******************************************************************************/
  
XrdXrootdAio *XrdXrootdAio::Alloc(XrdXrootdAioReq *arp, int bsize)
{
   XrdXrootdAio *aiop;

// Obtain an aio object
//
   fqMutex.Lock();
   if ((aiop = fqFirst)) fqFirst = aiop->Next;
      else if (maxAio) aiop = addBlock();
   if (aiop && (SI->AsyncNow > SI->AsyncMax)) SI->AsyncMax = SI->AsyncNow;
   fqMutex.UnLock();

// Allocate a buffer for this object
//
   if (aiop)
      {if (bsize && (aiop->buffp = BPool->Obtain(bsize)))
          {aiop->sfsAio.aio_buf = (void *)(aiop->buffp->buff);
           aiop->aioReq = arp;
           aiop->TIdent = arp->Link->ID;
          }
          else {aiop->Recycle(); aiop = 0;}
      }

// Return what we have
//
   return aiop;
}
 
/******************************************************************************/
/*                X r d X r o o t d A i o : : d o n e R e a d                 */
/******************************************************************************/

// Aio read requests are double buffered. So, there is only one aiocb active
// at a time. This is done for two reasons:
// 1) Provide a serial stream to the client, and
// 2) avoid swamping the network adapter.
// Additionally, double buffering requires minimal locking and simplifies the 
// redrive logic. While this knowledge violates OO design, it substantially 
// speeds up async I/O handling. This method is called out of the async event
// handler so it does very little work.
  
void XrdXrootdAio::doneRead()
{
// Plase this aio request on the completed queue
//
   aioReq->aioDone = this;

// Extract out any error conditions (keep only the first one)
//
   if (Result >= 0) aioReq->aioTotal += Result;
      else if (!aioReq->aioError) aioReq->aioError = Result;

// Schedule the associated arp to redrive the I/O
//
   Sched->Schedule((XrdJob *)aioReq);
}

/******************************************************************************/
/*               X r d X r o o t d A i o : : d o n e W r i t e                */
/******************************************************************************/

// Writes are more complicated because there may be several in transit. This
// is done to keep the client from swamping the network adapter. We try
// to optimize the handling of the aio object for the common cases. This method
// is called out of the async event handler so it does very little work.

void XrdXrootdAio::doneWrite()
{
   char recycle = 0;

// Lock the aioreq object against competition
//
   aioReq->Lock();
   aioReq->numActive--;

// Extract out any error conditions (keep only the first one).
//
   if (Result >= 0) {aioReq->myIOLen  -= Result;
                     aioReq->aioTotal += Result;
                    }
      else if (!aioReq->aioError) aioReq->aioError = Result;

// Redrive the protocol if so requested. It is impossible to have a proocol
// redrive and completed all of the I/O at the same time.
//
   if (aioReq->reDrive)
      {Sched->Schedule((XrdJob *)aioReq->Link);
       aioReq->reDrive = 0;
      }

// If more aio objects are needed, place this one on the free queue. Otherwise,
// schedule the AioReq object to complete handling the request if no more
// requests are outstanding. It is impossible to have a zero length with more
// requests outstanding.
//
   if (aioReq->myIOLen > 0)
      {Next = aioReq->aioFree; aioReq->aioFree = this;}
      else {if (!(aioReq->numActive)) Sched->Schedule((XrdJob *)aioReq);
            recycle = 1;
           }

// All done, perform early recycling if possible
//
   aioReq->UnLock();
   if (recycle) Recycle();
}

/******************************************************************************/
/*                 X r d X r o o t d A i o : : R e c y c l e                  */
/******************************************************************************/
  
void XrdXrootdAio::Recycle()
{

// Recycle the buffer
//
   if (buffp) {BPool->Release(buffp); buffp = 0;}

// Add this object to the free queue
//
   fqMutex.Lock();
   Next = fqFirst;
   fqFirst = this;
   if (--SI->AsyncNow < 0) SI->AsyncNow=0;
   fqMutex.UnLock();
}
  
/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                X r d X r o o t d A i o : : a d d B l o c k                 */
/******************************************************************************/
  
XrdXrootdAio *XrdXrootdAio::addBlock()
{
   const int numalloc = 4096/sizeof(XrdXrootdAio);
   int i = (numalloc <= maxAio ? numalloc : maxAio);
   XrdXrootdAio *aiop;

   TRACE(DEBUG, "Adding " <<i <<" aio objects; " <<maxAio <<" pending.");

   if ((aiop = new XrdXrootdAio[i]()))
      {maxAio -= i;
       while(--i) {aiop->Next = fqFirst; fqFirst = aiop; aiop++;}
      }

   return aiop;
}
  
/******************************************************************************/
/*                       X r d X r o o t d A i o R e q                        */
/******************************************************************************/
/******************************************************************************/
/*                X r d X r o o t d A i o R e q : : A l l o c                 */
/******************************************************************************/

// Implicit Parameters: prot->myIOLen   // Length of i/o request
//                      prot->myOffset  // Starting offset
//                      prot->myFile    // Target file
//                      prot->Link      // Link object
//                      prot->response  // Response object
  
XrdXrootdAioReq *XrdXrootdAioReq::Alloc(XrdXrootdProtocol *prot,
                                       char iotype, int numaio)
{
   int i, cntaio, myQuantum, iolen = prot->myIOLen;
   XrdXrootdAioReq *arp;
   XrdXrootdAio    *aiop;

// Obtain an aioreq object
//
   rqMutex.Lock();
   if ((arp = rqFirst)) rqFirst = arp->Next;
      else arp = addBlock();
   rqMutex.UnLock();

// Make sure we have one, fully reset it if we do
//
   if (!arp) return arp;
   arp->Clear(prot->Link);
   if (!numaio) numaio = maxAioPR;

// Compute the number of aio objects should get and the Quantum size we should
// use. This is a delicate balancing act. We don't want too many segments but
// neither do we want too large of an i/o size. So, if the i/o size is less than 
// the quantum then use half a quantum. If the number of segments is greater 
// than twice what we would like, then use a larger quantum size.
//
   if (iolen < Quantum) 
      {myQuantum = QuantumMin;
       if (!(cntaio = iolen / myQuantum)) cntaio = 1;
          else if (iolen % myQuantum) cntaio++;
      } else {cntaio = iolen / Quantum;
              if (cntaio <= maxAioPR2) myQuantum = Quantum;
                 else {myQuantum = QuantumMax;
                       cntaio = iolen / myQuantum;
                      }
              if (iolen % myQuantum) cntaio++;
             }

// Get appropriate number of aio objects
//
   i = (maxAioPR < cntaio ? maxAioPR : cntaio);
   while(i && (aiop = XrdXrootdAio::Alloc(arp, myQuantum)))
        {aiop->Next = arp->aioFree; arp->aioFree = aiop; i--;}

// Make sure we have at least the minimum number of aio objects
//
   if (i && (maxAioPR - i) < 2 && cntaio > 1)
      {arp->Recycle(0); return (XrdXrootdAioReq *)0;}

// Complete the request information
//
   if (iotype != 'w') prot->Link->setRef(1);
   arp->Instance   = prot->Link->Inst();
   arp->myIOLen    = iolen;  // Amount that is left to send
   arp->myOffset   = prot->myOffset;
   arp->myFile     = prot->myFile;
   arp->Response   = prot->Response;
   arp->aioType    = iotype;

// Return what we have
//
   return arp;
}

/******************************************************************************/
/*               X r d X r o o t d A i o R e q : : g e t A i o                */
/******************************************************************************/
  
XrdXrootdAio *XrdXrootdAioReq::getAio()
{
  XrdXrootdAio *aiop;

// Grab the next free aio object. If none, we return a null pointer. While this
// is a classic consumer/producer problem, normally handled by a semaphore,
// doing so would cause more threads to be tied up as the load increases. We
// want the opposite effect for scaling purposes. So, we use a redrive scheme.
//
   Lock();
   if ((aiop = aioFree)) {aioFree = aiop->Next; aiop->Next = 0;}
      else reDrive = 1;
   UnLock();
   return aiop;
}

/******************************************************************************/
/*                 X r d X r o o t d A i o R e q : : I n i t                  */
/******************************************************************************/
  
void XrdXrootdAioReq::Init(int iosize, int maxaiopr, int maxaio)
{
   XrdXrootdAio    *aiop;
   XrdXrootdAioReq *arp;

// Set the pointer to the buffer pool, scheduler and statistical area, these are
// only used by the Aio object
//
   XrdXrootdAio::Sched = XrdXrootdProtocol::Sched;
   XrdXrootdAio::BPool = XrdXrootdProtocol::BPool;
   XrdXrootdAio::SI    = XrdXrootdProtocol::SI;

// Set the pointer to the error object and compute the limits
//
   eDest       = &XrdXrootdProtocol::eDest;
   Quantum     = static_cast<size_t>(iosize);
   QuantumMin  = Quantum / 2;
   QuantumMax  = Quantum * 2;
   if (QuantumMax > XrdXrootdProtocol::maxBuffsz)
       QuantumMax = XrdXrootdProtocol::maxBuffsz;

// Set the maximum number of aio objects we can have (used by Aio object only)
// Note that sysconf(_SC_AIO_MAX) usually provides an unreliable number if it
// provides a number at all.
//
   maxAioPR  = (maxaiopr < 1 ? 8 : maxaiopr);
   maxAioPR2 = maxAioPR * 2;
   XrdXrootdAio::maxAio = (maxaio < maxAioPR ? maxAioPR : maxaio);

// Do some debuging
//
   TRACE(DEBUG, "Max aio/req=" <<maxAioPR
                <<"; aio/srv=" <<XrdXrootdAio::maxAio
                <<"; Quantum=" <<Quantum);

// Preallocate a block of AIO request objects AIO I/O objects
//
   if ((arp  =               addBlock())) {arp->Clear(0); arp->Recycle(0);}
   if ((aiop = XrdXrootdAio::addBlock())) aiop->Recycle();
}

/******************************************************************************/
/*                 X r d X r o o t d A i o R e q : : R e a d                  */
/******************************************************************************/
  
int XrdXrootdAioReq::Read()
{
   int rc;
   XrdXrootdAio *aiop;

// Get an aio object. No need to lock since we are simply double buffered.
// In fact, thsi interface is called only once to start the I/O. After the
// initial call, the I/O is propelled via aio redrive logic.
//
   if (!(aiop = aioFree)) return -ENOBUFS;
   aioFree = aiop->Next;
   aiop->Next = 0;

// Fill out the aiocb block
//
// aiop->sfsAio.aio_buf     = aiop->buffp->buff  (Filled in by Alloc())
   aiop->sfsAio.aio_offset  = myOffset;
   aiop->sfsAio.aio_nbytes  = (aiop->buffp->bsize>myIOLen ? myIOLen
                                                          : aiop->buffp->bsize);
// aiop->sfsAio.aio_reqprio = 0;                 (Filled in by XrdSfs Construct)
// aiop->sfsAio.aio_fildes  =                    (Filled in by XrdSfs aio read)

// Fire up the I/O (no need to lock this as it's simple double buffering)
//
   myIOLen  -= aiop->sfsAio.aio_nbytes;
   myOffset += aiop->sfsAio.aio_nbytes;
   numActive++;
   if ((rc = myFile->XrdSfsp->read((XrdSfsAio *)aiop))) 
      {numActive--; Recycle();} // Only 1!

// All done
//
   return rc;
}

/******************************************************************************/
/*              X r d X r o o t d A i o R e q : : R e c y c l e               */
/******************************************************************************/
  
void XrdXrootdAioReq::Recycle(int dref, XrdXrootdAio *oldp)
{
   XrdXrootdAio *aiop;

// Recycle any hanging aio object
//
// TRACE(DEBUG, "Recycling aioreq; dref=" <<dref <<" link=" <<Link);
   if (oldp) oldp->Recycle();

// When dref is <0, Recycle() was called to terminate an already started 
// operation. Make sure that everything is drained prior to recycling.
// Warining, the caller may not have the aioReq lock held in this case.
//
   if (dref < 0)
      {Lock();
       if (numActive)
          {aioError = -1; respDone = 1;
           UnLock();
           return;
          }
       UnLock();
      }

// Get rid of any aio objects that we might have
//
   while((aiop = aioDone)) {aioDone = aiop->Next; aiop->Recycle();}
   while((aiop = aioFree)) {aioFree = aiop->Next; aiop->Recycle();}

// If we have a link and it should be derefernced, do so now
//
   if (Link && dref && aioType != 'w') Link->setRef(-1);

// If this object is locked; remove the lock (caller must have obatined it)
//
   if (isLocked) UnLock();

// Put ourselves on the free queue
//
   rqMutex.Lock();
   Next = rqFirst;
   rqFirst = this;
   rqMutex.UnLock();
}

/******************************************************************************/
/*                X r d X r o o t d A i o R e q : : W r i t e                 */
/******************************************************************************/
  
int XrdXrootdAioReq::Write(XrdXrootdAio *aiop)
{
   int rc;

// For write, the aiop should or will be filled in as follows:
//
// aiop->sfsAio.aio_buf     = aiop->buffp->buff  (Filled in by Alloc())
// aiop->sfsAio.aio_offset  = Filled in by caller
// aiop->sfsAio.aio_nbytes  = Filled in by caller
// aiop->sfsAio.aio_reqprio = 0                  (Filled in by XrdSfs Construct)
// aiop->sfsAio.aio_fildes  =                    (Filled in by XrdSfs aio write)

// Fire up the I/O. Be optimistic that this will succeed.
//
   Lock(); numActive++; UnLock();
   if ((rc = myFile->XrdSfsp->write((XrdSfsAio *)aiop))) 
      {Lock(); numActive--; UnLock(); Recycle(-1);}

// All done
//
   return rc;
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*             X r d X r o o t d A i o R e q : : a d d B l o c k              */
/******************************************************************************/
  
XrdXrootdAioReq *XrdXrootdAioReq::addBlock()
{
   const int numalloc = 4096/sizeof(XrdXrootdAioReq);
   int i = numalloc;
   XrdXrootdAioReq *arp;

   if (!numalloc) return new XrdXrootdAioReq();
   TRACE(DEBUG, "Adding " <<numalloc <<" aioreq objects.");

   if ((arp = new XrdXrootdAioReq[numalloc]()))
      while(--i) {arp->Next = rqFirst; rqFirst = arp; arp++;}

   return arp;
}
  
/******************************************************************************/
/*                                 C l e a r                                  */
/******************************************************************************/

void XrdXrootdAioReq::Clear(XrdLink *lnkp)
{
Next      = 0;
myOffset  = 0;
myIOLen   = 0;
Instance  = 0;
Link      = lnkp;
myFile    = 0;
aioDone   = 0;
aioFree   = 0;
numActive = 0;
aioTotal  = 0;
aioError  = 0;
aioType   = 0;
respDone  = 0;
isLocked  = 0;
reDrive   = 0;
}
  
/******************************************************************************/
/*              X r d X r o o t d A i o R e q : : e n d R e a d               */
/******************************************************************************/
  
void XrdXrootdAioReq::endRead()
{
   XrdXrootdAio *aiop;
   int rc;

// For read requests, schedule the next read request and send the data we
// already have. Since we don't know if that read will complete before we
// can send the data of the just completed read, we must lock the AioReq.
// We do know that if we have the lock, absolutely nothing is in transit.
//
   Lock();
   numActive--;

// Do a sanity check. The link should not have changed hands but stranger
// things have happened.
//
   if (!(Link->isInstance(Instance))) {Scuttle("aio read"); return;}

// Dequeue the completed request (we know we're just double buffered but the
// queueing is structured so this works even we're n-buffered.
//
   aiop = aioDone;
   aioDone = aiop->Next;

// If we encountered an error, send off the error message now and terminate
//
   if (aioError
   || (myIOLen > 0 && aiop->Result == aiop->buffp->bsize && (aioError=Read())))
      {sendError((char *)aiop->TIdent);
       Recycle(1, aiop);
       return;
      }

// We may or may not have an I/O request in flight. However, send off
// whatever data we have at this point.
//
   rc = (numActive ?
         Response.Send(kXR_oksofar, aiop->buffp->buff, aiop->Result) :
         Response.Send(             aiop->buffp->buff, aiop->Result));

// Stop the operation if no I/O is in flight. Make the request stop-pending if
// we could not send the data to the client.
//
   if (!numActive) 
      {myFile->readCnt += aioTotal;
       Recycle(1, aiop);
      }
      else {aiop->Next = aioFree, aioFree = aiop;
            if (rc < 0) {aioError = -1; respDone = 1;}
            UnLock();
           }
}
  
/******************************************************************************/
/*             X r d X r o o t d A i o R e q : : e n d W r i t e              */
/******************************************************************************/
  
void XrdXrootdAioReq::endWrite()
{

// For write requests, this method is called when all of the I/O has completed
// There is no need to lock this object since nothing is pending. In any case,
// Do a sanity check. The link should not have changed hands but stranger
// things have happened.
//
   if (!(Link->isInstance(Instance))) {Scuttle("aio write"); return;}

// If we encountered an error, send off the error message else indicate all OK
//
   if (aioError) sendError(Link->ID);
      else Response.Send();

// Add in the bytes written. This is approzimate because it is done without
// obtaining any kind of lock. Fortunately, it only statistical in nature.
//
   myFile->writeCnt += aioTotal;

// We are done, simply recycle ouselves.
//
   Recycle();
}

/******************************************************************************/
/*              X r d X r o o t d A i o R e q : : S c u t t l e               */
/******************************************************************************/
  
void XrdXrootdAioReq::Scuttle(const char *opname)
{

// Log this event. We can't trust much of anything at this point.
//
   eDest->Emsg("scuttle",opname,"failed; link reassigned to",Link->ID);

// We can just recycle ourselves at this point since we know we are in a
// transition window where nothing is active w.r.t. this request.
//
   Recycle(0);
}

/******************************************************************************/
/*            X r d X r o o t d A i o R e q : : s e n d E r r o r             */
/******************************************************************************/
  
// Warning! The caller must have appropriately serialized the use of this method

void XrdXrootdAioReq::sendError(char *tident)
{
   char mbuff[4096];
   int rc;

// If a response was sent, don't send one again
//
   if (respDone) return;
   respDone = 1;

// Generate message text. We can't rely on the sfs interface to do this since
// that interface is synchronous.
//
   snprintf(mbuff, sizeof(mbuff)-1, "XrdXrootdAio: Unable to %s %s; %s",
           (aioType == 'r' ? "read" : "write"), myFile->XrdSfsp->FName(),
           eDest->ec2text(aioError));

// Please the error message in the log
//
   eDest->Emsg("aio", tident, mbuff);

// Remap the error from the filesystem
//
   rc = XrdXrootdProtocol::mapError(aioError);

// Send the erro back to the client (ignore any errors)
//
   Response.Send((XErrorCode)rc, mbuff);
}
