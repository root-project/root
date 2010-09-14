/******************************************************************************/
/*                                                                            */
/*                    X r d X r o o t d X e q A i o . c c                     */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//        $Id$

const char *XrdXrootdAiocbCVSID = "$Id$";

#include <unistd.h>

#include "Xrd/XrdBuffer.hh"
#include "Xrd/XrdLink.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdOuc/XrdOucErrInfo.hh"
#include "XrdSfs/XrdSfsInterface.hh"
#include "XrdXrootd/XrdXrootdAio.hh"
#include "XrdXrootd/XrdXrootdFile.hh"
#include "XrdXrootd/XrdXrootdProtocol.hh"
#include "XrdXrootd/XrdXrootdTrace.hh"
  
/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/

extern  XrdOucTrace  *XrdXrootdTrace;
  
/******************************************************************************/
/*                             a i o _ E r r o r                              */
/******************************************************************************/

int XrdXrootdProtocol::aio_Error(const char *op, int ecode)
{
   char *etext, buffer[MAXPATHLEN+80], unkbuff[64];

// Get the reason for the error
//
   if (!(etext = eDest.ec2text(ecode)))
      {sprintf(unkbuff, "reason unknown (%d)", ecode); etext = unkbuff;}

// Format the error message
//
    snprintf(buffer,sizeof(buffer),"Unable to %s %s; %s",
                    op, myFile->XrdSfsp->FName(), etext);

// Print it out if debugging is enabled
//
#ifndef NODEBUG
    eDest.Emsg("aio_Error", Link->ID, buffer);
#endif

// Place the error message in the error object and return
//
    myFile->XrdSfsp->error.setErrInfo(ecode, buffer);

// Prepare for recovery
//
   myAioReq = 0;
   return -EIO;
}
  
/******************************************************************************/
/*                              a i o _ R e a d                               */
/******************************************************************************/
  
// Implied Arguments:

// myFile   = file to be read
// myOffset = Offset at which to read
// myIOLen  = Number of bytes to read from file and write to socket

// Returns:
// >0      -> n/a
// =0      -> OK to continue with next operation.
// -EAGAIN -> Revert to synchronous I/O
// <0      -> Error, close link.
  
int XrdXrootdProtocol::aio_Read()
{
   XrdXrootdAioReq *arp;

// Allocate a request object to handle this request and fire off the first
// i/o (they are self-sustaining after that). Any errors at this point will
// force us to revert to synchronous i/o.
//
   if (!(arp=XrdXrootdAioReq::Alloc(this,'r',2)) || arp->Read()) return -EAGAIN;

// All done
//
   return 0;
}

/******************************************************************************/
/*                             a i o _ W r i t e                              */
/******************************************************************************/
  
// Implied Arguments:

// myFile   = file to be read
// myOffset = Offset at which to read
// myIOLen  = Number of bytes to read from file and write to socket
// myStalls = Number of stalls encountered last time we did I/O

// Returns:
// >0           -> Slow link, enable link and wait for more data.
// =0           -> OK to continue with next operation.
// -EAGAIN      -> Revert to synchronous I/O
// -EINPROGRESS -> Ran out of aio objects, leave link disabled
// -EIO         -> File system error, flush link.
// <0           -> Error, close link.
  
int XrdXrootdProtocol::aio_Write()
{

// Allocate a request object to handle this request
//
   if (!(myAioReq = XrdXrootdAioReq::Alloc(this, 'w'))) return -EAGAIN;

// Since the socket is synchronous in delivering data to write; only one
// write async request can occur at one time, though several may be in-flight
// after we drain the socket of data. While draining, we remember the AioReq
// object in case we must suspend operations and start the flow.
//
   return aio_WriteAll();
}

/******************************************************************************/
/*                          a i o _ W r i t e A l l                           */
/******************************************************************************/
  
// myFile   = file to be read
// myOffset = Offset at which to read
// myIOLen  = Number of bytes to read from file and write to socket
// myAioReq = -> Aio Request

// The steps taken are:
// 1) Obtain an aio object. If none available, a redrive will be scheduled for
//    the protocol and we return -EINPROGRESS which will keep the link disabled. 

// 2) Read the data from the link into the buffer using getData().

// 3) If the link is slow, return a 1 which will re-enable the link and
//    redrive the protocol when data is available. We will resume in 
//    aio_WriteCont() when the buffer has the required amount of data.

// 4) If the read from the link indicated an error then abort the operation
//    by recycling the AioReq object which will synchronize in-flight i/o.

// 5) Schedule the aio write. Errors will scuttle the operation and proceed to
//    flush the socket. The write() call will appropriately recycle the AioReq
//    object. We note that no error should be returned if aio resources are 
//    exhausted, the underlying implementation must revert to synchronous 
//    handling. That's a lot of overhead but we'll back off.

int XrdXrootdProtocol::aio_WriteAll()
{
   XrdXrootdAio *aiop;
   size_t Quantum;
   int rc = 0;

   if (myStalls) myStalls--;

   while (myIOLen > 0)
/*1*/    {if (!(aiop = myAioReq->getAio()))
             {Resume = &XrdXrootdProtocol::aio_WriteAll;
              myBlen = 0;
              return -EINPROGRESS;
             }

/*2*/     Quantum = (aiop->buffp->bsize > myIOLen ? myIOLen
                                                  : aiop->buffp->bsize);
          if ((rc = getData("aiodata", aiop->buffp->buff, Quantum)))
/*3*/       {if (rc > 0)
                {Resume = &XrdXrootdProtocol::aio_WriteCont;
                 myBlast = Quantum;
                 myAioReq->Push(aiop);
                 myStalls++;
                 return 1;
                }
/*4*/        myAioReq->Recycle(-1, aiop);
             break;
            }
/*5*/    aiop->sfsAio.aio_nbytes = Quantum;
         aiop->sfsAio.aio_offset = myOffset;
         myIOLen  -= Quantum; myOffset += Quantum;
         if ((rc = myAioReq->Write(aiop))) return aio_Error("write", rc);
         }

// We have completed
//
   if (myStalls <= as_maxstalls) myStalls = 0;
   myAioReq = 0;
   Resume   = 0;
   return rc;
}

/******************************************************************************/
/*                         a i o _ W r i t e C o n t                          */
/******************************************************************************/

// myFile   = file to be written
// myOffset = Offset at which to write
// myIOLen  = Number of bytes to read from socket and write to file
// myBlast  = Number of bytes already read from the socket
// myAio    = Pointer to the XrdXrootdAioReq object.
  
int XrdXrootdProtocol::aio_WriteCont()
{
   XrdXrootdAio *aiop = myAioReq->Pop();
   int rc;

// Write data that was finaly finished comming in. Note that we could simply
// pick up the current aio object without locks since this is synchronized
// via protocol object scheduling (only one can occur at a time).
//
   if ((rc = myAioReq->Write(aiop)))
      {myIOLen  = myIOLen-myBlast;
       return aio_Error("write", rc);
      }
    myOffset += myBlast; myIOLen -= myBlast;

// Either continue the request or return to enable the link
//
   if (myIOLen > 0) return aio_WriteAll();
   myAioReq = 0;
   return 0;
}
