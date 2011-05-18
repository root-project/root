#ifndef __XRDXROOTDAIO__
#define __XRDXROOTDAIO__
/******************************************************************************/
/*                                                                            */
/*                       X r d X r o o t d A i o . h h                        */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//        $Id$
  
#include "XProtocol/XPtypes.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdSfs/XrdSfsAio.hh"
#include "Xrd/XrdScheduler.hh"
#include "XrdXrootd/XrdXrootdResponse.hh"

/******************************************************************************/
/*                          X r d X r o o t d A i o                           */
/******************************************************************************/

// The XrdXrootdAio object represents a single aio read or write operation. One
// or more of these are allocated to the XrdXrootdAioReq and passed as upcast
// arguments to the sfs file object to effect asynchronous I/O.

class XrdBuffer;
class XrdBuffManager;
class XrdSysError;
class XrdXrootdAioReq;
class XrdXrootdStats;
  
class XrdXrootdAio : public XrdSfsAio
{
friend class XrdXrootdAioReq;
public:
        XrdBuffer    *buffp;   // -> Buffer object

virtual void          doneRead();

virtual void          doneWrite();

virtual void          Recycle();


              XrdXrootdAio() {Next=0; aioReq=0; buffp=0;}
             ~XrdXrootdAio() {};

private:

static  XrdXrootdAio    *Alloc(XrdXrootdAioReq *arp, int bsize=0);
static  XrdXrootdAio    *addBlock();

static  const char      *TraceID;
static  XrdBuffManager  *BPool;   // -> Buffer Manager
static  XrdScheduler    *Sched;   // -> System Scheduler
static  XrdXrootdStats  *SI;      // -> System Statistics
static  XrdSysMutex      fqMutex; // Locks static data
static  XrdXrootdAio    *fqFirst; // -> Object in free queue
static  int              maxAio;  // Maximum Aio objects we can yet have

        XrdXrootdAio    *Next;    // Chain pointer
        XrdXrootdAioReq *aioReq;  // -> Associated request object
};

/******************************************************************************/
/*                       X r d X r o o t d A i o R e q                        */
/******************************************************************************/

// The XrdXrootdAioReq object represents a complete aio request. It handles
// the appropriate translation of the synchrnous request to an async one,
// provides the redrive logic, and handles ending status.
//
class XrdLink;
class XrdXrootdFile;
class XrdXrootdProtocol;
  
class XrdXrootdAioReq : public XrdJob
{
friend class XrdXrootdAio;
public:

static XrdXrootdAioReq   *Alloc(XrdXrootdProtocol *p, char iot, int numaio=0);

       void               DoIt() {if (aioType == 'r') endRead();
                                     else endWrite();
                                 }

       XrdXrootdAio      *getAio();

inline XrdXrootdAio      *Pop() {XrdXrootdAio *aiop = aioDone;
                                 aioDone = aiop->Next; return aiop;
                                }

inline void               Push(XrdXrootdAio *newp)
                              {newp->Next = aioDone; aioDone = newp;}

static void               Init(int iosize, int maxaiopr, int maxaio=-80);

       int                Read();

       void               Recycle(int deref=1, XrdXrootdAio *aiop=0);

       int                Write(XrdXrootdAio *aiop);

       XrdXrootdAioReq() : XrdJob("aio request") {}
      ~XrdXrootdAioReq() {} // Never called

private:

        void               Clear(XrdLink *lnkp);

static  XrdXrootdAioReq   *addBlock();
        void               endRead();
        void               endWrite();
inline  void               Lock() {aioMutex.Lock(); isLocked = 1;}
        void               Scuttle(const char *opname);
        void               sendError(char *tident);
inline  void               UnLock() {isLocked = 0; aioMutex.UnLock();}

static  const char        *TraceID;
static  XrdSysError       *eDest;      // -> Error Object
static  XrdSysMutex        rqMutex;    // Locks static data
static  XrdXrootdAioReq   *rqFirst;    // -> Object in free queue
static  int                QuantumMin; // aio segment size (Quantum/2)
static  int                Quantum;    // aio segment size
static  int                QuantumMax; // aio segment size (Quantum*2)
static  int                maxAioPR;   // aio objects per request (max)
static  int                maxAioPR2;  // aio objects per request (max*2)

        XrdSysMutex        aioMutex;  // Locks private data
        XrdXrootdAioReq   *Next;      // -> Chain pointer

        off_t              myOffset;  // Next offset    (used for read's only)
        int                myIOLen;   // Size remaining (read and write end)
        unsigned int       Instance;  //    Network Link Instance
        XrdLink           *Link;      // -> Network link
        XrdXrootdFile     *myFile;    // -> Associated file

        XrdXrootdAio      *aioDone;   // Next aiocb that completed
        XrdXrootdAio      *aioFree;   // Next aiocb that we can use
        int                numActive; // Number of aio requests outstanding
        int                aioTotal;  // Actual number of disk bytes transferred
        int                aioError;  // First errno encounetered
        char               aioType;   // 'r' or 'w' or 's'
        char               respDone;  // 1 -> Response has been sent
        char               isLocked;  // 1 -> Object lock being held
        char               reDrive;   // 1 -> Link redrive is needed

        XrdXrootdResponse  Response;  // Copy of the original response object
};
#endif
