//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdCpMthrQueue                                                       //
//                                                                      //
// Author: Fabrizio Furano (INFN Padova, 2004)                          //
//                                                                      //
// A thread safe queue to be used for multithreaded producers-consumers //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//       $Id$

#include "XrdSys/XrdSysPthread.hh"
#include "XrdClient/XrdClientVector.hh"
#include "XrdSys/XrdSysSemWait.hh"
#include "XrdSys/XrdSysHeaders.hh"

using namespace std;

struct XrdCpMessage {
   void *buf;
   long long offs;
   int len;
};

// The max allowed size for this queue
// If this value is reached, then the writer has to wait...
#define CPMTQ_BUFFSIZE            50000000

class XrdCpMthrQueue {
 private:
   long                           fTotSize;
   XrdClientVector<XrdCpMessage*>            fMsgQue;      // queue for incoming messages
   int                                       fMsgIter;     // an iterator on it

   XrdSysRecMutex                        fMutex;       // mutex to protect data structures

   XrdSysSemWait                      fReadSem;     // variable to make the reader wait
                                                    // until some data is available
   XrdSysCondVar                      fWriteCnd;    // variable to make the writer wait
                                                    // if the queue is full
 public:

   XrdCpMthrQueue();
   ~XrdCpMthrQueue();

   int PutBuffer(void *buf, long long offs, int len);
   int GetBuffer(void **buf, long long &offs, int &len);
   int GetLength() { return fMsgQue.GetSize(); }
   void Clear();
};
   
