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

const char *XrdCpMthrQueueCVSID = "$Id$";

#include "XrdClient/XrdCpMthrQueue.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdClient/XrdClientDebug.hh"

XrdCpMthrQueue::XrdCpMthrQueue(): fReadSem(0) {
   // Constructor

   fMsgQue.Clear();
   fTotSize = 0;
}

XrdCpMthrQueue::~XrdCpMthrQueue() {
   // Destructor


}

int XrdCpMthrQueue::PutBuffer(void *buf, long long offs, int len) {
   XrdCpMessage *m;
   bool wantstowait = FALSE;

   {
      XrdSysMutexHelper mtx(fMutex);
      
      if (fTotSize > CPMTQ_BUFFSIZE) wantstowait = TRUE;
   }
   
   if (wantstowait) fWriteCnd.Wait(60);

   m = new XrdCpMessage;
   m->offs = offs;
   m->buf = buf;
   m->len = len;

   // Put message in the list
   {
      XrdSysMutexHelper mtx(fMutex);
    
      fMsgQue.Push_back(m);
      fTotSize += len;
   }
    
   fReadSem.Post(); 

   return 0;
}

int XrdCpMthrQueue::GetBuffer(void **buf, long long &offs, int &len) {
   XrdCpMessage *res;

   res = 0;
 
   // If there is no data for one hour, then give up with an error
   if (!fReadSem.Wait(3600)) {
	 XrdSysMutexHelper mtx(fMutex);

      	 if (fMsgQue.GetSize() > 0) {

	    // If there are messages to dequeue, we pick the oldest one
	    res = fMsgQue.Pop_front();
	    if (res) fTotSize -= res->len;
	 }
      }


   if (res) {
      *buf = res->buf;
      len = res->len;
      offs = res->offs;
      delete res;
      fWriteCnd.Signal();
   }

   return (res != 0);
}


void XrdCpMthrQueue::Clear() {
   void *buf;
   int len;
   long long offs;

   while (GetBuffer(&buf, offs, len)) {
      free(buf);
   }

   fTotSize = 0;

}

   
