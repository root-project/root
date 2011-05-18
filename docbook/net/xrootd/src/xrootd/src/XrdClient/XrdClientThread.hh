//////////////////////////////////////////////////////////////////////////
//                                                                      //
// XrdClientThread                                                      //
//                                                                      //
// An user friendly thread wrapper                                      //
// Author: F.Furano (INFN, 2005)                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//           $Id$

#ifndef XRC_THREAD_H
#define XRC_THREAD_H

#include "XrdSys/XrdSysPthread.hh"

void * XrdClientThreadDispatcher(void * arg);

class XrdClientThread {
private:
   pthread_t fThr;

   typedef void *(*VoidRtnFunc_t)(void *, XrdClientThread *);
   VoidRtnFunc_t ThreadFunc;
   friend void *XrdClientThreadDispatcher(void *);

 public:
   struct XrdClientThreadArgs {
      void *arg;
      XrdClientThread *threadobj;
   } fArg;
   
   
   XrdClientThread(VoidRtnFunc_t fn) {
#ifndef WIN32
      fThr = 0;
#endif
      ThreadFunc = fn;
   };

   virtual ~XrdClientThread() {

//      Cancel();
   };

   int Cancel() {
      return XrdSysThread::Cancel(fThr);
   };

   int Run(void *arg = 0) {
      fArg.arg = arg;
      fArg.threadobj = this;
      return XrdSysThread::Run(&fThr, XrdClientThreadDispatcher, (void *)&fArg,
			       XRDSYSTHREAD_HOLD, "");
   };

   int Detach() {
      return XrdSysThread::Detach(fThr);
   };

   int Join(void **ret = 0) {
      return XrdSysThread::Join(fThr, ret);
   };

   // these funcs are to be called only from INSIDE the thread loop
   int     SetCancelOn() {
      return XrdSysThread::SetCancelOn();
   };
   int     SetCancelOff() {
      return XrdSysThread::SetCancelOff();
   };
   int     SetCancelAsynchronous() {
      return XrdSysThread::SetCancelAsynchronous();
   };
   int     SetCancelDeferred() {
      return XrdSysThread::SetCancelDeferred();
   };
   void     CancelPoint() {
      XrdSysThread::CancelPoint();
   };

   int MaskSignal(int snum = 0, bool block = 1);
};



#endif
