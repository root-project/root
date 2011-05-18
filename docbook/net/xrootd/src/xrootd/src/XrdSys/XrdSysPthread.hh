#ifndef __SYS_PTHREAD__
#define __SYS_PTHREAD__
/******************************************************************************/
/*                                                                            */
/*                      X r d S y s P t h r e a d . h h                       */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//        $Id$

#include <errno.h>
#ifdef WIN32
#define HAVE_STRUCT_TIMESPEC 1
#endif
#include <pthread.h>
#include <signal.h>
#ifdef AIX
#include <sys/sem.h>
#else
#include <semaphore.h>
#endif

#include "XrdSys/XrdSysError.hh"

/******************************************************************************/
/*                         X r d S y s C o n d V a r                          */
/******************************************************************************/
  
// XrdSysCondVar implements the standard POSIX-compliant condition variable.
//               Methods correspond to the equivalent pthread condvar functions.

class XrdSysCondVar
{
public:

inline void  Lock()           {pthread_mutex_lock(&cmut);}

inline void  Signal()         {if (relMutex) pthread_mutex_lock(&cmut);
                               pthread_cond_signal(&cvar);
                               if (relMutex) pthread_mutex_unlock(&cmut);
                              }

inline void  Broadcast()      {if (relMutex) pthread_mutex_lock(&cmut);
                               pthread_cond_broadcast(&cvar);
                               if (relMutex) pthread_mutex_unlock(&cmut);
                              }

inline void  UnLock()         {pthread_mutex_unlock(&cmut);}

       int   Wait();
       int   Wait(int sec);
       int   WaitMS(int msec);

      XrdSysCondVar(      int   relm=1, // 0->Caller will handle lock/unlock
                    const char *cid=0   // ID string for debugging only
                   ) {pthread_cond_init(&cvar, NULL);
                      pthread_mutex_init(&cmut, NULL);
                      relMutex = relm; condID = (cid ? cid : "unk");
                     }
     ~XrdSysCondVar() {pthread_cond_destroy(&cvar);
                       pthread_mutex_destroy(&cmut);
                      }
private:

pthread_cond_t  cvar;
pthread_mutex_t cmut;
int             relMutex;
const char     *condID;
};



/******************************************************************************/
/*                     X r d S y s C o n d V a r H e l p e r                  */
/******************************************************************************/

// XrdSysCondVarHelper is used to implement monitors with the Lock of a a condvar.
//                     Monitors are used to lock
//                     whole regions of code (e.g., a method) and automatically
//                     unlock with exiting the region (e.g., return). The
//                     methods should be self-evident.
  
class XrdSysCondVarHelper
{
public:

inline void   Lock(XrdSysCondVar *CndVar)
                  {if (cnd) {if (cnd != CndVar) cnd->UnLock();
                                else return;
                            }
                   CndVar->Lock();
                   cnd = CndVar;
                  };

inline void UnLock() {if (cnd) {cnd->UnLock(); cnd = 0;}}

            XrdSysCondVarHelper(XrdSysCondVar *CndVar=0)
                 {if (CndVar) CndVar->Lock();
                  cnd = CndVar;
                 }
            XrdSysCondVarHelper(XrdSysCondVar &CndVar) {
	         CndVar.Lock();
		 cnd = &CndVar;
                 }

           ~XrdSysCondVarHelper() {if (cnd) UnLock();}
private:
XrdSysCondVar *cnd;
};


/******************************************************************************/
/*                           X r d S y s M u t e x                            */
/******************************************************************************/

// XrdSysMutex implements the standard POSIX mutex. The methods correspond
//             to the equivalent pthread mutex functions.
  
class XrdSysMutex
{
public:

inline int CondLock()
       {if (pthread_mutex_trylock( &cs )) return 0;
        return 1;
       }

inline void   Lock() {pthread_mutex_lock(&cs);}

inline void UnLock() {pthread_mutex_unlock(&cs);}

        XrdSysMutex() {pthread_mutex_init(&cs, NULL);}
       ~XrdSysMutex() {pthread_mutex_destroy(&cs);}

protected:

pthread_mutex_t cs;
};

/******************************************************************************/
/*                         X r d S y s R e c M u t e x                        */
/******************************************************************************/

// XrdSysRecMutex implements the recursive POSIX mutex. The methods correspond
//             to the equivalent pthread mutex functions.
  
class XrdSysRecMutex: public XrdSysMutex
{
public:

XrdSysRecMutex();

};


/******************************************************************************/
/*                     X r d S y s M u t e x H e l p e r                      */
/******************************************************************************/

// XrdSysMutexHelper us ised to implement monitors. Monitors are used to lock
//                   whole regions of code (e.g., a method) and automatically
//                   unlock with exiting the region (e.g., return). The
//                   methods should be self-evident.
  
class XrdSysMutexHelper
{
public:

inline void   Lock(XrdSysMutex *Mutex)
                  {if (mtx) {if (mtx != Mutex) mtx->UnLock();
                                else return;
                            }
                   Mutex->Lock();
                   mtx = Mutex;
                  };

inline void UnLock() {if (mtx) {mtx->UnLock(); mtx = 0;}}

            XrdSysMutexHelper(XrdSysMutex *mutex=0)
                 {if (mutex) mutex->Lock();
                  mtx = mutex;
                 }
            XrdSysMutexHelper(XrdSysMutex &mutex) {
	         mutex.Lock();
		 mtx = &mutex;
                 }

           ~XrdSysMutexHelper() {if (mtx) UnLock();}
private:
XrdSysMutex *mtx;
};

/******************************************************************************/
/*                       X r d S y s S e m a p h o r e                        */
/******************************************************************************/

// XrdSysSemaphore implements the classic counting semaphore. The methods
//                 should be self-evident. Note that on certain platforms
//                 semaphores need to be implemented based on condition
//                 variables since no native implementation is available.
  
#ifdef __macos__
class XrdSysSemaphore
{
public:

       int  CondWait();

       void Post();

       void Wait();

  XrdSysSemaphore(int semval=1,const char *cid=0) : semVar(0, cid)
                                  {semVal = semval; semWait = 0;}
 ~XrdSysSemaphore() {}

private:

XrdSysCondVar semVar;
int           semVal;
int           semWait;
};

#else

class XrdSysSemaphore
{
public:

inline int  CondWait()
       {while(sem_trywait( &h_semaphore ))
             {if (errno == EAGAIN) return 0;
              if (errno != EINTR) { throw "sem_CondWait() failed";}
             }
        return 1;
       }

inline void Post() {if (sem_post(&h_semaphore))
                       {throw "sem_post() failed";}
                   }

inline void Wait() {while (sem_wait(&h_semaphore))
                          {if (EINTR != errno) 
                              {throw "sem_wait() failed";}
                          }
                   }

  XrdSysSemaphore(int semval=1, const char * =0)
                               {if (sem_init(&h_semaphore, 0, semval))
				   {throw "sem_init() failed";}
                               }
 ~XrdSysSemaphore() {if (sem_destroy(&h_semaphore))
                       {throw "sem_destroy() failed";}
                   }

private:

sem_t h_semaphore;
};
#endif

/******************************************************************************/
/*                          X r d S y s T h r e a d                           */
/******************************************************************************/
  
// The C++ standard makes it impossible to link extern "C" methods with C++
// methods. Thus, making a full thread object is nearly impossible. So, this
// object is used as the thread manager. Since it is static for all intense
// and purposes, one does not need to create an instance of it.
//

// Options to Run()
//
// BIND creates threads that are bound to a kernel thread.
//
#define XRDSYSTHREAD_BIND 0x001

// HOLD creates a thread that needs to be joined to get its ending value.
//      Otherwise, a detached thread is created.
//
#define XRDSYSTHREAD_HOLD 0x002

class XrdSysThread
{
public:

static int          Cancel(pthread_t tid) {return pthread_cancel(tid);}

static int          Detach(pthread_t tid) {return pthread_detach(tid);}


static  int  SetCancelOff() {
      return pthread_setcancelstate(PTHREAD_CANCEL_DISABLE, 0);
 };

static  int  Join(pthread_t tid, void **ret) {
   return pthread_join(tid, ret);
 };

static  int  SetCancelOn() {
      return pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, 0);
 };

static  int  SetCancelAsynchronous() {
      return pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, 0);
 };

static int  SetCancelDeferred() {
      return pthread_setcanceltype(PTHREAD_CANCEL_DEFERRED, 0);
 };

static void  CancelPoint() {
      pthread_testcancel();
 };


static pthread_t    ID(void)              {return pthread_self();}

static int          Kill(pthread_t tid)   {return pthread_cancel(tid);}

static unsigned long Num(void)
                       {if (!initDone) doInit();
                        return (unsigned long)pthread_getspecific(threadNumkey);
                       }

static int          Run(pthread_t *, void *(*proc)(void *), void *arg, 
                        int opts=0, const char *desc = 0);

static int          Same(pthread_t t1, pthread_t t2)
                        {return pthread_equal(t1, t2);}

static void         setDebug(XrdSysError *erp) {eDest = erp;}

static void         setStackSize(size_t stsz) {stackSize = stsz;}

static int          Signal(pthread_t tid, int snum)
                       {return pthread_kill(tid, snum);}
 
static int          Wait(pthread_t tid);

                    XrdSysThread() {}
                   ~XrdSysThread() {}

private:
static void          doInit(void);
static XrdSysError  *eDest;
static pthread_key_t threadNumkey;
static size_t        stackSize;
static int           initDone;
};
#endif
