#ifndef __SYS_SEMWAIT__
#define __SYS_SEMWAIT__

/******************************************************************************/
/*                       X r d S y s S e m W a i t                            */
/*                                                                            */
/* Author: Fabrizio Furano (INFN, 2005)                                       */
/*                                                                            */
/* A counting semaphore with timed out wait primitive                         */
/******************************************************************************/

//           $Id$

#include "XrdSys/XrdSysPthread.hh"

class XrdSysSemWait {
 public:

   int  CondWait() {

      int rc = 0;
      // Wait until the sempahore value is positive. This will not be starvation
      // free is the OS implements an unfair mutex;
      // Returns 0 if signalled, non-0 if would block
      //

      semVar.Lock();
      if (semVal > 0) semVal--;
      else {
	 rc = 1;
      }

      semVar.UnLock();

      return rc;

   };
   
   void Post() {
      // Add one to the semaphore counter. If we the value is > 0 and there is a
      // thread waiting for the sempagore, signal it to get the semaphore.
      //
      semVar.Lock();

      if (semWait > 0) {
	 semVar.Signal();
	 semWait--;
      }
      else
	 semVal++;
      
      semVar.UnLock();
   };
   
   void Wait()   {
      // Wait until the sempahore value is positive. This will not be starvation
      // free is the OS implements an unfair mutex;
      //

      semVar.Lock();
      if (semVal > 0) semVal--;
      else {
	 semWait++;
	 semVar.Wait();
      }

      semVar.UnLock();

   };

   int Wait(int secs)  {
      int rc = 0;
      // Wait until the sempahore value is positive. This will not be starvation
      // free is the OS implements an unfair mutex;
      // Returns 0 if signalled, non-0 if timeout
      //

      semVar.Lock();
      if (semVal > 0) semVal--;
      else {
	 semWait++;
	 rc = semVar.Wait(secs);
	 if (rc) semWait--;
      }

      semVar.UnLock();

      return rc;
   };

   XrdSysSemWait(int semval=1,const char *cid=0) : semVar(0, cid) {
      semVal = semval; semWait = 0;
   }

   ~XrdSysSemWait() {}

private:

XrdSysCondVar semVar;
int           semVal;
int           semWait;
};



#endif
