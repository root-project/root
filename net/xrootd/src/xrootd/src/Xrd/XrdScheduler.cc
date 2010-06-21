/******************************************************************************/
/*                                                                            */
/*                       X r d S c h e d u l e r . h h                        */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//        $Id$

const char *XrdSchedulerCVSID = "$Id$";

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <sys/types.h>
#include <sys/wait.h>
#ifdef __macos__
#include <AvailabilityMacros.h>
#endif

#include "Xrd/XrdJob.hh"
#include "Xrd/XrdScheduler.hh"
#include "Xrd/XrdTrace.hh"
#include "XrdSys/XrdSysError.hh"

/******************************************************************************/
/*                        G l o b a l   O b j e c t s                         */
/******************************************************************************/

extern XrdSysError   XrdLog;
  
#ifndef NODEBUG
extern XrdOucTrace   XrdTrace;
#endif

       const char   *XrdScheduler::TraceID = "Sched";

/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/

class XrdSchedulerPID
     {public:
      XrdSchedulerPID *next;
      pid_t            pid;

      XrdSchedulerPID(pid_t newpid, XrdSchedulerPID *prev) 
                        {next = prev; pid = newpid;}
     ~XrdSchedulerPID() {}
     };
  
/******************************************************************************/
/*            E x t e r n a l   T h r e a d   I n t e r f a c e s             */
/******************************************************************************/
  
void *XrdStartReaper(void *carg)
      {XrdScheduler *sp = (XrdScheduler *)carg;
       sp->Reaper();
       return (void *)0;
      }

void *XrdStartTSched(void *carg)
      {XrdScheduler *sp = (XrdScheduler *)carg;
       sp->TimeSched();
       return (void *)0;
      }

void *XrdStartWorking(void *carg)
      {XrdScheduler *sp = (XrdScheduler *)carg;
       sp->Run();
       return (void *)0;
      }

/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdScheduler::XrdScheduler(int minw, int maxw, int maxi)
              : XrdJob("underused thread monitor"),
                WorkAvail(0, "sched work")
{
    min_Workers =  minw;
    max_Workers =  maxw;
    max_Workidl =  maxi;
    num_Workers =  0;
    num_JobsinQ =  0;
    stk_Workers =  maxw - (maxw/4*3);
    idl_Workers =  0;
    num_Jobs    =  0;
    max_QLength =  0;
    num_TCreate =  0;
    num_TDestroy=  0;
    num_Limited =  0;
    firstPID    =  0;
    WorkFirst = WorkLast = TimerQueue = 0;
}
 
/******************************************************************************/
/*                            D e s t r u c t o r                             */
/******************************************************************************/

XrdScheduler::~XrdScheduler()  // The scheduler is never deleted!
{
}
 
/******************************************************************************/
/*                                C a n c e l                                 */
/******************************************************************************/

void XrdScheduler::Cancel(XrdJob *jp)
{
   XrdJob *p, *pp = 0;

// Lock the queue
//
   TimerMutex.Lock();

// Find the matching job, if any
//
   p = TimerQueue;
   while(p && p != jp) {pp = p; p = p->NextJob;}

// Delete the job element
//
   if (p)
      {if (pp) pp->NextJob = p->NextJob;
          else  TimerQueue = p->NextJob;
       TRACE(SCHED, "time event " <<jp->Comment <<" cancelled");
      }

// All done
//
   TimerMutex.UnLock();
}
  
/******************************************************************************/
/*                                  D o I t                                   */
/******************************************************************************/

void XrdScheduler::DoIt()
{
   int num_kill, num_idle;

// Now check if there are too many idle threads (kill them if there are)
//
   if (!num_JobsinQ)
      {DispatchMutex.Lock(); num_idle = idl_Workers; DispatchMutex.UnLock();
       num_kill = num_idle - min_Workers;
       TRACE(SCHED, num_Workers <<" threads; " <<num_idle <<" idle");
       if (num_kill > 0)
          {if (num_kill > 1) num_kill = num_kill/2;
           SchedMutex.Lock();
           num_Layoffs = num_kill;
           while(num_kill--) WorkAvail.Post();
           SchedMutex.UnLock();
          }
      }

// Check if we should reschedule ourselves
//
   if (max_Workidl > 0) Schedule((XrdJob *)this, max_Workidl+time(0));
}

/******************************************************************************/
/*                                  F o r k                                   */
/******************************************************************************/
  
// This entry exists solely so that we can start a reaper thread for processes
//
pid_t XrdScheduler::Fork(const char *id)
{
   static int  retc, ReaperStarted = 0;
   pthread_t tid;
   pid_t pid;

// Fork
//
   if ((pid = fork()) < 0)
      {XrdLog.Emsg("Scheduler",errno,"fork to handle",id);
       return pid;
      }
   if (!pid) return pid;

// Obtain the status of the reaper thread.
//
   ReaperMutex.Lock();
   firstPID = new XrdSchedulerPID(pid, firstPID);
   retc = ReaperStarted;
   ReaperStarted = 1;
   ReaperMutex.UnLock();

// Start the reaper thread if it has not started.
//
   if (!retc)
      if ((retc = XrdSysThread::Run(&tid, XrdStartReaper, (void *)this,
                                    0, "Process reaper")))
         {XrdLog.Emsg("Scheduler", retc, "create reaper thread");
          ReaperStarted = 0;
         }

   return pid;
}

/******************************************************************************/
/*                                R e a p e r                                 */
/******************************************************************************/
  
void *XrdScheduler::Reaper()
{
   int status;
   pid_t pid;
   XrdSchedulerPID *tp, *ptp, *xtp;
#if defined(__macos__) && !defined(MAC_OS_X_VERSION_10_5)
   struct timespec ts = { 1, 0 };
#else
   sigset_t Sset;
   int signum;

// Set up for signal handling. Note: main() must block this signal at start)
//
   sigemptyset(&Sset);
   sigaddset(&Sset, SIGCHLD);
#endif

// Wait for all outstanding children
//
   do {ReaperMutex.Lock();
       tp = firstPID; ptp = 0;
       while(tp)
            {do {pid = waitpid(tp->pid, &status, WNOHANG);}
                while (pid < 0 && errno == EINTR);
             if (pid > 0)
                {if (TRACING(TRACE_SCHED)) traceExit(pid, status);
                 xtp = tp; tp = tp->next;
                 if (ptp) ptp->next = tp;
                    else   firstPID = tp;
                 delete xtp;
                } else {ptp = tp; tp = tp->next;}
             }
       ReaperMutex.UnLock();
#if defined(__macos__) && !defined(MAC_OS_X_VERSION_10_5)
       // Mac OS X sigwait() is broken on <= 10.4.
      } while (nanosleep(&ts, 0) <= 0);
#else
      } while(sigwait(&Sset, &signum) >= 0);
#endif
   return (void *)0;
}

/******************************************************************************/
/*                                   R u n                                    */
/******************************************************************************/
  
void XrdScheduler::Run()
{
   int waiting;
   XrdJob *jp;

// Wait for work then do it (an endless task for a worker thread)
//
   do {do {DispatchMutex.Lock();          idl_Workers++;DispatchMutex.UnLock();
           WorkAvail.Wait();
           DispatchMutex.Lock();waiting = --idl_Workers;DispatchMutex.UnLock();
           SchedMutex.Lock();
           if ((jp = WorkFirst))
              {if (!(WorkFirst = jp->NextJob)) WorkLast = 0;
               if (num_JobsinQ) num_JobsinQ--;
                  else XrdLog.Emsg("Scheduler","Job queue count underflow!");
              } else {
               num_JobsinQ = 0;
               if (num_Layoffs > 0)
                  {num_Layoffs--;
                   if (waiting)
                      {num_TDestroy++; num_Workers--;
                       TRACE(SCHED, "terminating thread; workers=" <<num_Workers);
                       SchedMutex.UnLock();
                       return;
                      }
                  }
              }
           SchedMutex.UnLock();
          } while(!jp);

    // Check if we should hire a new worker (we always want 1 idle thread)
    // before running this job.
    //
       if (!waiting) hireWorker();
       TRACE(SCHED, "running " <<jp->Comment <<" inq=" <<num_JobsinQ);
       jp->DoIt();
      } while(1);
}
 
/******************************************************************************/
/*                              S c h e d u l e                               */
/******************************************************************************/
  
void XrdScheduler::Schedule(XrdJob *jp)
{
// Lock down our data area
//
   SchedMutex.Lock();

// Place the request on the queue and broadcast it
//
   jp->NextJob  = 0;
   if (WorkFirst)
      {WorkLast->NextJob = jp;
       WorkLast = jp;
      } else {
       WorkFirst = jp;
       WorkLast  = jp;
      }
   WorkAvail.Post();

// Calculate statistics
//
   num_Jobs++;
   num_JobsinQ++;
   if (num_JobsinQ > max_QLength) max_QLength = num_JobsinQ;

// Unlock the data area and return
//
   SchedMutex.UnLock();
}

/******************************************************************************/
  
void XrdScheduler::Schedule(int numjobs, XrdJob *jfirst, XrdJob *jlast)
{

// Lock down our data area
//
   SchedMutex.Lock();

// Place the request list on the queue
//
   jlast->NextJob = 0;
   if (WorkFirst)
      {WorkLast->NextJob = jfirst;
       WorkLast = jlast;
      } else {
       WorkFirst = jfirst;
       WorkLast  = jlast;
      }

// Calculate statistics
//
   num_Jobs    += numjobs;
   num_JobsinQ += numjobs;
   if (num_JobsinQ > max_QLength) max_QLength = num_JobsinQ;

// Indicate number of jobs to work on
//
   while(numjobs--) WorkAvail.Post();

// Unlock the data area and return
//
   SchedMutex.UnLock();
}

/******************************************************************************/

void XrdScheduler::Schedule(XrdJob *jp, time_t atime)
{
   XrdJob *pp = 0, *p;

// Cancel this event, if scheduled
//
   Cancel(jp);

// Lock the queue
//
   TRACE(SCHED, "scheduling " <<jp->Comment <<" in " <<atime-time(0) <<" seconds");
   jp->SchedTime = atime;
   TimerMutex.Lock();

// Find the insertion point for the work element
//
   p = TimerQueue;
   while(p && p->SchedTime <= atime) {pp = p; p = p->NextJob;}

// Insert the job element
//
   jp->NextJob = p;
   if (pp)  pp->NextJob = jp;
      else {TimerQueue = jp; TimerRings.Signal();}

// All done
//
   TimerMutex.UnLock();
}

/******************************************************************************/
/*                              s e t P a r m s                               */
/******************************************************************************/
  
void XrdScheduler::setParms(int minw, int maxw, int avlw, int maxi, int once)
{
   static int isSet = 0;

// Lock the data area and check for 1-time set
//
   SchedMutex.Lock();
   if (once && isSet) {SchedMutex.UnLock(); return;}
   isSet = 1;

// get a consistent view of all the values
//
   if (maxw <= 0) maxw = max_Workers;
   if (minw < 0) minw = (maxw/10 ? maxw/10 : 1);
      else if (minw > maxw) minw = maxw;
   if (avlw < 0) avlw = maxw/4*3;
      else if (avlw > maxw) avlw = maxw;

// Set the values
//
   min_Workers = minw;
   max_Workers = maxw;
   stk_Workers = maxw - avlw;
   if (maxi >=0)  max_Workidl = maxi;

// Unlock the data area
//
   SchedMutex.UnLock();

// If we have an idle interval, schedule the idle check
//
   if (maxi > 0)
      {Cancel((XrdJob *)this);
       Schedule((XrdJob *)this, (time_t)maxi+time(0));
      }

// Debug the info
//
   TRACE(SCHED,"Set min_Workers=" <<min_Workers <<" max_Workers=" <<max_Workers);
   TRACE(SCHED,"Set stk_Workers=" <<stk_Workers <<" max_Workidl=" <<max_Workidl);
}

/******************************************************************************/
/*                                 S t a r t                                  */
/******************************************************************************/
  
void XrdScheduler::Start() // Serialized one time call!
{
    int retc, numw;
    pthread_t tid;

// Start a time based scheduler
//
   if ((retc = XrdSysThread::Run(&tid, XrdStartTSched, (void *)this,
                                 XRDSYSTHREAD_BIND, "Time scheduler")))
      XrdLog.Emsg("Scheduler", retc, "create time scheduler thread");

// If we an idle interval, schedule the idle check
//
   if (max_Workidl > 0) Schedule((XrdJob *)this, (time_t)max_Workidl+time(0));

// Start 1/3 of the minimum number of threads
//
   if (!(numw = min_Workers/3)) numw = 2;
   while(numw--) hireWorker(0);

// Unlock the data area
//
   TRACE(SCHED, "Starting with " <<num_Workers <<" workers" );
}

/******************************************************************************/
/*                                 S t a t s                                  */
/******************************************************************************/
  
int XrdScheduler::Stats(char *buff, int blen, int do_sync)
{
    int cnt_Jobs, cnt_JobsinQ, xam_QLength, cnt_Workers, cnt_idl;
    int cnt_TCreate, cnt_TDestroy, cnt_Limited;
    static char statfmt[] = "<stats id=\"sched\"><jobs>%d</jobs>"
                "<inq>%d</inq><maxinq>%d</maxinq>"
                "<threads>%d</threads><idle>%d</idle>"
                "<tcr>%d</tcr><tde>%d</tde>"
                "<tlimr>%d</tlimr></stats>";

// If only length wanted, do so
//
   if (!buff) return sizeof(statfmt) + 16*8;

// Get values protected by the Dispatch lock (avoid lock if no sync needed)
//
   if (do_sync) DispatchMutex.Lock();
   cnt_idl = idl_Workers;
   if (do_sync) DispatchMutex.UnLock();

// Get values protected by the Scheduler lock (avoid lock if no sync needed)
//
   if (do_sync) SchedMutex.Lock();
   cnt_Workers = num_Workers;
   cnt_Jobs    = num_Jobs;
   cnt_JobsinQ = num_JobsinQ;
   xam_QLength = max_QLength;
   cnt_TCreate = num_TCreate;
   cnt_TDestroy= num_TDestroy;
   cnt_Limited = num_Limited;
   if (do_sync) SchedMutex.UnLock();

// Format the stats and return them
//
   return snprintf(buff, blen, statfmt, cnt_Jobs, cnt_JobsinQ, xam_QLength,
                   cnt_Workers, cnt_idl, cnt_TCreate, cnt_TDestroy,
                   cnt_Limited);
}

/******************************************************************************/
/*                             T i m e S c h e d                              */
/******************************************************************************/
  
void XrdScheduler::TimeSched()
{
   XrdJob *jp;
   int wtime;

// Continuous loop until we find some work here
//
   do {TimerMutex.Lock();
       if (TimerQueue) wtime = TimerQueue->SchedTime-time(0);
          else wtime = 60*60;
       if (wtime > 0)
          {TimerMutex.UnLock();
           TimerRings.Wait(wtime);
          } else {
           jp = TimerQueue;
           TimerQueue = jp->NextJob;
           Schedule(jp);
           TimerMutex.UnLock();
          }
       } while(1);
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                           h i r e   W o r k e r                            */
/******************************************************************************/
  
void XrdScheduler::hireWorker(int dotrace)
{
   pthread_t tid;
   int retc;

// First check if we reached the maximum number of workers
//
   SchedMutex.Lock();
   if (num_Workers >= max_Workers)
      {num_Limited++;
       if ((num_Limited & 4095) == 1)
           XrdLog.Emsg("Scheduler","Thread limit has been reached!");
       SchedMutex.UnLock();
       return;
      }
   num_Workers++;
   num_TCreate++;
   SchedMutex.UnLock();

// Start a new thread. We do this without the schedMutex to avoid hang-ups. If
// we can't start a new thread, we recalculate the maximum number we can.
//
   retc = XrdSysThread::Run(&tid, XrdStartWorking, (void *)this, 0, "Worker");

// Now check the results and correct if we couldn't start the thread
//
   if (retc)
      {XrdLog.Emsg("Scheduler", retc, "create worker thread");
       SchedMutex.Lock();
       num_Workers--;
       num_TCreate--;
       max_Workers = num_Workers;
       min_Workers = (max_Workers/10 ? max_Workers/10 : 1);
       stk_Workers = max_Workers/4*3;
       SchedMutex.UnLock();
      } else if (dotrace) TRACE(SCHED, "Now have " <<num_Workers <<" workers" );
}
 
/******************************************************************************/
/*                             t r a c e E x i t                              */
/******************************************************************************/
  
void XrdScheduler::traceExit(pid_t pid, int status)
{  const char *why;
   int   retc;

   if (WIFEXITED(status))
      {retc = WEXITSTATUS(status);
       why = " exited with rc=";
      } else if (WIFSIGNALED(status))
                {retc = WTERMSIG(status);
                 why = " killed with signal ";
                } else {retc = 0;
                        why = " changed state ";
                       }
   TRACE(SCHED, "Process " <<pid <<why <<retc);
}
