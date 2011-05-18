/******************************************************************************/
/*                                                                            */
/*                          X r d O s s A i o . c c                           */
/*                                                                            */
/* (c) 2004 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/
  
//         $Id$

const char *XrdOssAioCVSID = "$Id$";

#include <signal.h>
#include <stdio.h>
#include <unistd.h>
#ifdef _POSIX_ASYNCHRONOUS_IO
#ifdef __FreeBSD__
#include <fcntl.h>
#endif
#ifdef __macos__
#include <sys/aio.h>
#else
#include <aio.h>
#endif
#endif

#include "XrdOss/XrdOssApi.hh"
#include "XrdOss/XrdOssTrace.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdSys/XrdSysPthread.hh"
#include "XrdSfs/XrdSfsAio.hh"

// All AIO interfaces are defined here.
 

// Currently we disable aio support for MacOS because it is way too
// buggy and incomplete. The two major problems are:
// 1) No implementation of sigwaitinfo(). Though we can simulate it...
// 2) Event notification returns an incomplete siginfo structure.
//
#ifdef __macos__
#undef _POSIX_ASYNCHRONOUS_IO
#endif

/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/
  
extern XrdOucTrace OssTrace;
//define tident aiop->TIdent

extern XrdSysError OssEroute;

int   XrdOssFile::AioFailure = 0;

#ifdef _POSIX_ASYNCHRONOUS_IO
#ifdef SIGRTMAX
const int OSS_AIO_READ_DONE  = SIGRTMAX-1;
const int OSS_AIO_WRITE_DONE = SIGRTMAX;
#else
#define OSS_AIO_READ_DONE  SIGUSR1
#define OSS_AIO_WRITE_DONE SIGUSR2
#endif
#endif

/******************************************************************************/
/*                                 F s y n c                                  */
/******************************************************************************/
  
/*
  Function: Async fsync() a file

  Input:    aiop      - A aio request object
*/

int XrdOssFile::Fsync(XrdSfsAio *aiop)
{

#ifdef _POSIX_ASYNCHRONOUS_IO
   int rc;

// Complete the aio request block and do the operation
//
   if (XrdOssSys::AioAllOk)
      {aiop->sfsAio.aio_fildes = fd;
       aiop->sfsAio.aio_sigevent.sigev_signo  = OSS_AIO_WRITE_DONE;
       aiop->TIdent = tident;

      // Start the operation
      //
         if (!(rc = aio_fsync(O_SYNC, &aiop->sfsAio))) return 0;
         if (errno != EAGAIN && errno != ENOSYS) return -errno;

      // Aio failed keep track of the problem (msg every 1024 events). Note
      // that the handling of the counter is sloppy because we do not lock it.
      //
         {int fcnt = AioFailure++;
          if ((fcnt & 0x3ff) == 1) OssEroute.Emsg("aio", errno, "fsync async");
         }
     }
#endif

// Execute this request in a synchronous fashion
//
   if ((aiop->Result = Fsync())) aiop->Result = -errno;

// Simply call the write completion routine and return as if all went well
//
   aiop->doneWrite();
   return 0;
}

/******************************************************************************/
/*                                  R e a d                                   */
/******************************************************************************/

/*
  Function: Async read `blen' bytes from the associated file, placing in 'buff'

  Input:    aiop      - An aio request object

   Output:  <0 -> Operation failed, value is negative errno value.
            =0 -> Operation queued
            >0 -> Operation not queued, system resources unavailable or
                                        asynchronous I/O is not supported.
*/
  
int XrdOssFile::Read(XrdSfsAio *aiop)
{

#ifdef _POSIX_ASYNCHRONOUS_IO
   EPNAME("AioRead");
   int rc;

// Complete the aio request block and do the operation
//
   if (XrdOssSys::AioAllOk)
      {aiop->sfsAio.aio_fildes = fd;
       aiop->sfsAio.aio_sigevent.sigev_signo  = OSS_AIO_READ_DONE;
       aiop->TIdent = tident;
       TRACE(Debug,  "Read " <<aiop->sfsAio.aio_nbytes <<'@'
                             <<aiop->sfsAio.aio_offset <<" started; aiocb="
                             <<std::hex <<aiop <<std::dec);

       // Start the operation
       //
          if (!(rc = aio_read(&aiop->sfsAio))) return 0;
          if (errno != EAGAIN && errno != ENOSYS) return -errno;

      // Aio failed keep track of the problem (msg every 1024 events). Note
      // that the handling of the counter is sloppy because we do not lock it.
      //
         {int fcnt = AioFailure++;
          if ((fcnt & 0x3ff) == 1) OssEroute.Emsg("aio", errno, "read async");
         }
     }
#endif

// Execute this request in a synchronous fashion
//
   aiop->Result = this->Read((void *)aiop->sfsAio.aio_buf,
                              (off_t)aiop->sfsAio.aio_offset,
                             (size_t)aiop->sfsAio.aio_nbytes);

// Simple call the read completion routine and return as if all went well
//
   aiop->doneRead();
   return 0;
}

/******************************************************************************/
/*                                 W r i t e                                  */
/******************************************************************************/
  
/*
  Function: Async write `blen' bytes from 'buff' into the associated file

  Input:    aiop      - An aio request object.

   Output:  <0 -> Operation failed, value is negative errno value.
            =0 -> Operation queued
            >0 -> Operation not queued, system resources unavailable or
                                        asynchronous I/O is not supported.
*/
  
int XrdOssFile::Write(XrdSfsAio *aiop)
{
#ifdef _POSIX_ASYNCHRONOUS_IO
   EPNAME("AioWrite");
   int rc;

// Complete the aio request block and do the operation
//
   if (XrdOssSys::AioAllOk)
      {aiop->sfsAio.aio_fildes = fd;
       aiop->sfsAio.aio_sigevent.sigev_signo  = OSS_AIO_WRITE_DONE;
       aiop->TIdent = tident;
       TRACE(Debug, "Write " <<aiop->sfsAio.aio_nbytes <<'@'
                             <<aiop->sfsAio.aio_offset <<" started; aiocb="
                             <<std::hex <<aiop <<std::dec);

       // Start the operation
       //
          if (!(rc = aio_write(&aiop->sfsAio))) return 0;
          if (errno != EAGAIN && errno != ENOSYS) return -errno;

       // Aio failed keep track of the problem (msg every 1024 events). Note
       // that the handling of the counter is sloppy because we do not lock it.
       //
          {int fcnt = AioFailure++;
           if ((fcnt & 0x3ff) == 1) OssEroute.Emsg("Write",errno,"write async");
          }
      }
#endif

// Execute this request in a synchronous fashion
//
   aiop->Result = this->Write((const void *)aiop->sfsAio.aio_buf,
                                     (off_t)aiop->sfsAio.aio_offset,
                                    (size_t)aiop->sfsAio.aio_nbytes);

// Simply call the write completion routine and return as if all went well
//
   aiop->doneWrite();
   return 0;
}

/******************************************************************************/
/*                 X r d O s s S y s   A I O   M e t h o d s                  */
/******************************************************************************/
/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/

int   XrdOssSys::AioAllOk = 0;
  
#if defined(_POSIX_ASYNCHRONOUS_IO) && !defined(HAVE_SIGWTI)
// The folowing is for sigwaitinfo() emulation
//
siginfo_t *XrdOssAioInfoR;
siginfo_t *XrdOssAioInfoW;
extern "C" {extern void XrdOssAioRSH(int, siginfo_t *, void *);}
extern "C" {extern void XrdOssAioWSH(int, siginfo_t *, void *);}
#endif

/******************************************************************************/
/*                               A i o I n i t                                */
/******************************************************************************/
/*
  Function: Initialize for AIO processing.

  Return:   True if successful, false otherwise.
*/

int XrdOssSys::AioInit()
{
#if defined(_POSIX_ASYNCHRONOUS_IO)
   EPNAME("AioInit");
   extern void *XrdOssAioWait(void *carg);
   pthread_t tid;
   int retc;

#ifndef HAVE_SIGWTI
// For those platforms that do not have sigwaitinfo(), we provide the
// appropriate emulation using a signal handler. We actually provide for
// two handlers since we separate reads from writes. To emulate synchronous
// signals, we prohibit one signal hander from interrupting another one.
//
    struct sigaction sa;

    sa.sa_sigaction = XrdOssAioRSH;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sigaddset(&sa.sa_mask, OSS_AIO_WRITE_DONE);
    if (sigaction(OSS_AIO_READ_DONE, &sa, NULL) < 0)
       {OssEroute.Emsg("AioInit", errno, "creating AIO read signal handler; "
                                 "AIO support terminated.");
        return 0;
       }

    sa.sa_sigaction = XrdOssAioWSH;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);
    sigaddset(&sa.sa_mask, OSS_AIO_READ_DONE);
    if (sigaction(OSS_AIO_WRITE_DONE, &sa, NULL) < 0)
       {OssEroute.Emsg("AioInit", errno, "creating AIO write signal handler; "
                                 "AIO support terminated.");
        return 0;
       }
#endif

// The AIO signal handler consists of two thread (one for read and one for
// write) that synhronously wait for AIO events. We assume, blithely, that
// the first two real-time signals have been blocked for all threads.
//
   if ((retc = XrdSysThread::Run(&tid, XrdOssAioWait,
                                (void *)(&OSS_AIO_READ_DONE))) < 0)
      OssEroute.Emsg("AioInit", retc, "creating AIO read signal thread; "
                                 "AIO support terminated.");
#ifdef __FreeBSD__
      else {DEBUG("started AIO read signal thread.");
#else
      else {DEBUG("started AIO read signal thread; tid=" <<(unsigned int)tid);
#endif
            if ((retc = XrdSysThread::Run(&tid, XrdOssAioWait,
                                (void *)(&OSS_AIO_WRITE_DONE))) < 0)
               OssEroute.Emsg("AioInit", retc, "creating AIO write signal thread; "
                                 "AIO support terminated.");
#ifdef __FreeBSD__
               else {DEBUG("started AIO write signal thread.");
#else
               else {DEBUG("started AIO write signal thread; tid=" <<(unsigned int)tid);
#endif
                     AioAllOk = 1;
                    }
           }

// All done
//
   return AioAllOk;
#else
   return 1;
#endif
}

/******************************************************************************/
/*                               A i o W a i t                                */
/******************************************************************************/
  
void *XrdOssAioWait(void *mySigarg)
{
#ifdef _POSIX_ASYNCHRONOUS_IO
   EPNAME("AioWait");
   int mySignum = *((int *)mySigarg);
   const char *sigType = (mySignum == OSS_AIO_READ_DONE ? "read" : "write");
   const int  isRead   = (mySignum == OSS_AIO_READ_DONE);
   sigset_t  mySigset;
   siginfo_t myInfo;
   XrdSfsAio *aiop;
   int rc, numsig;
   ssize_t retval;
#ifndef HAVE_SIGWTI
   extern int sigwaitinfo(const sigset_t *set, siginfo_t *info);
   extern siginfo_t *XrdOssAioInfoR;
   extern siginfo_t *XrdOssAioInfoW;

// We will catch one signal at a time. So, the address of siginfo_t can be
// placed in a global area where the signal handler will find it. We have one
// two places where this can go.
//
   if (isRead) XrdOssAioInfoR = &myInfo;
      else XrdOssAioInfoW = &myInfo;

// Initialize the signal we will be suspended for
//
   sigfillset(&mySigset);
   sigdelset(&mySigset, mySignum);
#else

// Initialize the signal we will be waiting for
//
   sigemptyset(&mySigset);
   sigaddset(&mySigset, mySignum);
#endif

// Simply wait for events and requeue the completed AIO operation
//
   do {do {numsig = sigwaitinfo((const sigset_t *)&mySigset, &myInfo);}
          while (numsig < 0 && errno == EINTR);
       if (numsig < 0)
          {OssEroute.Emsg("AioWait",errno,sigType,"wait for AIO signal");
           XrdOssSys::AioAllOk = 0;
           break;
          }
       if (numsig != mySignum || myInfo.si_code != SI_ASYNCIO)
          {char buff[80];
           sprintf(buff, "%d %d", myInfo.si_code, numsig);
           OssEroute.Emsg("AioWait", "received unexpected signal", buff);
           continue;
          }

#ifdef __macos__
       aiop = (XrdSfsAio *)myInfo.si_value.sigval_ptr;
#else
       aiop = (XrdSfsAio *)myInfo.si_value.sival_ptr;
#endif

       while ((rc = aio_error(&aiop->sfsAio)) == EINPROGRESS) {}
       retval = (ssize_t)aio_return(&aiop->sfsAio);

       DEBUG(sigType <<" completed for " <<aiop->TIdent <<"; rc=" <<rc 
             <<" result=" <<retval <<" aiocb=" <<std::hex <<aiop <<std::dec);

       if (retval < 0) aiop->Result = -rc;
          else         aiop->Result = retval;

       if (isRead) aiop->doneRead();
          else     aiop->doneWrite();
      } while(1);
#endif
   return (void *)0;
}
 
#if defined( _POSIX_ASYNCHRONOUS_IO) && !defined(HAVE_SIGWTI)
/******************************************************************************/
/*                           s i g w a i t i n f o                            */
/******************************************************************************/
  
// Some platforms do not have sigwaitinfo() (e.g., MacOS). We provide for
// emulation here. It's not as good as the kernel version and the 
// implementation is very specific to the task at hand.
//
int sigwaitinfo(const sigset_t *set, siginfo_t *info)
{
// Now enable the signal handler by unblocking the signal. It will move the
// siginfo into the waiting struct and we can return.
//
   sigsuspend(set);
   return info->si_signo;
}
 
/******************************************************************************/
/*                          X r d O s s A i o R S H                           */
/******************************************************************************/
  
// XrdOssAioRSH handles AIO read signals. This handler was setup at AIO
// initialization time but only when this platform does not have sigwaitinfo().
//
extern "C"
{
void XrdOssAioRSH(int signum, siginfo_t *info, void *ucontext)
{
   extern siginfo_t *XrdOssAioInfoR;

// If we received a signal, it must have been for an AIO read and the read
// signal thread enabled this signal. This means that a valid address exists
// in the global read info pointer that we can now fill out.
//
   XrdOssAioInfoR->si_signo = info->si_signo;
   XrdOssAioInfoR->si_errno = info->si_errno;
   XrdOssAioInfoR->si_code  = info->si_code;
#ifdef __macos__
   XrdOssAioInfoR->si_value.sigval_ptr = info->si_addr;
#else
   XrdOssAioInfoR->si_value.sival_ptr = info->si_value.sival_ptr;
#endif
}
}
 
/******************************************************************************/
/*                          X r d O s s A i o W S H                           */
/******************************************************************************/
  
// XrdOssAioRSH handles AIO read signals. This handler was setup at AIO
// initialization time but only when this platform does not have sigwaitinfo().
//
extern "C"
{
void XrdOssAioWSH(int signum, siginfo_t *info, void *ucontext)
{
   extern siginfo_t *XrdOssAioInfoW;

// If we received a signal, it must have been for an AIO read and the read
// signal thread enabled this signal. This means that a valid address exists
// in the global read info pointer that we can now fill out.
//
   XrdOssAioInfoW->si_signo = info->si_signo;
   XrdOssAioInfoW->si_errno = info->si_errno;
   XrdOssAioInfoW->si_code  = info->si_code;
#ifdef __macos__
   XrdOssAioInfoW->si_value.sigval_ptr = info->si_addr;
#else
   XrdOssAioInfoW->si_value.sival_ptr = info->si_value.sival_ptr;
#endif
}
}
#endif
