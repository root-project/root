/******************************************************************************/
/*                                                                            */
/*                          X r d W a i t 4 1 . c c                           */
/*                                                                            */
/* (c) 2009 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//           $Id$

const char *XrdWait41CVSID = "$Id$";

/* This unitily waits for the first of n file locks. The syntax is:

   wait41 <path> [<path> [. . .]]

*/

/******************************************************************************/
/*                         i n c l u d e   f i l e s                          */
/******************************************************************************/
  
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <sys/param.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "XrdOuc/XrdOucTList.hh"
#include "XrdSys/XrdSysHeaders.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdSys/XrdSysPthread.hh"

/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/

class XrdW41Gate
{
public:

static void   Serialize(XrdOucTList *gfP, int Wait=1);

static int    Wait41(XrdOucTList *fP);

       XrdW41Gate() {}
      ~XrdW41Gate() {}

private:
static XrdSysMutex      gateMutex;
static XrdSysSemaphore  gateSem;
static int              gateOpen;
};

XrdSysMutex     XrdW41Gate::gateMutex;
XrdSysSemaphore XrdW41Gate::gateSem(0);
int             XrdW41Gate::gateOpen = 0;

class XrdW41Dirs
{
public:

static XrdOucTList *Expand(const char *Path, XrdOucTList *ptl);
};
  
/******************************************************************************/
/*            E x t e r n a l   T h r e a d   I n t e r f a c e s             */
/******************************************************************************/

namespace XrdWait41
{
void *GateWait(void *parg)
{
   XrdOucTList *fP = (XrdOucTList *)parg;

// Serialize
//
   XrdW41Gate::Serialize(fP);
   return (void *)0;
}
}

using namespace XrdWait41;

/******************************************************************************/
/*                                  m a i n                                   */
/******************************************************************************/
  
int main(int argc, char *argv[])
{
   sigset_t myset;
   XrdOucTList *gateFiles = 0;
   struct stat Stat;
   const char *eText;
   char buff[8];
   int i;

// Turn off sigpipe and host a variety of others before we start any threads
//
   signal(SIGPIPE, SIG_IGN);  // Solaris optimization
   sigemptyset(&myset);
   sigaddset(&myset, SIGPIPE);
   sigaddset(&myset, SIGCHLD);
   pthread_sigmask(SIG_BLOCK, &myset, NULL);

// Set the default stack size here
//
   if (sizeof(long) > 4) XrdSysThread::setStackSize((size_t)1048576);
      else               XrdSysThread::setStackSize((size_t)786432);

// Construct a list of files. For each directory, expand that to a list
//
   for (i = 1; i < argc; i++)
       {if (stat(argv[i], &Stat))
           {eText = strerror(errno);
            cerr <<"wait41: " <<eText <<" processing " <<argv[i] <<endl;
            continue;
           }
             if (S_ISREG(Stat.st_mode))
                gateFiles =    new XrdOucTList(argv[i],0,gateFiles);
        else if (S_ISDIR(Stat.st_mode))
                gateFiles = XrdW41Dirs::Expand(argv[i],  gateFiles);
       }

// If we have no waiters then fail
//
   if (!gateFiles)
      {cerr <<"wait41: Nothing to wait on!" <<endl;
       cout <<"BAD\n" <<endl;
       _exit(1);
      }

// Now wait for the first lock
//
   eText = (XrdW41Gate::Wait41(gateFiles) ? "OK\n" : "BAD\n");
   cout <<eText <<endl;

// Now wait for the process to die
//
   if (read(STDIN_FILENO, buff, sizeof(buff))) {}
   exit(0);
}

/******************************************************************************/
/*       C l a s s   X r d W 4 1 D i r s   I m p l e m e n t a t i o n        */
/******************************************************************************/
/******************************************************************************/
/*                                E x p a n d                                 */
/******************************************************************************/
  
XrdOucTList *XrdW41Dirs::Expand(const char *Path, XrdOucTList *ptl)
{
    struct dirent *dp;
    struct stat Stat;
    const char *eText;
    char buff[1024], *sfxDir;
    DIR *DFD;

    if (!(DFD = opendir(Path)))
       {eText = strerror(errno);
        cerr <<"wait41: " <<eText <<" opening directory" <<Path <<endl;
        return ptl;
       }

    strcpy(buff, Path); sfxDir = buff + strlen(Path);
    if (*(sfxDir-1) != '/') *sfxDir++ = '/';

    errno = 0;
    while((dp = readdir(DFD)))
         {if (!strcmp(dp->d_name, ".") || !strcmp(dp->d_name, "..")) continue;
          strcpy(sfxDir, dp->d_name);
          if (stat(buff, &Stat))
             {eText = strerror(errno);
              cerr <<"wait41: " <<eText <<" processing " <<buff <<endl;
              continue;
             }
          if (S_ISREG(Stat.st_mode)) ptl = new XrdOucTList(buff, 0, ptl);
          errno = 0;
         }

    if (errno)
       {eText = strerror(errno);
        cerr <<"wait41: " <<eText <<" reading directory" <<Path <<endl;
       }

    closedir(DFD);
    return ptl;
}

/******************************************************************************/
/*       C l a s s   X r d W 4 1 G a t e   I m p l e m e n t a t i o n        */
/******************************************************************************/
/******************************************************************************/
/*                             S e r i a l i z e                              */
/******************************************************************************/
  
void XrdW41Gate::Serialize(XrdOucTList *gfP, int Wait)
{
   FLOCK_t lock_args;
   int Act, rc;

// Establish locking options
//
   bzero(&lock_args, sizeof(lock_args));
   lock_args.l_type = F_WRLCK;
   Act = (Wait ? F_SETLKW : F_SETLK);

// Now perform the action
//
   do {rc = fcntl(gfP->val, Act, &lock_args);} while(rc == -1 && errno == EINTR);

// Determine result
//
   if (rc != -1) rc = 0;
      else {rc = errno;
            cerr <<"Serialize: " <<strerror(rc) <<" locking FD " <<gfP->text <<endl;
           }

// Reflect what happened here
//
   gateMutex.Lock();
   if (rc || gateOpen) close(gfP->val);
      else gateOpen = 1;
   gateSem.Post();
   gateMutex.UnLock();
}

/******************************************************************************/
/*                                W a i t 4 1                                 */
/******************************************************************************/
  
int XrdW41Gate::Wait41(XrdOucTList *gfP)
{
   static const int AMode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
   pthread_t tid;
   const char *eTxt;
   int rc, Num = 0;
  
// Run through the chain of files setting up a wait. We try to do a fast
// redispatch in case we get a lock early.
//
   while(gfP)
        {if (Num)
            {gateMutex.Lock();
             if (gateOpen) {gateMutex.UnLock(); return 1;}
             gateMutex.UnLock();
            }
              if ((gfP->val = open(gfP->text, O_CREAT|O_RDWR, AMode)) < 0)
                 {eTxt = strerror(errno);
                  cerr <<"Wait41: " <<eTxt <<" opening " <<gfP->text <<endl;
                 }
         else if ((rc = XrdSysThread::Run(&tid, GateWait, (void *)gfP,
                                      XRDSYSTHREAD_BIND, "Gate Wait")))
                 {eTxt = strerror(errno);
                  cerr <<"Wait41: " <<eTxt <<" creating gate thread for "
                                    <<gfP->text <<endl;
                  close(gfP->val);
                 } else Num++;
          gfP = gfP->next;
         }

// At this point we will have to wait for the lock if we have any threads
//
   while(Num--)
        {gateSem.Wait();
         gateMutex.Lock();
         if (gateOpen) {gateMutex.UnLock(); return 1;}
         gateMutex.UnLock();
        }

// No such luck, every thread failed
//
   return 0;
}
