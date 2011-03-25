/******************************************************************************/
/*                                                                            */
/*                          X r d O s s M i o . c c                           */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <unistd.h>
#include <stdio.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/stat.h>

#if defined(_POSIX_MAPPED_FILES)
#include <sys/mman.h>
#endif

#include "XrdSys/XrdSysPthread.hh"
#include "XrdOss/XrdOssMio.hh"
#include "XrdOss/XrdOssMioFile.hh"
#include "XrdOss/XrdOssTrace.hh"

/******************************************************************************/
/*                      S t a t i c   V a r i a b l e s                       */
/******************************************************************************/

XrdOucHash<XrdOssMioFile> XrdOssMio::MM_Hash;

XrdSysMutex    XrdOssMio::MM_Mutex;

XrdOssMioFile *XrdOssMio::MM_Perm     = 0;
XrdOssMioFile *XrdOssMio::MM_Idle     = 0;
XrdOssMioFile *XrdOssMio::MM_IdleLast = 0;

char           XrdOssMio::MM_on       = 1;
char           XrdOssMio::MM_chk      = 0;
char           XrdOssMio::MM_okmlock  = 1;
char           XrdOssMio::MM_preld    = 0;
long long      XrdOssMio::MM_pagsz    = (long long)sysconf(_SC_PAGESIZE);
#ifdef __macos__
long long      XrdOssMio::MM_pages    = 1024*1024*1024;
#else
long long      XrdOssMio::MM_pages    = (long long)sysconf(_SC_PHYS_PAGES);
#endif
long long      XrdOssMio::MM_max      = MM_pagsz*MM_pages/2;
long long      XrdOssMio::MM_inuse    = 0;

extern XrdSysError OssEroute;

extern XrdOucTrace OssTrace;
  
/******************************************************************************/
/*                               D i s p l a y                                */
/******************************************************************************/

void XrdOssMio::Display(XrdSysError &Eroute)
{
     char buff[1080];
     snprintf(buff, sizeof(buff), "       oss.memfile %s%s%s max %lld",
             (MM_on      ? ""            : "off "),
             (MM_preld   ? "preload"     : ""),
             (MM_chk     ? "check xattr" : ""), MM_max);
     Eroute.Say(buff);
}

/******************************************************************************/
/*                                   M a p                                    */
/******************************************************************************/
  
XrdOssMioFile *XrdOssMio::Map(char *path, int fd, int opts)
{
#if defined(_POSIX_MAPPED_FILES)
   EPNAME("MioMap");
   XrdSysMutexHelper mapMutex;
   struct stat statb;
   XrdOssMioFile *mp;
   void *thefile;
   char hashname[64];

// Get the size of the file
//
   if (fstat(fd, &statb))
      {OssEroute.Emsg("Mio", errno, "fstat file", path);
       return 0;
      }

// Develop hash name for this file
//
   XrdOucTrace::bin2hex((char *)&statb.st_dev,
                         int(sizeof(statb.st_dev)), hashname);
   XrdOucTrace::bin2hex((char *)&statb.st_ino, int(sizeof(statb.st_ino)),
                                         hashname+(sizeof(statb.st_dev)*2));

// Because of potntial race conditions, we must serialize execution
//
   mapMutex.Lock(&MM_Mutex);

// Check if we already have this mapping
//
   if ((mp = MM_Hash.Find(hashname)))
      {DEBUG("Reusing mmap; usecnt=" <<mp->inUse <<" path=" <<path);
       if (!(mp->Status & OSSMIO_MPRM) && !mp->inUse) Reclaim(mp);
       mp->inUse++;
       return mp;
      }

// Check if memory will be over committed
//
   if (MM_inuse + statb.st_size > MM_max)
      {if (!Reclaim(statb.st_size))
          {OssEroute.Emsg("Mio", "Unable to reclaim enough storage to mmap",path);
           return 0;
          }
      }
   MM_inuse += statb.st_size;

// Memory map the file
//
   if ((thefile = mmap(0,statb.st_size,PROT_READ,MAP_PRIVATE,fd,0))==MAP_FAILED)
      {OssEroute.Emsg("Mio", errno, "mmap file", path);
       return 0;
      } else {DEBUG("mmap " <<statb.st_size <<" bytes for " <<path);}

// Lock the file, if need be. Turn off locking if we don't have privs
//
   if (MM_okmlock && (opts & OSSMIO_MLOK))
      {if (mlock((char *)thefile, statb.st_size))
          {     if (errno == ENOSYS)
                   {OssEroute.Emsg("Mio","mlock() not supported; feature disabled.");
                    MM_okmlock = 0;
                   }
           else if (errno == EPERM)
                   {OssEroute.Emsg("Mio","Not privileged for mlock(); feature disabled.");
                    MM_okmlock = 0;
                   }
           else  OssEroute.Emsg("Mio", errno, "mlock file", path);
          } else {DEBUG("Locked " <<statb.st_size <<" bytes for " <<path);}
      }

// get a new file object
//
   if (!(mp = new XrdOssMioFile(hashname)))
      {OssEroute.Emsg("Mio", "Unable to allocate mmap file object for", path);
       munmap((char *)thefile, statb.st_size);
       return 0;
      }

// Complete the object here
//
   mp->Base   = thefile;
   mp->Size   = statb.st_size;
   mp->Dev    = statb.st_dev;
   mp->Ino    = statb.st_ino;
   mp->Status = opts;

// Add the mapping to our hash table
//
   if (MM_Hash.Add(hashname, mp))
      {OssEroute.Emsg("Mio", "Hash add failed for", path);
       munmap((char *)thefile, statb.st_size);
       delete mp;
       return 0;
      }

// If this is a permanent file, place it on the permanent queue
//
   if (opts & OSSMIO_MPRM)
      {mp->Next = MM_Perm; MM_Perm = mp;
       DEBUG("Placed file on permanent queue " <<path);
      }

// If this file is to be preloaded, start it now
//
   if (MM_preld && mp->inUse == 1)
      {pthread_t tid;
       int retc;
       mp->inUse++;
       if ((retc = XrdSysThread::Run(&tid, preLoad, (void *)mp)) < 0)
          {OssEroute.Emsg("Mio", retc, "creating mmap preload thread");
           mp->inUse--;
          }
          else DEBUG("started mmap preload thread; tid=" <<(unsigned long)tid);
      }

// All done
//
   return mp;
#else
   return 0;
#endif
}

/******************************************************************************/
/*                               p r e L o a d                                */
/******************************************************************************/
  
void *XrdOssMio::preLoad(void *arg)
{
   XrdOssMioFile *mp = (XrdOssMioFile *)arg;
   char *Base = (char *)(mp->Base);
   char *Bend = Base + mp->Size;
   char  Byte;

// Reference each page until we are done
//
   while(Base < Bend) {Byte = *Base; Base += MM_pagsz;}

// All done
//
   Recycle(mp);
   return (void *)0;
}

/******************************************************************************/
/*                               R e c l a i m                                */
/******************************************************************************/
  
// Reclaim() can only be called if the caller has the MM_Mutex lock!
//
int XrdOssMio::Reclaim(off_t amount)
{
   EPNAME("MioReclaim");
   XrdOssMioFile *mp;
   DEBUG("Trying to reclaim " <<amount <<" bytes.");

// Try to reclaim memory
//
   while((mp = MM_Idle) && amount > 0)
        {MM_Idle = mp->Next;
         MM_inuse -= mp->Size;
         amount   -= mp->Size;
         MM_Hash.Del(mp->HashName);  // This will delete the object
        }

// Indicate whether we cleared enough
//
   return amount <= 0;
}

/******************************************************************************/

int XrdOssMio::Reclaim(XrdOssMioFile *mp)
{
   EPNAME("MioReclaim");
   XrdOssMioFile *pmp = 0, *cmp = MM_Idle;

// Try to find the mapping
//
   while(cmp && mp != cmp) {pmp = cmp; cmp = cmp->Next;}

// Remove mapping from the idle list
//
   if (cmp)
      {if (pmp) pmp->Next = mp->Next;
          else  MM_Idle   = mp->Next;
       if (MM_IdleLast == cmp) MM_IdleLast = pmp;
      }
      else {DEBUG("Cannot find mapping for " <<mp->Dev <<':' <<mp->Ino);}

   return (cmp != 0);
}
 
/******************************************************************************/
/*                               R e c y c l e                                */
/******************************************************************************/
  
void XrdOssMio::Recycle(XrdOssMioFile *mp)
{
   XrdSysMutexHelper mmMutex(&MM_Mutex);

// Decrement the use count
//
   mp->inUse--;
   if (mp->inUse < 0)
      {OssEroute.Emsg("Mio", "MM usecount underflow for ", mp->HashName);
       mp->inUse = 0;
      } else if (mp->inUse > 0) return;

// If this is not a kept mapping, put it on the reclaim list
//
   if (!(mp->Status & OSSMIO_MPRM))
      {if (MM_IdleLast) MM_IdleLast->Next = mp;
          else MM_Idle = mp;
       MM_IdleLast = mp;
       mp->Next = 0;
      }
}
  
/******************************************************************************/
/*                                   S e t                                    */
/******************************************************************************/
  
void XrdOssMio::Set(int V_on, int V_preld,  int V_check)
{
   if (V_on      >= 0) MM_on      = (char)V_on;
   if (V_preld   >= 0) MM_preld   = (char)V_preld;
   if (V_check   >= 0) MM_chk     = (char)V_check;
}

void XrdOssMio::Set(long long V_max)
{
   if (V_max > 0) MM_max = V_max;
      else if (V_max < 0) MM_max = MM_pagsz*MM_pages*(-V_max)/100;
}
 
/******************************************************************************/
/*             X r d O s s d M i o F i l e   D e s t r u c t o r              */
/******************************************************************************/
  
XrdOssMioFile::~XrdOssMioFile()
{
#if defined(_POSIX_MAPPED_FILES)
    munmap((char *)Base, Size);
#endif
}
