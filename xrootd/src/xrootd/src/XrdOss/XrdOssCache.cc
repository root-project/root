/******************************************************************************/
/*                                                                            */
/*                        X r d O s s C a c h e . c c                         */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdOssCacheCVSID = "$Id$";

#include <unistd.h>
#include <errno.h>
#include <strings.h>
#include <iostream.h>
#include <stdio.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "XrdOss/XrdOssApi.hh"
#include "XrdOss/XrdOssCache.hh"
#include "XrdOss/XrdOssTrace.hh"
  
/******************************************************************************/
/*                 S t o r a g e   S y s t e m   O b j e c t                  */
/******************************************************************************/
  
extern XrdOssSys *XrdOssSS;

/******************************************************************************/
/*           G l o b a l   E r r o r   R o u t i n g   O b j e c t            */
/******************************************************************************/

extern XrdSysError OssEroute;

extern XrdOucTrace OssTrace;
  
/******************************************************************************/
/*              X r d O s s C a c h e _ L o c k   M e t h o d s               */
/******************************************************************************/

// We would do these as inline methods but the C++ preprocessor gets messed up
// when we try to do recursive (but protected) includes.
//

XrdOssCache_Lock::XrdOssCache_Lock()  {XrdOssSS->CacheContext.Lock(); locked = 1;}

XrdOssCache_Lock::~XrdOssCache_Lock() {if (locked) XrdOssSS->CacheContext.UnLock();}
  
void   XrdOssCache_Lock::Lock() {if (!locked) {XrdOssSS->CacheContext.Lock();
                                locked = 1;}
                               }

void XrdOssCache_Lock::UnLock() {if ( locked) {XrdOssSS->CacheContext.UnLock();
                                locked = 0;}
                               }

/******************************************************************************/
/*            X r d O s s C a c h e _ F S D a t a   M e t h o d s             */
/******************************************************************************/
  
XrdOssCache_FSData::XrdOssCache_FSData(const char *fsp, 
                                       STATFS_t   &fsbuff,
                                       dev_t       fsID)
{

     path = strdup(fsp);
     size = (long long)fsbuff.f_blocks*fsbuff.FS_BLKSZ;
     frsz = (long long)fsbuff.f_bavail*fsbuff.FS_BLKSZ;
     fsid = fsID;
     updt = time(0);
     next = 0;
     stat = 0;
}
  
/******************************************************************************/
/*            X r d O s s C a c h e _ F S   C o n s t r u c t o r             */
/******************************************************************************/
  
XrdOssCache_FS::XrdOssCache_FS(int &retc,
                               const char *fsg, 
                               const char *fsp,
                               const int inplace)
{
   STATFS_t fsbuff;
   struct stat sfbuff;
   XrdOssCache_FSData *fdp;
   XrdOssCache_Group  *cgp;
   dev_t tmpid = 0;

// set the groupname and the path supplied
//
   group  = strdup(fsg);
   path   = 0;
   retc   = ENOMEM;

// Find the filesystem for this object
//
   if (FS_Stat(fsp, &fsbuff) || stat(fsp, &sfbuff)) {retc = errno; return;}

// If this is an in-place creation, then we must get the cache lock
//
   if (inplace) {XrdOssSS->CacheContext.Lock(); fdp = XrdOssSS->fsdata;}
      else fdp = XrdOssSS->xsdata;

// Find the matching filesystem data
//
   while(fdp) {if (fdp->fsid == sfbuff.st_dev) break; fdp = fdp->next;}

// If we didn't find the filesystem, then create one
//
   if (!fdp)
      if (!(fdp = new XrdOssCache_FSData(fsp, fsbuff, tmpid))) return;
         else if (inplace) {fdp->next = XrdOssSS->fsdata; XrdOssSS->fsdata = fdp;}
                 else      {fdp->next = XrdOssSS->xsdata; XrdOssSS->xsdata = fdp;}

// Complete the filesystem block
//
   path   = strdup(fsp);
   plen   = strlen(fsp);
   fsdata = fdp;
   retc   = 0;

// Link this filesystem into the filesystem chain
//
   if (inplace)
      {if (!XrdOssSS->fsfirst) {next = this;              XrdOssSS->fscurr = this;
                                XrdOssSS->fsfirst = this; XrdOssSS->fslast = this;
                               }
          else {next = XrdOssSS->fslast->next; XrdOssSS->fslast->next = this;
                       XrdOssSS->fslast = this;
               }
      } else {
       if (!XrdOssSS->xsfirst) {next = this;            ; XrdOssSS->xscurr = this;
                                XrdOssSS->xsfirst = this; XrdOssSS->xslast = this;
                               }
          else {next = XrdOssSS->xslast->next; XrdOssSS->xslast->next = this;
                       XrdOssSS->xslast = this;
               }
      }

// Check if this is the first group allocation
//
   cgp = (inplace ? XrdOssSS->fsgroups : XrdOssSS->xsgroups);
   while(cgp && strcmp(fsg, cgp->group)) cgp = cgp->next;
   if (!cgp && (cgp = new XrdOssCache_Group(fsg, this)))
      if (inplace) {cgp->next = XrdOssSS->fsgroups; XrdOssSS->fsgroups = cgp;}
         else      {cgp->next = XrdOssSS->xsgroups; XrdOssSS->xsgroups = cgp;}

// Release the cache lock if we obtained it
//
   if (inplace) XrdOssSS->CacheContext.UnLock();
}

/******************************************************************************/
/*                                A d j u s t                                 */
/******************************************************************************/
  
off_t  XrdOssSys::Adjust(dev_t devid, off_t size)
{
   EPNAME("Adjust")
   XrdOssCache_FSData *fsdp;

// Obtain cache lock
//
   CacheContext.Lock();

// Search for matching filesystem
//
   fsdp = fsdata;
   while(fsdp && fsdp->fsid != devid) fsdp = fsdp->next;

// Process the result
//
   if (fsdp) {DEBUG("size=" <<fsdp->frsz <<'+' <<size <<" path=" <<fsdp->path);
              size = (fsdp->frsz += size);
              fsdp->stat |= XrdOssFSData_ADJUSTED;
             } else {
              DEBUG("dev " <<devid <<" not found.");
             }

// All done
//
   CacheContext.UnLock();
   return size;
}

/******************************************************************************/
/*                            L i s t _ C a c h e                             */
/******************************************************************************/
  
void XrdOssSys::List_Cache(char *lname, int self, XrdSysError &Eroute)
{
     XrdOssCache_FS *fsp;
     char buff[4096];

     CacheContext.Lock();
     if ((fsp = fsfirst))do
        {if (self && !(self & fsp->fsdata->stat)) continue;
                snprintf(buff, sizeof(buff), "%s %s %s",
                         lname, fsp->group, fsp->path);
                Eroute.Say(buff); 
                fsp = fsp->next;
        } while(fsp != fsfirst);
     CacheContext.UnLock();
}
 
/******************************************************************************/
/*                             C a c h e S c a n                              */
/******************************************************************************/

void *XrdOssSys::CacheScan(void *carg)
{
   EPNAME("CacheScan")
   XrdOssCache_FSData *fsdp;
   STATFS_t fsbuff;
   const struct timespec naptime = {XrdOssSS->cscanint, 0};

// Loop scanning the cache
//
   while(1)
        {nanosleep(&naptime, 0);

        // Get the cache context lock
        //
           XrdOssSS->CacheContext.Lock();

        // Scan through all filesystems skip filesystem that have been
        // recently adjusted to avoid fs statstics latency problems.
        //
           fsdp = XrdOssSS->fsdata;
           while(fsdp)
                {if ((fsdp->stat & XrdOssFSData_REFRESH)
                 || !(fsdp->stat & XrdOssFSData_ADJUSTED))
                     {if (FS_Stat(fsdp->path, &fsbuff))
                         OssEroute.Emsg("XrdOssCacheScan", errno ,
                                    "state file system ",(char *)fsdp->path);
                         else {fsdp->frsz = fsbuff.f_bavail*fsbuff.FS_BLKSZ;
                               fsdp->stat &= ~(XrdOssFSData_REFRESH |
                                               XrdOssFSData_ADJUSTED);
                               DEBUG("New free=" <<fsdp->frsz <<" path=" <<fsdp->path);
                               }
                     } else fsdp->stat |= XrdOssFSData_REFRESH;
                 fsdp = fsdp->next;
                }

         // Unlock the cache and go wait for the next interval
         //
            XrdOssSS->CacheContext.UnLock();
         }

// Keep the compiler happy
//
   return (void *)0;
}

/******************************************************************************/
/*                        o o s s _ F i n d _ P r t y                         */
/******************************************************************************/
  
int XrdOssFind_Prty(XrdOssCache_Req *req, void *carg)
{
    int prty = *(int *)carg;
    return (req->prty >= prty);
}
  
/******************************************************************************/
/*                         o o s s _ F i n d _ R e q                          */
/******************************************************************************/

int XrdOssFind_Req(XrdOssCache_Req *req, void *carg)
{
    XrdOssCache_Req *xreq = (XrdOssCache_Req *)carg;
    return (req->hash == xreq->hash) && !strcmp(req->path, xreq->path);
}

/******************************************************************************/
/*                               R e C a c h e                                */
/******************************************************************************/
  
void XrdOssSys::ReCache()
{
     XrdOssCache_FS     *fsp, *fspnext;
     XrdOssCache_FSData *fdp, *fdpnext;
     XrdOssCache_Group  *cgp, *cgpnext;

// Get exclsuive control over the cache
//
   CacheContext.Lock();

// Delete all filesystem definitions
//
   fsp = fsfirst;
   if (fsp)
      do {fspnext = fsp->next; delete fsp; fsp = fspnext;} 
         while (fsp != fsfirst);
   fsfirst = xsfirst; fslast = xslast; fscurr = xscurr;
   xsfirst = 0;       xslast = 0;      xscurr = 0;

   fdp = fsdata;
   while(fdp)
         {fdpnext = fdp->next; delete fdp; fdp = fdpnext;}
   fsdata = xsdata; xsdata = 0;

   cgp = fsgroups;
   while(cgp)
         {cgpnext = cgp->next; delete cgp; cgp = cgpnext;}
   fsgroups = xsgroups; xsgroups = 0;

// Release the cache lock
//
   CacheContext.UnLock();
}
