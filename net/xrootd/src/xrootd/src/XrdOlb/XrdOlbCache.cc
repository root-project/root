/******************************************************************************/
/*                                                                            */
/*                        X r d O l b C a c h e . c c                         */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdOlbCacheCVSID = "$Id$";
  
#include <sys/types.h>

#include "XrdOlb/XrdOlbCache.hh"
#include "XrdOlb/XrdOlbRRQ.hh"

using namespace XrdOlb;

/******************************************************************************/
/*                      L o c a l   S t r u c t u r e s                       */
/******************************************************************************/
  
struct XrdOlbBNCArgs
       {SMask_t          smask;
        char            *ppfx;
        int              plen;
       };
  
struct XrdOlbEXTArgs
       {XrdOucHash<char> *hp;
        char             *ppfx;
        int               plen;
       };

/******************************************************************************/
/*                               G l o b a l s                                */
/******************************************************************************/
  
XrdOlbCache XrdOlb::Cache;

/******************************************************************************/
/*                    E x t e r n a l   F u n c t i o n s                     */
/******************************************************************************/
/******************************************************************************/
/*                       X r d O l b B o u n c e A l l                        */
/******************************************************************************/
  
int XrdOlbBounceAll(const char *key, XrdOlbCInfo *cinfo, void *maskp)
{
    SMask_t xmask, smask = *(SMask_t *)maskp;

// Clear the vector for this server and indicate it bounced
//
   xmask = ~smask;
   cinfo->hfvec &= xmask;
   cinfo->pfvec &= xmask;
   cinfo->sbvec |= smask;

// Return a zero to keep this hash table
//
   return 0;
}

/******************************************************************************/
/*                      X r d O l b B o u n c e S o m e                       */
/******************************************************************************/

int XrdOlbBounceSome(const char *key, XrdOlbCInfo *cip, void *xargp)
{
    struct XrdOlbBNCArgs *xargs = (struct XrdOlbBNCArgs *)xargp;

// Check for match
//
   if (!strncmp(key, xargs->ppfx, xargs->plen))
      return XrdOlbBounceAll(key, cip, (void *)&xargs->smask);

// Keep the entry otherwise
//
   return 0;
}

/******************************************************************************/
/*                        X r d O l b C l e a r V e c                         */
/******************************************************************************/
  
int XrdOlbClearVec(const char *key, XrdOlbCInfo *cinfo, void *sid)
{
    const SMask_t smask_1 = 1;  // Avoid compiler promotion errors
    SMask_t smask = smask_1<<*(int *)sid;

// Clear the vector for this server
//
   smask = ~smask;
   cinfo->hfvec &= smask;
   cinfo->pfvec &= smask;
   cinfo->sbvec &= smask;

// Return indicating whether we should delete this or not
//
   return (cinfo->hfvec ? 0 : -1);
}

/******************************************************************************/
/*                       X r d O l b E x t r a c t F N                        */
/******************************************************************************/
  
int XrdOlbExtractFN(const char *key, XrdOlbCInfo *cip, void *xargp)
{
    struct XrdOlbEXTArgs *xargs = (struct XrdOlbEXTArgs *)xargp;

// Check for match
//
   if (xargs->plen <= (int)strlen(key)
   &&  !strncmp(key, xargs->ppfx, xargs->plen)) xargs->hp->Add(key, 0);

// All done
//
   return 0;
}

/******************************************************************************/
/*                       X r d O l b S c r u b S c a n                        */
/******************************************************************************/
  
int XrdOlbScrubScan(const char *key, XrdOlbCInfo *cip, void *xargp)
{
   return 0;
}

/******************************************************************************/
/*                               A d d F i l e                                */
/******************************************************************************/
  
int XrdOlbCache::AddFile(const char    *path,
                         SMask_t        mask,
                         int            isrw,
                         int            dltime,
                         XrdOlbRRQInfo *Info)
{
   XrdOlbPInfo  pinfo;
   XrdOlbCInfo *cinfo;
   int isnew = 0;

// Find if this server can handle the file in r/w mode
//
   if (isrw < 0)
      if (!Paths.Find(path, pinfo)) isrw = 0;
         else isrw = (pinfo.rwvec & mask) != 0;

// Lock the hash table
//
   PTMutex.Lock();

// Add/Modify the entry
//
   if ((cinfo = PTable.Find(path)))
      {if (dltime > 0) 
          {cinfo->deadline = dltime + time(0);
           cinfo->hfvec = 0; cinfo->pfvec = 0; cinfo->sbvec = 0;
           if (Info) Add2Q(Info, cinfo, isrw);
          } else {
           isnew = (cinfo->hfvec == 0);
           cinfo->hfvec |=  mask; cinfo->sbvec &= ~mask;
           if (isrw) {cinfo->deadline = 0;
                      if (cinfo->roPend || cinfo->rwPend)
                         Dispatch(cinfo, cinfo->roPend, cinfo->rwPend);
                     }
              else   {if (!cinfo->rwPend) cinfo->deadline = 0;
                      if (cinfo->roPend) Dispatch(cinfo, cinfo->roPend, 0);
                     }
          }
      } else if (dltime)
                {cinfo = new XrdOlbCInfo();
                 cinfo->hfvec = mask; cinfo->pfvec=cinfo->sbvec = 0; isnew = 1;
                 if (dltime > 0) cinfo->deadline = dltime + time(0);
                 PTable.Add(path, cinfo, LifeTime);
                 if (Info) Add2Q(Info, cinfo, isrw);
                }

// All done
//
   PTMutex.UnLock();
   return isnew;
}
  
/******************************************************************************/
/*                              D e l C a c h e                               */
/******************************************************************************/

void XrdOlbCache::DelCache(const char *path)
{

// Lock the hash table
//
   PTMutex.Lock();

// Delete the cache line
//
   PTable.Del(path);

// All done
//
   PTMutex.UnLock();
}
  
/******************************************************************************/
/*                               D e l F i l e                                */
/******************************************************************************/
  
int XrdOlbCache::DelFile(const char    *path,
                         SMask_t        mask,
                         int            dltime)
{
   XrdOlbCInfo *cinfo;
   int gone4good;

// Lock the hash table
//
   PTMutex.Lock();

// Look up the entry and remove server
//
   if ((cinfo = PTable.Find(path)))
      {cinfo->hfvec &= ~mask;
       cinfo->pfvec &= ~mask;
       cinfo->sbvec &= ~mask;
       gone4good = (cinfo->hfvec == 0);
       if (dltime > 0) cinfo->deadline = dltime + time(0);
          else if (gone4good) PTable.Del(path);
      } else gone4good = 0;

// All done
//
   PTMutex.UnLock();
   return gone4good;
}
  
/******************************************************************************/
/*                               G e t F i l e                                */
/******************************************************************************/
  
int  XrdOlbCache::GetFile(const char    *path,
                          XrdOlbCInfo   &cinfo,
                          int            isrw,
                          XrdOlbRRQInfo *Info)
{
   XrdOlbCInfo *info;

// Lock the hash table
//
   PTMutex.Lock();

// Look up the entry and remove server
//
   if ((info = PTable.Find(path)))
      {cinfo.hfvec = info->hfvec;
       cinfo.pfvec = info->pfvec;
       cinfo.sbvec = info->sbvec;
       if (info->deadline && info->deadline <= time(0))
          info->deadline = 0;
          else if (Info && info->deadline && !info->sbvec) Add2Q(Info,info,isrw);
       cinfo.deadline = info->deadline;
      }

// All done
//
   PTMutex.UnLock();
   return (info != 0);
}

/******************************************************************************/
/*                                 A p p l y                                  */
/******************************************************************************/
  
void XrdOlbCache::Apply(int (*func)(const char *,XrdOlbCInfo *,void *), void *Arg)
{
     PTMutex.Lock();
     PTable.Apply(func, Arg);
     PTMutex.UnLock();
}
 
/******************************************************************************/
/*                                B o u n c e                                 */
/******************************************************************************/

void XrdOlbCache::Bounce(SMask_t smask, char *path)
{

// Remove server from cache entries and indicate that it bounced
//
   if (!path)
      {PTMutex.Lock();
       PTable.Apply(XrdOlbBounceAll, (void *)&smask);
       PTMutex.UnLock();
      } else {
       struct XrdOlbBNCArgs xargs = {smask, path, strlen(path)};
       PTMutex.Lock();
       PTable.Apply(XrdOlbBounceSome, (void *)&xargs);
       PTMutex.UnLock();
      }
}
  
/******************************************************************************/
/*                               E x t r a c t                                */
/******************************************************************************/

void XrdOlbCache::Extract(const char *pathpfx, XrdOucHash<char> *hashp)
{
   struct XrdOlbEXTArgs xargs = {hashp, (char *)pathpfx, strlen(pathpfx)};

// Search the cache for all matching elements and insert them into the new hash
//
   PTMutex.Lock();
   PTable.Apply(XrdOlbExtractFN, (void *)&xargs);
   PTMutex.UnLock();
}
  
/******************************************************************************/
/*                                 R e s e t                                  */
/******************************************************************************/
  
void XrdOlbCache::Reset(int servid)
{
     PTMutex.Lock();
     PTable.Apply(XrdOlbClearVec, (void *)&servid);
     PTMutex.UnLock();
}

/******************************************************************************/
/*                                 S c r u b                                  */
/******************************************************************************/
  
void XrdOlbCache::Scrub()
{
     PTMutex.Lock();
     PTable.Apply(XrdOlbScrubScan, (void *)0);
     PTMutex.UnLock();
}

/******************************************************************************/
/*                       P r i v a t e   M e t h o d s                        */
/******************************************************************************/
/******************************************************************************/
/*                                 A d d 2 Q                                  */
/******************************************************************************/
  
void XrdOlbCache::Add2Q(XrdOlbRRQInfo *Info, XrdOlbCInfo *cp, int isrw)
{
   short Slot = (isrw ? cp->rwPend : cp->roPend);

// Add the request to the appropriate pending queue
//
   Info->Key = cp;
   Info->isRW= isrw;
   if (!(Slot = RRQ.Add(Slot, Info))) Info->Key = 0;
      else if (isrw) cp->rwPend = Slot;
               else  cp->roPend = Slot;
}

/******************************************************************************/
/*                              D i s p a t c h                               */
/******************************************************************************/
  
void XrdOlbCache::Dispatch(XrdOlbCInfo *cinfo, short roQ, short rwQ)
{

// Dispach the waiting elements
//
   if (roQ) {RRQ.Ready(roQ, cinfo, cinfo->hfvec);
             cinfo->roPend = 0;
            }
   if (rwQ) {RRQ.Ready(rwQ, cinfo, cinfo->hfvec);
             cinfo->rwPend = 0;
            }
}

/******************************************************************************/
/*                X r d O l b C I n f o   D e s t r u c t o r                 */
/******************************************************************************/
  
XrdOlbCInfo::~XrdOlbCInfo()
{
   if (roPend) RRQ.Del(roPend, this);
   if (rwPend) RRQ.Del(rwPend, this);
}
