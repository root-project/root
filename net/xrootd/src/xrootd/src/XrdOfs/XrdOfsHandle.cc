/******************************************************************************/
/*                                                                            */
/*                       X r d O f s H a n d l e . c c                        */
/*                                                                            */
/* (c) 2008 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdOfsHandleCVSID = "$Id$";

#include "XrdOfs/XrdOfsHandle.hh"
#include "XrdOss/XrdOss.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdSys/XrdSysTimer.hh"

/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/

class XrdOfsHanOss : public XrdOssDF
{
public:
                // Directory oriented methods
        int     Opendir(const char *)                        {return -EBADF;}
        int     Readdir(char *buff, int blen)                {return -EBADF;}

                // File oriented methods
        int     Fstat(struct stat *)                         {return -EBADF;}
        int     Fsync()                                      {return -EBADF;}
        int     Fsync(XrdSfsAio *aiop)                       {return -EBADF;}
        int     Ftruncate(unsigned long long)                {return -EBADF;}
        off_t   getMmap(void **addr)                         {return 0;}
        int     isCompressed(char *cxidp=0)                  {return 0;}
        int     Open(const char *, int, mode_t, XrdOucEnv &) {return -EBADF;}
        ssize_t Read(off_t, size_t)                 {return (ssize_t)-EBADF;}
        ssize_t Read(void *, off_t, size_t)         {return (ssize_t)-EBADF;}
        int     Read(XrdSfsAio *aoip)               {return (ssize_t)-EBADF;}
        ssize_t ReadRaw(    void *, off_t, size_t)  {return (ssize_t)-EBADF;}
        ssize_t Write(const void *, off_t, size_t)  {return (ssize_t)-EBADF;}
        int     Write(XrdSfsAio *aiop)              {return (ssize_t)-EBADF;}

                // Methods common to both
        int     Close(long long *retsz=0)                    {return -EBADF;}
inline  int     Handle() {return -1;}

                XrdOfsHanOss() {}
               ~XrdOfsHanOss() {}

};

/******************************************************************************/
/*                        S t a t i c   O b j e c t s                         */
/******************************************************************************/
  
XrdSysMutex   XrdOfsHandle::myMutex;
XrdOfsHanTab  XrdOfsHandle::roTable;
XrdOfsHanTab  XrdOfsHandle::rwTable;
XrdOssDF     *XrdOfsHandle::ossDF = (XrdOssDF *)new XrdOfsHanOss;
XrdOfsHandle *XrdOfsHandle::Free = 0;

/******************************************************************************/
/*                    c l a s s   X r d O f s H a n d l e                     */
/******************************************************************************/
/******************************************************************************/
/* static public                A l l o c   # 1                               */
/******************************************************************************/
  
int XrdOfsHandle::Alloc(const char *thePath, int isrw, XrdOfsHandle **Handle)
{
   XrdOfsHandle *hP;
   XrdOfsHanTab *theTable = (isrw ? &rwTable : &roTable);
   XrdOfsHanKey theKey(thePath, (int)strlen(thePath));
   int          retc;

// Lock the search table and try to find the key. If found, increment the
// the link count (can only be done with the global lock) then release the
// lock and try to lock the handle. It can't escape between lock calls because
// the link count is positive. If we can't lock the handle then it must be the
// that a long running operation is occuring. Return the handle to its former
// state and return a delay. Otherwise, return the handle.
//
   myMutex.Lock();
   if ((hP = theTable->Find(theKey)) && hP->Path.Links != 0xffff)
      {hP->Path.Links++; myMutex.UnLock();
       if (hP->WaitLock()) {*Handle = hP; return 0;}
       myMutex.Lock(); hP->Path.Links--; myMutex.UnLock();
       return nolokDelay;
      }

// Get a new handle
//
   if (!(retc = Alloc(theKey, isrw, Handle))) theTable->Add(*Handle);

// All done
//
   myMutex.UnLock();
   return retc;
}

/******************************************************************************/
/* static public                A l l o c   # 2                               */
/******************************************************************************/

int XrdOfsHandle::Alloc(XrdOfsHandle **Handle)
{
    XrdOfsHanKey myKey("dummy", 5);
    int retc;

    myMutex.Lock();
    if (!(retc = Alloc(myKey, 0, Handle))) 
       {(*Handle)->Path.Links = 0; (*Handle)->UnLock();}
    myMutex.UnLock();
    return retc;
}

/******************************************************************************/
/* private                      A l l o c   # 3                               */
/******************************************************************************/
  
int XrdOfsHandle::Alloc(XrdOfsHanKey theKey, int isrw, XrdOfsHandle **Handle)
{
   static const int minAlloc = 4096/sizeof(XrdOfsHandle);
   XrdOfsHandle *hP;

// No handle currently in the table. Get a new one off the free list
//
   if (!Free && (hP = new XrdOfsHandle[minAlloc]))
      {int i = minAlloc; while(i--) {hP->Next = Free; Free = hP; hP++;}}
   if ((hP = Free)) Free = hP->Next;

// Initialize the new handle, if we have one, and add it to the table
//
   if (hP)
      {hP->Path         = theKey;
       hP->Path.Links   = 1;
       hP->isChanged    = 0;                       // File changed
       hP->isCompressed = 0;                       // Compression
       hP->isPending    = 0;                       // Pending output
       hP->isRW         = isrw;                    // File mode
       hP->ssi          = ossDF;                   // No storage system yet
       hP->Lock();                                 // Wait is not possible
       *Handle = hP;
       return 0;
      }
   return nomemDelay;                              // Delay client
}

/******************************************************************************/
/* static public                    H i d e                                   */
/******************************************************************************/

void XrdOfsHandle::Hide(const char *thePath)
{
   XrdOfsHandle *hP;
   XrdOfsHanKey theKey(thePath, (int)strlen(thePath));

// Lock the search table and try to find the key in each table. If found,
// clear the length field to effectively hide the item.
//
   myMutex.Lock();
   if ((hP = roTable.Find(theKey))) hP->Path.Len = 0;
   if ((hP = rwTable.Find(theKey))) hP->Path.Len = 0;
   myMutex.UnLock();
}

/******************************************************************************/
/*                              I n a c t i v e                               */
/******************************************************************************/

int XrdOfsHandle::Inactive()
{
    return (ssi == ossDF);
}
  
/******************************************************************************/
/* public                         R e t i r e                                 */
/******************************************************************************/

// The handle must be locked upon entry! It is unlocked upon exit.

int XrdOfsHandle::Retire(long long *retsz, char *buff, int blen)
{
   extern XrdSysError OfsEroute;
   int numLeft;

// Get the global lock as the links field can only be manipulated with it.
// Decrement the links count and if zero, remove it from the table and
// place it on the free list. Otherwise, it is still in use.
//
   myMutex.Lock();
   if (Path.Links == 1)
      {if (buff) strlcpy(buff, Path.Val, blen);
       numLeft = 0;
       if ( (isRW ? rwTable.Remove(this) : roTable.Remove(this)) )
         {Next = Free; Free = this;
          if (Path.Val) {free((void *)Path.Val); Path.Val = (char *)"";}
          Path.Len = 0;
          if (ssi && ssi != ossDF)
             {ssi->Close(retsz); delete ssi; ssi = ossDF;}
         } else OfsEroute.Emsg("Retire", "Lost handle to", Path.Val);
      } else numLeft = --Path.Links;
   UnLock();
   myMutex.UnLock();
   return numLeft;
}

/******************************************************************************/
/* public                       W a i t L o c k                               */
/******************************************************************************/
  
int XrdOfsHandle::WaitLock(void)
{
   int ntry = LockTries;

// Try to obtain a lock within the retry parameters
//
   while(ntry--)
        {if (hMutex.CondLock()) return 1;
         if (ntry) XrdSysTimer::Wait(LockWait);
        }

// Indicate we could not get a lock
//
   return 0;
}

/******************************************************************************/
/*                    C l a s s   X r d O f s H a n T a b                     */
/******************************************************************************/
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/
  
XrdOfsHanTab::XrdOfsHanTab(int psize, int csize)
{
     prevtablesize = psize;
     nashtablesize = csize;
     Threshold     = (csize * LoadMax) / 100;
     nashnum       = 0;
     nashtable     = (XrdOfsHandle **)
                     malloc( (size_t)(csize*sizeof(XrdOfsHandle *)) );
     memset((void *)nashtable, 0, (size_t)(csize*sizeof(XrdOfsHandle *)));
}

/******************************************************************************/
/* public                            A d d                                    */
/******************************************************************************/
  
void XrdOfsHanTab::Add(XrdOfsHandle *hip)
{
   unsigned int kent;

// Check if we should expand the table
//
   if (++nashnum > Threshold) Expand();

// Add the entry to the table
//
   kent = hip->Path.Hash % nashtablesize;
   hip->Next = nashtable[kent];
   nashtable[kent] = hip;
}
  
/******************************************************************************/
/* private                        E x p a n d                                 */
/******************************************************************************/
  
void XrdOfsHanTab::Expand()
{
   int newsize, newent, i;
   size_t memlen;
   XrdOfsHandle **newtab, *nip, *nextnip;

// Compute new size for table using a fibonacci series
//
   newsize = prevtablesize + nashtablesize;

// Allocate the new table
//
   memlen = (size_t)(newsize*sizeof(XrdOfsHandle *));
   if (!(newtab = (XrdOfsHandle **) malloc(memlen))) return;
   memset((void *)newtab, 0, memlen);

// Redistribute all of the current items
//
   for (i = 0; i < nashtablesize; i++)
       {nip = nashtable[i];
        while(nip)
             {nextnip = nip->Next;
              newent  = nip->Path.Hash % newsize;
              nip->Next = newtab[newent];
              newtab[newent] = nip;
              nip = nextnip;
             }
       }

// Free the old table and plug in the new table
//
   free((void *)nashtable);
   nashtable     = newtab;
   prevtablesize = nashtablesize;
   nashtablesize = newsize;

// Compute new expansion threshold
//
   Threshold = static_cast<int>((static_cast<long long>(newsize)*LoadMax)/100);
}

/******************************************************************************/
/* public                           F i n d                                   */
/******************************************************************************/
  
XrdOfsHandle *XrdOfsHanTab::Find(XrdOfsHanKey &Key)
{
  XrdOfsHandle *nip;
  unsigned int kent;

// Compute position of the hash table entry
//
   kent = Key.Hash%nashtablesize;

// Find the entry
//
   nip = nashtable[kent];
   while(nip && nip->Path != Key) nip = nip->Next;
   return nip;
}

/******************************************************************************/
/* public                         R e m o v e                                 */
/******************************************************************************/
  
int XrdOfsHanTab::Remove(XrdOfsHandle *rip)
{
   XrdOfsHandle *nip, *pip = 0;
   unsigned int kent;

// Compute position of the hash table entry
//
   kent = rip->Path.Hash%nashtablesize;

// Find the entry
//
   nip = nashtable[kent];
   while(nip && nip != rip) {pip = nip; nip = nip->Next;}

// Remove if found
//
   if (nip)
      {if (pip) pip->Next = nip->Next;
          else nashtable[kent] = nip->Next;
       nashnum--;
      }
   return nip != 0;
}
