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

#include <stdio.h>
#include <time.h>
#include <sys/errno.h>
#include <sys/types.h>

#include "XrdOfs/XrdOfsHandle.hh"
#include "XrdOfs/XrdOfsStats.hh"
#include "XrdOss/XrdOss.hh"
#include "XrdSys/XrdSysError.hh"
#include "XrdSys/XrdSysPlatform.hh"
#include "XrdSys/XrdSysTimer.hh"

/******************************************************************************/
/*                         L o c a l   C l a s s e s                          */
/******************************************************************************/
/******************************************************************************/
/*                          X r d O f s H a n O s s                           */
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
/*                          X r d O f s H a n X p r                           */
/******************************************************************************/
  
class XrdOfsHanXpr
{
friend class XrdOfsHandle;
public:

       void          add2Q(int doLK=1);

       void          Deref()
                        {xqCV.Lock(); Handle=0; Call=0; xTNew=0; xqCV.UnLock();}

static XrdOfsHanXpr *Get();

       void          Set(XrdOfsHanCB *cbP, time_t xtm)
                        {xqCV.Lock(); Call = cbP; xTNew = xtm; xqCV.UnLock();}

       XrdOfsHanXpr(XrdOfsHandle *hP, XrdOfsHanCB *cbP, time_t xtm)
                   : Next(0), Handle(hP), Call(cbP), xTime(xtm), xTNew(0) {}
      ~XrdOfsHanXpr() {}

private:
       XrdOfsHanXpr *Next;
       XrdOfsHandle *Handle;
       XrdOfsHanCB  *Call;
       time_t        xTime;
       time_t        xTNew;

static XrdSysCondVar xqCV;
static XrdOfsHanXpr *xprQ;
};

XrdSysCondVar  XrdOfsHanXpr::xqCV(0, "HanXpr cv");
XrdOfsHanXpr  *XrdOfsHanXpr::xprQ = 0;

/******************************************************************************/
/*                          X r d O f s H a n P s c                           */
/******************************************************************************/
  
class XrdOfsHanPsc
{
public:

union {
XrdOfsHanPsc  *Next;
char          *User;   // -> Owner for posc files (user.pid:fd@host)
      };
XrdOfsHanXpr  *xprP;   // -> Associate Xpr object if active
int            Unum;   // -> Offset in poscq
short          Ulen;   //    Length of user.pid
short          Uhst;   // -> Host portion
short          Mode;   //    Mode file is to have

static
XrdOfsHanPsc  *Alloc();

void           Recycle();

               XrdOfsHanPsc() : User(0), xprP(0), Unum(0), Ulen(0),
                                Uhst(0), Mode(0)  {}
              ~XrdOfsHanPsc() {}
private:

static XrdSysMutex    pscMutex;
static XrdOfsHanPsc  *Free;
};

XrdSysMutex    XrdOfsHanPsc::pscMutex;
XrdOfsHanPsc  *XrdOfsHanPsc::Free = 0;

/******************************************************************************/
/*                     E x t e r n a l   L i n k a g e s                      */
/******************************************************************************/
  
void *XrdOfsHanXpire(void *pp)
{
     XrdOfsHandle::StartXpr();
     return (void *)0;
}

extern XrdSysError OfsEroute;

extern XrdOfsStats OfsStats;

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
  
int XrdOfsHandle::Alloc(const char *thePath, int Opts, XrdOfsHandle **Handle)
{
   XrdOfsHandle *hP;
   XrdOfsHanTab *theTable = (Opts & opRW ? &rwTable : &roTable);
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
   if (!(retc = Alloc(theKey, Opts, Handle))) theTable->Add(*Handle);
   OfsStats.Add(OfsStats.Data.numHandles);

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
  
int XrdOfsHandle::Alloc(XrdOfsHanKey theKey, int Opts, XrdOfsHandle **Handle)
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
       hP->isRW         = (Opts & opPC);           // File mode
       hP->ssi          = ossDF;                   // No storage system yet
       hP->Posc         = 0;                       // No creator
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
/* public                        P o s c G e t                                */
/******************************************************************************/
  
// Warning: the handle must be locked!

int XrdOfsHandle::PoscGet(short &Mode, int Done)
{
   XrdOfsHanPsc *pP;
   int pnum;

   if (Posc)
      {pnum = Posc->Unum;
       Mode = Posc->Mode;
       if (Done)
          {pP = Posc; Posc = 0;
           if (pP->xprP) {myMutex.Lock(); Path.Links--; myMutex.UnLock();}
           pP->Recycle();
          }
       return pnum;
      }

   Mode = 0;
   return 0;
}
  
/******************************************************************************/
/* public                        P o s c S e t                                */
/******************************************************************************/
  
// Warning: the handle must be locked!

int XrdOfsHandle::PoscSet(const char *User, int Unum, short Umod)
{
   static const char *Who = "?:0.0@?", *Whc = Who+1, *Whh = Who+5;
   const char *Col, *At;
   int retval = 0;

// If we have no posc object then we may just be able to return
//
   if (!Posc)
      {if (Unum > 0) Posc = XrdOfsHanPsc::Alloc();
          else return 0;
      }

// Find the markers in the incomming user
//
   if (!(Col = index(User, ':')) || !(At = index(User, '@')))
      {User = Who; Col = Whc; At = Whh;}

// If we already have a user check if it matches
//
   if (Posc->User)
      {if (!Unum)
          {if (!strncmp(User, Posc->User, Posc->Ulen)
           &&  !strcmp(Posc->User + Posc->Uhst, At+1)) return 0;
           return -ETXTBSY;
          } else {
           char buff[1024];
           sprintf(buff, "%s to %s for", Posc->User, User);
           OfsEroute.Emsg("Posc", "Creator changed from", buff, Path.Val);
           if (Unum < 0) Unum = Posc->Unum;
              else if (Unum !=  Posc->Unum) retval = Posc->Unum;
          }
       free(Posc->User);
      }

// Assign creation values
//
   Posc->User = strdup(User);
   Posc->Ulen = Col - User + 1;
   Posc->Uhst = At  - User + 1;
   Posc->Unum = Unum;
   Posc->Mode = Umod;
   return retval;
}
  
/******************************************************************************/
/* public                        P o s c U s r                                */
/******************************************************************************/
  
// Warning: the handle must be locked!

const char *XrdOfsHandle::PoscUsr()
{
   if (Posc) return Posc->User;
   return "?@?";
}
  
/******************************************************************************/
/* public                         R e t i r e                                 */
/******************************************************************************/

// The handle must be locked upon entry! It is unlocked upon exit.

int XrdOfsHandle::Retire(long long *retsz, char *buff, int blen)
{
   int numLeft;

// Get the global lock as the links field can only be manipulated with it.
// Decrement the links count and if zero, remove it from the table and
// place it on the free list. Otherwise, it is still in use.
//
   myMutex.Lock();
   if (Path.Links == 1)
      {if (buff) strlcpy(buff, Path.Val, blen);
       numLeft = 0; OfsStats.Dec(OfsStats.Data.numHandles);
       if ( (isRW ? rwTable.Remove(this) : roTable.Remove(this)) )
         {Next = Free; Free = this;
          if (Posc) {Posc->Recycle(); Posc = 0;}
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

int XrdOfsHandle::Retire(XrdOfsHanCB *cbP, int hTime)
{
   static int allOK = StartXpr(1);
   XrdOfsHanXpr *xP;

// The handle can only be held by one reference and only if it's a POSC and
// defered handling was properly set up.
//
   myMutex.Lock();
   if (!Posc || !allOK)
      {OfsEroute.Emsg("Retire", "ignoring deferred retire of", Path.Val);
       if (Path.Links != 1 || !Posc || !cbP) myMutex.UnLock();
          else {myMutex.UnLock(); cbP->Retired(this);}
       return Retire();
      }
   myMutex.UnLock();

// If this object already has an xpr object (happens for bouncing connections)
// then reuse that object. Otherwise create a new one and put it on the queue.
//
   if (Posc->xprP) Posc->xprP->Set(cbP, hTime+time(0));
      else {xP = Posc->xprP = new XrdOfsHanXpr(this, cbP, hTime+time(0));
            xP->add2Q();
           }
   UnLock();
   return 0;
}

/******************************************************************************/
/* public                       S t a r t X p r                               */
/******************************************************************************/
  
int XrdOfsHandle::StartXpr(int Init)
{
   static int InitDone = 0;
   XrdOfsHanXpr *xP;
   XrdOfsHandle *hP;

// If this is the initial all and we have not been initialized do so
//
   if (Init)
      {pthread_t tid;
       int rc;
       if (InitDone) return InitDone == 1;
       if ((rc = XrdSysThread::Run(&tid, XrdOfsHanXpire, (void *)0,
                                   0, "Handle Timeout")))
          {OfsEroute.Emsg("StartXpr", rc, "create handle timeout thread");
           InitDone = -1; return 0;
          }
       InitDone = 1; return 1;
      }

// Simply loop waiting for expired handles to become available. The Get() will
// return an Xpr object with the associated handle locked. 
//
do{xP = XrdOfsHanXpr::Get(); hP = xP->Handle;

// Perform validity check on the handle to catch instances where the handle
// was closed while we were in the process of getting it. While this is safe
// it should never happen, so issue a message so we know to fix it.
//
   if (hP->Posc && xP == hP->Posc->xprP) hP->Posc->xprP = 0;
      else {OfsEroute.Emsg("StarXtpr", "Invalid xpr ref to", hP->Path.Val);
            hP->UnLock(); delete xP; continue;
           }

// As the handle is locked we can get the global handle lock to prevent
// additions and removals of handles as we need a stable reference count to
// effect the callout, if any. Do so only if the reference count is one (for us)
// and the handle is active. In all cases, drop the global lock.
//
   myMutex.Lock();
   if (hP->Path.Links != 1 || !xP->Call) myMutex.UnLock();
      else {myMutex.UnLock();
            xP->Call->Retired(hP);
           }

// We can now officially retire the handle and delete the xpr object
//
   hP->Retire();
   delete xP;
  } while(1);

// Keep the compiler happy
//
   return 0;
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
/*                    C l a s s   X r d O f s H a n P s c                     */
/******************************************************************************/
/******************************************************************************/
/*                                 A l l o c                                  */
/******************************************************************************/
  
XrdOfsHanPsc *XrdOfsHanPsc::Alloc()
{
   XrdOfsHanPsc *pP;

// Grab or allocate an object
//
   pscMutex.Lock();
   if ((pP = Free)) {Free = pP->Next; pP->Next = 0;}
      else pP = new XrdOfsHanPsc;
   pscMutex.UnLock();

   return pP;
}

/******************************************************************************/
/*                               R e c y c l e                                */
/******************************************************************************/
  
void XrdOfsHanPsc::Recycle()
{

// Release any storgae appendages and clear other field
//
   if (xprP) {xprP->Deref(); xprP = 0;}
   if (User) free(User);
   Unum = 0;
   Ulen = 0;
   Uhst = 0;
   Mode = 0;

// Place element on free chain. We keep them all as there are never too many
//
   pscMutex.Lock();
   Next = Free; Free = this;
   pscMutex.UnLock();
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

/******************************************************************************/
/*                    C l a s s   X r d O f s H a n x p r                     */
/******************************************************************************/
/******************************************************************************/
/*                                 a d d 2 Q                                  */
/******************************************************************************/

void XrdOfsHanXpr::add2Q(int doLK)
{
   XrdOfsHanXpr *xPP, *xP;

// Place this object on the defered queue
//
   if (doLK) xqCV.Lock();
   xPP = 0; xP = xprQ;

   while(xP && xP->xTime < xTime) {xPP = xP; xP = xP->Next;}

   Next = xP;
   if (xPP) {xPP->Next = this; if (doLK)  xqCV.UnLock();}
      else  {     xprQ = this; if (doLK) {xqCV.Signal(); xqCV.UnLock();}}
};

/******************************************************************************/
/* public                            G e t                                    */
/******************************************************************************/

XrdOfsHanXpr *XrdOfsHanXpr::Get()
{
   XrdOfsHanXpr *xP;
   XrdOfsHandle *hP;
   int waitTime = 2592000;

// Obtain the xqCV lock as we need it to inspect/modify the queue and elements
// This lock is automatically released when we wait on the associated condvar.
//
   xqCV.Lock();

// Caculate the next wait time based on the first element, if any, in the queue.
// If the wait time is positive then loop back to wait that amount of time. Note
// that we have the xqCV lock that is needed to touch an inq Xpr object.
//
do{do{if (!(xP = xprQ)) waitTime = 2592000;
         else waitTime = xP->xTime - time(0);
      if (waitTime > 0) break;
      xprQ = xP->Next;

// Get the associated file handle. If none, simply delete the Xpr object.
//
      if (!(hP = xP->Handle)) {delete xP; continue;}

// If a new wait time is indicated then reschedule this object
//
      if (xP->xTNew)
         {xP->xTime = xP->xTNew; xP->xTNew = 0;
          xP->add2Q(0);
          continue;
         }

// Since we are still holding the xqCV lock we must get a conditional lock on
// the handle. If we can't then reschedule this object for later.
//
      if (!(hP->WaitLock()))
         {OfsEroute.Emsg("Retire", "defering retire of", hP->Path.Val);
          xP->xTime = time(0)+30;
          xP->add2Q(0);
          continue;
         }

// Drop the xqCV lock prior to returning the Xpr object to the caller. The
// caller will delete the object as needed.
//
   xqCV.UnLock();
   return xP;

     } while(1);

// We have the xqCV lock so we can now wait for an event or a timeout
//
   xqCV.Wait(waitTime);
  } while(1);
}
