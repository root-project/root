/******************************************************************************/
/*                                                                            */
/*                       X r d O f s H a n d l e . c c                        */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

const char *XrdOfsHandleCVSID = "$Id$";

#include "XrdOfs/XrdOfs.hh"
#include "XrdOfs/XrdOfsConfig.hh"
#include "XrdOfs/XrdOfsHandle.hh"
#include "XrdSys/XrdSysTimer.hh"

/******************************************************************************/
/*                    F i l e   S y s t e m   O b j e c t                     */
/******************************************************************************/
  
extern XrdOfs XrdOfsFS;

/******************************************************************************/
/*                          X r d O f s H a n d l e                           */
/******************************************************************************/
/******************************************************************************/
/*                           C o n s t r u c t o r                            */
/******************************************************************************/

XrdOfsHandle::XrdOfsHandle(unsigned long          hval,
                           const char            *oname,
                           int                    opn_mode,
                           time_t                 opn_time,
                           XrdOfsHandleAnchor    *origin,
                           XrdOssDF              *ossdf)
{
    anchor       = origin;                  // Indicate where we belong
    flags        = OFS_TCLOSE|OFS_INPROG;   // Handle is being opened
    oflag        = opn_mode;                // Open flags
    optod        = opn_time;                // Time handle was opened
    links        = 1;                       // Number links to handle
    activ        = 0;                       // Not held
    hash         = hval;                    // Name hash value
    name         = strdup(oname);           // File name
    ssi          = ossdf;                   // Storage System Interface
    ecode        = 0;                       // Preset error code
    cxrsz        = 0;                       // Compression
    cxid[0]      = '\0';                    // Compression

    Lock();                                 // Return handle locked!!!!
    fullList.setItem(this);                 // Initialize list item pointer
    openList.setItem(this);                 // Initialize list item pointer
    pathid = origin->Insert(*this);         // Insert item into the chain
}

/******************************************************************************/
/*                            D e a c t i v a t e                             */
/******************************************************************************/
  
void XrdOfsHandle::Deactivate(int GetLock)
{

// Get anchor lock if needed
//
   if (GetLock) anchor->Lock();

// Remove outselves from the open list and decrease open count
//
   openList.Remove();
   XrdOfsFS.FDOpen--;
   flags |=   OFS_TCLOSE;
   flags &= ~(OFS_PENDIO | OFS_INPROG);

// Release the lock if we acquired it and exit
//
   if (GetLock) anchor->UnLock();
}

/******************************************************************************/
/*                                R e t i r e                                 */
/******************************************************************************/

void XrdOfsHandle::Retire(int GetLock)
{
// We can only retirte once (though this is an idempotent operation)
//
   if (!(flags & OFS_RETIRED))
      {if (GetLock) anchor->Lock();
       openList.Remove(); fullList.Remove();
       anchor->Detach(name);
       flags |= OFS_RETIRED;
       if (GetLock) anchor->UnLock();
      }
}

/******************************************************************************/
/*                    X r d O f s H a n d l e A n c h o r                     */
/******************************************************************************/
/******************************************************************************/
/*                              A d d 2 O p e n                               */
/******************************************************************************/
  
void XrdOfsHandleAnchor::Add2Open(XrdOfsHandle &item)
{
// Lock the anchor, insert the element, and then unlock and exit
//
   Lock(); 
   openList.Insert(&item.openList);
   XrdOfsFS.FDOpen++;
   UnLock();
}

/******************************************************************************/
/*                                 A p p l y                                  */
/******************************************************************************/
  
XrdOfsHandle *XrdOfsHandleAnchor::Apply(XrdOucDLlist<XrdOfsHandle> &List,
                                       int (*func)(XrdOfsHandle *, void *),
                                       unsigned long a1, const char *a2)
{struct XrdOfsHandle_Args args(a1, a2);

// Lock the anchor, apply the function to all elements, unlock and exit
//
   Lock();
   XrdOfsHandle *p = List.Apply(func, (void *)&args);
   UnLock();
   return p;
}

XrdOfsHandle *XrdOfsHandleAnchor::Apply(XrdOucDLlist<XrdOfsHandle> &List,
                                       int (*func)(XrdOfsHandle *, void *),
                                       void *args)
{
// Lock the anchor, apply the function to all elements, unlock and exit
//
   Lock();
   XrdOfsHandle *p = List.Apply(func, args);
   UnLock();
   return p;
}

/******************************************************************************/
/*                                A t t a c h                                 */
/******************************************************************************/
  
XrdOfsHandle *XrdOfsHandleAnchor::Attach(const char *path)
{
   XrdOfsHandle *fh;

// If we are not sharing file descriptors, then indicate nothing found
//
   if (XrdOfsFS.Options & XrdOfsFDNOSHARE) return 0;

// Try to find an open file. If found, increase the link count to prevent the 
// handle from being deleted (link count can only be diddled w/ anchor locked).
//
   Lock();
   if ((fh = fhtab.Find(path))) fh->links++;
   UnLock();
   return fh;
}

/******************************************************************************/
/*                                D e t a c h                                 */
/******************************************************************************/
  
// This method may only be called with the anchor lock held!
//
void XrdOfsHandleAnchor::Detach(const char *path)
{

// Remove entry from the file handle table
//
   if (!(XrdOfsFS.Options & XrdOfsFDNOSHARE)) fhtab.Del(path);
}

/******************************************************************************/
/*                                I n s e r t                                 */
/******************************************************************************/
  
unsigned int XrdOfsHandleAnchor::Insert(XrdOfsHandle &item)
{
   unsigned int newpid;

// Lock the anchor, insert the element, and then unlock and return an
// incremented pathid to the caller.
//
   Lock();
   fullList.Insert(&item.fullList);

// If unlimited sharing is possible, enter item into hash table
//
   if (!(XrdOfsFS.Options & XrdOfsFDNOSHARE))
      fhtab.Add(item.Name(), &item, 0, Hash_keep);

// Generate a new pathid and unlock the anchor
//
   newpid = pathid +=4;
   UnLock();

// Return new pathid to the caller
//
   return newpid;
}

/******************************************************************************/
/*                              W a i t L o c k                               */
/******************************************************************************/
  
int XrdOfsHandle_Common::WaitLock(void)
{
    static XrdSysTimer timer;
    int ntry=0;

// Try to obtain a lock within the retry parameters
//
    do { if (ntry && XrdOfsFS.LockWait) timer.Wait(XrdOfsFS.LockWait);
         if (CondLock()) return 1;
       } while(ntry++ < XrdOfsFS.LockTries);

// Indicate we could not get a lock
//
   return 0;
}

  
/******************************************************************************/
/*        E x t e r n a l   F u n c t i o n s   f o r   H a n d l e s         */
/******************************************************************************/
  
// The following fucntions do not belong to any class. This is necessitated by
// the fact that C++ does not allow class functions to be passed to a typeless
// interface and we cannot cast away a functions classness. So, this is the
// only other alternative.

/******************************************************************************/
/*                    X r d O f s H a n d l e _ M a t c h                     */
/******************************************************************************/
  
int XrdOfsHandle_Match(XrdOfsHandle *oh, void *varg)
{
/* This method simply returns true when a match is found. This causes the
   invoking function to return the pointer to the matching node. If a match
   is found, we attempt to lock the handle using retry logic, if necessary,
   before returning because the caller must have the assurance that the found
   handle actually exists since will be returning a pointer to it. If we can't
   lock it, then we pretend we didn't find it. Bad, but better than deadlock.
*/
   XrdOfsHandle_Args *args = static_cast<XrdOfsHandle_Args *>(varg);

// Return if this node does not match (no lock needed for this test)
//
   if (args->hval != oh->Hash() || strcmp(args->name, oh->Name())) return 0;

// Now, obtain a lock on this node and return result
//
   return oh->WaitLock();
}

/******************************************************************************/
/*                      X r d O f s H a n d l e _ Z a p                       */
/******************************************************************************/
  
int XrdOfsHandle_Zap(XrdOfsHandle *oh, void *varg)
{
/* This method simply hides all matching items by zeroing out the hash field.
   This effectively makes the item unmatchable. We need not lock the target
   handle since this is an idempotent operation and the handle can't escape
   while we have the anchor locked.
*/
   XrdOfsHandle_Args *args = static_cast<XrdOfsHandle_Args *>(varg);

   if (args->hval == oh->Hash() && !strcmp(args->name,oh->Name())) oh->Zap();
   return 0;
}
