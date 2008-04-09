#ifndef __OFS_HANDLE__
#define __OFS_HANDLE__
/******************************************************************************/
/*                                                                            */
/*                       X r d O f s H a n d l e . h h                        */
/*                                                                            */
/* (c) 2003 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

//         $Id$

/* These are private data structures. They are allocated dynamically to the
   appropriate size (yes, that means dbx has a tough time).
*/

#include <stdlib.h>

#include "XrdOss/XrdOss.hh"
#include "XrdOuc/XrdOucDLlist.hh"
#include "XrdOuc/XrdOucHash.hh"
#include "XrdSys/XrdSysPthread.hh"

/******************************************************************************/
/*                     X r d O f s H a n d l e _ A r g s                      */
/******************************************************************************/
  
class XrdOfsHandle_Args
      {public:
       unsigned long hval;
       const char   *name;
       XrdOfsHandle_Args(unsigned long a1, const char *a2)
                        {hval = a1; name = a2;}
      ~XrdOfsHandle_Args() {}
      };

/******************************************************************************/
/*                    X r d O f s H a n d l e   F l a g s                     */
/******************************************************************************/
  
#define OFS_TCLOSE   0x0001
#define OFS_EOF      0x0002
#define OFS_INPROG   0x0010
#define OFS_PENDIO   0x0020
#define OFS_CHANGED  0x4000
#define OFS_RETIRED  0x8000
  
/******************************************************************************/
/*             C l a s s   X r d O f s H a n d l e _ C o m m o n              */
/******************************************************************************/

class XrdOfsHandle;

// Define items that are in common between a handle achor and an actual handle.
//
class XrdOfsHandle_Common
{
public:

unsigned long       Hash() {return hash;}
const char         *Name() {return name;}
unsigned int        PHID() {return pathid;}

void                Lock() {mutex.Lock();}
int             CondLock() {return mutex.CondLock();}
void              UnLock() {mutex.UnLock();}
int             WaitLock();

void                 Zap() {hash = 0;}

protected:

XrdOucDLlist<XrdOfsHandle> fullList;
XrdOucDLlist<XrdOfsHandle> openList;

unsigned long            hash;         // Hash value for the name
const    char           *name;
unsigned int           pathid;         // ID to uniquely distinguish handles

private:

XrdSysMutex   mutex;
};
  
/******************************************************************************/
/*              C l a s s   X r d O f s H a n d l e A n c h o r               */
/******************************************************************************/

class XrdOfsHandleAnchor : public XrdOfsHandle_Common
{
public:

time_t         IdleDeadline;

void           Add2Open(XrdOfsHandle &);

XrdOfsHandle  *Apply2Full(int (*func)(XrdOfsHandle *, void *), void *arg)
                   {return Apply(fullList, func, arg);}

XrdOfsHandle  *Apply2Open(int (*func)(XrdOfsHandle *, void *), void *arg)
                       {return Apply(openList, func, arg);}

XrdOfsHandle  *Attach(const char *path);

void           Detach(const char *path);

XrdOfsHandle  *Find(unsigned int a1, const char *a2)
                  {extern int XrdOfsHandle_Match(XrdOfsHandle *, void *);
                   return Apply(fullList, XrdOfsHandle_Match, a1, a2);
                  }

void           Hide(unsigned long a1, const char *a2)
                  {extern int XrdOfsHandle_Zap(XrdOfsHandle *, void *);
                   Apply(fullList, XrdOfsHandle_Zap, a1, a2);
                  }

unsigned int   Insert(XrdOfsHandle &);

     XrdOfsHandleAnchor(const char *type="???", unsigned int pid=0)
                 {name = type; hash = 0; IdleDeadline = 0; pathid = pid;}
    ~XrdOfsHandleAnchor() {}

/******************************************************************************/

friend int XrdOfsHandle_Match(XrdOfsHandle *, void *);
friend int XrdOfsHandle_Zap(XrdOfsHandle *, void *);

/******************************************************************************/

private:

XrdOucHash<XrdOfsHandle> fhtab;

XrdOfsHandle *Apply(XrdOucDLlist<XrdOfsHandle> &List,
                   int (*func)(XrdOfsHandle *, void *),
                   unsigned long a1, const char *a2);

XrdOfsHandle *Apply(XrdOucDLlist<XrdOfsHandle> &List,
                   int (*func)(XrdOfsHandle *, void *), void *args);
};

/******************************************************************************/
/*                    C l a s s   X r d O f s H a n d l e                     */
/******************************************************************************/

class XrdOssDF;

class XrdOfsHandle : public XrdOfsHandle_Common
{
friend class XrdOfsHandleAnchor;

public:
int                 flags;       // General    flags
int                 oflag;       // Open       flags
int                 links;       // Number of users for this handle (see note)
int                 activ;       // Number of active ops
int                 ecode;       // INPROG error code, if any
time_t              optod;       // TOD of last operation
int                 cxrsz;       // Compression region size
char                cxid[4];     // Compression Algorithm

void                Activate() {flags &= ~(OFS_TCLOSE|OFS_INPROG);
                                anchor->Add2Open(*this);
                               }

XrdOfsHandleAnchor &Anchor() {return *anchor;}

void                 Deactivate(int GetLock=1);

void                 Retire(int GetLock=1);

XrdOssDF            &Select(void) {return *ssi;}   // To allow for mt interfaces

const char          *Qname() {return anchor->Name();}

void                 LockAnchor() {anchor->Lock();}
void               UnLockAnchor() {anchor->UnLock();}

          XrdOfsHandle(unsigned long        hval,
                       const char          *oname,
                       int                  opn_mode,
                       time_t               opn_time,
                       XrdOfsHandleAnchor  *origin,
                       XrdOssDF            *oossdf);

         ~XrdOfsHandle()
               {Retire();
                if (name) free((void *)name);
                if (ssi) delete ssi;
               }

private:
XrdOfsHandleAnchor *anchor;     // Pointer to anchor node
XrdOssDF           *ssi;        // Storage System Interface
};

// Note: The links field is protected under the anchor lock; *not* the
//       the handle lock. This means this field can only be modified and
//       reliably inspected while holding the anchor lock. This poses problems
//       when deleting this object since the destructor needs the anchor lock
//       and the caller may or may not have it. The anchorLocked variable is
//       used to indicate the state of the lock during object deletion.
#endif
