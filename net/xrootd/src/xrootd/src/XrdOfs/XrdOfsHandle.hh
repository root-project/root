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

#include "XrdOuc/XrdOucCRC.hh"
#include "XrdSys/XrdSysPthread.hh"

/******************************************************************************/
/*                    C l a s s   X r d O f s H a n K e y                     */
/******************************************************************************/
  
class XrdOfsHanKey
{
public:

const char          *Val;
unsigned int         Hash;
short                Len;
unsigned short       Links;

inline XrdOfsHanKey& operator=(const XrdOfsHanKey &rhs)
                                 {Val = strdup(rhs.Val); Hash = rhs.Hash;
                                  Len = rhs.Len;
                                  return *this;
                                 }

inline int           operator==(const XrdOfsHanKey &oth)
                                 {return Hash == oth.Hash && Len == oth.Len
                                      && !strcmp(Val, oth.Val);
                                 }

inline int           operator!=(const XrdOfsHanKey &oth)
                                 {return Hash != oth.Hash || Len != oth.Len
                                      || strcmp(Val, oth.Val);
                                 }

                    XrdOfsHanKey(const char *key=0, int kln=0)
                                : Val(key), Len(kln), Links(0)
                    {Hash = (key && kln ?
                          XrdOucCRC::CRC32((const unsigned char *)key,kln) : 0);
                    }

                   ~XrdOfsHanKey() {};
};

/******************************************************************************/
/*                    C l a s s   X r d O f s H a n T a b                     */
/******************************************************************************/

class XrdOfsHandle;
  
class XrdOfsHanTab
{
public:
void           Add(XrdOfsHandle *hP);

XrdOfsHandle  *Find(XrdOfsHanKey &Key);

int            Remove(XrdOfsHandle *rip);

// When allocateing a new nash, specify the required starting size. Make
// sure that the previous number is the correct Fibonocci antecedent. The
// series is simply n[j] = n[j-1] + n[j-2].
//
    XrdOfsHanTab(int psize = 987, int size = 1597);
   ~XrdOfsHanTab() {} // Never gets deleted

private:

static const int LoadMax = 80;

void             Expand();

XrdOfsHandle   **nashtable;
int              prevtablesize;
int              nashtablesize;
int              nashnum;
int              Threshold;
};

/******************************************************************************/
/*                    C l a s s   X r d O f s H a n d l e                     */
/******************************************************************************/
  
class XrdOssDF;
class XrdOfsHanCB;
class XrdOfsHanPsc;

class XrdOfsHandle
{
friend class XrdOfsHanTab;
friend class XrdOfsHanXpr;
public:

char                isPending;    // 1-> File  is pending sync()
char                isChanged;    // 1-> File was modified
char                isCompressed; // 1-> File  is compressed
char                isRW;         // T-> File  is open in r/w mode

void                Activate(XrdOssDF *ssP) {ssi = ssP;}

static const int    opRW = 1;
static const int    opPC = 3;

static       int    Alloc(const char *thePath,int Opts,XrdOfsHandle **Handle);
static       int    Alloc(                             XrdOfsHandle **Handle);

static       void   Hide(const char *thePath);

inline       int    Inactive() {return (ssi == ossDF);}

inline const char  *Name() {return Path.Val;}

             int    PoscGet(short &Mode, int Done=0);

             int    PoscSet(const char *User, int Unum, short Mode);

       const char  *PoscUsr();

             int    Retire(long long *retsz=0, char *buff=0, int blen=0);

             int    Retire(XrdOfsHanCB *, int DSec);

XrdOssDF           &Select(void) {return *ssi;}   // To allow for mt interfaces

static       int    StartXpr(int Init=0);         // Internal use only!

             int    Usage() {return Path.Links;}

inline       void   Lock()   {hMutex.Lock();}
inline       void   UnLock() {hMutex.UnLock();}

          XrdOfsHandle() : Path(0,0) {}

         ~XrdOfsHandle() {Retire();}

private:
static int           Alloc(XrdOfsHanKey, int Opts, XrdOfsHandle **Handle);
       int           WaitLock(void);

static const int     LockTries =   3; // Times to try for a lock
static const int     LockWait  = 333; // Mills to wait between tries
static const int     nolokDelay=   3; // Secs to delay client when lock failed
static const int     nomemDelay=  15; // Secs to delay client when ENOMEM

static XrdSysMutex   myMutex;
static XrdOfsHanTab  roTable;    // File handles open r/o
static XrdOfsHanTab  rwTable;    // File Handles open r/w
static XrdOssDF     *ossDF;      // Dummy storage sysem
static XrdOfsHandle *Free;       // List of free handles

       XrdSysMutex   hMutex;
       XrdOssDF     *ssi;        // Storage System Interface
       XrdOfsHandle *Next;
       XrdOfsHanKey  Path;       // Path for this handle
       XrdOfsHanPsc *Posc;       // -> Info for posc-type files
};
  
/******************************************************************************/
/*                     C l a s s   X r d O f s H a n C B                      */
/******************************************************************************/
  
class XrdOfsHanCB
{
public:

virtual void Retired(XrdOfsHandle *) = 0;

             XrdOfsHanCB() {}
virtual     ~XrdOfsHanCB() {}
};
#endif
