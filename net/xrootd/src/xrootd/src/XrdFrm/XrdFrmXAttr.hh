#ifndef __XRDFRMXATTR_HH__
#define __XRDFRMXATTR_HH__
/******************************************************************************/
/*                                                                            */
/*                        X r d F r m X A t t r . h h                         */
/*                                                                            */
/* (c) 2010 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*                            All Rights Reserved                             */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC02-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#include <sys/types.h>
#include <inttypes.h>
#include <netinet/in.h>
#include <sys/types.h>

#include "XrdSys/XrdSysPlatform.hh"

/* XrdFrmXAttr encapsulates the extended attributes needed to determine
   file residency. It is used by the FRM in migrate and purge processing as well
   as for the OSS to determine file residency in memory. It is self-contained
   to prevent circular dependencies.
*/

/******************************************************************************/
/*                        X r d F r m X A t t r C p y                         */
/******************************************************************************/
  
class XrdFrmXAttrCpy
{
public:

long long cpyTime;     // Last time file was copied
char      Rsvd[16];    // Reserved fields

/* postGet() will put cpyTime in host byte order (see preSet()).
*/
       int             postGet(int Result)
                              {if (Result > 0) cpyTime = ntohll(cpyTime);
                               return Result;
                              }

/* preSet() will put cpyTime in network byte order to allow the attribute to
            to be copied to different architectures and still work.
*/
       XrdFrmXAttrCpy *preSet(XrdFrmXAttrCpy &tmp)
                             {tmp.cpyTime = htonll(cpyTime); return &tmp;}

/* Name() returns the extended attribute name for this object.
*/
static const char     *Name() {return "XrdFrm.Cpy";}

/* sizeGet() and sizeSet() return the actual size of the object is used.
*/
static int             sizeGet() {return sizeof(XrdFrmXAttrCpy);}
static int             sizeSet() {return sizeof(XrdFrmXAttrCpy);}

       XrdFrmXAttrCpy() : cpyTime(0) {memset(Rsvd, 0, sizeof(Rsvd));}
      ~XrdFrmXAttrCpy() {}
};
  
/******************************************************************************/
/*                        X r d F r m X A t t r M e m                         */
/******************************************************************************/
  
class XrdFrmXAttrMem
{
public:

char      Flags;       // See definitions below
char      Rsvd[7];     // Reserved fields

// The following flags are defined for Flags
//
static const char memMap  = 0x01; // Mmap the file
static const char memKeep = 0x02; // Mmap the file and keep mapping
static const char memLock = 0x04; // Mmap the file and lock it in memory

/* postGet() and preSet() are minimal as no chages are needed
*/
static int             postGet(int Result)         {return Result;}
       XrdFrmXAttrMem *preSet(XrdFrmXAttrMem &tmp) {return this;}

/* Name() returns the extended attribute name for this object.
*/
static const char     *Name() {return "XrdFrm.Mem";}

/* sizeGet() and sizeSet() return the actual size of the object is used.
*/
static int             sizeGet() {return sizeof(XrdFrmXAttrMem);}
static int             sizeSet() {return sizeof(XrdFrmXAttrMem);}

       XrdFrmXAttrMem() : Flags(0) {memset(Rsvd, 0, sizeof(Rsvd));}
      ~XrdFrmXAttrMem() {}
};

/******************************************************************************/
/*                        X r d F r m X A t t r P i n                         */
/******************************************************************************/
  
class XrdFrmXAttrPin
{
public:

long long pinTime;     // Pin-to-time or pin-for-time value
char      Flags;       // See definitions below
char      Rsvd[7];     // Reserved fields

// The following flags are defined for Flags
//
static const char pinPerm = 0x01; // Pin forever
static const char pinIdle = 0x02; // Pin unless pinTime idle met
static const char pinKeep = 0x04; // Pin until  pinTime
static const char pinSet  = 0x07; // Pin is valid

/* postGet() will put pinTime in host byte order (see preSet()).
*/
       int             postGet(int Result)
                              {if (Result > 0) pinTime = ntohll(pinTime);
                               return Result;
                              }

/* preSet() will put pinTime in network byte order to allow the attribute to
            to be copied to different architectures and still work.
*/
       XrdFrmXAttrPin *preSet(XrdFrmXAttrPin &tmp)
                             {tmp.pinTime = htonll(pinTime); tmp.Flags = Flags;
                              return &tmp;
                             }

/* Name() returns the extended attribute name for this object.
*/
static const char     *Name() {return "XrdFrm.Pin";}


/* sizeGet() and sizeSet() return the actual size of the object is used.
*/
static int             sizeGet() {return sizeof(XrdFrmXAttrCpy);}
static int             sizeSet() {return sizeof(XrdFrmXAttrCpy);}

       XrdFrmXAttrPin() : pinTime(0), Flags(0) {memset(Rsvd, 0, sizeof(Rsvd));}
      ~XrdFrmXAttrPin() {}
};

/******************************************************************************/
/*                        X r d F r m X A t t r P f n                         */
/******************************************************************************/
  
class XrdFrmXAttrPfn
{
public:

char      Pfn[MAXPATHLEN+8]; // Enough room for the Pfn

/* postGet() and preSet() are minimal as no chages are needed
*/
static int             postGet(int Result)         {return Result;}
       XrdFrmXAttrPfn *preSet(XrdFrmXAttrPfn &tmp) {return this;}

/* Name() returns the extended attribute name for this object.
*/
static const char     *Name() {return "XrdFrm.Pfn";}

/* sizeGet() return the actual size of the object is used.
*/
static int             sizeGet() {return sizeof(XrdFrmXAttrPfn);}

/* sizeSet() returns the length of the Pfn string plus the null byte.
*/
       int             sizeSet() {return strlen(Pfn)+1;}

       XrdFrmXAttrPfn() {*Pfn = 0;}
      ~XrdFrmXAttrPfn() {}
};
#endif
