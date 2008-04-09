// $Id$
#ifndef __SUT_PFILE_H
#define __SUT_PFILE_H

/******************************************************************************/
/*                                                                            */
/*                      X r d S u t P F i l e . h h                           */
/*                                                                            */
/* (c) 2005 by the Board of Trustees of the Leland Stanford, Jr., University  */
/*       All Rights Reserved. See XrdInfo.cc for complete License Terms       */
/*   Produced by Andrew Hanushevsky for Stanford University under contract    */
/*              DE-AC03-76-SFO0515 with the Department of Energy              */
/******************************************************************************/

#ifndef __XPROTOCOL_H
#include <XProtocol/XProtocol.hh>
#endif
#ifndef __OOUC_HASH__
#include <XrdOuc/XrdOucHash.hh>
#endif
#ifndef __OUC_STRING_H__
#include <XrdOuc/XrdOucString.hh>
#endif
/******************************************************************************/
/*                                                                            */
/*  Interface class to file to store login-related information                */
/*                                                                            */
/******************************************************************************/

#define kFileIDSize  8
#define kDefFileID  "XrdIF"
#define kXrdIFVersion  1

#define kOfsFileID   0
#define kOfsVersion  8     // == kFileIDSize (if this changes remember to scale
#define kOfsCtime    12    //                 accordingly the other offsets ...)
#define kOfsItime    16
#define kOfsEntries  20
#define kOfsIndOfs   24
#define kOfsJnkSiz   28

#define kPFEcreate   0x1
#define kPFEopen     0x2

#define kMaxLockTries  3

enum EPFileErrors {
   kPFErrBadInputs,
   kPFErrFileAlreadyOpen,
   kPFErrNoFile,
   kPFErrFileRename,
   kPFErrStat,
   kPFErrFileOpen,
   kPFErrFileNotOpen,
   kPFErrLocking,
   kPFErrUnlocking,
   kPFErrFileLocked,
   kPFErrSeek,
   kPFErrRead,
   kPFErrOutOfMemory,
   kPFErrLenMismatch,
   kPFErrBadOp
};

class XrdSutPFEntry;

class XrdSutPFEntInd {
public:
   char  *name;
   kXR_int32  nxtofs;
   kXR_int32  entofs;
   kXR_int32  entsiz;
   XrdSutPFEntInd(const char *n = 0,
                  kXR_int32 no = 0, kXR_int32 eo = 0, kXR_int32 es = 0);
   XrdSutPFEntInd(const XrdSutPFEntInd &ei);
   virtual ~XrdSutPFEntInd() { if (name) delete[] name; } 

   kXR_int32 Length() const { return (strlen(name) + 4*sizeof(kXR_int32)); }
   void SetName(const char *n = 0);

   // Assignement operator
   XrdSutPFEntInd &operator=(const XrdSutPFEntInd ei);
};

class XrdSutPFHeader {
public:
   char   fileID[kFileIDSize];
   kXR_int32  version;
   kXR_int32  ctime;           // time of file change
   kXR_int32  itime;           // time of index change
   kXR_int32  entries;
   kXR_int32  indofs;
   kXR_int32  jnksiz;          // number of unreachable bytes
   XrdSutPFHeader(const char *id = "       ", kXR_int32 v = 0, kXR_int32 ct = 0,
                  kXR_int32 it = 0, kXR_int32 ent = 0, kXR_int32 ofs = 0);
   XrdSutPFHeader(const XrdSutPFHeader &fh);
   virtual ~XrdSutPFHeader() { }
   void Print() const;

   static kXR_int32 Length() { return (kFileIDSize + 6*sizeof(kXR_int32)); }
};


class XrdSutPFile {

   friend class XrdSutCache;          // for open/close operation;

private:
   char                  *name;
   bool                   valid;      // If the file is usable ...
   kXR_int32              fFd;
   XrdOucHash<kXR_int32> *fHashTable; // Reflects the file index structure
   kXR_int32              fHTutime;   // time at which fHashTable was updated
   kXR_int32              fError;     // last error
   XrdOucString           fErrStr;    // description of last error

   // Entry low level access
   kXR_int32 WriteHeader(XrdSutPFHeader hd);
   kXR_int32 ReadHeader(XrdSutPFHeader &hd);
   kXR_int32 WriteInd(kXR_int32 ofs, XrdSutPFEntInd ind);
   kXR_int32 ReadInd(kXR_int32 ofs, XrdSutPFEntInd &ind);
   kXR_int32 WriteEnt(kXR_int32 ofs, XrdSutPFEntry ent);
   kXR_int32 ReadEnt(kXR_int32 ofs, XrdSutPFEntry &ent);

   // Reset (set inactive)
   kXR_int32 Reset(kXR_int32 ofs, kXR_int32 size);

   // Hash table operations
   kXR_int32 UpdateHashTable(bool force = 0);

   // For errors
   kXR_int32 Err(kXR_int32 code, const char *loc,
                 const char *em1 = 0, const char *em2 = 0);

public:
   XrdSutPFile(const char *n, kXR_int32 openmode = kPFEcreate,
               kXR_int32 createmode = 0600, bool hashtab = 1);
   XrdSutPFile(const XrdSutPFile &f);
   virtual ~XrdSutPFile();

   // Initialization method
   bool Init(const char *n, kXR_int32 openmode = kPFEcreate,
             kXR_int32 createmode = 0600, bool hashtab = 1);

   // Open/Close operations
   kXR_int32 Open(kXR_int32 opt, bool *wasopen = 0,
                  const char *nam = 0, kXR_int32 createmode = 0600);
   kXR_int32 Close(kXR_int32 d = -1);

   // File name
   const char *Name() const { return (const char *)name; }
   // (Un)Successful attachement
   bool IsValid() const { return valid; }
   // Last error
   kXR_int32 LastError() const { return fError; }
   const char *LastErrStr() const { return (const char *)fErrStr.c_str(); }

   // Update Methods
   kXR_int32 RemoveEntry(const char *name);
   kXR_int32 RemoveEntry(kXR_int32 ofs);
   kXR_int32 RemoveEntries(const char *name, char opt);
   kXR_int32 Trim(const char *fbak = 0);
   kXR_int32 UpdateHeader(XrdSutPFHeader hd);
   kXR_int32 WriteEntry(XrdSutPFEntry ent);
   kXR_int32 UpdateCount(const char *nm, int *cnt = 0, int step = 1, bool reset = 0);
   kXR_int32 ResetCount(const char *nm) { return UpdateCount(nm,0,0,1); }
   kXR_int32 ReadCount(const char *nm, int &cnt) { return UpdateCount(nm,&cnt,0); }

   // Access methods   
   kXR_int32 RetrieveHeader(XrdSutPFHeader &hd);
   kXR_int32 ReadEntry(const char *name, XrdSutPFEntry &ent, int opt = 0);
   kXR_int32 ReadEntry(kXR_int32 ofs, XrdSutPFEntry &ent);
   kXR_int32 SearchEntries(const char *name, char opt,
                           kXR_int32 *ofs = 0, kXR_int32 nofs = 1);
   kXR_int32 SearchSpecialEntries(kXR_int32 *ofs = 0, kXR_int32 nofs = 1);

   // Browser
   kXR_int32 Browse(void *out = 0);
};

#endif
