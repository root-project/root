// @(#)root/hdfs:$Id$
// Author: Brian Bockelman 29/09/2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_THDFSFile
#define ROOT_THDFSFile

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THDFSFile                                                            //
//                                                                      //
// A THDFSFile is like a normal TFile except that it reads and writes   //
// its data via the HDFS protocols.  For more information, see          //
// http://hadoop.apache.org/hdfs/.                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TFile
#include "TFile.h"
#endif

#ifndef ROOT_TSystem
#include "TSystem.h"
#endif

class THDFSFile : public TFile {

private:
   void     *fHdfsFH;    // HDFS file handle
   void     *fFS;        // HDFS user handle
   Long64_t  fSize;      // file size
   Long64_t  fSysOffset; // Seek offset in file
   char     *fPath;      // HDFS path name

   Int_t    SysOpen(const char *pathname, Int_t flags, UInt_t mode);
   Int_t    SysClose(Int_t fd);
   Int_t    SysRead(Int_t fd, void *buf, Int_t len);
   Int_t    SysWrite(Int_t fd, const void *buf, Int_t len);
   Long64_t SysSeek(Int_t fd, Long64_t offset, Int_t whence);
   Int_t    SysStat(Int_t fd, Long_t *id, Long64_t *size, Long_t *flags, Long_t *modtime);
   Int_t    SysSync(Int_t fd);

public:
   THDFSFile(const char *path, Option_t *option="",
             const char *ftitle="", Int_t compress=1);
   virtual ~THDFSFile();

   void ResetErrno() const;

   ClassDef(THDFSFile, 0) //A ROOT file that reads/writes via HDFS
};



class THDFSSystem : public TSystem {

private:
   void  *fFH;           // HDFS filesystem handle.
   void  *fDirp;         // Pointer to the array of file information.
   TUrl  *fUrlp;         // Pointer to the array of directory content URLs.
   Int_t  fDirEntries;   // The number of entries in the fDirp array.
   Int_t  fDirCtr;       // The current position in the fDirp array.

public:
    THDFSSystem();
    virtual ~THDFSSystem() { }

    Int_t       MakeDirectory(const char *name);
    void       *OpenDirectory(const char *name);
    void        FreeDirectory(void *dirp);
    const char *GetDirEntry(void *dirp);
    Int_t       GetPathInfo(const char *path, FileStat_t &buf);
    Bool_t      AccessPathName(const char *path, EAccessMode mode);
    Int_t       Unlink(const char *path);

    ClassDef(THDFSSystem,0)   // Directory handler for HDFS (THDFSFile)
};

#endif
