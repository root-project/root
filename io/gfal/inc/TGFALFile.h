// @(#)root/gfal:$Id$
// Author: Fons Rademakers   8/12/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGFALFile
#define ROOT_TGFALFile

#include "TFile.h"
#include "TSystem.h"


class TGFALFile : public TFile {

private:
   Bool_t        fStatCached;  ///<! (transient) is file status cached?
   struct stat64 fStatBuffer;  ///<! (transient) Cached file status buffer (for performance)

   TGFALFile() : fStatCached(kFALSE) { }

   // Interface to basic system I/O routines
   Int_t    SysOpen(const char *pathname, Int_t flags, UInt_t mode);
   Int_t    SysClose(Int_t fd);
   Int_t    SysRead(Int_t fd, void *buf, Int_t len);
   Int_t    SysWrite(Int_t fd, const void *buf, Int_t len);
   Long64_t SysSeek(Int_t fd, Long64_t offset, Int_t whence);
   Int_t    SysStat(Int_t fd, Long_t *id, Long64_t *size, Long_t *flags, Long_t *modtime);
   Int_t    SysSync(Int_t) { /* no fsync for GFAL */ return 0; }

public:
   TGFALFile(const char *url, Option_t *option="",
             const char *ftitle="", Int_t compress = ROOT::RCompressionSetting::EDefaults::kUseCompiledDefault);
   ~TGFALFile();

   Bool_t  ReadBuffer(char *buf, Int_t len);
   Bool_t  ReadBuffer(char *buf, Long64_t pos, Int_t len);
   Bool_t  WriteBuffer(const char *buf, Int_t len);

   ClassDef(TGFALFile,1)  //A ROOT file that reads/writes via a GFAL
};


class TGFALSystem : public TSystem {

private:
   void    *fDirp;   // directory handler

   void    *GetDirPtr() const { return fDirp; }

public:
   TGFALSystem();
   virtual ~TGFALSystem() { }

   Int_t       MakeDirectory(const char *name);
   void       *OpenDirectory(const char *name);
   void        FreeDirectory(void *dirp);
   const char *GetDirEntry(void *dirp);
   Int_t       GetPathInfo(const char *path, FileStat_t &buf);
   Bool_t      AccessPathName(const char *path, EAccessMode mode);

   ClassDef(TGFALSystem,0)  // Directory handler for GFAL
};

#endif
