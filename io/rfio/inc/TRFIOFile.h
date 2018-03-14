// @(#)root/rfio:$Id$
// Author: Fons Rademakers  20/01/99 + Giulia Taurelli  29/06/2006

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRFIOFile
#define ROOT_TRFIOFile

#include "TFile.h"
#include "TSystem.h"


class TRFIOFile : public TFile {

private:
   TRFIOFile() { }

   // Interface to basic system I/O routines
   Int_t    SysOpen(const char *pathname, Int_t flags, UInt_t mode);
   Int_t    SysClose(Int_t fd);
   Int_t    SysRead(Int_t fd, void *buf, Int_t len);
   Int_t    SysWrite(Int_t fd, const void *buf, Int_t len);
   Long64_t SysSeek(Int_t fd, Long64_t offset, Int_t whence);
   Int_t    SysStat(Int_t fd, Long_t *id, Long64_t *size, Long_t *flags, Long_t *modtime);
   Int_t    SysSync(Int_t) { /* no fsync for RFIO */ return 0; }
   Bool_t   ReadBuffers(char *buf, Long64_t *pos, Int_t *len, Int_t nbuf);

public:
   TRFIOFile(const char *url, Option_t *option="",
             const char *ftitle="", Int_t compress=4);
   ~TRFIOFile();

   Int_t   GetErrno() const;
   void    ResetErrno() const;

   ClassDef(TRFIOFile,1)  //A ROOT file that reads/writes via a rfiod server
};

/**
\class TRFIOSystem
\ingroup IO

Directory handler for RFIO
*/

class TRFIOSystem : public TSystem {

private:
   void    *fDirp;   // directory handler

   void    *GetDirPtr() const { return fDirp; }

public:
   TRFIOSystem();
   virtual ~TRFIOSystem() { }

   Int_t       MakeDirectory(const char *name);
   void       *OpenDirectory(const char *name);
   void        FreeDirectory(void *dirp);
   const char *GetDirEntry(void *dirp);
   Int_t       GetPathInfo(const char *path, FileStat_t &buf);
   Bool_t      AccessPathName(const char *path, EAccessMode mode);
   Int_t       Unlink(const char *path);

   ClassDef(TRFIOSystem,0)  // Directory handler for RFIO
};

#endif
