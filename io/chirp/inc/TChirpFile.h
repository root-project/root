// @(#)root/chirp:$Id$
// Author: Dan Bradley, Michael Albrecht, Douglas Thain

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TChirpFile
#define ROOT_TChirpFile

#ifndef ROOT_TFile
#include "TFile.h"
#endif

#ifndef ROOT_TSystem
#include "TSystem.h"
#endif

class TChirpFile:public TFile {
private:
   TChirpFile();
   struct chirp_file *chirp_file_ptr;

   Int_t SysOpen(const char *pathname, Int_t flags, UInt_t mode);
   Int_t SysClose(Int_t fd);
   Int_t SysRead(Int_t fd, void *buf, Int_t len);
   Int_t SysWrite(Int_t fd, const void *buf, Int_t len);
   Long64_t SysSeek(Int_t fd, Long64_t offset, Int_t whence);
   Int_t SysStat(Int_t fd, Long_t * id, Long64_t * size, Long_t * flags, Long_t * modtime);
   Int_t SysSync(Int_t fd);

public:
   TChirpFile(const char *path, Option_t * option = "", const char *ftitle = "", Int_t compress = 1);

   ~TChirpFile();

   Bool_t ReadBuffers(char *buf, Long64_t * pos, Int_t * len, Int_t nbuf);

   ClassDef(TChirpFile, 0)
};


class TChirpSystem:public TSystem {
public:
   TChirpSystem();
   virtual ~ TChirpSystem();

   Int_t MakeDirectory(const char *name);
   void *OpenDirectory(const char *name);
   void FreeDirectory(void *dirp);
   const char *GetDirEntry(void *dirp);
   Int_t GetPathInfo(const char *path, FileStat_t & buf);
   Bool_t AccessPathName(const char *path, EAccessMode mode);
   Int_t Unlink(const char *path);

   int Rename(const char *from, const char *to);
   int Link(const char *from, const char *to);
   int Symlink(const char *from, const char *to);
   int GetFsInfo(const char *path, Long_t * id, Long_t * bsize, Long_t * blocks, Long_t * bfree);
   int Chmod(const char *file, UInt_t mode);
   int Utime(const char *file, Long_t modtime, Long_t actime);

   ClassDef(TChirpSystem, 0)
};

#endif
