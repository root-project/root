// @(#)root/base:$Id$
// Author: Andreas Peters  15/05/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAlienSystem
#define ROOT_TAlienSystem

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienSystem                                                         //
//                                                                      //
// TSystem Implementation of the AliEn GRID plugin.                     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include <stdio.h>
#endif

#include "TSystem.h"

class TAlienSystem : public TSystem {

protected:
   char fWorkingDirectory[1024];

public:
   TAlienSystem(const char *name = "Generic", const char *title = "Generic System");
   virtual ~TAlienSystem();

   //---- Misc
   virtual Bool_t          Init();

   //---- Directories
   virtual int             MakeDirectory(const char *name);
   virtual void           *OpenDirectory(const char *name);
   virtual void            FreeDirectory(void *dirp);
   virtual const char     *GetDirEntry(void *dirp);
   virtual void           *GetDirPtr() const { return 0; }
   virtual Bool_t          ChangeDirectory(const char *path);
   virtual const char     *WorkingDirectory();
   virtual const char     *HomeDirectory(const char *userName = 0);
   virtual int             mkdir(const char *name, Bool_t recursive = kFALSE);
   Bool_t                  cd(const char *path) { return ChangeDirectory(path); }
   const char             *pwd() { return WorkingDirectory(); }

   //---- Paths & Files
   virtual int             CopyFile(const char *from, const char *to, Bool_t overwrite = kFALSE);
   virtual int             Rename(const char *from, const char *to);
   virtual int             Link(const char *from, const char *to);
   virtual int             Symlink(const char *from, const char *to);
   virtual int             Unlink(const char *name);
   int                     GetPathInfo(const char *path, Long_t *id, Long_t *size, Long_t *flags, Long_t *modtime);
   int                     GetPathInfo(const char *path, Long_t *id, Long64_t *size, Long_t *flags, Long_t *modtime);
   virtual int             GetPathInfo(const char *path, FileStat_t &buf);
   virtual int             AlienFilestat(const char *fpath, FileStat_t &buf);
   virtual int             GetFsInfo(const char *path, Long_t *id, Long_t *bsize, Long_t *blocks, Long_t *bfree);
   virtual int             Chmod(const char *file, UInt_t mode);
   virtual int             Umask(Int_t mask);
   virtual int             Utime(const char *file, Long_t modtime, Long_t actime);
   virtual const char     *FindFile(const char *search, TString& file, EAccessMode mode = kFileExists);
   virtual Bool_t          AccessPathName(const char *path, EAccessMode mode = kFileExists);

   //---- Users & Groups
   virtual Int_t           GetUid(const char *user = 0);
   virtual Int_t           GetGid(const char *group = 0);
   virtual Int_t           GetEffectiveUid();
   virtual Int_t           GetEffectiveGid();
   virtual UserGroup_t    *GetUserInfo(Int_t uid);
   virtual UserGroup_t    *GetUserInfo(const char *user = 0);
   virtual UserGroup_t    *GetGroupInfo(Int_t gid);
   virtual UserGroup_t    *GetGroupInfo(const char *group = 0);

   ClassDef(TAlienSystem,0)  // System interface to the Alien Catalogue
};

#endif
