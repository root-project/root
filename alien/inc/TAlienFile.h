// @(#)root/alien:$Name:  $:$Id: TAlienFile.h,v 1.1 2003/11/13 15:15:11 rdm Exp $
// Author: Andreas Peters 11/09/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAlienFile
#define ROOT_TAlienFile


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienFile                                                           //
//                                                                      //
// A TAlienFile is like a normal TFile except that it reads and writes  //
// its data via an Alien service.                                       //
// Filenames are standard URL format with protocol "alien".             //
// The following are valid TAlienFile URL's:                            //
//                                                                      //
//    alien:///alice/cern.ch/user/p/peters/test.root                    //
//    alien://alien.cern.ch/alice/cern.ch/user/p/peters/test.root       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TFile
#include "TFile.h"
#endif
#ifndef ROOT_TUrl
#include "TUrl.h"
#endif
#ifndef ROOT_TSystem
#include "TSystem.h"
#endif


class TAlienFile : public TFile {

private:
   TUrl      fUrl;                 //URL of file
   Long64_t  fOffset;              //seek offet
   TFile    *fSubFile;             //sub file (PFN)

   TAlienFile() : fUrl("dummy") { }

   // Interface to basic system I/O routines
   Int_t    SysOpen(const char *pathname, Int_t flags, UInt_t mode);
   Int_t    SysClose(Int_t fd);
   Int_t    SysRead(Int_t fd, void *buf, Int_t len);
   Int_t    SysWrite(Int_t fd, const void *buf, Int_t len);
   Long64_t SysSeek(Int_t fd, Long64_t offset, Int_t whence);
   Int_t    SysStat(Int_t fd, Long_t *id, Long_t *size, Long_t *flags,
                    Long_t *modtime);
   Int_t    SysSync(Int_t);

public:
   TAlienFile(const char *url, Option_t * option = "",
              const char *ftitle = "", Int_t compress = 1);
   virtual ~TAlienFile();

   Bool_t ReadBuffer(char *buf, Int_t len);
   Bool_t WriteBuffer(const char *buf, Int_t len);

   Int_t GetErrno() const;
   void ResetErrno() const;

   ClassDef(TAlienFile,1)  //A ROOT file that reads/writes via AliEn Services
};


class TAlienSystem : public TSystem {

private:
   void     *fDirp;       // directory handler

   void *GetDirPtr() const { return fDirp; }

public:
   TAlienSystem();
   virtual ~TAlienSystem() { }
   Int_t       MakeDirectory(const char *name);
   void       *OpenDirectory(const char *name);
   void        FreeDirectory(void *dirp);
   const char *GetDirEntry(void *dirp);
   Int_t       GetPathInfo(const char *path, Long_t *id, Long_t *size,
                           Long_t *flags, Long_t *modtime);
   Bool_t      AccessPathName(const char *path, EAccessMode mode);

   ClassDef(TAlienSystem,0)  // Directory handler for AliEn
};

#endif
