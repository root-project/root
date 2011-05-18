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


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGFALFile                                                            //
//                                                                      //
// A TGFALFile is like a normal TFile except that it reads and writes   //
// its data via the underlaying Grid access mechanism.                  //
// TGFALFile file names are either a logical file name, a guid, an      //
// SURL or a TURL, like:                                                //
//                                                                      //
//    gfal:/lfn/user/r/rdm/galice.root                                  //
//                                                                      //
// Grid storage interactions today require using several existing       //
// software components:                                                 //
//  - The replica catalog services to locate valid replicas of          //
//    files.                                                            //
//  - The SRM software to ensure:                                       //
//     - files  exist on disk (they are recalled from mass              //
//       storage if necessary) or                                       //
//     - space is allocated on disk for new files (they are possibly    //
//       migrated to mass storage later)                                //
//  - A file access mechanism to access files from the storage          //
//    system on the worker node.                                        //
//                                                                      //
// The GFAL library hides these interactions and presents a Posix       //
// interface for the I/O operations. The currently supported protocols  //
// are: file for local access, dcap, gsidcap and kdcap (dCache access   //
// protocol) and rfio (CASTOR access protocol).                         //
//                                                                      //
// File naming convention:                                              //
// A file name can be a Logical File Name (LFN), a Grid Unique          //
// IDentifier (GUID), a file replica (SURL) or a Transport file         //
// name (TURL):                                                         //
//                                                                      //
//     an LFN starts with lfn:                                          //
//        for example lfn:baud/testgfal15                               //
//                                                                      //
//     a GUID starts with guid:                                         //
//        for example guid:2cd59291-7ae7-4778-af6d-b1f423719441         //
//                                                                      //
//     an SURL starts with srm://                                       //
//         for example srm://wacdr002d.cern.ch:8443/castor/             //
//                    cern.ch/user/b/baud/testgfal15                    //
//                                                                      //
//      a TURL starts with a protocol name:                             //
//          for example rfio:////castor/cern.ch/user/b/baud/testgfal15  //
//                                                                      //
// Note that for the TGFALFile plugin to work, all these pathnames      //
// should be prepended by gfal:.                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TFile
#include "TFile.h"
#endif
#ifndef ROOT_TSystem
#include "TSystem.h"
#endif


class TGFALFile : public TFile {

private:
   Bool_t        fStatCached;  //! (transient) is file status cached?
   struct stat64 fStatBuffer;  //! (transient) Cached file status buffer (for performance)

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
             const char *ftitle="", Int_t compress=1);
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
