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

///////////////////////////////////////////////////////////////////////////
//                                                                       //
// TRFIOFile                                                             //
//                                                                       //
// A TRFIOFile is like a normal TFile except that it reads and writes    //
// its data via a rfiod server (for more on the rfiod daemon see         //
// http://wwwinfo.cern.ch/pdp/serv/shift.html). TRFIOFile file names     //
// are in standard URL format with protocol "rfio". The following are    //
// valid TRFIOFile URL's:                                                //
//                                                                       //
//    rfio:/afs/cern.ch/user/r/rdm/galice.root                           //
//         where galice.root is a symlink of the type /shift/.../...     //
//    rfio:na49db1:/data1/raw.root                                       //
//    rfio:/castor/cern.ch/user/r/rdm/test.root                          //
//                                                                       //
// If Castor 2.1 is used the file names can be given also in the         //
// following ways:                                                       //
//                                                                       //
//  rfio://host:port/?path=FILEPATH                                      //
//  rfio://host/?path=FILEPATH                                           //
//  rfio:///castor?path=FILEPATH                                         //
//  rfio://stager_host:stager_port/?path=/castor/cern.ch/user/r/         //
//    rdm/bla.root&svcClass=MYSVCLASS&castorVersion=MYCASTORVERSION      //
//  rfio://stager_host/?path=/castor/cern.ch/user/r/                     //
//    rdm/bla.root&svcClass=MYSVCLASS&castorVersion=MYCASTORVERSION      //
//  rfio:///castor?path=/castor/cern.ch/user/r/                          //
//    rdm/bla.root&svcClass=MYSVCLASS&castorVersion=MYCASTORVERSION      //
//                                                                       //
// path is mandatory as parameter but all the other ones are optional.   //
//                                                                       //
// For the ultimate description of supported urls see:                   //
//    https://twiki.cern.ch/twiki/bin/view/FIOgroup/RfioRootTurl         //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TFile
#include "TFile.h"
#endif
#ifndef ROOT_TSystem
#include "TSystem.h"
#endif


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
             const char *ftitle="", Int_t compress=1);
   ~TRFIOFile();

   Int_t   GetErrno() const;
   void    ResetErrno() const;

   ClassDef(TRFIOFile,1)  //A ROOT file that reads/writes via a rfiod server
};


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
