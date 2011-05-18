// @(#)root/chirp:$Id$
// Author: Dan Bradley   17/12/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TChirpFile
#define ROOT_TChirpFile


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TChirpFile                                                           //
//                                                                      //
// A TChirpFile is like a normal TFile except that it reads and writes  //
// its data via a Chirp server.  For more information, see              //
// http://www.cs.wisc.edu/condor/chirp.                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TFile
#include "TFile.h"
#endif

class TChirpFile : public TFile {

private:
   struct chirp_client *chirp_client;

   TChirpFile() : chirp_client(0) { }

   // Interface to basic system I/O routines
   Int_t    SysOpen(const char *pathname, Int_t flags, UInt_t mode);
   Int_t    SysClose(Int_t fd);
   Int_t    SysRead(Int_t fd, void *buf, Int_t len);
   Int_t    SysWrite(Int_t fd, const void *buf, Int_t len);
   Long64_t SysSeek(Int_t fd, Long64_t offset, Int_t whence);
   Int_t    SysStat(Int_t fd, Long_t *id, Long64_t *size, Long_t *flags, Long_t *modtime);
   Int_t    SysSync(Int_t fd);

   Int_t OpenChirpClient(const char *URL,char const **path);
   Int_t CloseChirpClient();

public:
   TChirpFile(const char *path, Option_t *option="",
              const char *ftitle="", Int_t compress=1);

   ~TChirpFile();

   Bool_t  ReadBuffer(char *buf, Int_t len);
   Bool_t  ReadBuffer(char *buf, Long64_t pos, Int_t len);
   Bool_t  WriteBuffer(const char *buf, Int_t len);

   void    ResetErrno() const;

   ClassDef(TChirpFile,0)  //A ROOT file that reads/writes via a Chirp server
};

#endif
