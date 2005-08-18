// @(#)root/alien:$Name:  $:$Id: TAlienFile.h,v 1.9 2005/08/18 14:16:28 rdm Exp $
// Author: Andreas Peters 11/09/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
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
   TFile    *fSubFile;             //sub file (PFN)
   TString   fAuthz;               //authorization envelope
   TString   fLfn;                 //logical filename

   TAlienFile() : fUrl("dummy") { }

public:
   TAlienFile(const char *url, Option_t *option = "",
              const char *ftitle = "", Int_t compress = 1);
   virtual ~TAlienFile();

   TString    AccessURL(const char *url, Option_t *option = "",
                        const char *ftitle = "", Int_t compress = 1);

   Bool_t     ReadBuffer(char *buf, Int_t len);
   Bool_t     WriteBuffer(const char *buf, Int_t len);

   void       Seek(Long64_t offset, ERelativeTo pos = kBeg);
   void       Close(Option_t *option="");

   Int_t      Write(const char *name=0, Int_t opt=0, Int_t bufsiz=0) const
                 { return (fSubFile) ? fSubFile->Write(name,opt,bufsiz) : -1; }
   Int_t      Write(const char *name=0, Int_t opt=0, Int_t bufsiz=0)
                 { return (fSubFile) ? fSubFile->Write(name,opt,bufsiz) : -1; }
   Long64_t   GetBytesRead() const
                 { return (fSubFile) ? fSubFile->GetBytesRead() : -1; }
   Long64_t   GetBytesWritten() const
                 { return (fSubFile) ? fSubFile->GetBytesWritten() : -1; }
   Long64_t   GetSize() const
                 { return (fSubFile) ? fSubFile->GetSize() : -1; }
   Bool_t     cd(const char *path)
                 { return (fSubFile) ? fSubFile->cd(path) : kFALSE; }
   const char *GetPath() const
                 { return (fSubFile) ? fSubFile->GetPath() : 0; }
   TObject    *Get(const char *namecycle)
                 { return (fSubFile) ? fSubFile->Get(namecycle) : 0; }
   TFile      *GetFile() const
                 { return (fSubFile) ? fSubFile->GetFile() : 0; }
   TKey       *GetKey(const char *name, Short_t cycle=9999) const
                 { return (fSubFile) ? fSubFile->GetKey(name, cycle) : 0; };
   TList      *GetList() const
                 { return (fSubFile) ? fSubFile->GetList() : 0; }
   TList      *GetListOfKeys() const
                 { return (fSubFile) ? fSubFile->GetListOfKeys() : 0; }

   TFile      *GetSubFile() const { return fSubFile; }

   ClassDef(TAlienFile,1)  //A ROOT file that reads/writes via AliEn services and sub protocols
};



#endif
