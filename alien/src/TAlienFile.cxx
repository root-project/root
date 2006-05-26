// @(#)root/alien:$Name:  $:$Id: TAlienFile.cxx,v 1.18 2006/05/19 07:30:04 brun Exp $
// Author: Andreas Peters 11/09/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienFile                                                           //
//                                                                      //
// A TAlienFile is like a normal TFile except that it reads and writes  //
// its data via an AliEn service.                                       //
// Filenames are standard URL format with protocol "alien".             //
// The following are valid TAlienFile URL's:                            //
//                                                                      //
//    alien:///alice/cern.ch/user/p/peters/test.root                    //
//    alien://alien.cern.ch/alice/cern.ch/user/p/peters/test.root       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TAlienFile.h"
#include "TAlienResult.h"
#include "TAlien.h"
#include "TROOT.h"
#include "TObjString.h"
#include "TMap.h"
#include "TObjArray.h"
#include "TString.h"
#include "Rtypes.h"
#include "TSystem.h"
#include "TVirtualMutex.h"
#include "TProcessUUID.h"
#include "TArchiveFile.h"
#include "TUrl.h"

ClassImp(TAlienFile)

//______________________________________________________________________________
TAlienFile::TAlienFile(const char *url, Option_t *option,
                       const char *ftitle, Int_t compress,Bool_t parallelopen)
{
   // Create an Alien File Object. An AliEn File is the same as a TFile
   // except that it is being accessed via an Alien service. The url
   // argument must be of the form: alien:/[machine]/path/file.root
   // Using the option access, another access protocol (PFN) can be
   // specified for an LFN e.g.:
   //     "alien:///alice/test.root"
   // If you want to write a file on specific storage element use the syntax
   //     "alien:///alice/test.root?&se=Alice::CERN::Storage"
   // The default SE is specified by the enviroment variable alien_CLOSE_SE
   //
   // The URL option "?locate=1" can be appended to a URL to use the TAlienFile
   // interface to locate a file which is accessed by a logical file name.
   // The file name is replaced by an TURL containing the file catalogue
   // authorization envelope. This can be used f.e. to get file access authorization
   // through a client to be used on a PROOF cluster reading data from
   // authorization-envelope enabled xrootd servers. The "locate" option
   // enforces only the retrieval of the access envelope but does not
   // create a physical connection to an xrootd server.
   //
   // If the file specified in the URL does not exist, is not accessable
   // or can not be created the kZombie bit will be set in the TAlienFile
   // object. Use IsZombie() to see if the file is accessable.
   // For a description of the option and other arguments see the TFile ctor.
   // The preferred interface to this constructor is via TFile::Open().

   fSubFileHandle=0;
   fUrl = TUrl(url);

   TUrl lUrl(url);

   TString name(TString("alien://")+TString(lUrl.GetFile()));
   SetName(name);
   TFile::TFile(name, "NET", ftitle, compress);

   fOption = option;

   TString newurl = AccessURL(lUrl.GetUrl(), fOption, ftitle, compress);
   Bool_t lLocate = kFALSE;
   // get the options and check if this is just to prelocate the file ....
   TString urloptions=lUrl.GetOptions();
   TObjArray *objOptions = urloptions.Tokenize("&");
   for (Int_t n = 0; n < objOptions->GetEntries(); n++) {
      TString loption = ((TObjString*)objOptions->At(n))->GetName();
      TObjArray *objTags = loption.Tokenize("=");
      if (objTags->GetEntries() == 2) {
         TString key   =  ((TObjString*)objTags->At(0))->GetName();
         TString value =  ((TObjString*)objTags->At(1))->GetName();
         if ( (key == "locate") && (value == "1") ) {
            lLocate=kTRUE;
         }
      }
      delete objTags;
   }
   delete objOptions;

   if (lLocate) {
      SetName(newurl);
      return;
   }

   TUrl nUrl(newurl.Data());
   TString oldopt;
   TString newopt;

   if (newurl == "")
      goto zombie;

   oldopt = lUrl.GetOptions();

   newopt = nUrl.GetOptions();

   // add the original options from the alien URL
   nUrl.SetOptions(newopt + TString("&") + oldopt + TString("&"));

   if (parallelopen) {
     fSubFileHandle = TFile::AsyncOpen(nUrl.GetUrl(), fOption, ftitle, compress);
     return;
   } else {
     fSubFile = TFile::Open(nUrl.GetUrl(), fOption, ftitle, compress);
   }

   if ((!fSubFile) || (fSubFile->IsZombie())) {
      Error("TAlienFile", "cannot open %s!", url);
      goto zombie;
   }

   nUrl.SetOptions("");
   fSubFile->SetName(nUrl.GetUrl());
   fSubFile->SetTitle(name);

   Init(kFALSE);
   return;

zombie:
   // error in file opening occured, make this object a zombie
   MakeZombie();
   gDirectory = gROOT;
   return;
}

//______________________________________________________________________________
TString TAlienFile::AccessURL(const char *url, Option_t *option,
                              const char *, Int_t)
{
   //access a URL
   TString stmp;
   Bool_t create;
   Bool_t recreate;
   Bool_t update;
   Bool_t read;

   TUrl purl(url);

   // find out the storage element and the lfn from the given url
   TString storageelement="";
   TString file = purl.GetFile();

   Bool_t publicaccess=kFALSE;
   storageelement = gSystem->Getenv("alien_CLOSE_SE");

   // get the options and set the storage element
   TString urloptions=purl.GetOptions();
   TObjArray *objOptions = urloptions.Tokenize("&");
   for (Int_t n = 0; n < objOptions->GetEntries(); n++) {
      TString loption = ((TObjString*)objOptions->At(n))->GetName();
      TObjArray *objTags = loption.Tokenize("=");
      if (objTags->GetEntries() == 2) {
         TString key   =  ((TObjString*)objTags->At(0))->GetName();
         TString value =  ((TObjString*)objTags->At(1))->GetName();
         if ( (key == "se") || (key == "SE") || (key == "Se") ) {
            storageelement = value;
         }
         if ((key == "publicaccess") || (key == "PublicAccess") ||
             (key == "PUBLICACCESS")) {
            if (atoi( value.Data()))
               publicaccess = kTRUE;
         }
      }
      delete objTags;
   }
   delete objOptions;

   fOption = option;
   fOption.ToUpper();
   fSubFile = 0;

   TObjString* urlStr=0;
   TObjString* authzStr=0;

   TString command;

   TIterator* iter = 0;
   TObject* object = 0;

   TGridResult* result;
   TAlienResult* alienResult;
   TList* list;

   TString stringurl;
   TString anchor;
   TObjArray* tokens;
   anchor="";

   TString newurl;
   if (fOption == "NEW")
      fOption = "CREATE";

   create = (fOption == "CREATE") ? kTRUE : kFALSE;
   recreate = (fOption == "RECREATE") ? kTRUE : kFALSE;
   update = (fOption == "UPDATE") ? kTRUE : kFALSE;
   read = (fOption == "READ") ? kTRUE : kFALSE;

   if (!create && !recreate && !update && !read) {
      read = kTRUE;
      fOption = "READ";
   }

   if (create || recreate || update) {
      fWritable=kTRUE;
   }

   if (recreate) {
      fOption = "CREATE";
      create = kTRUE;
   }

   /////////////////////////////////////////////////////////////////////////////////////////
   // first get an active Grid connection

   if (!gGrid) {
      // no TAlien existing ....
      Error("TAlienFile", "no active GRID connection found");
      goto zombie2;
   } else {
      if ((strcmp(gGrid->GetGrid(), "alien"))) {
         Error("TAlienFile", "you don't have an active <alien> grid!");
         goto zombie2;
      }
   }

   /////////////////////////////////////////////////////////////////////////////////////////
   // get the authorization from the catalogue

   if (read) {
      // get the read access
      if (publicaccess)
         command = TString("access -p read ");
      else
         command = TString("access read ");
   }

   if (create) {
      command = TString("access write-once ");
   }

   if (recreate) {
      command = TString("access write-version ");
   }

   if (update) {
      command = TString("access write-version ");
   }

   command += file;

   if (fWritable) {
      // append the storage element environment variable
      command += " ";
      command += storageelement;
   }

   fLfn = file;
   result = gGrid->Command(command.Data(),kFALSE,TAlien::kOUTPUT);
   alienResult = dynamic_cast<TAlienResult*>(result);
   list = dynamic_cast<TList*>(alienResult);
   if (!list) {
      if (result) {
         delete result;
      }
      Error("TAlienFile", "cannot get the access envelope for %s",purl.GetUrl());
      goto zombie2;
   }

   iter = list->MakeIterator();
   object = 0;

   while ((object = iter->Next()) != 0) {
      TMap* map = dynamic_cast<TMap*>(object);

      TObject* urlObject = map->GetValue("url");
      urlStr = dynamic_cast<TObjString*>(urlObject);

      TObject* authzObject = map->GetValue("envelope");
      authzStr = dynamic_cast<TObjString*>(authzObject);

      // there is only one result line .... in case it is at all ....
      break;
   }

   if ((!urlStr) || (!authzStr)) {
      if (fWritable) {
         Error("TAlienFile", "didn't get the authorization to write %s",purl.GetUrl());
      } else {
         Error("TAlienFile", "didn't get the authorization to read %s",purl.GetUrl());
      }

      Info("TAlienFile","Command::Stdout !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
      gGrid->Stdout();
      Info("TAlienFile","Command::Stderr !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
      gGrid->Stderr();
      Info("TAlienFile","End of Output   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
      delete iter;
      delete result;
      goto zombie2;
   }

   delete iter;
   delete result;

   fAuthz = authzStr->GetName();
   stringurl = urlStr->GetName();

   tokens = stringurl.Tokenize("#");

   if (tokens->GetEntries() == 2) {
     anchor = ((TObjString*)tokens->At(1))->GetName();
   }
   urlStr->SetString(((TObjString*)tokens->At(0))->GetName());
   if (tokens) {
     delete tokens;
   }


   newurl = urlStr->GetName();
   stmp = purl.GetAnchor();
   newurl += TString("?&authz=");
   newurl += authzStr->GetName();
   if (stmp != "") {
     newurl += "#";
     newurl += purl.GetAnchor();
   } else {
     if (anchor.Length()) {
       newurl += "#";
       newurl += anchor;
     }
     //=======
     //      newurl += "#";
     //      newurl += purl.GetAnchor();
     //>>>>>>> 1.17
   }
   return newurl;

zombie2:
   // error in file opening occured, make this object a zombie
   return "";
}

//______________________________________________________________________________
TAlienFile::~TAlienFile()
{
   // TAlienFile file dtor.
  R__LOCKGUARD2(gROOTMutex);
   if (fSubFile) {
      Close();

      gROOT->GetListOfFiles()->Remove(fSubFile);
      gROOT->GetUUIDs()->RemoveUUID(fSubFile->GetUniqueID());
      SafeDelete(fSubFile);
   }
   //   gROOT->GetListOfFiles()->Remove(this);
   //   gROOT->GetUUIDs()->RemoveUUID(this->GetUniqueID());

   fSubFile = 0;
   gFile = 0;
   gDirectory = gROOT;
   if (gDebug)
      Info("~TAlienFile", "dtor called for %s", GetName());
}

//______________________________________________________________________________
Bool_t TAlienFile::ReadBuffer(char *buf, Int_t len)
{
   // Read specified byte range.
   // Returns kTRUE in case of error.

   if (fSubFile)
      return fSubFile->ReadBuffer(buf, len);
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TAlienFile::WriteBuffer(const char *buf, Int_t len)
{
   // Write specified byte range
   // Returns kTRUE in case of error.

   if (fSubFile)
      return fSubFile->WriteBuffer(buf, len);
   return kTRUE;
}


//______________________________________________________________________________
void TAlienFile::Seek(Long64_t offset, ERelativeTo pos)
{
   // Seek into file.

   if (fSubFile) {
      fSubFile->Seek(offset, pos);
   }
}

//______________________________________________________________________________
void TAlienFile::Close(Option_t *option)
{
   // Close file.

   if (fOption == "READ")
      return;

   // set GCLIENT_EXTRA_ARG environment
   gSystem->Setenv("GCLIENT_EXTRA_ARG",fAuthz.Data());

   // commit the envelope
   TString command("commit ");

   command += (Long_t)fSubFile->GetSize();
   command += " ";
   command += fLfn;

   if (fSubFile) {
     fSubFile->Close(option);
   }

   TGridResult* result = gGrid->Command(command, kFALSE,TAlien::kOUTPUT);
   TAlienResult* alienResult = dynamic_cast<TAlienResult*>(result);
   TList* list = dynamic_cast<TList*>(alienResult);
   if (!list) {
      if (result) {
         delete result;
      }
      Error("Close", "cannot commit envelope for %s", fLfn.Data());
   }
   TIterator* iter = list->MakeIterator();
   TObject* object = 0;
   if (fWritable) {
      while ((object = iter->Next()) != 0) {
         TMap* map = dynamic_cast<TMap*>(object);
         TObject* commitObject = map->GetValue(fLfn.Data());
         if (commitObject) {
            TObjString* commitStr = dynamic_cast<TObjString*>(commitObject);
            if (!(strcmp(commitStr->GetName(),"1"))) {
               // the file has been committed
               break;
            }
         }

         Error("Close", "cannot register %s!", fLfn.Data());
         // there is only one result line .... in case it is at all ....
         break;
      }
      delete iter;
      delete result;
   }

   gSystem->Unsetenv("GCLIENT_EXTRA_ARG");
}

//______________________________________________________________________________
void
TAlienFile::Init(Bool_t create) {
   gFile=this;
   if (fSubFileHandle) {
     fSubFile = TFile::Open(fSubFileHandle);
     fSubFileHandle=0;
     if ((!fSubFile) || (fSubFile->IsZombie())) {
       Error("TAlienFile", "cannot open %s!", GetName());
       gFile = 0;
       gDirectory = gROOT;
       MakeZombie();
       return;
     }
   }
   {
     R__LOCKGUARD2(gROOTMutex);
     //     gROOT->GetListOfFiles()->Remove(fSubFile);
     //     gROOT->GetUUIDs()->RemoveUUID(fSubFile->GetUniqueID());
     //     gROOT->GetListOfFiles()->Add(this);
     //     gROOT->GetUUIDs()->AddUUID(fUUID,this);
   }

}
