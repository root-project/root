// @(#)root/alien:$Id$
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
// it's data via TXNetFile and gets authorization and the TXNetFile     //
// URL from an alien service.                                           //
//                                                                      //
// Filenames are standard URL format with protocol "alien".             //
// The following are valid TAlienFile URL's:                            //
//                                                                      //
//    alien:///alice/cern.ch/user/p/peters/test.root                    //
//    /alien/alice/cern.ch/user/p/peters/test.root                      //
//                                                                      //
//    - notice that URLs like /alien/alice... are converted internally  //
//      to alien://alice...                                             //
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
#include "TStopwatch.h"
#include "TVirtualMonitoring.h"
#include "TVirtualMutex.h"
#include "TProcessUUID.h"
#include "TUrl.h"
#include "TError.h"
#include <cstdlib>

ClassImp(TAlienFile)

#define MAX_FILE_IMAGES 16

//______________________________________________________________________________
TAlienFile::TAlienFile(const char *purl, Option_t *option,
                       const char *ftitle, Int_t compress,
                       Bool_t parallelopen, const char *lurl,
                       const char *authz) :
   TXNetFile(purl, option, ftitle, compress, 0, parallelopen, lurl)
{
   // Create an Alien File Object. An AliEn File is the same as a TFile
   // except that its real tranfer URL is resolved via an Alien service. The url
   // argument must be of the form: alien:/[machine]/path/file.root
   // Using the option access, another access protocol (PFN) can be
   // specified for an LFN e.g.:
   //     "alien:///alice/test.root"
   // If you want to write a file on specific storage element use the syntax
   //     "alien:///alice/test.root?&se=Alice::CERN::Storage"
   // The default SE is specified by the environment variable alien_CLOSE_SE
   //
   // If you read a file, the closest file image to alien_CLOSE_SE is taken.
   // If the file cannot opened from the closest image, the next image is tried,
   // until there is no image location left to be tried.
   //
   // If the file specified in the URL does not exist, is not accessable
   // or can not be created the kZombie bit will be set in the TAlienFile
   // object. Use IsZombie() to see if the file is accessable.
   // For a description of the option and other arguments see the TFile ctor.
   // The preferred interface to this constructor is via TFile::Open().
   //
   // Warning: TAlienFile objects should only be created through the factory functions:
   //    TFile::Open("alien://...");
   // or
   //    TAlienFile::Open("alien://...");
   //
   // Don't use "new TAlienFile" directly unless you know, what you are doing
   //

   TUrl logicalurl(lurl);
   fLfn = logicalurl.GetFile();
   fAuthz = authz;
   fGUID = "";
   fUrl = "";
   fPfn = "";
   fSE = "";
   fImage = 0;
   fNreplicas = 0;
   fOpenedAt = time(0);
}

//______________________________________________________________________________
TAlienFile *TAlienFile::Open(const char *url, Option_t *option,
                             const char *ftitle, Int_t compress,
                             Bool_t parallelopen)
{
   // Static method used to create a TAlienFile object. For options see
   // TAlienFile ctor.

   if (!gGrid) {
      ::Error("TAlienFileAccess", "No GRID connection available!");
      return 0;
   }
   TUrl lUrl(url);

   // Report this open phase as a temp one, since we have no object yet
   if (gMonitoringWriter)
      gMonitoringWriter->SendFileOpenProgress(0, 0, "alienopen", kFALSE);

   TString name(TString("alien://") + TString(lUrl.GetFile()));
   TString fAName = name;
   TString fAOption = option;
   TUrl fAUrl;
   TString sguid, pfnStr;
   Int_t nreplicas = 0;
   Bool_t fAWritable;
   fAWritable = kFALSE;
   TString authz;

   // Access a URL.

   TString stmp;
   Bool_t create;
   Bool_t recreate;
   Bool_t update;
   Bool_t read;

   TUrl purl(url);

   // find out the storage element and the lfn from the given url
   TString storageelement;
   storageelement = "";
   TString file = purl.GetFile();

   Bool_t publicaccess = kFALSE;
   storageelement = gSystem->Getenv("alien_CLOSE_SE");

   // get the options and set the storage element
   TString urloptions = purl.GetOptions();
   TObjArray *objOptions = urloptions.Tokenize("&");
   for (Int_t n = 0; n < objOptions->GetEntries(); n++) {
      TString loption = ((TObjString *) objOptions->At(n))->GetName();
      TObjArray *objTags = loption.Tokenize("=");
      if (objTags->GetEntries() == 2) {
         TString key = ((TObjString *) objTags->At(0))->GetName();
         TString value = ((TObjString *) objTags->At(1))->GetName();
         if (!key.CompareTo("se", TString::kIgnoreCase)) {
            storageelement = value;
         }
         if (!key.CompareTo("publicaccess")) {
            if (atoi(value.Data()))
               publicaccess = kTRUE;
         }
      }
      delete objTags;
   }
   delete objOptions;

   fAOption = option;
   fAOption.ToUpper();

   TObjString *urlStr = 0;
   TString urlStrs;
   TObjString *authzStr = 0;
   TObjString *seStr = 0;
   TString seStrs, snreplicas;

   TString command;
   TString repcommand;
   TIterator *iter = 0;
   TObject *object = 0;

   TGridResult *result;
   TAlienResult *alienResult;
   TList *list;

   TString stringurl;
   TString anchor;
   TObjArray *tokens;
   anchor = "";

   TString newurl;
   if (fAOption == "NEW")
      fAOption = "CREATE";

   create = (fAOption == "CREATE") ? kTRUE : kFALSE;
   recreate = (fAOption == "RECREATE") ? kTRUE : kFALSE;
   update = (fAOption == "UPDATE") ? kTRUE : kFALSE;
   read = (fAOption == "READ") ? kTRUE : kFALSE;

   if (!create && !recreate && !update && !read) {
      read = kTRUE;
      fAOption = "READ";
   }

   if (create || recreate || update) {
      fAWritable = kTRUE;
   }

   if (recreate) {
      fAOption = "RECREATE";
      create = kTRUE;
   }
   /////////////////////////////////////////////////////////////////////////////////////////
   // first get an active Grid connection

   if (!gGrid) {
      // no TAlien existing ....
      ::Error("TAlienFile::Open", "no active GRID connection found");
      fAUrl = "";

      // Reset the temp monitoring info
      if (gMonitoringWriter)
         gMonitoringWriter->SendFileOpenProgress(0, 0, 0, kFALSE);

      return 0;
   } else {
      if ((strcmp(gGrid->GetGrid(), "alien"))) {
         ::Error("TAlienFile::Open", "you don't have an active <alien> grid!");
         fAUrl = "";

         // Reset the temp monitoring info
         if (gMonitoringWriter)
            gMonitoringWriter->SendFileOpenProgress(0, 0, 0, kFALSE);

         return 0;
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

   if (fAWritable) {
      // append the storage element environment variable
      command += " ";
      command += storageelement;
   }

   TString fALfn = file;

   int imagenr = 0;

   do {
      imagenr++;
      repcommand = command;
      if (!fAWritable) {
         // in the read case, try all image locations
         if (storageelement != "") {
            repcommand += " ";
            repcommand += storageelement;
         } else {
            repcommand += " ";
            repcommand += "unknown";
         }
         repcommand += " 0 ";
         repcommand += imagenr;
      }

      if (gSystem->Getenv("ALIEN_SITE")) {
         repcommand += " 0 ";
         repcommand += gSystem->Getenv("ALIEN_SITE");
      }
      result = gGrid->Command(repcommand.Data(), kFALSE, TAlien::kOUTPUT);
      alienResult = dynamic_cast < TAlienResult * >(result);
      list = dynamic_cast < TList * >(alienResult);
      if (!list) {

         if (seStr)
            ::Error("TAlienFile::Open",
                    "cannot get the access envelope for %s and image %u in SE <%s>",
                    purl.GetUrl(), imagenr, seStr->GetName());
         else
            ::Error("TAlienFile::Open",
                    "cannot get the access envelope for %s and image %u",
                    purl.GetUrl(), imagenr);
         fAUrl = "";
         if (result) {
            delete result;
         }

         // Reset the temp monitoring info
         if (gMonitoringWriter)
            gMonitoringWriter->SendFileOpenProgress(0, 0, 0, kFALSE);

         return 0;
      }

      iter = list->MakeIterator();
      object = 0;

      Bool_t imageeof = kFALSE;

      while ((object = iter->Next()) != 0) {
         TMap *map = dynamic_cast < TMap * >(object);

         TObject *urlObject = map->GetValue("url");
         urlStr = dynamic_cast < TObjString * >(urlObject);
         if (urlStr) urlStrs = urlStr->GetName();

         TObject *authzObject = map->GetValue("envelope");
         authzStr = dynamic_cast < TObjString * >(authzObject);

         TObject *seObject = map->GetValue("se");
         seStr = dynamic_cast < TObjString * >(seObject);
         if (seStr) seStrs = seStr->GetName();

         TObject *nreplicasObject = map->GetValue("nSEs");
         if (nreplicasObject) {
            snreplicas = nreplicasObject->GetName();
            nreplicas = snreplicas.Atoi();
         }

         TObject *guidObject = map->GetValue("guid");
         if (guidObject) sguid = guidObject->GetName();

         TObject *pfnObject = map->GetValue("pfn");
         if (pfnObject) pfnStr = pfnObject->GetName();

         if (map->GetValue("eof")) {
            imageeof = kTRUE;
            // there is only one result line .... in case it is at all ....
         }
         break;
      }

      if ((!urlStr) || (!authzStr)) {
         if (fAWritable) {
            ::Error("TAlienFile::Open",
                    "didn't get the authorization to write %s",
                    purl.GetUrl());
         } else {
            if (!imageeof) {
               ::Error("TAlienFile::Open",
                       "didn't get the authorization to read %s from location %u",
                       purl.GetUrl(), imagenr);
            }
         }
         if (!imageeof) {
            ::Info("TAlienFile::Open",
               "Command::Stdout !!!");
            gGrid->Stdout();
            ::Info("TAlienFile::Open",
               "Command::Stderr !!!");
            gGrid->Stderr();
            ::Info("TAlienFile::Open",
               "End of Output   !!!");
         }
         delete iter;
         delete result;
         fAUrl = "";
         if (!imageeof) {
            continue;
         } else {
            // if the service signals eof, it makes no sense to check more replicas
            ::Error("TAlienFile::Open",
                    "No more images to try - giving up");

            // Reset the temp monitoring info
            if (gMonitoringWriter)
               gMonitoringWriter->SendFileOpenProgress(0, 0, 0, kFALSE);

            return 0;
         }
      }

      delete iter;

      authz = authzStr->GetName();
      stringurl = urlStr->GetName();

      tokens = stringurl.Tokenize("#");

      if (tokens->GetEntries() == 2) {
         anchor = ((TObjString *) tokens->At(1))->GetName();
         urlStr->SetString(((TObjString *) tokens->At(0))->GetName());
      }

      delete tokens;

      newurl = urlStr->GetName();
      stmp = purl.GetAnchor();
      newurl += TString("?&authz=");
      newurl += authzStr->GetName();

      if (!fAWritable) {
         if (seStr)
            ::Info("TAlienFile::Open", "Accessing image %u of %s in SE <%s>",
                   imagenr, purl.GetUrl(), seStr->GetName());
         else
            ::Info("TAlienFile::Open", "Accessing image %u of %s", imagenr, purl.GetUrl());
      }
      // the treatement of ZIP files is done in the following way:
      // LFNs in AliEn pointing to files in ZIP archives don't contain the .zip suffix in the file name
      // to tell TArchiveFile about the ZIP nature, we add to the URL options 'zip=<member>'
      // This options are not visible in the file name, they are passed through TXNetFile to TNetFile to TArchiveFile

      if (stmp != "") {
         newurl += "#";
         newurl += purl.GetAnchor();
         TString lUrlfile = lUrl.GetFile();
         TString lUrloption;
         lUrloption = "zip=";
         lUrloption += purl.GetAnchor();
         lUrloption += "&mkpath=1";
         lUrl.SetFile(lUrlfile);
         lUrl.SetOptions(lUrloption);
      } else {
         if (anchor.Length()) {
            newurl += "#";
            newurl += anchor;
            TString lUrlfile = lUrl.GetFile();
            TString lUrloption;
            lUrloption = "zip=";
            lUrloption += anchor;
            lUrloption += "&mkpath=1";
            lUrl.SetFile(lUrlfile);
            // lUrl.SetAnchor(anchor);
            lUrl.SetOptions(lUrloption);
         } else {
            TString loption;
            loption = lUrl.GetOptions();
            if (loption.Length()) {
               loption += "&mkpath=1";
               lUrl.SetOptions(loption.Data());
            } else {
               lUrl.SetOptions("mkpath=1");
            }
         }
      }

      fAUrl = TUrl(newurl);

      // append the original options
      TString oldopt;
      TString newopt;

      if (TString(fAUrl.GetUrl()) == "") {
         // error in file opening occured

          // Reset the temp monitoring info
         if (gMonitoringWriter)
            gMonitoringWriter->SendFileOpenProgress(0, 0, 0, kFALSE);

         return 0;
      }

      TUrl nUrl = fAUrl;
      TUrl oUrl(url);

      oldopt = oUrl.GetOptions();
      newopt = nUrl.GetOptions();

      // add the original options from the alien URL
      if (oldopt.Length()) {
         nUrl.SetOptions(newopt + TString("&") + oldopt);
      } else {
         nUrl.SetOptions(newopt);
      }

      fAUrl = nUrl;
      delete result;
      if (gDebug > 1)
         ::Info("TAlienFile","Opening AUrl <%s> lUrl <%s>",fAUrl.GetUrl(),lUrl.GetUrl());
      TStopwatch timer;
      timer.Start();
      TAlienFile *alienfile =
          new TAlienFile(fAUrl.GetUrl(), fAOption, ftitle, compress,
                         parallelopen, lUrl.GetUrl(), authz);
      timer.Stop();
      if (alienfile->IsZombie()) {
         delete alienfile;
         if (fAWritable) {
            // for the moment we support only 1 try during writing - no alternative locations
            break;
         }
         continue;
      } else {
         alienfile->SetSE(seStrs);
         alienfile->SetPfn(pfnStr);
         alienfile->SetImage(imagenr);
         alienfile->SetNreplicas(nreplicas);
         alienfile->SetGUID(sguid);
         alienfile->SetUrl(urlStrs);
         alienfile->SetElapsed(timer.RealTime());
         return alienfile;
      }
   } while (imagenr < MAX_FILE_IMAGES);

   if (!fAWritable) {
      ::Error("TAlienFile::Open",
              "Couldn't open any of the file images of %s", lUrl.GetUrl());
   }

   // Reset the temp monitoring info
   if (gMonitoringWriter)
      gMonitoringWriter->SendFileOpenProgress(0, 0, 0, kFALSE);

   return 0;
}

//______________________________________________________________________________
TAlienFile::~TAlienFile()
{
   // TAlienFile file dtor.

   if (IsOpen()) {
      Close();
   }
   if (gDebug)
      Info("~TAlienFile", "dtor called for %s", GetName());
}

//______________________________________________________________________________
void TAlienFile::Close(Option_t * option)
{
   // Close the file.

   if (!IsOpen()) return;


   // Close file
   TXNetFile::Close(option);

   if (fOption == "READ")
      return;

   // set GCLIENT_EXTRA_ARG environment
   gSystem->Setenv("GCLIENT_EXTRA_ARG", fAuthz.Data());

   // commit the envelope
   TString command("commit ");

   Long64_t siz = GetSize();
   if (siz <= 0)
      Error("Close", "the reported size of the written file is <= 0");

   command += siz;
   command += " ";
   command += fLfn;

   TGridResult *result = gGrid->Command(command, kFALSE, TAlien::kOUTPUT);
   TAlienResult *alienResult = dynamic_cast < TAlienResult * >(result);
   TList *list = dynamic_cast < TList * >(alienResult);
   if (!list) {
      if (result) {
         delete result;
      }
      Error("Close", "cannot commit envelope for %s", fLfn.Data());
      gSystem->Unlink(fLfn);
   }
   TIterator *iter = list->MakeIterator();
   TObject *object = 0;
   if (fWritable) {
      while ((object = iter->Next()) != 0) {
         TMap *map = dynamic_cast < TMap * >(object);
         TObject *commitObject = map->GetValue(fLfn.Data());
         if (commitObject) {
            TObjString *commitStr =
                dynamic_cast < TObjString * >(commitObject);
            if (!(strcmp(commitStr->GetName(), "1"))) {
               // the file has been committed
               break;
            }
         }

         Error("Close", "cannot register %s!", fLfn.Data());
         gSystem->Unlink(fLfn);
         // there is only one result line .... in case it is at all ....
         break;
      }
   }
   delete iter;
   delete result;

   gSystem->Unsetenv("GCLIENT_EXTRA_ARG");

}

//______________________________________________________________________________
TString TAlienFile::SUrl(const char *lfn)
{
   // Get surl from lfn by asking AliEn catalog.

   TString command;
   TString surl;

   if (!lfn) {
      return surl;
   }

   TUrl lurl(lfn);
   command = "access -p read ";
   command += lurl.GetFile();

   TGridResult* result;

   if (!gGrid) {
      ::Error("TAlienFile::SUrl","no grid connection");
      return surl;
   }

   result = gGrid->Command(command.Data(), kFALSE, TAlien::kOUTPUT);
   if (!result) {
      ::Error("TAlienFile::SUrl","couldn't get access URL for alien file %s", lfn);
      return surl;
   }

   TIterator *iter = result->MakeIterator();
   TObject *object=0;
   TObjString *urlStr=0;

   object = iter->Next();
   if (object) {
      TMap *map = dynamic_cast < TMap * >(object);
      TObject *urlObject = map->GetValue("url");
      urlStr = dynamic_cast < TObjString * >(urlObject);

      if (urlStr) {
         surl = urlStr->GetName();
         delete object;
         return surl;
      }
   }

   ::Error("TAlienFile::SUrl","couldn't get surl for alien file %s", lfn);
   return surl;
}
