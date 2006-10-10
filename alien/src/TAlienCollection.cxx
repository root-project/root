// @(#)root/alien:$Name:  $:$Id: TAlienCollection.cxx,v 1.9 2006/10/05 14:56:24 rdm Exp $
// Author: Andreas-Joachim Peters 9/5/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TAlienCollection                                                     //
//                                                                      //
// Class which manages collection of files on AliEn middleware.         //
// The file collection is in the form of an XML file.                   //
//                                                                      //
// The internal list is managed as follows:                             //
// TList* ===> TMap*(file) ===> TMap*(attributes)                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TAlienCollection.h"
#include "TAlienResult.h"
#include "TList.h"
#include "TMap.h"
#include "TFile.h"
#include "TXMLEngine.h"
#include "TObjString.h"
#include "TEventList.h"
#include "TObjArray.h"


ClassImp(TAlienCollection)

//______________________________________________________________________________
TAlienCollection::TAlienCollection(const char *localcollectionfile)
{
   // Create Alien event collection, by reading collection for the specified
   // file.

   fXmlFile = localcollectionfile;
   fEventList = new TList();
   fEventList->SetOwner(kTRUE);
   fEventListIter = new TIter(fEventList);
   fCurrent = 0;
   if (localcollectionfile != 0) {
      ParseXML();
   }
}

//______________________________________________________________________________
TAlienCollection::~TAlienCollection()
{
   // Clean up event file collection.

   if (fEventList)
      delete fEventList;

   if (fEventListIter)
      delete fEventListIter;
}

//______________________________________________________________________________
TAlienCollection *TAlienCollection::Open(const char *localcollectionfile)
{
   // Static method used to create an Alien event collection, by reading
   // collection for the specified file.

   TAlienCollection *collection = new TAlienCollection(localcollectionfile);
   return collection;
}

//______________________________________________________________________________
TFile *TAlienCollection::OpenFile(const char *filename) const
{
   // Open an file via its TURL.

   const char *turl = GetTURL(filename);
   if (turl) {
      return TFile::Open(turl);
   }
   return 0;
}

//______________________________________________________________________________
void TAlienCollection::Reset()
{
   // Reset file iterator.

   fEventListIter->Reset();
   fCurrent = 0;
}

//______________________________________________________________________________
void TAlienCollection::ParseXML()
{
   // Parse event file collection XML file.

   TXMLEngine xml;

   XMLDocPointer_t xdoc = xml.ParseFile(fXmlFile);
   if (!xdoc) {
      Error("ParseXML", "cannot parse the xml file %s", fXmlFile.Data());
      return;
   }

   XMLNodePointer_t xalien = xml.DocGetRootElement(xdoc);
   if (!xalien) {
      Error("ParseXML", "cannot find the <alien> tag in %s",
            fXmlFile.Data());
      return;
   }

   XMLNodePointer_t xcollection = xml.GetChild(xalien);
   if (!xcollection) {
      Error("ParseXML", "cannot find the <collection> tag in %s",
            fXmlFile.Data());
      return;
   }

   XMLNodePointer_t xevent = xml.GetChild(xcollection);;
   if (!xevent) {
      Error("ParseXML", "cannot find the <event> tag in %s",
            fXmlFile.Data());
      return;
   }
   if (!xevent)
      return;

   do {
      if (xml.GetAttr(xevent, "name")) {
         TMap *files = new TMap();

         // here is our event
         //      printf("Found event: %s\n",xml.GetAttr(xevent,"name"));

         // files
         XMLNodePointer_t xfile = xml.GetChild(xevent);
         if (!xfile)
            continue;

         Bool_t firstfile = kTRUE;
         do {
            // here we have an event file
            // get the attributes;
            xml.GetAttr(xfile, "lfn");
            xml.GetAttr(xfile, "turl");

            TMap *attributes  = new TMap();
            TObjString *oname = new TObjString(xml.GetAttr(xfile, "name"));
            TObjString *oturl = new TObjString(xml.GetAttr(xfile, "turl"));
            TObjString *olfn  = new TObjString(xml.GetAttr(xfile, "lfn"));
            TObjString *omd5  = new TObjString(xml.GetAttr(xfile, "md5"));
            TObjString *osize = new TObjString(xml.GetAttr(xfile, "size"));
            TObjString *oguid = new TObjString(xml.GetAttr(xfile, "guid"));
            TObjString *oseStringlist =
               new TObjString(xml.GetAttr(xfile, "seStringlist"));
            TObjString *oevlist =
               new TObjString(xml.GetAttr(xfile, "evlist"));
            // if oevlist is defined, we parse it and fill a TEventList
            printf("The Eventlist is %s\n", oevlist->GetName());
            if (oevlist->GetName() != "") {
               TEventList *xmlevlist =
                  new TEventList(oturl->GetName(), oguid->GetName());
               TString stringevlist = oevlist->GetName();
               TObjArray *evlist = stringevlist.Tokenize(",");
               for (Int_t n = 0; n < evlist->GetEntries(); n++) {
                  xmlevlist->Enter(atol(((TObjString *) evlist->At(n))->GetName()));
               }
               attributes->Add(new TObjString("evlist"), xmlevlist);
            }
            attributes->Add(new TObjString("name"), oname);
            attributes->Add(new TObjString("turl"), oturl);
            attributes->Add(new TObjString("lfn"), olfn);
            attributes->Add(new TObjString("md5"), omd5);
            attributes->Add(new TObjString("size"), osize);
            attributes->Add(new TObjString("guid"), oguid);
            attributes->Add(new TObjString("seStringlist"), oseStringlist);
            files->Add(new TObjString(xml.GetAttr(xfile, "name")), attributes);

            // we add the first file always as a file without name to the map
            if (firstfile) {
               files->Add(new TObjString(""), attributes);
               firstfile = kFALSE;
            }
         } while ((xfile = xml.GetNext(xfile)));
         fEventList->Add(files);
      }
   } while ((xevent = xml.GetNext(xevent)));
}

//______________________________________________________________________________
TMap *TAlienCollection::Next()
{
   // Return next event file map.

   fCurrent = (TMap *) fEventListIter->Next();
   return fCurrent;
}

//______________________________________________________________________________
Bool_t TAlienCollection::Remove(TMap * map)
{
   // Return next event file map.
   if (fEventList->Remove(map)) {
      return kTRUE;
   } else {
      return kFALSE;
   }
}

//______________________________________________________________________________
const char *TAlienCollection::GetTURL(const char *filename) const
{
   // Get a file's transport URL (TURL). Returns 0 in case of error.

   if (fCurrent) {
      TMap *obj = (TMap *) fCurrent->GetValue(filename);
      if (obj) {
         if (obj->GetValue("turl")) {
            return (((TObjString *) obj->GetValue("turl"))->GetName());
         }
      }
   }
   Error("GetTURL", "cannot get TURL of file %s", filename);
   return 0;
}

//______________________________________________________________________________
const char *TAlienCollection::GetLFN() const
{
   // Get a file's LFN. Returns 0 in case of error.

   if (fCurrent) {
      TMap *obj = (TMap *) fCurrent->GetValue("");
      if (obj) {
         if (obj->GetValue("lfn")) {
            return (((TObjString *) obj->GetValue("lfn"))->GetName());
         }
      }
   }
   Error("GetLFN", "cannot get LFN");
   return 0;
}

//______________________________________________________________________________
void TAlienCollection::Print(Option_t *) const
{
   // Print event file collection.

   Info("Print", "dumping %d elements", fEventList->GetSize());
   TIter next(fEventList);
   TMap *filemap;
   Int_t count = 0;
   while ((filemap = (TMap *) next())) {
      count++;
      Info("Print", "printing element %d", count);
      filemap->Print();
   }
}

//______________________________________________________________________________
TDSet *TAlienCollection::GetDataset(const char *type, const char *objname,
                                    const char *dir)
{
   //Get data set.

   Reset();
   TMap *mapp;
   TDSet *dset = new TDSet(type, objname, dir);
   if (!dset) {
      return 0;
   }

   while ((mapp = Next())) {
      if (((TObjString *) fCurrent->GetValue("")))
         dset->Add(((TMap *) (fCurrent->GetValue("")))->GetValue("turl")->
                   GetName());;
   }
   return dset;
}

//______________________________________________________________________________
TGridResult *TAlienCollection::GetGridResult(const char *filename,
                                             Bool_t publicaccess)
{
   //Return grid result.

   Reset();
   TMap *mapp;
   TGridResult *result = new TAlienResult();

   while ((mapp = Next())) {
      if (((TObjString *) fCurrent->GetValue(filename))) {
         TMap *attributes = (TMap *) fCurrent->GetValue(filename)->Clone();
         if (publicaccess) {
            attributes->Add(new TObjString("options"),
                            new TObjString("&publicaccess=1"));
         }
         result->Add(attributes);
      }
   }
   return dynamic_cast < TGridResult * >(result);
}

//______________________________________________________________________________
Bool_t TAlienCollection::OverlapCollection(TAlienCollection * comparator)
{
   // return kTRUE if comparator overlaps with this
   if ((!comparator)) {
      return kFALSE;
   }

   loopagain:
   // loop over col1 and try to find it in col2
   this->Reset();
   // loop over all elements in reference (=this)
   TMap *overlapmap;
   while ((overlapmap = this->Next())) {
      comparator->Reset();
      Bool_t found = kFALSE;
      // try to find in the comparator collection
      while ((comparator->Next())) {
         TString s1 = this->GetLFN();
         TString s2 = comparator->GetLFN();
         if (s1 == s2) {
            found = kTRUE;
            break;
         }
      }
      if (!found) {
         this->Remove(overlapmap);
         goto loopagain;
      }
   }
   return kTRUE;
}
