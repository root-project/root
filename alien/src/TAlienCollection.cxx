// @(#)root/alien:$Name:  $:$Id: TAlienCollection.cxx,v 1.1 2005/05/20 11:13:30 rdm Exp $
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
//////////////////////////////////////////////////////////////////////////

#include "TAlienCollection.h"
#include "TList.h"
#include "TMap.h"
#include "TFile.h"
#include "TXMLEngine.h"
#include "TObjString.h"


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

   ParseXML();
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
      Error("ParseXML","cannot parse the xml file %s",fXmlFile.Data());
      return;
   }

   XMLNodePointer_t xalien = xml.DocGetRootElement(xdoc);
   if (!xalien) {
      Error("ParseXML","cannot find the <alien> tag in %s",fXmlFile.Data());
      return;
   }

   XMLNodePointer_t xcollection = xml.GetChild(xalien);
   if (!xcollection) {
      Error("ParseXML","cannot find the <collection> tag in %s",fXmlFile.Data());
      return;
   }

   XMLNodePointer_t xevent = xml.GetChild(xcollection);;
   if (!xevent) {
      Error("ParseXML","cannot find the <event> tag in %s",fXmlFile.Data());
      return;
   }
   if (!xevent) return;

   do {
      if (xml.GetAttr(xevent, "name")) {
         TMap *files = new TMap();

         // here is our event
         //      printf("Found event: %s\n",xml.GetAttr(xevent,"name"));

         // files
         XMLNodePointer_t xfile = xml.GetChild(xevent);
         if (!xfile) continue;

         do {
            // here we have an event file
            // get the attributes;
            xml.GetAttr(xfile, "lfn");
            xml.GetAttr(xfile, "turl");
            // Use turl
            files->Add(new TObjString(xml.GetAttr(xfile,"name")) , new TObjString(xml.GetAttr(xfile,"turl")));
         } while ((xfile = xml.GetNext(xfile)));
         fEventList->Add(files);
      }
   } while ((xevent =  xml.GetNext(xevent)));
}

//______________________________________________________________________________
TMap *TAlienCollection::Next()
{
   // Return next event file map.

   fCurrent = (TMap*)fEventListIter->Next();
   return fCurrent;
}

//______________________________________________________________________________
const char *TAlienCollection::GetTURL(const char* filename) const
{
   // Get a file's transport URL (TURL). Returns 0 in case of error.

   if (fCurrent) {
      TObjString *obj = (TObjString*)fCurrent->GetValue(filename);
      if (obj) {
         if (strlen(obj->GetName()))
            return (obj->GetName());
      }
   }
   Error("GetTURL","cannot get TURL of file %s",filename);
   return 0;
}

//______________________________________________________________________________
void TAlienCollection::Print(Option_t *opt) const
{
   // Print event file collection.

   Info("Print", "dumping %d elements", fEventList->GetSize());
   TIter next(fEventList);
   TMap *filemap;
   Int_t count=0;
   while ((filemap = (TMap*)next())) {
      count++;
      Info("Print", "printing element %d", count);
      filemap->Print();
   }
}
