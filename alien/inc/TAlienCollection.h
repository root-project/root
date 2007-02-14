// @(#)root/alien:$Name:  $:$Id: TAlienCollection.h,v 1.4 2006/10/05 14:56:24 rdm Exp $
// Author: Andreas-Joachim Peters 9/5/2005

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TAlienCollection
#define ROOT_TAlienCollection

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

#ifndef ROOT_TGridCollection
#include "TGridCollection.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TDSet
#include "TDSet.h"
#endif
#ifndef ROOT_TGridResult
#include "TGridResult.h"
#endif

class TMap;
class TList;
class TIter;
class TFile;


class TAlienCollection : public TGridCollection {

private:
   TString     fXmlFile;        // collection XML file
   TList      *fEventList;      // list with event file maps
   TIter      *fEventListIter;  // event file list iterator
   TMap       *fCurrent;        // current event file map

   TAlienCollection(const TAlienCollection &); // Not implemented
   TAlienCollection& operator=(const TAlienCollection &); // Not implemented
   virtual void ParseXML();

public:
   TAlienCollection() : fXmlFile(), fEventList(0), fEventListIter(0), fCurrent(0) { }
   TAlienCollection(const char *localCollectionFile);

   virtual ~TAlienCollection();

   void         Reset();
   TMap        *Next();
   Bool_t       Remove(TMap *map);
   const char  *GetTURL(const char *name) const;
   const char  *GetLFN() const;
   void         Print(Option_t *opt) const;
   TFile       *OpenFile(const char *filename) const;
   TList       *GetEventList() const { return fEventList; }
   Bool_t       OverlapCollection(TAlienCollection *comparator);

   TDSet       *GetDataset(const char *type, const char *objname = "*", const char *dir = "/");
   TGridResult *GetGridResult(const char *filename="", Bool_t publicaccess=kFALSE);

   static TAlienCollection *Open(const char *localcollectionfile);

   ClassDef(TAlienCollection,1)  // Manages collection of files on AliEn
};

#endif
