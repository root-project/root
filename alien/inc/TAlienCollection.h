// @(#)root/alien:$Name:  $:$Id: TAlienCollection.h,v 1.1 2005/05/20 11:13:30 rdm Exp $
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

   virtual void ParseXML();

public:
   TAlienCollection() : fEventList(0), fEventListIter(0), fCurrent(0) { }
   TAlienCollection(const char *localCollectionFile);

   virtual ~TAlienCollection();

   void        Reset();
   TMap       *Next();
   const char *GetTURL(const char *name) const;
   void        Print(Option_t *opt) const;
   TFile      *OpenFile(const char *filename) const;
   TList      *GetEventList() const { return fEventList; }

   TDSet      *GetDataset(const char *type, const char *objname = "*", const char *dir = "/");

   static TAlienCollection *Open(const char *localcollectionfile);
   static TAlienCollection *Query(const char *path, const char *pattern, Int_t maxfiles=1000);

   ClassDef(TAlienCollection,1)  // Manages collection of files on AliEn
};

#endif
