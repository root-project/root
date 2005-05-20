// @(#)root/alien:$Name:  $:$Id: TAlienJDL.h,v 1.3 2004/11/01 17:38:08 jgrosseo Exp $
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

   static TAlienCollection *Open(const char *localcollectionfile);

   ClassDef(TAlienCollection,1)  // Manages collection of files on AliEn
};

#endif
