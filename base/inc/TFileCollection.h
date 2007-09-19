// @(#)root/base:$Id$
// Author: Jan Fiete Grosse-Oetringhaus  01/06/07

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFileCollection
#define ROOT_TFileCollection

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFileCollection                                                      //
//                                                                      //
// Class that contains a list of TFileInfo's and accumulated meta       //
// data information about its entries. This class is used to describe   //
// file sets as stored by Grid file catalogs, by PROOF or any other     //
// collection of TFile names.                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

class THashList;
class TList;
class TFileInfo;
class TFileInfoMeta;


class TFileCollection : public TNamed {

private:
   THashList  *fList;               //-> list of TFileInfos
   TList      *fMetaDataList;       //-> generic list of file meta data object(s) (summed over entries of fList)
   Long64_t    fTotalSize;          // total size of files in the list
   Float_t     fStagedPercentage;   // percentage of files staged

   TFileCollection(const TFileCollection&);             // not implemented
   TFileCollection& operator=(const TFileCollection&);  // not implemented

public:
   TFileCollection(const char *name = 0, const char *title = 0, const char *file = 0);
   virtual ~TFileCollection();

   void            Add(TFileInfo *info);
   void            AddFromFile(const char *file);
   void            AddFromDirectory(const char *dir);
   TList          *GetList() { return (TList*) fList; }

   void            Update();
   void            Sort();
   void            SetAnchor(const char *anchor) const;
   void            Print(Option_t *option = "") const;

   Long64_t        GetTotalSize() const { return fTotalSize; }
   Float_t         GetStagedPercentage() const { return fStagedPercentage; }
   Float_t         GetCorruptedPercentage() const;

   const char     *GetDefaultTreeName() const;
   Long64_t        GetTotalEntries(const char *tree) const;
   TFileInfoMeta  *GetMetaData(const char *meta = 0) const;

   TFileCollection *GetStagedSubset();

   ClassDef(TFileCollection, 1)  // Collection of TFileInfo objects
};

#endif
