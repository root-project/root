// @(#)root/cont:$Id$
// Author: Fons Rademakers   11/08/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TClassTable
#define ROOT_TClassTable


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClassTable                                                          //
//                                                                      //
// This class registers for all classes their name, id and dictionary   //
// function in a hash table.                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TClassRec {
public:
   char            *fName;
   Version_t        fId;
   Int_t            fBits;
   VoidFuncPtr_t    fDict;
   const type_info *fInfo;
   TClassRec       *fNext;
};

namespace ROOT {
   class TMapTypeToClassRec;
}

class TClassTable : public TObject {

friend  void ROOT::ResetClassVersion(TClass*, const char*, Short_t);

private:
   typedef ROOT::TMapTypeToClassRec IdMap_t;

   static TClassRec  **fgTable;
   static TClassRec  **fgSortedTable;
   static IdMap_t     *fgIdMap;
   static int          fgSize;
   static int          fgTally;
   static Bool_t       fgSorted;
   static int          fgCursor;

   TClassTable();

   static TClassRec   *FindElementImpl(const char *cname, Bool_t insert);
   static TClassRec   *FindElement(const char *cname, Bool_t insert=kFALSE);
   static void         SortTable();

public:
   // bits that can be set in pragmabits
   enum { kNoStreamer = 0x01, kNoInputOperator = 0x02, kAutoStreamer = 0x04 };

   ~TClassTable();

   static void          Add(const char *cname, Version_t id,
                            const type_info &info, VoidFuncPtr_t dict,
                            Int_t pragmabits);
   static char         *At(int index);
   int                  Classes();
   static Version_t     GetID(const char *cname);
   static Int_t         GetPragmaBits(const char *name);
   static VoidFuncPtr_t GetDict(const char *cname);
   static VoidFuncPtr_t GetDict(const type_info& info);
   static void          Init();
   static char         *Next();
   void                 Print(Option_t *option="") const;
   static void          PrintTable();
   static void          Remove(const char *cname);
   static void          Terminate();

   ClassDef(TClassTable,0)  //Table of known classes
};

R__EXTERN TClassTable *gClassTable;

namespace ROOT {
   extern void AddClass(const char *cname, Version_t id, VoidFuncPtr_t dict,
                        Int_t pragmabits);
   extern void RemoveClass(const char *cname);
}

#endif
