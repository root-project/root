// @(#)root/cont:$Name$:$Id$
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


struct ClassRec_t {
   char          *name;
   Version_t      id;
   VoidFuncPtr_t  dict;
   ClassRec_t    *next;
};


class TClassTable : public TObject {

private:
   TClassTable();

   static ClassRec_t  *FindElement(const char *cname, Bool_t insert=kFALSE);
   static void         SortTable();

   static ClassRec_t **fgTable;
   static ClassRec_t **fgSortedTable;
   static int          fgSize;
   static int          fgTally;
   static Bool_t       fgSorted;
   static int          fgCursor;

public:
   ~TClassTable();

   static void          Add(const char *cname, Version_t id, VoidFuncPtr_t dict);
   int                  Classes();
   static Version_t     GetID(const char *cname);
   static VoidFuncPtr_t GetDict(const char *cname);
   static void          Init();
   static char         *Next();
   void                 Print(Option_t *option="");
   static void          PrintTable();
   static void          Remove(const char *cname);
   static void          Terminate();

   ClassDef(TClassTable,0)  //Table of known classes
};

R__EXTERN TClassTable *gClassTable;

extern void AddClass(const char *cname, Version_t id, VoidFuncPtr_t dict);
extern void RemoveClass(const char *cname);

#endif
