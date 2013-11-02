// @(#)root/cont
// Author: Philippe Canal Aug 2013

/*************************************************************************
 * Copyright (C) 1995-2013, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TListOfFunctions
#define ROOT_TListOfFunctions

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TListOfFunctions                                                     //
//                                                                      //
// A collection of TFunction objects designed for fast access given a   //
// DeclId_t and for keep track of TFunction that were described         //
// unloaded function.                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_THastList
#include "THashList.h"
#endif

#ifndef ROOT_TDictionary
#include "TDictionary.h"
#endif

class TExMap;
class TFunction;

class TListOfFunctions : public THashList
{
private:
   typedef TDictionary::DeclId_t DeclId_t;
   TClass    *fClass; // Context of this list.  Not owned.
   
   TExMap    *fIds;      // Map from DeclId_t to TFunction*
   THashList *fUnloaded; // Holder of TFunction for unloaded functions.
   
   TListOfFunctions(const TListOfFunctions&);              // not implemented
   TListOfFunctions& operator=(const TListOfFunctions&);   // not implemented
   
public:
   
   TListOfFunctions(TClass *cl);
   ~TListOfFunctions();

   virtual void Clear(Option_t *option);
   virtual void Delete(Option_t *option="");

   using THashList::FindObject;
   virtual TObject   *FindObject(const char *name) const;

   TFunction *Get(DeclId_t id);

   void       AddFirst(TObject *obj);
   void       AddFirst(TObject *obj, Option_t *opt);
   void       AddLast(TObject *obj);
   void       AddLast(TObject *obj, Option_t *opt);
   void       AddAt(TObject *obj, Int_t idx);
   void       AddAfter(const TObject *after, TObject *obj);
   void       AddAfter(TObjLink *after, TObject *obj);
   void       AddBefore(const TObject *before, TObject *obj);
   void       AddBefore(TObjLink *before, TObject *obj);

   void       RecursiveRemove(TObject *obj);
   TObject   *Remove(TObject *obj);
   TObject   *Remove(TObjLink *lnk);

   void Load();
   void Unload();
   void Unload(TFunction *func);
   
   ClassDef(TListOfFunctions,0);  // List of TFunctions for a class
};

#endif // ROOT_TListOfFunctions
