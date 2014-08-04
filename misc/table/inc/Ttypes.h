/* @(#)root/table:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Ttypes
#define ROOT_Ttypes

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Stypes                                                               //
//                                                                      //
// Basic types used by STAF - ROOT interface.                           //
//                                                                      //
// This header file contains the set of the macro definitions           //
// to generate a ROOT dictionary for "pure" C-strucutre the way ROOT    //
// does it for the "normal" C++ classes                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "Rtypes.h"

#ifndef _QUOTE2_
# ifdef ANSICPP
#  define _QUOTE2_(name1,name2) _QUOTE_(name1##name2)
#else
#  define _QUOTE2_(name1,name2) _QUOTE_(_NAME1_(name1)name2)
# endif
#endif

// #if ROOT_VERSION_CODE >= ROOT_VERSION(3,03,5)

//___________________________________________________________________
#define _TableClassImp_(className,structName)

//___________________________________________________________________
#define TableClassStreamerImp(className)                            \
void className::Streamer(TBuffer &R__b) {                           \
   TTable::Streamer(R__b); }

//___________________________________________________________________
#define TableClassImp(className,structName)                         \
   const char* className::TableDictionary()                         \
   {return TTable::TableDictionary(_QUOTE_(className),_QUOTE_(structName),fgColDescriptors);}\
   _TableClassImp_(className,structName)

//___________________________________________________________________
#define TableClassImpl(className,structName)                        \
  TTableDescriptor *className::fgColDescriptors = 0;                \
  TableClassImp(className,structName)                               \
  TableClassStreamerImp(className)


#define TableImpl(name)                                            \
  TTableDescriptor *_NAME2_(St_,name)::fgColDescriptors = 0;       \
  TableClassImp(_NAME2_(St_,name), _NAME2_(name,_st))              \
  TableClassStreamerImp(_NAME2_(St_,name))

#define TableImp(name)  TableClassImp(_NAME2_(St_,name),_QUOTE2_(St_,name))

#define ClassDefTable(className,structName)         \
  public:                                           \
     static const char* TableDictionary();          \
  protected:                                        \
     static TTableDescriptor *fgColDescriptors;     \
     virtual TTableDescriptor *GetDescriptorPointer() const { return fgColDescriptors;}                 \
virtual void SetDescriptorPointer(TTableDescriptor *list)  { fgColDescriptors = list;}                  \
  public:                                           \
    typedef structName* iterator;                   \
    className() : TTable(_QUOTE_(className),sizeof(structName))    {SetType(_QUOTE_(structName));}      \
    className(const char *name) : TTable(name,sizeof(structName)) {SetType(_QUOTE_(structName));}     \
    className(Int_t n) : TTable(_QUOTE_(className),n,sizeof(structName)) {SetType(_QUOTE_(structName));}\
    className(const char *name,Int_t n) : TTable(name,n,sizeof(structName)) {SetType(_QUOTE_(structName));}\
    structName *GetTable(Int_t i=0) const { return ((structName *)GetArray())+i;}                       \
    structName &operator[](Int_t i){ assert(i>=0 && i < GetNRows()); return *GetTable(i); }             \
    const structName &operator[](Int_t i) const { assert(i>=0 && i < GetNRows()); return *((const structName *)(GetTable(i))); } \
    structName *begin() const  {                      return GetNRows()? GetTable(0):0;}\
    structName *end()   const  {Long_t i = GetNRows(); return          i? GetTable(i):0;}

// -- The member function "begin()" returns a pointer to the first table row
//   (or just zero if the table is empty).
// -- The member function "end()" returns a pointer to the last+1 table row
//   (or just zero if the table is empty).

//  protected:
//    _NAME2_(className,C)() : TChair() {;}
//  public:
//    _NAME2_(className,C)(className *tableClass) : TChair(tableClass) {;}

#define ClassDefineChair(classChairName,classTableName,structName)    \
  public:                                               \
    typedef structName* iterator;                       \
    structName *GetTable(Int_t i) const  {              \
              if (fLastIndx != UInt_t(i)) {             \
                ((classChairName *)this)->fLastIndx = i;        \
                ((classChairName *)this)->fLastRow =            \
                  ((classTableName *)GetThisTable())->GetTable(i);    \
           }; return (structName *)fLastRow; };          \
    structName &operator[](Int_t i){ assert(i>=0 && i < GetNRows()); return *GetTable(i); }    \
    const structName &operator[](Int_t i) const { assert(i>=0 && i < GetNRows()); return *((const structName *)(GetTable(i))); }\
    structName *begin() const  {                      return GetNRows()? GetTable(0):0;}\
    structName *end()   const  {Int_t i = GetNRows(); return          i? GetTable(i):0;}

//
//    ClassDefineChair(_NAME2_(className,C),className,structName)
//    We have to define this macro in full because RootCint doesn't provide the deep CPP evaluation
//    V.Fine 17/12/2003
#define ClassDefChair(className,structName)             \
  public:                                               \
    typedef structName* iterator;                       \
    structName *GetTable(Int_t i) const  {              \
              if (fLastIndx != UInt_t(i)) {             \
                ((_NAME2_(className,C) *)this)->fLastIndx = i;        \
                ((_NAME2_(className,C) *)this)->fLastRow =            \
                  ((className *)GetThisTable())->GetTable(i);              \
           }; return (structName *)fLastRow; }          \
    structName &operator[](Int_t i){ assert(i>=0 && i < GetNRows()); return *GetTable(i); }    \
    const structName &operator[](Int_t i) const { assert(i>=0 && i < GetNRows()); return *((const structName *)(GetTable(i))); }\
    structName *begin() const  {                      return GetNRows()? GetTable(0):0;}\
    structName *end()   const  {Int_t i = GetNRows(); return          i? GetTable(i):0;}


namespace ROOT {
   template <class T> class TTableInitBehavior: public TDefaultInitBehavior {
   public:
      static const char* fgStructName; // Need to be instantiated
      virtual TClass *CreateClass(const char *cname, Version_t id,
                                  const type_info &info, TVirtualIsAProxy *isa,
                                  const char *dfil, const char *ifil,
                                  Int_t dl, Int_t il) const
      {
         TClass *cl = TDefaultInitBehavior::CreateClass(cname, id, info, isa,
                                                        dfil, ifil,dl, il);
         fgStructName = T::TableDictionary();
         return cl;
      }
      virtual void Unregister(const char* classname) const
      {
         TDefaultInitBehavior::Unregister(classname);
         TDefaultInitBehavior::Unregister(fgStructName);
      }
   };
   template <class T> const char * TTableInitBehavior<T >::fgStructName = 0;
}

class TTable;
namespace ROOT {
   template <class RootClass>
      const ROOT::TTableInitBehavior<RootClass> *DefineBehavior(TTable*, RootClass*)
      {
         static ROOT::TTableInitBehavior<RootClass> behave;
         return &behave;
      }
}

#endif
