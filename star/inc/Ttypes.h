/* @(#)root/star:$Name:  $:$Id: Ttypes.h,v 1.3 2000/08/09 08:41:22 brun Exp $ */

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
// $Id: Ttypes.h,v 1.3 2000/08/09 08:41:22 brun Exp $
// Basic types used by STAF - ROOT interface.                           //
//                                                                      //
// This header file contains the set of the macro definitions           //
// to generate a ROOT dictionary for "pure" C-strucutre the way ROOT    //
// does it for the "normal" C++ classes                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//*KEEP,Rtypes.
#include "Rtypes.h"
//*KEND.

#ifdef ANSICPP
#   define _QUOTE2_(name1,name2) _QUOTE_(name1##name2)
#else
#   define _QUOTE2_(name1,name2) _QUOTE_(_NAME1_(name1)name2)
#endif

//___________________________________________________________________
#define _TableClassInit_(className,structName)                      \
   extern void AddClass(const char *cname, Version_t id, VoidFuncPtr_t dict, Int_t); \
   extern void RemoveClass(const char *cname);                      \
   class _NAME2_(R__Init,className) {                               \
      public:                                                       \
         _NAME2_(R__Init,className)() {                             \
            AddClass(_QUOTE_(className), className::Class_Version(),\
                     &className::Dictionary, 0);                       \
         }                                                          \
         ~_NAME2_(R__Init,className)() {                            \
            RemoveClass(_QUOTE_(className));                        \
            RemoveClass(_QUOTE_(structName));                       \
         }                                                          \
   };

//___________________________________________________________________
#define _TableClassImp_(className,structName)                       \
   TClass *className::Class()                                       \
          { if (!fgIsA) className::Dictionary(); return fgIsA; }    \
   const char *className::ImplFileName() { return __FILE__; }       \
   int className::ImplFileLine() { return __LINE__; }               \
   TClass *className::fgIsA = 0;                                    \
   _TableClassInit_(className,structName)

//___________________________________________________________________
#define TableClassStreamerImp(className)                            \
void className::Streamer(TBuffer &R__b) {                           \
   TTable::Streamer(R__b); }

//___________________________________________________________________
#define TableClassImp(className,structName)                         \
   void className::Dictionary()                                     \
   {                                                                \
      TClass *c = CreateClass(_QUOTE_(className), Class_Version(),  \
                              DeclFileName(), ImplFileName(),       \
                              DeclFileLine(), ImplFileLine());      \
                                                                    \
      char *structBuf = new char[strlen(_QUOTE2_(structName,.h))+2];\
      strcpy(structBuf,_QUOTE2_(structName,.h));                    \
      char *s = strstr(structBuf,"_st.h");                          \
      if (s) { *s = 0;  strcat(structBuf,".h"); }                   \
      TClass *r = CreateClass(_QUOTE_(structName), Class_Version(), \
                              structBuf, structBuf, 1,  1 );        \
      fgIsA = c;                                                    \
      fgColDescriptors = new TTableDescriptor(r);                   \
   }                                                                \
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
  protected:                                        \
     static TTableDescriptor *fgColDescriptors;     \
     virtual TTableDescriptor *GetDescriptorPointer() const { return fgColDescriptors;}                 \
     virtual void SetDescriptorPointer(TTableDescriptor *list) const { fgColDescriptors = list;}        \
  public:                                           \
    typedef structName* iterator;                   \
    className() : TTable(_QUOTE_(className),sizeof(structName))    {SetType(_QUOTE_(structName));}      \
    className(const Text_t *name) : TTable(name,sizeof(structName)) {SetType(_QUOTE_(structName));}     \
    className(Int_t n) : TTable(_QUOTE_(className),n,sizeof(structName)) {SetType(_QUOTE_(structName));}\
    className(const Text_t *name,Int_t n) : TTable(name,n,sizeof(structName)) {SetType(_QUOTE_(structName));}\
    structName *GetTable(Int_t i=0) const { return ((structName *)GetArray())+i;}                       \
    structName &operator[](Int_t i){ assert(i>=0 && i < GetNRows()); return *GetTable(i); }             \
    const structName &operator[](Int_t i) const { assert(i>=0 && i < GetNRows()); return *((const structName *)(GetTable(i))); } \
    structName *begin() const  {                      return GetNRows()? GetTable(0):0;}\
    structName *end()   const  {Int_t i = GetNRows(); return          i? GetTable(i):0;}

// -- The member function "begin()" returns a pointer to the first table row
//   (or just zero if the table is empty).
// -- The member function "end()" returns a pointer to the last+1 table row
//   (or just zero if the table is empty).

//  protected:
//    _NAME2_(className,C)() : TChair() {;}
//  public:
//    _NAME2_(className,C)(className *tableClass) : TChair(tableClass) {;}

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

#endif
