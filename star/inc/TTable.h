// @(#)root/star:$Name:  $:$Id: TTable.h,v 1.4 2000/09/05 09:21:24 brun Exp $
// Author: Valery Fine(fine@mail.cern.ch)   03/07/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
// $Id: TTable.h,v 1.4 2000/09/05 09:21:24 brun Exp $
#ifndef ROOT_TTable
#define ROOT_TTable

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TTable                                                              //
//                                                                      //
//  It is a base class to create a "wrapper" class                      //
//  holding the plain C-structure array                                 //
//  (1 element of the structure per element)                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifdef __CINT__
#pragma Ccomment on
#endif

#include "Ttypes.h"
#include "TDataSet.h"
#include "TArray.h"
#include "tableDescriptor.h"
#include "TCut.h"


#ifndef __CINT__
#  include <string.h>
#include <fstream.h>
#include <assert.h>
#endif

enum ETableBits {
    kIsNotOwn         = BIT(23)   // if the TTable wrapper doesn't own the STAF table                                 // As result of the Update() method for example
};
class G__DataMemberInfo;
class TTableDescriptor;
class TH1;

class TTable : public TDataSet, public TArray {
   friend class TDataSet;
   friend class St_XDFFile;
private:
   Long_t         fSize;       // Size of the one element (row) of the table

protected:
   Char_t    *fTable;       // Array of (fN*fSize) longs

   void       CopyStruct(Char_t *dest, const Char_t *src);
   Char_t    *Create();
   virtual    void       Clear(Option_t *opt="");
   virtual    void       Delete(Option_t *opt="");
//   virtual    TClass    *GetRowClass(const Char_t *rowName)  const ;
   virtual Bool_t  EntryLoop(const Char_t *exprFileName,Int_t &action, TObject *obj, Int_t nentries=1000000000, Int_t firstentry=0, Option_t *option="");
   Int_t      SetfN(Long_t len);
   void       SetTablePointer(void *table);
   void       SetUsedRows(Int_t n);
   void       SetType(const Text_t *const type);
   void       StreamerHeader(TBuffer &b,Version_t version=3);
   int        PointerToPointer(G__DataMemberInfo &m);
   void       StreamerTable(TBuffer &b,Version_t version=3);
   virtual TTableDescriptor *GetDescriptorPointer() const;
   virtual void  SetDescriptorPointer(TTableDescriptor *list) const ;

   Long_t     fMaxIndex;   // The used capacity of this array

   void      ReAlloc(Int_t newsize);

public:

   enum EColumnType {kNAN, kFloat, kInt, kLong, kShort, kDouble, kUInt
                          ,kULong, kUShort, kUChar, kChar };

   TTable(const Text_t *name=0, Int_t size=0);
   TTable(const Text_t *name, Int_t n,Int_t size);
   TTable(const Text_t *name, Int_t n, Char_t *array,Int_t size);
   TTable(const Text_t *name, const Text_t *type, Int_t n, Char_t *array, Int_t size);
   TTable(const TTable &table);
   TTable    &operator=(const TTable &rhs);
   virtual    ~TTable();

   virtual     void       Adopt(Int_t n, void *array);
   virtual     Int_t      AddAt(const void *c);
   virtual     void       AddAt(const void *c, Int_t i);
   virtual     void       AsString(void *buf, const char *name, Int_t width=0) const;
              const void *At(Int_t i) const;
   virtual     void       Browse(TBrowser *b);
   virtual     void       CopySet(TTable &array);
               Int_t      CopyRows(const TTable *srcTable, Int_t srcRow=0, Int_t dstRow=0, Int_t nRows=0, Bool_t expand=kFALSE);
   virtual     void       Draw(Option_t *opt);
   virtual     TH1       *Draw(TCut varexp, TCut selection, Option_t *option=""
                         ,Int_t nentries=1000000000, Int_t firstentry=0);
   virtual     TH1       *Draw(const Text_t *varexp, const Text_t *selection, Option_t *option=""
                              ,Int_t nentries=1000000000, Int_t firstentry=0); // *MENU*
               void      *GetArray()     const ;
   virtual     TClass    *GetRowClass()  const ;
   virtual     Long_t     GetNRows()     const;
   virtual     Long_t     GetRowSize()   const;
   virtual     Long_t     GetTableSize() const;
   virtual     TTableDescriptor *GetTableDescriptors() const;
   virtual     TTableDescriptor *GetRowDescriptors()   const;
   virtual     const Char_t *GetType()   const;
   virtual     void       Fit(const Text_t *formula ,const Text_t *varexp, const Text_t *selection="",Option_t *option="" ,Option_t *goption=""
                              ,Int_t nentries=1000000000, Int_t firstentry=0); // *MENU*

   virtual     Long_t     HasData() const { return 1; }
   virtual     Bool_t     IsFolder() const;
   virtual     void       ls(Option_t *option="");
   virtual     void       ls(Int_t deep);
               Int_t      NaN();
   static      TTable    *New(const Char_t *name, const Char_t *type, void *array, UInt_t size);
   virtual     Char_t    *MakeExpression(const Char_t *expressions[],Int_t nExpressions);
   virtual     Char_t    *Print(Char_t *buf,Int_t n) const ;
   virtual     void       Print(Option_t *opt="");
   virtual  const Char_t *Print(Int_t row, Int_t rownumber=10,
                                const Char_t *colfirst="", const Char_t *collast="") const; // *MENU*
   virtual  const Char_t *PrintHeader() const; // *MENU*
   virtual     void       Project(const Text_t *hname, const Text_t *varexp, const Text_t *selection="", Option_t *option=""
                                 ,Int_t nentries=1000000000, Int_t firstentry=0);

   virtual    Int_t      Purge(Option_t *opt="");

               void      *ReAllocate(Int_t newsize);
               void      *ReAllocate();
   virtual     void       SavePrimitive(ofstream &out, Option_t *option);
   virtual     void       Set(Int_t n);
   virtual     void       Set(Int_t n, Char_t *array);
   virtual     void       SetNRows(Int_t n);
   virtual     void       Reset(Int_t c=0);
   virtual     void       Update();
   virtual     void       Update(TDataSet *set,UInt_t opt=0);
               void      *operator[](Int_t i);
               const void *operator[](Int_t i) const;


 //  ----   Table descriptor service   ------

   virtual   Int_t        GetColumnIndex(const Char_t *columnName) const;
   virtual  const Char_t *GetColumnName(Int_t columnIndex)      const;
   virtual   const UInt_t *GetIndexArray(Int_t columnIndex)     const;
   virtual   UInt_t       GetNumberOfColumns()                  const;
   virtual   UInt_t       GetOffset(Int_t columnIndex)          const;
   virtual   Int_t        GetOffset(const Char_t *columnName=0) const;
   virtual   UInt_t       GetColumnSize(Int_t columnIndex)      const;
   virtual   Int_t        GetColumnSize(const Char_t *columnName=0) const;
   virtual   UInt_t       GetTypeSize(Int_t columnIndex)        const;
   virtual   Int_t        GetTypeSize(const Char_t *columnName=0) const ;
   virtual   UInt_t       GetDimensions(Int_t columnIndex)      const;
   virtual   Int_t        GetDimensions(const Char_t *columnName=0) const ;
   virtual   EColumnType  GetColumnType(Int_t columnIndex)      const;
   virtual   EColumnType  GetColumnType(const Char_t *columnName=0) const;

   ClassDef(TTable,3)  // Array of the C structures
};

inline  void  *TTable::GetArray() const { return (void *)fTable;}

inline  void   TTable::Print(Option_t *) { Print((Char_t *)0,Int_t(0)); }

inline  void   TTable::SetUsedRows(Int_t n) { fMaxIndex = n;}
inline  void   TTable::SetNRows(Int_t n) {SetUsedRows(n);}
//   ULong_t   &operator(){ return GetTable();}


inline void *TTable::operator[](Int_t i)
{
   if (!BoundsOk("TTable::operator[]", i))
      i = 0;
    return (void *)(fTable+i*fSize);
}

inline const void *TTable::operator[](Int_t i) const
{
   if (!BoundsOk("TTable::operator[]", i))
      i = 0;
    return (const void *)(fTable+i*fSize);
}

inline void TTable::Draw(Option_t *opt)
{ Draw(opt, "", "", 1000000000, 0); }

#include "TTableDescriptor.h"

#endif
