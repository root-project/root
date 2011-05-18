// @(#)root/table:$Id$
// Author: Valery Fine(fine@mail.cern.ch)   03/07/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

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
#include "tableDescriptor.h"
#ifndef ROOT_TCut
# include "TCut.h"
#endif

# ifndef ROOT_Riosfwd
#  include "Riosfwd.h"
# endif

#ifndef __CINT__
#  include <string.h>
#  include <assert.h>
#endif

#include <vector>

class TTableDescriptor;
class TH1;
class TTableMap;
typedef TTableMap*  Ptr_t;

class TTable : public TDataSet {
   friend class TDataSet;
   friend class St_XDFFile;
protected:
   Long_t     fSize;       // Size of the one element (row) of the table

protected:

   Int_t      fN;           //Number of array elements
   Char_t    *fTable;       // Array of (fN*fSize) longs
   Long_t     fMaxIndex;    // The used capacity of this array

   Bool_t    BoundsOk(const char *where, Int_t at) const;
   Bool_t    OutOfBoundsError(const char *where, Int_t i) const;

   void       CopyStruct(Char_t *dest, const Char_t *src);
   Char_t    *Create();
   virtual    void       Clear(Option_t *opt="");
   virtual    void       Delete(Option_t *opt="");
   virtual Bool_t  EntryLoop(const Char_t *exprFileName,Int_t &action, TObject *obj, Int_t nentries=1000000000, Int_t firstentry=0, Option_t *option="");
   Int_t      SetfN(Long_t len);
   void       SetTablePointer(void *table);
   void       SetUsedRows(Int_t n);
   virtual void SetType(const char *const type);
   void       StreamerHeader(TBuffer &b,Version_t version=3);
   void       StreamerTable(TBuffer &b,Version_t version=3);
   virtual TTableDescriptor *GetDescriptorPointer() const;
   virtual void  SetDescriptorPointer(TTableDescriptor *list);

   void       ReAlloc(Int_t newsize);
   static const char *TableDictionary(const char *className,const char *structName,TTableDescriptor *&ColDescriptors);

public:

   enum EColumnType {kNAN, kFloat, kInt, kLong, kShort, kDouble, kUInt
                     ,kULong, kUShort, kUChar, kChar, kPtr, kBool
                     ,kEndColumnType };
   enum ETableBits {
      kIsNotOwn      = BIT(23)   // if the TTable wrapper doesn't own the STAF table
                                 // As result of the Update() method for example
   };
   static const char *fgTypeName[kEndColumnType];
   TTable(const char *name=0, Int_t size=0);
   TTable(const char *name, Int_t n,Int_t size);
   TTable(const char *name, Int_t n, Char_t *array,Int_t size);
   TTable(const char *name, const char *type, Int_t n, Char_t *array, Int_t size);
   TTable(const TTable &table);
   TTable    &operator=(const TTable &rhs);
   virtual    ~TTable();

   virtual     void       Adopt(Int_t n, void *array);
   virtual     Int_t      AddAt(const void *c);
   virtual     void       AddAt(const void *c, Int_t i);
   virtual     void       AddAt(TDataSet *dataset,Int_t idx=0);
   virtual     Long_t     AppendRows(const void *row, UInt_t nRows);
   virtual     void       AsString(void *buf, EColumnType type, Int_t width, ostream &out) const;
              const void *At(Int_t i) const;
   virtual     void       Browse(TBrowser *b);
   virtual     void       CopySet(TTable &array);
               Int_t      CopyRows(const TTable *srcTable,Long_t srcRow=0, Long_t dstRow=0, Long_t nRows=0, Bool_t expand=kFALSE);
   virtual     void       DeleteRows(Long_t indx,UInt_t nRows=1);
   virtual     void       Draw(Option_t *opt);
   virtual     TH1       *Draw(TCut varexp, TCut selection, Option_t *option=""
                         ,Int_t nentries=1000000000, Int_t firstentry=0);
   virtual     TH1       *Draw(const char *varexp, const char *selection, Option_t *option=""
                              ,Int_t nentries=1000000000, Int_t firstentry=0); // *MENU*
               void      *GetArray()     const ;
   virtual     TClass    *GetRowClass()  const ;
               Int_t      GetSize() const { return fN; }
   virtual     Long_t     GetNRows()     const;
   virtual     Long_t     GetRowSize()   const;
   virtual     Long_t     GetTableSize() const;
   virtual     TTableDescriptor *GetTableDescriptors() const;
   virtual     TTableDescriptor *GetRowDescriptors()   const;
   virtual     const Char_t *GetType()   const;
   virtual     void       Fit(const char *formula ,const char *varexp, const char *selection="",Option_t *option="" ,Option_t *goption=""
                              ,Int_t nentries=1000000000, Int_t firstentry=0); // *MENU*

   virtual     Long_t     HasData() const { return 1; }
   virtual     Long_t     InsertRows(const void *rows, Long_t indx, UInt_t nRows=1);
   virtual     Bool_t     IsFolder() const;
               Int_t      NaN();
   static      TTable    *New(const Char_t *name, const Char_t *type, void *array, UInt_t size);
   virtual     Char_t    *MakeExpression(const Char_t *expressions[],Int_t nExpressions);
   virtual     Char_t    *Print(Char_t *buf,Int_t n) const ;
   virtual     void       Print(Option_t *opt="") const;
   virtual  const Char_t *Print(Int_t row, Int_t rownumber=10,
                                const Char_t *colfirst="", const Char_t *collast="") const; // *MENU*
   virtual     void       PrintContents(Option_t *opt="") const;
   virtual  const Char_t *PrintHeader() const; // *MENU*
   virtual     void       Project(const char *hname, const char *varexp, const char *selection="", Option_t *option=""
                                 ,Int_t nentries=1000000000, Int_t firstentry=0);

   virtual    Int_t       Purge(Option_t *opt="");

               void      *ReAllocate(Int_t newsize);
               void      *ReAllocate();
   virtual     void       SavePrimitive(ostream &out, Option_t *option = "");
   virtual     void       Set(Int_t n);
   virtual     void       Set(Int_t n, Char_t *array);
   virtual     void       SetNRows(Int_t n);
   virtual     void       Reset(Int_t c=0);
   virtual     void       ResetMap(Bool_t wipe=kTRUE);
   virtual     void       Update();
   virtual     void       Update(TDataSet *set,UInt_t opt=0);
               void      *operator[](Int_t i);
               const void *operator[](Int_t i) const;


 //  ----   Table descriptor service   ------

   virtual   Int_t        GetColumnIndex(const Char_t *columnName) const;
   virtual  const Char_t *GetColumnName(Int_t columnIndex)      const;
   virtual  const UInt_t *GetIndexArray(Int_t columnIndex)      const;
   virtual  UInt_t        GetNumberOfColumns()                  const;
   virtual  UInt_t        GetOffset(Int_t columnIndex)          const;
   virtual  Int_t         GetOffset(const Char_t *columnName=0) const;
   virtual  UInt_t        GetColumnSize(Int_t columnIndex)      const;
   virtual  Int_t         GetColumnSize(const Char_t *columnName=0) const;
   virtual  UInt_t        GetTypeSize(Int_t columnIndex)        const;
   virtual  Int_t         GetTypeSize(const Char_t *columnName=0) const ;
   virtual  UInt_t        GetDimensions(Int_t columnIndex)      const;
   virtual  Int_t         GetDimensions(const Char_t *columnName=0) const ;
   virtual  EColumnType   GetColumnType(Int_t columnIndex)      const;
   virtual  EColumnType   GetColumnType(const Char_t *columnName=0) const;
   virtual  const Char_t *GetColumnComment(Int_t columnIndex) const;

   static const char *GetTypeName(EColumnType type);
   static EColumnType GetTypeId(const char *typeName);

   // Table index iterator:
   class iterator {
      public:
         typedef std::vector<Long_t>::iterator vec_iterator;
         typedef std::vector<Long_t>::const_iterator vec_const_iterator;
      private:
         Long_t        fRowSize;
         const TTable *fThisTable;
         vec_iterator  fCurrentRow;
      public:
         iterator(): fRowSize(0), fThisTable(0) {;}
         iterator(const TTable &table, vec_iterator &arowPtr) :
            fRowSize(table.GetRowSize()), fThisTable(&table), fCurrentRow(arowPtr) {;}
         iterator(const TTable &table, vec_const_iterator &arowPtr) :
            fRowSize(table.GetRowSize()), fThisTable(&table),
            fCurrentRow(*(std::vector<Long_t>::iterator *)(void *)&arowPtr) {;}
            //fCurrentRow(* const_cast<vector<Long_t>::iterator *>(&arowPtr) ) {;}
         iterator(const iterator& iter) : fRowSize (iter.fRowSize), fThisTable(iter.fThisTable),fCurrentRow(iter.fCurrentRow){}
         void operator=(const iterator& iter)   { fRowSize = iter.fRowSize; fThisTable = iter.fThisTable; fCurrentRow=iter.fCurrentRow; }
         void operator++()    { ++fCurrentRow;   }
         void operator++(int) {   fCurrentRow++; }
         void operator--()    { --fCurrentRow;   }
         void operator--(int) {   fCurrentRow--; }
         iterator operator+(Int_t idx)   { std::vector<Long_t>::iterator addition   = fCurrentRow+idx; return  iterator(*fThisTable,addition); }
         iterator operator-(Int_t idx)   { std::vector<Long_t>::iterator subtraction = fCurrentRow-idx; return  iterator(*fThisTable,subtraction); }
         void operator+=(Int_t idx)  {  fCurrentRow+=idx; }
         void operator-=(Int_t idx)  {  fCurrentRow-=idx; }
         void *rowPtr() const { return  (void *)(((const char *)fThisTable->GetArray()) + (*fCurrentRow)*fRowSize ); }
         operator void *() const { return rowPtr(); }
         Int_t operator-(const iterator &it) const { return (*fCurrentRow)-(*(it.fCurrentRow)); }
         Long_t operator *() const { return  *fCurrentRow; }
         Bool_t operator==(const iterator &t) const { return  ( (fCurrentRow == t.fCurrentRow) && (fThisTable == t.fThisTable) ); }
         Bool_t operator!=(const iterator &t) const { return !operator==(t); }

         const TTable &Table()   const { return *fThisTable;}
         const Long_t &RowSize() const { return fRowSize;}
#ifndef __CINT__
         const std::vector<Long_t>::iterator &Row() const { return fCurrentRow;}
#endif
   };

#ifndef __CINT__
   //  pointer iterator
   // This create an iterator to iterate over all table column of the
   // type provided.
   // For example" piterator(table,kPtr) is to iterate over
   // all cells of (TTableMap *) type

   class piterator {
      private:
         std::vector<ULong_t>  fPtrs;
         UInt_t                fCurrentRowIndex;
         UInt_t                fCurrentColIndex;
         UInt_t                fRowSize;
         const Char_t         *fCurrentRowPtr;
         void                **fCurrentColPtr;

      protected:
         void **column() {return  fCurrentColPtr = (void **)(fCurrentRowPtr + fPtrs[fCurrentColIndex]);}

      public:
         piterator(const TTable *t=0,EColumnType type=kPtr);
         piterator(const piterator& iter);
         void operator=(const piterator& iter);

         void operator++();
         void operator++(int);
         void operator--();
         void operator--(int);

//        operator const char *() const;
         void **operator *();

         Bool_t operator==(const piterator &t) const;
         Bool_t operator!=(const piterator &t) const;

         UInt_t Row()    const;
         UInt_t Column() const;

         void  MakeEnd(UInt_t lastRowIndex);
   }; // class iterator over pointers

   piterator pbegin(){ return piterator(this); }
   piterator pend(){ piterator pLast(this); pLast.MakeEnd(GetNRows()); return pLast; }
#endif
   static const char *TableDictionary() { return 0; };
   ClassDef(TTable,4)  // vector of the C structures
};

//________________________________________________________________________
inline void  TTable::AddAt(TDataSet *dataset,Int_t idx)
{ TDataSet::AddAt(dataset,idx); }

//________________________________________________________________________
inline Bool_t TTable::BoundsOk(const char *where, Int_t at) const
{
   return (at < 0 || at >= fN)
                  ? OutOfBoundsError(where, at)
                  : kTRUE;
}

//________________________________________________________________________
inline  void  *TTable::GetArray() const { return (void *)fTable;}

//________________________________________________________________________
inline  void   TTable::Print(Option_t *) const { Print((Char_t *)0,Int_t(0)); }

//________________________________________________________________________
inline  void   TTable::SetUsedRows(Int_t n) { fMaxIndex = n;}
//________________________________________________________________________
inline  void   TTable::SetNRows(Int_t n) {SetUsedRows(n);}
//   ULong_t   &operator(){ return GetTable();}

//________________________________________________________________________
inline void *TTable::operator[](Int_t i)
{
   if (!BoundsOk("TTable::operator[]", i))
      i = 0;
    return (void *)(fTable+i*fSize);
}

//________________________________________________________________________
inline const void *TTable::operator[](Int_t i) const
{
   if (!BoundsOk("TTable::operator[]", i))
      i = 0;
    return (const void *)(fTable+i*fSize);
}

//________________________________________________________________________
inline void TTable::Draw(Option_t *opt)
{ Draw(opt, "", "", 1000000000, 0); }

#ifndef __CINT__
    //________________________________________________________________________________________________________________
    inline TTable::piterator::piterator(const piterator& iter) :
           fPtrs (iter.fPtrs),
           fCurrentRowIndex(iter.fCurrentRowIndex),
           fCurrentColIndex(iter.fCurrentColIndex),
           fRowSize(iter.fRowSize),
           fCurrentRowPtr(iter.fCurrentRowPtr),
           fCurrentColPtr(iter.fCurrentColPtr)
    {}
    //________________________________________________________________________________________________________________
    inline void TTable::piterator::operator=(const piterator& iter){
        fPtrs            = iter.fPtrs;
        fCurrentRowIndex = iter.fCurrentRowIndex;
        fCurrentColIndex = iter.fCurrentColIndex;
        fRowSize         = iter.fRowSize;
        fCurrentRowPtr   = iter.fCurrentRowPtr;
        fCurrentColPtr   = iter.fCurrentColPtr;
    }
    //________________________________________________________________________________________________________________
    inline void TTable::piterator::operator++()
    {
       ++fCurrentColIndex;
         if (fCurrentColIndex >= fPtrs.size()) {
           fCurrentColIndex = 0;
         ++fCurrentRowIndex;
           fCurrentRowPtr += fRowSize;
         }
         column();
    }
    //________________________________________________________________________________________________________________
    inline void TTable::piterator::operator++(int) {  operator++(); }
    //________________________________________________________________________________________________________________
    inline void TTable::piterator::operator--()
    {
       if (fCurrentColIndex > 0) {
          fCurrentColIndex--;
          fCurrentColIndex = fPtrs.size()-1;
        --fCurrentRowIndex;
          fCurrentRowPtr -= fRowSize;
       } else {
          fCurrentColIndex--;
       }
       column();
    }
    //________________________________________________________________________________________________________________
    inline void TTable::piterator::operator--(int) {  operator--();  }
    //________________________________________________________________________________________________________________
    // inline TTable::piterator::operator const char *() const { return fCurrentColPtr; }
    //________________________________________________________________________________________________________________
    inline void **TTable::piterator::operator *()            { return fCurrentColPtr; }
    //________________________________________________________________________________________________________________
    inline Bool_t TTable::piterator::operator==(const piterator &t) const {
        return  (
                 (fCurrentRowIndex== t.fCurrentRowIndex)
              && (fCurrentColIndex == t.fCurrentColIndex)
//              && (fCurrentRowPtr == t.fCurrentRowPtr )
//              && (fCurrentColPtr == t.fCurrentColPtr )
                );
    }
    //________________________________________________________________________________________________________________
    inline Bool_t TTable::piterator::operator!=(const piterator &t) const { return !operator==(t); }
    //________________________________________________________________________________________________________________
    inline void  TTable::piterator::MakeEnd(UInt_t lastRowIndex){fCurrentColIndex = 0; fCurrentRowIndex = lastRowIndex;}
    //________________________________________________________________________________________________________________
    inline UInt_t TTable::piterator::Row()    const { return fCurrentRowIndex;}
    //________________________________________________________________________________________________________________
    inline UInt_t TTable::piterator::Column() const { return fCurrentColIndex;}
#endif
#include "TTableDescriptor.h"

#endif
