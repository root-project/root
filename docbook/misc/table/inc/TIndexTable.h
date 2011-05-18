// @(#)root/table:$Id$
// Author: Valery Fine(fine@bnl.gov)   01/03/2001

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2001 [BNL] Brookhaven National Laboratory.              *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TIndexTable
#define ROOT_TIndexTable

#ifndef ROOT_TTable
#include "TTable.h"
#endif

//////////////////////////////////////////////////////
//
// Class TIndexTable
// Iterator of the table with extra index array
//
//////////////////////////////////////////////////////


class TIndexTable : public TTable {
protected:
   const TTable *fRefTable;
public:
   class iterator {
   protected:
      const TTable *fTable;
      const int   *fCurrentRow;
      iterator(): fTable(0), fCurrentRow(0) {}
   public:
      iterator(const TTable &t, const int &rowPtr): fTable(&t), fCurrentRow(&rowPtr){}
      iterator(const TTable &t): fTable(&t),fCurrentRow(0){}
      iterator(const iterator& iter) : fTable(iter.fTable), fCurrentRow(iter.fCurrentRow){}
      iterator &operator=(const iterator& iter) {fTable = iter.fTable; fCurrentRow = iter.fCurrentRow; return *this;}
      iterator &operator++()    { if (fCurrentRow) ++fCurrentRow; return *this;}
      void     operator++(int)  { if (fCurrentRow) fCurrentRow++;}
      iterator &operator--()    { if (fCurrentRow) --fCurrentRow; return *this;}
      void     operator--(int) { if (fCurrentRow) fCurrentRow--;}
      iterator &operator+(Int_t idx) { if (fCurrentRow) fCurrentRow+=idx; return *this;}
      iterator &operator-(Int_t idx) { if (fCurrentRow) fCurrentRow-=idx; return *this;}
      Int_t operator-(const iterator &it) const { return fCurrentRow-it.fCurrentRow; }
      void *operator *(){ return (void *)(fTable?((char *)fTable->GetArray())+(*fCurrentRow)*(fTable->GetRowSize()):0);}
      operator int()  { return *fCurrentRow;}
      Bool_t operator==(const iterator &t) const { return (fCurrentRow == t.fCurrentRow); }
      Bool_t operator!=(const iterator &t) const { return !operator==(t); }
   };
   TIndexTable(const TTable *table);
   TIndexTable(const TIndexTable &indx): TTable(indx){}
   int  *GetTable(Int_t i=0);
   Bool_t  IsValid() const;
   void    push_back(Long_t next);

   const TTable *Table() const;
   iterator begin()        { return ((const TIndexTable *)this)->begin();}
   iterator begin() const  { return GetNRows() ? iterator(*Table(),*GetTable(0)):end();}
   iterator end()   { return ((const TIndexTable *)this)->end(); }
   iterator end()   const  {Long_t i = GetNRows(); return i? iterator(*Table(), *GetTable(i)):iterator(*this);}

protected:
   static TTableDescriptor *CreateDescriptor();

// define ClassDefTable(TIndexTable,int)
protected:
   static TTableDescriptor *fgColDescriptors;
   virtual TTableDescriptor *GetDescriptorPointer() const;
   virtual void SetDescriptorPointer(TTableDescriptor *list);
public:
   TIndexTable() : TTable("TIndexTable",sizeof(int))    {SetType("int");}
   TIndexTable(const char *name) : TTable(name,sizeof(int)) {SetType("int");}
   TIndexTable(Int_t n) : TTable("TIndexTable",n,sizeof(int)) {SetType("int");}
   TIndexTable(const char *name,Int_t n) : TTable(name,n,sizeof(int)) {SetType("int");}
   virtual ~TIndexTable() {}
   const int *GetTable(Int_t i=0) const;
   int &operator[](Int_t i){ assert(i>=0 && i < GetNRows()); return *GetTable(i); }
   const int &operator[](Int_t i) const { assert(i>=0 && i < GetNRows()); return *((const int *)(GetTable(i))); }
   ClassDef(TIndexTable,4) // "Index" array for TTable object
};

//___________________________________________________________________________________________________________
inline  int  *TIndexTable::GetTable(Int_t i) { return ((int *)GetArray())+i;}
//___________________________________________________________________________________________________________
inline  const int *TIndexTable::GetTable(Int_t i) const { return ((int *)GetArray())+i;}
//___________________________________________________________________________________________________________
inline  Bool_t TIndexTable::IsValid() const
{
   // Check whether all "map" values do belong the table
   const TTable *cont= Table();
   if (!cont) return kFALSE;

   iterator i      = begin();
   iterator finish = end();
   Int_t totalSize         = cont->GetNRows();

   for (; i != finish; i++) {
      int th = i;
      if (  th == -1 || (0 <= th && th < totalSize) ) continue;
      return kFALSE;
   }
   return kTRUE;
}
//___________________________________________________________________________________________________________
inline void TIndexTable::push_back(Long_t next){ AddAt(&next); }

#endif
