// @(#)root/table:$Id$
// Author: Valery Fine(fine@bnl.gov)   30/06/2001

#ifndef ROOT_TGenericTable
#define ROOT_TGenericTable

#include "TTable.h"

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGenericTable                                                       //
//                                                                      //
//  This is the class to represent the array of C-struct                //
//  defined at run-time                                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
class TGenericTable : public TTable {
protected:
   TTableDescriptor *fColDescriptors;
   virtual TTableDescriptor *GetDescriptorPointer() const { return fColDescriptors;}
   virtual void SetDescriptorPointer(TTableDescriptor *list) { fColDescriptors = list;}
   void SetGenericType(){ TTable::SetType(GetDescriptorPointer()->GetName()); }

public:
   class iterator {
   protected:
      UInt_t  fRowSize;
      char   *fCurrentRow;
      iterator():  fRowSize(0), fCurrentRow(0) {}
   public:
      iterator(UInt_t size, char &rowPtr): fRowSize(size), fCurrentRow(&rowPtr){}
      iterator(const TTable &t, char &rowPtr): fRowSize(t.GetRowSize()), fCurrentRow(&rowPtr){}
      iterator(const TTable &t): fRowSize(t.GetRowSize()), fCurrentRow(0){}
      iterator(const iterator& iter) : fRowSize (iter.fRowSize), fCurrentRow(iter.fCurrentRow){}
      iterator &operator=(const iterator& iter) { fRowSize = iter.fRowSize; fCurrentRow = iter.fCurrentRow; return *this;}
      iterator &operator++()    { if (fCurrentRow) fCurrentRow+=fRowSize; return *this;}
      void     operator++(int) { if (fCurrentRow) fCurrentRow+=fRowSize;}
      iterator &operator--()    { if (fCurrentRow) fCurrentRow-=fRowSize; return *this;}
      void      operator--(int) { if (fCurrentRow) fCurrentRow-=fRowSize;}
      iterator &operator+(Int_t idx) { if (fCurrentRow) fCurrentRow+=idx*fRowSize; return *this;}
      iterator &operator-(Int_t idx) { if (fCurrentRow) fCurrentRow-=idx*fRowSize; return *this;}
      Int_t operator-(const iterator &it) const { return (fCurrentRow-it.fCurrentRow)/fRowSize; }
             char *operator *(){ return fCurrentRow;}
             Bool_t operator==(const iterator &t) const { return (fCurrentRow == t.fCurrentRow); }
      Bool_t operator!=(const iterator &t) const { return !operator==(t); }
   };
   TGenericTable() : TTable("TGenericTable",-1), fColDescriptors(0) {SetType("generic");}

   // Create TGenericTable by C structure name provided
   TGenericTable(const char *structName, const char *name);
   TGenericTable(const char *structName, Int_t n);
   TGenericTable(const char *structName, const char *name,Int_t n);

   // Create TGenericTable by TTableDescriptor pointer
   TGenericTable(const TTableDescriptor &dsc, const char *name);
   TGenericTable(const TTableDescriptor &dsc, Int_t n);
   TGenericTable(const TTableDescriptor &dsc,const char *name,Int_t n);

   virtual ~TGenericTable();

   char  *GetTable(Int_t i=0)   const { return ((char *)GetArray())+i*GetRowSize();}
   TTableDescriptor  *GetTableDescriptors() const { return GetDescriptorPointer();}
   TTableDescriptor  *GetRowDescriptors()   const { return GetDescriptorPointer();}
   char &operator[](Int_t i){ assert(i>=0 && i < GetNRows()); return *GetTable(i); }
   const char &operator[](Int_t i) const { assert(i>=0 && i < GetNRows()); return *((const char *)(GetTable(i))); }
   iterator begin()        { return ((const TGenericTable *)this)->begin();}
   iterator begin() const  {                      return GetNRows() ? iterator(*this, *GetTable(0)):end();}
   iterator end()   { return ((const TGenericTable *)this)->end(); }
   iterator end()   const  {Long_t i = GetNRows(); return i? iterator(*this, *GetTable(i)):iterator(*this);}
   ClassDef(TGenericTable,4) // Generic array of C-structure (a'la STL vector)
};

#endif
