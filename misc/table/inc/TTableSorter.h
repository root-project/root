// @(#)root/table:$Id$
// Author: Valery Fine   26/01/99  (E-mail: fine@bnl.gov)

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TTableSorter
#define ROOT_TTableSorter

#include "TNamed.h"
#include "TTableDescriptor.h"

////////////////////////////////////////////////////////////////////////////////////////
//
//  TTableSorter  - Is an "observer" class to sort the TTable objects
//                    The class provides an interface to the standard "C/C++"
//
// qsort and bsearch subroutine (for further information see your local C/C++ docs)
// =====     =======
//
//  - This class DOESN'T change / touch the "host" table  itself
//    For any TTable object one can create as many different "sorter"
//    as he/she finds useful for his/her code
//  - Any instance of this class is meaningful as long as the "host" object
//    "TTable" does exist and is not changed
//  - Any attempt to access this TTableSorter after the "host" object deleted
//    causes the program abnormal termination
//  - Any attempt to access this TTableSorter after the "host" object been changed
//    causes an unpredictable result
//  - Any instance (object) of this class is NOT deleted "by automatic" just
//    the "host object "TTable" deleted. It is the responsibility of the user's code
//    keeping TTableSorter and the the "host" TTable objects consistent.
//
////////////////////////////////////////////////////////////////////////////////////////


class table_head_st;

typedef Int_t (*COMPAREMETHOD)(const void **, const void **);
typedef Int_t (*SEARCHMETHOD) (const void *, const void **);

class TTableSorter : public TNamed {
private:
   union {  Char_t   fChar;
   Int_t    fInt;
   Long_t   fLong;
   Float_t  fFloat;
   Double_t fDouble;
   } fValue;

protected:
   //   enum EColumnType {kNAN, kFloat, kInt, kLong, kShort, kDouble, kUInt
   //                           ,kULong, kUShort, kUChar, kChar };
   void    **fSortIndex;    // Array of pointers to columns of the sorted table
   Int_t     fLastFound;    // The index of the last found index within fSortIndex
   Int_t     fFirstRow;     // first row of the table to be sorted
   Int_t     fNumberOfRows; // number of rows of the table to be sorted
   TString   fColName;      //
   Int_t     fColOffset;    //
   Int_t     fColSize;      // The size of the selected column in bytes
   Int_t    *fIndexArray;   // "parsed" indecis
   Int_t     fColDimensions;// The number of the dimensions for array (=-1 means it is a "simple" array)
   const Char_t *fsimpleArray;    // Pointer to the "simple" array;
   const TTable *fParentTable;    //!- the back pointer to the sorted table
   SEARCHMETHOD  fSearchMethod;   // Function selected to search values
   COMPAREMETHOD fCompareMethod;  // Function to sort the original array
   TTable::EColumnType  fColType; // data type of the selected column
   Long_t  fParentRowSize;        // To be filled from TTable::GetRowSize() method
   const char *fFirstParentRow;   //! pointer to the internal array of TTable object;

   static int CompareFloat_t     (const void **, const void **);
   static int CompareInt_t       (const void **, const void **);
   static int CompareLong_t      (const void **, const void **);
   static int CompareULong_t     (const void **, const void **);
   static int CompareUInt_t      (const void **, const void **);
   static int CompareShort_t     (const void **, const void **);
   static int CompareDouble_t    (const void **, const void **);
   static int CompareUShort_t    (const void **, const void **);
   static int CompareUChar_t     (const void **, const void **);
   static int CompareChar_t      (const void **, const void **);
   static int CompareBool_t      (const void **, const void **);

   Int_t  BSearch(const void *value) const;

   Int_t BSearch(Float_t  value ) const;
   Int_t BSearch(Int_t    value ) const;
   Int_t BSearch(ULong_t  value ) const;
   Int_t BSearch(Long_t   value ) const;
   Int_t BSearch(UInt_t   value ) const;
   Int_t BSearch(Short_t  value ) const;
   Int_t BSearch(Double_t value ) const;
   Int_t BSearch(UShort_t value ) const;
   Int_t BSearch(UChar_t  value ) const;
   Int_t BSearch(Char_t   value ) const;
   Int_t BSearch(Bool_t   value ) const;

   //  Int_t BSearch(const Char_t *value) const;
   //  Int_t BSearch(TString &value)     ;

   Bool_t FillIndexArray();
   Long_t GetRowSize();
   void   QSort();
   void   LearnTable();

   static int SearchFloat_t     (const void *, const void **);
   static int SearchInt_t       (const void *, const void **);
   static int SearchULong_t     (const void *, const void **);
   static int SearchLong_t      (const void *, const void **);
   static int SearchUInt_t      (const void *, const void **);
   static int SearchShort_t     (const void *, const void **);
   static int SearchDouble_t    (const void *, const void **);
   static int SearchUShort_t    (const void *, const void **);
   static int SearchUChar_t     (const void *, const void **);
   static int SearchChar_t      (const void *, const void **);
   static int SearchBool_t      (const void *, const void **);

   Int_t SelectSearch(Float_t  value ) const;
   Int_t SelectSearch(Int_t    value ) const;
   Int_t SelectSearch(ULong_t  value ) const;
   Int_t SelectSearch(Long_t   value ) const;
   Int_t SelectSearch(UInt_t   value ) const;
   Int_t SelectSearch(Short_t  value ) const;
   Int_t SelectSearch(Double_t value ) const;
   Int_t SelectSearch(UShort_t value ) const;
   Int_t SelectSearch(UChar_t  value ) const;
   Int_t SelectSearch(Char_t   value ) const;
   Int_t SelectSearch(Bool_t   value ) const;

   void  SetSearchMethod();
   void  SetSimpleArray(Int_t arraySize, Int_t firstRow,Int_t numberRows);
   void  BuildSorter(TString &colName, Int_t firstRow, Int_t numberRows);
   const char *At(Int_t i) const;

public:
   TTableSorter();
   TTableSorter(const TTable &table, TString &colName, Int_t firstRow=0,Int_t numbeRows=0);
   TTableSorter(const TTable *table, TString &colName, Int_t firstRow=0,Int_t numbeRows=0);

   TTableSorter(const TTable &table, SEARCHMETHOD search, COMPAREMETHOD compare, Int_t firstRow=0,Int_t numbeRows=0);
   TTableSorter(const TTable *table, SEARCHMETHOD search, COMPAREMETHOD compare, Int_t firstRow=0,Int_t numbeRows=0);

   TTableSorter(const Float_t  *simpleArray, Int_t arraySize, Int_t firstRow=0,Int_t numberRows=0);
   TTableSorter(const Double_t *simpleArray, Int_t arraySize, Int_t firstRow=0,Int_t numberRows=0);
   TTableSorter(const Long_t   *simpleArray, Int_t arraySize, Int_t firstRow=0,Int_t numberRows=0);
   virtual ~TTableSorter();

   virtual Int_t CountKey(const void *key, Int_t firstIndx=0,Bool_t bSearch=kTRUE,Int_t *firstRow=0) const;
   virtual Int_t CountKeys() const;
   virtual Int_t FindFirstKey(const void *key) const;

   Int_t BinarySearch(Float_t  value ) const;
   Int_t BinarySearch(Int_t    value ) const;
   Int_t BinarySearch(ULong_t  value ) const;
   Int_t BinarySearch(Long_t   value ) const;
   Int_t BinarySearch(UInt_t   value ) const;
   Int_t BinarySearch(Short_t  value ) const;
   Int_t BinarySearch(Double_t value ) const;
   Int_t BinarySearch(UShort_t value ) const;
   Int_t BinarySearch(UChar_t  value ) const;
   Int_t BinarySearch(Char_t   value ) const;
   Int_t BinarySearch(Bool_t   value ) const;

   virtual const char   *GetColumnName() const { return fColName.Data();}
   Int_t     GetIndex(UInt_t sortedIndex) const;
   virtual const void     *GetKeyAddress(Int_t indx) { return (fSortIndex && indx >= 0) ?fSortIndex[indx]:(void *)(-1);}
   virtual       Int_t     GetLastFound()  const { return fLastFound; }
   virtual const char   *GetTableName()  const;
   virtual const char   *GetTableTitle() const;
   virtual const char   *GetTableType()  const;
   virtual       TTable   *GetTable()      const;
   virtual       Int_t     GetNRows()      const { return fNumberOfRows;}
   virtual       Int_t     GetFirstRow()   const { return fFirstRow;}

   Int_t operator[](Int_t value)    const;
   Int_t operator[](Long_t value)   const;
   Int_t operator[](Double_t value) const;
   Int_t operator[](void *value)    const;
   //    Int_t operator[](const Char_t *value) const;
   //    Int_t operator[](TString &value) const { return BSearch(value); }  // to be implemented

   Int_t operator()(Float_t value);
   Int_t operator()(Int_t value);
   Int_t operator()(Long_t value);
   Int_t operator()(Double_t value);
   //    Int_t operator()(const Char_t *value) { return BinarySearch(*value); } // to be implemented
   //    Int_t operator()(TString &value)    { return *this(value.Data());  }   // to be implemented

   ClassDef(TTableSorter,0) // Is an "observer" class to sort the TTable objects
};

inline const char *TTableSorter::At(Int_t i) const {return fFirstParentRow + i*fParentRowSize;}
inline Long_t TTableSorter::GetRowSize() { return fParentRowSize; }

inline Int_t TTableSorter::operator[](Int_t value)    const { return BSearch(value); }
inline Int_t TTableSorter::operator[](Long_t value)   const { return BSearch(value); }
inline Int_t TTableSorter::operator[](Double_t value) const { return BSearch(value); }
inline Int_t TTableSorter::operator[](void *value)    const { return BSearch(value); }

inline Int_t TTableSorter::operator()(Float_t value)  { return BinarySearch(value); }
inline Int_t TTableSorter::operator()(Int_t value)    { return BinarySearch(value); }
inline Int_t TTableSorter::operator()(Long_t value)   { return BinarySearch(value); }
inline Int_t TTableSorter::operator()(Double_t value) { return BinarySearch(value); }

#endif
