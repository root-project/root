// @(#)root/table:$Id$
// Author: Valery Fine   26/01/99  (E-mail: fine@bnl.gov)

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include <stdlib.h>
#include "TTableSorter.h"
#include "TTable.h"
#include "TClass.h"
#include "TDataMember.h"
#include "TDataType.h"
#include "TMemberInspector.h"
extern "C" {
   typedef Int_t (*CALLQSORT)    (const void *, const void *);
}

/////////////////////////////////////////////////////////////////////////////////////////
//
//  TTableSorter  - Is an "observer" class to sort the TTable objects
//                    The class provides an interface to the standard "C/C++"
//
// qsort and bsearch subroutines (for further information see your local C/C++ docs)
// =====     =======
//
//  - This class DOESN'T change / touch the "host" table  itself
//    For any TTable object one can create as many different "sorter"
//    as one finds useful for one's code
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
// "To do" list
//
//  1. A separate method to provide lexicographical sort if the "sorted" column is a kind of array
//
//  Usage:
//    1. Create an instance of the sorter for the selected column of your table
//
//        new TTableSorter(TTable &table, TString &colName,Int_t firstRow,Int_t numberRows)
//
//        All sort actions are performed within TTableSorter ctor.
//        This means one needs no extra effort to SORT table. "Sorter" contains
//        the "sorted index array" as soon as you create the sorter
//
//        TTableSorter sorter(MyTable,"id",20, 34);
//          - Creates a sorter for MyTable column "id" ordering
//            its 34 rows from 20 row with standard "C" qsort subroutine
//
//    2.  You may use this instance to search any "id" value with operator []
//          to get the table row index as follows:
//
//          Int_t id = 5;
//          Int_t index =  sorter[id]; // Look for the row index with id = 5
//                                     // using the standard "C"  "bsearch" binary search
//                                     // subroutine
//          Int_t index =  sorter(id); // Look for the row index with id "nearest" to 5
//                                     // using the internal "BinarySearch" method
//
//    3. Some useful methods of this class:
//
//        3.1. CountKeys()
//        3.2  CountKey(const void *key, Int_t firstIndx=0,Bool_t bSearch=kTRUE,Int_t *firstRow=0)
//        3.3. FindFirstKey(const void *key)
//        3.4. GetIndex(UInt_t sortedIndex)
//
/////////////////////////////////////////////////////////////////////////////////////////


ClassImp(TTableSorter)

//_____________________________________________________________________________
//TTableSorter::TTableSorter() : fsimpleArray(0),fParentTable(*((const TTable *)0))
TTableSorter::TTableSorter() : fsimpleArray(0),fParentTable(0)
{
   // default ctor for RootCint dictionary
   fLastFound    = -1;
   fFirstRow     =  0;
   fSortIndex    =  0;
   fSearchMethod =  0;
   fNumberOfRows =  0;
   fColType      = TTable::kNAN;
   fsimpleArray    = 0;
   fParentRowSize  = 0;
   fFirstParentRow = 0;
   fCompareMethod  = 0;
   fIndexArray     = 0;
   fColDimensions  = 0;
   fColOffset      = 0;
   fColSize        = 0;
}

//_____________________________________________________________________________
TTableSorter::TTableSorter(const TTable &table, TString &colName,Int_t firstRow
                               ,Int_t numberRows):fsimpleArray(0),fParentTable(&table)
{
  //
  // TTableSorter ctor sorts the input table along its column defined with colName
  //
  //    - colName    - may be followed by the square brackets with integer number inside,
  //                   if that columm is an array (for example "phys[3]").
  //                   NO expression inside of [], only a single integer number allowed !
  //    - firstRow   - the first table row to sort from (=0 by default)
  //    - numberRows - the number of the table rows to sort (=0 by default)
  //                   = 0 means sort all rows from the "firstRow" by the end of table
  //

   fCompareMethod =  0;
   fSearchMethod  =  0;

   BuildSorter(colName, firstRow, numberRows);
}

//_____________________________________________________________________________
TTableSorter::TTableSorter(const TTable *table, TString &colName,Int_t firstRow
                               ,Int_t numberRows):fsimpleArray(0),fParentTable(table)
{
  //
  // TTableSorter ctor sorts the input table along its column defined with colName
  //
  //    - colName    - may be followed by the square brackets with integer number inside,
  //                   if that columm is an array (for example "phys[3]").
  //                   NO expression inside of [], only a single integer number allowed !
  //    - firstRow   - the first table row to sort from (=0 by default)
  //    - numberRows - the number of the table rows to sort (=0 by default)
  //                   = 0 means sort all rows from the "firstRow" by the end of table
  //

   fCompareMethod =  0;
   fSearchMethod  =  0;

   BuildSorter(colName, firstRow, numberRows);
}
//_____________________________________________________________________________
TTableSorter::TTableSorter(const TTable &table, SEARCHMETHOD search,
                           COMPAREMETHOD compare, Int_t firstRow,Int_t numberRows)
                          :fsimpleArray(0),fParentTable(&table)
{
  //
  // TTableSorter ctor sorts the input table according the function "search"
  //
  //    - search     - the function to compare the "key" and the table rows during sorting
  //                   typedef Int_t (*SEARCHMETHOD) (const void *, const void **);
  //
  //    - compare    - the function to compare two table rows during searching
  //                   typedef Int_t (*COMPAREMETHOD)(const void **, const void **);
  //
  //    - firstRow   - the first table row to sort from (=0 by default)
  //    - numberRows - the number of the table rows to sort (=0 by default)
  //                   = 0 means sort all rows from the "firstRow" by the end of table
  //  Note:  This is a base class. If one fears it is not safe
  //  -----  to allow "void *" one may potect the end-user code
  //         providing a derived class with the appropriated type
  //         of the parameters.
  //
   fCompareMethod =  compare;
   fSearchMethod  =  search;
   TString colName = "user's defined";
   BuildSorter(colName, firstRow, numberRows);
}
//_____________________________________________________________________________
TTableSorter::TTableSorter(const TTable *table, SEARCHMETHOD search,
                           COMPAREMETHOD compare, Int_t firstRow,Int_t numberRows)
                          :fsimpleArray(0),fParentTable(table)
{
  //
  // TTableSorter ctor sorts the input table according the function "search"
  //
  //    - search     - the function to compare the "key" and the table rows during sorting
  //                   typedef Int_t (*SEARCHMETHOD) (const void *, const void **);
  //
  //    - compare    - the function to compare two table rows during searching
  //                   typedef Int_t (*COMPAREMETHOD)(const void **, const void **);
  //
  //    - firstRow   - the first table row to sort from (=0 by default)
  //    - numberRows - the number of the table rows to sort (=0 by default)
  //                   = 0 means sort all rows from the "firstRow" by the end of table
  //  Note:  This is a base class. If one fears it is not safe
  //  -----  to allow "void *" one may potect the end-user code
  //         providing a derived class with the appropriated type
  //         of the parameters.
  //

   fCompareMethod =  compare;
   fSearchMethod  =  search;
   TString colName = "user's defined";
   BuildSorter(colName, firstRow, numberRows);
}

//_____________________________________________________________________________
void TTableSorter::BuildSorter(TString &colName, Int_t firstRow, Int_t numberRows)
{
  //
  // BuildSorter backs TTableSorter ctor
  //
  //    - colName    - may be followed by the square brackets with integer number inside,
  //                   if that columm is an array (for example "phys[3]").
  //                   NO expression inside of [], only a single integer number allowed !
  //    - firstRow   - the first table row to sort from (=0 by default)
  //    - numberRows - the number of the table rows to sort (=0 by default)
  //                   = 0 means sort all rows from the "firstRow" by the end of table
  //

   assert(fParentTable!=0);

   fLastFound     = -1;
   fNumberOfRows  =  0;
   fColType       =  TTable::kNAN;
   fsimpleArray   =  0;
   //yf  fCompareMethod =  0;
   fSortIndex     =  0;
   //yf fSearchMethod  =  0;
   fColDimensions =  0;
   fColOffset     =  0;

   // Generate new object name
   TString n = fParentTable->GetName();
   n += ".";
   n += colName;
   SetName(n);

   Char_t *name = (Char_t *) colName.Data();
   if (!(name || strlen(colName.Data()))) { MakeZombie(); delete [] name; return; }
   name = StrDup(colName.Data());

   // check bounds:
   if (firstRow > fParentTable->GetNRows()) { MakeZombie(); delete [] name; return; }
   fFirstRow = firstRow;

   fNumberOfRows = fParentTable->GetNRows()- fFirstRow;
   if (numberRows > 0)  fNumberOfRows = TMath::Min(numberRows,fNumberOfRows);
   fParentRowSize = fParentTable->GetRowSize();
   fFirstParentRow= (const char *)fParentTable->GetArray();

   // Allocate index array
   if (fNumberOfRows <=0 ) { MakeZombie(); delete [] name; return; }
   fSortIndex = new void*[fNumberOfRows];

   // define dimensions if any;
   // count the open "["
   Char_t *br = name - 1;
   while((br = strchr(br+1,'['))) {
      if (!fColDimensions) *br = 0;
      fColDimensions++;
   }

   // Define the column name
   fColName = name;
   delete [] name;

   fIndexArray = 0;
   if (fColDimensions) {
      fIndexArray = new Int_t[fColDimensions];
      memset(fIndexArray,0,fColDimensions*sizeof(Int_t));
      // Define the index
      const char *openBracket  = colName.Data()-1;
      const char *closeBracket = colName.Data()-1;
      for (Int_t i=0; i< fColDimensions; i++) {
         openBracket  = strchr(openBracket+1, '[');
         closeBracket = strchr(closeBracket+1,']');
         if (closeBracket > openBracket)
            fIndexArray[i] = atoi(openBracket+1);
         else {
            Error("TTable ctor", "Wrong parethethis <%s>",colName.Data());
            MakeZombie();
            return;
         }
      }
   }
   if (colName != "user's defined") {
      LearnTable();
      SetSearchMethod();
   }
   if (!FillIndexArray()) QSort();
}

//_____________________________________________________________________________
TTableSorter::TTableSorter(const Float_t *simpleArray, Int_t arraySize, Int_t firstRow
                               ,Int_t numberRows)
                               :fsimpleArray((const Char_t*)simpleArray)
                               ,fParentTable(0)
{
  //
  // TTableSorter ctor sort the input "simpleArray"
  //
  //    - arraySize  - the size of the full array
  //    - firstRow   - the first table row to sort from (=0 by default)
  //    - numberRows - the number of the table rows to sort (=0 by default)
  //                   = 0 means sort all rows from the "firstRow" by the end of table
  //

   fLastFound    = -1;

   SetSimpleArray(arraySize,firstRow,numberRows);
   if (!fsimpleArray) { MakeZombie(); return; }

 //  LearnTable();

   fColName = "Float";
   fColType   = TTable::kFloat;
   fColSize   = sizeof(Float_t);
   fParentRowSize = fColSize;

  // FillIndexArray();

   Float_t *p = ((Float_t *)fsimpleArray) + fFirstRow;
   Bool_t isPreSorted = kTRUE;
   Float_t sample = *p;
   for (Int_t i=0; i < fNumberOfRows;i++,p++) {
      fSortIndex[i-fFirstRow] = p;
      if ( isPreSorted) {
         if (sample > *p) isPreSorted = kFALSE;
         else sample = *p;
      }
   }

   SetSearchMethod();
   if (!isPreSorted) QSort();
}

//_____________________________________________________________________________
TTableSorter::TTableSorter(const Double_t *simpleArray, Int_t arraySize, Int_t firstRow
                               ,Int_t numberRows)
                               :fsimpleArray((const Char_t*)simpleArray)
                               ,fParentTable(0)
{
  //
  // TTableSorter ctor sort the input "simpleArray"
  //
  //    - arraySize  - the size of the full array
  //    - firstRow   - the first table row to sort from (=0 by default)
  //    - numberRows - the number of the table rows to sort (=0 by default)
  //                   = 0 means sort all rows from the "firstRow" by the end of table
  //

   fLastFound    = -1;

   SetSimpleArray(arraySize,firstRow,numberRows);
   if (!fsimpleArray)  {MakeZombie(); return; }

 //  LearnTable();

   fColName = "Double";
   fColType = TTable::kDouble;
   fColSize = sizeof(Double_t);
   fParentRowSize = fColSize;

  // FillIndexArray();

   Double_t *p = ((Double_t *)simpleArray) + fFirstRow;
   Bool_t isPreSorted = kTRUE;
   Double_t sample = *p;
   for (Int_t i=0; i < fNumberOfRows;i++,p++) {
      fSortIndex[i-fFirstRow] = p;
      if ( isPreSorted) {
         if (sample > *p) isPreSorted = kFALSE;
         else sample = *p;
      }
   }

   SetSearchMethod();
   if (!isPreSorted) QSort();
}

//_____________________________________________________________________________
TTableSorter::TTableSorter(const Long_t *simpleArray, Int_t arraySize, Int_t firstRow
                               ,Int_t numberRows)
                               :fsimpleArray((const Char_t*)simpleArray)
                               ,fParentTable(0)
{
  //
  // TTableSorter ctor sort the input "simpleArray"
  //
  //    - arraySize  - the sie of the full array
  //    - firstRow   - the first table row to sort from (=0 by default)
  //    - numberRows - the number of the table rows to sort (=0 by default)
  //                   = 0 means sort all rows from the "firstRow" by the end of table
  //

   fLastFound    = -1;

   SetSimpleArray(arraySize,firstRow,numberRows);
   if (!simpleArray) { MakeZombie(); return; }

 //  LearnTable();

   fColName = "Long";
   fColType = TTable::kLong;
   fColSize = sizeof(Long_t);
   fParentRowSize = fColSize;

  // FillIndexArray();

   Long_t *p = ((Long_t *)simpleArray) + fFirstRow;
   Bool_t isPreSorted = kTRUE;
   Long_t sample = *p;
   for (Int_t i=0; i < fNumberOfRows;i++,p++) {
      fSortIndex[i-fFirstRow] = p;
      if ( isPreSorted) {
         if (sample > *p) isPreSorted = kFALSE;
         else sample = *p;
      }
   }
   SetSearchMethod();
   if (!isPreSorted) QSort();

}

//_____________________________________________________________________________
void TTableSorter::SetSimpleArray(Int_t arraySize, Int_t firstRow,Int_t numberRows)
{
   // Set some common parameteres for the "simple" arrays
   SetName("Array");

   fSortIndex     = 0;
   fSearchMethod  = 0;
   fColDimensions = 0;
   delete [] fIndexArray;
   fIndexArray    = 0;
   fColOffset     = 0;

   // check bounds:
   if (firstRow > arraySize) return;
   fFirstRow = firstRow;

   fNumberOfRows = arraySize - fFirstRow;
   if (numberRows > 0)  fNumberOfRows = TMath::Min(numberRows,fNumberOfRows);

   // Allocate index array
   delete [] fSortIndex;
   if (fNumberOfRows > 0) fSortIndex = new void*[fNumberOfRows];
}

//_____________________________________________________________________________
TTableSorter::~TTableSorter()
{
   //to be documented
   delete [] fSortIndex; fSortIndex = 0; fNumberOfRows=0;
   delete [] fIndexArray;
}

//_____________________________________________________________________________
//______________________________________________________________________________
//*-*-*-*-*-*-*Binary search in an array of n values to locate value*-*-*-*-*-*-*
//*-*          ==================================================
//*-*  If match is found, the function returns the position (index) of the element.
//*-*  If no match found, the function gives the index of the nearest element
//*-*  smaller than key value.
//*-*  Note: The function returns the negative result if the key value
//*-*  ----  is smaller any table value.
//*-*
//*-* This method is based on TMath::BinarySearch
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*

#define BINARYSEARCH(valuetype) Int_t TTableSorter::BinarySearch(valuetype value) const {\
   switch (fColType) {                                \
         case  TTable::kFloat:                        \
           return SelectSearch(Float_t(value));       \
         case  TTable::kInt :                         \
           return SelectSearch(Int_t(value));         \
         case  TTable::kLong :                        \
           return SelectSearch(Long_t(value));        \
         case  TTable::kShort :                       \
           return SelectSearch(Short_t(value));       \
         case  TTable::kDouble :                      \
           return SelectSearch(Double_t(value));      \
         case  TTable::kUInt:                         \
           return SelectSearch(UInt_t(value));        \
         case  TTable::kULong :                       \
           return SelectSearch(ULong_t(value));       \
         case  TTable::kUShort:                       \
           return SelectSearch(UShort_t(value));      \
         case  TTable::kBool:                         \
           return SelectSearch(Bool_t(value));        \
         case  TTable::kUChar:                        \
           return SelectSearch(UChar_t(value));       \
         case  TTable::kChar:                         \
           return SelectSearch(Char_t(value));        \
         default:                                     \
           return -1;                                 \
           break;                                     \
      };                                              \
}                                                     \
Int_t TTableSorter::BSearch(valuetype value) const{   \
  union {  Bool_t   Bool;                             \
           Char_t   Char;                             \
           UChar_t  UChar;                            \
           Short_t  Short;                            \
           UShort_t UShort;                           \
           Int_t    Int;                              \
           UInt_t   UInt;                             \
           Long_t   Long;                             \
           ULong_t  ULong;                            \
           Float_t  Float;                            \
           Double_t Double;                           \
         } value_;                                    \
                                                      \
   switch (fColType) {                                \
         case  TTable::kFloat:                        \
           value_.Float = Float_t(value); break;      \
         case  TTable::kInt :                         \
           value_.Int   = Int_t(value);   break;      \
         case  TTable::kLong :                        \
           value_.Long  = Long_t(value);  break;      \
         case  TTable::kShort :                       \
           value_.Short = Short_t(value); break;      \
         case  TTable::kDouble :                      \
           value_.Double=  Double_t(value);break;     \
         case  TTable::kUInt:                         \
           value_.UInt  = UInt_t(value);  break;      \
         case  TTable::kULong :                       \
           value_.ULong = ULong_t(value); break;      \
         case  TTable::kUShort:                       \
           value_.UShort= UShort_t(value);break;      \
         case  TTable::kUChar:                        \
           value_.UChar = UChar_t(value); break;      \
         case  TTable::kChar:                         \
           value_.Char  = Char_t(value);  break;      \
         case  TTable::kBool:                         \
           value_.Bool  = Bool_t(value);  break;      \
         default:                                     \
           return -1;                                 \
           break;                                     \
   };                                                 \
   return BSearch(&value_);                           \
}                                                     \
Int_t TTableSorter::SelectSearch(valuetype value) const {         \
   valuetype **array = (valuetype **)fSortIndex;                  \
   Int_t nabove, nbelow, middle;                                  \
   nabove = fNumberOfRows+1;                                      \
   nbelow = 0;                                                    \
   while(nabove-nbelow > 1) {                                     \
      middle = (nabove+nbelow)/2;                                 \
      if (value == *array[middle-1]) { nbelow = middle; break; }  \
      if (value  < *array[middle-1]) nabove = middle;             \
      else                           nbelow = middle;             \
   }                                                              \
   nbelow--;                                                      \
   ((TTableSorter *)this)->fLastFound    = nbelow;                \
   if (nbelow < 0) return nbelow;                                 \
   return GetIndex(nbelow);                                       \
}

#define COMPAREFLOATVALUES(valuetype)                 \
int TTableSorter::Search##valuetype  (const void *elem1, const void **elem2) { \
         valuetype *value1 = (valuetype *)(elem1);    \
         valuetype *value2 = (valuetype *)(*elem2);   \
         valuetype diff = *value1-*value2;            \
         Int_t res = 0;                               \
         if (diff > 0)      res =  1;                 \
         else if (diff < 0) res = -1;                 \
         return res;                                  \
}                                                     \
int TTableSorter::Compare##valuetype  (const void **elem1, const void **elem2) {\
         valuetype *value1 = (valuetype *)(*elem1);   \
         valuetype *value2 = (valuetype *)(*elem2);   \
         valuetype diff = *value1-*value2;            \
         Int_t res = 0;                               \
         if (diff > 0  )    res =  1;                 \
         else if (diff < 0) res = -1;                 \
         if (res) return res;                         \
         return int(value1-value2);                   \
}                                                     \
BINARYSEARCH(valuetype)

//_____________________________________________________________________________
#define COMPAREVALUES(valuetype)  \
int TTableSorter::Search##valuetype  (const void *elem1, const void **elem2) { \
         valuetype *value1 = (valuetype *)(elem1);    \
         valuetype *value2 = (valuetype *)(*elem2);   \
         return    int(*value1-*value2);              \
}                                                     \
int TTableSorter::Compare##valuetype  (const void **elem1, const void **elem2) { \
         valuetype *value1 = (valuetype *)(*elem1);   \
         valuetype *value2 = (valuetype *)(*elem2);   \
         valuetype diff = *value1-*value2;            \
         if (diff ) return diff;                      \
         return int(value1-value2);                   \
}                                                     \
BINARYSEARCH(valuetype)

   COMPAREFLOATVALUES(Float_t)
   COMPAREVALUES(Int_t)
   COMPAREVALUES(Long_t)
   COMPAREVALUES(ULong_t)
   COMPAREVALUES(UInt_t)
   COMPAREVALUES(Short_t)
   COMPAREFLOATVALUES(Double_t)
   COMPAREVALUES(UShort_t)
   COMPAREVALUES(UChar_t)
   COMPAREVALUES(Char_t)
   COMPAREVALUES(Bool_t)

#define COMPAREORDER(valuetype) Compare##valuetype
#define SEARCHORDER(valuetype) Search##valuetype

//_____________________________________________________________________________
Int_t TTableSorter::BSearch(const void *value) const
{
   //to be documented
   Int_t index = -1;
   if (fSearchMethod) {
      void **p = (void **)::bsearch((void *) value,  // Object to search for
                   fSortIndex,          // Pointer to base of search data
                   fNumberOfRows,       // Number of elements
                   sizeof(void *),       // Width of elements
                   CALLQSORT(fSearchMethod));
      ((TTableSorter *)this)->fLastFound = -1;
      if (p) {
         const Char_t *res = (const Char_t *)(*p);
         ((TTableSorter *)this)->fLastFound = ((Char_t *)p - (Char_t *)fSortIndex)/sizeof(void *);
         // calculate index:
         if (!fsimpleArray)
            index =  fFirstRow + (res - (At(fFirstRow)+ fColOffset))/fParentRowSize;
         else
            index = ULong_t(res) - ULong_t(fsimpleArray)/fColSize;
      }
   }
   return index;
}

//_____________________________________________________________________________
Int_t TTableSorter::GetIndex(UInt_t sortedIndex) const
{
   // returns the original index of the row by its sorted index
   Int_t indx = -1;
   if (sortedIndex < UInt_t(fNumberOfRows) )  {
      void *p = fSortIndex[sortedIndex];
      if (p) {
         const Char_t *res = (const Char_t *)p;
         // calculate index:
         if (!fsimpleArray)
            indx = fFirstRow + (res - (At(fFirstRow) + fColOffset))/fParentRowSize;
         else
            indx = (ULong_t(res) - ULong_t(fsimpleArray))/fColSize;
      }
   }
   return indx;
}

#if 0
//_____________________________________________________________________________
int TTableSorter::CompareUChar  (const void *elem1, const void *elem2)
{
   //to be documented
   UChar_t *value1 = (UChar_t *)(*elem1);
   UChar_t *value2 = (UChar_t *)(*elem2);
   COMPAREVALUES(value1,value2)
}

//_____________________________________________________________________________
int TTableSorter::CompareChar   (const void *elem1, const void *elem2)
{
   //to be documented
   Char_t *value1 = (Char_t *)(*elem1);
   Char_t *value2 = (Char_t *)(*elem2);
   COMPAREVALUES(value1,value2)
}
#endif

//_____________________________________________________________________________
Int_t TTableSorter::CountKey(const void *key, Int_t firstIndx, Bool_t bSearch, Int_t *firstRow) const
{
 //
 //  CountKey counts the number of rows with the key value equal "key"
 //
 //  key      - it is a POINTER to the key value
 //  fistIndx - the first index within sorted array to star search
 //              = 0 by default
 //  bSearch  = kTRUE - binary search (by default) is used otherwise linear one
 //

   Int_t count = 0;
   if (firstRow) *firstRow = -1;
   if (fSearchMethod) {
      Int_t indx = firstIndx;
      Int_t nRows = GetNRows();
      if (!bSearch) {
         while ( indx < nRows && fSearchMethod(key,(const void **)&fSortIndex[indx])){indx++;}
         // Remember the first row been asked:
      } else {
         indx = FindFirstKey(key);
         if (indx >= 0 ) {  // Key was found let's count it
            count = TMath::Max(0,GetLastFound() - indx + 1);
            indx  = TMath::Max(GetLastFound()+1,firstIndx);
            // Forward pass
         }
      }
      if (indx >= 0) {
         while ( indx < nRows &&!fSearchMethod(key,(const void **)&fSortIndex[indx])){indx++; count++;}
         if (firstRow && count) *firstRow = indx-count;
      }
   }
   return count;
}

//_____________________________________________________________________________
Int_t TTableSorter::CountKeys() const
{
   //
   // Counts the number of different key values
   //
   Int_t count = 0;
   if (fSortIndex && fSortIndex[0]) {
      void *key = fSortIndex[0];
      Int_t indx = 0;
      while (indx < GetNRows()){
         indx += CountKey(key,indx,kFALSE);
         count++;
         key = fSortIndex[indx];
      }
   }
   return count;
}

//_____________________________________________________________________________
Bool_t TTableSorter::FillIndexArray(){
  //////////////////////////////////////////////////////////////
  // File the array of the pointers and check whether
  // the original table has been sorted to avoid an extra job.
  //
  // Return: kTRUE  - the table has been sorted
  //         kFALSE - otherwise
  //////////////////////////////////////////////////////////////
   assert(fSortIndex!=0);
   const char *row = At(fFirstRow) + fColOffset;
   Bool_t isPreSorted = kTRUE;
   const void  *sample = row;
   for (Int_t i=fFirstRow; i < fFirstRow+fNumberOfRows;i++,row += fParentRowSize) {
      fSortIndex[i-fFirstRow] = (char *)row;
      if ( isPreSorted) {
         void *ptr = &row;
         if (fCompareMethod(&sample,(const void **)ptr)>0) isPreSorted = kFALSE;
         else sample = row;
      }
   }
   return isPreSorted;
}

//_____________________________________________________________________________
Int_t TTableSorter::FindFirstKey(const void *key) const
{
 //
 // Looks for the first index of the "key"
 // within SORTED table AFTER sorting
 //
 // Returns: = -1 if the "key" was not found
 //
 // Note: This method has no sense for
 // ====  the float and double key
 //
 //       To get the index within the original
 //       unsorted table the GetIndex() method
 //       may be used like this:
 //       GetIndex(FindFirstKey(key))
 //
   Int_t indx = -1;
   if (BSearch(key)>=0) {
      indx = GetLastFound();
      if (indx >=0)
         while (indx > 0 && !fSearchMethod(key,(const void **)&fSortIndex[indx-1])) indx--;
   }
   return indx;
}

//_____________________________________________________________________________
const char * TTableSorter::GetTableName() const
{
   //to be documented
   return fParentTable ? fParentTable->GetName():"";
}

//_____________________________________________________________________________
const char * TTableSorter::GetTableTitle() const
{
   //to be documented
   return fParentTable ? fParentTable->GetTitle():"";
}

 //_____________________________________________________________________________
const char * TTableSorter::GetTableType() const
{
   //to be documented
   return fParentTable ? fParentTable->GetType():"";
}

//_____________________________________________________________________________
TTable *TTableSorter::GetTable() const
{
   //to be documented
   return (TTable *)fParentTable;
}

//_____________________________________________________________________________
void  TTableSorter::SetSearchMethod()
{
   // Select search function at once
   if (!fSearchMethod) {
      switch (fColType) {
         case  TTable::kFloat:
            fSearchMethod  = SEARCHORDER(Float_t);
            fCompareMethod = COMPAREORDER(Float_t);
            break;
         case  TTable::kInt :
            fSearchMethod = SEARCHORDER(Int_t);
            fCompareMethod = COMPAREORDER(Int_t);
            break;
         case  TTable::kLong :
            fSearchMethod = SEARCHORDER(Long_t);
            fCompareMethod = COMPAREORDER(Long_t);
            break;
         case  TTable::kShort :
            fSearchMethod  = SEARCHORDER(Short_t);
            fCompareMethod = COMPAREORDER(Short_t);
            break;
         case  TTable::kDouble :
            fSearchMethod = SEARCHORDER(Double_t);
            fCompareMethod = COMPAREORDER(Double_t);
            break;
         case  TTable::kUInt:
            fSearchMethod = SEARCHORDER(UInt_t);
            fCompareMethod = COMPAREORDER(UInt_t);
            break;
         case  TTable::kULong :
            fSearchMethod= SEARCHORDER(ULong_t);
            fCompareMethod = COMPAREORDER(ULong_t);
            break;
         case  TTable::kUShort:
            fSearchMethod = SEARCHORDER(UShort_t);
            fCompareMethod = COMPAREORDER(UShort_t);
            break;
         case  TTable::kUChar:
            fSearchMethod = SEARCHORDER(UChar_t);
            fCompareMethod = COMPAREORDER(UChar_t);
            break;
         case  TTable::kChar:
            fSearchMethod = SEARCHORDER(Char_t);
            fCompareMethod = COMPAREORDER(Char_t);
            break;
         case  TTable::kBool:
            fSearchMethod = SEARCHORDER(Bool_t);
            fCompareMethod = COMPAREORDER(Bool_t);
            break;
         default:
            break;

      };
   }
}

//____________________________________________________________________________
void  TTableSorter::QSort(){
 // Call the standard C run-time library "qsort" function
 //
   if (fCompareMethod)
      ::qsort(fSortIndex,       //Start of target array
              fNumberOfRows,       //Array size in elements
              sizeof(void *),      //Element size in bytes
              CALLQSORT(fCompareMethod));
}

//____________________________________________________________________________
void TTableSorter::LearnTable()
{
//
//  LearnTable() allows the TTableSorter to learn the structure of the
//  tables used to fill the ntuple.
//  table     - the name of the table
//  buildTree - if kTRUE, then add TBranches to the TTree for each table
//              column (default=kFALSE)
//
   TClass *classPtr = fParentTable->GetRowClass();
   if (!classPtr) return;

   if (!classPtr->GetListOfRealData()) classPtr->BuildRealData();
   if (!(classPtr->GetNdata())) return;

   const Char_t *types;
   Char_t *varname;

   TIter next(classPtr->GetListOfDataMembers());
   TDataMember *member = 0;
   while ( (member = (TDataMember *) next()) ) {
      varname = (Char_t *) member->GetName();

      if (strcmp(varname,fColName.Data())) continue;

      TDataType *memberType = member->GetDataType();
      types = memberType->GetTypeName();
      SetTitle(types);
      if (!strcmp("float", types))
         fColType = TTable::kFloat ;
      else if (!strcmp("int", types))
         fColType = TTable::kInt   ;
      else if (!strcmp("long", types))
         fColType = TTable::kLong  ;
      else if (!strcmp("short", types))
         fColType = TTable::kShort ;
      else if (!strcmp("double", types))
         fColType = TTable::kDouble;
      else if (!strcmp("unsigned int", types))
         fColType = TTable::kUInt  ;
      else if (!strcmp("unsigned long", types))
         fColType = TTable::kULong ;
      else if (!strcmp("unsigned short", types))
         fColType = TTable::kUShort;
      else if (!strcmp("unsigned char", types))
         fColType = TTable::kUChar;
      else if (!strcmp("char", types))
         fColType= TTable::kChar;
      else if (!strcmp("bool", types))
         fColType= TTable::kBool;

      if (fColType != TTable::kNAN) {
         Int_t dim = 0;
         Int_t globalIndex = 0;
         if ( (dim = member->GetArrayDim()) ) {
            // Check dimensions
            if (dim != fColDimensions) {
               Error("LearnTable","Wrong dimension");
               TTable *t = (TTable *)fParentTable;
               t->Print();
               return;
            }
            // Calculate the global index
            for( Int_t indx=0; indx < fColDimensions; indx++ ){
               globalIndex *= member->GetMaxIndex(indx);
               globalIndex += fIndexArray[indx];
            }
         }
         fColSize   = memberType->Size();
         fColOffset = member->GetOffset() + memberType->Size() * globalIndex;
      }
      break;
   }
}

#undef COMPAREVALUES
#undef COMPAREORDER
#undef COMPAREFLOATVALUES
#undef BINARYSEARCH
