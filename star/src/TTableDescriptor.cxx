// @(#)root/star:$Name:  $:$Id: TTableDescriptor.cxx,v 1.8 2001/05/31 06:36:36 brun Exp $
// Author: Valery Fine   09/08/99  (E-mail: fine@bnl.gov)
// $Id: TTableDescriptor.cxx,v 1.8 2001/05/31 06:36:36 brun Exp $
#include <stdlib.h>

#include "TROOT.h"
#include "TTableDescriptor.h"
#include "TTable.h"
#include "TClass.h"
#include "TDataMember.h"
#include "TDataType.h"
#include "Ttypes.h"
#include "TInterpreter.h"

TTableDescriptor *TTableDescriptor::fgColDescriptors = 0;
TableClassImp(TTableDescriptor,tableDescriptor_st)

//______________________________________________________________________________
void TTableDescriptor::Streamer(TBuffer &R__b)
{
  // The custom Streamer for this table
  fSecondDescriptor = 0;
  TTable::Streamer(R__b);
}

//______________________________________________________________________________
TTableDescriptor::TTableDescriptor(const TTable *parentTable)
 : TTable("tableDescriptor",sizeof(tableDescriptor_st)), fRowClass(0),fSecondDescriptor(0)
{
  if (parentTable) {
     TClass *classPtr = parentTable->GetRowClass();
     Init(classPtr);
  }
  else MakeZombie();
}

//______________________________________________________________________________
TTableDescriptor::TTableDescriptor(TClass *classPtr)
 : TTable("tableDescriptor",sizeof(tableDescriptor_st)),fRowClass(0),fSecondDescriptor(0)
{
  // Create a descriptor of the C-structure defined by TClass
  // TClass *classPtr must be a valid pointer to TClass object for
  // "plain" C_struture only !!!
  Init(classPtr);
}
//______________________________________________________________________________
void TTableDescriptor::Init(TClass *classPtr)
{
  // Create a descriptor of the C-structure defined by TClass
  // TClass *classPtr must be a valid pointer to TClass object for
  // "plain" C_structure only !!!
  fSecondDescriptor = 0;
  SetType("tableDescriptor");
  if (classPtr) {
    fRowClass = classPtr; // remember my row class
    SetName(classPtr->GetName());
    LearnTable(classPtr);
  }
  else
    MakeZombie();
}
//______________________________________________________________________________
TTableDescriptor::~TTableDescriptor()
{
#ifdef NORESTRICTIONS
  if (!IsZombie()) {
    for (Int_t i=0;i<GetNRows();i++) {
      Char_t *name = (Char_t *)ColumnName(i);
      if (name) delete [] name;
      UInt_t  *indxArray = (UInt_t *)IndexArray(i);
      if (indxArray) delete [] indxArray;
    }
  }
#endif
  delete fSecondDescriptor;
}

//____________________________________________________________________________
TString TTableDescriptor::CreateLeafList() const
{
  // Create a list of leaf to be useful for TBranch::TBranch ctor
  const Char_t TypeMapTBranch[]="\0FIISDiisbBC";
  Int_t maxRows = NumberOfColumns();
  TString string;
  for (Int_t i=0;i<maxRows;i++){
    if (i) string += ":";
    UInt_t nDim = Dimensions(i);

    UInt_t totalSize = 1;
    UInt_t k = 0;

    if (nDim) {
      const UInt_t *indx = IndexArray(i);
      if (!indx){
        string = "";
        Error("CreateLeafList()","Can not create leaflist for arrays");
        return string;
      }
      for (k=0;k< nDim; k++) totalSize *= indx[k];
    }
    const Char_t *colName = ColumnName(i);
    if (totalSize > 1) {
       for ( k = 0; k < totalSize; k++) {
         Char_t buf[10];
         sprintf(buf,"_%d",k);
         string += colName;
         string += buf;
         if (k==0) {
           string += "/";
           string += TypeMapTBranch[ColumnType(i)];
         }
         if (k != totalSize -1) string += ":";
      }
    } else {
       string += ColumnName(i);
       string += "/";
       string += TypeMapTBranch[ColumnType(i)];
    }
  }
  return string;
}

//______________________________________________________________________________
TTableDescriptor *TTableDescriptor::MakeDescriptor(const char *structName)
{
	///////////////////////////////////////////////////////////
	//
	// MakeDescriptor(const char *structName) - static method
	//                structName - the name of the C structure 
	//                             to create descriptor of
	// return a new instance of the TTableDescriptor or 0 
	// if the "structName is not present with the dictionary
	//
	///////////////////////////////////////////////////////////
	TTableDescriptor *dsc = 0;
//    TClass *cl = gROOT->GetClass(structName, kFALSE);
    TClass *cl = new TClass(structName,1,0,0);
	assert(cl);
	dsc = new TTableDescriptor(cl);
	return dsc;
}

//____________________________________________________________________________
void TTableDescriptor::LearnTable(const TTable *parentTable)
{
  if (!parentTable) {
    MakeZombie();
    return;
  }
  LearnTable(parentTable->GetRowClass());
}

//____________________________________________________________________________
void TTableDescriptor::LearnTable(TClass *classPtr)
{
//
//  LearnTable() creates an array of the descriptors for elements of the row
//
// It creates a descriptor of the C-structure defined by TClass
// TClass *classPtr must be a valid pointer to TClass object for
// "plain" C-structure only !!!
//
//  This is to introduce an artificial restriction demanded by STAR database group
//
//    1. the name may be 31 symbols at most
//    2. the number the dimension is 3 at most
//
//  To lift this restriction one has to provide -DNORESTRICTIONS CPP symbol and
//  recompile code (and debug code NOW!)
//

  if (!classPtr) return;

  if (!classPtr->GetListOfRealData()) {
	  const char *name = classPtr->GetName();
	   char buffer[512];
	   sprintf(buffer," new %s();",name);
 	   void *p = (void *)gInterpreter->Calc(buffer);
 	   classPtr->BuildRealData(p);
  	   sprintf(buffer,"delete ((%s *)(0x%lx);",name,(Long_t)p);

  }
  if (!(classPtr->GetNdata())) return;

  Char_t *varname;

  tableDescriptor_st elementDescriptor;

  ReAllocate(classPtr->GetListOfDataMembers()->GetSize());
  Int_t columnIndex = 0;
  TIter next(classPtr->GetListOfDataMembers());
  TDataMember *member = 0;
  TDataSet *comments = new TDataSet(".comments",this,kTRUE);
  while ( (member = (TDataMember *) next()) ) {
    memset(&elementDescriptor,0,sizeof(tableDescriptor_st));
    varname = (Char_t *) member->GetName();
#ifdef NORESTRICTIONS
//  This is remove to introduce an artificial restriction demanded by STAR infrastructure group
                                             elementDescriptor.fColumnName = StrDup(varname);
#else
                                             elementDescriptor.fColumnName[0] = '\0';
         strncat(elementDescriptor.fColumnName,varname,sizeof(elementDescriptor.fColumnName));
#endif
    // define index
    TDataType *memberType = member->GetDataType();
                                              elementDescriptor.fTypeSize = memberType->Size();
    elementDescriptor.fType = TTable::GetTypeId(memberType->GetTypeName());
    Int_t globalIndex = 1;
    if (elementDescriptor.fType != kNAN) {
      Int_t dim = 0;
      if ( (dim = member->GetArrayDim()) ) {
                                              elementDescriptor.fDimensions = dim;
#ifdef NORESTRICTIONS
                                              elementDescriptor.fIndexArray = new UInt_t(dim);
#else
       UInt_t maxDim = sizeof(elementDescriptor.fIndexArray)/sizeof(UInt_t *);
       if (UInt_t(dim) > maxDim) {
                Error("LearnTable","Too many dimenstions - %d", dim);
                dim =  maxDim;
       }
#endif
        for( Int_t indx=0; indx < dim; indx++ ){
                                              elementDescriptor.fIndexArray[indx] = member->GetMaxIndex(indx);
          globalIndex *= elementDescriptor.fIndexArray[indx];
        }
      }
    }
    else Error("LearnTable","Wrong data type for <%s> structure",classPtr->GetName());
    elementDescriptor.fSize   =  globalIndex * (elementDescriptor.fTypeSize);
    elementDescriptor.fOffset = member->GetOffset();
    AddAt(&elementDescriptor,columnIndex); columnIndex++;
    (new TDataSet(varname,comments))->SetTitle(member->GetTitle());
  }
}
 
//______________________________________________________________________________
Int_t TTableDescriptor::UpdateOffsets(const TTableDescriptor *newDescriptor)
{
  //                  "Schema evolution"
  // Method updates the offsets with a new ones from another descriptor
  // 
  Int_t maxColumns = NumberOfColumns();
  Int_t mismathes = 0;
 
  if (   (UInt_t(maxColumns) == newDescriptor->NumberOfColumns())
      && (memcmp(GetArray(),newDescriptor->GetArray(),sizeof(tableDescriptor_st)*GetNRows()) == 0)
     ) return mismathes; // everything fine for sure !

  // Something wrong here, we have to check things piece by piece
  for (Int_t colCounter=0; colCounter < maxColumns; colCounter++)
  {
    Int_t colNewIndx = newDescriptor->ColumnByName(ColumnName(colCounter));
    // look for analog
    EColumnType newType = newDescriptor->ColumnType(colNewIndx);
#ifdef __STAR__
    if (newType == kInt)       newType = kLong;
    else if (newType == kUInt) newType = kULong;
#endif
    if (    colNewIndx >=0
         && Dimensions(colCounter) == newDescriptor->Dimensions(colNewIndx)
         && ColumnType(colCounter) == newType
       )  {
      SetOffset(newDescriptor->Offset(colNewIndx),colCounter);
      if (colNewIndx != colCounter) {
        printf("Schema evolution: \t%d column of the \"%s\" table has been moved to %d-th column\n",
        colCounter,ColumnName(colCounter),colNewIndx);
        mismathes++;
      }
    }
    else {
      printf("Schema evolution: \t%d column \"%s\" of %d type has been lost\n",
        colCounter,ColumnName(colCounter),ColumnType(colCounter));
      printf(" Indx = %d, name = %s \n", colNewIndx, ColumnName(colCounter));
      SetOffset(UInt_t(-1),colCounter);
      mismathes++;
    }
  }
  if (!mismathes && UInt_t(maxColumns) != newDescriptor->NumberOfColumns()) {
      mismathes++;
      printf("Warning: One extra column has been introduced\n");
  }
  return mismathes;
}

//____________________________________________________________________________
const Int_t TTableDescriptor::ColumnByName(const Char_t *columnName) const
{
 // Find the column index but the column name
 const tableDescriptor_st *elementDescriptor = ((TTableDescriptor *)this)->GetTable();
 Int_t i = -1;
 if (!elementDescriptor) return i;
 Int_t nRows = GetNRows();
 char *bracket = 0;
 if (nRows) {
   char *name = StrDup(columnName);
   if ((bracket = strchr(name,'[')) )  *bracket = 0;
   for (i=0; i < nRows; i++,elementDescriptor++)
     if (strcmp(name,elementDescriptor->fColumnName) == 0) break;
   delete [] name;
 }
 if (i==nRows) i = -1;
 // Check array
 if (bracket && !Dimensions(i)) {
     i = -1;
     Warning("ColumnByName","%s column contains a scalar value",columnName);
 }
 return i;
}

//____________________________________________________________________________
Int_t TTableDescriptor::Offset(const Char_t *columnName) const
{
  // Return offset of the column defined by "columnName"
  // Take in account index if provided
  // Can not handle multidimensional indeces yet.

   Int_t indx = ColumnByName(columnName);
   Int_t offset = -1;
   if (indx >= 0 ) {
      offset = Offset(indx);
      const char *openBracket = 0;
      if ( (openBracket = strchr(columnName,'['))  )
         offset += atoi(openBracket+1)*TypeSize(indx);
   }
   return offset;
}

//____________________________________________________________________________
Int_t TTableDescriptor::ColumnSize(const Char_t *columnName) const
{
  Int_t indx = ColumnByName(columnName);
  if (indx >= 0 ) indx = ColumnSize(indx);
  return indx;
}

//____________________________________________________________________________
Int_t TTableDescriptor::TypeSize(const Char_t *columnName) const
{
  Int_t indx = ColumnByName(columnName);
  if (indx >= 0 ) indx = TypeSize(indx);
  return indx;
}

//____________________________________________________________________________
Int_t TTableDescriptor::Dimensions(const Char_t *columnName) const
{
  Int_t indx = ColumnByName(columnName);
  if (indx >= 0 ) indx = Dimensions(indx);
  return indx;
}

//____________________________________________________________________________
TTable::EColumnType TTableDescriptor::ColumnType(const Char_t *columnName) const
{
  Int_t indx = ColumnByName(columnName);
  if (indx >= 0 ) indx = ColumnType(indx);
  return EColumnType(indx);
}

