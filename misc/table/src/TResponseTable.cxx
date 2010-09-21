// @(#)root/table:$Id$
// Author: Valery Fine(fine@bnl.gov)   03/04/2002

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "TResponseTable.h"

//______________________________________________________________________________
//
// TResponseTable is an example of the custom version of the TGenericTable class
//______________________________________________________________________________

ClassImp(TResponseTable)
TableClassStreamerImp(TResponseTable)

//______________________________________________________________________________
TResponseTable::TResponseTable():TGenericTable(), fResponseLocation(-1)
{
   //to be documented
}

//______________________________________________________________________________
TResponseTable::TResponseTable(const char *name,const char *volumePath, const char *responseDefinition, Int_t /*allocSize*/)
 :  TGenericTable(), fResponseLocation(-1)
{
   // Set an empty descriptor
   SetDescriptorPointer(new TTableDescriptor(name));

   // The first element is always "int TRACK;"
   AddElement("TRACK",kInt);
   AddVolumePath(volumePath);
   AddResponse(responseDefinition);
   fSize = GetDescriptorPointer()->Sizeof();
   fResponseLocation = FindResponseLocation(*GetDescriptorPointer());
   SetType("DetectorResponse");
}
//______________________________________________________________________________
void TResponseTable::AddVolumePath(const char *path)
{
   //to be documented
   Int_t counter = 0;
   const Int_t maxResponseCounter = 15;
   const char *next = &path[0];
   while( ( *next && *next != ' ') &&  counter < maxResponseCounter ) {
      TString elName;
      for (int j=0; j<4 && (next[j] != ' ');j++)  elName += next[j];
      AddElement(elName,kInt);
      next += 4;
      counter++;
   }
}
//______________________________________________________________________________
void TResponseTable::AddResponse(const char *chit)
{
   //to be documented
   Int_t counter = 0;
   const Int_t maxResponseCounter = 15;
   const char *next = &chit[0];
   while( ( *next != ' ' ) &&  counter < maxResponseCounter )  {
      TString elName;
      for (int j=0; j<4 && (next[j] != ' ');j++)  elName += next[j];
      AddElement(elName,kFloat);
      next += 4;
      counter++;
   }
}
//______________________________________________________________________________
void TResponseTable::AddElement(const char *path,EColumnType type)
{
   //to be documented
   assert( (type == kInt || type == kFloat ) );

   TTableDescriptor  &dsc = *GetTableDescriptors();
   Int_t nRow = dsc.GetNRows();
   tableDescriptor_st row;

   memset(&row,0,sizeof(row));
   strlcpy(row.fColumnName,path,sizeof(row.fColumnName));
   if (nRow) row.fOffset = dsc[nRow-1].fOffset + dsc[nRow-1].fSize;

   row.fType = type;
   if (type == kInt)
      row.fTypeSize = sizeof(Int_t);
   else
      row.fTypeSize = sizeof(Float_t);

   row.fSize = row.fTypeSize;
   dsc.AddAt(&row);
}

//______________________________________________________________________________
void TResponseTable::SetResponse(int track, int *nvl, float *response)
{
   // Add one extra his/digit to the table
   // Reallocate the table if needed
   char    *charBuffer     = new char[GetRowSize()];
   Int_t   *nvlBuffer      = (Int_t *)charBuffer;
   Float_t *responseBuffer = (Float_t *)charBuffer;
   Int_t jResponse  = 0;
   Int_t jNvl       = 0;

   // Loop for the response information
   TTableDescriptor  &dsc = *GetTableDescriptors();
   Int_t nRow = dsc.GetNRows();
   tableDescriptor_st *row = dsc.GetTable();
   nvlBuffer[0] =  track; row++;
   for (int i=1;i<nRow;i++,row++) {
      if (row->fType == kFloat) {
         responseBuffer[i] = response[jResponse++];
      } else {
         nvlBuffer[i] = nvl[jNvl++];
      }
   }
   AddAt(charBuffer);
   delete [] charBuffer;
}

//______________________________________________________________________________
Int_t TResponseTable::FindResponseLocation(TTableDescriptor  &dsc)
{
 // Look up the table descriptor to find the
 // first respnse value location
 // TResponsetable layout:
 //  offset
 //   +0    int TRACK
 //   +1
 //   ...   int <volume path description>
 //  +nVl.
 //  +nVl+1  <----  fResponseLocation
 //   ...   response values
 //  RowSize

   // responseLocation is an offset of the first float data-member
   Int_t responseLocation = -1;
   Int_t nRow = dsc.GetNRows();
   tableDescriptor_st *row = dsc.GetTable();
   for (int i=0;i<nRow;i++,row++) {
      if (row->fType == kFloat) {
         // found
         responseLocation = i;
         break;
      }
   }
   return responseLocation;
}
