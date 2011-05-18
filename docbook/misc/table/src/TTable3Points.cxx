// @(#)root/table:$Id$
// Author: Valery Fine   10/05/99  (E-mail: fine@bnl.gov)

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TTable3Points.h"

///////////////////////////////////////////////////////////////////////////////////
//
//   TTable3Points class is to create 3D view of any 3 columns of the TTable objects
//   with one and the same "key column value".
//
//   For example all values of the column "x[0]" "x[1]" "x[2]" of the begin_html <a href="http://www.rhic.bnl.gov/STAR/html/comp_l/root/html/g2t_tpc_hit_st.html"> g2t_tpc_hit </a> end_html table
//   from the rows with one and same "track_id" column value will be regarded
//   as an image of one and the same "track".
//   The last means all those points will be painted with one and the same 3D
//   attributes like "color", "size", "style", "light","markers", "connections"  etc.
//
//   The original TTable object must be pre-sorted by "key column" via TTableSorter
//   class
//
// void   CreatePoints(Tg2t_tpc_hit *points)
// {
//   g2t_tpc_hit_st *p = points->GetTable();
//
//  TTable3Points *track = 0;
//  TString tr;
//  tr = "track_p";
//  TTable &ttt = *((TTable *)points);
//  // Track2Line MUST be on heap otherwise 3D view will crash just code leaves this
//  // subroutine
//  We will assemble all points by its "track_p" field.
//
//  TTableSorter *Track2Line = new TTableSorter (ttt,"track_p");
//
//  Int_t i = 0;
//  Char_t buffer[10];
//  Int_t ntracks = 0;
//  const Int_t maxtracks = 5;
////---------------------------- Fill tracks -------------------
//  long currentId = -1;
//  long newId =  0;
//  g2t_tpc_hit_st *hitPoint = 0;
//  TVolume *thisTrack[7] = {0,0,0,0,0,0,0}; // seven volumes for 7 different colors
//  Int_t  MaxRowtoCount = 5000; // 5000;
//  Int_t  MaxTracks = Track2Line->CountKeys();
//  MaxTracks = 100;
//  for (i=0;i<Track2Line->GetNRows() && ntracks <  MaxTracks ;i++)
//  {
//   hitPoint = p + Track2Line->GetIndex(i);
//   newId =  hitPoint->track_p;
//   if (newId != currentId)  { // The hit for the new track has been found
//
//     const Char_t *xName = "x[0]";
//     const Char_t *yName = "x[1]";
//     const Char_t *zName = "x[2]";
//
//     track =  new TTable3Points(Track2Line,(const void *)&newId,xName,yName,zName);
//
//     // Create a shape for this node
//     TPolyLineShape *trackShape  =  new TPolyLineShape(track);
//     trackShape->SetVisibility(1);
//     Int_t colorIndx = ntracks%7;
//     trackShape->SetColorAttribute(colorIndx+kGreen);
//     trackShape->SetLineStyle(1);
//     trackShape->SetSizeAttribute(2);
//     // Create a node to hold it
//     if (!thisTrack[colorIndx])  {
//         thisTrack[colorIndx] = new TVolume("hits","hits",trackShape);
//         thisTrack[colorIndx]->Mark();
//         thisTrack[colorIndx]->SetVisibility();
//         TVolumePosition *pp = hall->Add(thisTrack[colorIndx]);
//         if (!pp) printf(" no position %d\n",ntrack);
//     }
//     else
//       thisTrack[colorIndx]->Add(trackShape);
//     currentId = newId;
//     ntracks++;
//   }
// }
//
///////////////////////////////////////////////////////////////////////////////////

ClassImp(TTable3Points)

//________________________________________________________________________________
TTable3Points::TTable3Points():fColumnOffset(0)
{
   //to be documented
}

//________________________________________________________________________________
TTable3Points::TTable3Points(TTableSorter *sorter,const void *key,
                       const Char_t *xName, const Char_t *yName, const Char_t *zName
                      ,Option_t *opt)
                : TTablePoints(sorter,key,opt)

{
   //to be documented
   fColumnOffset =  new ULong_t [kTotalSize];
   SetXColumn(xName);  SetYColumn(yName);  SetZColumn(zName);
}

//________________________________________________________________________________
TTable3Points::TTable3Points(TTableSorter *sorter,Int_t keyIndex,
                       const Char_t *xName, const Char_t *yName, const Char_t *zName
                      ,Option_t *opt)
                : TTablePoints(sorter,keyIndex,opt)

{
   //to be documented
   fColumnOffset =  new ULong_t [kTotalSize];
   SetXColumn(xName);  SetYColumn(yName);  SetZColumn(zName);
}

//________________________________________________________________________________
TTable3Points::~TTable3Points()
{
   //to be documented
   SafeDelete(fColumnOffset);
}

//________________________________________________________________________________
Float_t TTable3Points::GetAnyPoint(Int_t idx, EPointDirection xAxis) const
{
   //to be documented
   Float_t point  = 0;
   TTable  *table = 0;
   if (fTableSorter) table = fTableSorter->GetTable();
   if (table) {
      const Char_t *tablePtr = ((Char_t *)table->At(Indx(idx))) + fColumnOffset[xAxis] ;
      point = *((Float_t *)tablePtr);
   }
   return point;
}

//____________________________________________________________________________
void TTable3Points::SetAnyColumn(const Char_t *anyName, EPointDirection indx)
{
   //to be documented
   fColumnOffset[indx] = fTableSorter->GetTable()->GetOffset(anyName);
   if (fColumnOffset[indx] == ULong_t(-1))  MakeZombie();
}

//____________________________________________________________________________
Float_t *TTable3Points::GetXYZ(Float_t *xyz,Int_t idx, Int_t num) const
{
   //to be documented
   if (xyz) {
      Int_t size = TMath::Min(idx+num,Size());
      Int_t j=0;
      for (Int_t i=idx;i<size;i++) {
         xyz[j++] = GetX(i);
         xyz[j++] = GetY(i);
         xyz[j++] = GetZ(i);
      }
   }
   return xyz;
}
