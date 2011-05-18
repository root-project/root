// @(#)root/table:$Id$
// Author: Valery Fine   14/05/99  (E-mail: fine@bnl.gov)

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TTablePoints.h"

///////////////////////////////////////////////////////////////////////////////////////
//                                                                                   //
//   Defines the TTable as an element of "event" geometry                            //
//                                                                                   //
//  +SEQ,TTablePoints.                                                               //
//  +SEQ,T<your_table_name_here>.                                                    //
//                                                                                   //
//  class T<your_table_name_here>_Points : public TTablePoints                       //
//  {                                                                                //
//    public:                                                                        //
//       T<your_table_name_here>_Points(TTableSorter *sorter,const void *key,Option_t *opt):
//                TTablePoints(sorter,key,opt){}                                     //
//       virtual  ~T<your_table_name_here>_Points(){} // default destructor          //
//       virtual Float_t GetX(Int_t indx) { return ((<your_table_name_here>_st *)fRows)[Indx(idx)]-> <x>;}               //
//       virtual Float_t GetY(Int_t indx) { return ((<your_table_name_here>_st *)fRows)[Indx(idx)]-> <y>;}               //
//       virtual Float_t GetZ(Int_t indx) { return ((<your_table_name_here>_st *)fRows)[Indx(idx)]-> <z>;}               //
//  };                                                                               //
//                                                                                   //
///////////////////////////////////////////////////////////////////////////////////////

ClassImp(TTablePoints)

//____________________________________________________________________________
TTablePoints::TTablePoints()
{
   //to be documented
   fTableSorter =  0;
   fKey         =  0;
   fFirstRow    = -1;
   fSize        =  0;
   fRows        =  0;
}

//____________________________________________________________________________
TTablePoints::TTablePoints(TTableSorter *sorter,const void *key,Option_t *opt)
{
   //to be documented
   fTableSorter =  0;
   fKey         =  0;
   fFirstRow    = -1;
   fSize        =  0;
   fRows        =  0;
   if (sorter) {
      fTableSorter = sorter;
      fKey         = key;
      fSize        = sorter->CountKey(fKey,0,kTRUE,&fFirstRow);
      SetTablePointer(GetTable());
   }
   SetOption(opt);
}

//____________________________________________________________________________
TTablePoints::TTablePoints(TTableSorter *sorter, Int_t keyIndex,Option_t *opt)
{
   //to be documented
   fTableSorter =  0;
   fKey         =  0;
   fFirstRow    = -1;
   fSize        =  0;
   fRows        =  0;
   if (sorter) {
      fTableSorter = sorter;
      fKey         = sorter->GetKeyAddress(keyIndex);
      fSize        = sorter->CountKey(fKey,keyIndex,kFALSE,&fFirstRow);
      SetTablePointer(GetTable());
   }
   SetOption(opt);
}

//______________________________________________________________________________
Int_t TTablePoints::DistancetoPrimitive(Int_t /*px*/, Int_t /*py*/)
{
//*-*-*-*-*-*-*Compute distance from point px,py to a 3-D points *-*-*-*-*-*-*
//*-*          =====================================================
//*-*
//*-*  Compute the closest distance of approach from point px,py to each segment
//*-*  of the polyline.
//*-*  Returns when the distance found is below DistanceMaximum.
//*-*  The distance is computed in pixels units.
//*-*
//*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
   return -1;
}
