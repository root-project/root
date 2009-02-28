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

#include "TArrayL.h"
#include "TTableMap.h"

//////////////////////////////////////////////////////////////////////////////
// TTableMap class is helper class to keep the list of the referencs to the
// TTable rows and iterate over it.
// TTableMap is a persistent class.
// The pointer to the TTableMap object may be used as an element
// of the TTable row and saved with the table all together.
//
// For example, the track table may contain a member to the "map" of the hits
//  struct {
//    float helix;
//    TTableMap *hits;
//  } tracks_t;
//
//   // Create track table:
//   LArTrackTable *tracks = new LArTrackTable(...);
//
//   // Get pointer to the hit table
//   LArHitTable *hits = GiveMeHits();
//   // Loop over all tracks
//   LArTrackTable::iterator track = tracks->begin();
//   LArTrackTable::iterator last = tracks->end();
//   for (;track != last;track++) {
//     // Find all hits of this track
//      LArHitTable::iterator hit     = hits->begin();
//      LArHitTable::iterator lastHit = hits->end();
//      Long_t hitIndx = 0;
//      // Create an empty list of this track hits
//      (*track).hits = new TTableMap(hits);
//      for(;hit != lastHit;hit++,hitIndx) {
//        if (IsMyHit(*hit)) {  // add this hit index to the current track
//           (*track).hits->push_back(hitIndx);
//        }
//      }
//   }
//___________________________________________________________________

ClassImp(TTableMap)

TTableMap::TTableMap(const TTable *table)
          : fTable(table)
{
   //to be documented
}

//___________________________________________________________________
void TTableMap::Streamer(TBuffer &R__b)
{
   // UInt_t R__s, R__c;
   TArrayL vecIO;
   if (R__b.IsReading()) {
      Version_t v =  R__b.ReadVersion();
      if (v) { }
      // read Table
      R__b >> fTable;
      // Read index array
      vecIO.Streamer(R__b);
      Int_t n = vecIO.GetSize();
      Int_t i = 0;
      reserve(n);
      Long_t *thisArr = vecIO.GetArray();
      for (i=0; i<n; i++,thisArr++) push_back(*thisArr);
   } else {
      // Write TTable
      assert(IsValid());
      R__b.WriteVersion(IsA());
      R__b << fTable;
      //  Write index array
      TTableMap::iterator ptr = begin();
      vecIO.Adopt(size(),&(*ptr));
      vecIO.Streamer(R__b);
      vecIO.fArray=0;  // we should not destroy the real array
   }
}
