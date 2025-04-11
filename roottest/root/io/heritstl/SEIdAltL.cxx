////////////////////////////////////////////////////////////////////////////
// $Id$
//
// SEIdAltL
//
// SEIdAltL is a vector+iterator for SEIdAltLItems
//
// Author:  R. Hatcher 2001.10.22
//
////////////////////////////////////////////////////////////////////////////

#include "SEIdAltL.h"
#include "PlexCalib.h"

#include "TMath.h"
#include "assert.h"

#include <iostream>
#include <iomanip>
#include <algorithm>
#include <functional>

using namespace std;

ClassImp(SEIdAltL)

//_____________________________________________________________________________
ostream& operator<<(ostream& os, const SEIdAltL& alt)
{
  // Print out the list
  if (alt.size() <= 0) {
     os << "Empty SEIdAltL " << endl;
  }
  else {
     char marker;
     //Int_t seid;
     os << "     i | StripEndId (size=" << alt.size() 
        << ",cur=" << alt.fCurrent << ") "
        << "| weight    " 
        << "| muon unit " 
        << "| t offset  " 
        << endl;
     os << "-------+-----------------------------"
        << "+-----------" 
        << "+-----------" 
        << "+-----------" 
        << endl;

     for (unsigned int i=0; i<alt.size(); ++i) {
        (i==alt.fCurrent) ? marker = '*' : marker = ' ';
        SEIdAltLItem item = alt[i];
        // seid   = item.GetSEId();
        os << " " << marker << " " << i 
           << " " << item
           << endl;
     }
  }
  return os;
}

//______________________________________________________________________________
SEIdAltL::SEIdAltL()
   : fCurrent(0), fError(kUnchecked)
{
   // Default constructor
}

//______________________________________________________________________________
SEIdAltL::SEIdAltL(const SEIdAltL &rhs)
   : std::vector<SEIdAltLItem>(rhs), fCurrent(rhs.fCurrent), fError(kUnchecked)
{

   // deep copy constructor
   for (unsigned int i=0; i<rhs.size(); ++i) {
      this->push_back(rhs[i]);
   }

}

//______________________________________________________________________________
SEIdAltL::~SEIdAltL()
{
   // delete all the owned sub-objects
}

//______________________________________________________________________________
void SEIdAltL::AddStripEndId(const Int_t& pseid, Float_t weight,
                                 const PlexCalib* calib, Int_t adc, Double_t time)
{
   // add a new item to the list

   fError = kUnchecked;  // adding a new strip makes consistency unknown

   if (calib) {
      SEIdAltLItem item = calib->CalibStripEnd(pseid,adc,time);
      item.SetWeight(weight);
      this->push_back(item);
   }
   else {
      SEIdAltLItem item(pseid,weight);
      this->push_back(item);
   }
}

//______________________________________________________________________________
void SEIdAltL::DropCurrent()
{
   // Remove current item from the list.
   // Do not reset current position of iterator.
   // An iterative removal should start from Last() and use Previous().

   UInt_t n = size();

   if (n <= 0) return;  // one cannot drop what one doesn't have
   if (fCurrent>=n) {
      cout
         << "can not DropCurrent (fCurrent=" << fCurrent 
         << ") on a list of " << n << " items " << endl;
      return;
   }

   SEIdAltLIter cursor = this->begin() + fCurrent;
   // delete the owned item
   this->erase(cursor);

}

//______________________________________________________________________________
void SEIdAltL::DropZeroWeights()
{
   /*
   // Remove pairs from the list that have weight == 0

   if (size() <= 0) return;  // one cannot drop what one doesn't have

   // move all zero items to the end
   SEIdAltLIter new_end = 
      remove_if(this->begin(), this->end(), 
                mem_fun_ref(&SEIdAltLItem::IsZeroWeight));

   // erase the moved items out of the array 
   this->erase(new_end,this->end());
*/
}

//______________________________________________________________________________
void SEIdAltL::KeepTopWeights(UInt_t n, Bool_t keeporder)
{
   // Remove all but "n" pairs from the list (top "n" sorted by weight)
   // Final relative order of elements in the list is unchanged
   // if keeporder=kTRUE otherwise list ordered by decending weights.
   // If "n" would separate values of the same weight then
   // more than "n" are kept.

   unsigned int i;
   const unsigned int cnt(size());

   if (cnt <= 0) return;  // one cannot drop what one doesn't have

   if (n >= cnt) {
      if (keeporder) return; // keep everything, no change in order
      else n = cnt;          // perform sort but don't go beyond end
   }

   SEIdAltL& self = *this;

   if (n <= 0) { // special case for new size=0
      this->clear();
      return;
   }

   // make a copy of the weights
   // sort it
   // select the nth down the list for lowest weight value to keep
   vector<Float_t> sortedwgt(cnt);
   vector<unsigned int> sortedindx(cnt);
   for (i=0; i<cnt; i++) sortedwgt[i] = self[i].GetWeight();
   Bool_t down=kTRUE;
   TMath::Sort(cnt,&(sortedwgt[0]),&(sortedindx[0]),down);
   // n-1 because C arrays start with 0
   Int_t   icut = sortedindx[n-1];
   Float_t  cut = sortedwgt[icut]; 

   // count final size
   // may not be ==n because of two entries with same weight
   vector<SEIdAltLItem> tempVector;
   for (i=0; i<cnt; i++) {
      unsigned int indxold = i;
      if (!keeporder) indxold = sortedindx[i];
      if (self[indxold].GetWeight() >= cut) {
         tempVector.push_back(self[indxold]);
      }
   }

   // replace current vector with newly created temporary
   this->swap(tempVector);

}

//______________________________________________________________________________
void SEIdAltL::ClearWeights()
{
   // set all the weights to zero

   SEIdAltL& self = *this;
   for (unsigned int i=0; i<size(); i++) (self[i]).SetWeight(0.0);
}

//______________________________________________________________________________
const SEIdAltLItem& SEIdAltL::GetBestItem() const
{
   // find the SEIdAltLItem with the highest weight (const version)

   if (size() <= 0) abort();
   
   SEIdAltLConstIter cursor = this->begin();
   SEIdAltLConstIter best   = this->begin();
   Float_t wgt, maxwgt = -1.0e-37;
   while (cursor != this->end()) {
      const SEIdAltLItem& item = *cursor;
      if ( (wgt = item.GetWeight()) > maxwgt ) {
         maxwgt = wgt; best = cursor;
      }
      cursor++;
   }
   return *best;
      
}
//______________________________________________________________________________
SEIdAltLItem& SEIdAltL::GetBestItem()
{
   // find the SEIdAltLItem with the highest weight
   if (size() <= 0) abort();
   
   SEIdAltLIter cursor = this->begin();
   SEIdAltLIter best   = this->begin();
   Float_t wgt, maxwgt = -1.0e-37;
   while (cursor != this->end()) {
      SEIdAltLItem& item = *cursor;
      if ( (wgt = item.GetWeight()) > maxwgt ) {
         maxwgt = wgt; best = cursor;
      }
      cursor++;
   }
   return *best;
      
}
//______________________________________________________________________________
Int_t SEIdAltL::GetBestSEId() const
{
   // find the Int_t with the highest weight and return by value

   if (size() <= 0) return Int_t(); // return bad value if no list
  
   return GetBestItem().GetSEId();

}

//______________________________________________________________________________
Float_t SEIdAltL::GetBestWeight() const
{
   // find the highest weight

   if (size() <= 0) return -1.0e30;
   
   return GetBestItem().GetWeight();

}

//______________________________________________________________________________
const SEIdAltLItem& SEIdAltL::GetCurrentItem() const
{
   // return by value current SEIdAltLItem (const version)

   // if (!IsValid()) return 0; // no list or out of range
   assert(IsValid());

   SEIdAltLConstIter cursor = this->begin() + fCurrent;
   return *cursor;

}

//______________________________________________________________________________
SEIdAltLItem& SEIdAltL::GetCurrentItem()
{
   // return by value current SEIdAltLItem

   // if (!IsValid()) return 0; // no list or out of range
   assert(IsValid());

   SEIdAltLIter cursor = this->begin() + fCurrent;
   return *cursor;

}

//______________________________________________________________________________
Int_t SEIdAltL::GetCurrentSEId() const
{
   // return by value current Int_t

   // if (!IsValid()) return Int_t(); // no list or out of range
   assert(IsValid());
   return GetCurrentItem().GetSEId();

}

//______________________________________________________________________________
Float_t SEIdAltL::GetCurrentWeight() const
{
   // return the weight attached to the current Int_t

   // if (!IsValid()) return -1.0e37;  // no list or out of range
   assert(IsValid());
   return GetCurrentItem().GetWeight();

}

//______________________________________________________________________________
Bool_t SEIdAltL::IsValid() const
{
   // is current position a valid entry

   return size()>0 &&  fCurrent<size();

}

//______________________________________________________________________________
void SEIdAltL::Print(Option_t *option) const
{
   // Print out the list

   unsigned int n = size();

   if (n <= 0) {
      printf("Empty SEIdAltL\n");
      return;
   }

   const SEIdAltLItem& best = GetBestItem();

   const SEIdAltL& self = *this;
   unsigned int i;
   char cursormarker, bestmarker;
   Int_t seid;
//   Float_t        weight;

   switch (option[0]) {
   case 'c':
   case 'C':
      // compact notation assumes that there isn't a mixup
      // and all items share a common detector/plane/subpart/end
      seid = GetCurrentSEId(); // any will do
      printf("[ det/plane/...]");
      for (i=0; i<n; i++) {
         const SEIdAltLItem& item = self[i];
         bestmarker   = (item == best)     ? '!' : ' ';
         cursormarker = (i    == fCurrent) ? '@' : ' ';
         seid   = item.GetSEId();
         printf(" %c%c%3d",bestmarker,cursormarker,seid);

      }
      if (option[0] == 'C') {
         printf("\n          wgt ");
         for (i=0; i<n; i++) {
            const SEIdAltLItem& item = self[i];
            printf(" %5.3f",item.GetWeight());
         }
      }
      printf("\n");

      break;
   default:

      printf("      i | StripEndId  | weight    |  PE    | Linear    | S2S Corr  | time\n");
      printf(" -------+-------------+-----------+--------+-----------+-----------+-----------\n");
      for (i=0; i<n; i++) {
         const SEIdAltLItem& item = self[i];
         bestmarker   = (item == best)     ? '!' : ' ';
         cursormarker = (i    == fCurrent) ? '@' : ' ';
         printf(" %c%c %3d | ",bestmarker,cursormarker,i);
         item.Print("c");
         printf("\n");
      }
   }
}

//______________________________________________________________________________
void SEIdAltL::SetCurrentWeight(Float_t weight)
{
   // set the weight attached to the current Int_t

   if (!IsValid()) return; // no list or out of range
   SEIdAltLIter cursor = this->begin() + fCurrent;
   (*cursor).SetWeight(weight);

}

//______________________________________________________________________________
void SEIdAltL::AddToCurrentWeight(Float_t wgtadd)
{
   // add to the weight attached to the current Int_t

   if (!IsValid()) return; // no list or out of range
   SEIdAltLIter cursor = this->begin() + fCurrent;
   (*cursor).AddToWeight(wgtadd);

}

//______________________________________________________________________________
void SEIdAltL::NormalizeWeights(Float_t /* wgtsum */)
{
   // normalize the weights so sum adds up to "wgtsum"
   // if all values are exactly zero, then this sets them
   // to wgtsum/fSize

   unsigned int n=size();

   if (n < 1) return;

   Float_t sum = 0.0;

   SEIdAltLIter iter, the_end=this->end();

   iter = this->begin();
   while (iter != the_end) { sum += (*iter).GetWeight(); iter++; }

   iter = this->begin();
   if ( sum != 0.0 ) {
      Float_t scale = 1.0/sum;
      while (iter != the_end) {
         Float_t wgt = (*iter).GetWeight() * scale;
         (*iter).SetWeight(wgt);
         iter++;
      }
   } else {
      // all weight values were zero
      Float_t equalwgt = 1.0/(float)n;
      while (iter != the_end) { (*iter).SetWeight(equalwgt); iter++; }
   }

}

//______________________________________________________________________________

void SEIdAltL::TestConsistency() const
{
   // Set the fError flag if the list is inconsistent
   // in terms of GetDetector, GetEnd, GetPlane, GetPlaneView

   // skip if already checked (adding values clears this flag)
   if ( kUnchecked != fError ) return;

   // clear all errors
   fError = 0;

   unsigned int n = size();

   // a single entry is consistent with itself
   if ( 1 == n ) return;

   // an empty list is consistent with nothing
   // and asking for these value will be problematic
   if ( 0 == n ) {
      fError = kBadDetector | kBadEnd | kBadPlane | kBadPlaneView;
      return;
   }

   SEIdAltLConstIter iter = this->begin();
   SEIdAltLConstIter the_end = this->end();

   /* Int_t first_seid = */ (*iter).GetSEId();

   iter++;  // no need to recheck the first against itself
   while (iter != the_end) {
      /* Int_t alt_seid = */ (*iter).GetSEId();
      iter++;
   }
   
   if ( fError != kOkay ) {
      cout 
         << "SEIdAltL::TestConsistency - list is inconsistent " 
         << endl;
      this->Print();
   }

}
//______________________________________________________________________________
