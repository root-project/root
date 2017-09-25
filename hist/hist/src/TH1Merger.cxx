// Helper clas implementing some of the TH1 functionality

#include "TH1Merger.h"
#include "TH1.h"
#include "TH2.h"
#include "TH3.h"
#include "TAxis.h"
#include "TError.h"
#include "THashList.h"
#include "TClass.h"
#include <iostream>


Bool_t TH1Merger::AxesHaveLimits(const TH1 * h) {
   Bool_t hasLimits = h->GetXaxis()->GetXmin() < h->GetXaxis()->GetXmax();
   if (h->GetDimension() > 1) hasLimits &=  h->GetYaxis()->GetXmin() < h->GetYaxis()->GetXmax();
   if (h->GetDimension() > 2) hasLimits &=  h->GetZaxis()->GetXmin() < h->GetZaxis()->GetXmax();
   return hasLimits; 
}

/// Function performing the actual merge
Bool_t TH1Merger::operator() () {

   
   EMergerType type = ExamineHistograms();

   if (gDebug) Info("Merge","Histogram Merge type is %d and new axis flag is %d",(int) type,(int) fNewAxisFlag);

   if (type == kNotCompatible) return kFALSE;

   if (type == kAllSameAxes)
      return SameAxesMerge();

   if (type == kAllLabel)
      return LabelMerge();

   if (type == kAllNoLimits)
      return BufferMerge();

   // this is the mixed case - more complicated 
   if (type == kHasNewLimits) {
      // we need to define some new axes     
      DefineNewAxes();
      // we might need to merge some histogram using the buffer
      Bool_t ret =  BufferMerge();
      // if ret is true the merge is completed and we can exit 
      if (ret) return kTRUE; 
      // in the other cases then we merge using FindBin
      return DifferentAxesMerge(); 
   }
   Error("TH1Merger","Unknown type of Merge for histogram %s",fH0->GetName());
   return kFALSE;
}

/**
   Examine the list of histograms to find out which type of Merge we need to do
   Pass the input list containing the histogram to merge and h0 which is the initial histogram 
   on which all the histogram of the list will be merged 
   This are the possible cases:
    - 1. All histogram have the same axis (allSameLimits = true)
    - 2. Histogram have different axis but compatible  (allSameLimits = false) and sameLimitsX,Y,Z specifies which axis
         has different limits
    - 3. Histogram do not have limits (so the Buffer is used)  allHaveLimits = false
    - 3b. One histogram has limits the other not : allHaveLimits = false AND initialLimitsFound = true
    - 4. Histogram Have labels  = allHaveLabels = true
   

*/
TH1Merger::EMergerType TH1Merger::ExamineHistograms() {



   Bool_t initialLimitsFound = kFALSE;
   Bool_t allHaveLabels = kTRUE;  // assume all histo have labels and check later
   Bool_t allHaveLimits = kTRUE;
   Bool_t allSameLimits = kTRUE;
   Bool_t sameLimitsX = kTRUE;
   Bool_t sameLimitsY = kTRUE;
   Bool_t sameLimitsZ = kTRUE;
   Bool_t foundLabelHist = kFALSE;
   Bool_t haveWeights = kFALSE; 
   
   // TAxis newXAxis;
   // TAxis newYAxis;
   // TAxis newZAxis;

   TIter next(&fInputList);
   TH1 * h = fH0;  // start with fH0

   int dimension = fH0->GetDimension(); 

   // start looping on the histograms 

   do  {

      // check first histogram compatibility
      if (h != fH0) {
         if (h->GetDimension() != dimension) {
            Error("Merge", "Cannot merge histogram - dimensions are different\n "
                  "%s has dim=%d and %s has dim=%d",fH0->GetName(),dimension,h->GetName(),h->GetDimension());
            return kNotCompatible; 
         }
      }

      // check if one of the histogram is weighted
      haveWeights |= h->GetSumw2N() != 0;

      // do not skip anymore empty histograms
      // since are used to set the limits
      Bool_t hasLimits = TH1Merger::AxesHaveLimits(h);
      allHaveLimits = allHaveLimits && hasLimits;
      allSameLimits &= allHaveLimits;

      
      if (hasLimits) {
         h->BufferEmpty();

//          // this is done in case the first histograms are empty and
//          // the histogram have different limits
// #ifdef LATER
//          if (firstHistWithLimits ) {
//             // set axis limits in the case the first histogram did not have limits
//             if (h != this && !SameLimitsAndNBins( fXaxis, *h->GetXaxis()) ) {
//               if (h->GetXaxis()->GetXbins()->GetSize() != 0) fXaxis.Set(h->GetXaxis()->GetNbins(), h->GetXaxis()->GetXbins()->GetArray());
//               else                                           fXaxis.Set(h->GetXaxis()->GetNbins(), h->GetXaxis()->GetXmin(), h->GetXaxis()->GetXmax());
//             }
//             firstHistWithLimits = kFALSE;
//          }
// #endif

         // this is executed the first time an histogram with limits is found
         // to set some initial values on the new axis
         if (!initialLimitsFound) {
            initialLimitsFound = kTRUE;
            if (h->GetXaxis()->GetXbins()->GetSize() != 0)
               fNewXAxis.Set(h->GetXaxis()->GetNbins(), h->GetXaxis()->GetXbins()->GetArray());
            else
               fNewXAxis.Set(h->GetXaxis()->GetNbins(), h->GetXaxis()->GetXmin(), h->GetXaxis()->GetXmax());
            if (dimension > 1) {
               if (h->GetYaxis()->GetXbins()->GetSize() != 0)
                  fNewYAxis.Set(h->GetYaxis()->GetNbins(), h->GetYaxis()->GetXbins()->GetArray());
               else
                  fNewYAxis.Set(h->GetYaxis()->GetNbins(), h->GetYaxis()->GetXmin(), h->GetYaxis()->GetXmax());
            }
            if (dimension > 2) {
               if (h->GetZaxis()->GetXbins()->GetSize() != 0)
                  fNewZAxis.Set(h->GetZaxis()->GetNbins(), h->GetZaxis()->GetXbins()->GetArray());
               else
                  fNewZAxis.Set(h->GetZaxis()->GetNbins(), h->GetZaxis()->GetXmin(), h->GetZaxis()->GetXmax());

            }
         }
         else {
            // check first if histograms have same bins in X
            if (!TH1::SameLimitsAndNBins(fNewXAxis, *(h->GetXaxis())) ) {
               sameLimitsX = kFALSE;
               // recompute the limits in this case the optimal limits
               // The condition to works is that the histogram have same bin with
               // and one common bin edge
               if (!TH1::RecomputeAxisLimits(fNewXAxis, *(h->GetXaxis()))) {
                  Error("Merge", "Cannot merge histograms - limits are inconsistent:\n "
                        "first: %s (%d, %f, %f), second: %s (%d, %f, %f)", fH0->GetName(),
                        fNewXAxis.GetNbins(), fNewXAxis.GetXmin(), fNewXAxis.GetXmax(), 
                        h->GetName(),h->GetXaxis()->GetNbins(), h->GetXaxis()->GetXmin(),
                        h->GetXaxis()->GetXmax());
                  return kNotCompatible;
               }
            }
            // check first if histograms have same bins in Y
            if (dimension > 1 && !TH1::SameLimitsAndNBins(fNewYAxis, *(h->GetYaxis()))) {
             sameLimitsY = kFALSE;
             // recompute in this case the optimal limits
             // The condition to works is that the histogram have same bin with
             // and one common bin edge
             if (!TH1::RecomputeAxisLimits(fNewYAxis, *(h->GetYaxis()))) {
               Error("Merge", "Cannot merge histograms - limits are inconsistent:\n "
                     "first: %s (%d, %f, %f), second: %s (%d, %f, %f)", fH0->GetName(),
                     fNewYAxis.GetNbins(), fNewYAxis.GetXmin(), fNewYAxis.GetXmax(),
                     h->GetName(), h->GetYaxis()->GetNbins(), h->GetYaxis()->GetXmin(),
                     h->GetYaxis()->GetXmax());
               return kNotCompatible;
             }
           }
           if(dimension > 2 && !fH0->SameLimitsAndNBins(fNewZAxis, *(h->GetZaxis()))) {
             sameLimitsZ = kFALSE;
             if (!TH1::RecomputeAxisLimits(fNewZAxis, *(h->GetZaxis()))) {
               Error("Merge", "Cannot merge histograms - limits are inconsistent:\n "
                     "first: %s (%d, %f, %f), second: %s (%d, %f, %f)", fH0->GetName(),
                     fNewZAxis.GetNbins(), fNewZAxis.GetXmin(), fNewZAxis.GetXmax(),
                     h->GetName(),h->GetZaxis()->GetNbins(), h->GetZaxis()->GetXmin(),
                     h->GetZaxis()->GetXmax());
               return kNotCompatible;
             }
           }
           allSameLimits = sameLimitsX && sameLimitsY && sameLimitsZ;


         }
      }
      Bool_t histoIsEmpty = h->IsEmpty(); 
      // std::cout << "considering histo " << h->GetName() << "  labels - " << allHaveLabels << " is empty "
      //           <<  histoIsEmpty << std::endl;

      // if histogram is empty it does not matter if it has label or not 
      if (allHaveLabels && !histoIsEmpty) {
         THashList* hlabels=h->GetXaxis()->GetLabels();
         Bool_t haveOneLabel = (hlabels != nullptr);
         // do here to print message only one time
         if (foundLabelHist && allHaveLabels && !haveOneLabel) {
            Warning("Merge","Not all histograms have labels. I will ignore labels,"
            " falling back to bin numbering mode.");
         }

         allHaveLabels &= (haveOneLabel);
         // for the error message
         if (haveOneLabel) foundLabelHist = kTRUE;

         if (foundLabelHist && gDebug)
            Info("TH1Merger::ExamineHistogram","Histogram %s has labels",h->GetName() );
         
         // If histograms have labels but CanExtendAllAxes() is false
         // use bin center mode
         if (allHaveLabels && !fH0->CanExtendAllAxes())  {
            // special case for this histogram when is empty
            // and axis cannot be extended (because it is the default)
            if ( fH0->IsEmpty()  ) {
                if (gDebug) 
                   Info("TH1Merger::ExamineHistogram","Histogram %s to be merged is empty and we are merging with %s that has labels. Force the axis to be extended",fH0->GetName(),h->GetName());
               UInt_t bitMaskX = fH0->GetXaxis()->CanBeAlphanumeric() & TH1::kXaxis;
               UInt_t bitMaskY = (fH0->GetYaxis()->CanBeAlphanumeric() << 1 ) & TH1::kYaxis;
               UInt_t bitMaskZ = (fH0->GetZaxis()->CanBeAlphanumeric() << 2 ) & TH1::kZaxis; 
               fH0->SetCanExtend(bitMaskX | bitMaskY | bitMaskZ );
            }
            if (!fH0->CanExtendAllAxes()) {
               if (gDebug) 
                  Info("TH1Merger::ExamineHistogram","Histogram %s to be merged has label but axis cannot be extended - using bin numeric mode to merge. Call TH1::SetExtendAllAxes() if want to merge using label mode",fH0->GetName());
               allHaveLabels = kFALSE;
            }
         }
         // I could add a check if histogram contains bins without a label
         // and with non-zero bin content
         // Do we want to support this ???
         // only in case the !h->CanExtendAllAxes()
         if (allHaveLabels && !h->CanExtendAllAxes()) {
            // count number of bins with non-null content
            Int_t non_zero_bins = 0;
            Int_t nbins = h->GetXaxis()->GetNbins();
            if (nbins > hlabels->GetEntries() ) {
               for (Int_t i = 1; i <= nbins; i++) {
                  if (h->RetrieveBinContent(i) != 0 || (fH0->fSumw2.fN && h->GetBinError(i) != 0) ) {
                     non_zero_bins++;
                  }
               }
               if (non_zero_bins > hlabels->GetEntries() ) {
                  Warning("TH1Merger::ExamineHistograms","Histogram %s contains non-empty bins without labels - falling back to bin numbering mode",h->GetName() );
                  allHaveLabels = kFALSE;
               }
            }
         }
      }
      if (gDebug)
         Info("TH1Merger::ExamineHistogram","Examine histogram %s - labels %d - same limits %d - axis found %d",h->GetName(),allHaveLabels,allSameLimits,initialLimitsFound );

   }    while ( ( h = dynamic_cast<TH1*> ( next() ) ) != NULL );

   if (!h && (*next) ) {
      Error("Merge","Attempt to merge object of class: %s to a %s",
            (*next)->ClassName(),fH0->ClassName());
      return kNotCompatible;
   }

   // in case of weighted histogram set Sumw2() on fH0 is is not weighted
   if (haveWeights && fH0->GetSumw2N() == 0) fH0->Sumw2(); 
   
   // return the type of merge
   if (allHaveLabels) return kAllLabel;
   if (allSameLimits) return kAllSameAxes;
   if (!initialLimitsFound) {
      R__ASSERT(!allHaveLimits);
      // case where no limits are found and the buffer is used
      return kAllNoLimits;
   }
   // remaining case should be the mixed one. Some histogram have limits some not
   fNewAxisFlag = 0;
   if (!sameLimitsX) fNewAxisFlag |= TH1::kXaxis;
   if (!sameLimitsY) fNewAxisFlag |= TH1::kYaxis;
   if (!sameLimitsZ) fNewAxisFlag |= TH1::kZaxis;

   // we need to set the flag also in case this histogram has no limits
   // we need to set explicitly the flag to re-define  a new axis
   if (fH0->GetXaxis()->GetXmin() >= fH0->GetXaxis()->GetXmax()) fNewAxisFlag |= TH1::kXaxis;
   if (dimension > 1 && fH0->GetYaxis()->GetXmin() >= fH0->GetYaxis()->GetXmax()) fNewAxisFlag |= TH1::kYaxis;
   if (dimension > 2 && fH0->GetZaxis()->GetXmin() >= fH0->GetZaxis()->GetXmax()) fNewAxisFlag |= TH1::kZaxis;

   
   return kHasNewLimits; 
   
}

/**
   Function to define new histogram axis when merging
   It is call only in case of merging with different axis or with the 
   buffer  (kHasNewLimits)
*/

void TH1Merger::DefineNewAxes() { 

   // first we need to create a copy of the histogram in case is not empty

   if (!fH0->IsEmpty() ) {
      Bool_t mustCleanup = fH0->TestBit(kMustCleanup);
      if (mustCleanup) fH0->ResetBit(kMustCleanup);
      fHClone = (TH1*)fH0->IsA()->New();
      fHClone->SetDirectory(0);
      fH0->Copy(*fHClone);
      if (mustCleanup) fH0->SetBit(kMustCleanup);
      fH0->BufferEmpty(1);         // To remove buffer.
      fH0->Reset();                // BufferEmpty sets limits so we can't use it later.
      fH0->SetEntries(0);
      fInputList.AddFirst(fHClone);

   }

   bool newLimitsX = (fNewAxisFlag & TH1::kXaxis);
   bool newLimitsY = (fNewAxisFlag & TH1::kYaxis);
   bool newLimitsZ = (fNewAxisFlag & TH1::kZaxis);
   if (newLimitsX) {
      fH0->fXaxis.SetRange(0,0);
      if (fNewXAxis.GetXbins()->GetSize() != 0)
          fH0->fXaxis.Set(fNewXAxis.GetNbins(),fNewXAxis.GetXbins()->GetArray());
       else
          fH0->fXaxis.Set(fNewXAxis.GetNbins(),fNewXAxis.GetXmin(), fNewXAxis.GetXmax());
   }
   if (newLimitsY) {
      fH0->fYaxis.SetRange(0,0);
      if (fNewYAxis.GetXbins()->GetSize() != 0)
          fH0->fYaxis.Set(fNewYAxis.GetNbins(),fNewYAxis.GetXbins()->GetArray());
       else
          fH0->fYaxis.Set(fNewYAxis.GetNbins(),fNewYAxis.GetXmin(), fNewYAxis.GetXmax());
   }
   if (newLimitsZ) {
      fH0->fZaxis.SetRange(0,0);
      if (fNewZAxis.GetXbins()->GetSize() != 0)
          fH0->fZaxis.Set(fNewZAxis.GetNbins(),fNewZAxis.GetXbins()->GetArray());
       else
          fH0->fZaxis.Set(fNewZAxis.GetNbins(),fNewZAxis.GetXmin(), fNewZAxis.GetXmax());
   }

   // we need to recompute fNcells and set the array size (as in TH1::SetBins)
   fH0->fNcells = fH0->fXaxis.GetNbins()+2;
   if (fH0->fDimension > 1) fH0->fNcells *= fH0->fYaxis.GetNbins()+2;
   if (fH0->fDimension > 2) fH0->fNcells *= fH0->fZaxis.GetNbins()+2;
   fH0->SetBinsLength(fH0->fNcells);
   if (fH0->fSumw2.fN) fH0->fSumw2.Set(fH0->fNcells);
   // set dummy Y and Z axis for lower dim histogras
   if (fH0->fDimension < 3)  fH0->fZaxis.Set(1,0,1);
   if (fH0->fDimension < 2)  fH0->fYaxis.Set(1,0,1);

   if (gDebug) {
      if (newLimitsX) Info("DefineNewAxis","A new X axis has been defined Nbins=%d , [%f,%f]", fH0->fXaxis.GetNbins(),
                           fH0->fXaxis.GetXmin(), fH0->fXaxis.GetXmax() );
      if (newLimitsY) Info("DefineNewAxis","A new Y axis has been defined Nbins=%d , [%f,%f]", fH0->fYaxis.GetNbins(),
                           fH0->fYaxis.GetXmin(), fH0->fYaxis.GetXmax() );
      if (newLimitsZ) Info("DefineNewAxis","A new Z axis has been defined Nbins=%d , [%f,%f]", fH0->fZaxis.GetNbins(),
                           fH0->fZaxis.GetXmin(), fH0->fZaxis.GetXmax() );
   }

   return;

}

Bool_t TH1Merger::BufferMerge() { 

   TIter next(&fInputList); 
   while (TH1* hist = (TH1*)next()) {
      // support also case where some histogram have limits and some have the buffer
      if ( !TH1Merger::AxesHaveLimits(hist) && hist->fBuffer  ) {

         if (gDebug)
            Info("TH1Merger::BufferMerge","Merging histogram %s into %s",hist->GetName(), fH0->GetName() );


         // case of no limits
         // Entries from buffers have to be filled one by one
         // because FillN doesn't resize histograms.
         Int_t nbentries = (Int_t)hist->fBuffer[0];
         if (fH0->fDimension  == 1) {
            for (Int_t i = 0; i < nbentries; i++)
               fH0->Fill(hist->fBuffer[2*i + 2], hist->fBuffer[2*i + 1]);
         }
         if (fH0->fDimension == 2) {
            auto h2 = dynamic_cast<TH2*>(fH0);
            R__ASSERT(h2);
            for (Int_t i = 0; i < nbentries; i++)
               h2->Fill(hist->fBuffer[3*i + 2], hist->fBuffer[3*i + 3],hist->fBuffer[3*i + 1] );
         }
         if (fH0->fDimension == 3) {
            auto h3 = dynamic_cast<TH3*>(fH0);
            R__ASSERT(h3);
            for (Int_t i = 0; i < nbentries; i++)
               h3->Fill(hist->fBuffer[4*i + 2], hist->fBuffer[4*i + 3],hist->fBuffer[4*i + 4], hist->fBuffer[4*i + 1] );
         }
         fInputList.Remove(hist);
      }
   }
   // return true if the merge is completed
   if (fInputList.GetSize() == 0) {
      // all histo have been merged
      return kTRUE; 
   }
   // we need to reset the buffer in case of merging later on
   // is this really needed ???
   if (fH0->fBuffer) fH0->BufferEmpty(1);
   
   return kFALSE; 
}

Bool_t TH1Merger::SameAxesMerge() { 


   Double_t stats[TH1::kNstat], totstats[TH1::kNstat];
   for (Int_t i=0;i<TH1::kNstat;i++) {
      totstats[i] = stats[i] = 0;
   }
   fH0->GetStats(totstats);
   Double_t nentries = fH0->GetEntries();
   
   TIter next(&fInputList); 
   while (TH1* hist=(TH1*)next()) {
      // process only if the histogram has limits; otherwise it was processed before
      // in the case of an existing buffer (see if statement just before)

      if (gDebug)
         Info("TH1Merger::SameAxesMerge","Merging histogram %s into %s",hist->GetName(), fH0->GetName() );

      // skip empty histograms
      if (hist->IsEmpty()) continue; 

      // import statistics
      hist->GetStats(stats);
      for (Int_t i=0; i<TH1::kNstat; i++)
         totstats[i] += stats[i];
      nentries += hist->GetEntries();

         //Int_t nx = hist->GetXaxis()->GetNbins();
         // loop on bins of the histogram and do the merge
      for (Int_t ibin = 0; ibin < hist->fNcells; ibin++) {

         Double_t cu = hist->RetrieveBinContent(ibin);
         Double_t e1sq = TMath::Abs(cu);
         if (fH0->fSumw2.fN) e1sq= hist->GetBinErrorSqUnchecked(ibin);

         fH0->AddBinContent(ibin,cu);
         if (fH0->fSumw2.fN) fH0->fSumw2.fArray[ibin] += e1sq;

      }
   }
   //copy merged stats
   fH0->PutStats(totstats);
   fH0->SetEntries(nentries);

   return kTRUE;
}


/**
   Merged histogram when axis can be different. 
   Histograms are merged looking at bin center positions
   
 */
Bool_t TH1Merger::DifferentAxesMerge() { 
   
   Double_t stats[TH1::kNstat], totstats[TH1::kNstat];
   for (Int_t i=0;i<TH1::kNstat;i++) {totstats[i] = stats[i] = 0;}
   fH0->GetStats(totstats);
   Double_t nentries = fH0->GetEntries();

   TIter next(&fInputList); 
   while (TH1* hist=(TH1*)next()) {

      if (gDebug)
         Info("TH1Merger::DifferentAxesMerge","Merging histogram %s into %s",hist->GetName(), fH0->GetName() );
      
      // skip empty histograms
      if (hist->IsEmpty()) continue; 

      // import statistics
      hist->GetStats(stats);
      for (Int_t i=0;i<TH1::kNstat;i++)
         totstats[i] += stats[i];
      nentries += hist->GetEntries();

      // loop on bins of the histogram and do the merge
      for (Int_t ibin = 0; ibin < hist->fNcells; ibin++) {

         Double_t cu = hist->RetrieveBinContent(ibin);
         Double_t e1sq = TMath::Abs(cu);
         if (fH0->fSumw2.fN) e1sq= hist->GetBinErrorSqUnchecked(ibin);

         // if bin is empty we can skip it
         if (cu == 0 && e1sq == 0) continue;

         Int_t binx,biny,binz;
         hist->GetBinXYZ(ibin, binx, biny, binz);

         // case of underflow/overflows in the histogram being merged
         if (binx <= 0 || binx >= hist->GetNbinsX() + 1) {
            if (fH0->fXaxis.CanExtend() || ( hist->fXaxis.GetBinCenter(binx) > fH0->fXaxis.GetXmin() && hist->fXaxis.GetBinCenter(binx) < fH0->fXaxis.GetXmax()) ) {
               Error("TH1Merger::DifferentAxesMerge", "Cannot merge histograms - the histograms %s can extend the X axis or have"
                     " different limits and underflows/overflows are present in the histogram %s.",fH0->GetName(),hist->GetName());
                  return kFALSE;
            }
         }
         if (biny <= 0 || biny >= hist->GetNbinsY() + 1) {
            if (fH0->fYaxis.CanExtend() || ( hist->fYaxis.GetBinCenter(biny) > fH0->fYaxis.GetXmin() && hist->fYaxis.GetBinCenter(biny) < fH0->fYaxis.GetXmax()) ) {
               Error("TH1Merger::DifferentAxesMerge", "Cannot merge histograms - the histograms %s can extend the Y axis or have"
                     " different limits and underflows/overflows are present in the histogram %s.",fH0->GetName(),hist->GetName());
                  return kFALSE;
            }
         }
         if (binz <= 0 || binz >= hist->GetNbinsZ() + 1) {
            if (fH0->fZaxis.CanExtend() || ( hist->fZaxis.GetBinCenter(binz) > fH0->fZaxis.GetXmin() && hist->fZaxis.GetBinCenter(binz) < fH0->fZaxis.GetXmax()) ) {
               Error("TH1Merger::DifferentAxesMerge", "Cannot merge histograms - the histograms %s can extend the Z axis or have"
                     " different limits and underflows/overflows are present in the histogram %s.",fH0->GetName(),hist->GetName());
                  return kFALSE;
            }
         }

         Int_t ix = 0;
         Int_t iy = 0;
         Int_t iz = 0;

         // we can extend eventually the axis if histogram is capable of doing it
         // by using FindBin
          ix = fH0->fXaxis.FindBin(hist->GetXaxis()->GetBinCenter(binx));
          if (fH0->fDimension > 1)
             iy = fH0->fYaxis.FindBin(hist->GetYaxis()->GetBinCenter(biny));
          if (fH0->fDimension > 2)
             iz = fH0->fZaxis.FindBin(hist->GetZaxis()->GetBinCenter(binz));

         Int_t ib = fH0->GetBin(ix,iy,iz); 
         if (ib < 0 || ib > fH0->fNcells) {
            Fatal("TH1Merger::LabelMerge","Fatal error merging histogram %s - bin number is %d and array size is %d",
                  fH0->GetName(), ib,fH0->fNcells);
         }

          fH0->AddBinContent(ib,cu);
          if (fH0->fSumw2.fN) fH0->fSumw2.fArray[ib] += e1sq;
      }
   }
   //copy merged stats
   fH0->PutStats(totstats);
   fH0->SetEntries(nentries);

   return kTRUE;
}


/**
   Merge histograms with labels 
*/
Bool_t TH1Merger::LabelMerge() { 

   
   Double_t stats[TH1::kNstat], totstats[TH1::kNstat];
   for (Int_t i=0;i<TH1::kNstat;i++) {totstats[i] = stats[i] = 0;}
   fH0->GetStats(totstats);
   Double_t nentries = fH0->GetEntries();

   TIter next(&fInputList); 
   while (TH1* hist=(TH1*)next()) {

      if (gDebug)
         Info("TH1Merger::LabelMerge","Merging histogram %s into %s",hist->GetName(), fH0->GetName() );
      
      // skip empty histograms
      if (hist->IsEmpty()) continue; 

      // import statistics
      hist->GetStats(stats);
      for (Int_t i=0;i<TH1::kNstat;i++)
         totstats[i] += stats[i];
      nentries += hist->GetEntries();

      auto labelsX = hist->GetXaxis()->GetLabels();
      auto labelsY = hist->GetYaxis()->GetLabels();
      auto labelsZ = hist->GetZaxis()->GetLabels();
      R__ASSERT(!( labelsX == nullptr  && labelsY == nullptr && labelsZ == nullptr));

      // loop on bins of the histogram and do the merge
      for (Int_t ibin = 0; ibin < hist->fNcells; ibin++) {

         Double_t cu = hist->RetrieveBinContent(ibin);
         Double_t e1sq = cu;
         if (fH0->fSumw2.fN) e1sq= hist->GetBinErrorSqUnchecked(ibin);

         // if bin is empty we can skip it
         if (cu == 0 && e1sq == 0) continue; 
         
         Int_t binx,biny,binz;
         hist->GetBinXYZ(ibin, binx, biny, binz);

         // here only in the case of bins with labels
         const char * labelX = 0; 
         const char * labelY = 0; 
         const char * labelZ = 0; 
         labelX=hist->GetXaxis()->GetBinLabel(binx);
         if (fH0->fDimension > 1) labelY = hist->GetYaxis()->GetBinLabel(biny);
         if (fH0->fDimension > 2) labelZ = hist->GetYaxis()->GetBinLabel(binz);
         // do we need to support case when there are bins with labels and bins without them ??
         // this case should have been detected before when examining the histograms 
         

         Int_t ix = -1;
         Int_t iy = (fH0->fDimension > 1) ? -1 : 0;
         Int_t iz = (fH0->fDimension > 2) ? -1 : 0; 

         // special case for underflow/overflows which have normally empty labels 
         if (binx == 0 && TString(labelX) == "" ) ix = 0;
         if (binx == hist->fXaxis.GetNbins() +1 && TString(labelX) == "" ) ix = fH0->fXaxis.GetNbins() +1;
         if (fH0->fDimension > 1 ) {
            if (biny == 0 && TString(labelY) == "" ) iy = 0;
            if (biny == hist->fYaxis.GetNbins() +1 && TString(labelY) == "" ) iy = fH0->fYaxis.GetNbins() +1;
         }
         if (fH0->fDimension > 2 ) {
            if (binz == 0 && TString(labelZ) == "" ) iz = 0;
            if (binz == hist->fZaxis.GetNbins() +1 && TString(labelZ) == "" ) iz = fH0->fZaxis.GetNbins() +1;
         }



         // find corresponding case (in case bin is not overflow)
         if (ix == -1) {
            if (labelsX)
               ix = fH0->fXaxis.FindBin(labelX);
            else
               ix = FindFixBinNumber(binx, hist->fXaxis, fH0->fXaxis);
         }

         if (iy == -1 && fH0->fDimension> 1 ) { // check on dim should not be needed
            if (labelsY)
               iy= fH0->fYaxis.FindBin(labelY);
            else 
               iy = FindFixBinNumber(biny, hist->fYaxis, fH0->fYaxis);
         }
         if (iz == -1 && fH0->fDimension> 2)  {
            if (labelsZ)
               iz= fH0->fZaxis.FindBin(labelZ);
            else 
               iz = FindFixBinNumber(binz, hist->fZaxis, fH0->fZaxis);
         }

         if (gDebug)
            Info("TH1Merge::LabelMerge","Merge bin [%d,%d,%d] with label [%s,%s,%s] into bin [%d,%d,%d]",
                 binx,biny,binz,labelX,labelY,labelZ,ix,iy,iz);

         Int_t ib = fH0->GetBin(ix,iy,iz); 
         if (ib < 0 || ib >= fH0->fNcells) {
            Fatal("TH1Merger::LabelMerge","Fatal error merging histogram %s - bin number is %d and array size is %d",
                  fH0->GetName(), ib,fH0->fNcells);
         }

          fH0->AddBinContent(ib,cu);
          if (fH0->fSumw2.fN) fH0->fSumw2.fArray[ib] += e1sq;
      }
   }
   //copy merged stats
   fH0->PutStats(totstats);
   fH0->SetEntries(nentries);

   return kTRUE;
}
