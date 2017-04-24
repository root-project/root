#include "TProfile2Poly.h"
#include "TProfileHelper.h"

#include "TMultiGraph.h"
#include "TGraph.h"
#include "TClass.h"
#include "TList.h"
#include "TMath.h"

ClassImp(TProfile2Poly)

   // -------------- TProfile2PolyBin  --------------

   TProfile2PolyBin::TProfile2PolyBin()
{
   fSumw = 0;
   fSumw2 = 0;
   fSumwz = 0;
   fSumwz2 = 0;
   fNumEntries = 0;
   fError = 0;
   fAverage = 0;
}

TProfile2PolyBin::TProfile2PolyBin(TObject *poly, Int_t bin_number) : TH2PolyBin(poly, bin_number)
{
   fSumw = 0;
   fSumw2 = 0;
   fSumwz = 0;
   fSumwz2 = 0;
   fNumEntries = 0;
   fError = 0;
   fAverage = 0;
}

void TProfile2PolyBin::Update()
{
   UpdateAverage();
   UpdateError();
   SetChanged(true);
}

void TProfile2PolyBin::UpdateAverage()
{
   fAverage = fSumw / fNumEntries;
}

void TProfile2PolyBin::UpdateError()
{
   fError = std::sqrt((fSumw2 / fNumEntries) - (fContent * fContent));
}

void TProfile2PolyBin::ClearStats()
{
   fSumw = 0;
   fSumw2 = 0;
   fSumwz = 0;
   fSumwz2 = 0;
   fNumEntries = 0;
   fError = 0;
   fAverage = 0;
}

// -------------- TProfile2Poly  --------------

TProfile2Poly::TProfile2Poly(const char *name, const char *title, Double_t xlow, Double_t xup, Double_t ylow,
                             Double_t yup)
   : TH2Poly(name, title, xlow, xup, ylow, yup)
{
}

TProfile2Poly::TProfile2Poly(const char *name, const char *title, Int_t nX, Double_t xlow, Double_t xup, Int_t nY,
                             Double_t ylow, Double_t yup)
   : TH2Poly(name, title, nX, xlow, xup, nY, ylow, yup)
{
}

TProfile2PolyBin *TProfile2Poly::CreateBin(TObject *poly)
{
   if (!poly) return 0;

   if (fBins == 0) {
      fBins = new TList();
      fBins->SetOwner();
   }

   fNcells++;
   Int_t ibin = fNcells - kNOverflow;
   return new TProfile2PolyBin(poly, ibin);
}

Int_t TProfile2Poly::Fill(Double_t xcoord, Double_t ycoord, Double_t value)
{
   return Fill(xcoord, ycoord, value, 1);
}

Int_t TProfile2Poly::Fill(Double_t xcoord, Double_t ycoord, Double_t value, Double_t weight)
{

   // TODO: is this correct?
   if (fNcells <= kNOverflow) return 0;
   Int_t overflow = 0;
   if (ycoord > fYaxis.GetXmax())
      overflow += -1;
   else if (ycoord > fYaxis.GetXmin())
      overflow += -4;
   else
      overflow += -7;
   if (xcoord > fXaxis.GetXmax())
      overflow += -2;
   else if (xcoord > fXaxis.GetXmin())
      overflow += -1;
   if (overflow != -5) {
      fOverflow[-overflow - 1] += weight;
      if (fSumw2.fN) fSumw2.fArray[-overflow - 1] += weight * weight;
      return overflow;
   }

   // Finds the cell (x,y) coordinates belong to
   Int_t n = (Int_t)(floor((xcoord - fXaxis.GetXmin()) / fStepX));
   Int_t m = (Int_t)(floor((ycoord - fYaxis.GetXmin()) / fStepY));

   // Make sure the array indices are correct.
   if (n >= fCellX) n = fCellX - 1;
   if (m >= fCellY) m = fCellY - 1;
   if (n < 0) n = 0;
   if (m < 0) m = 0;

   // TODO: is this correct?
   if (fIsEmpty[n + fCellX * m]) {
      fOverflow[4] += weight;
      if (fSumw2.fN) fSumw2.fArray[4] += weight * weight;
      return -5;
   }

   // ------------ Update global (per histo) statistics
   fTsumw += weight;
   fTsumw2 += weight * weight;
   fTsumwx += weight * xcoord;
   fTsumwx2 += weight * xcoord * xcoord;
   fTsumwy += weight * ycoord;
   fTsumwy2 += weight * ycoord * ycoord;
   fTsumwxy += weight * xcoord * ycoord;
   fTsumwz += weight * value;
   fTsumwz2 += weight * value * value;

   // ------------ Update local (per bin) statistics
   TProfile2PolyBin *bin;
   TIter next(&fCells[n + fCellX * m]);
   TObject *obj;
   while ((obj = next())) {
      bin = (TProfile2PolyBin *)obj;
      if (bin->IsInside(xcoord, ycoord)) {
         fEntries++;
         bin->SetFNumEntries(bin->GetFNumEntries() + 1);
         bin->SetFSumw(bin->GetFSumw() + value);
         bin->SetFSumw2(bin->GetFSumw2() + (value * value));
         bin->SetFSumwz(bin->GetFSumwz() + (value * weight));

         bin->Update();
         bin->SetContent(bin->GetAverage());

         return bin->GetBinNumber();
      }
   }

   fOverflow[4] += weight;
   if (fSumw2.fN) fSumw2.fArray[4] += weight * weight * value;
   return -5;
}

Long64_t TProfile2Poly::Merge(TCollection *in)
{
   Int_t size = in->GetSize();

   std::vector<TProfile2Poly *> list;
   list.reserve(size);

   for (int i = 0; i < size ; i++) {
      list.push_back((TProfile2Poly *)((TList *)in)->At(i));
   }
   return this->Merge(list);
}

Long64_t TProfile2Poly::Merge(std::vector<TProfile2Poly *> list)
{
   if (list.size() == 0) {
      std::cout << "[FAIL] TProfile2Poly::Merge: No objects to be merged " << std::endl;
      return -1;
   }

   // ------------ Check that bin numbers of TP2P's to be merged are equal
   std::set<Int_t> numBinUnique;
   for (const auto &histo : list) {
      numBinUnique.insert(histo->fBins->GetSize());
   }
   if (numBinUnique.size() != 1) {
      std::cout << "[FAIL] TProfile2Poly::Merge: Bin numbers of TProfile2Polys to be merged differ!" << std::endl;
      return -1;
   }
   Int_t nbins = *numBinUnique.begin();

   // ------------ Update global (per histo) statistics
   for (const auto &histo : list) {
      this->fEntries += histo->fEntries;
      this->fTsumw += histo->fTsumw;
      this->fTsumw2 += histo->fTsumw2;
      this->fTsumwx += histo->fTsumwx;
      this->fTsumwx2 += histo->fTsumwx2;
      this->fTsumwy += histo->fTsumwy;
      this->fTsumwy2 += histo->fTsumwy2;
      this->fTsumwxy += histo->fTsumwxy;
      this->fTsumwz += histo->fTsumwz;
      this->fTsumwz2 += histo->fTsumwz2;

      // Merge overflow bins
      for (Int_t i = 0; i < 9; ++i) {
         this->fOverflow[i] += histo->GetOverflowContent(i);
      }
   }

   // ------------ Update local (per bin) statistics
   TProfile2PolyBin *dst = nullptr;
   TProfile2PolyBin *src = nullptr;

   for (Int_t i = 0; i < nbins; i++) {
      dst = (TProfile2PolyBin *)fBins->At(i);

      Double_t sumw_acc = dst->GetFSumw();
      Double_t sumw2_acc = dst->GetFSumw2();
      Double_t numEntries_acc = dst->GetFNumEntries();

      // accumulate values of interest in the input vector
      for (const auto &e : list) {
         src = (TProfile2PolyBin *)e->fBins->At(i);
         sumw_acc += src->GetFSumw();
         sumw2_acc += src->GetFSumw2();
         numEntries_acc += src->GetFNumEntries();
      }

      // set values of accumulation
      dst->SetFSumw(sumw_acc);
      dst->SetFSumw2(sumw2_acc);
      dst->SetFNumEntries(numEntries_acc);

      // update averages, errors
      dst->Update();
   }
   this->SetContentToAverageW();
   return 1;
}

void TProfile2Poly::SetContentToAverageW()
{
   Int_t nbins = fBins->GetSize();
   for (Int_t i = 0; i < nbins; i++) {
      TProfile2PolyBin *bin = (TProfile2PolyBin *)fBins->At(i);
      bin->SetContent(bin->GetAverage());
   }
}

void TProfile2Poly::SetContentToErrorW()
{
   Int_t nbins = fBins->GetSize();
   for (Int_t i = 0; i < nbins; i++) {
      TProfile2PolyBin *bin = (TProfile2PolyBin *)fBins->At(i);
      bin->Update();
      bin->SetContent(bin->GetError());
   }
}

void TProfile2Poly::Reset(Option_t *opt)
{
   TIter next(fBins);
   TObject *obj;
   TProfile2PolyBin *bin;

   // Clears bin contents
   while ((obj = next())) {
      bin = (TProfile2PolyBin *)obj;
      bin->ClearContent();
      bin->ClearStats();
   }
   TH2::Reset(opt);
}
