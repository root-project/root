#include "TProfile2Poly.h"
#include "TProfileHelper.h"

#include "TMultiGraph.h"
#include "TGraph.h"
#include "TClass.h"
#include "TList.h"
#include "TMath.h"

ClassImp(TProfile2Poly)

// -------------- TProfile2PolyBin  --------------

TProfile2PolyBin::TProfile2PolyBin() {
    fSumw = 0;
    fSumw2 = 0;
    fSumwz = 0;
    fSumwz2 = 0;

    fNumEntries = 0;
}

TProfile2PolyBin::TProfile2PolyBin(TObject* poly, Int_t bin_number)
    : TH2PolyBin(poly, bin_number){
    fSumw = 0;
    fSumw2 = 0;
    fSumwz = 0;
    fSumwz2 = 0;

    fNumEntries = 0;
}

void TProfile2PolyBin::UpdateAverage(){
    fContent =  fSumw / fNumEntries;
    SetChanged(true);
}

void TProfile2PolyBin::ClearStats(){
   fSumw       = 0;
   fSumw2      = 0;
   fSumwz      = 0;
   fSumwz2     = 0;
   fNumEntries = 0;
}

// -------------- TProfile2Poly  --------------

TProfile2Poly::TProfile2Poly(const char *name, const char *title,
                             Double_t xlow, Double_t xup,
                             Double_t ylow, Double_t yup)
    : TH2Poly(name, title, xlow, xup, ylow, yup) {}

TProfile2Poly::TProfile2Poly(const char *name, const char *title,
                             Int_t nX, Double_t xlow, Double_t xup,
                             Int_t nY, Double_t ylow, Double_t yup)
    : TH2Poly(name, title, nX, xlow, xup, nY, ylow, yup) {}

Int_t TProfile2Poly::AddBin(TObject *poly)
{
    if (!poly) return 0;

    if (fBins == 0) {
       fBins = new TList();
       fBins->SetOwner();
    }

    fNcells++;
    Int_t ibin = fNcells-kNOverflow;
    TProfile2PolyBin *bin = new TProfile2PolyBin(poly, ibin);

    // If the bin lies outside histogram boundaries, then extends the boundaries.
    // Also changes the partition information accordingly
    Bool_t flag = kFALSE;
    if (fFloat) {
       if (fXaxis.GetXmin() > bin->GetXMin()) {
          fXaxis.Set(100, bin->GetXMin(), fXaxis.GetXmax());
          flag = kTRUE;
       }
       if (fXaxis.GetXmax() < bin->GetXMax()) {
          fXaxis.Set(100, fXaxis.GetXmin(), bin->GetXMax());
          flag = kTRUE;
       }
       if (fYaxis.GetXmin() > bin->GetYMin()) {
          fYaxis.Set(100, bin->GetYMin(), fYaxis.GetXmax());
          flag = kTRUE;
       }
       if (fYaxis.GetXmax() < bin->GetYMax()) {
          fYaxis.Set(100, fYaxis.GetXmin(), bin->GetYMax());
          flag = kTRUE;
       }
       if (flag) ChangePartition(fCellX, fCellY);
    } else {
       /*Implement polygon clipping code here*/
    }

    fBins->Add((TObject*) bin);
    SetNewBinAdded(kTRUE);

    // Adds the bin to the partition matrix
    AddBinToPartition(bin);

    return ibin;
}

Int_t TProfile2Poly::Fill(Double_t xcoord, Double_t ycoord, Double_t value){
    return Fill(xcoord, ycoord, value, 1);
}

Int_t TProfile2Poly::Fill(Double_t xcoord, Double_t ycoord, Double_t value, Double_t weight) {

    //TODO: is this correct?
    if (fNcells <= kNOverflow) return 0;
    Int_t overflow = 0;
    if      (ycoord > fYaxis.GetXmax()) overflow += -1;
    else if (ycoord > fYaxis.GetXmin()) overflow += -4;
    else                           overflow += -7;
    if      (xcoord > fXaxis.GetXmax()) overflow += -2;
    else if(xcoord > fXaxis.GetXmin())  overflow += -1;
    if (overflow != -5) {
        fOverflow[-overflow - 1]+= weight;
        if (fSumw2.fN) fSumw2.fArray[-overflow - 1] += weight*weight;
        return overflow;
    }

    // Finds the cell (x,y) coordinates belong to
    Int_t n = (Int_t)(floor((xcoord-fXaxis.GetXmin())/fStepX));
    Int_t m = (Int_t)(floor((ycoord-fYaxis.GetXmin())/fStepY));

    // Make sure the array indices are correct.
    if (n>=fCellX) n = fCellX-1;
    if (m>=fCellY) m = fCellY-1;
    if (n<0)       n = 0;
    if (m<0)       m = 0;

    //TODO: is this correct?
    if (fIsEmpty[n+fCellX*m]) {
        fOverflow[4]+= weight;
        if (fSumw2.fN) fSumw2.fArray[4] += weight*weight;
        return -5;
    }

    // ------------ Update global (per histo) statistics
    fTsumw   += weight;
    fTsumw2  += weight*weight;
    fTsumwx  += weight*xcoord;
    fTsumwx2 += weight*xcoord*xcoord;
    fTsumwy  += weight*ycoord;
    fTsumwy2 += weight*ycoord*ycoord;
    fTsumwxy += weight*xcoord*ycoord;
    fTsumwz  += weight*value;
    fTsumwz2 += weight*value*value;

    // ------------ Update local (per bin) statistics
    TProfile2PolyBin* bin;
    TIter next(&fCells[n+fCellX*m]);
    TObject *obj;
    while ((obj=next())) {
        bin  = (TProfile2PolyBin*)obj;
        if (bin->IsInside(xcoord,ycoord)) {
            fEntries++;
            bin->SetFNumEntries( bin->GetFNumEntries() + 1 );
            bin->SetFSumw( bin->GetFSumw() + value );
            bin->SetFSumwz( bin->GetFSumwz() + value*weight );
            bin->UpdateAverage(); // fContent = fSumw / fNumEntries;

            return bin->GetBinNumber();
        }
    }

    fOverflow[4]+= weight;
    if (fSumw2.fN) fSumw2.fArray[4] += weight*weight*value;
    return -5;
}

Long64_t TProfile2Poly::Merge(TCollection* in){

    std::vector<TProfile2Poly*> list;

    for(int i=0; i<in->GetSize(); i++){
        list.push_back((TProfile2Poly*)((TList*)in)->At(i));
    }
    this->Merge(list);
    return 0;
}

void  TProfile2Poly::Merge(std::vector<TProfile2Poly*> list){

    // TODO: Build checks to see if merge is allowed on "this" / "list"

    TProfile2PolyBin* dst = nullptr;
    TProfile2PolyBin* src = nullptr;

    // TODO: CHECK SIZES OF ALL INPUT ELEMENTS TO VERYIFY THAT WE CAN ACTUALLY MERGE THESE SHITS TOGETHER.
    Int_t numBins = list[0]->fBins->GetSize();

    // ------------ Update global (per histo) statistics
    for(const auto& histo : list){
        this->fEntries += histo->fEntries;
        this->fTsumw   += histo->fTsumw  ;
        this->fTsumw2  += histo->fTsumw2 ;
        this->fTsumwx  += histo->fTsumwx ;
        this->fTsumwx2 += histo->fTsumwx2;
        this->fTsumwy  += histo->fTsumwy ;
        this->fTsumwy2 += histo->fTsumwy2;
        this->fTsumwxy += histo->fTsumwxy;
        this->fTsumwz  += histo->fTsumwz ;
        this->fTsumwz2 += histo->fTsumwz2;
    }

    // ------------ Update local (per bin) statistics
    for(Int_t i=0; i<numBins; i++){
        dst = (TProfile2PolyBin*)fBins->At(i);

        Int_t current_src        = 0;
        Double_t Sumw_srcs       = 0;
        Double_t NumEntries_srcs = 0;

        // accumulate values of interest in the input vector
        for(const auto& e : list){
            src  = ((TProfile2PolyBin*)list[current_src]->fBins->At(i));
            Sumw_srcs += src->GetFSumw();
            NumEntries_srcs += src->GetFNumEntries();
            current_src++;
        }

        // set values of accumulation
        dst->SetFSumw(Sumw_srcs + dst->GetFSumw());
        dst->SetFNumEntries(NumEntries_srcs + dst->GetFNumEntries());
        dst->UpdateAverage();
    }
}

void TProfile2Poly::Reset(Option_t *opt){
   TIter next(fBins);
   TObject* obj;
   TProfile2PolyBin* bin;

   // Clears bin contents
   while ((obj = next())) {
      bin = (TProfile2PolyBin*) obj;
      bin->ClearContent();
      bin->ClearStats();
   }
   TH2::Reset(opt);
}


