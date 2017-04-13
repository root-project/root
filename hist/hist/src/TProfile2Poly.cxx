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
    fSumV = 0;
    fSumV2 = 0;
    fSumVW = 0;
    fSumVW2 = 0;

    fNumEntries = 0;
}

TProfile2PolyBin::TProfile2PolyBin(TObject* poly, Int_t bin_number)
    : TH2PolyBin(poly, bin_number){
    fSumV = 0;
    fSumV2 = 0;
    fSumVW = 0;
    fSumVW2 = 0;

    fNumEntries = 0;
}

TProfile2PolyBin::~TProfile2PolyBin(){
}

void TProfile2PolyBin::UpdateAverage(){
    fContent =  fSumV / fNumEntries;
    SetChanged(true);
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

TProfile2Poly::~TProfile2Poly(){
}

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

    TProfile2PolyBin* bin;

    TIter next(&fCells[n+fCellX*m]);
    TObject *obj;

    while ((obj=next())) {
        bin  = (TProfile2PolyBin*)obj;
        if (bin->IsInside(xcoord,ycoord)) {

            fEntries++;

            bin->fNumEntries++;
            bin->fSumV += value;
            bin->fSumVW += value*weight;

            bin->UpdateAverage();

            return bin->GetBinNumber();
        }
    }

    fOverflow[4]+= weight;
    if (fSumw2.fN) fSumw2.fArray[4] += weight*weight*value;
    return -5;
}

void  TProfile2Poly::Merge(std::vector<TProfile2Poly*> list){

    TIter next(list[0]->fBins);
    TObject* obj;

    Int_t n=0;
    while ((obj = next())) {
        n++;
    }

    Int_t current_element=0;
    for(auto& e : list ){
        for(Int_t i=0; i<n; i++){

            TProfile2PolyBin* dst = (TProfile2PolyBin*)fBins->At(i);
            TProfile2PolyBin* src  = ((TProfile2PolyBin*)list[current_element]->fBins->At(i));

            /* DEBUG INFO
            std::cout << "fCont | current " << dst->GetContent() << "\t"
                      << "to_merge "        << src->GetContent() << std::endl;

            std::cout << "fEntr | current " << dst->getFNumEntries() << "\t"
                      << "to_merge "        << src->getFNumEntries() << std::endl;

            std::cout << "fSumV | current " << dst->getFSumV() << "\t"
                      << "to_merge "        << src->getFSumV() << std::endl;

            std::cout << std::endl;
            */

            dst->setFSumV(dst->fSumV + src->fSumV);
            dst->setFNumEntries(dst->fNumEntries + src->fNumEntries);
            dst->UpdateAverage();
        }
     current_element++;
    }
}
