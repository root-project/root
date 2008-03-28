// @(#)root/foam:$Id$
// Author: S. Jadach <mailto:Stanislaw.jadach@ifj.edu.pl>, P.Sawicki <mailto:Pawel.Sawicki@ifj.edu.pl>

//____________________________________________________________________________
//
// Class  TFoamMaxwt
// =================
// Small auxiliary class for controlling MC weight.
// It provides certain measure of the "maximum weight"
// depending on small user-parameter "epsilon".
// It creates and uses 2 histograms of the TH1D class.
// User defines no. of bins nBin,  nBin=1000 is  recommended
// wmax defines weight range (1,wmax), it is adjusted "manually"
//
//____________________________________________________________________________


#include "Riostream.h"
#include "TH1.h"
#include "TFoamMaxwt.h"

ClassImp(TFoamMaxwt);

//____________________________________________________________________________
TFoamMaxwt::TFoamMaxwt()
{
// Constructor for streamer
   fNent = 0;
   fnBin = 0;
   fWtHst1 = 0;
   fWtHst2 = 0;
}

//____________________________________________________________________________
TFoamMaxwt::TFoamMaxwt(Double_t wmax, Int_t nBin)
{
// Principal user constructor
   fNent = 0;
   fnBin = nBin;
   fwmax = wmax;
   fWtHst1 = new TH1D("TFoamMaxwt_hst_Wt1","Histo of weight   ",nBin,0.0,wmax);
   fWtHst2 = new TH1D("TFoamMaxwt_hst_Wt2","Histo of weight**2",nBin,0.0,wmax);
   fWtHst1->SetDirectory(0);// exclude from diskfile
   fWtHst2->SetDirectory(0);// and enable deleting
}

//______________________________________________________________________________
TFoamMaxwt::TFoamMaxwt(TFoamMaxwt &From): TObject(From)
{
// Explicit COPY CONSTRUCTOR (unused, so far)
   fnBin   = From.fnBin;
   fwmax   = From.fwmax;
   fWtHst1 = From.fWtHst1;
   fWtHst2 = From.fWtHst2;
   Error("TFoamMaxwt","COPY CONSTRUCTOR NOT TESTED!");
}

//_______________________________________________________________________________
TFoamMaxwt::~TFoamMaxwt()
{
// Destructor
   delete fWtHst1; // For this SetDirectory(0) is needed!
   delete fWtHst2; //
   fWtHst1=0;
   fWtHst2=0;
}
//_______________________________________________________________________________
void TFoamMaxwt::Reset()
{
// Reseting weight analysis
   fNent = 0;
   fWtHst1->Reset();
   fWtHst2->Reset();
}

//_______________________________________________________________________________
TFoamMaxwt& TFoamMaxwt::operator=(const TFoamMaxwt &From)
{
// substitution =
   if (&From == this) return *this;
   fnBin = From.fnBin;
   fwmax = From.fwmax;
   fWtHst1 = From.fWtHst1;
   fWtHst2 = From.fWtHst2;
   return *this;
}

//________________________________________________________________________________
void TFoamMaxwt::Fill(Double_t wt)
{
// Filling analyzed weight
   fNent =  fNent+1.0;
   fWtHst1->Fill(wt,1.0);
   fWtHst2->Fill(wt,wt);
}

//________________________________________________________________________________
void TFoamMaxwt::Make(Double_t eps, Double_t &MCeff)
{
// Calculates Efficiency= aveWt/wtLim for a given tolerance level epsilon<<1
// To be called at the end of the MC run.

   Double_t wtLim,aveWt;
   GetMCeff(eps, MCeff, wtLim);
   aveWt = MCeff*wtLim;
   cout<< "00000000000000000000000000000000000000000000000000000000000000000000000"<<endl;
   cout<< "00 -->wtLim: No_evt ="<<fNent<<"   <Wt> = "<<aveWt<<"  wtLim=  "<<wtLim<<endl;
   cout<< "00 -->wtLim: For eps = "<<eps  <<"    EFFICIENCY <Wt>/wtLim= "<<MCeff<<endl;
   cout<< "00000000000000000000000000000000000000000000000000000000000000000000000"<<endl;
}

//_________________________________________________________________________________
void TFoamMaxwt::GetMCeff(Double_t eps, Double_t &MCeff, Double_t &wtLim)
{
// Calculates Efficiency= aveWt/wtLim for a given tolerance level epsilon<<1
// using information stored in two histograms.
// To be called at the end of the MC run.

   Int_t ib,ibX;
   Double_t lowEdge,bin,bin1;
   Double_t aveWt, aveWt1;

   fWtHst1->Print();
   fWtHst2->Print();

// Convention on bin-numbering: nb=1 for 1-st bin, underflow nb=0, overflow nb=Nb+1
   Double_t sum   = 0.0;
   Double_t sumWt = 0.0;
   for(ib=0;ib<=fnBin+1;ib++) {
      sum   += fWtHst1->GetBinContent(ib);
      sumWt += fWtHst2->GetBinContent(ib);
   }
   if( (sum == 0.0) || (sumWt == 0.0) ) {
      cout<<"TFoamMaxwt::Make: zero content of histogram !!!,sum,sumWt ="<<sum<<sumWt<<endl;
   }
   aveWt = sumWt/sum;
   //--------------------------------------
   for( ibX=fnBin+1; ibX>0; ibX--) {
      lowEdge = (ibX-1.0)*fwmax/fnBin;
      sum   = 0.0;
      sumWt = 0.0;
      for( ib=0; ib<=fnBin+1; ib++) {
         bin  = fWtHst1->GetBinContent(ib);
         bin1 = fWtHst2->GetBinContent(ib);
         if(ib >= ibX) bin1=lowEdge*bin;
         sum   += bin;
         sumWt += bin1;
      }
      aveWt1 = sumWt/sum;
      if( TMath::Abs(1.0-aveWt1/aveWt) > eps ) break;
   }
   //---------------------------
   if(ibX == (fnBin+1) ) {
      wtLim = 1.0e200;
      MCeff   = 0.0;
      cout<< "+++++ wtLim undefined. Higher uper limit in histogram"<<endl;
   } else if( ibX == 1) {
      wtLim = 0.0;
      MCeff   =-1.0;
      cout<< "+++++ wtLim undefined. Lower uper limit or more bins "<<endl;
   } else {
      wtLim= (ibX)*fwmax/fnBin; // We over-estimate wtLim, under-estimate MCeff
      MCeff  = aveWt/wtLim;
   }
}
///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//      End of    Class  TFoamMaxwt                                          //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
