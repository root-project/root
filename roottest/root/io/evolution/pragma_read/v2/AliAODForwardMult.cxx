//
// Class that contains the forward multiplicity data per event 
//
// This class contains a histogram of 
// @f[
//   \frac{d^2N_{ch}}{d\eta d\phi}\quad,
// @f]
// as well as a trigger mask for each analysed event.  
// 
// The eta acceptance of the event is stored in the underflow bins of
// the histogram.  So to build the final histogram, one needs to
// correct for this acceptance (properly weighted by the events), and
// the vertex efficiency.  This simply boils down to defining a 2D
// histogram and summing the event histograms in that histogram.  One
// should of course also do proper book-keeping of the accepted event.
//
#include "AliAODForwardMult.h"
#include <TBrowser.h>
#include <iostream>
#include <TMath.h>
#include <TObjString.h>
#include <TObjArray.h>
// #include "AliLog.h"
ClassImp(AliAODForwardMult)
#ifdef DOXY_INPUT
; // For Emacs 
#endif

//____________________________________________________________________
AliAODForwardMult::AliAODForwardMult()
  : fIsMC(false),
    fHist(),
    fHeader(0)
{
  // 
  // Constructor 
  // 
}

//____________________________________________________________________
AliAODForwardMult::AliAODForwardMult(Bool_t isMC) 
  : fIsMC(isMC),
    fHist("forwardMult", "d^{2}N_{ch}/d#etad#varphi in the forward regions", 
	  200, -4, 6, 20, 0, 2*TMath::Pi()),
    fHeader()
{
  // 
  // Constructor 
  // 
  // Parameters: 
  //  isMC   If set to true this is for MC data (effects branch name)
  // 
  fHist.SetXTitle("#eta");
  fHist.SetYTitle("#varphi [radians]");
  fHist.SetZTitle("#frac{d^{2}N_{ch}}{d#etad#varphi}");
  fHist.SetDirectory(0);
  fHist.Sumw2();
}

//____________________________________________________________________
void
AliAODForwardMult::Init(const TAxis& etaAxis)
{
  // Initialize the histogram with an eta axis 
  // 
  // Parameters: 
  //   etaAxis       Eta axis to use 
  // 
  fHist.SetBins(etaAxis.GetNbins(), etaAxis.GetXmin(), etaAxis.GetXmax(), 
		20, 0, 2*TMath::Pi());
}
//____________________________________________________________________
void
AliAODForwardMult::SetSNN(UShort_t snn)
{
  // set the center of mass energy per nucleon pair (GeV). 
  // This is stored in bin (0,0) of the histogram 
  // 
  // Parameters: 
  //   sNN   Center of mass energy per nuclean 
  fHist.SetBinContent(0,0,snn);
}
//____________________________________________________________________
void
AliAODForwardMult::SetSystem(UShort_t sys)
{
  // set the center of mass energy per nucleon pair (GeV). 
  // This is stored in bin (N+1,0) of the histogram 
  // 
  // Parameters: 
  //   sys   Collision system number 
  fHist.SetBinContent(fHist.GetNbinsX()+1,0,sys);
}

//____________________________________________________________________
void
AliAODForwardMult::Clear(Option_t* option)
{
  // Clear (or reset) internal values 
  // 
  // Parameters: 
  //  option   Passed to TH1::Reset 
  // 
  fHist.Reset(option);
  if (fHeader) fHeader->Clear();
}

//____________________________________________________________________
UShort_t
AliAODForwardMult::GetSNN() const
{
  // set the center of mass energy per nucleon pair (GeV). 
  // This is stored in bin (0,0) of the histogram 
  // 
  // Parameters: 
  //   sNN   Center of mass energy per nuclean 
  return UShort_t(fHist.GetBinContent(0,0));
}

//____________________________________________________________________
UShort_t
AliAODForwardMult::GetSystem() const
{
  // set the center of mass energy per nucleon pair (GeV). 
  // This is stored in bin (N+1,0) of the histogram 
  // 
  // Parameters: 
  //   sNN   Center of mass energy per nuclean 
  return UShort_t(fHist.GetBinContent(fHist.GetNbinsX()+1,0));
}

//____________________________________________________________________
void
AliAODForwardMult::Browse(TBrowser* b)
{
  // Browse this object 
  // 
  // Parameters: 
  //   b   Browser to use 
  b->Add(&fHist);
  if (fHeader) b->Add(fHeader);
}


//____________________________________________________________________
void
AliAODForwardMult::Print(Option_t* option) const
{
  // Print this object 
  // 
  // Parameters: 
  //  option   Passed to TH1::Print 
  // fHist.Print(option);
  const TAxis& x = *(fHist.GetXaxis());
  const TAxis& y = *(fHist.GetYaxis());

  UShort_t sys = GetSystem();
  TString  str = "unknown";
  switch (sys) { 
  case 1:  str = "pp"; break;
  case 2:  str = "PbPb"; break;
  case 3:  str = "pPb" ; break;
  }

  fHist.Print(option);
  std::cout << "N_ch(eta,phi): (" 
	    << x.GetNbins() << "," << x.GetXmin() << "-" << x.GetXmax() << ")x("
	    << y.GetNbins() << "," << y.GetXmin() << "-" << y.GetXmax() << ")\n"
	    << "sNN:           " << GetSNN() << "GeV\n" 
	    << "System:        " << str << std::endl;

  if (fHeader) fHeader->Print();
}

//____________________________________________________________________
void
AliAODForwardMult::CreateHeader(UInt_t   triggers, 
				Float_t  ipZ, 
				Float_t  centrality, 
				UShort_t nClusters)
{
  Info("", "Creating header");
  fHeader = new AliAODForwardHeader();
  fHeader->SetTriggerMask(triggers);
  fHeader->SetIpZ(ipZ);
  fHeader->SetCentrality(centrality);
  fHeader->SetNClusters(nClusters);
  // fHeader->Print();
}

#if CUSTOM_STREAMER
//______________________________________________________________________________
void AliAODForwardMult::Streamer(TBuffer &R__b)
{
  // Stream an object of class AliAODForwardMult.
  if (R__b.IsReading()) {
    Printf("Reading AliAODForwardMult object in streamer");
    R__b.ReadClassBuffer(AliAODForwardMult::Class(),this);
  } else {
    R__b.WriteClassBuffer(AliAODForwardMult::Class(),this);
  }
}
#endif

//____________________________________________________________________
//
// EOF
//
