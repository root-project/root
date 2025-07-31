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
#define AliWarningGeneral(X, fmt, Y) do { ::Warning((X), (fmt), (Y)); } while(false)
ClassImp(AliAODForwardMult)
#ifdef DOXY_INPUT
; // For Emacs
#endif

//____________________________________________________________________
const Float_t AliAODForwardMult::fgkInvalidIpZ = 1e6;

//____________________________________________________________________
AliAODForwardMult::AliAODForwardMult()
  : fIsMC(false),
    fHist(),
    fTriggers(0),
    fIpZ(fgkInvalidIpZ),
    fCentrality(-1),
    fNClusters(0)
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
    fTriggers(0),
    fIpZ(fgkInvalidIpZ),
    fCentrality(-1),
    fNClusters(0)
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
AliAODForwardMult::Clear(Option_t* option)
{
  // Clear (or reset) internal values
  //
  // Parameters:
  //  option   Passed to TH1::Reset
  //
  fHist.Reset(option);
  fTriggers  = 0;
  fIpZ       = fgkInvalidIpZ;
  fNClusters = 0;
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
Bool_t
AliAODForwardMult::HasIpZ() const
{
  // Check if we have valid z coordinate of the interaction point
  //
  // Return:
  //   true if the z coordinate of the interaction point is valid
  //
  return TMath::Abs(fIpZ - fgkInvalidIpZ) > 1;
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
  static TObjString ipz;
  static TObjString trg;
  static TObjString cnt;
  static TObjString ncl;
  ipz = Form("ip_z=%fcm", fIpZ);
  trg = GetTriggerString(fTriggers);
  cnt = Form("%+6.1f%%", fCentrality);
  ncl = Form("%d clusters", fNClusters);
  b->Add(&fHist);
  b->Add(&ipz);
  b->Add(&trg);
  b->Add(&cnt);
  b->Add(&ncl);
}

namespace {
  void AppendAnd(TString& trg, const TString& what)
  {
    if (!trg.IsNull()) trg.Append(" & ");
    trg.Append(what);
  }
}
//____________________________________________________________________
const Char_t*
AliAODForwardMult::GetTriggerString(UInt_t mask)
{
  // Get a string that describes the triggers
  //
  // Parameters:
  //   mask  Bit pattern of triggers
  // Return:
  //   Character string representation of mask
  static TString trg;
  trg = "";
  if ((mask & kInel)        != 0x0) AppendAnd(trg, "INEL");
  if ((mask & kInelGt0)     != 0x0) AppendAnd(trg, "INEL>0");
  if ((mask & kNSD)         != 0x0) AppendAnd(trg, "NSD");
  if ((mask & kV0AND)       != 0x0) AppendAnd(trg, "V0AND");
  if ((mask & kA)           != 0x0) AppendAnd(trg, "A");
  if ((mask & kB)           != 0x0) AppendAnd(trg, "B");
  if ((mask & kC)           != 0x0) AppendAnd(trg, "C");
  if ((mask & kE)           != 0x0) AppendAnd(trg, "E");
  if ((mask & kMCNSD)       != 0x0) AppendAnd(trg, "MCNSD");
  if ((mask & kNClusterGt0) != 0x0) AppendAnd(trg, "NCluster>0");
  if ((mask & kSatellite)   != 0x0) AppendAnd(trg, "Satellite");
  return trg.Data();
}

//____________________________________________________________________
TH1I*
AliAODForwardMult::MakeTriggerHistogram(const char* name, Int_t mask)
{
  //
  // Make a histogram to record triggers in.
  //
  // The bins defined by the trigger enumeration in this class.  One
  // can use this enumeration to retrieve the number of triggers for
  // each class.
  //
  // Parameters:
  //    name Name of the histogram
  //
  // Return:
  //    Newly allocated histogram
  //
  TString sel("");
  TString andSel("");
  if (mask > 0) {
    sel    = GetTriggerString(mask);
    andSel = GetTriggerString(mask & ~kB);
    andSel.Prepend(" & ");
  }
  TH1I* ret = new TH1I(name, "Triggers", (int) kAccepted + 1, -.5, (int) kAccepted + .5);
  ret->SetYTitle("Events");
  ret->SetFillColor(kRed+1);
  ret->SetFillStyle(3001);
  ret->GetXaxis()->SetBinLabel(kBinAll,         "All events");
  ret->GetXaxis()->SetBinLabel(kBinB,           Form("B (Coll.)%s",
						     andSel.Data()));
  ret->GetXaxis()->SetBinLabel(kBinA,           Form("A%s", andSel.Data()));
  ret->GetXaxis()->SetBinLabel(kBinC,           Form("C%s", andSel.Data()));
  ret->GetXaxis()->SetBinLabel(kBinE,           Form("E%s", andSel.Data()));
  ret->GetXaxis()->SetBinLabel(kBinInel,        "Coll. & INEL");
  ret->GetXaxis()->SetBinLabel(kBinInelGt0,     "Coll. & INEL>0");
  ret->GetXaxis()->SetBinLabel(kBinNSD,         "Coll. & NSD");
  ret->GetXaxis()->SetBinLabel(kBinV0AND,       "Coll. & V0AND");
  ret->GetXaxis()->SetBinLabel(kBinMCNSD,       "NSD (MC truth)");
  ret->GetXaxis()->SetBinLabel(kBinSatellite,   "Satellite");
  ret->GetXaxis()->SetBinLabel(kBinPileUp,      "w/Pileup");
  ret->GetXaxis()->SetBinLabel(kBinOffline,     "w/Offline");
  ret->GetXaxis()->SetBinLabel(kBinNClusterGt0, "w/N_{cluster}>1");
  ret->GetXaxis()->SetBinLabel(kWithVertex,     "w/Vertex");
  ret->GetXaxis()->SetBinLabel(kWithTrigger,    Form("w/Selected trigger (%s)",
						     sel.Data()));
  ret->GetXaxis()->SetBinLabel(kAccepted,       "Accepted by cut");
  ret->GetXaxis()->SetNdivisions(kAccepted, false);
  ret->SetStats(0);

  return ret;
}

//____________________________________________________________________
TH1I*
AliAODForwardMult::MakeStatusHistogram(const char* name)
{
  //
  // Make a histogram to record status in.
  //
  // The bins defined by the status enumeration in this class.
  //
  // Parameters:
  //    name Name of the histogram
  //
  // Return:
  //    Newly allocated histogram
  //
  TH1I* ret = new TH1I(name, "Event selection status", (int) kWrongVertex + 1, -.5, (int) kWrongVertex + .5);
  ret->SetYTitle("Events");
  ret->SetFillColor(kBlue+1);
  ret->SetFillStyle(3001);
  ret->GetXaxis()->SetBinLabel(kGoodEvent+1,       "Good");
  ret->GetXaxis()->SetBinLabel(kWrongCentrality+1, "Out-of-range centrality");
  ret->GetXaxis()->SetBinLabel(kWrongTrigger+1,    "Wrong trigger");
  ret->GetXaxis()->SetBinLabel(kIsPileup+1,        "Pile-up");
  ret->GetXaxis()->SetBinLabel(kNoVertex+1,        "No IP_{z}");
  ret->GetXaxis()->SetBinLabel(kWrongVertex+1,     "Out-or-range IP_{z}");
  ret->GetXaxis()->SetNdivisions(kWrongVertex, false);
  ret->SetStats(0);
  return ret;
}

//____________________________________________________________________
UInt_t
AliAODForwardMult::MakeTriggerMask(const char* what)
{
  UShort_t    trgMask = 0;
  TString     trgs(what);
  trgs.ToUpper();
  TObjArray*  parts = trgs.Tokenize("&");
  TObjString* trg;
  TIter       next(parts);
  while ((trg = static_cast<TObjString*>(next()))) {
    TString s(trg->GetString());
    s.Strip(TString::kBoth, ' ');
    s.ToUpper();
    if      (s.IsNull()) continue;
    if      (s.CompareTo("INEL")       == 0) trgMask |= kInel;
    else if (s.CompareTo("INEL>0")     == 0) trgMask |= kInelGt0;
    else if (s.CompareTo("INELGT0")    == 0) trgMask |= kInelGt0;
    else if (s.CompareTo("NSD")        == 0) trgMask |= kNSD;
    else if (s.CompareTo("V0AND")      == 0) trgMask |= kV0AND;
    else if (s.CompareTo("MCNSD")      == 0) trgMask |= kMCNSD;
    else if (s.CompareTo("B")          == 0) trgMask |= kB;
    else if (s.CompareTo("A")          == 0) trgMask |= kA;
    else if (s.CompareTo("C")          == 0) trgMask |= kC;
    else if (s.CompareTo("SAT")        == 0) trgMask |= kSatellite;
    else if (s.CompareTo("E")          == 0) trgMask |= kE;
    else if (s.CompareTo("NCLUSTER>0") == 0) trgMask |= kNClusterGt0;
    else
      AliWarningGeneral("MakeTriggerMask", "Unknown trigger %s", s.Data());
  }
  delete parts;
  return trgMask;
}

//____________________________________________________________________
Bool_t
AliAODForwardMult::CheckEvent(Int_t    triggerMask,
			      Double_t vzMin, Double_t vzMax,
			      UShort_t cMin,  UShort_t cMax,
			      TH1*     hist,  TH1*     status) const
{
  //
  // Check if event meets the passses requirements.
  //
  // It returns true if @e all of the following is true
  //
  // - The trigger is within the bit mask passed.
  // - The vertex is within the specified limits.
  // - The centrality is within the specified limits, or if lower
  //   limit is equal to or larger than the upper limit.
  //
  // If a histogram is passed in the last parameter, then that
  // histogram is filled with the trigger bits.
  //
  // Parameters:
  //    triggerMask  Trigger mask
  //    vzMin        Minimum @f$ v_z@f$ (in centimeters)
  //    vzMax        Maximum @f$ v_z@f$ (in centimeters)
  //    cMin         Minimum centrality (in percent)
  //    cMax         Maximum centrality (in percent)
  //    hist         Histogram to fill
  //
  // Return:
  //    @c true if the event meets the requirements
  //
  if (cMin < cMax && (cMin > fCentrality || cMax <= fCentrality)) {
    if (status) status->Fill(kWrongCentrality);
    return false;
  }

  if (hist) {
    Int_t tmp = triggerMask & ~kB;
    hist->AddBinContent(kBinAll);
    if (IsTriggerBits(kB|tmp))          hist->AddBinContent(kBinB);
    if (IsTriggerBits(kA|tmp))          hist->AddBinContent(kBinA);
    if (IsTriggerBits(kC|tmp))          hist->AddBinContent(kBinC);
    if (IsTriggerBits(kE|tmp))          hist->AddBinContent(kBinE);
    if (IsTriggerBits(kB|kInel))        hist->AddBinContent(kBinInel);
    if (IsTriggerBits(kB|kInelGt0))     hist->AddBinContent(kBinInelGt0);
    if (IsTriggerBits(kB|kNSD))         hist->AddBinContent(kBinNSD);
    if (IsTriggerBits(kB|kV0AND))       hist->AddBinContent(kBinV0AND);
    if (IsTriggerBits(kPileUp))         hist->AddBinContent(kBinPileUp);
    if (IsTriggerBits(kMCNSD))          hist->AddBinContent(kBinMCNSD);
    if (IsTriggerBits(kOffline))        hist->AddBinContent(kBinOffline);
    if (IsTriggerBits(kNClusterGt0))    hist->AddBinContent(kBinNClusterGt0);
    if (IsTriggerBits(kSatellite))      hist->AddBinContent(kBinSatellite);
    if (IsTriggerBits(triggerMask) && !IsTriggerBits(kB|tmp))
      Warning("CheckEvent", "event: 0x%x, mask: 0x%x, tmp: 0x%x, tmp|b: 0x%x",
	     fTriggers, triggerMask, tmp, tmp|kB);
  }
  // Check if we have an event of interest.
  Int_t mask = triggerMask; //|kB
  if (!IsTriggerBits(mask)) {
    if (status) status->Fill(kWrongTrigger);
    return false;
  }

  // Check for pileup
  if (IsTriggerBits(kPileUp)) {
    if (status) status->Fill(kIsPileup);
    return false;
  }
  if (hist) hist->AddBinContent(kWithTrigger);

  // Check that we have a valid vertex
  if (vzMin < vzMax && !HasIpZ()) {
    if (status) status->Fill(kNoVertex);
    return false;
  }
  if (hist) hist->AddBinContent(kWithVertex);

  // Check that vertex is within cuts
  if (vzMin < vzMax && !InRange(vzMin, vzMax)) {
    if (status) status->Fill(kWrongVertex);
    return false;
  }
  if (hist) hist->AddBinContent(kAccepted);

  if (status) status->Fill(kGoodEvent);
  return true;
}

//____________________________________________________________________
void
AliAODForwardMult::Print(Option_t* option) const
{
  // Print this object
  //
  // Parameters:
  //  option   Passed to TH1::Print
  fHist.Print(option);
  UShort_t sys = GetSystem();
  TString  str = "unknown";
  switch (sys) {
  case 1:  str = "pp"; break;
  case 2:  str = "PbPb"; break;
  case 3:  str = "pPb" ; break;
  }
  std::cout << "Ipz:         " << fIpZ << "cm " << (HasIpZ() ? "" : "in")
	    << "valid\n"
	    << "Triggers:    " << GetTriggerString(fTriggers)  << "\n"
	    << "sNN:         " << GetSNN() << "GeV\n"
	    << "System:      " << str << "\n"
	    << "Centrality:  " << fCentrality << "%"
	    << std::endl;
}

//____________________________________________________________________
//
// EOF
//
