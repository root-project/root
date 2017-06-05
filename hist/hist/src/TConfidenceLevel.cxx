// @(#)root/hist:$Id$
// Author: Christophe.Delaere@cern.ch   21/08/2002

////////////////////////////////////////////////////////////////////////////////
/** \class TConfidenceLevel
    \ingroup Hist
    \brief Class to compute 95% CL limits
*///////////////////////////////////////////////////////////////////////////////

/*************************************************************************
 * C.Delaere                                                             *
 * adapted from the mclimit code from Tom Junk                           *
 * see http://cern.ch/thomasj/searchlimits/ecl.html                      *
 *************************************************************************/

#include "TConfidenceLevel.h"
#include "TH1F.h"
#include "TMath.h"
#include "Riostream.h"

ClassImp(TConfidenceLevel);

Double_t const TConfidenceLevel::fgMCLM2S = 0.025;
Double_t const TConfidenceLevel::fgMCLM1S = 0.16;
Double_t const TConfidenceLevel::fgMCLMED = 0.5;
Double_t const TConfidenceLevel::fgMCLP1S = 0.84;
Double_t const TConfidenceLevel::fgMCLP2S = 0.975;
// LHWG "one-sided" definition
Double_t const TConfidenceLevel::fgMCL3S1S = 2.6998E-3;
Double_t const TConfidenceLevel::fgMCL5S1S = 5.7330E-7;
// the other definition (not chosen by the LHWG)
Double_t const TConfidenceLevel::fgMCL3S2S = 1.349898E-3;
Double_t const TConfidenceLevel::fgMCL5S2S = 2.866516E-7;


////////////////////////////////////////////////////////////////////////////////
/// Default constructor

TConfidenceLevel::TConfidenceLevel()
{
   fStot = 0;
   fBtot = 0;
   fDtot = 0;
   fTSD  = 0;
   fTSB  = 0;
   fTSS  = 0;
   fLRS  = 0;
   fLRB  = 0;
   fNMC  = 0;
   fNNMC = 0;
   fISS  = 0;
   fISB  = 0;
   fMCL3S = fgMCL3S1S;
   fMCL5S = fgMCL5S1S;
}


////////////////////////////////////////////////////////////////////////////////
/// Constructor that fix some conventions
/// \param mc is the number of Monte Carlo experiments
/// \param onesided specifies if the intervals are one-sided or not.

TConfidenceLevel::TConfidenceLevel(Int_t mc, bool onesided)
{
   fStot = 0;
   fBtot = 0;
   fDtot = 0;
   fTSD  = 0;
   fTSB  = 0;
   fTSS  = 0;
   fLRS  = 0;
   fLRB  = 0;
   fNMC  = mc;
   fNNMC = mc;
   fISS  = new Int_t[mc];
   fISB  = new Int_t[mc];
   fMCL3S = onesided ? fgMCL3S1S : fgMCL3S2S;
   fMCL5S = onesided ? fgMCL5S1S : fgMCL5S2S;
}


////////////////////////////////////////////////////////////////////////////////
/// The destructor

TConfidenceLevel::~TConfidenceLevel()
{
   if (fISS)
      delete[]fISS;
   if (fISB)
      delete[]fISB;
   if (fTSB)
      delete[]fTSB;
   if (fTSS)
      delete[]fTSS;
   if (fLRS)
      delete[]fLRS;
   if (fLRB)
      delete[]fLRB;
}


////////////////////////////////////////////////////////////////////////////////
/// Get the expected statistic value in the background only hypothesis

Double_t TConfidenceLevel::GetExpectedStatistic_b(Int_t sigma) const
{
   switch (sigma) {
   case -2:
      return (-2 *((fTSB[fISB[TMath::Min((Int_t) fNMC,(Int_t) TMath::Max((Int_t) 1,(Int_t) (fNMC * fgMCLP2S)))]]) - fStot));
   case -1:
      return (-2 *((fTSB[fISB[TMath::Min((Int_t) fNMC,(Int_t) TMath::Max((Int_t) 1,(Int_t) (fNMC * fgMCLP1S)))]]) - fStot));
   case 0:
      return (-2 *((fTSB[fISB[TMath::Min((Int_t) fNMC,(Int_t) TMath::Max((Int_t) 1,(Int_t) (fNMC * fgMCLMED)))]]) - fStot));
   case 1:
      return (-2 *((fTSB[fISB[TMath::Min((Int_t) fNMC,(Int_t) TMath::Max((Int_t) 1,(Int_t) (fNMC * fgMCLM1S)))]]) - fStot));
   case 2:
      return (-2 *((fTSB[fISB[TMath::Min((Int_t) fNMC,(Int_t) TMath::Max((Int_t) 1,(Int_t) (fNMC * fgMCLM2S)))]]) - fStot));
   default:
      return 0;
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Get the expected statistic value in the signal plus background hypothesis

Double_t TConfidenceLevel::GetExpectedStatistic_sb(Int_t sigma) const
{
   switch (sigma) {
   case -2:
      return (-2 *((fTSS[fISS[TMath::Min((Int_t) fNMC,(Int_t) TMath::Max((Int_t) 1,(Int_t) (fNMC * fgMCLP2S)))]]) - fStot));
   case -1:
      return (-2 *((fTSS[fISS[TMath::Min((Int_t) fNMC,(Int_t) TMath::Max((Int_t) 1,(Int_t) (fNMC * fgMCLP1S)))]]) - fStot));
   case 0:
      return (-2 *((fTSS[fISS[TMath::Min((Int_t) fNMC,(Int_t) TMath::Max((Int_t) 1,(Int_t) (fNMC * fgMCLMED)))]]) - fStot));
   case 1:
      return (-2 *((fTSS[fISS[TMath::Min((Int_t) fNMC,(Int_t) TMath::Max((Int_t) 1,(Int_t) (fNMC * fgMCLM1S)))]]) - fStot));
   case 2:
      return (-2 *((fTSS[fISS[TMath::Min((Int_t) fNMC,(Int_t) TMath::Max((Int_t) 1,(Int_t) (fNMC * fgMCLM2S)))]]) - fStot));
   default:
      return 0;
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Get the Confidence Level for the background only

Double_t TConfidenceLevel::CLb(bool use_sMC) const
{
   Double_t result = 0;
   if (use_sMC) {
      for (Int_t i = 0; i < fNMC; i++)
         if (fTSS[fISS[i]] < fTSD)
            result += (1 / (fLRS[fISS[i]] * fNMC));
   } else {
      for (Int_t i = 0; i < fNMC; i++)
         if (fTSB[fISB[i]] < fTSD)
            result = (Double_t(i + 1)) / fNMC;
   }
   return result;
}


////////////////////////////////////////////////////////////////////////////////
/// Get the Confidence Level for the signal plus background hypothesis

Double_t TConfidenceLevel::CLsb(bool use_sMC) const
{
   Double_t result = 0;
   if (use_sMC) {
      for (Int_t i = 0; i < fNMC; i++)
         if (fTSS[fISS[i]] <= fTSD)
            result = i / fNMC;
   } else {
      for (Int_t i = 0; i < fNMC; i++)
         if (fTSB[fISB[i]] <= fTSD)
            result += (fLRB[fISB[i]]) / fNMC;
   }
   return result;
}


////////////////////////////////////////////////////////////////////////////////
/// Get the Confidence Level defined by CLs = CLsb/CLb.
/// This quantity is stable w.r.t. background fluctuations.

Double_t TConfidenceLevel::CLs(bool use_sMC) const
{
   Double_t clb = CLb(kFALSE);
   Double_t clsb = CLsb(use_sMC);
   if(clb==0) { std::cout << "Warning: clb = 0 !" << std::endl; return 0;}
   else return clsb/clb;
}


////////////////////////////////////////////////////////////////////////////////
/// Get the expected Confidence Level for the signal plus background hypothesis
/// if there is only background.

Double_t TConfidenceLevel::GetExpectedCLsb_b(Int_t sigma) const
{
   Double_t result = 0;
   switch (sigma) {
   case -2:
      {
         for (Int_t i = 0; i < fNMC; i++)
            if (fTSB[fISB[i]] <= fTSB[fISB[TMath::Min((Int_t) fNMC,(Int_t) TMath::Max((Int_t) 1,(Int_t) (fNMC * fgMCLP2S)))]])
               result += fLRB[fISB[i]] / fNMC;
         return result;
      }
   case -1:
      {
         for (Int_t i = 0; i < fNMC; i++)
            if (fTSB[fISB[i]] <= fTSB[fISB[TMath::Min((Int_t) fNMC,(Int_t) TMath::Max((Int_t) 1,(Int_t) (fNMC * fgMCLP1S)))]])
               result += fLRB[fISB[i]] / fNMC;
         return result;
      }
   case 0:
      {
         for (Int_t i = 0; i < fNMC; i++)
            if (fTSB[fISB[i]] <= fTSB[fISB[TMath::Min((Int_t) fNMC,(Int_t) TMath::Max((Int_t) 1,(Int_t) (fNMC * fgMCLMED)))]])
               result += fLRB[fISB[i]] / fNMC;
         return result;
      }
   case 1:
      {
         for (Int_t i = 0; i < fNMC; i++)
            if (fTSB[fISB[i]] <= fTSB[fISB[TMath::Min((Int_t) fNMC,(Int_t) TMath::Max((Int_t) 1,(Int_t) (fNMC * fgMCLM1S)))]])
               result += fLRB[fISB[i]] / fNMC;
         return result;
      }
   case 2:
      {
         for (Int_t i = 0; i < fNMC; i++)
            if (fTSB[fISB[i]] <= fTSB[fISB[TMath::Min((Int_t) fNMC,(Int_t) TMath::Max((Int_t) 1,(Int_t) (fNMC * fgMCLM2S)))]])
               result += fLRB[fISB[i]] / fNMC;
         return result;
      }
   default:
      return 0;
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Get the expected Confidence Level for the background only
/// if there is signal and background.

Double_t TConfidenceLevel::GetExpectedCLb_sb(Int_t sigma) const
{
   Double_t result = 0;
   switch (sigma) {
   case 2:
      {
         for (Int_t i = 0; i < fNMC; i++)
            if (fTSS[fISS[i]] <= fTSS[fISS[TMath::Min((Int_t) fNMC,(Int_t) TMath::Max((Int_t) 1,(Int_t) (fNMC * fgMCLP2S)))]])
               result += fLRS[fISS[i]] / fNMC;
         return result;
      }
   case 1:
      {
         for (Int_t i = 0; i < fNMC; i++)
            if (fTSS[fISS[i]] <= fTSS[fISS[TMath::Min((Int_t) fNMC,(Int_t) TMath::Max((Int_t) 1,(Int_t) (fNMC * fgMCLP1S)))]])
               result += fLRS[fISS[i]] / fNMC;
         return result;
      }
   case 0:
      {
         for (Int_t i = 0; i < fNMC; i++)
            if (fTSS[fISS[i]] <= fTSS[fISS[TMath::Min((Int_t) fNMC,(Int_t) TMath::Max((Int_t) 1,(Int_t) (fNMC * fgMCLMED)))]])
               result += fLRS[fISS[i]] / fNMC;
         return result;
      }
   case -1:
      {
         for (Int_t i = 0; i < fNMC; i++)
            if (fTSS[fISS[i]] <= fTSS[fISS[TMath::Min((Int_t) fNMC,(Int_t) TMath::Max((Int_t) 1,(Int_t) (fNMC * fgMCLM1S)))]])
               result += fLRS[fISS[i]] / fNMC;
         return result;
      }
   case -2:
      {
         for (Int_t i = 0; i < fNMC; i++)
            if (fTSS[fISS[i]] <= fTSS[fISS[TMath::Min((Int_t) fNMC,(Int_t) TMath::Max((Int_t) 1,(Int_t) (fNMC * fgMCLM2S)))]])
               result += fLRS[fISS[i]] / fNMC;
         return result;
      }
   default:
      return 0;
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Get the expected Confidence Level for the background only
/// if there is only background.

Double_t TConfidenceLevel::GetExpectedCLb_b(Int_t sigma) const
{
   Double_t result = 0;
   switch (sigma) {
   case 2:
      {
         for (Int_t i = 0; i < fNMC; i++)
            if (fTSB[fISB[i]] <= fTSB[fISB[TMath::Min((Int_t) fNMC,(Int_t) TMath::Max((Int_t) 1,(Int_t) (fNMC * fgMCLM2S)))]])
               result = (i + 1) / double (fNMC);
         return result;
      }
   case 1:
      {
         for (Int_t i = 0; i < fNMC; i++)
            if (fTSB[fISB[i]] <= fTSB[fISB[TMath::Min((Int_t) fNMC,(Int_t) TMath::Max((Int_t) 1,(Int_t) (fNMC * fgMCLM1S)))]])
               result = (i + 1) / double (fNMC);
         return result;
      }
   case 0:
      {
         for (Int_t i = 0; i < fNMC; i++)
            if (fTSB[fISB[i]] <= fTSB[fISB[TMath::Min((Int_t) fNMC,(Int_t) TMath::Max((Int_t) 1,(Int_t) (fNMC * fgMCLMED)))]])
               result = (i + 1) / double (fNMC);
         return result;
      }
   case -1:
      {
         for (Int_t i = 0; i < fNMC; i++)
            if (fTSB[fISB[i]] <= fTSB[fISB[TMath::Min((Int_t) fNMC,(Int_t) TMath::Max((Int_t) 1,(Int_t) (fNMC * fgMCLP1S)))]])
               result = (i + 1) / double (fNMC);
         return result;
      }
   case -2:
      {
         for (Int_t i = 0; i < fNMC; i++)
            if (fTSB[fISB[i]] <= fTSB[fISB[TMath::Min((Int_t) fNMC,(Int_t) TMath::Max((Int_t) 1,(Int_t) (fNMC * fgMCLP2S)))]])
               result = (i + 1) / double (fNMC);
         return result;
      }
   }
   return result;
}


////////////////////////////////////////////////////////////////////////////////
/// Get average CLsb.

Double_t TConfidenceLevel::GetAverageCLsb() const
{
   Double_t result = 0;
   Double_t psumsb = 0;
   for (Int_t i = 0; i < fNMC; i++) {
      psumsb += fLRB[fISB[i]] / fNMC;
      result += psumsb / fNMC;
   }
   return result;
}


////////////////////////////////////////////////////////////////////////////////
/// Get average CLs.

Double_t TConfidenceLevel::GetAverageCLs() const
{
   Double_t result = 0;
   Double_t psumsb = 0;
   for (Int_t i = 0; i < fNMC; i++) {
      psumsb += fLRB[fISB[i]] / fNMC;
      result += ((psumsb / fNMC) / ((i + 1) / fNMC));
   }
   return result;
}


////////////////////////////////////////////////////////////////////////////////
/// Get 3s probability.

Double_t TConfidenceLevel::Get3sProbability() const
{
   Double_t result = 0;
   Double_t psumbs = 0;
   for (Int_t i = 0; i < fNMC; i++) {
      psumbs += 1 / (Double_t) (fLRS[(fISS[(Int_t) (fNMC - i)])] * fNMC);
      if (psumbs <= fMCL3S)
         result = i / fNMC;
   }
   return result;
}


////////////////////////////////////////////////////////////////////////////////
/// Get 5s probability.

Double_t TConfidenceLevel::Get5sProbability() const
{
   Double_t result = 0;
   Double_t psumbs = 0;
   for (Int_t i = 0; i < fNMC; i++) {
      psumbs += 1 / (Double_t) (fLRS[(fISS[(Int_t) (fNMC - i)])] * fNMC);
      if (psumbs <= fMCL5S)
         result = i / fNMC;
   }
   return result;
}


////////////////////////////////////////////////////////////////////////////////
/// Display sort of a "canonical" -2lnQ plot.
/// This results in a plot with 2 elements:
///
/// - The histogram of -2lnQ for background hypothesis (full)
/// - The histogram of -2lnQ for signal and background hypothesis (dashed)
///
/// The 2 histograms are respectively named b_hist and sb_hist.

void  TConfidenceLevel::Draw(const Option_t*)
{
   TH1F h("TConfidenceLevel_Draw","",50,0,0);
   Int_t i;
   for (i=0; i<fNMC; i++) {
      h.Fill(-2*(fTSB[i]-fStot));
      h.Fill(-2*(fTSS[i]-fStot));
   }
   TH1F* b_hist  = new TH1F("b_hist", "-2lnQ",50,h.GetXaxis()->GetXmin(),h.GetXaxis()->GetXmax());
   TH1F* sb_hist = new TH1F("sb_hist","-2lnQ",50,h.GetXaxis()->GetXmin(),h.GetXaxis()->GetXmax());
   for (i=0; i<fNMC; i++) {
      b_hist->Fill(-2*(fTSB[i]-fStot));
      sb_hist->Fill(-2*(fTSS[i]-fStot));
   }
   b_hist->Draw();
   sb_hist->Draw("Same");
   sb_hist->SetLineStyle(3);
}


////////////////////////////////////////////////////////////////////////////////
/// Set the TSB.

void  TConfidenceLevel::SetTSB(Double_t * in)
{
   fTSB = in;
   TMath::Sort(fNNMC, fTSB, fISB, 0);
}


////////////////////////////////////////////////////////////////////////////////
/// Set the TSS.

void  TConfidenceLevel::SetTSS(Double_t * in)
{
   fTSS = in;
   TMath::Sort(fNNMC, fTSS, fISS, 0);
}
