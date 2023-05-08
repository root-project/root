// @(#)root/hist:$Id$
// Author: Christophe.Delaere@cern.ch   21/08/2002

#ifndef ROOT_TConfidenceLevel
#define ROOT_TConfidenceLevel

#include "TObject.h"

//____________________________________________________________________
//
// TConfidenceLevel
//
// This class serves as output for the TLimit::ComputeLimit method.
// It is created just after the time-consuming part and can be stored
// in a TFile for further processing. It contains
// light methods to return CLs, CLb and other interesting quantities.
//____________________________________________________________________


class TConfidenceLevel : public TObject {
 public:
   TConfidenceLevel();
   TConfidenceLevel(Int_t mc, bool onesided = kTRUE);
   ~TConfidenceLevel() override;
   inline void SetTSD(Double_t in) { fTSD = in; }
   void SetTSB(Double_t * in);
   void SetTSS(Double_t * in);
   inline void SetLRS(Double_t * in) { fLRS = in; }
   inline void SetLRB(Double_t * in) { fLRB = in; }
   inline void SetBtot(Double_t in) { fBtot = in; }
   inline void SetStot(Double_t in) { fStot = in; }
   inline void SetDtot(Int_t in) { fDtot = in; }
   inline Double_t GetStatistic() const { return -2 * (fTSD - fStot); }
   void Draw(const Option_t *option="") override;
   Double_t GetExpectedStatistic_b(Int_t sigma = 0) const;
   Double_t GetExpectedStatistic_sb(Int_t sigma = 0) const;
   Double_t CLb(bool use_sMC = kFALSE) const;
   Double_t CLsb(bool use_sMC = kFALSE) const;
   Double_t CLs(bool use_sMC = kFALSE) const;
   Double_t GetExpectedCLb_sb(Int_t sigma = 0) const;
   Double_t GetExpectedCLb_b(Int_t sigma = 0) const;
   Double_t GetExpectedCLsb_b(Int_t sigma = 0) const;
   inline Double_t GetExpectedCLs_b(Int_t sigma = 0) const { return (GetExpectedCLsb_b(sigma) / GetExpectedCLb_b(sigma)); }
   Double_t GetAverageCLs() const;
   Double_t GetAverageCLsb() const;
   Double_t Get3sProbability() const;
   Double_t Get5sProbability() const;
   inline Int_t GetDtot() const { return fDtot; }
   inline Double_t GetStot() const { return fStot; }
   inline Double_t GetBtot() const { return fBtot; }
 private:
   // data members used for the limits calculation
   Int_t      fNNMC;
   Int_t      fDtot;
   Double_t   fStot;
   Double_t   fBtot;
   Double_t   fTSD;
   Double_t   fNMC;
   Double_t   fMCL3S;
   Double_t   fMCL5S;
   Double_t  *fTSB;              //[fNNMC]
   Double_t  *fTSS;              //[fNNMC]
   Double_t  *fLRS;              //[fNNMC]
   Double_t  *fLRB;              //[fNNMC]
   Int_t     *fISS;              //[fNNMC]
   Int_t     *fISB;              //[fNNMC]
   // cumulative probabilities for defining the bands on plots
   static const Double_t fgMCLM2S;
   static const Double_t fgMCLM1S;
   static const Double_t fgMCLMED;
   static const Double_t fgMCLP1S;
   static const Double_t fgMCLP2S;
   static const Double_t fgMCL3S1S;
   static const Double_t fgMCL5S1S;
   static const Double_t fgMCL3S2S;
   static const Double_t fgMCL5S2S;
   ClassDefOverride(TConfidenceLevel, 1) // output for TLimit functions
};

#endif

