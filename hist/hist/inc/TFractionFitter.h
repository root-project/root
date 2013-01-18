// @(#)root/hist:$Id$
// Author: Frank Filthaut F.Filthaut@science.ru.nl  20/05/2002
   
#ifndef ROOT_TFractionFitter
#define ROOT_TFractionFitter

#ifndef ROOT_TVirtualFitter
# include "TVirtualFitter.h"
#endif
#ifndef ROOT_TObjArray
# include "TObjArray.h"
#endif
#include <vector>

class TH1;

///////////////////////////////////////////////////////////////////////////////
// TFractionFitter
//
// Fits MC fractions to data histogram (a la HMCMLL, see R. Barlow and C. Beeston,
// Comp. Phys. Comm. 77 (1993) 219-228, and http://www.hep.man.ac.uk/~roger/hfrac.f).
// 
///////////////////////////////////////////////////////////////////////////////

class TFractionFitter: public TObject {
public:
   TFractionFitter();
   TFractionFitter(TH1* data, TObjArray *MCs, Option_t *option="");
   virtual ~TFractionFitter();

   TVirtualFitter* GetFitter() const;
   void ErrorAnalysis(Double_t UP);
   void SetRangeX(Int_t low, Int_t high);
   void ReleaseRangeX();
   void SetRangeY(Int_t low, Int_t high);
   void ReleaseRangeY();
   void SetRangeZ(Int_t low, Int_t high);
   void ReleaseRangeZ();
   void ExcludeBin(Int_t bin);
   void IncludeBin(Int_t bin);
   void Constrain(Int_t parm, Double_t low, Double_t high);
   void UnConstrain(Int_t parm);
   void SetData(TH1 *data);
   void SetMC(Int_t parm, TH1 *MC);
   void SetWeight(Int_t parm, TH1* weight);
   Int_t Fit();

   void GetResult(Int_t parm, Double_t& value, Double_t& error) const;
   TH1* GetPlot();

   // This global function needs access to computeFCN()
   friend void TFractionFitFCN(Int_t& npar, Double_t* gin, Double_t& f, Double_t* par, Int_t flag);

   // Goodness of fit
   Double_t GetChisquare() const;
   Int_t GetNDF() const;
   Double_t GetProb() const;

   // MC predictions (smeared templates)
   TH1* GetMCPrediction(Int_t parm) const;

private:
   void CheckParNo(Int_t parm) const;
   void CheckConsistency();
   void FindPrediction(int bin, double& t_i, int& k_0, double& A_ki) const;
   void ComputeFCN(Int_t& npar, Double_t* gin, Double_t& f, Double_t* par, Int_t flag);
   void GetRanges(Int_t& minX, Int_t& maxX, Int_t& minY, Int_t& maxY,
                  Int_t& minZ, Int_t& maxZ) const;
   void ComputeChisquareLambda();
   bool IsExcluded(Int_t bin) const;

protected:
   Bool_t   fFitDone;             // flags whether a valid fit has been performed
   Int_t    fLowLimitX;           // first bin in X dimension
   Int_t    fHighLimitX;          // last  bin in X dimension
   Int_t    fLowLimitY;           // first bin in Y dimension
   Int_t    fHighLimitY;          // last  bin in Y dimension
   Int_t    fLowLimitZ;           // first bin in Z dimension
   Int_t    fHighLimitZ;          // last  bin in Z dimension
   std::vector<Int_t> fExcludedBins; // bins excluded from the fit

   Int_t    fNpfits;              // Number of points used in the fit
   Int_t    fNDF;                 // Number of degrees of freedom in the fit
   Double_t fChisquare;           // Template fit chisquare

   TObjArray fAji;                // array of pointers to predictions of real template distributions

   // Histograms
   TH1*      fData;               // pointer to the "data" histogram to be fitted to
   TObjArray fMCs;                // array of pointers to template histograms
   TObjArray fWeights;            // array of pointers to corresponding weight factors (may be null)
   Double_t  fIntegralData;       // "data" histogram content integral over allowed fit range
   Double_t* fIntegralMCs;        // same for template histograms (weights not taken into account)
   Double_t* fFractions;          // template fractions scaled to the "data" histogram statistics
   TH1*      fPlot;               // pointer to histogram containing summed template predictions

   Int_t     fNpar;               // number of fit parameters

   ClassDef(TFractionFitter, 0)   // Fits MC fractions to data histogram
};

//
//  TFractionFitFCN
//
//  Computes negative log-likelihood for TFractionFitter
//

void TFractionFitFCN(Int_t& npar, Double_t* gin, Double_t& f, Double_t* par, Int_t flag);

#endif // ROOT_TFractionFitter
