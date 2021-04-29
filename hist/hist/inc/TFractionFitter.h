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

#include "TFitResultPtr.h"

#include <vector>

class TH1;

namespace ROOT {
   namespace Fit  {
      class Fitter;
   }
}

class TFractionFitter: public TObject {
public:
   TFractionFitter();
   TFractionFitter(TH1* data, TObjArray *MCs, Option_t *option="");
   virtual ~TFractionFitter();

   //TVirtualFitter* GetFitter() const;
   ROOT::Fit::Fitter* GetFitter() const;
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
   TFitResultPtr Fit();

   void GetResult(Int_t parm, Double_t& value, Double_t& error) const;
   TH1* GetPlot();

   // This global function needs access to computeFCN()
   //friend void TFractionFitFCN(Int_t& npar, Double_t* gin, Double_t& f, Double_t* par, Int_t flag);

   // Goodness of fit
   Double_t GetChisquare() const;
   Int_t GetNDF() const;
   Double_t GetProb() const;

   // MC predictions (smeared templates)
   TH1* GetMCPrediction(Int_t parm) const;

   // FCN evaluation
   Double_t EvaluateFCN(const Double_t * par) {
      Double_t f = 0;
      ComputeFCN(f, par, 0);
      return f;
   }

private:
   void CheckParNo(Int_t parm) const;
   void CheckConsistency();
   void FindPrediction(int bin, double& t_i, int& k_0, double& A_ki) const;
   void ComputeFCN(Double_t& f, const Double_t* par, Int_t flag);
   void GetRanges(Int_t& minX, Int_t& maxX, Int_t& minY, Int_t& maxY,
                  Int_t& minZ, Int_t& maxZ) const;
   void ComputeChisquareLambda();
   bool IsExcluded(Int_t bin) const;

protected:
   Bool_t   fFitDone;                   ///< Flags whether a valid fit has been performed
   Int_t    fLowLimitX;                 ///< First bin in X dimension
   Int_t    fHighLimitX;                ///< Last  bin in X dimension
   Int_t    fLowLimitY;                 ///< First bin in Y dimension
   Int_t    fHighLimitY;                ///< Last  bin in Y dimension
   Int_t    fLowLimitZ;                 ///< First bin in Z dimension
   Int_t    fHighLimitZ;                ///< Last  bin in Z dimension
   std::vector<Int_t> fExcludedBins;    ///< Bins excluded from the fit

   Int_t    fNpfits;                    ///< Number of points used in the fit
   Int_t    fNDF;                       ///< Number of degrees of freedom in the fit
   Double_t fChisquare;                 ///< Template fit chisquare

   TObjArray fAji;                      ///< Array of pointers to predictions of real template distributions

   ///@name Histograms
   ///@{
   TH1*      fData;                     ///< Pointer to the "data" histogram to be fitted to
   TObjArray fMCs;                      ///< Array of pointers to template histograms
   TObjArray fWeights;                  ///< Array of pointers to corresponding weight factors (may be null)
   Double_t  fIntegralData;             ///< "data" histogram content integral over allowed fit range
   Double_t* fIntegralMCs;              ///< Same for template histograms (weights not taken into account)
   Double_t* fFractions;                ///< Template fractions scaled to the "data" histogram statistics
   TH1*      fPlot;                     ///< Pointer to histogram containing summed template predictions
   ROOT::Fit::Fitter *fFractionFitter;  ///< Pointer to Fitter class
   ///@}

   Int_t     fNpar;                     ///< number of fit parameters

   ClassDef(TFractionFitter, 0); // Fits MC fractions to data histogram
};

//
//  TFractionFitFCN
//
//  Computes negative log-likelihood for TFractionFitter
//

void TFractionFitFCN(Int_t& npar, Double_t* gin, Double_t& f, Double_t* par, Int_t flag);

#endif // ROOT_TFractionFitter
