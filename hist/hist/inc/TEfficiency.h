#ifndef ROOT_TEfficiency
#define ROOT_TEfficiency

//standard header
#include <vector>
#include <utility>

//ROOT header
#include "TNamed.h"

#include "TAttLine.h"

#include "TAttFill.h"

#include "TAttMarker.h"

#include "TFitResultPtr.h"


class TCollection;
class TF1;
class TGraphAsymmErrors;
class TH1;
class TH2;
class TList;

class TEfficiency: public TNamed, public TAttLine, public TAttFill, public TAttMarker
{
public:
   /// Enumeration type for different statistic options for calculating confidence intervals
   /// kF* ... frequentist methods; kB* ... bayesian methods
   enum EStatOption {
      kFCP = 0,                         ///< Clopper-Pearson interval (recommended by PDG)
      kFNormal,                         ///< Normal approximation
      kFWilson,                         ///< Wilson interval
      kFAC,                             ///< Agresti-Coull interval
      kFFC,                             ///< Feldman-Cousins interval
      kBJeffrey,                        ///< Jeffrey interval (Prior ~ Beta(0.5,0.5)
      kBUniform,                        ///< Prior ~ Uniform = Beta(1,1)
      kBBayesian,                       ///< User specified Prior ~ Beta(fBeta_alpha,fBeta_beta)
      kMidP                             ///< Mid-P Lancaster interval
   };

protected:

      Double_t      fBeta_alpha;             ///< Global parameter for prior beta distribution (default = 1)
      Double_t      fBeta_beta;              ///< Global parameter for prior beta distribution (default = 1)
      std::vector<std::pair<Double_t, Double_t> > fBeta_bin_params;  ///< Parameter for prior beta distribution different bin by bin
                                                                     ///< (default vector is empty)
      Double_t      (*fBoundary)(Double_t,Double_t,Double_t,Bool_t); ///<! Pointer to a method calculating the boundaries of confidence intervals
      Double_t      fConfLevel;              ///<  Confidence level (default = 0.683, 1 sigma)
      TDirectory*   fDirectory;              ///<! Pointer to directory holding this TEfficiency object
      TList*        fFunctions;              ///<->Pointer to list of functions
      TGraphAsymmErrors* fPaintGraph;        ///<! Temporary graph for painting
      TH2*          fPaintHisto;             ///<! Temporary histogram for painting
      TH1*          fPassedHistogram;        ///<  Histogram for events which passed certain criteria
      EStatOption   fStatisticOption;        ///<  Defines how the confidence intervals are determined
      TH1*          fTotalHistogram;         ///<  Histogram for total number of events
      Double_t      fWeight;                 ///<  Weight for all events (default = 1)

      enum EStatusBits {
         kIsBayesian       = BIT(14),  ///< Bayesian statistics are used
         kPosteriorMode    = BIT(15),  ///< Use posterior mean for best estimate (Bayesian statistics)
         kShortestInterval = BIT(16),  ///< Use shortest interval
         kUseBinPrior      = BIT(17),  ///< Use a different prior for each bin
         kUseWeights       = BIT(18)   ///< Use weights
      };

      void          Build(const char* name,const char* title);
      void          FillGraph(TGraphAsymmErrors * graph, Option_t * opt) const;
      void          FillHistogram(TH2 * h2) const;

public:
      TEfficiency();
      TEfficiency(const TH1& passed,const TH1& total);
      TEfficiency(const char* name,const char* title,Int_t nbins,
                  const Double_t* xbins);
      TEfficiency(const char* name,const char* title,Int_t nbins,Double_t xlow,
                  Double_t xup);
      TEfficiency(const char* name,const char* title,Int_t nbinsx,
                  Double_t xlow,Double_t xup,Int_t nbinsy,Double_t ylow,
                  Double_t yup);
      TEfficiency(const char* name,const char* title,Int_t nbinsx,
                  const Double_t* xbins,Int_t nbinsy,const Double_t* ybins);
      TEfficiency(const char* name,const char* title,Int_t nbinsx,
                  Double_t xlow,Double_t xup,Int_t nbinsy,Double_t ylow,
                  Double_t yup,Int_t nbinsz,Double_t zlow,Double_t zup);
      TEfficiency(const char* name,const char* title,Int_t nbinsx,
                  const Double_t* xbins,Int_t nbinsy,const Double_t* ybins,
                  Int_t nbinsz,const Double_t* zbins);
      TEfficiency(const TEfficiency& heff);
      ~TEfficiency() override;

      void          Add(const TEfficiency& rEff) {*this += rEff;}
      void          Browse(TBrowser*) override{Draw();}
      TGraphAsymmErrors*   CreateGraph(Option_t * opt = "") const;
      TH2*          CreateHistogram(Option_t * opt = "") const;
      Int_t DistancetoPrimitive(Int_t px, Int_t py) override;
      void          Draw(Option_t* opt = "") override;
      void  ExecuteEvent(Int_t event, Int_t px, Int_t py) override;
      void          Fill(Bool_t bPassed,Double_t x,Double_t y=0,Double_t z=0);
      void          FillWeighted(Bool_t bPassed,Double_t weight,Double_t x,Double_t y=0,Double_t z=0);
      Int_t         FindFixBin(Double_t x,Double_t y=0,Double_t z=0) const;
      TFitResultPtr Fit(TF1* f1,Option_t* opt="");
      // use trick of -1 to return global parameters
      Double_t      GetBetaAlpha(Int_t bin = -1) const {return (fBeta_bin_params.size() > (UInt_t)bin) ? fBeta_bin_params[bin].first : fBeta_alpha;}
      Double_t      GetBetaBeta(Int_t bin =  -1) const {return (fBeta_bin_params.size() > (UInt_t)bin) ? fBeta_bin_params[bin].second : fBeta_beta;}
      Double_t      GetConfidenceLevel() const {return fConfLevel;}
      TH1*          GetCopyPassedHisto() const;
      TH1*          GetCopyTotalHisto() const;
      Int_t         GetDimension() const;
      TDirectory*   GetDirectory() const {return fDirectory;}
      Double_t      GetEfficiency(Int_t bin) const;
      Double_t      GetEfficiencyErrorLow(Int_t bin) const;
      Double_t      GetEfficiencyErrorUp(Int_t bin) const;
      Int_t         GetGlobalBin(Int_t binx,Int_t biny=0,Int_t binz=0) const;
      TGraphAsymmErrors*   GetPaintedGraph() const { return fPaintGraph; }
      TH2*          GetPaintedHistogram() const { return fPaintHisto; }
      TList*        GetListOfFunctions();
      const TH1*    GetPassedHistogram() const {return fPassedHistogram;}
      EStatOption   GetStatisticOption() const {return fStatisticOption;}
      const TH1*    GetTotalHistogram() const {return fTotalHistogram;}
      Double_t      GetWeight() const {return fWeight;}
      Long64_t      Merge(TCollection* list);
      TEfficiency&  operator+=(const TEfficiency& rhs);
      TEfficiency&  operator=(const TEfficiency& rhs);
      void          Paint(Option_t* opt) override;
      void          SavePrimitive(std::ostream& out,Option_t* opt="") override;
      void          SetBetaAlpha(Double_t alpha);
      void          SetBetaBeta(Double_t beta);
      void          SetBetaBinParameters(Int_t bin, Double_t alpha, Double_t beta);
      void          SetConfidenceLevel(Double_t level);
      void          SetDirectory(TDirectory* dir);
      void          SetName(const char* name) override;
      Bool_t        SetPassedEvents(Int_t bin,Int_t events);
      Bool_t        SetPassedHistogram(const TH1& rPassed,Option_t* opt);
      void          SetPosteriorMode(Bool_t on = true) { SetBit(kPosteriorMode,on); SetShortestInterval(on); }
      void          SetPosteriorAverage(Bool_t on = true) { SetBit(kPosteriorMode,!on); }
      void          SetShortestInterval(Bool_t on = true) { SetBit(kShortestInterval,on); }
      void          SetCentralInterval(Bool_t on = true) { SetBit(kShortestInterval,!on); }
      void          SetStatisticOption(EStatOption option);
      Bool_t        SetBins(Int_t nx, Double_t xmin, Double_t xmax);
      Bool_t        SetBins(Int_t nx, const Double_t *xBins);
      Bool_t        SetBins(Int_t nx, Double_t xmin, Double_t xmax, Int_t ny, Double_t ymin, Double_t ymax);
      Bool_t        SetBins(Int_t nx, const Double_t *xBins, Int_t ny, const Double_t *yBins);
      Bool_t        SetBins(Int_t nx, Double_t xmin, Double_t xmax, Int_t ny, Double_t ymin, Double_t ymax,
                            Int_t nz, Double_t zmin, Double_t zmax);
      Bool_t        SetBins(Int_t nx, const Double_t *xBins, Int_t ny, const Double_t * yBins, Int_t nz,
                            const Double_t *zBins);

      void          SetTitle(const char* title) override;
      Bool_t        SetTotalEvents(Int_t bin,Int_t events);
      Bool_t        SetTotalHistogram(const TH1& rTotal,Option_t* opt);
      void          SetUseWeightedEvents(Bool_t on = kTRUE);
      void          SetWeight(Double_t weight);
      Bool_t        UsesBayesianStat() const {return TestBit(kIsBayesian);}
      Bool_t        UsesPosteriorMode() const   {return TestBit(kPosteriorMode) && TestBit(kIsBayesian);}
      Bool_t        UsesShortestInterval() const   {return TestBit(kShortestInterval) && TestBit(kIsBayesian);}
      Bool_t        UsesPosteriorAverage() const   {return !UsesPosteriorMode();}
      Bool_t        UsesCentralInterval() const   {return !UsesShortestInterval();}
      Bool_t        UsesWeights() const {return TestBit(kUseWeights);}

      static Bool_t CheckBinning(const TH1& pass,const TH1& total);
      static Bool_t CheckConsistency(const TH1& pass,const TH1& total,Option_t* opt="");
      static Bool_t CheckEntries(const TH1& pass,const TH1& total,Option_t* opt="");
      static Bool_t CheckWeights(const TH1& pass,const TH1& total);
      static Double_t Combine(Double_t& up,Double_t& low,Int_t n,const Int_t* pass,const Int_t* total,
                              Double_t alpha,Double_t beta,Double_t level=0.683,
                              const Double_t* w=0,Option_t* opt="");
      static TGraphAsymmErrors* Combine(TCollection* pList,Option_t* opt="",Int_t n=0,const Double_t* w=0);

      //calculating boundaries of confidence intervals
      static Double_t AgrestiCoull(Double_t total,Double_t passed,Double_t level,Bool_t bUpper);
      static Double_t ClopperPearson(Double_t total,Double_t passed,Double_t level,Bool_t bUpper);
      static Double_t Normal(Double_t total,Double_t passed,Double_t level,Bool_t bUpper);
      static Double_t Wilson(Double_t total,Double_t passed,Double_t level,Bool_t bUpper);
      static Double_t FeldmanCousins(Double_t total,Double_t passed,Double_t level,Bool_t bUpper);
      static Bool_t FeldmanCousinsInterval(Double_t total,Double_t passed,Double_t level,Double_t & lower, Double_t & upper);
      static Double_t MidPInterval(Double_t total,Double_t passed,Double_t level,Bool_t bUpper);
      // Bayesian functions
      static Double_t Bayesian(Double_t total,Double_t passed,Double_t level,Double_t alpha,Double_t beta,Bool_t bUpper, Bool_t bShortest = false);
      // helper functions for Bayesian statistics
      static Double_t BetaCentralInterval(Double_t level,Double_t alpha,Double_t beta,Bool_t bUpper);
      static Bool_t   BetaShortestInterval(Double_t level,Double_t alpha,Double_t beta,Double_t & lower, Double_t & upper);
      static Double_t BetaMean(Double_t alpha,Double_t beta);
      static Double_t BetaMode(Double_t alpha,Double_t beta);

      ClassDefOverride(TEfficiency,2)     //calculating efficiencies
};

const TEfficiency operator+(const TEfficiency& lhs,const TEfficiency& rhs);

#endif
