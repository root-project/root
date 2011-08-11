#ifndef ROOT_TEfficiency
#define ROOT_TEfficiency

//standard header
#include <vector>
#include <utility>

//ROOT header
#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

#ifndef ROOT_TAttLine
#include "TAttLine.h"
#endif

#ifndef ROOT_TAttFill
#include "TAttFill.h"
#endif

#ifndef ROOT_TAttMarker
#include "TAttMarker.h"
#endif

class TCollection;
class TF1;
class TGraphAsymmErrors;
class TH1;
class TH2;
class TList;

//|TEfficiency
//------------------------

class TEfficiency: public TNamed, public TAttLine, public TAttFill, public TAttMarker
{
public:  
      //enumaration type for different statistic options for calculating confidence intervals
      //kF* ... frequentist methods; kB* ... bayesian methods      
      enum EStatOption {
	 kFCP = 0,                         //Clopper-Pearson interval (recommended by PDG)
	 kFNormal,                         //normal approximation
	 kFWilson,                         //Wilson interval
	 kFAC,                             //Agresti-Coull interval
	 kFFC,                             //Feldman-Cousins interval
	 kBJeffrey,                        //Jeffrey interval (Prior ~ Beta(0.5,0.5)
	 kBUniform,                        //Prior ~ Uniform = Beta(1,1)
	 kBBayesian                        //user specified Prior ~ Beta(fBeta_alpha,fBeta_beta)
      };

protected:

      Double_t      fBeta_alpha;             //global parameter for prior beta distribution (default = 1)
      Double_t      fBeta_beta;              //global parameter for prior beta distribution (default = 1)
      std::vector<std::pair<Double_t, Double_t> > fBeta_bin_params;  // parameter for prior beta distribution different bin by bin 
                                                                 // (default vector is empty)
      Double_t      (*fBoundary)(Int_t,Int_t,Double_t,Bool_t);               //!pointer to a method calculating the boundaries of confidence intervals
      Double_t      fConfLevel;              //confidence level (default = 0.95)
      TDirectory*   fDirectory;              //!pointer to directory holding this TEfficiency object
      TList*        fFunctions;              //->pointer to list of functions
      TGraphAsymmErrors* fPaintGraph;        //!temporary graph for painting
      TH2*          fPaintHisto;             //!temporary histogram for painting      
      TH1*          fPassedHistogram;        //histogram for events which passed certain criteria
      EStatOption   fStatisticOption;        //defines how the confidence intervals are determined
      TH1*          fTotalHistogram;         //histogram for total number of events
      Double_t      fWeight;                 //weight for all events (default = 1)

      enum{
	 kIsBayesian       = BIT(14),              //bayesian statistics are used
         kPosteriorMode    = BIT(15),              //use posterior mean for best estimate (Bayesian statistics)
         kShortestInterval = BIT(16),              // use shortest interval
         kUseBinPrior      = BIT(17),              // use a different prior for each bin
         kUseWeights       = BIT(18)               // use weights
      };

      void          Build(const char* name,const char* title);      
      
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
      ~TEfficiency();
      
      void          Add(const TEfficiency& rEff) {*this += rEff;}
      virtual Int_t DistancetoPrimitive(Int_t px, Int_t py);
      void          Draw(Option_t* opt = "");
      virtual void  ExecuteEvent(Int_t event, Int_t px, Int_t py);
      void          Fill(Bool_t bPassed,Double_t x,Double_t y=0,Double_t z=0);
      void          FillWeighted(Bool_t bPassed,Double_t weight,Double_t x,Double_t y=0,Double_t z=0);
      Int_t         FindFixBin(Double_t x,Double_t y=0,Double_t z=0) const;
      Int_t         Fit(TF1* f1,Option_t* opt="");
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
      void          Paint(Option_t* opt);
      void          SavePrimitive(ostream& out,Option_t* opt="");
      void          SetBetaAlpha(Double_t alpha);
      void          SetBetaBeta(Double_t beta);    
      void          SetBetaBinParameters(Int_t bin, Double_t alpha, Double_t beta);
      void          SetConfidenceLevel(Double_t level);
      void          SetDirectory(TDirectory* dir);
      void          SetName(const char* name);
      Bool_t        SetPassedEvents(Int_t bin,Int_t events);
      Bool_t        SetPassedHistogram(const TH1& rPassed,Option_t* opt);
      void          SetPosteriorMode(Bool_t on = true) { SetBit(kPosteriorMode,on); SetShortestInterval(on); } 
      void          SetPosteriorAverage(Bool_t on = true) { SetBit(kPosteriorMode,!on); } 
      void          SetShortestInterval(Bool_t on = true) { SetBit(kShortestInterval,on); } 
      void          SetCentralInterval(Bool_t on = true) { SetBit(kShortestInterval,!on); } 
      void          SetStatisticOption(EStatOption option);
      void          SetTitle(const char* title);
      Bool_t        SetTotalEvents(Int_t bin,Int_t events);
      Bool_t        SetTotalHistogram(const TH1& rTotal,Option_t* opt);
      void          SetUseWeightedEvents();
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
      static Double_t Combine(Double_t& up,Double_t& low,Int_t n,const Int_t* pass,const Int_t* total,
			      Double_t alpha,Double_t beta,Double_t level=0.683,
			      const Double_t* w=0,Option_t* opt="");
      static TGraphAsymmErrors* Combine(TCollection* pList,Option_t* opt="",Int_t n=0,const Double_t* w=0);
      
      //calculating boundaries of confidence intervals
      static Double_t AgrestiCoull(Int_t total,Int_t passed,Double_t level,Bool_t bUpper);
      static Double_t ClopperPearson(Int_t total,Int_t passed,Double_t level,Bool_t bUpper);
      static Double_t Normal(Int_t total,Int_t passed,Double_t level,Bool_t bUpper);
      static Double_t Wilson(Int_t total,Int_t passed,Double_t level,Bool_t bUpper);
      static Double_t FeldmanCousins(Int_t total,Int_t passed,Double_t level,Bool_t bUpper);
      static Bool_t FeldmanCousinsInterval(Int_t total,Int_t passed,Double_t level,Double_t & lower, Double_t & upper);
      // Bayesian functions 
      static Double_t Bayesian(Int_t total,Int_t passed,Double_t level,Double_t alpha,Double_t beta,Bool_t bUpper, Bool_t bShortest = false);
      // helper functions for Bayesian statistics  
      static Double_t BetaCentralInterval(Double_t level,Double_t alpha,Double_t beta,Bool_t bUpper);
      static Bool_t   BetaShortestInterval(Double_t level,Double_t alpha,Double_t beta,Double_t & lower, Double_t & upper);
      static Double_t BetaMean(Double_t alpha,Double_t beta);
      static Double_t BetaMode(Double_t alpha,Double_t beta);
      
      ClassDef(TEfficiency,1)     //calculating efficiencies
};

const TEfficiency operator+(const TEfficiency& lhs,const TEfficiency& rhs);

#endif
