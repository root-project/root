#ifndef __Validation_RecoParticleFlow_TH2Analyzer__
#define __Validation_RecoParticleFlow_TH2Analyzer__

#include <vector>

#include <TObject.h>

class TH2;
class TH1D;
class TH2D;

// EN FAIT NON ? 
// check why white window in colin's case
// are you making copies of histograms without changing the name? 
// names could be handled in the following way:
// - give a name to each instance of TH2Analyzer in constructor
// - all histograms of the corresponding TH2Analyzer are created with a name (key)
// which starts with name_RMS

class TH2Analyzer : public TObject {

 public:
  TH2Analyzer( const TH2* h, int rebin=1) : 
    hist2D_(h), 
    rebinnedHist2D_(0),
    average_(0),
    RMS_(0),
    sigmaGauss_(0),
    meanXslice_(0) {
    Eval(rebin);
  }

  TH2Analyzer( const TH2* h, const int binxmin, const int binxmax,
	       const int rebin, const bool cst_binning=true) : 
    hist2D_(h), 
    rebinnedHist2D_(0),
    average_(0),
    RMS_(0),
    sigmaGauss_(0),
    meanXslice_(0) {
    Eval(rebin, binxmin, binxmax, cst_binning);
  } 

  ~TH2Analyzer() override {Reset(); }

  void Reset() {}

  void SetHisto( const TH2* h ) {hist2D_ = h;}

  void Eval(const int rebinFactor);
  void Eval(const int rebinFactor, const int binxmin, const int binxmax,
	    const bool cst_binning);
  
  TH1D* Average() { return average_; }
  TH1D* RMS() { return RMS_; }
  TH1D* SigmaGauss() { return sigmaGauss_; }
  TH1D* MeanX() { return meanXslice_; }

  // add an histo for chi2 /  ndof 
  // add a function FitSlice(int i)
  // not now: work along Y

 private:

  void ProcessSlices(  const TH2D* /* histo */ ) {}

  // no need for const, because i is copied
  void ProcessSlice(const int /* i */, TH1D* /* histo*/ ) const {}

  const TH2* hist2D_;
  TH2D*      rebinnedHist2D_;
  TH1D*      average_;
  TH1D*      RMS_;
  TH1D*      sigmaGauss_;
  TH1D*      meanXslice_;

  //std::vector< TH1D* > parameters_; // if we are not fitting with a gauss function
  ClassDefOverride(TH2Analyzer, 1);

};

#endif 
