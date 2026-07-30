
 /*****************************************************************************
  * Project: RooFit                                                           *
  *                                                                           *
  * Copyright (c) 2000-2005, Regents of the University of California          *
  *                          and Stanford University. All rights reserved.    *
  *                                                                           *
  * Redistribution and use in source and binary forms,                        *
  * with or without modification, are permitted according to the terms        *
  * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
  *****************************************************************************/

//////////////////////////////////////////////////////////////////////////////
/// \class RooFFTConvPdf
/// \ingroup Roofitcore
///
/// This class implements a generic one-dimensional numeric convolution of two PDFs,
/// and can convolve any two RooAbsPdfs. The class exploits the convolution theorem
/// \f[
///       f(x) * g(x) \rightarrow F(k_i) \cdot G(k_i)
/// \f]
/// to calculate the convolution by calculating a forward FFT of both input PDFs,
/// multiplying the complex coefficients and performing the inverse FFT to get the
/// result in the input space. The FFT is implemented internally so that no external
/// FFT library (e.g. FFTW3) is required.
///
/// Note that the performance in terms of speed and stability of RooFFTConvPdf is
/// vastly superior to that of RooNumConvPdf.
///
/// An important feature of FFT convolutions is that the observable is assumed to be
/// cyclical. This is correct for cyclical observables such as angles,
/// but does not hold in general. For non-cyclical variables, wrap-around artifacts may be
/// encountered, *e.g.* if the PDF is zero at xMin and non-zero at xMax. A rising tail may appear at xMin.
/// This is inevitable when using FFTs. A distribution with 3 bins therefore looks like:
/// ```
/// ... 0 1 2 0 1 2 0 1 2 ...
/// ```
///
/// Therefore, if bins 0 and 2 are not equal, the FFT sees a cyclical function with a step at the 2|0 boundary, which causes
/// artifacts in Fourier space.
///
/// The spillover or discontinuity can be reduced or eliminated by
/// introducing a buffer zone in the FFT calculation. If this feature is activated (on by default),
/// the sampling array for the FFT calculation is extended in both directions,
/// and padded with the lowest/highest bin.
/// Example:
/// ```
///     original:                -5 -4 -3 -2 -1 0 +1 +2 +3 +4 +5
///     add buffer zones:    U U -5 -4 -3 -2 -1 0 +1 +2 +3 +4 +5 O O
///     rotate:              0 +1 +2 +3 +4 +5 O O U U -5 -4 -3 -2 -1
/// ```
/// The buffer bins are stripped away when the FFT output values
/// are transferred back to the p.d.f cache. The default buffer size is 10% of the
/// observable domain size, and can be changed with the `setBufferFraction()` member function.
///
/// The RooFFTConvPdf uses caching inherited from a RooAbsCachedPdf. If it is
/// evaluated for a particular value of x, the FFT and convolution is calculated
/// for all bins in the observable space for the given choice of parameters,
/// which are also stored in the cache. Subsequent evaluations for different values of the convolution observable and
/// identical parameters will be retrieved from the cache. If one or more
/// of the parameters change, the cache will be updated, *i.e.*, a new FFT runs.
///
/// The sampling density of the FFT is controlled by the binning of the
/// the convolution observable, which can be changed using RooRealVar::setBins(N).
/// For good results, N should be large (>=1000). Additional interpolation
/// between the bins may improve the result if coarse binnings are chosen. These can be
/// activated in the constructor or by calling `setInterpolationOrder()`.
/// For N >> 1000, interpolation will not substantially improve the accuracy.
///
/// Additionial information on caching can be displayed by monitoring
/// the message stream with topic "Caching" at the INFO level, *i.e.*
/// by calling `RooMsgService::instance().addStream(RooMsgService::INFO,Topic("Caching"))`
/// to see these message on stdout.
///
/// Multi-dimensional convolutions are not supported at the moment.
///

#include "RooFFTConvPdf.h"

#include "RooAbsReal.h"
#include "RooMsgService.h"
#include "RooDataHist.h"
#include "RooHistPdf.h"
#include "RooRealVar.h"
#include "RooGenContext.h"
#include "RooConvGenContext.h"
#include "RooBinning.h"
#include "RooLinearVar.h"
#include "RooCustomizer.h"
#include "RooGlobalFunc.h"
#include "RooConstVar.h"
#include "RooUniformBinning.h"
#include "RooFitImplHelpers.h"

#include "TClass.h"
#include "TMath.h"

#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

namespace {

using FFTComplex = std::complex<double>;

// Iterative in-place radix-2 Cooley-Tukey FFT. The length of `a` must be a power
// of two. `sign` is -1 for the forward DFT and +1 for the backward DFT (no
// normalization is applied in either direction, matching the FFTW convention).
void fftRadix2(std::vector<FFTComplex> &a, int sign)
{
   const std::size_t n = a.size();
   if (n < 2)
      return;

   // Bit-reversal permutation.
   for (std::size_t i = 1, j = 0; i < n; ++i) {
      std::size_t bit = n >> 1;
      for (; j & bit; bit >>= 1) {
         j ^= bit;
      }
      j ^= bit;
      if (i < j) {
         std::swap(a[i], a[j]);
      }
   }

   // Twiddle factors of the largest stage: tw[k] = exp(sign * 2*pi*i * k / n)
   // for k < n/2. A stage of length `len` uses the subset with stride n/len.
   // Computing each factor directly (instead of advancing a running phase by
   // repeated multiplication) keeps the round-off error at the epsilon level
   // independently of n.
   std::vector<FFTComplex> tw(n >> 1);
   const double angle = sign * 2.0 * TMath::Pi() / static_cast<double>(n);
   for (std::size_t k = 0; k < tw.size(); ++k) {
      tw[k] = std::polar(1.0, angle * static_cast<double>(k));
   }

   // Butterfly stages.
   for (std::size_t len = 2; len <= n; len <<= 1) {
      const std::size_t half = len >> 1;
      const std::size_t stride = n / len;
      for (std::size_t i = 0; i < n; i += len) {
         for (std::size_t k = 0; k < half; ++k) {
            const FFTComplex u = a[i + k];
            const FFTComplex v = a[i + k + half] * tw[k * stride];
            a[i + k] = u + v;
            a[i + k + half] = u - v;
         }
      }
   }
}

// Replicates the FFTW r2c -> multiply -> c2r pipeline used by RooFFTConvPdf.
// The convention is unnormalized (matching FFTW), so the output equals
// n * (input1 ⊛ input2) where ⊛ denotes circular convolution. The absolute
// scale does not matter since RooHistPdf renormalizes the cached histogram.
//
// If n is a power of two, the circular convolution is computed directly with
// two transforms of length n. Otherwise, the inputs are zero-padded to the
// smallest power of two m >= 2n - 1, which yields their linear convolution,
// and the circular convolution is recovered exactly by folding the result
// back modulo n. Both real inputs are packed into a single complex sequence
// z = input1 + i*input2, so that one forward transform yields both spectra
// via Hermitian symmetry.
void doFFT(int n, double const *input1, double const *input2, double *output)
{
   if (n <= 0)
      return;
   const std::size_t N = static_cast<std::size_t>(n);

   std::size_t m = N;
   if ((N & (N - 1)) != 0) {
      m = 1;
      while (m < 2 * N - 1) {
         m <<= 1;
      }
   }

   std::vector<FFTComplex> z(m, FFTComplex(0.0, 0.0));
   bool nonNegativeInputs = true;
   for (std::size_t k = 0; k < N; ++k) {
      nonNegativeInputs &= input1[k] >= 0.0 && input2[k] >= 0.0;
      z[k] = FFTComplex(input1[k], input2[k]);
   }
   fftRadix2(z, -1);

   // Unpack the spectra A = FFT(input1) and B = FFT(input2) from z and
   // overwrite z with the product A*B. Both spectra are Hermitian
   // (A[m-k] = conj(A[k])), and so is their product, so only half of the
   // bins need to be computed. Each iteration reads and writes the pair
   // (k, m-k), so the update can safely be done in place.
   for (std::size_t k = 0; k <= m / 2; ++k) {
      const std::size_t j = (m - k) & (m - 1); // (m - k) mod m
      const FFTComplex zk = z[k];
      const FFTComplex zjc = std::conj(z[j]);
      const FFTComplex ak = 0.5 * (zk + zjc);
      const FFTComplex bk = FFTComplex(0.0, -0.5) * (zk - zjc);
      const FFTComplex pk = ak * bk;
      z[k] = pk;
      z[j] = std::conj(pk);
   }
   fftRadix2(z, +1);

   // The unnormalized round trip yields m times the (linear) convolution.
   // Rescale to the documented n * (input1 ⊛ input2) convention, and for the
   // padded case fold the linear convolution (length 2n - 1 <= m) modulo n.
   const double scale = static_cast<double>(N) / static_cast<double>(m);
   for (std::size_t k = 0; k < N; ++k) {
      double val = z[k].real();
      if (m != N) {
         val += z[k + N].real();
      }
      val *= scale;
      // The convolution of two non-negative sequences is non-negative, but
      // round-off can push bins whose true value is ~0 slightly negative,
      // which RooFit later reports as pdf evaluation errors. Clamp those.
      if (nonNegativeInputs && val < 0.0) {
         val = 0.0;
      }
      output[k] = val;
   }
}

} // namespace

using std::endl, std::string, std::ostream;


////////////////////////////////////////////////////////////////////////////////
/// Constructor for numerical (FFT) convolution of PDFs.
/// \param[in] name Name of this PDF
/// \param[in] title Title for plotting this PDF
/// \param[in] convVar Observable to convolve the PDFs in \attention Use a high number of bins (>= 1000) for good accuracy.
/// \param[in] pdf1 First PDF to be convolved
/// \param[in] pdf2 Second PDF to be convolved
/// \param[in] ipOrder Order for interpolation between bins (since FFT is discrete)
/// The binning used for the FFT sampling is controlled by the binning named "cache" in the convolution observable `convVar`.
/// If such a binning is not set, the same number of bins as for `convVar` will be used.

RooFFTConvPdf::RooFFTConvPdf(const char *name, const char *title, RooRealVar &convVar, RooAbsPdf &pdf1, RooAbsPdf &pdf2,
                             Int_t ipOrder)
   : RooAbsCachedPdf(name, title, ipOrder),
     _x("!x", "Convolution Variable", this, convVar),
     _xprime("!xprime", "External Convolution Variable", this, false),
     _pdf1("!pdf1", "pdf1", this, pdf1, false),
     _pdf2("!pdf2", "pdf2", this, pdf2, false),
     _params("!params", "effective parameters", this),
     _bufFrac(0.1),
     _bufStrat(Extend),
     _shift1(0),
     _shift2((convVar.getMax("cache") + convVar.getMin("cache")) / 2),
     _cacheObs("!cacheObs", "Cached observables", this, false, false)
{
  prepareFFTBinning(convVar);

  calcParams() ;

}

////////////////////////////////////////////////////////////////////////////////
/// \copydoc RooFFTConvPdf(const char*, const char*, RooRealVar&, RooAbsPdf&, RooAbsPdf&, Int_t)
/// \param[in] pdfConvVar If the variable used for convolution is a function, itself, pass the function here, and pass
/// the convolution variable to `convVar`. See also rf210_angularconv.C in the <a
/// href="https://root.cern/root/html/tutorials/roofit/index.html.">roofit tutorials</a>

RooFFTConvPdf::RooFFTConvPdf(const char *name, const char *title, RooAbsReal &pdfConvVar, RooRealVar &convVar,
                             RooAbsPdf &pdf1, RooAbsPdf &pdf2, Int_t ipOrder)
   : RooAbsCachedPdf(name, title, ipOrder),
     _x("!x", "Convolution Variable", this, convVar, false, false),
     _xprime("!xprime", "External Convolution Variable", this, pdfConvVar),
     _pdf1("!pdf1", "pdf1", this, pdf1, false),
     _pdf2("!pdf2", "pdf2", this, pdf2, false),
     _params("!params", "effective parameters", this),
     _bufFrac(0.1),
     _bufStrat(Extend),
     _shift1(0),
     _shift2((convVar.getMax("cache") + convVar.getMin("cache")) / 2),
     _cacheObs("!cacheObs", "Cached observables", this, false, false)
{
  prepareFFTBinning(convVar);

  calcParams() ;
}



////////////////////////////////////////////////////////////////////////////////
/// Copy constructor

RooFFTConvPdf::RooFFTConvPdf(const RooFFTConvPdf& other, const char* name) :
  RooAbsCachedPdf(other,name),
  _x("!x",this,other._x),
  _xprime("!xprime",this,other._xprime),
  _pdf1("!pdf1",this,other._pdf1),
  _pdf2("!pdf2",this,other._pdf2),
  _params("!params",this,other._params),
  _bufFrac(other._bufFrac),
  _bufStrat(other._bufStrat),
  _shift1(other._shift1),
  _shift2(other._shift2),
  _cacheObs("!cacheObs",this,other._cacheObs)
 {
 }



////////////////////////////////////////////////////////////////////////////////
/// Destructor

RooFFTConvPdf::~RooFFTConvPdf()
{
}


////////////////////////////////////////////////////////////////////////////////
/// Try to improve the binning and inform user if possible.
/// With a 10% buffer fraction, 930 raw bins yield 1024 FFT bins,
/// a sweet spot for the speed of FFTW.

void RooFFTConvPdf::prepareFFTBinning(RooRealVar& convVar) const {
  if (!convVar.hasBinning("cache")) {
    const RooAbsBinning& varBinning = convVar.getBinning();
    const int optimal = static_cast<Int_t>(1024/(1.+_bufFrac));

    //Can improve precision if binning is uniform
    if (varBinning.numBins() < optimal && varBinning.isUniform()) {
      coutI(Caching) << "Changing internal binning of variable '" << convVar.GetName()
          << "' in FFT '" << fName << "'"
          << " from " << varBinning.numBins()
          << " to " << optimal << " to improve the precision of the numerical FFT."
          << " This can be done manually by setting an additional binning named 'cache'." << std::endl;
      convVar.setBinning(RooUniformBinning(varBinning.lowBound(), varBinning.highBound(), optimal, "cache"), "cache");
    } else {
      coutE(Caching) << "The internal binning of variable " << convVar.GetName()
          << " is not uniform. The numerical FFT will likely yield wrong results." << std::endl;
      convVar.setBinning(varBinning, "cache");
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Return base name component for cache components in this case 'PDF1_CONV_PDF2'

const char* RooFFTConvPdf::inputBaseName() const
{
  static TString name ;
  name = _pdf1.arg().GetName() ;
  name.Append("_CONV_") ;
  name.Append(_pdf2.arg().GetName()) ;
  return name.Data() ;
}




////////////////////////////////////////////////////////////////////////////////
/// Return specialized cache subclass for FFT calculations

RooFFTConvPdf::PdfCacheElem* RooFFTConvPdf::createCache(const RooArgSet* nset) const
{
  return new FFTCacheElem(*this,nset) ;
}




////////////////////////////////////////////////////////////////////////////////
/// Clone input pdf and attach to dataset

RooFFTConvPdf::FFTCacheElem::FFTCacheElem(const RooFFTConvPdf& self, const RooArgSet* nsetIn) :
  PdfCacheElem(self,nsetIn)
{
  RooAbsPdf* clonePdf1 = static_cast<RooAbsPdf*>(self._pdf1.arg().cloneTree()) ;
  RooAbsPdf* clonePdf2 = static_cast<RooAbsPdf*>(self._pdf2.arg().cloneTree()) ;
  clonePdf1->attachDataSet(*hist()) ;
  clonePdf2->attachDataSet(*hist()) ;

   // Shift observable
   RooRealVar* convObs = static_cast<RooRealVar*>(hist()->get()->find(self._x.arg().GetName())) ;

   // Install FFT reference range
   string refName = Form("refrange_fft_%s",self.GetName()) ;
   convObs->setRange(refName.c_str(),convObs->getMin(),convObs->getMax()) ;

   if (self._shift1!=0) {
     RooLinearVar* shiftObs1 = new RooLinearVar(Form("%s_shifted_FFTBuffer1",convObs->GetName()),"shiftObs1",
                      *convObs,RooFit::RooConst(1),RooFit::RooConst(-1*self._shift1)) ;

    RooArgSet clonedBranches1 ;
    RooCustomizer cust(*clonePdf1,"fft") ;
    cust.replaceArg(*convObs,*shiftObs1) ;

    pdf1Clone.reset(static_cast<RooAbsPdf*>(cust.build()));

    pdf1Clone->addOwnedComponents(*shiftObs1) ;
    pdf1Clone->addOwnedComponents(*clonePdf1) ;

  } else {
    pdf1Clone.reset(clonePdf1) ;
  }

  if (self._shift2!=0) {
    RooLinearVar* shiftObs2 = new RooLinearVar(Form("%s_shifted_FFTBuffer2",convObs->GetName()),"shiftObs2",
                      *convObs,RooFit::RooConst(1),RooFit::RooConst(-1*self._shift2)) ;

    RooArgSet clonedBranches2 ;
    RooCustomizer cust(*clonePdf2,"fft") ;
    cust.replaceArg(*convObs,*shiftObs2) ;

    pdf1Clone->addOwnedComponents(*shiftObs2) ;
    pdf1Clone->addOwnedComponents(*clonePdf2) ;

    pdf2Clone.reset(static_cast<RooAbsPdf*>(cust.build()));

  } else {
    pdf2Clone.reset(clonePdf2) ;
  }

  // Cache normalization integral values, since we know they don't change for
  // the given normalization set in this cache object. When using this cache,
  // we evaluate the pdfs without normalization set and then do the
  // normalization manually using these cached values. This has less overhead
  // compared to letting RooAbsPdf::getVal(normSet) figure out if the normSet
  // has changed and get the caching right.
  normVal1 = pdf1Clone->getNorm(hist()->get());
  normVal2 = pdf2Clone->getNorm(hist()->get());

  // Attach cloned pdf to all original parameters of self
  RooArgSet convObsSet{*convObs};
  RooArgSet fftParams;
  self.getParameters(&convObsSet, fftParams) ;

  // Remove all cache histogram from fftParams as these
  // observable need to remain attached to the histogram
  fftParams.remove(*hist()->get(),true,true) ;

  pdf1Clone->recursiveRedirectServers(fftParams) ;
  pdf2Clone->recursiveRedirectServers(fftParams) ;
  pdf1Clone->fixAddCoefRange(refName.c_str(), true) ;
  pdf2Clone->fixAddCoefRange(refName.c_str(), true) ;

  // Ensure that coefficients for Add PDFs are only interpreted with respect to the convolution observable
  RooArgSet convSet(self._x.arg());
  pdf1Clone->fixAddCoefNormalization(convSet, true);
  pdf2Clone->fixAddCoefNormalization(convSet, true);

  // Save copy of original histX binning and make alternate binning
  // for extended range scanning

  const Int_t N = convObs->numBins();
  if (N < 900) {
    oocoutW(&self, Eval) << "The FFT convolution '" << self.GetName() << "' will run with " << N
        << " bins. A decent accuracy for difficult convolutions is typically only reached with n >= 1000. Suggest to increase the number"
        " of bins of the observable '" << convObs->GetName() << "'." << std::endl;
  }
  Int_t Nbuf = static_cast<Int_t>((N*self.bufferFraction())/2 + 0.5) ;
  double obw = (convObs->getMax() - convObs->getMin())/N ;
  Int_t N2 = N+2*Nbuf ;

  scanBinning = std::make_unique<RooUniformBinning>(convObs->getMin()-Nbuf*obw,convObs->getMax()+Nbuf*obw,N2);
  histBinning.reset(convObs->getBinning().clone());

  // Deactivate dirty state propagation on datahist observables
  // and set all nodes on both pdfs to operMode AlwaysDirty
  hist()->setDirtyProp(false) ;
  convObs->setOperMode(ADirty,true) ;
}


////////////////////////////////////////////////////////////////////////////////
/// Suffix for cache histogram (added in addition to suffix for cache name)

TString RooFFTConvPdf::histNameSuffix() const
{
  return TString(Form("_BufFrac%3.1f_BufStrat%d",_bufFrac,_bufStrat)) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Returns all RooAbsArg objects contained in the cache element

RooArgList RooFFTConvPdf::FFTCacheElem::containedArgs(Action a)
{
  RooArgList ret(PdfCacheElem::containedArgs(a)) ;

  ret.add(*pdf1Clone) ;
  ret.add(*pdf2Clone) ;
  if (pdf1Clone->ownedComponents()) {
    ret.add(*pdf1Clone->ownedComponents()) ;
  }
  if (pdf2Clone->ownedComponents()) {
    ret.add(*pdf2Clone->ownedComponents()) ;
  }

  return ret ;
}



////////////////////////////////////////////////////////////////////////////////
/// Fill the contents of the cache the FFT convolution output

void RooFFTConvPdf::fillCacheObject(RooAbsCachedPdf::PdfCacheElem& cache) const
{
  RooDataHist& cacheHist = *cache.hist() ;

  (static_cast<FFTCacheElem&>(cache)).pdf1Clone->setOperMode(ADirty,true) ;
  (static_cast<FFTCacheElem&>(cache)).pdf2Clone->setOperMode(ADirty,true) ;

  // Determine if there other observables than the convolution observable in the cache
  RooArgSet otherObs ;
  RooArgSet(*cacheHist.get()).snapshot(otherObs) ;

  RooAbsArg* histArg = otherObs.find(_x.arg().GetName()) ;
  if (histArg) {
    otherObs.remove(*histArg,true,true) ;
  }

  //cout << "RooFFTConvPdf::fillCacheObject() otherObs = " << otherObs << std::endl ;

  // Handle trivial scenario -- no other observables
  if (otherObs.empty()) {
    fillCacheSlice(static_cast<FFTCacheElem&>(cache),RooArgSet()) ;
    return ;
  }

  // Handle cases where there are other cache slices
  // Iterator over available slice positions and fill each

  // Determine number of bins for each slice position observable
  Int_t n = otherObs.size() ;
  std::vector<int> binCur(n+1);
  std::vector<int> binMax(n+1);
  Int_t curObs = 0 ;

  std::vector<RooAbsLValue*> obsLV(n);
  Int_t i(0) ;
  for (auto* lvarg : dynamic_range_cast<RooAbsLValue*>(otherObs)) {
    obsLV[i] = lvarg ;
    binCur[i] = 0 ;
    // coverity[FORWARD_NULL]
    binMax[i] = lvarg->numBins(binningName())-1 ;
    i++ ;
  }

  bool loop(true) ;
  while(loop) {
    // Set current slice position
    for (Int_t j=0 ; j<n ; j++) { obsLV[j]->setBin(binCur[j],binningName()) ; }

//     std::cout << "filling slice: bin of obsLV[0] = " << obsLV[0]->getBin() << std::endl ;

    // Fill current slice
    fillCacheSlice(static_cast<FFTCacheElem&>(cache),otherObs) ;

    // Determine which iterator to increment
    while(binCur[curObs]==binMax[curObs]) {

      // Reset current iterator and consider next iterator ;
      binCur[curObs]=0 ;
      curObs++ ;

      // master termination condition
      if (curObs==n) {
   loop=false ;
   break ;
      }
    }

    // Increment current iterator
    binCur[curObs]++ ;
    curObs=0 ;

  }

}


////////////////////////////////////////////////////////////////////////////////
/// Fill a slice of cachePdf with the output of the FFT convolution calculation

void RooFFTConvPdf::fillCacheSlice(FFTCacheElem& aux, const RooArgSet& slicePos) const
{
  // Extract histogram that is the basis of the RooHistPdf
  RooDataHist& cacheHist = *aux.hist() ;

  // Sample array of input points from both pdfs
  // Note that returned arrays have optional buffers zones below and above range ends
  // to reduce cyclical effects and have been cyclically rotated so that bin containing
  // zero value is at position zero. Example:
  //
  //     original:                -5 -4 -3 -2 -1 0 +1 +2 +3 +4 +5
  //     add buffer zones:    U U -5 -4 -3 -2 -1 0 +1 +2 +3 +4 +5 O O
  //     rotate:              0 +1 +2 +3 +4 +5 O O U U -5 -4 -3 -2 -1
  //
  //

  Int_t N;
  Int_t N2;
  Int_t binShift1;
  Int_t binShift2;

  RooRealVar* histX = static_cast<RooRealVar*>(cacheHist.get()->find(_x.arg().GetName())) ;
  if (_bufStrat==Extend) histX->setBinning(*aux.scanBinning) ;
  RooRealVar &xVar = const_cast<RooRealVar &>(static_cast<RooRealVar const&>(_x.arg()));
  std::vector<double> input1 = scanPdf(xVar,*aux.pdf1Clone,aux.normVal1,cacheHist,slicePos,N,N2,binShift1,_shift1) ;
  std::vector<double> input2 = scanPdf(xVar,*aux.pdf2Clone,aux.normVal2,cacheHist,slicePos,N,N2,binShift2,_shift2) ;
  if (_bufStrat==Extend) histX->setBinning(*aux.histBinning) ;

  std::vector<double> output(N2);
  doFFT(N2, input1.data(), input2.data(), output.data());

  Int_t totalShift = binShift1 + (N2-N)/2 ;

  // Store FFT result in cache

  std::unique_ptr<TIterator> iter{const_cast<RooDataHist&>(cacheHist).sliceIterator(const_cast<RooAbsReal&>(_x.arg()),slicePos)};
  for (Int_t i =0 ; i<N ; i++) {

    // Cyclically shift array back so that bin containing zero is back in zeroBin
    Int_t j = i + totalShift ;
    while (j<0) j+= N2 ;
    while (j>=N2) j-= N2 ;

    iter->Next() ;
    const std::size_t binIdx = cacheHist.getIndex(*cacheHist.get(), /*fast=*/true);
    cacheHist.set(binIdx, output[j], -1.);
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Scan the values of 'pdf' in observable 'obs' using the bin values stored in 'hist' at slice position 'slicePos'
/// N is filled with the number of bins defined in hist, N2 is filled with N plus the number of buffer bins
/// The return value is an array of doubles of length N2 with the sampled values. The caller takes ownership
/// of the array

std::vector<double> RooFFTConvPdf::scanPdf(RooRealVar &obs, RooAbsPdf &pdf, double normVal, const RooDataHist &hist,
                                           const RooArgSet &slicePos, Int_t &N, Int_t &N2, Int_t &zeroBin,
                                           double shift) const
{

  RooRealVar* histX = static_cast<RooRealVar*>(hist.get()->find(obs.GetName())) ;

  // Calculate number of buffer bins on each size to avoid cyclical flow
  N = histX->numBins(binningName()) ;
  Int_t Nbuf = static_cast<Int_t>((N*bufferFraction())/2 + 0.5) ;
  N2 = N+2*Nbuf ;


  // Allocate array of sampling size plus optional buffer zones
  std::vector<double> array(N2);

  // Set position of non-convolution observable to that of the cache slice that were are processing now
  hist.get(slicePos) ;

  // Find bin ID that contains zero value
  zeroBin = 0 ;
  if (histX->getMax()>=0 && histX->getMin()<=0) {
    zeroBin = histX->getBinning().binNumber(0) ;
  } else if (histX->getMin()>0) {
    double bw = (histX->getMax() - histX->getMin())/N2 ;
    zeroBin = Int_t(-histX->getMin()/bw) ;
  } else {
    double bw = (histX->getMax() - histX->getMin())/N2 ;
    zeroBin = Int_t(-1*histX->getMax()/bw) ;
  }

  Int_t binShift = Int_t((N2* shift) / (histX->getMax()-histX->getMin())) ;

  zeroBin += binShift ;
  while(zeroBin>=N2) zeroBin-= N2 ;
  while(zeroBin<0) zeroBin+= N2 ;

  // To mimic exactly the normalization code in RooAbsPdf::getValV()
  auto getPdfVal = [&]() {
     double rawVal = pdf.getVal();
     return RooFit::Detail::normalizeWithNaNPacking(pdf, rawVal, normVal);
  };

  // First scan hist into temp array
  std::vector<double> tmp(N2);
  Int_t k(0) ;
  switch(_bufStrat) {

  case Extend:
    // Sample entire extended range (N2 samples)
    for (k=0 ; k<N2 ; k++) {
      histX->setBin(k) ;
      tmp[k] = getPdfVal();
    }
    break ;

  case Flat:
    // Sample original range (N samples) and fill lower and upper buffer
    // bins with p.d.f. value at respective boundary
    {
      histX->setBin(0) ;
      double val = getPdfVal();
      for (k=0 ; k<Nbuf ; k++) {
   tmp[k] = val ;
      }
      for (k=0 ; k<N ; k++) {
   histX->setBin(k) ;
   tmp[k+Nbuf] = getPdfVal();
      }
      histX->setBin(N-1) ;
      val = getPdfVal();
      for (k=0 ; k<Nbuf ; k++) {
   tmp[N+Nbuf+k] = val ;
      }
    }
    break ;

  case Mirror:
    // Sample original range (N samples) and fill lower and upper buffer
    // bins with mirror image of sampled range
    for (k=0 ; k<N ; k++) {
      histX->setBin(k) ;
      tmp[k+Nbuf] = getPdfVal();
    }
    for (k=1 ; k<=Nbuf ; k++) {
      histX->setBin(k) ;
      tmp[Nbuf-k] = getPdfVal();
      histX->setBin(N-k) ;
      tmp[Nbuf+N+k-1] = getPdfVal();
    }
    break ;
  }

  // Scan function and store values in array
  for (Int_t i=0 ; i<N2 ; i++) {
    // Cyclically shift writing location by zero bin position
    Int_t j = i - (zeroBin) ;
    if (j<0) j+= N2 ;
    if (j>=N2) j-= N2 ;
    array[i] = tmp[j] ;
  }

  return array ;
}



////////////////////////////////////////////////////////////////////////////////
/// Return the observables to be cached given the normalization set nset.
///
/// If the cache observable is in nset then this is
///    - the convolution observable plus
///    - any member of nset that is either a RooCategory,
///    - or was previously specified through setCacheObservables().
///
/// In case the cache observable is *not* in nset, then it is
///    - the convolution observable plus
///    - all member of nset that are observables of this p.d.f.
///

RooFit::OwningPtr<RooArgSet> RooFFTConvPdf::actualObservables(const RooArgSet& nset) const
{
  // Get complete list of observables
  auto obs1 = new RooArgSet{};
  RooArgSet obs2;
  _pdf1.arg().getObservables(&nset, *obs1) ;
  _pdf2.arg().getObservables(&nset, obs2) ;
  obs1->add(obs2, true) ;

  // Check if convolution observable is in nset
  if (nset.contains(_x.arg())) {

    // Now strip out all non-category observables
    RooArgSet killList ;
    for(auto const& arg : *obs1) {
      if (arg->IsA()->InheritsFrom(RooAbsReal::Class()) && !_cacheObs.find(arg->GetName())) {
        killList.add(*arg) ;
      }
    }
    obs1->remove(killList) ;

    // And add back the convolution observables
    obs1->add(_x.arg(),true) ;

    obs1->add(_cacheObs) ;

  } else {

    // If cacheObs was filled, cache only observables in there
    if (!_cacheObs.empty()) {
      RooArgSet killList ;
      for(auto const& arg : *obs1) {
        if (arg->IsA()->InheritsFrom(RooAbsReal::Class()) && !_cacheObs.find(arg->GetName())) {
          killList.add(*arg) ;
        }
      }
      obs1->remove(killList) ;
    }


    // Make sure convolution observable is always in there
    obs1->add(_x.arg(),true) ;

  }

  return RooFit::OwningPtr<RooArgSet>{obs1};
}



////////////////////////////////////////////////////////////////////////////////
/// Return the parameters on which the cache depends given normalization
/// set nset. For this p.d.f these are the parameters of the input p.d.f.
/// but never the convolution variable, in case it is not part of nset.

RooFit::OwningPtr<RooArgSet> RooFFTConvPdf::actualParameters(const RooArgSet& nset) const
{
  std::unique_ptr<RooArgSet> vars{getVariables()};
  vars->remove(*std::unique_ptr<RooArgSet>{actualObservables(nset)});

  return RooFit::makeOwningPtr(std::move(vars));
}



////////////////////////////////////////////////////////////////////////////////
/// Return p.d.f. observable (which can be a function) to substitute given
/// p.d.f. observable. Substitutes x by xprime if xprime is set.

RooAbsArg& RooFFTConvPdf::pdfObservable(RooAbsArg& histObservable) const
{
  if (_xprime.absArg() && string(histObservable.GetName())==_x.absArg()->GetName()) {
    return (*_xprime.absArg()) ;
  }
  return histObservable ;
}



////////////////////////////////////////////////////////////////////////////////
/// Create appropriate generator context for this convolution. If both input p.d.f.s support
/// internal generation, if it is safe to use them and if no observables other than the convolution
/// observable are requested for generation, use the specialized convolution generator context
/// which implements a smearing strategy in the convolution observable. If not return the
/// regular accept/reject generator context

RooAbsGenContext* RooFFTConvPdf::genContext(const RooArgSet &vars, const RooDataSet *prototype,
                   const RooArgSet* auxProto, bool verbose) const
{
  RooArgSet vars2(vars) ;
  vars2.remove(_x.arg(),true,true) ;
  Int_t numAddDep = vars2.size() ;

  RooArgSet dummy ;
  bool pdfCanDir = ((static_cast<RooAbsPdf const &>(_pdf1.arg())).getGenerator(_x.arg(), dummy) != 0 &&
                    (static_cast<RooAbsPdf const &>(_pdf1.arg())).isDirectGenSafe(_x.arg()));
  bool resCanDir = ((static_cast<RooAbsPdf const &>(_pdf2.arg())).getGenerator(_x.arg(), dummy) != 0 &&
                    (static_cast<RooAbsPdf const &>(_pdf2.arg())).isDirectGenSafe(_x.arg()));

  if (pdfCanDir) {
    cxcoutI(Generation) << "RooFFTConvPdf::genContext() input p.d.f " << _pdf1.arg().GetName()
         << " has internal generator that is safe to use in current context" << std::endl ;
  }
  if (resCanDir) {
    cxcoutI(Generation) << "RooFFTConvPdf::genContext() input p.d.f. " << _pdf2.arg().GetName()
         << " has internal generator that is safe to use in current context" << std::endl ;
  }
  if (numAddDep>0) {
    cxcoutI(Generation) << "RooFFTConvPdf::genContext() generation requested for observables other than the convolution observable " << _x.arg().GetName() << std::endl ;
  }


  if (numAddDep>0 || !pdfCanDir || !resCanDir) {
    // Any resolution model with more dependents than the convolution variable
    // or pdf or resmodel do not support direct generation
    cxcoutI(Generation) << "RooFFTConvPdf::genContext() selecting accept/reject generator context because one or both of the input "
         << "p.d.f.s cannot use internal generator and/or "
         << "observables other than the convolution variable are requested for generation" << std::endl ;
    return new RooGenContext(*this,vars,prototype,auxProto,verbose) ;
  }

  // Any other resolution model: use specialized generator context
  cxcoutI(Generation) << "RooFFTConvPdf::genContext() selecting specialized convolution generator context as both input "
            << "p.d.fs are safe for internal generator and only "
            << "the convolution observables is requested for generation" << std::endl ;
  return new RooConvGenContext(*this,vars,prototype,auxProto,verbose) ;
}



////////////////////////////////////////////////////////////////////////////////
/// Change the size of the buffer on either side of the observable range to `frac` times the
/// size of the range of the convolution observable.

void RooFFTConvPdf::setBufferFraction(double frac)
{
  if (frac<0) {
    coutE(InputArguments) << "RooFFTConvPdf::setBufferFraction(" << GetName() << ") fraction should be greater than or equal to zero" << std::endl ;
    return ;
  }
  _bufFrac = frac ;

  // Sterilize the cache as certain partial results depend on buffer fraction
  _cacheMgr.sterilize() ;
}


////////////////////////////////////////////////////////////////////////////////
/// Change strategy to fill the overflow buffer on either side of the convolution observable range.
///
/// - `Extend` means is that the input p.d.f convolution observable range is widened to include the buffer range
/// - `Flat` means that the buffer is filled with the p.d.f. value at the boundary of the observable range
/// - `Mirror` means that the buffer is filled with a mirror image of the p.d.f. around the convolution observable boundary
///
/// The default strategy is extend. If one of the input p.d.f.s is a RooAddPdf, it is configured so that the interpretation
/// range of the fraction coefficients is kept at the nominal convolutions observable range (instead of interpreting coefficients
/// in the widened range including the buffer).

void RooFFTConvPdf::setBufferStrategy(BufStrat bs)
{
  _bufStrat = bs ;
}



////////////////////////////////////////////////////////////////////////////////
/// Customized printing of arguments of a RooNumConvPdf to more intuitively reflect the contents of the
/// product operator construction

void RooFFTConvPdf::printMetaArgs(ostream& os) const
{
  os << _pdf1.arg().GetName() << "(" << _x.arg().GetName() << ") (*) " << _pdf2.arg().GetName() << "(" << _x.arg().GetName() << ") " ;
}



////////////////////////////////////////////////////////////////////////////////
/// (Re)calculate effective parameters of this p.d.f.

void RooFFTConvPdf::calcParams()
{
  RooArgSet argSet{_x.arg()};
  RooArgSet params1;
  RooArgSet params2;
  _pdf1.arg().getParameters(&argSet, params1);
  _pdf2.arg().getParameters(&argSet, params2);
  _params.removeAll() ;
  _params.add(params1) ;
  _params.add(params2,true) ;
}
