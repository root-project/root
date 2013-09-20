
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
//
 // 
 // This class implement a generic one-dimensional numeric convolution of two p.d.f.
 // and can convolve any two RooAbsPdfs. The class exploits the convolution theorem
 //
 //       f(x) (*) g(x) --F--> f(k_i) * g(k_i)
 //
 // and calculate the convolution by calculate a Real->Complex FFT of both input p.d.fs
 // multiplying the complex coefficients and performing the reverse Complex->Real FFT
 // to get the result in the input space. This class using the ROOT FFT Interface to
 // the (free) FFTW3 package (www.fftw.org) and requires that your ROOT installation is
 // compiled with the --enable-fftw3 option (instructions for Linux follow)
 //
 // Note that the performance in terms of speed and stability of RooFFTConvPdf is 
 // vastly superior to that of RooNumConvPdf 
 //
 // An important feature of FFT convolutions is that the observable is treated in a
 // cyclical way. This is correct & desirable behavior for cyclical observables such as angles,
 // but it may not be for other observables. The effect that is observed is that if
 // p.d.f is zero at xMin and non-zero at xMax some spillover occurs and
 // a rising tail may appear at xMin. This effect can be reduced or eliminated by
 // introducing a buffer zone in the FFT calculation. If this feature is activated
 // input the sampling array for the FFT calculation is extended in both directions
 // and filled with repetitions of the lowest bin value and highest bin value
 // respectively. The buffer bins are stripped again when the FFT output values
 // are transferred to the p.d.f cache. The default buffer size is 10% of the
 // observable domain size and can be changed with setBufferFraction() member function.
 // 
 // This class is a caching p.d.f inheriting from RooAbsCachedPdf. If this p.d.f 
 // is evaluated for a particular value of x, the FFT calculate the values for the
 // p.d.f at all points in observables space for the given choice of parameters,
 // which are stored in the cache. Subsequent evaluations of RooFFTConvPdf with
 // identical parameters will retrieve results from the cache. If one or more
 // of the parameters change, the cache will be updated.
 // 
 // The sampling density of the cache is controlled by the binning of the 
 // the convolution observable, which can be changed from RooRealVar::setBins(N)
 // For good results N should be large (>1000). Additional interpolation of
 // cache values may improve the result if courser binning are chosen. These can be 
 // set in the constructor or through the setInterpolationOrder() member function. 
 // For N>1000 interpolation will not substantially improve the performance.
 //
 // Additionial information on caching activities can be displayed by monitoring
 // the message stream with topic "Caching" at the INFO level, i.e. 
 // do RooMsgService::instance().addStream(RooMsgService::INFO,Topic("Caching")) 
 // to see these message on stdout
 //
 // Multi-dimensional convolutions are not supported yet, but will be in the future
 // as FFTW can calculate them
 //
 // ---
 // 
 // Installing a copy of FFTW on Linux and compiling ROOT to use it
 // 
 // 1) Go to www.fftw.org and download the latest stable version (a .tar.gz file)
 //
 // If you have root access to your machine and want to make a system installation of FFTW
 //
 //   2) Untar fftw-XXX.tar.gz in /tmp, cd into the untarred directory 
 //       and type './configure' followed by 'make install'. 
 //       This will install fftw in /usr/local/bin,lib etc...
 //
 //   3) Start from a source installation of ROOT. If you now have a binary distribution,
 //      first download a source tar ball from root.cern.ch for your ROOT version and untar it.
 //      Run 'configure', following the instruction from 'configure --help' but be sure run 'configure' 
 //      with additional flags '--enable-fftw3' and '--enable-roofit', then run 'make'
 //         
 // 
 // If you do not have root access and want to make a private installation of FFTW
 //
 //   2) Make a private install area for FFTW, e.g. /home/myself/fftw
 //
 //   3) Untar fftw-XXX.tar.gz in /tmp, cd into the untarred directory
 //       and type './configure --prefix=/home/myself/fftw' followed by 'make install'. 
 //       Substitute /home/myself/fftw with a directory of your choice. This
 //       procedure will install FFTW in the location designated by you
 // 
 //   4) Start from a source installation of ROOT. If you now have a binary distribution,
 //      first download a source tar ball from root.cern.ch for your ROOT version and untar it.
 //      Run 'configure', following the instruction from 'configure --help' but be sure run 'configure' 
 //      with additional flags
 //       '--enable-fftw3', 
 //       '--with-fftw3-incdir=/home/myself/fftw/include', 
 //       '--width-fftw3-libdir=/home/myself/fftw/lib' and 
 //       '--enable-roofit' 
 //      Then run 'make'
//


#include "Riostream.h" 

#include "RooFit.h"
#include "RooFFTConvPdf.h" 
#include "RooAbsReal.h" 
#include "RooMsgService.h"
#include "RooDataHist.h"
#include "RooHistPdf.h"
#include "RooRealVar.h"
#include "TComplex.h"
#include "TVirtualFFT.h"
#include "RooGenContext.h"
#include "RooConvGenContext.h"
#include "RooBinning.h"
#include "RooLinearVar.h"
#include "RooCustomizer.h"
#include "RooGlobalFunc.h"
#include "RooLinearVar.h"
#include "RooConstVar.h"
#include "TClass.h"
#include "TSystem.h"

using namespace std ;

ClassImp(RooFFTConvPdf) 



//_____________________________________________________________________________
RooFFTConvPdf::RooFFTConvPdf(const char *name, const char *title, RooRealVar& convVar, RooAbsPdf& pdf1, RooAbsPdf& pdf2, Int_t ipOrder) :
  RooAbsCachedPdf(name,title,ipOrder),
  _x("!x","Convolution Variable",this,convVar),
  _xprime("!xprime","External Convolution Variable",this,0),
  _pdf1("!pdf1","pdf1",this,pdf1,kFALSE),
  _pdf2("!pdf2","pdf2",this,pdf2,kFALSE),
  _params("!params","effective parameters",this),
  _bufFrac(0.1),
  _bufStrat(Extend),
  _shift1(0),
  _shift2(0),
  _cacheObs("!cacheObs","Cached observables",this,kFALSE,kFALSE)
 { 
   // Constructor for convolution of pdf1 (x) pdf2 in observable convVar. The binning used for the FFT sampling is controlled
   // by the binning named "cache" in the convolution observable. The resulting FFT convolved histogram is interpolated at
   // order 'ipOrder' A minimum binning of 1000 bins is recommended.

   if (!convVar.hasBinning("cache")) {
     convVar.setBinning(convVar.getBinning(),"cache") ;
   }
   
   _shift2 = (convVar.getMax("cache")+convVar.getMin("cache"))/2 ;

   calcParams() ;

 } 



//_____________________________________________________________________________
RooFFTConvPdf::RooFFTConvPdf(const char *name, const char *title, RooAbsReal& pdfConvVar, RooRealVar& convVar, RooAbsPdf& pdf1, RooAbsPdf& pdf2, Int_t ipOrder) :
  RooAbsCachedPdf(name,title,ipOrder),
  _x("!x","Convolution Variable",this,convVar,kFALSE,kFALSE),
  _xprime("!xprime","External Convolution Variable",this,pdfConvVar),
  _pdf1("!pdf1","pdf1",this,pdf1,kFALSE),
  _pdf2("!pdf2","pdf2",this,pdf2,kFALSE),
  _params("!params","effective parameters",this),
  _bufFrac(0.1),
  _bufStrat(Extend),
  _shift1(0),
  _shift2(0),
  _cacheObs("!cacheObs","Cached observables",this,kFALSE,kFALSE)
 { 
   // Constructor for convolution of pdf1 (x) pdf2 in observable convVar. The binning used for the FFT sampling is controlled
   // by the binning named "cache" in the convolution observable. The resulting FFT convolved histogram is interpolated at
   // order 'ipOrder' A minimum binning of 1000 bins is recommended.

   if (!convVar.hasBinning("cache")) {
     convVar.setBinning(convVar.getBinning(),"cache") ;
   }
   
   _shift2 = (convVar.getMax("cache")+convVar.getMin("cache"))/2 ;

   calcParams() ;
 } 



//_____________________________________________________________________________
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
   // Copy constructor
 } 



//_____________________________________________________________________________
RooFFTConvPdf::~RooFFTConvPdf() 
{
  // Destructor 
}



//_____________________________________________________________________________
const char* RooFFTConvPdf::inputBaseName() const 
{
  // Return base name component for cache components in this case 'PDF1_CONV_PDF2'

  static TString name ;
  name = _pdf1.arg().GetName() ;
  name.Append("_CONV_") ;
  name.Append(_pdf2.arg().GetName()) ;
  return name.Data() ;
}




//_____________________________________________________________________________
RooFFTConvPdf::PdfCacheElem* RooFFTConvPdf::createCache(const RooArgSet* nset) const 
{
  // Return specialized cache subclass for FFT calculations
  return new FFTCacheElem(*this,nset) ;
}




//_____________________________________________________________________________
RooFFTConvPdf::FFTCacheElem::FFTCacheElem(const RooFFTConvPdf& self, const RooArgSet* nsetIn) : 
  PdfCacheElem(self,nsetIn),
  fftr2c1(0),fftr2c2(0),fftc2r(0) 
{
  // Clone input pdf and attach to dataset
  RooAbsPdf* clonePdf1 = (RooAbsPdf*) self._pdf1.arg().cloneTree() ;
  RooAbsPdf* clonePdf2 = (RooAbsPdf*) self._pdf2.arg().cloneTree() ;
  clonePdf1->attachDataSet(*hist()) ;
  clonePdf2->attachDataSet(*hist()) ;

   // Shift observable
   RooRealVar* convObs = (RooRealVar*) hist()->get()->find(self._x.arg().GetName()) ;

   // Install FFT reference range 
   string refName = Form("refrange_fft_%s",self.GetName()) ;
   convObs->setRange(refName.c_str(),convObs->getMin(),convObs->getMax()) ;   

   if (self._shift1!=0) {
     RooLinearVar* shiftObs1 = new RooLinearVar(Form("%s_shifted_FFTBuffer1",convObs->GetName()),"shiftObs1",
					       *convObs,RooFit::RooConst(1),RooFit::RooConst(-1*self._shift1)) ;

    RooArgSet clonedBranches1 ;
    RooCustomizer cust(*clonePdf1,"fft") ;
    cust.replaceArg(*convObs,*shiftObs1) ;  

    pdf1Clone = (RooAbsPdf*) cust.build() ;

    pdf1Clone->addOwnedComponents(*shiftObs1) ;
    pdf1Clone->addOwnedComponents(*clonePdf1) ;

  } else {
    pdf1Clone = clonePdf1 ;
  }

  if (self._shift2!=0) {
    RooLinearVar* shiftObs2 = new RooLinearVar(Form("%s_shifted_FFTBuffer2",convObs->GetName()),"shiftObs2",
					       *convObs,RooFit::RooConst(1),RooFit::RooConst(-1*self._shift2)) ;

    RooArgSet clonedBranches2 ;
    RooCustomizer cust(*clonePdf2,"fft") ;
    cust.replaceArg(*convObs,*shiftObs2) ;  

    pdf1Clone->addOwnedComponents(*shiftObs2) ;
    pdf1Clone->addOwnedComponents(*clonePdf2) ;

    pdf2Clone = (RooAbsPdf*) cust.build() ;

  } else {
    pdf2Clone = clonePdf2 ;
  }


  // Attach cloned pdf to all original parameters of self
  RooArgSet* fftParams = self.getParameters(*convObs) ;

  // Remove all cache histogram from fftParams as these
  // observable need to remain attached to the histogram
  fftParams->remove(*hist()->get(),kTRUE,kTRUE) ;

  pdf1Clone->recursiveRedirectServers(*fftParams) ;
  pdf2Clone->recursiveRedirectServers(*fftParams) ;
  pdf1Clone->fixAddCoefRange(refName.c_str()) ;
  pdf2Clone->fixAddCoefRange(refName.c_str()) ;

  delete fftParams ;

  // Save copy of original histX binning and make alternate binning
  // for extended range scanning

  Int_t N = convObs->numBins() ;
  Int_t Nbuf = static_cast<Int_t>((N*self.bufferFraction())/2 + 0.5) ;
  Double_t obw = (convObs->getMax() - convObs->getMin())/N ;
  Int_t N2 = N+2*Nbuf ;

  scanBinning = new RooUniformBinning (convObs->getMin()-Nbuf*obw,convObs->getMax()+Nbuf*obw,N2) ;
  histBinning = convObs->getBinning().clone() ;

  // Deactivate dirty state propagation on datahist observables
  // and set all nodes on both pdfs to operMode AlwaysDirty
  hist()->setDirtyProp(kFALSE) ;  
  convObs->setOperMode(ADirty,kTRUE) ;
} 


//_____________________________________________________________________________
TString RooFFTConvPdf::histNameSuffix() const
{
  // Suffix for cache histogram (added in addition to suffix for cache name)
  return TString(Form("_BufFrac%3.1f_BufStrat%d",_bufFrac,_bufStrat)) ;
}



//_____________________________________________________________________________
RooArgList RooFFTConvPdf::FFTCacheElem::containedArgs(Action a) 
{
  // Returns all RooAbsArg objects contained in the cache element
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


//_____________________________________________________________________________
RooFFTConvPdf::FFTCacheElem::~FFTCacheElem() 
{ 
  delete fftr2c1 ; 
  delete fftr2c2 ; 
  delete fftc2r ; 

  delete pdf1Clone ;
  delete pdf2Clone ;

  delete histBinning ;
  delete scanBinning ;

}




//_____________________________________________________________________________
void RooFFTConvPdf::fillCacheObject(RooAbsCachedPdf::PdfCacheElem& cache) const 
{
  // Fill the contents of the cache the FFT convolution output
  RooDataHist& cacheHist = *cache.hist() ;
  
  ((FFTCacheElem&)cache).pdf1Clone->setOperMode(ADirty,kTRUE) ;
  ((FFTCacheElem&)cache).pdf2Clone->setOperMode(ADirty,kTRUE) ;

  // Determine if there other observables than the convolution observable in the cache
  RooArgSet otherObs ;
  RooArgSet(*cacheHist.get()).snapshot(otherObs) ;

  RooAbsArg* histArg = otherObs.find(_x.arg().GetName()) ;
  if (histArg) {
    otherObs.remove(*histArg,kTRUE,kTRUE) ;
    delete histArg ;
  } 

  //cout << "RooFFTConvPdf::fillCacheObject() otherObs = " << otherObs << endl ;

  // Handle trivial scenario -- no other observables
  if (otherObs.getSize()==0) {
    fillCacheSlice((FFTCacheElem&)cache,RooArgSet()) ;
    return ;
  }

  // Handle cases where there are other cache slices
  // Iterator over available slice positions and fill each

  // Determine number of bins for each slice position observable
  Int_t n = otherObs.getSize() ;
  Int_t* binCur = new Int_t[n+1] ;
  Int_t* binMax = new Int_t[n+1] ;
  Int_t curObs = 0 ;

  RooAbsLValue** obsLV = new RooAbsLValue*[n] ;
  TIterator* iter = otherObs.createIterator() ;
  RooAbsArg* arg ;
  Int_t i(0) ;
  while((arg=(RooAbsArg*)iter->Next())) {
    RooAbsLValue* lvarg = dynamic_cast<RooAbsLValue*>(arg) ;
    obsLV[i] = lvarg ;
    binCur[i] = 0 ;
    // coverity[FORWARD_NULL]
    binMax[i] = lvarg->numBins(binningName())-1 ;    
    i++ ;
  }
  delete iter ;

  Bool_t loop(kTRUE) ;
  while(loop) {
    // Set current slice position
    for (Int_t j=0 ; j<n ; j++) { obsLV[j]->setBin(binCur[j],binningName()) ; }

//     cout << "filling slice: bin of obsLV[0] = " << obsLV[0]->getBin() << endl ;

    // Fill current slice
    fillCacheSlice((FFTCacheElem&)cache,otherObs) ;

    // Determine which iterator to increment
    while(binCur[curObs]==binMax[curObs]) {
      
      // Reset current iterator and consider next iterator ;
      binCur[curObs]=0 ;      
      curObs++ ;

      // master termination condition
      if (curObs==n) {
	loop=kFALSE ;
	break ;
      }
    }

    // Increment current iterator
    binCur[curObs]++ ;
    curObs=0 ;      
    
  }

  delete[] obsLV ;
  delete[] binMax ;
  delete[] binCur ;
  
}


//_____________________________________________________________________________
void RooFFTConvPdf::fillCacheSlice(FFTCacheElem& aux, const RooArgSet& slicePos) const 
{
  // Fill a slice of cachePdf with the output of the FFT convolution calculation

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

  Int_t N,N2,binShift1,binShift2 ;
  
  RooRealVar* histX = (RooRealVar*) cacheHist.get()->find(_x.arg().GetName()) ;
  if (_bufStrat==Extend) histX->setBinning(*aux.scanBinning) ;
  Double_t* input1 = scanPdf((RooRealVar&)_x.arg(),*aux.pdf1Clone,cacheHist,slicePos,N,N2,binShift1,_shift1) ;
  Double_t* input2 = scanPdf((RooRealVar&)_x.arg(),*aux.pdf2Clone,cacheHist,slicePos,N,N2,binShift2,_shift2) ;
  if (_bufStrat==Extend) histX->setBinning(*aux.histBinning) ;




  // Retrieve previously defined FFT transformation plans
  if (!aux.fftr2c1) {
    aux.fftr2c1 = TVirtualFFT::FFT(1, &N2, "R2CK");
    aux.fftr2c2 = TVirtualFFT::FFT(1, &N2, "R2CK");
    aux.fftc2r  = TVirtualFFT::FFT(1, &N2, "C2RK");
  }
  
  // Real->Complex FFT Transform on p.d.f. 1 sampling
  aux.fftr2c1->SetPoints(input1);
  aux.fftr2c1->Transform();

  // Real->Complex FFT Transform on p.d.f 2 sampling
  aux.fftr2c2->SetPoints(input2);
  aux.fftr2c2->Transform();

  // Loop over first half +1 of complex output results, multiply 
  // and set as input of reverse transform
  for (Int_t i=0 ; i<N2/2+1 ; i++) {
    Double_t re1,re2,im1,im2 ;
    aux.fftr2c1->GetPointComplex(i,re1,im1) ;
    aux.fftr2c2->GetPointComplex(i,re2,im2) ;
    Double_t re = re1*re2 - im1*im2 ;
    Double_t im = re1*im2 + re2*im1 ;
    TComplex t(re,im) ;
    aux.fftc2r->SetPointComplex(i,t) ;
  }

  // Reverse Complex->Real FFT transform product
  aux.fftc2r->Transform() ;

  Int_t totalShift = binShift1 + (N2-N)/2 ;

  // Store FFT result in cache

  TIterator* iter = const_cast<RooDataHist&>(cacheHist).sliceIterator(const_cast<RooAbsReal&>(_x.arg()),slicePos) ;
  for (Int_t i =0 ; i<N ; i++) {

    // Cyclically shift array back so that bin containing zero is back in zeroBin
    Int_t j = i + totalShift ;
    while (j<0) j+= N2 ;
    while (j>=N2) j-= N2 ;

    iter->Next() ;
    cacheHist.set(aux.fftc2r->GetPointReal(j)) ;    
  }
  delete iter ;

  // cacheHist.dump2() ;

  // Delete input arrays
  delete[] input1 ;
  delete[] input2 ;

}


//_____________________________________________________________________________
Double_t*  RooFFTConvPdf::scanPdf(RooRealVar& obs, RooAbsPdf& pdf, const RooDataHist& hist, const RooArgSet& slicePos, 
				  Int_t& N, Int_t& N2, Int_t& zeroBin, Double_t shift) const
{
  // Scan the values of 'pdf' in observable 'obs' using the bin values stored in 'hist' at slice position 'slicePos'
  // N is filled with the number of bins defined in hist, N2 is filled with N plus the number of buffer bins
  // The return value is an array of doubles of length N2 with the sampled values. The caller takes ownership
  // of the array


  RooRealVar* histX = (RooRealVar*) hist.get()->find(obs.GetName()) ;

  // Calculate number of buffer bins on each size to avoid cyclical flow
  N = histX->numBins(binningName()) ;
  Int_t Nbuf = static_cast<Int_t>((N*bufferFraction())/2 + 0.5) ;
  N2 = N+2*Nbuf ;

  
  // Allocate array of sampling size plus optional buffer zones
  Double_t* array = new Double_t[N2] ;
  
  // Set position of non-convolution observable to that of the cache slice that were are processing now
  hist.get(slicePos) ;

  // Find bin ID that contains zero value
  zeroBin = 0 ;
  if (histX->getMax()>=0 && histX->getMin()<=0) {
    zeroBin = histX->getBinning().binNumber(0) ;
  } else if (histX->getMin()>0) {
    Double_t bw = (histX->getMax() - histX->getMin())/N2 ;
    zeroBin = Int_t(-histX->getMin()/bw) ;
  } else {
    Double_t bw = (histX->getMax() - histX->getMin())/N2 ;
    zeroBin = Int_t(-1*histX->getMax()/bw) ;
  }

  Int_t binShift = Int_t((N2* shift) / (histX->getMax()-histX->getMin())) ;

  zeroBin += binShift ;
  while(zeroBin>=N2) zeroBin-= N2 ;
  while(zeroBin<0) zeroBin+= N2 ;

  // First scan hist into temp array 
  Double_t *tmp = new Double_t[N2] ;
  Int_t k(0) ;
  switch(_bufStrat) {

  case Extend:
    // Sample entire extended range (N2 samples)
    for (k=0 ; k<N2 ; k++) {
      histX->setBin(k) ;
      tmp[k] = pdf.getVal(hist.get()) ;    
    }  
    break ;

  case Flat:    
    // Sample original range (N samples) and fill lower and upper buffer
    // bins with p.d.f. value at respective boundary
    {
      histX->setBin(0) ;
      Double_t val = pdf.getVal(hist.get()) ;  
      for (k=0 ; k<Nbuf ; k++) {
	tmp[k] = val ;
      }
      for (k=0 ; k<N ; k++) {
	histX->setBin(k) ;
	tmp[k+Nbuf] = pdf.getVal(hist.get()) ;    
      }  
      histX->setBin(N-1) ;
      val = pdf.getVal(hist.get()) ;  
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
      tmp[k+Nbuf] = pdf.getVal(hist.get()) ;    
    }  
    for (k=1 ; k<=Nbuf ; k++) {
      histX->setBin(k) ;
      tmp[Nbuf-k] = pdf.getVal(hist.get()) ;    
      histX->setBin(N-k) ;
      tmp[Nbuf+N+k-1] = pdf.getVal(hist.get()) ;    
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

  // Cleanup 
  delete[] tmp ;
  return array ;
}



//_____________________________________________________________________________
RooArgSet* RooFFTConvPdf::actualObservables(const RooArgSet& nset) const 
{
  // Return the observables to be cached given the normalization set nset
  //
  // If the cache observables is in nset then this is 
  //    - the convolution observable plus 
  //    - any member of nset that is either a RooCategory, 
  //    - or was previously specified through setCacheObservables().
  //
  // In case the cache observable is _not_ in nset, then it is
  //    - the convolution observable plus 
  //    - all member of nset are observables of this p.d.f.
  // 

  // Get complete list of observables 
  RooArgSet* obs1 = _pdf1.arg().getObservables(nset) ;
  RooArgSet* obs2 = _pdf2.arg().getObservables(nset) ;
  obs1->add(*obs2,kTRUE) ;

  // Check if convolution observable is in nset
  if (nset.contains(_x.arg())) {

    // Now strip out all non-category observables
    TIterator* iter = obs1->createIterator() ;
    RooAbsArg* arg ;
    RooArgSet killList ;
    while((arg=(RooAbsArg*)iter->Next())) {
      if (arg->IsA()->InheritsFrom(RooAbsReal::Class()) && !_cacheObs.find(arg->GetName())) {
	killList.add(*arg) ;
      }
    }
    delete iter ;
    obs1->remove(killList) ;
    
    // And add back the convolution observables
    obs1->add(_x.arg(),kTRUE) ; 

    obs1->add(_cacheObs) ;

    delete obs2 ;
    
  } else {

    // If cacheObs was filled, cache only observables in there
    if (_cacheObs.getSize()>0) {
      TIterator* iter = obs1->createIterator() ;
      RooAbsArg* arg ;
      RooArgSet killList ;
      while((arg=(RooAbsArg*)iter->Next())) {
	if (arg->IsA()->InheritsFrom(RooAbsReal::Class()) && !_cacheObs.find(arg->GetName())) {
	  killList.add(*arg) ;
	}
      }
      delete iter ;
      obs1->remove(killList) ;
    }


    // Make sure convolution observable is always in there
    obs1->add(_x.arg(),kTRUE) ; 
    delete obs2 ;
    
  }

  return obs1 ;  
}



//_____________________________________________________________________________
RooArgSet* RooFFTConvPdf::actualParameters(const RooArgSet& nset) const 
{  
  // Return the parameters on which the cache depends given normalization
  // set nset. For this p.d.f these are the parameters of the input p.d.f.
  // but never the convolution variable, it case it is not part of nset

  RooArgSet* vars = getVariables() ;
  RooArgSet* obs = actualObservables(nset) ;
  vars->remove(*obs) ;
  delete obs ;

  return vars ;
}



//_____________________________________________________________________________
RooAbsArg& RooFFTConvPdf::pdfObservable(RooAbsArg& histObservable) const 
{
  // Return p.d.f. observable (which can be a function) to substitute given
  // p.d.f. observable. Substitute x by xprime if xprime is set

  if (_xprime.absArg() && string(histObservable.GetName())==_x.absArg()->GetName()) {
    return (*_xprime.absArg()) ;
  }
  return histObservable ;
}



//_____________________________________________________________________________
RooAbsGenContext* RooFFTConvPdf::genContext(const RooArgSet &vars, const RooDataSet *prototype, 
					    const RooArgSet* auxProto, Bool_t verbose) const 
{
  // Create appropriate generator context for this convolution. If both input p.d.f.s support
  // internal generation, if it is safe to use them and if no observables other than the convolution
  // observable are requested for generation, use the specialized convolution generator context
  // which implements a smearing strategy in the convolution observable. If not return the
  // regular accept/reject generator context

  RooArgSet vars2(vars) ;
  vars2.remove(_x.arg(),kTRUE,kTRUE) ;
  Int_t numAddDep = vars2.getSize() ;

  RooArgSet dummy ;
  Bool_t pdfCanDir = (((RooAbsPdf&)_pdf1.arg()).getGenerator(_x.arg(),dummy) != 0 && \
		      ((RooAbsPdf&)_pdf1.arg()).isDirectGenSafe(_x.arg())) ;
  Bool_t resCanDir = (((RooAbsPdf&)_pdf2.arg()).getGenerator(_x.arg(),dummy) !=0  && 
		      ((RooAbsPdf&)_pdf2.arg()).isDirectGenSafe(_x.arg())) ;

  if (pdfCanDir) {
    cxcoutI(Generation) << "RooFFTConvPdf::genContext() input p.d.f " << _pdf1.arg().GetName() 
			<< " has internal generator that is safe to use in current context" << endl ;
  }
  if (resCanDir) {
    cxcoutI(Generation) << "RooFFTConvPdf::genContext() input p.d.f. " << _pdf2.arg().GetName() 
			<< " has internal generator that is safe to use in current context" << endl ;
  }
  if (numAddDep>0) {
    cxcoutI(Generation) << "RooFFTConvPdf::genContext() generation requested for observables other than the convolution observable " << _x.arg().GetName() << endl ;
  }  

  
  if (numAddDep>0 || !pdfCanDir || !resCanDir) {
    // Any resolution model with more dependents than the convolution variable
    // or pdf or resmodel do not support direct generation
    cxcoutI(Generation) << "RooFFTConvPdf::genContext() selecting accept/reject generator context because one or both of the input "
			<< "p.d.f.s cannot use internal generator and/or " 
			<< "observables other than the convolution variable are requested for generation" << endl ;
    return new RooGenContext(*this,vars,prototype,auxProto,verbose) ;
  } 
  
  // Any other resolution model: use specialized generator context
  cxcoutI(Generation) << "RooFFTConvPdf::genContext() selecting specialized convolution generator context as both input "
		      << "p.d.fs are safe for internal generator and only "
		      << "the convolution observables is requested for generation" << endl ;
  return new RooConvGenContext(*this,vars,prototype,auxProto,verbose) ;
}



//_____________________________________________________________________________
void RooFFTConvPdf::setBufferFraction(Double_t frac) 
{
  // Change the size of the buffer on either side of the observable range to frac times the
  // size of the range of the convolution observable

  if (frac<0) {
    coutE(InputArguments) << "RooFFTConvPdf::setBufferFraction(" << GetName() << ") fraction should be greater than or equal to zero" << endl ;
    return ;
  }
  _bufFrac = frac ;

  // Sterilize the cache as certain partial results depend on buffer fraction
  _cacheMgr.sterilize() ;
}


//_____________________________________________________________________________
void RooFFTConvPdf::setBufferStrategy(BufStrat bs) 
{
  // Change strategy to fill the overflow buffer on either side of the convolution observable range.
  //
  // 'Extend' means is that the input p.d.f convolution observable range is widened to include the buffer range
  // 'Flat' means that the buffer is filled with the p.d.f. value at the boundary of the observable range
  // 'Mirror' means that the buffer is filled with a ,irror image of the p.d.f. around the convolution observable boundary 
  //
  // The default strategy is extend. If one of the input p.d.f.s is a RooAddPdf, it is configured so that the interpretation
  // range of the fraction coefficients is kept at the nominal convolutions observable range (instead of interpreting coefficients
  // in the widened range including the buffer)

  _bufStrat = bs ;
}



//_____________________________________________________________________________
void RooFFTConvPdf::printMetaArgs(ostream& os) const 
{
  // Customized printing of arguments of a RooNumConvPdf to more intuitively reflect the contents of the
  // product operator construction

  os << _pdf1.arg().GetName() << "(" << _x.arg().GetName() << ") (*) " << _pdf2.arg().GetName() << "(" << _x.arg().GetName() << ") " ;
}



//_____________________________________________________________________________
void RooFFTConvPdf::calcParams() 
{
  // (Re)calculate effective parameters of this p.d.f.

  RooArgSet* params1 = _pdf1.arg().getParameters(_x.arg()) ;
  RooArgSet* params2 = _pdf2.arg().getParameters(_x.arg()) ;
  _params.removeAll() ;
  _params.add(*params1) ;
  _params.add(*params2,kTRUE) ;
  delete params1 ;
  delete params2 ;
}



//_____________________________________________________________________________
Bool_t RooFFTConvPdf::redirectServersHook(const RooAbsCollection& /*newServerList*/, Bool_t /*mustReplaceAll*/, Bool_t /*nameChange*/, Bool_t /*isRecursive*/) 
{
  //calcParams() ;
  return kFALSE ;
}
