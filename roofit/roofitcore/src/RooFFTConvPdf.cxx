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

 // -- CLASS DESCRIPTION [PDF] -- 
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

using namespace std ;

ClassImp(RooFFTConvPdf) 



RooFFTConvPdf::RooFFTConvPdf(const char *name, const char *title, RooRealVar& convVar, RooAbsPdf& pdf1, RooAbsPdf& pdf2, Int_t ipOrder) :
  RooAbsCachedPdf(name,title,ipOrder),
  _x("x","Convolution Variable",this,convVar),
  _pdf1("pdf1","pdf1",this,pdf1),
  _pdf2("pdf2","pdf2",this,pdf2),
  _bufFrac(0.1)
 { 
   // Constructor for convolution of pdf1 x pdf2 in observable convVar. The cache is interpolated to order ipOrder
 } 


RooFFTConvPdf::RooFFTConvPdf(const RooFFTConvPdf& other, const char* name) :  
  RooAbsCachedPdf(other,name),
  _x("x",this,other._x),
  _pdf1("pdf1",this,other._pdf1),
  _pdf2("pdf2",this,other._pdf2),
  _bufFrac(other._bufFrac)
 { 
   // Copy constructor
 } 


RooFFTConvPdf::~RooFFTConvPdf() 
{
  // Destructor 

  // Delete FFT engines 
  map<const RooHistPdf*,CacheAuxInfo*>::iterator iter =  _cacheAuxInfo.begin() ;
  for (; iter!=_cacheAuxInfo.end() ; ++iter) {
    delete iter->second ;
  }
}


const char* RooFFTConvPdf::inputBaseName() const 
{
  // Return base name component for cache components that are auto-generated by RooAbsCachedPdf base class
  static TString name ;
  name = _pdf1.GetName() ;
  name.Append("_CONV_") ;
  name.Append(_pdf2.GetName()) ;
  return name.Data() ;
}


void RooFFTConvPdf::fillCacheObject(RooAbsCachedPdf::CacheElem& cache) const 
{
  RooDataHist& cacheHist = *cache._hist ;
  RooHistPdf& cachePdf = *cache._pdf ;
  
  // Determine if there other observables than the convolution observable in the cache
  RooArgSet otherObs ;
  RooArgSet(*cacheHist.get()).snapshot(otherObs) ;
  otherObs.remove(_x.arg(),kTRUE,kTRUE) ;

  // Handle trivial scenario -- no other observables
  if (otherObs.getSize()==0) {
    fillCacheSlice(cachePdf,RooArgSet()) ;
    return ;
  }

  // Handle cases where there are other cache slices
  // Iterator over available slice positions and fill each

  // Determine number of bins for each slice position observable
  Int_t n = otherObs.getSize() ;
  Int_t* binCur = new Int_t[n] ;
  Int_t* binMax = new Int_t[n] ;
  Int_t curObs = 0 ;

  RooAbsLValue** obsLV = new RooAbsLValue*[n] ;
  TIterator* iter = otherObs.createIterator() ;
  RooAbsArg* arg ;
  Int_t i(0) ;
  while((arg=(RooAbsArg*)iter->Next())) {
    RooAbsLValue* lvarg = dynamic_cast<RooAbsLValue*>(arg) ;
    obsLV[i] = lvarg ;
    binCur[i] = 0 ;
    binMax[i] = lvarg->numBins()-1 ;    
    i++ ;
  }
  delete iter ;

  Bool_t loop(kTRUE) ;
  while(loop) {
    // Set current slice position
    for (Int_t j=0 ; j<n ; j++) { obsLV[j]->setBin(binCur[j]) ; }

    // Fill current slice
    fillCacheSlice(cachePdf,otherObs) ;

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
  
  
}

void RooFFTConvPdf::fillCacheSlice(RooHistPdf& cachePdf, const RooArgSet& slicePos) const 
{
  // (Re)Fill the cache represented by the given RooHistPdf.

  // Extract histogram that is the basis of the RooHistPdf
  RooDataHist& cacheHist = cachePdf.dataHist() ;

  RooAbsPdf& pdf1 = (RooAbsPdf&)_pdf1.arg() ;
  RooAbsPdf& pdf2 = (RooAbsPdf&)_pdf2.arg() ;
  
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

  Int_t N,N2 ;
  Double_t* input1 = scanPdf((RooRealVar&)_x.arg(),pdf1,cacheHist,slicePos,N,N2) ;
  Double_t* input2 = scanPdf((RooRealVar&)_x.arg(),pdf2,cacheHist,slicePos,N,N2) ;

  // Retrieve previously defined FFT transformation plans
  CacheAuxInfo* aux = _cacheAuxInfo[&cachePdf] ;
  if (!aux) {
    // If they do not exist make them now and keep in cache
    aux = new CacheAuxInfo ;
    _cacheAuxInfo[&cachePdf] = aux ;
    aux->fftr2c1 = TVirtualFFT::FFT(1, &N2, "R2CK");
    aux->fftr2c2 = TVirtualFFT::FFT(1, &N2, "R2CK");
    aux->fftc2r  = TVirtualFFT::FFT(1, &N2, "C2RK");
  }
  
  // Real->Complex FFT Transform on p.d.f. 1 sampling
  aux->fftr2c1->SetPoints(input1);
  aux->fftr2c1->Transform();

  // Real->Complex FFT Transform on p.d.f 2 sampling
  aux->fftr2c2->SetPoints(input2);
  aux->fftr2c2->Transform();

  // Loop over first half +1 of complex output results, multiply 
  // and set as input of reverse transform
  for (Int_t i=0 ; i<N2/2+1 ; i++) {
    Double_t re1,re2,im1,im2 ;
    aux->fftr2c1->GetPointComplex(i,re1,im1) ;
    aux->fftr2c2->GetPointComplex(i,re2,im2) ;
    Double_t re = re1*re2 - im1*im2 ;
    Double_t im = re1*im2 + re2*im1 ;
    TComplex t(re,im) ;
    aux->fftc2r->SetPointComplex(i,t) ;
  }

  // Reverse Complex->Real FFT transform product
  aux->fftc2r->Transform() ;

  // Find bin ID that contains zero value
  Int_t zeroBin = 0 ;
  if (_x.min()<0 && _x.max()>0) {
    zeroBin = ((RooAbsRealLValue&)_x.arg()).getBinning().binNumber(0.)+1 ;
  }

  // Store FFT result in cache
  TIterator* iter = const_cast<RooDataHist&>(cacheHist).sliceIterator(const_cast<RooAbsReal&>(_x.arg()),slicePos) ;
  for (Int_t i =0 ; i<N ; i++) {

    // Cyclically shift array back so that bin containing zero is back in zeroBin
    Int_t j ;
    if (i<zeroBin) {
      j = i + (N2-zeroBin) ;
    } else {
      j = i - zeroBin ;
    }

    iter->Next() ;
    cacheHist.set(aux->fftc2r->GetPointReal(j)) ;    
  }
  
  // cacheHist.dump2() ;

  // Delete input arrays
  delete[] input1 ;
  delete[] input2 ;

}

Double_t*  RooFFTConvPdf::scanPdf(RooRealVar& obs, RooAbsPdf& pdf, const RooDataHist& hist, const RooArgSet& slicePos, Int_t& N, Int_t& N2) const
{
  // Clone input pdf and attach to dataset
  RooArgSet* cloneSet = (RooArgSet*) RooArgSet(pdf).snapshot(kTRUE) ;
  RooAbsPdf* theClone = (RooAbsPdf*) cloneSet->find(pdf.GetName()) ;
  theClone->attachDataSet(hist) ;

  RooRealVar* histX = (RooRealVar*) hist.get()->find(obs.GetName()) ;
  N = histX->numBins() ;
  
  // Calculate number of buffer bins on each size to avoid cyclical flow
  Int_t Nbuf = static_cast<Int_t>((N*bufferFraction())/2 + 0.5) ;
  N2 = N+2*Nbuf ;

  // Allocate array of sampling size plus optional buffer zones
  Double_t* array = new Double_t[N2] ;
  
  // Find bin ID that contains zero value
  Int_t zeroBin = -1 ;
  if (histX->getMin()<0 && histX->getMax()>0) {
    zeroBin = histX->getBinning().binNumber(0.)+1 ;
  }

  // First scan hist into temp array 
  Double_t *tmp = new Double_t[N] ;
  TIterator* iter = const_cast<RooDataHist&>(hist).sliceIterator(obs,slicePos) ;
  Int_t k=0 ;
  RooAbsArg* arg ;
  while((arg=(RooAbsArg*)iter->Next())) {
    tmp[k++] = theClone->getVal(hist.get()) ;
  }
  delete iter ;

  // Get underflow and overflow values
  Double_t valFirst = tmp[0] ;
  Double_t valLast = tmp[N-1] ;
  
  // Scan function and store values in array
  // i = bin index of RooRealVar [0,N-1]
  // j = array index of sample array [0,N+2Nbuf-1]

  for (Int_t i=0 ; i<N2 ; i++) {

    // Account fo
    Int_t j = i-Nbuf ;    

    // Determine value of pdf for this bin j     
    Double_t valJ ;    
    if (j<0) {
      // Underflow buffer, value of first bin
      valJ = valFirst ;
    } else if (j>=N) {
      // Overflow buffer, value of last bin
      valJ = valLast ;
    } else { 
      // In range, value of pdf
      valJ = tmp[j] ;
    }       
    
    // Cyclically shift writing location by zero bin position    
    if (zeroBin>=0) {
      if (j>= zeroBin) {
	// Write positive value bins
	array[i-(zeroBin+Nbuf)] = valJ ;
      } else {
	// Write negative value bins
	array[i-(zeroBin+Nbuf)+N2] = valJ ;
      }
    } else {
      array[i] = valJ ;
    }

  }


  // Cleanup 
  delete cloneSet ;
  delete[] tmp ;
  return array ;
}


RooArgSet* RooFFTConvPdf::actualObservables(const RooArgSet& nset) const 
{
  RooArgSet* obs1 = _pdf1.arg().getObservables(nset) ;
  RooArgSet* obs2 = _pdf2.arg().getObservables(nset) ;
  obs1->add(*obs2,kTRUE) ;
  obs1->add(_x.arg(),kTRUE) ; // always add convolution observable
  delete obs2 ;
  return obs1 ;  
}


RooArgSet* RooFFTConvPdf::actualParameters(const RooArgSet& nset) const 
{  
  RooArgSet* par1 = _pdf1.arg().getParameters(nset) ;
  RooArgSet* par2 = _pdf2.arg().getParameters(nset) ;
  par1->add(*par2,kTRUE) ;
  par1->remove(_x.arg(),kTRUE,kTRUE) ;
  delete par2 ;
  return par1 ;
}


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
    cxcoutI(Generation) << "RooFFTConvPdf::genContext() input p.d.f " << _pdf1.arg().GetName() << " has internal generator that is safe to use in current context" << endl ;
  }
  if (resCanDir) {
    cxcoutI(Generation) << "RooFFTConvPdf::genContext() input p.d.f. " << _pdf2.arg().GetName() << " has internal generator that is safe to use in current context" << endl ;
  }
  if (numAddDep>0) {
    cxcoutI(Generation) << "RooFFTConvPdf::genContext() generation requested for observables other than the convolution observable " << _x.arg().GetName() << endl ;
  }  

  
  if (numAddDep>0 || !pdfCanDir || !resCanDir) {
    // Any resolution model with more dependents than the convolution variable
    // or pdf or resmodel do not support direct generation
    cxcoutI(Generation) << "RooFFTConvPdf::genContext() selecting accept/reject generator context because one or both of the input p.d.f.s cannot use internal generator and/or " 
			  << "observables other than the convolution variable are requested for generation" << endl ;
    return new RooGenContext(*this,vars,prototype,auxProto,verbose) ;
  } 
  
  // Any other resolution model: use specialized generator context
  cxcoutI(Generation) << "RooFFTConvPdf::genContext() selecting specialized convolution generator context as both input p.d.fs are safe for internal generator and only "
			<< "the convolution observables is requested for generation" << endl ;
  return new RooConvGenContext(*this,vars,prototype,auxProto,verbose) ;
}


void RooFFTConvPdf::setBufferFraction(Double_t frac) 
{
  if (frac<0) {
    coutE(InputArguments) << "RooFFTConvPdf::setBufferFraction(" << GetName() << ") fraction should be greater than or equal to zero" << endl ;
    return ;
  }
  _bufFrac = frac ;
}

