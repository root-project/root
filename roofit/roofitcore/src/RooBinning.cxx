/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
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
// BEGIN_HTML 
// Class RooBinning is an implements RooAbsBinning in terms
// of an array of boundary values, posing no constraints on the choice
// of binning, thus allowing variable bin sizes. Various methods allow
// the user to add single bin boundaries, mirrored pairs, or sets of
// uniformly spaced boundaries.  
// END_HTML
//

#include "RooFit.h"

#include "Riostream.h"
#include "Riostream.h"
#include "RooBinning.h"
#include "RooDouble.h"
#include "RooAbsPdf.h"
#include "RooRealVar.h"
#include "RooNumber.h"
#include "RooMsgService.h"
#include "TList.h"

using namespace std;

ClassImp(RooBinning)
;


//_____________________________________________________________________________
RooBinning::RooBinning(Double_t xlo, Double_t xhi, const char* name) : 
  RooAbsBinning(name), 
  _xlo(0),
  _xhi(0),
  _ownBoundLo(kTRUE), 
  _ownBoundHi(kTRUE), 
  _array(0)
{
  // Constructor for an initially empty binning defining the range [xlo,xhi]

  setRange(xlo,xhi) ;
}



//_____________________________________________________________________________
RooBinning::RooBinning(Int_t nbins, Double_t xlo, Double_t xhi, const char* name) : 
  RooAbsBinning(name), 
  _xlo(0),
  _xhi(0),
  _ownBoundLo(kTRUE), 
  _ownBoundHi(kTRUE), 
  _array(0)
{
  // Constructor for a uniform binning in 'nbins' bins in the range [xlo,xhi]

  // Uniform bin size constructor
  setRange(xlo,xhi) ;
  addUniform(nbins,xlo,xhi) ;
}




//_____________________________________________________________________________
RooBinning::RooBinning(Int_t nbins, const Double_t* boundaries, const char* name) : 
  RooAbsBinning(name),
  _xlo(0),
  _xhi(0),
  _ownBoundLo(kTRUE), 
  _ownBoundHi(kTRUE), 
  _array(0)
{
  // Constructor for a binning in the range[xlo,xhi] with 'nbins' bin boundaries listed
  // array 'boundaries'

  // Variable bin size constructor
  setRange(boundaries[0],boundaries[nbins]) ;
  while(nbins--) addBoundary(boundaries[nbins]) ;
}



//_____________________________________________________________________________
RooBinning::RooBinning(const RooBinning& other, const char* name) : 
  RooAbsBinning(name),
  _boundaries(other._boundaries),
  _array(0)
{ 
  // Copy constructor
  _xlo = other._xlo ;
  _xhi = other._xhi ;
  _ownBoundLo = other._ownBoundLo ;
  _ownBoundHi = other._ownBoundHi ;
  _nbins = other._nbins ;
}



//_____________________________________________________________________________
RooBinning::~RooBinning() 
{
  // Destructor

  if (_array) delete[] _array ;

}


//_____________________________________________________________________________
Bool_t RooBinning::addBoundary(Double_t boundary) 
{  
  // Add bin boundary at given value

  if (_boundaries.find(boundary)!=_boundaries.end()) {
    // If boundary previously existed as range delimiter, 
    //                    convert to regular boundary now
    if (boundary==_xlo) _ownBoundLo = kFALSE ;
    if (boundary==_xhi) _ownBoundHi = kFALSE ;
    return kFALSE ;    
  }

  // Add a new boundary
  _boundaries.insert(boundary) ;
  updateBinCount() ;
  return kTRUE ;
}



//_____________________________________________________________________________
void RooBinning::addBoundaryPair(Double_t boundary, Double_t mirrorPoint) 
{
  // Add pair of boundaries: one at 'boundary' and one at 2*mirrorPoint-boundary

  addBoundary(boundary) ;
  addBoundary(2*mirrorPoint-boundary) ;
}



//_____________________________________________________________________________
Bool_t RooBinning::removeBoundary(Double_t boundary)
{
  // Remove boundary at given value

  if (_boundaries.find(boundary)!=_boundaries.end()) {
    _boundaries.erase(boundary) ;
    return kFALSE ;
  }

  // Return error status - no boundary found
  return kTRUE ;
}



//_____________________________________________________________________________
Bool_t RooBinning::hasBoundary(Double_t boundary)
{
  // Check if boundary exists at given value

  return (_boundaries.find(boundary)!=_boundaries.end()) ;
}



//_____________________________________________________________________________
void RooBinning::addUniform(Int_t nbins, Double_t xlo, Double_t xhi)
{
  // Add array of nbins uniformly sized bins in range [xlo,xhi]

  Int_t i ;
  Double_t binw = (xhi-xlo)/nbins ;
  for (i=0 ; i<=nbins ; i++) 
    addBoundary(xlo+i*binw) ;  
}



//_____________________________________________________________________________
Int_t RooBinning::binNumber(Double_t x) const
{
  // Return sequential bin number that contains value x where bin
  // zero is the first bin with an upper boundary above the lower bound
  // of the range

  Int_t n(0) ;
  for (set<Double_t>::const_iterator iter = _boundaries.begin() ; iter!=_boundaries.end() ; ++iter) {
    if (x<*iter) {
      return n ;
    }

    // Only increment counter in valid range
    if (*iter> _xlo && n<_nbins-1) n++ ;    
  }  
  return n;
}



//_____________________________________________________________________________
Int_t RooBinning::rawBinNumber(Double_t x) const 
{
  // Return sequential bin number that contains value x where bin
  // zero is the first bin that is defined, regardless if that bin
  // is outside the current defined range
  
 
  // Determine 'raw' bin number (i.e counting all defined boundaries) for given value
  Int_t n(0) ;

  for (set<Double_t>::const_iterator iter = _boundaries.begin() ; iter!=_boundaries.end() ; ++iter) {    
    if (x<*iter) return n>0?n-1:0 ;
    n++ ;
  }
  return n-1;
}



//_____________________________________________________________________________
Double_t RooBinning::nearestBoundary(Double_t x) const 
{
  // Return the value of the nearest boundary to x

  Int_t bn = binNumber(x) ;
  if (fabs(binLow(bn)-x)<fabs(binHigh(bn)-x)) {
    return binLow(bn) ;
  } else {
    return binHigh(bn) ;
  }
}



//_____________________________________________________________________________
Double_t* RooBinning::array() const
{
  // Return array of boundary values

  if (_array) delete[] _array ;
  _array = new Double_t[numBoundaries()] ;

  Int_t i(0) ;
  for (set<Double_t>::const_iterator iter = _boundaries.begin() ; iter!=_boundaries.end() ; ++iter) {
    if (*iter>=_xlo && *iter <=_xhi) {
      _array[i++] = *iter ;
    }
  }
  return _array ;
}



//_____________________________________________________________________________
void RooBinning::setRange(Double_t xlo, Double_t xhi) 
{
  // Change the defined range associated with this binning.
  // Bins that lie outside the new range [xlo,xhi] will not be
  // removed, but will be 'inactive', i.e. the new 0 bin will
  // be the first bin with an upper boundarie > xlo

  if (xlo>xhi) {
    coutE(InputArguments) << "RooUniformBinning::setRange: ERROR low bound > high bound" << endl ;
    return ;
  }
  
  // Remove previous boundaries 
  
  for (set<Double_t>::iterator iter = _boundaries.begin() ; iter!=_boundaries.end() ;) {    
    if ((*iter == _xlo && _ownBoundLo) || (*iter == _xhi && _ownBoundHi)) {
      _boundaries.erase(iter++) ;
    } else {
      ++iter ;
    }
  }

  // Insert boundaries at range delimiter, if necessary 
  _ownBoundLo = kFALSE ;
  _ownBoundHi = kFALSE ;
  if (!hasBoundary(xlo)) {
    addBoundary(xlo) ;
    _ownBoundLo = kTRUE ;
  }
  if (!hasBoundary(xhi)) {
    addBoundary(xhi) ;
    _ownBoundHi = kTRUE ;
  }

  _xlo = xlo ;
  _xhi = xhi ;
  
  // Count number of bins with new range 
  updateBinCount() ;
}




//_____________________________________________________________________________
void RooBinning::updateBinCount()
{
  // Update the internal bin counter

  Int_t i(-1) ;
  for (set<Double_t>::const_iterator iter = _boundaries.begin() ; iter!=_boundaries.end() ; ++iter) {    
    if (*iter>=_xlo && *iter <=_xhi) {
      i++ ;
    }
  }
  _nbins = i ;
}



//_____________________________________________________________________________
Bool_t RooBinning::binEdges(Int_t bin, Double_t& xlo, Double_t& xhi) const 
{
  // Return upper and lower bound of bin 'bin'. If the return value
  // is true an error occurred

  if (bin<0 || bin>= _nbins) {
    coutE(InputArguments) << "RooBinning::binEdges ERROR: bin number must be in range (0," << _nbins << ")" << endl ; 
    return kTRUE ;
  }
  
  // Determine sequential bin number for given value
  Int_t n(0) ;
  for (set<Double_t>::const_iterator iter = _boundaries.begin() ; iter!=_boundaries.end() ; ++iter) {    

    if (n==bin && *iter>=_xlo) {
      xlo = *iter ;
      iter++ ;
      xhi = *iter ;
      return kFALSE ;
    }

    // Only increment counter in valid range
    if (*iter>= _xlo && n<_nbins-1) n++ ;
  }

  return kTRUE ;
}



//_____________________________________________________________________________
Double_t RooBinning::binCenter(Int_t bin) const 
{
  // Return the position of the center of bin 'bin'

  Double_t xlo,xhi ;
  if (binEdges(bin,xlo,xhi)) return 0 ;
  return (xlo+xhi)/2 ;
}



//_____________________________________________________________________________
Double_t RooBinning::binWidth(Int_t bin) const 
{
  // Return the width of the requested bin

  Double_t xlo,xhi ;
  if (binEdges(bin,xlo,xhi)) return 0 ;
  return (xhi-xlo);
}



//_____________________________________________________________________________
Double_t RooBinning::binLow(Int_t bin) const 
{
  // Return the lower bound of the requested bin

  Double_t xlo,xhi ;
  if (binEdges(bin,xlo,xhi)) return 0 ;
  return xlo ;
}



//_____________________________________________________________________________
Double_t RooBinning::binHigh(Int_t bin) const  
{
  // Return the upper bound of the requested bin

  Double_t xlo,xhi ;
  if (binEdges(bin,xlo,xhi)) return  0 ;
  return xhi ;
}



//______________________________________________________________________________
void RooBinning::Streamer(TBuffer &R__b)
{
   // Custom streamer that provides backward compatibility to read v1 data

   if (R__b.IsReading()) {

     UInt_t R__s, R__c;
     Version_t R__v = R__b.ReadVersion(&R__s, &R__c); if (R__v) { }
     if (R__v>1) {
       R__b.ReadClassBuffer(RooBinning::Class(),this,R__v,R__s,R__c);
     } else {
       RooAbsBinning::Streamer(R__b);
       R__b >> _xlo;
       R__b >> _xhi;
       R__b >> _ownBoundLo;
       R__b >> _ownBoundHi;
       R__b >> _nbins;
       
       // Convert TList to set<double>
       TList boundaries ;
       boundaries.Streamer(R__b);
       TIterator* iter = boundaries.MakeIterator() ;
       RooDouble* elem ;
       while((elem=(RooDouble*)iter->Next())) {
	 _boundaries.insert(*elem) ;
       }
       delete iter ;
       
       R__b.CheckByteCount(R__s, R__c, RooBinning::IsA());
     }
   } else {
     R__b.WriteClassBuffer(RooBinning::Class(),this);
   }
}


