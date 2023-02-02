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

/**
\file RooGrid.cxx
\class RooGrid
\ingroup Roofitcore

RooGrid is a utility class for RooMCIntegrator which
implements an adaptive multi-dimensional Monte Carlo numerical
integration, following the VEGAS algorithm.
**/

#include "RooGrid.h"
#include "RooAbsFunc.h"
#include "RooNumber.h"
#include "RooRandom.h"
#include "TMath.h"
#include "RooMsgService.h"

#include <math.h>
#include "Riostream.h"
#include <iomanip>



////////////////////////////////////////////////////////////////////////////////
/// Constructor with given function binding

RooGrid::RooGrid(const RooAbsFunc &function)
  : _valid(true)
{
  // check that the input function is valid
  if(!(_valid= function.isValid())) {
    oocoutE(nullptr,InputArguments) << "RooGrid: cannot initialize using an invalid function" << std::endl;
    return;
  }

  // allocate workspace memory
  _dim= function.getDimension();
  _xl.resize(_dim);
  _xu.resize(_dim);
  _delx.resize(_dim);
  _d.resize(_dim*maxBins);
  _xi.resize(_dim*(maxBins+1));
  _xin.resize(maxBins+1);
  _weight.resize(maxBins);

  // initialize the grid
  _valid= initialize(function);
}


////////////////////////////////////////////////////////////////////////////////
/// Calculate and store the grid dimensions and volume using the
/// specified function, and initialize the grid using a single bin.
/// Return true, or else false if the range is not valid.

bool RooGrid::initialize(const RooAbsFunc &function)
{
  _vol= 1;
  _bins= 1;
  for(UInt_t index= 0; index < _dim; index++) {
    _xl[index]= function.getMinLimit(index);
    if(RooNumber::isInfinite(_xl[index])) {
      oocoutE(nullptr,Integration) << "RooGrid: lower limit of dimension " << index << " is infinite" << std::endl;
      return false;
    }
    _xu[index]= function.getMaxLimit(index);
    if(RooNumber::isInfinite(_xl[index])) {
      oocoutE(nullptr,Integration) << "RooGrid: upper limit of dimension " << index << " is infinite" << std::endl;
      return false;
    }
    double dx= _xu[index] - _xl[index];
    if(dx <= 0) {
      oocoutE(nullptr,Integration) << "RooGrid: bad range for dimension " << index << ": [" << _xl[index]
                   << "," << _xu[index] << "]" << std::endl;
      return false;
    }
    _delx[index]= dx;
    _vol*= dx;
    coord(0,index) = 0;
    coord(1,index) = 1;
  }
  return true;
}


////////////////////////////////////////////////////////////////////////////////
/// Adjust the subdivision of each axis to give the specified
/// number of bins, using an algorithm that preserves relative
/// bin density. The new binning can be finer or coarser than
/// the original binning.

void RooGrid::resize(UInt_t bins)
{
  // is there anything to do?
  if(bins == _bins) return;

  // weight is ratio of bin sizes
  double pts_per_bin = (double) _bins / (double) bins;

  // loop over grid dimensions
  for (UInt_t j = 0; j < _dim; j++) {
    double xold,xnew(0),dw(0);
    Int_t i = 1;
    // loop over bins in this dimension and load _xin[] with new bin edges

    UInt_t k;
    for(k = 1; k <= _bins; k++) {
      dw += 1.0;
      xold = xnew;
      xnew = coord(k,j);
      while(dw > pts_per_bin) {
   dw -= pts_per_bin;
   newCoord(i++)= xnew - (xnew - xold) * dw;
      }
    }
    // copy the new edges into _xi[j]
    for(k = 1 ; k < bins; k++) {
      coord(k, j) = newCoord(k);
    }
    coord(bins, j) = 1;
  }
  _bins = bins;
}


////////////////////////////////////////////////////////////////////////////////
/// Reset the values associated with each grid cell.

void RooGrid::resetValues()
{
  for(UInt_t i = 0; i < _bins; i++) {
    for (UInt_t j = 0; j < _dim; j++) {
      value(i,j)= 0.0;
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Generate a random vector in the specified box and store its
/// coordinates in the x[] array provided, the corresponding bin
/// indices in the bin[] array, and the volume of this bin in vol.
/// The box is specified by the array box[] of box integer indices
/// that each range from 0 to getNBoxes()-1.

void RooGrid::generatePoint(const UInt_t box[], double x[], UInt_t bin[], double &vol,
             bool useQuasiRandom) const
{
  vol= 1;

  // generate a vector of quasi-random numbers to use
  if(useQuasiRandom) {
    RooRandom::quasi(_dim,x);
  }
  else {
    RooRandom::uniform(_dim,x);
  }

  // loop over coordinate axes
  for(UInt_t j= 0; j < _dim; ++j) {

    // generate a random point uniformly distributed (in box space)
    // within the box[j]-th box of coordinate axis j.
    double z= ((box[j] + x[j])/_boxes)*_bins;

    // store the bin in which this point lies along the j-th
    // coordinate axis and calculate its width and position y
    // in normalized bin coordinates.
    Int_t k= static_cast<Int_t>(z);
    bin[j] = k;
    double y, bin_width;
    if(k == 0) {
      bin_width= coord(1,j);
      y= z * bin_width;
    }
    else {
      bin_width= coord(k+1,j) - coord(k,j);
      y= coord(k,j) + (z-k)*bin_width;
    }
    // transform from normalized bin coordinates to x space.
    x[j] = _xl[j] + y*_delx[j];

    // update this bin's calculated volume
    vol *= bin_width;
  }
}



////////////////////////////////////////////////////////////////////////////////
/// Reset the specified array of box indices to refer to the first box
/// in the standard traversal order.

void RooGrid::firstBox(UInt_t box[]) const
{
  for(UInt_t i= 0; i < _dim; i++) box[i]= 0;
}



////////////////////////////////////////////////////////////////////////////////
/// Update the specified array of box indices to refer to the next box
/// in the standard traversal order and return true, or else return
/// false if we the indices already refer to the last box.

bool RooGrid::nextBox(UInt_t box[]) const
{
  // try incrementing each index until we find one that does not roll
  // over, starting from the last index.
  Int_t j(_dim-1);
  while (j >= 0) {
    box[j]= (box[j] + 1) % _boxes;
    if (0 != box[j]) return true;
    j--;
  }
  // if we get here, then there are no more boxes
  return false;
}



////////////////////////////////////////////////////////////////////////////////
/// Print info about this object to the specified stream.

void RooGrid::print(std::ostream& os, bool verbose, std::string const& indent) const
{
  os << "RooGrid: volume = " << getVolume() << std::endl;
  os << indent << "  Has " << getDimension() << " dimension(s) each subdivided into "
     << getNBins() << " bin(s) and sampled with " << _boxes << " box(es)" << std::endl;
  for(std::size_t index= 0; index < getDimension(); index++) {
    os << indent << "  (" << index << ") ["
       << std::setw(10) << _xl[index] << "," << std::setw(10) << _xu[index] << "]" << std::endl;
    if(!verbose) continue;
    for(std::size_t bin= 0; bin < _bins; bin++) {
      os << indent << "    bin-" << bin << " : x = " << coord(bin,index) << " , y = "
    << value(bin,index) << std::endl;
    }
  }
}


////////////////////////////////////////////////////////////////////////////////
/// Add the specified amount to bin[j] of the 1D histograms associated
/// with each axis j.

void RooGrid::accumulate(const UInt_t bin[], double amount)
{
  for(UInt_t j = 0; j < _dim; j++) value(bin[j],j) += amount;
}


////////////////////////////////////////////////////////////////////////////////
/// Refine the grid using the values that have been accumulated so far.
/// The parameter alpha controls the stiffness of the rebinning and should
/// usually be between 1 (stiffer) and 2 (more flexible). A value of zero
/// prevents any rebinning.

void RooGrid::refine(double alpha)
{
  for (UInt_t j = 0; j < _dim; j++) {

    // smooth this dimension's histogram of grid values and calculate the
    // new sum of the histogram contents as grid_tot_j
    double oldg = value(0,j);
    double newg = value(1,j);
    value(0,j)= (oldg + newg)/2;
    double grid_tot_j = value(0,j);
    // this loop implements value(i,j) = ( value(i-1,j)+value(i,j)+value(i+1,j) ) / 3

    UInt_t i;
    for (i = 1; i < _bins - 1; i++) {
      double rc = oldg + newg;
      oldg = newg;
      newg = value(i+1,j);
      value(i,j)= (rc + newg)/3;
      grid_tot_j+= value(i,j);
    }
    value(_bins-1,j)= (newg + oldg)/2;
    grid_tot_j+= value(_bins-1,j);

    // calculate the weights for each bin of this dimension's histogram of values
    // and their sum
    double tot_weight(0);
    for (i = 0; i < _bins; i++) {
      _weight[i] = 0;
      if (value(i,j) > 0) {
   oldg = grid_tot_j/value(i,j);
   /* damped change */
   _weight[i] = TMath::Power(((oldg-1.0)/oldg/log(oldg)), alpha);
      }
      tot_weight += _weight[i];
    }

    double pts_per_bin = tot_weight / _bins;

    double xold;
    double xnew = 0;
    double dw = 0;

    i = 1;
    for (UInt_t k = 0; k < _bins; k++) {
      dw += _weight[k];
      xold = xnew;
      xnew = coord(k+1,j);

      while(dw > pts_per_bin) {
   dw -= pts_per_bin;
   newCoord(i++) = xnew - (xnew - xold) * dw / _weight[k];
      }
    }

    for (UInt_t k = 1 ; k < _bins ; k++) {
      coord( k, j) = newCoord(k);
    }

    coord(_bins, j) = 1;
  }
}
