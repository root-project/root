/*****************************************************************************
 * Project: BaBar detector at the SLAC PEP-II B-factory
 * Package: RooFitCore
 *    File: $Id: RooMCIntegrator.cc,v 1.1 2001/08/17 15:51:58 david Exp $
 * Authors:
 *   DK, David Kirkby, Stanford University, kirkby@hep.stanford.edu
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu
 * History:
 *   08-Aug-2001 WV Created initial version
 *
 * Copyright (C) 2001 Stanford University
 *****************************************************************************/

// -- CLASS DESCRIPTION --
// RooMCIntegrator implements an adaptive multi-dimensional Monte Carlo
// numerical integration, following the VEGAS algorithm originally described
// in G. P. Lepage, J. Comp. Phys. 27, 192(1978). This implementation is
// based on a C version from the 0.9 beta release of the GNU scientific library.

#include "RooFitCore/RooMCIntegrator.hh"
#include "RooFitCore/RooNumber.hh"

#include <math.h>
#include <assert.h>

ClassImp(RooMCIntegrator)
;

RooMCIntegrator::RooMCIntegrator(const RooAbsFunc& function, SamplingMode mode,
				 GeneratorType genType, Bool_t verbose) :
  RooAbsIntegrator(function), _grid(function), _verbose(verbose),
  _genType(genType), _mode(mode), _alpha(1.5)
{
  // check that our grid initialized without errors
  if(!(_valid= _grid.isValid())) return;
  if(_verbose) _grid.Print();
} 

RooMCIntegrator::~RooMCIntegrator() {
}

Double_t RooMCIntegrator::integral() {
  // Evaluate the integral using a fixed number of calls to evaluate the integrand
  // equal to about 10k per dimension. Use the first 5k calls to refine the grid
  // over 5 iterations of 1k calls each, and the remaining 5k calls for a single
  // high statistics integration.

  vegas(AllStages,1000*_grid.getDimension(),5);
  return vegas(ReuseGrid,5000*_grid.getDimension(),1);
}

Double_t RooMCIntegrator::vegas(Stage stage, UInt_t calls, UInt_t iterations, Double_t *absError) {
  // Perform one step of Monte Carlo integration using the specified number of iterations
  // with (approximately) the specified number of integrand evaluation calls per iteration.
  // Use the VEGAS algorithm, starting from the specified stage. Returns the best estimate
  // of the integral. Also sets *absError to the estimated absolute error of the integral
  // estimate if absError is non-zero.

  // reset the grid to its initial state if we are starting from scratch
  if(stage == AllStages) _grid.initialize(*_function);

  // reset the results of previous calculations on this grid, but reuse the grid itself.
  if(stage <= ReuseGrid) {
    _wtd_int_sum = 0;
    _sum_wgts = 0;
    _chi_sum = 0;
    _it_num = 1;
    _samples = 0;
  }

  // refine the results of previous calculations on the current grid.
  if(stage <= RefineGrid) {
    UInt_t bins = RooGrid::maxBins;
    UInt_t boxes = 1;
    UInt_t dim(_grid.getDimension());

    // select the sampling mode for the next step
    if(_mode != ImportanceOnly) {
      // calculate the largest number of equal subdivisions ("boxes") along each
      // axis that results in an average of no more than 2 integrand calls per cell
      boxes = (UInt_t)floor(pow(calls/2.0,1.0/dim));
      // use stratified sampling if we are allowed enough calls (or equivalently,
      // if the dimension is low enough)
      _mode = Importance;
      if (2*boxes >= RooGrid::maxBins) {
	_mode = Stratified;
	// adjust the number of bins and boxes to give an integral number >= 1 of boxes per bin
	Int_t box_per_bin= (boxes > RooGrid::maxBins) ? boxes/RooGrid::maxBins : 1;
	bins= boxes/box_per_bin;
	if(bins > RooGrid::maxBins) bins= RooGrid::maxBins;
	boxes = box_per_bin * bins;	
	if(_verbose) cout << "RooMCIntegrator: using stratified sampling with " << bins << " bins and "
			  << box_per_bin << " boxes/bin" << endl;
      }
      else {
	if(_verbose) cout << "RooMCIntegrator: using importance sampling with " << bins << " bins and "
			  << boxes << " boxes" << endl;
      }
    }

    // calculate the total number of n-dim boxes for this step
    Double_t tot_boxes = pow((Double_t)boxes,(Double_t)dim);

    // increase the total number of calls to get at least 2 calls per box, if necessary
    _calls_per_box = (UInt_t)(calls/tot_boxes);
    if(_calls_per_box < 2) _calls_per_box= 2;
    calls= (UInt_t)(_calls_per_box*tot_boxes);

    // calculate the Jacobean factor: volume/(avg # of calls/bin)
    _jac = _grid.getVolume()*pow((Double_t)bins,(Double_t)dim)/calls;

    // setup our grid to use the calculated number of boxes and bins
    _grid.setNBoxes(boxes);
    if(bins != _grid.getNBins()) _grid.resize(bins);
  }

  // allocate memory for some book-keeping arrays
  UInt_t *box= _grid.createIndexVector();
  UInt_t *bin= _grid.createIndexVector();
  Double_t *x= _grid.createPoint();

  // loop over iterations for this step
  Double_t cum_int(0),cum_sig(0);
  _it_start = _it_num;
  _chisq = 0.0;
  for (UInt_t it = 0; it < iterations; it++) {
    Double_t intgrl(0),intgrl_sq(0),sig(0),jacbin(_jac);
    
    _it_num = _it_start + it;
    
    // reset the values associated with each grid cell
    _grid.resetValues();

    // loop over grid boxes
    _grid.firstBox(box);
    do {
      Double_t m(0),q(0);
      // loop over integrand evaluations within this grid box
      for(UInt_t k = 0; k < _calls_per_box; k++) {
	// generate a random point in this box
	Double_t bin_vol(0);
	_grid.generatePoint(box, x, bin, bin_vol, _genType == QuasiRandom ? kTRUE : kFALSE);
	// evaluate the integrand at the generated point
	Double_t fval= jacbin*bin_vol*integrand(x);	
	// update mean and variance calculations
	Double_t d = fval - m;
	m+= d / (k + 1.0);
	q+= d * d * (k / (k + 1.0));
	// accumulate the results of this evaluation (importance sampling only)
	if (_mode != Stratified) _grid.accumulate(bin, fval*fval);
      }
      intgrl += m * _calls_per_box;
      Double_t f_sq_sum = q * _calls_per_box ;
      sig += f_sq_sum ;

      // accumulate the results for this grid box (stratified sampling only)      
      if (_mode == Stratified) _grid.accumulate(bin, f_sq_sum);
    } while(_grid.nextBox(box));

    // compute final results for this iteration
    Double_t wgt;
    sig = sig / (_calls_per_box - 1.0)  ;    
    if (sig > 0) {
      wgt = 1.0 / sig;
    }
    else if (_sum_wgts > 0) {
      wgt = _sum_wgts / _samples;
    }
    else {
      wgt = 0.0;
    }
    intgrl_sq = intgrl * intgrl;
    _result = intgrl;
    _sigma  = sqrt(sig);
    
    if (wgt > 0.0) {
      _samples++ ;
      _sum_wgts += wgt;
      _wtd_int_sum += intgrl * wgt;
      _chi_sum += intgrl_sq * wgt;
      
      cum_int = _wtd_int_sum / _sum_wgts;
      cum_sig = sqrt (1 / _sum_wgts);
      
      if (_samples > 1) {
	_chisq = (_chi_sum - _wtd_int_sum * cum_int)/(_samples - 1.0);
      }
    }
    else {
      cum_int += (intgrl - cum_int) / (it + 1.0);
      cum_sig = 0.0;
    }         
    if (_verbose) {
      cout << "=== Iteration " << _it_num << " : I = " << intgrl << " +/- " << sqrt(sig) << endl
	   << "    Cummulative : I = " << cum_int << " +/- " << cum_sig << "( chi2 = " << _chisq
	   << ")" << endl;
      // print the grid after the final iteration
      if(it + 1 == iterations) _grid.Print("V");
    }
    _grid.refine(_alpha);
  }

  // cleanup
  delete[] bin;
  delete[] box;
  delete[] x;

  if(absError) *absError = cum_sig;
  return cum_int;
}
