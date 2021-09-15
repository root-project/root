/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   PB, Patrick Bos,     NL eScience Center, p.bos@esciencecenter.nl        *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

#ifndef ROOT_TEST_LIB_H
#define ROOT_TEST_LIB_H

#include "RooWorkspace.h"
#include "RooRandom.h"
#include "RooAddPdf.h"
#include "RooDataSet.h"
#include "RooRealVar.h" // for the dynamic cast to have a complete type

#include <sstream>
#include <memory>  // make_unique
#include <vector>

RooAbsPdf * generate_1D_gaussian_pdf(RooWorkspace &w)
{
   w.factory("Gaussian::g(x[-5,5],mu[0,-3,3],sigma[1])");
   RooAbsPdf *pdf = w.pdf("g");
   return pdf;
}

RooDataSet * generate_1D_dataset(RooWorkspace &w, RooAbsPdf *pdf, unsigned long N_events)
{
   RooDataSet *data = pdf->generate(RooArgSet(*w.var("x")), N_events);
   return data;
}


std::tuple<std::unique_ptr<RooAbsReal>, RooAbsPdf *, RooDataSet *, std::unique_ptr<RooArgSet>>
generate_1D_gaussian_pdf_nll(RooWorkspace &w, unsigned long N_events)
{
   RooAbsPdf *pdf = generate_1D_gaussian_pdf(w);

   RooDataSet *data = generate_1D_dataset(w, pdf, N_events);

   RooRealVar *mu = w.var("mu");
   mu->setVal(-2.9);

   std::unique_ptr<RooAbsReal> nll{pdf->createNLL(*data)};

   // save initial values for the start of all minimizations
   std::unique_ptr<RooArgSet> values = std::make_unique<RooArgSet>(*mu, *pdf, *nll, "values");

   return std::make_tuple(std::move(nll), pdf, data, std::move(values));
}

// return two unique_ptrs, the first because nll is a pointer,
// the second because RooArgSet doesn't have a move ctor
std::tuple<std::unique_ptr<RooAbsReal>, RooAbsPdf *, RooDataSet *, std::unique_ptr<RooArgSet>>
generate_ND_gaussian_pdf_nll(RooWorkspace &w, unsigned int n, unsigned long N_events, bool batch_mode = false) {
  RooArgSet obs_set;

  // create gaussian parameters
  std::vector<double> mean(n), sigma(n);
  for (unsigned ix = 0; ix < n; ++ix) {
    mean[ix] = RooRandom::randomGenerator()->Gaus(0, 2);
    sigma[ix] = 0.1 + abs(RooRandom::randomGenerator()->Gaus(0, 2));
  }

  // create gaussians and also the observables and parameters they depend on
  for (unsigned ix = 0; ix < n; ++ix) {
    std::ostringstream os;
    os << "Gaussian::g" << ix
       << "(x" << ix << "[-10,10],"
       << "m" << ix << "[" << mean[ix] << ",-10,10],"
       << "s" << ix << "[" << sigma[ix] << ",0.1,10])";
    w.factory(os.str().c_str());
  }

  // create uniform background signals on each observable
  for (unsigned ix = 0; ix < n; ++ix) {
    {
      std::ostringstream os;
      os << "Uniform::u" << ix << "(x" << ix << ")";
      w.factory(os.str().c_str());
    }

    // gather the observables in a list for data generation below
    {
      std::ostringstream os;
      os << "x" << ix;
      obs_set.add(*w.arg(os.str().c_str()));
    }
  }

  RooArgSet pdf_set = w.allPdfs();

  // create event counts for all pdfs
  RooArgSet count_set;

  // ... for the gaussians
  for (unsigned ix = 0; ix < n; ++ix) {
    std::stringstream os, os2;
    os << "Nsig" << ix;
    os2 << "#signal events comp " << ix;
    RooRealVar a(os.str().c_str(), os2.str().c_str(), N_events/10, 0., 10*N_events);
    w.import(a);
    // gather in count_set
    count_set.add(*w.arg(os.str().c_str()));
  }
  // ... and for the uniform background components
  for (unsigned ix = 0; ix < n; ++ix) {
    std::stringstream os, os2;
    os << "Nbkg" << ix;
    os2 << "#background events comp " << ix;
    RooRealVar a(os.str().c_str(), os2.str().c_str(), N_events/10, 0., 10*N_events);
    w.import(a);
    // gather in count_set
    count_set.add(*w.arg(os.str().c_str()));
  }

  RooAddPdf* sum = new RooAddPdf("sum", "gaussians+uniforms", pdf_set, count_set);
  w.import(*sum);  // keep sum around after returning

  // --- Generate a toyMC sample from composite PDF ---
  RooDataSet *data = sum->generate(obs_set, N_events);

  std::unique_ptr<RooAbsReal> nll {sum->createNLL(*data, RooFit::BatchMode(batch_mode))};

  // set values randomly so that they actually need to do some fitting
  for (unsigned ix = 0; ix < n; ++ix) {
    {
      std::ostringstream os;
      os << "m" << ix;
      dynamic_cast<RooRealVar *>(w.arg(os.str().c_str()))->setVal(RooRandom::randomGenerator()->Gaus(0, 2));
    }
    {
      std::ostringstream os;
      os << "s" << ix;
      dynamic_cast<RooRealVar *>(w.arg(os.str().c_str()))->setVal(0.1 + abs(RooRandom::randomGenerator()->Gaus(0, 2)));
    }
  }

  // gather all values of parameters, pdfs and nll here for easy
  // saving and restoring
  std::unique_ptr<RooArgSet> all_values = std::make_unique<RooArgSet>(pdf_set, count_set, "all_values");
  all_values->add(*nll);
  all_values->add(*sum);
  for (unsigned ix = 0; ix < n; ++ix) {
    {
      std::ostringstream os;
      os << "m" << ix;
      all_values->add(*w.arg(os.str().c_str()));
    }
    {
      std::ostringstream os;
      os << "s" << ix;
      all_values->add(*w.arg(os.str().c_str()));
    }
  }

  return std::make_tuple(std::move(nll), sum, data, std::move(all_values));
}


#endif //ROOT_TEST_LIB_H
