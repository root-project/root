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

#include <RooAddPdf.h>
#include <RooDataSet.h>
#include <RooProdPdf.h>
#include <RooRandom.h>
#include <RooRealVar.h> // for the dynamic cast to have a complete type
#include <RooWorkspace.h>

#include <sstream>
#include <memory> // make_unique
#include <vector>

RooAbsPdf *generate_1D_gaussian_pdf(RooWorkspace &ws)
{
   ws.factory("Gaussian::g(x[-5,5],mu[0,-3,3],sigma[1])");
   RooAbsPdf *pdf = ws.pdf("g");
   return pdf;
}

std::unique_ptr<RooDataSet> generate_1D_dataset(RooWorkspace &ws, RooAbsPdf *pdf, unsigned long nEvents)
{
   return std::unique_ptr<RooDataSet>{pdf->generate(RooArgSet(*ws.var("x")), nEvents)};
}

std::tuple<std::unique_ptr<RooAbsReal>, RooAbsPdf *, std::unique_ptr<RooDataSet>, std::unique_ptr<RooArgSet>>
generate_1D_gaussian_pdf_nll(RooWorkspace &ws, unsigned long nEvents)
{
   RooAbsPdf *pdf = generate_1D_gaussian_pdf(ws);

   std::unique_ptr<RooDataSet> data{generate_1D_dataset(ws, pdf, nEvents)};

   RooRealVar *mu = ws.var("mu");
   mu->setVal(-2.9);

   std::unique_ptr<RooAbsReal> nll{pdf->createNLL(*data)};

   // save initial values for the start of all minimizations
   auto values = std::make_unique<RooArgSet>(*mu, *pdf, *nll, "values");

   return {std::move(nll), pdf, std::move(data), std::move(values)};
}

// return two unique_ptrs, the first because nll is a pointer,
// the second because RooArgSet doesn't have a move constructor
std::tuple<std::unique_ptr<RooAbsReal>, RooAbsPdf *, std::unique_ptr<RooDataSet>, std::unique_ptr<RooArgSet>>
generate_ND_gaussian_pdf_nll(RooWorkspace &ws, unsigned int n, unsigned long nEvents, RooFit::EvalBackend evalBackend)
{
   RooArgSet obs_set;

   // create gaussian parameters
   std::vector<double> mean(n);
   std::vector<double> sigma(n);
   for (unsigned ix = 0; ix < n; ++ix) {
      mean[ix] = RooRandom::randomGenerator()->Gaus(0, 2);
      sigma[ix] = 0.1 + std::abs(RooRandom::randomGenerator()->Gaus(0, 2));
   }

   // create gaussians and also the observables and parameters they depend on
   RooArgSet signals;
   for (unsigned ix = 0; ix < n; ++ix) {
      std::ostringstream os;
      os << "Gaussian::g" << ix << "(x" << ix << "[-10,10],"
         << "m" << ix << "[" << mean[ix] << ",-10,10],"
         << "s" << ix << "[" << sigma[ix] << ",0.1,10])";
      signals.add(*ws.factory(os.str()));
   }

   // create uniform background signals on each observable
   RooArgSet backgrounds;
   for (unsigned ix = 0; ix < n; ++ix) {
      {
         std::ostringstream os;
         os << "Uniform::u" << ix << "(x" << ix << ")";
         backgrounds.add(*ws.factory(os.str()));
      }

      // gather the observables in a list for data generation below
      {
         std::ostringstream os;
         os << "x" << ix;
         obs_set.add(*ws.arg(os.str()));
      }
   }

   // The ND signal and background pdfs
   RooProdPdf sigPdf{"sig_pdf", "sig_pdf", signals};
   RooProdPdf bkgPdf{"bkg_pdf", "bkg_pdf", backgrounds};

   // Signal and background yields
   RooRealVar nSig{"n_sig", "n_sig", nEvents / 2., 0., 5. * nEvents};
   RooRealVar nBkg{"n_bkg", "n_bkg", nEvents / 2., 0., 5. * nEvents};

   ws.import(RooAddPdf("sum", "gaussians+uniforms", {sigPdf, bkgPdf}, {nSig, nBkg}));
   RooAbsPdf *sum = ws.pdf("sum");

   // --- Generate a toyMC sample from composite PDF ---
   std::unique_ptr<RooDataSet> data{sum->generate(obs_set)};

   std::unique_ptr<RooAbsReal> nll{sum->createNLL(*data, evalBackend)};

   // set values randomly so that they actually need to do some fitting
   for (unsigned ix = 0; ix < n; ++ix) {
      {
         std::ostringstream os;
         os << "m" << ix;
         auto *val = static_cast<RooRealVar *>(ws.arg(os.str().c_str()));
         val->setVal(RooRandom::randomGenerator()->Uniform(val->getMin(), val->getMax()));
      }
      {
         std::ostringstream os;
         os << "s" << ix;
         auto *val = static_cast<RooRealVar *>(ws.arg(os.str().c_str()));
         val->setVal(RooRandom::randomGenerator()->Uniform(val->getMin(), val->getMax()));
      }
   }

   // gather all values of parameters, pdfs and nll here for easy
   // saving and restoring
   auto all_values = std::make_unique<RooArgSet>(*ws.arg("n_sig"), *ws.arg("n_bkg"), "all_values");
   for (unsigned ix = 0; ix < n; ++ix) {
      {
         std::ostringstream os;
         os << "m" << ix;
         all_values->add(*ws.arg(os.str()));
      }
      {
         std::ostringstream os;
         os << "s" << ix;
         all_values->add(*ws.arg(os.str()));
      }
   }

   return {std::move(nll), sum, std::move(data), std::move(all_values)};
}

#endif // ROOT_TEST_LIB_H
