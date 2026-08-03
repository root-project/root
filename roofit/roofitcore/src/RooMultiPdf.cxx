/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitModels                                                     *
 * @(#)root/roofit:$Id$
 * Authors:                                                                  *
 *   RW, Ruddick William  UC Colorado        wor@slac.stanford.edu           *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
/** \class RooMultiPdf
   \ingroup Roofit

   The class RooMultiPdf allows for the creation of a RooMultiPdf object,
   which can switch between previously set probability density functions (PDF)
   by setting their associated indices.*/

#include <RooMultiPdf.h>
#include <RooFit/Detail/MathFuncs.h>
#include <RooConstVar.h>

#include <sstream>
#include <stdexcept>

// Constructing a RooMultiPdf
//  parameter name : The name of the RooMultiPdf object
//  parameter title : Display title in plots
//  parameter _x :  variable used to select which PDF that selects the active PDF.
//  parameter _c : A list of the pdfs.The index of each PDF in the list should match the values in _x
RooMultiPdf::RooMultiPdf(const char *name, const char *title, RooCategory &_x, const RooArgList &_c)
   : RooAbsPdf(name, title), // call of constructor base class RooAbsPdf passing it name and title
     c("_pdfs", "The list of pdfs", this),
     corr("_corrs", "The list of correction factors", this),
     x("_index", "the pdf index", this, _x)

// parameter corr : Holds correction factors - number of free parameters
// in each PDF held in the RooMultiPdf object
{
   int count = 0;

   c.add(_c);
   for (RooAbsArg *pdf : c) {
      // This is done by the user BUT is there a way to do it at construction?
      _x.defineType(("_pdf" + std::to_string(count)).c_str(), count);
      std::unique_ptr<RooArgSet> variables(pdf->getVariables());
      std::unique_ptr<RooAbsCollection> nonConstVariables(variables->selectByAttrib("Constant", false));
      // Isn't there a better way to hold on to these values?
      std::string corrName = std::string{"const"} + pdf->GetName();
      corr.addOwned(std::make_unique<RooConstVar>(corrName.c_str(), "", nonConstVariables->size()));
      count++;
   }
   _oldIndex = fIndex;
}

// Here new RooMultiPdf copy is created that references the same components as the original.
// Copies c, corr, x
RooMultiPdf::RooMultiPdf(const RooMultiPdf &other, const char *name)
   : RooAbsPdf(other, name), c("_pdfs", this, other.c), corr("_corrs", this, other.corr), x("_index", this, other.x)
{
   fIndex = other.fIndex;
   _oldIndex = fIndex;
   cFactor = other.cFactor; // correction to 2*NLL by default is -> 2*0.5 per param
}

// evaluate() and getLogVal() define how the value and log-value of the RooMultiPdf are computed at a given point.
// RooMultiPdf must have both of these so it RooFit can treat it like a PDF
Double_t RooMultiPdf::evaluate() const
{
   double val = getCurrentPdf()->getVal(c.nset());
   _oldIndex = x;
   return val;
}

Double_t RooMultiPdf::getLogVal(const RooArgSet *nset) const
{
   double logval = getCurrentPdf()->getLogVal(nset);
   _oldIndex = x;
   return logval;
}

void RooMultiPdf::getParametersHook(const RooArgSet *nset, RooArgSet *list, bool stripDisconnected) const
{
   if (!stripDisconnected)
      return;

   list->removeAll();
   getCurrentPdf()->getParameters(nset, *list, stripDisconnected);
   list->add(*x);
}

////////////////////////////////////////////////////////////////////////////////
/// The index category of a RooMultiPdf selects between alternative hypotheses
/// for the observables (as used in the discrete profiling method), it does not
/// partition the data into subsamples like the index category of a
/// RooSimultaneous. It therefore has no well-defined prior probability, and
/// there is no meaningful way to sample it in generate(). Fix the index to a
/// specific state (e.g. with RooCategory::setIndex()) and generate from the
/// resulting single-component pdf instead.

RooAbsGenContext *RooMultiPdf::genContext(const RooArgSet &vars, const RooDataSet *prototype,
                                           const RooArgSet *auxProto, bool verbose) const
{
   RooArgSet allVars{vars};
   if (prototype)
      allVars.add(*prototype->get(), true);

   if (allVars.find(x->GetName())) {
      std::stringstream ss;
      ss << "RooMultiPdf::genContext(" << GetName() << "): the index category \"" << x->GetName()
         << "\" selects between alternative hypotheses for the observables and is not itself an observable "
            "with a well-defined prior probability, unlike the index category of a RooSimultaneous. "
            "Generating data with this index among the variables to generate is therefore not supported. "
            "Fix the index to a specific state (e.g. with RooCategory::setIndex()) and generate from the "
            "resulting single-component pdf instead.";
      throw std::invalid_argument(ss.str());
   }

   return RooAbsPdf::genContext(vars, prototype, auxProto, verbose);
}
