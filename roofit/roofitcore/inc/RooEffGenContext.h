/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 *    File: $Id: RooEffGenContext.h,v 1.2 2007/05/11 09:11:30 verkerke Exp $
 * Authors:                                                                  *
 *   GR, Gerhard Raven, NIKHEF/VU, Gerhard.Raven@nikhf.nl                    *
 *                                                                           *
 * Copyright (c) 2005, NIKHEF.  All rights reserved.                         *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/
#ifndef ROO_EFF_GEN_CONTEXT
#define ROO_EFF_GEN_CONTEXT

#include "RooAbsGenContext.h"
class RooAbsPdf;
class RooArgSet;
class RooDataSet;
class RooAbsReal;

class RooEffGenContext : public RooAbsGenContext {
public:
   RooEffGenContext(const RooAbsPdf &model, const RooAbsPdf &pdf, const RooAbsReal &eff, const RooArgSet &vars,
                    const RooDataSet *prototype = nullptr, const RooArgSet *auxProto = nullptr, bool verbose = false,
                    const RooArgSet *forceDirect = nullptr);

   void printMultiline(std::ostream &os, Int_t content, bool verbose = false, TString indent = "") const override;

protected:
   void initGenerator(const RooArgSet &theEvent) override;
   void generateEvent(RooArgSet &theEvent, Int_t remaining) override;

private:
   inline void initializeEff(RooAbsReal const &eff)
   {
      _eff = dynamic_cast<RooAbsReal *>(_cloneSet.find(eff.GetName()));
   }

   RooArgSet _cloneSet;          ///< Internal clone of p.d.f.
   RooAbsReal *_eff;             ///< Pointer to efficiency function
   std::unique_ptr<RooAbsGenContext> _generator; ///< Generator context for p.d.f
   RooArgSet _vars;              ///< Vars to generate
   double _maxEff;               ///< Maximum of efficiency in vars

   ClassDefOverride(RooEffGenContext, 0); // Context for generating a dataset from a PDF
};
#endif
