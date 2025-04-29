/// \cond ROOFIT_INTERNAL

/*
 * Project: RooFit
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef ROO_FORMULA
#define ROO_FORMULA

#include "RooPrintable.h"
#include "RooArgList.h"
#include "RooArgSet.h"
#include "RooFit/EvalContext.h"

#include "TFormula.h"

#include <memory>
#include <vector>
#include <string>

class RooAbsReal;

class RooFormula : public TNamed {
public:
   // Constructors etc.
   RooFormula(const char *name, const char *formula, const RooArgList &varList, bool checkVariables = true);
   RooFormula(const RooFormula &other, const char *name = nullptr);
   TObject *Clone(const char *newName = nullptr) const override { return new RooFormula(*this, newName); }

   RooFormula &operator=(const RooFormula &other) = delete;
   RooFormula &operator=(RooFormula &&other) = delete;

   ////////////////////////////////////////////////////////////////////////////////
   /// Return list of arguments which are used in the formula.
   RooArgSet actualDependents() const { return usedVariables(); }
   bool changeDependents(const RooAbsCollection &newDeps, bool mustReplaceAll, bool nameChange);

   bool ok() const { return _tFormula != nullptr; }
   /// Evaluate all parameters/observables, and then evaluate formula.
   double eval(const RooArgSet *nset = nullptr) const;
   void doEval(RooArgList const &actualVars, RooFit::EvalContext &) const;

   void printMultiline(std::ostream &os, Int_t contents, bool verbose = false, TString indent = "") const;

   std::string formulaString() const { return _tFormula ? _tFormula->GetTitle() : ""; }
   TFormula* getTFormula() const { return _tFormula.get(); }

private:
   std::string processFormula(std::string origFormula) const;
   RooArgList usedVariables() const;
   void installFormulaOrThrow(const std::string &formula);

   std::vector<bool> _varIsUsed;        ///<! Track whether a given variable is in use or not
   RooArgList _origList;                ///<! Original list of dependents
   std::unique_ptr<TFormula> _tFormula; ///<! The formula used to compute values
};

#endif

/// \endcond
