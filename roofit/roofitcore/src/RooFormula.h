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

class RooFormula : public TNamed, public RooPrintable {
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

   /// Return pointer to the parameter with given name.
   /// \return Parameter if in use, nullptr if not in use.
   RooAbsArg *getParameter(const char *name) const { return usedVariables().find(name); }

   /// Return pointer to parameter at given index. This returns
   /// irrespective of whether the parameter is in use.
   RooAbsArg *getParameter(Int_t index) const { return _origList.at(index); }

   bool ok() const { return _tFormula != nullptr; }
   /// Evaluate all parameters/observables, and then evaluate formula.
   double eval(const RooArgSet *nset = nullptr) const;
   void doEval(RooFit::EvalContext &) const;

   /// DEBUG: Dump state information
   void dump() const;

   void printValue(std::ostream &os) const override;
   void printName(std::ostream &os) const override;
   void printTitle(std::ostream &os) const override;
   void printClassName(std::ostream &os) const override;
   void printArgs(std::ostream &os) const override;
   void printMultiline(std::ostream &os, Int_t contents, bool verbose = false, TString indent = "") const override;

   void Print(Option_t *options = nullptr) const override
   {
      // Printing interface (human readable)
      printStream(defaultPrintStream(), defaultPrintContents(options), defaultPrintStyle(options));
   }

   std::string formulaString() const { return _tFormula ? _tFormula->GetTitle() : ""; }
   TFormula* getTFormula() const { return _tFormula.get(); }

private:
   std::string processFormula(std::string origFormula) const;
   RooArgList usedVariables() const;
   std::string reconstructFormula(std::string internalRepr) const;
   void installFormulaOrThrow(const std::string &formulaa);

   RooArgList _origList;                ///<! Original list of dependents
   std::vector<bool> _isCategory;       ///<! Whether an element of the _origList is a category.
   std::unique_ptr<TFormula> _tFormula; ///<! The formula used to compute values
};

#endif

/// \endcond
