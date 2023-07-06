/*
 * Project: RooFit
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_RooExpPoly_h
#define RooFit_RooExpPoly_h

#include <RooAbsPdf.h>
#include <RooRealProxy.h>
#include <RooListProxy.h>

class RooExpPoly : public RooAbsPdf {
public:
   RooExpPoly() {}
   RooExpPoly(const char *name, const char *title, RooAbsReal &x, const RooArgList &coefList, int lowestOrder = 1);

   RooExpPoly(const RooExpPoly &other, const char *name = nullptr);
   TObject *clone(const char *newname) const override { return new RooExpPoly(*this, newname); }

   /// Get the x variable.
   RooAbsReal const &x() const { return _x.arg(); }

   /// Get the coefficient list.
   RooArgList const &coefList() const { return _coefList; }

   /// Return the order for the first coefficient in the list.
   int lowestOrder() const { return _lowestOrder; }

   double getLogVal(const RooArgSet *nset) const override;

   std::string getFormulaExpression(bool expand) const;

   int getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char *rangeName = nullptr) const override;
   double analyticalIntegral(int code, const char *rangeName = nullptr) const override;

   void adjustLimits();

protected:
   RooRealProxy _x;
   RooListProxy _coefList;
   int _lowestOrder;

   // CUDA support
   void computeBatch(double *output, size_t size, RooFit::Detail::DataMap const &) const override;
   inline bool canComputeBatchWithCuda() const override { return true; }

   /// Evaluation
   double evaluate() const override;
   double evaluateLog() const;

   ClassDefOverride(RooExpPoly, 1) // ExpPoly PDF
};

#endif
