/*
 * Project: RooFit
 * Author:
 *   Ruggero Turra <ruggero.turra@cern.ch>, 2016
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_RooSpline_h
#define RooFit_RooSpline_h

#include <RooAbsReal.h>
#include <RooRealProxy.h>

#include <TSpline.h>

#include <ROOT/RSpan.hxx>

#include <vector>

class TGraph;

class RooSpline : public RooAbsReal {
public:
   RooSpline() = default;
   RooSpline(const char *name, const char *title, RooAbsReal &x, std::span<const double> x0, std::span<const double> y0,
             int order = 3, bool logx = false, bool logy = false);
   RooSpline(const char *name, const char *title, RooAbsReal &x, const TGraph &gr, int order = 3, bool logx = false,
             bool logy = false);
   RooSpline(const RooSpline &other, const char *name = nullptr);

   /// Virtual copy constructor.
   /// \param[in] newname The name of the cloned object (optional).
   TObject *clone(const char *newname) const override { return new RooSpline(*this, newname); }

protected:
   double evaluate() const override;

private:
   std::unique_ptr<TSpline> _spline; ///< The spline object.
   RooRealProxy _x;                  ///< The independent variable.
   bool _logx = false;               ///< Flag indicating logarithmic scaling of x values.
   bool _logy = false;               ///< Flag indicating logarithmic scaling of y values.

   ClassDefOverride(RooSpline, 1); // A RooFit class for creating spline functions
};
#endif
