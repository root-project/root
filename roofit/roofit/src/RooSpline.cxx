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

#include <RooSpline.h>

#include <RooMsgService.h>

#include <TGraph.h>

#include <sstream>
#include <string>
#include <vector>

/** \class RooSpline
    \ingroup Roofit
    \brief A RooFit class for creating spline functions.

This class provides the functionality to create spline functions in RooFit,
, using ROOT TSpline. It supports splines of order 3 or 5. It also support
interpolation in the log-space (x or y), for example
exp(spline({x0}, {log y0})), useful when you have something (as xsections)
that is more similar to exponentials than polynomials.

Usage example:
~~~ {.cpp}
RooRealVar x{"x", "x", 0, 5};

std::vector<double> x0{1., 2, 3};
std::vector<double> y0{10., 20, 50};

RooSpline spline{"myspline", "my spline", x, x0, y0};

auto frame = x.frame();
spline.plotOn(frame);
frame->Draw();
~~~
*/


/// Constructor for creating a spline from a TGraph.
/// \param[in] name The name of the spline.
/// \param[in] title The title of the spline.
/// \param[in] x The independent variable.
/// \param[in] gr The input TGraph containing the data points.
/// \param[in] order The order of the spline (3 or 5).
/// \param[in] logx If true, the x values are logarithmically scaled before spline creation.
/// \param[in] logy If true, the y values are logarithmically scaled before spline creation.
RooSpline::RooSpline(const char *name, const char *title, RooAbsReal &x, const TGraph &gr, int order, bool logy,
                     bool logx)
   : RooSpline(name, title, x, {gr.GetX(), gr.GetX() + gr.GetN()}, {gr.GetY(), gr.GetY() + gr.GetN()}, order, logx,
               logy)
{
}

/// Constructor for creating a spline from raw data.
/// \param[in] name The name of the spline.
/// \param[in] title The title of the spline.
/// \param[in] x The independent variable.
/// \param[in] x0 The array of x values for the spline points.
/// \param[in] y0 The array of y values for the spline points.
/// \param[in] order The order of the spline (3 or 5).
/// \param[in] logx If true, the x values are logarithmically scaled before spline creation.
/// \param[in] logy If true, the y values are logarithmically scaled before spline creation.
RooSpline::RooSpline(const char *name, const char *title, RooAbsReal &x, std::span<const double> x0,
                     std::span<const double> y0, int order, bool logx, bool logy)
   : RooAbsReal{name, title}, _x{"x", "x", this, x}, _logx{logx}, _logy{logy}
{
   const std::string title_spline = std::string(title) + "_spline";
   if (x0.size() != y0.size()) {
      std::stringstream errMsg;
      errMsg << "RooSpline::ctor(" << GetName() << ") ERROR: size of x and y are not equal";
      coutE(InputArguments) << errMsg.str() << std::endl;
      throw std::invalid_argument(errMsg.str());
   }

   // To do the logarithm inplace if necessary.
   std::vector<double> x0Copy;
   x0Copy.assign(x0.begin(), x0.end());
   std::vector<double> y0Copy;
   y0Copy.assign(y0.begin(), y0.end());

   if (_logx) {
      for (auto &xRef : x0Copy) {
         xRef = std::log(xRef);
      }
   }
   if (_logy) {
      for (auto &yRef : y0Copy) {
         yRef = std::log(yRef);
      }
   }

   if (order == 3) {
      _spline = std::make_unique<TSpline3>(title_spline.c_str(), &x0Copy[0], &y0Copy[0], x0.size());
   } else if (order == 5) {
      _spline = std::make_unique<TSpline5>(title_spline.c_str(), &x0Copy[0], &y0Copy[0], x0.size());
   } else {
      std::stringstream errMsg;
      errMsg << "supported orders are 3 or 5";
      coutE(InputArguments) << errMsg.str() << std::endl;
      throw std::invalid_argument(errMsg.str());
   }
}

/// Copy constructor.
/// \param[in] other The RooSpline object to copy from.
/// \param[in] name The name of the new RooSpline object (optional).
RooSpline::RooSpline(const RooSpline &other, const char *name)
   : RooAbsReal(other, name),
     _spline(static_cast<TSpline *>(other._spline->Clone())),
     _x("x", this, other._x),
     _logx(other._logx),
     _logy(other._logy)
{
}

/// Evaluate the spline function at the current point.
double RooSpline::evaluate() const
{
   const double x_val = (!_logx) ? _x : std::exp(_x);
   return (!_logy) ? _spline->Eval(x_val) : std::exp(_spline->Eval(x_val));
}
