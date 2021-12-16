// Author: Rahul Balasubramanian, CERN  12/2021

#include "RooPolyFunc.h"

#include "RooPolynomial.h"
#include "RooRealVar.h"
#include "RooFormulaVar.h"
#include "RooWrapperPdf.h"

#include <initializer_list>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>

#include "gtest/gtest.h"

namespace {

std::unique_ptr<RooPolyFunc> makePolyFunc1D(RooRealVar const &x)
{
   // 1 + x + x^2
   auto polyfunc = std::make_unique<RooPolyFunc>("f", "f", RooArgSet(x));
   polyfunc->addTerm(1.0, x, 0);
   polyfunc->addTerm(1.0, x, 1);
   polyfunc->addTerm(1.0, x, 2);

   return polyfunc;
}

std::unique_ptr<RooPolyFunc> makePolyFunc2D(RooRealVar const &x, RooRealVar const &y)
{
   // 1 + x + y + x^2 + y^2
   auto polyfunc = std::make_unique<RooPolyFunc>("f", "f", RooArgSet(x, y));
   polyfunc->addTerm(1.0, x, 0, y, 0);
   polyfunc->addTerm(1.0, x, 0, y, 1);
   polyfunc->addTerm(1.0, x, 1, y, 0);
   polyfunc->addTerm(1.0, x, 2, y, 0);
   polyfunc->addTerm(1.0, x, 0, y, 2);

   return polyfunc;
}
} // namespace

using Doubles = std::initializer_list<double>;

TEST(RooPolyFunc, WrappedPdfClosure)
{
   RooRealVar x("x", "x", 0., -10., 10.);
   auto polyfunc = makePolyFunc1D(x);
   RooWrapperPdf wrapperpdf("wrappdf", "wrappdf", *polyfunc);

   RooRealVar c0("c0", "c0", 1.);
   RooRealVar c1("c1", "c1", 1.);
   RooRealVar c2("c2", "c2", 1.);
   RooPolynomial pdf("pdf", "pdf", x, RooArgList(c0, c1, c2), /*lowestOrder=*/0.0);

   RooArgSet normSet{x};
   for (double theX : Doubles{-10., -5., -1., -0.5, 0., 0.5, 1., 5., 10.}) {
      x = theX;
      // wrapped pdf of RooPolyFunc should match RooPolynomial
      // EXPECT_FLOAT_EQ(wrapperpdf.getVal(), pdf.getVal())
      EXPECT_FLOAT_EQ(wrapperpdf.getVal(normSet), pdf.getVal(normSet)) << theX;
   }
}

TEST(RooPolyFunc, FormulaVarClosure)
{
   RooRealVar x("x", "x", 0., -10., 10.);
   auto polyfunc = makePolyFunc1D(x);
   RooFormulaVar formula("formula", "formula", "1.0 + 1.0*pow(@0,1) + 1.0*pow(@0,2)", RooArgList(x));

   for (double theX : Doubles{0., 0.5, 1., 5., 10.}) {
      x = theX;
      // RooPolyFunc should match RooPolynomial
      EXPECT_FLOAT_EQ(polyfunc->getVal(), formula.getVal()) << theX;
   }
}

TEST(RooPolyFunc, TaylorExpansionClosure1D)
{
   RooRealVar x("x", "x", 0., -10., 10.);
   auto polyfunc = makePolyFunc1D(x);
   auto taylor = RooPolyFunc::taylorExpand("taylor", "taylor expansion", *polyfunc, RooArgSet{x}, 0.0, 2);
   for (double theX : Doubles{0., 0.5, 1., 5., 10.}) {
      x = theX;
      // Taylor epansion of 2nd degree polynomial from RooPolyFunc
      // should match parabola upto to numerical precision
      EXPECT_FLOAT_EQ(taylor->getVal(), polyfunc->getVal()) << theX;
   }
}

TEST(RooPolyFunc, TaylorExpansionClosure2D)
{
   RooRealVar x("x", "x", 0., -10., 10.);
   RooRealVar y("y", "y", 0., -10., 10.);
   auto polyfunc = makePolyFunc2D(x, y);
   auto taylor = RooPolyFunc::taylorExpand("taylor", "taylor expansion", *polyfunc, RooArgSet{x, y}, 0.0, 2);
   for (double theX : Doubles{0., 0.5, 1., 5., 10.}) {
      for (double theY : Doubles{0., 0.5, 1., 5., 10.}) {
         x = theX;
         y = theY;
         // Taylor epansion of 2nd degree polynomial from RooPolyFunc
         // should match parabola upto to numerical precision
         EXPECT_NEAR(taylor->getVal() / polyfunc->getVal(), 1.0, 0.001) << theX << "," << theY;
      }
   }
}
