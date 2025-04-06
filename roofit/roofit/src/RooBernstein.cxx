/*
 * Project: RooFit
 *
 * Copyright (c) 2024, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

/** \class RooBernstein
    \ingroup Roofit

Bernstein basis polynomials are positive-definite in the range [0,1].
In this implementation, we extend [0,1] to be the range of the parameter.
There are n+1 Bernstein basis polynomials of degree n:
\f[
 B_{i,n}(x) = \begin{pmatrix}n \\\ i \end{pmatrix} x^i \cdot (1-x)^{n-i}
\f]
Thus, by providing n coefficients that are positive-definite, there
is a natural way to have well-behaved polynomial PDFs. For any n, the n+1 polynomials
'form a partition of unity', i.e., they sum to one for all values of x.
They can be used as a basis to span the space of polynomials with degree n or less:
\f[
 PDF(x, c_0, ..., c_n) = \mathcal{N} \cdot \sum_{i=0}^{n} c_i \cdot B_{i,n}(x).
\f]
By giving n+1 coefficients in the constructor, this class constructs the n+1
polynomials of degree n, and sums them to form an element of the space of polynomials
of degree n. \f$ \mathcal{N} \f$ is a normalisation constant that takes care of the
cases where the \f$ c_i \f$ are not all equal to one.

See also
http://www.idav.ucdavis.edu/education/CAGDNotes/Bernstein-Polynomials.pdf
**/

#include <RooBernstein.h>
#include <RooRealVar.h>
#include <RooBatchCompute.h>

#include <RooFit/Detail/MathFuncs.h>


RooBernstein::RooBernstein(const char *name, const char *title, RooAbsRealLValue &x, const RooArgList &coefList)
   : RooAbsPdf(name, title), _x("x", "Dependent", this, x), _coefList("coefList", "List of coefficients", this)
{
   _coefList.addTyped<RooAbsReal>(coefList);
}

RooBernstein::RooBernstein(const RooBernstein &other, const char *name)
   : RooAbsPdf(other, name),
     _x("x", this, other._x),
     _coefList(this, other._coefList),
     _refRangeName{other._refRangeName},
     _buffer{other._buffer}
{
}

/// Force use of a given normalisation range.
/// Needed for functions or PDFs (e.g. RooAddPdf) whose shape depends on the choice of normalisation.
void RooBernstein::selectNormalizationRange(const char *rangeName, bool force)
{
   if (rangeName && (force || !_refRangeName.empty())) {
      _refRangeName = rangeName;
   }
}

void RooBernstein::fillBuffer() const
{
   _buffer.resize(_coefList.size() + 2); // will usually be a no-op because size stays the same
   std::size_t n = _coefList.size();
   for (std::size_t i = 0; i < n; ++i) {
      _buffer[i] = static_cast<RooAbsReal &>(_coefList[i]).getVal();
   }
   std::tie(_buffer[n], _buffer[n + 1]) = _x->getRange(_refRangeName.empty() ? nullptr : _refRangeName.c_str());
}

double RooBernstein::evaluate() const
{
   fillBuffer();
   return RooFit::Detail::MathFuncs::bernstein(_x, xmin(), xmax(), _buffer.data(), _coefList.size());
}

/// Compute multiple values of Bernstein distribution.
void RooBernstein::doEval(RooFit::EvalContext &ctx) const
{
   fillBuffer();
   RooBatchCompute::compute(ctx.config(this), RooBatchCompute::Bernstein, ctx.output(), {ctx.at(_x)}, _buffer);
}

Int_t RooBernstein::getAnalyticalIntegral(RooArgSet &allVars, RooArgSet &analVars, const char * /*rangeName*/) const
{
   return matchArgs(allVars, analVars, _x) ? 1 : 0;
}

double RooBernstein::analyticalIntegral(Int_t /*code*/, const char *rangeName) const
{
   fillBuffer();
   return RooFit::Detail::MathFuncs::bernsteinIntegral(_x.min(rangeName), _x.max(rangeName), xmin(), xmax(),
                                                                 _buffer.data(), _coefList.size());
}
