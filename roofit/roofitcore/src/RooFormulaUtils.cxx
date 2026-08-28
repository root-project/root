/// \cond ROOFIT_INTERNAL

/*****************************************************************************
 * Project: RooFit                                                           *
 * Package: RooFitCore                                                       *
 * @(#)root/roofitcore:$Id$
 * Authors:                                                                  *
 *   WV, Wouter Verkerke, UC Santa Barbara, verkerke@slac.stanford.edu       *
 *   DK, David Kirkby,    UC Irvine,         dkirkby@uci.edu                 *
 *                                                                           *
 * Copyright (c) 2000-2005, Regents of the University of California          *
 *                          and Stanford University. All rights reserved.    *
 *                                                                           *
 * Redistribution and use in source and binary forms,                        *
 * with or without modification, are permitted according to the terms        *
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)             *
 *****************************************************************************/

/**
\file RooFormulaUtils.cxx
\ingroup Roofitcore

Free functions to translate and evaluate user-defined expressions of
RooAbsArgs. See RooFormulaUtils.h for a description of the supported
expression dialect. To debug the formula preprocessing, activate the
RooFit::DEBUG message level for the RooFit::InputArguments topic.

### Evaluation backends
By default, makeEvaluator() compiles the expression with a small built-in
parser and evaluates it without any use of the interpreter/JIT. Expressions
the parser does not support silently fall back to the traditional TFormula
(cling JIT) backend, so any expression that worked before keeps working. The
environment variable `ROOFIT_FORMULA_BACKEND` overrides this: `tformula`
always uses the TFormula backend, and `ast` disables the fallback, turning
unsupported expressions into hard errors (useful for testing).

Batch evaluation (doEvalFormula()) of expressions on the built-in backend is
chunk-vectorized via the RooBatchCompute library, see doEvalFormula() for the
numerical implications.
**/

#include "RooFormulaUtils.h"
#include "RooAbsBinning.h"
#include "RooAbsCategory.h"
#include "RooAbsReal.h"
#include "RooAbsRealLValue.h"
#include "RooArgList.h"
#include "RooCurve.h"
#include "RooFitImplHelpers.h"
#include "RooMsgService.h"
#include "RooBatchCompute.h"
#include "RooExprEvaluator.h"
#include "RooFormulaParser.h"
#include "RooTFormulaEvaluator.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <sstream>

using std::sregex_iterator;

namespace {

/// Which evaluation backend the formula should use, from ROOFIT_FORMULA_BACKEND.
enum class FormulaBackend {
   AstWithFallback = 0, ///< default: JIT-free parser, silent TFormula fallback
   AstOnly = 1,         ///< `ast`: fail loudly instead of falling back
   TFormulaOnly = 2     ///< `tformula`: always use the TFormula backend
};

std::mutex gFormulaBackendMutex;
int gFormulaBackendCached = -1; ///< -1: environment variable not read yet
bool gFormulaBackendWarnedChange = false;
std::string gFormulaBackendEnvSeen; ///< raw env value at first read ("" if unset)

/// Read ROOFIT_FORMULA_BACKEND once (thread-safely) and cache the result. On
/// later reads, warn (once) if the environment variable no longer matches the
/// cached value: changing it after the first evaluator creation has no
/// effect.
FormulaBackend formulaBackend()
{
   std::lock_guard<std::mutex> lock{gFormulaBackendMutex};
   const char *env = std::getenv("ROOFIT_FORMULA_BACKEND");
   const std::string val = env ? env : "";
   if (gFormulaBackendCached < 0) {
      FormulaBackend mode = FormulaBackend::AstWithFallback;
      if (val == "ast") {
         mode = FormulaBackend::AstOnly;
      } else if (val == "tformula") {
         mode = FormulaBackend::TFormulaOnly;
      } else if (!val.empty()) {
         oocoutW(nullptr, InputArguments) << "Ignoring unknown ROOFIT_FORMULA_BACKEND value '" << val
                                          << "' (supported: \"ast\", \"tformula\")" << std::endl;
      }
      gFormulaBackendCached = static_cast<int>(mode);
      gFormulaBackendEnvSeen = val;
      gFormulaBackendWarnedChange = false;
   } else if (val != gFormulaBackendEnvSeen && !gFormulaBackendWarnedChange) {
      gFormulaBackendWarnedChange = true;
      oocoutW(nullptr, InputArguments) << "ROOFIT_FORMULA_BACKEND changed from '" << gFormulaBackendEnvSeen << "' to '"
                                       << val << "' after it was first read; the change has no effect in this process"
                                       << std::endl;
   }
   return static_cast<FormulaBackend>(gFormulaBackendCached);
}

} // namespace

namespace RooFormulaInternal {

void resetFormulaBackendForTesting()
{
   std::lock_guard<std::mutex> lock{gFormulaBackendMutex};
   gFormulaBackendCached = -1;
   gFormulaBackendWarnedChange = false;
   gFormulaBackendEnvSeen.clear();
}

} // namespace RooFormulaInternal

namespace {

/// Convert `@i`-style references to `x[i]`.
void convertArobaseReferences(std::string &formula)
{
   bool match = false;
   for (std::size_t i = 0; i < formula.size(); ++i) {
      if (match && !isdigit(formula[i])) {
         formula.insert(formula.begin() + i, ']');
         i += 1;
         match = false;
      } else if (!match && formula[i] == '@') {
         formula[i] = 'x';
         formula.insert(formula.begin() + i + 1, '[');
         i += 1;
         match = true;
      }
   }
   if (match)
      formula += ']';
}

/// Replace all occurrences of `what` with `with` inside of `inOut`.
void replaceAll(std::string &inOut, std::string_view what, std::string_view with)
{
   for (std::string::size_type pos{}; inOut.npos != (pos = inOut.find(what.data(), pos, what.length()));
        pos += with.length()) {
      inOut.replace(pos, what.length(), with.data(), with.length());
   }
}

/// Find the word boundaries with a static std::regex and return a bool vector
/// flagging their positions. The end of the string is considered a word
/// boundary.
std::vector<bool> getWordBoundaryFlags(std::string const &s)
{
   static const std::regex r{"\\b"};
   std::vector<bool> out(s.size() + 1);

   for (auto i = std::sregex_iterator(s.begin(), s.end(), r); i != std::sregex_iterator(); ++i) {
      std::smatch m = *i;
      out[m.position()] = true;
   }

   // The end of a string is also a word boundary
   out[s.size()] = true;

   return out;
}

// Check if a RooConstVar whose name is a number (e.g. from RooFit::RooConst())
// has a value that matches its name.
bool isNumericNameValid(RooAbsArg &arg)
{
   // Extract the value from the RooAbsArg
   std::stringstream ss;
   ss << arg;
   try {
      return std::stod(arg.GetName()) == std::stod(ss.str());
   } catch (const std::exception &) {
      throw std::invalid_argument(std::string("RooConstVar named ") + arg.GetName() +
                                  " has a name or value that cannot be converted to a valid number");
   }
}

/// Replace all named references with "x[i]"-style.
void replaceVarNamesWithIndexStyle(std::string &formula, RooArgList const &varList)
{
   std::vector<bool> isWordBoundary = getWordBoundaryFlags(formula);
   for (unsigned int i = 0; i < varList.size(); ++i) {
      std::string_view varName = varList[i].GetName();

      // If the RooAbsArg has a number as name, we perform checks
      std::string varNameStr{varName};
      static const std::regex pureNumberNameRegex("^\\s*\\d+(\\.\\d+)?\\s*$");
      if (std::regex_match(varNameStr, pureNumberNameRegex)) { // Name is a number
         // If the RooAbsArg is a RooConstVar having (double)name == value
         // we don't perform substitution
         if (varList[i].InheritsFrom("RooConstVar") && isNumericNameValid(varList[i])) {
            continue;
         } else {
            std::stringstream exceptionSs;
            exceptionSs << "Variable '" << varName << "' is not a valid argument for RooFormulaVar. "
                        << "Variables with a name that is a number can only be of type RooConstVar "
                        << "and have value equal to the name";
            throw std::invalid_argument(exceptionSs.str());
         }
      }

      std::stringstream replacementStream;
      replacementStream << "x[" << i << "]";
      std::string replacement = replacementStream.str();

      for (std::string::size_type pos{}; formula.npos != (pos = formula.find(varName.data(), pos, varName.length()));
           pos += replacement.size()) {

         std::string::size_type next = pos + varName.length();

         // The matched variable name has to be surrounded by word boundaries
         if (!isWordBoundary[pos] || !isWordBoundary[next])
            continue;

         // Veto '[' and ']' as next characters. If the variable is called `x`
         // or `0`, this might otherwise replace `x[0]`.
         if (next < formula.size() && (formula[next] == '[' || formula[next] == ']')) {
            continue;
         }

         // As we replace substrings in the middle of the string, we also have
         // to update the word boundary flag vector. Note that we don't care
         // the word boundaries in the `x[i]` are correct, as it has already
         // been replaced.
         std::size_t nOld = varName.length();
         std::size_t nNew = replacement.size();
         auto wbIter = isWordBoundary.begin() + pos;
         if (nNew > nOld) {
            isWordBoundary.insert(wbIter + nOld, nNew - nOld, false);
         } else if (nNew < nOld) {
            isWordBoundary.erase(wbIter + nNew, wbIter + nOld);
         }

         // Do the actual replacement
         formula.replace(pos, varName.length(), replacement);
      }

      oocxcoutD(static_cast<TObject *>(nullptr), InputArguments)
         << "Preprocessing formula: replace named references: " << varName << " --> " << replacement << "\n\t"
         << formula << std::endl;
   }
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
/// Process a formula by replacing all ordinal and name references by `x[i]`,
/// where `i` matches the position of the argument in `varList`, and category
/// state references such as `leptonMulti::one` by the category index. The
/// caller name is used in debug and error messages.
std::string
RooFormulaUtils::processFormula(std::string formula, RooArgList const &varList, std::string const &callerName)
{
   // WARNING to developers: people use these functions a lot via RooGenericPdf
   // and RooFormulaVar! Performance matters here. Avoid non-static
   // std::regex, because constructing these can become a bottleneck because
   // of the regex compilation.

   oocxcoutD(static_cast<TObject *>(nullptr), InputArguments)
      << "Preprocessing formula step 1: find category tags (catName::catState) in " << formula << std::endl;

   // Step 1: Find all category tags and the corresponding index numbers
   static const std::regex categoryReg("(\\w+)::(\\w+)");
   std::map<std::string, int> categoryStates;
   for (sregex_iterator matchIt = sregex_iterator(formula.begin(), formula.end(), categoryReg);
        matchIt != sregex_iterator(); ++matchIt) {
      assert(matchIt->size() == 3);
      const std::string fullMatch = (*matchIt)[0];
      const std::string catName = (*matchIt)[1];
      const std::string catState = (*matchIt)[2];

      const auto catVariable = dynamic_cast<const RooAbsCategory *>(varList.find(catName.c_str()));
      if (!catVariable) {
         oocxcoutD(static_cast<TObject *>(nullptr), InputArguments)
            << "Formula " << callerName << " uses '::' to reference a category state as '" << fullMatch
            << "' but a category '" << catName << "' cannot be found in the input variables." << std::endl;
         continue;
      }

      if (!catVariable->hasLabel(catState)) {
         oocoutE(static_cast<TObject *>(nullptr), InputArguments)
            << "Formula " << callerName << " uses '::' to reference a category state as '" << fullMatch
            << "' but the category '" << catName << "' does not seem to have the state '" << catState << "'."
            << std::endl;
         throw std::invalid_argument(formula);
      }
      const int catNum = catVariable->lookupIndex(catState);

      categoryStates[fullMatch] = catNum;
      oocxcoutD(static_cast<TObject *>(nullptr), InputArguments)
         << "\n\t" << fullMatch << "\tname=" << catName << "\tstate=" << catState << "=" << catNum;
   }
   oocxcoutD(static_cast<TObject *>(nullptr), InputArguments) << "-- End of category tags --" << std::endl;

   // Step 2: Replace all category tags
   for (const auto &catState : categoryStates) {
      replaceAll(formula, catState.first, std::to_string(catState.second));
   }

   oocxcoutD(static_cast<TObject *>(nullptr), InputArguments)
      << "Preprocessing formula step 2: replace category tags\n\t" << formula << std::endl;

   // Step 3: Convert `@i`-style references to `x[i]`
   convertArobaseReferences(formula);

   oocxcoutD(static_cast<TObject *>(nullptr), InputArguments)
      << "Preprocessing formula step 3: replace '@'-references\n\t" << formula << std::endl;

   // Step 4: Replace all named references with "x[i]"-style
   replaceVarNamesWithIndexStyle(formula, varList);

   oocxcoutD(static_cast<TObject *>(nullptr), InputArguments) << "Final formula:\n\t" << formula << std::endl;

   return formula;
}

namespace {

/// Prune the variables that a processed formula doesn't reference as `x[i]`,
/// adding the used subset of `varList` to `actualVars` and returning the
/// formula with the `x[i]` indices remapped to the pruned list. The remapping
/// keeps the persisted pair (formula string, dependents) self-consistent, see
/// https://github.com/root-project/root/issues/21371. Out-of-range references
/// are kept as they are; they are reported when the evaluator is created.
std::string pruneAndReindexFormula(std::string const &processedFormula, RooArgList const &varList,
                                   RooArgList &actualVars)
{
   static const std::regex ordinalRegex("\\bx\\[([0-9]+)\\]");

   // First pass: find out which variables are actually referenced as `x[i]`.
   std::vector<bool> varIsUsed(varList.size());
   for (auto matchIt = sregex_iterator(processedFormula.begin(), processedFormula.end(), ordinalRegex);
        matchIt != sregex_iterator(); ++matchIt) {
      const std::size_t i = std::stoi((*matchIt)[1].str());
      if (i < varIsUsed.size()) {
         varIsUsed[i] = true;
      }
   }

   // Map each original index to its position among the used variables; pruned
   // entries get -1 and are never looked up (they don't appear in the formula).
   std::vector<int> newIndex(varList.size(), -1);
   for (std::size_t i = 0; i < varList.size(); ++i) {
      if (varIsUsed[i]) {
         newIndex[i] = actualVars.size();
         actualVars.add(varList[i]);
      }
   }

   // Second pass: rewrite every x[old] to x[newIndex[old]].
   std::string result;
   result.reserve(processedFormula.size());
   std::size_t lastPos = 0;
   for (auto matchIt = sregex_iterator(processedFormula.begin(), processedFormula.end(), ordinalRegex);
        matchIt != sregex_iterator(); ++matchIt) {
      std::smatch match = *matchIt;
      result.append(processedFormula, lastPos, match.position() - lastPos);
      const std::size_t oldIdx = std::stoi(match[1].str());
      result += oldIdx < newIndex.size() ? "x[" + std::to_string(newIndex[oldIdx]) + "]" : match[0].str();
      lastPos = match.position() + match.length();
   }
   result.append(processedFormula, lastPos, std::string::npos);

   return result;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
/// Reconstruct a user-facing formula string by replacing the index
/// placeholders in the internal representation with the variable names, or
/// with `fixedReplacement` if given.
std::string
RooFormulaUtils::reconstructFormula(std::string internalRepr, RooArgList const &args, const char *fixedReplacement)
{
   const auto nArgs = args.size();
   for (unsigned int i = 0; i < nArgs; ++i) {
      std::stringstream regexStr;
      regexStr << "x\\[" << i << "\\]|@" << i;
      std::regex regex(regexStr.str());

      std::string replacement = fixedReplacement ? fixedReplacement : std::string("[") + args[i].GetName() + "]";
      internalRepr = std::regex_replace(internalRepr, regex, replacement);
   }

   return internalRepr;
}

////////////////////////////////////////////////////////////////////////////////
/// Create the evaluation engine for a processed formula, checking that the
/// formula compiles and also fulfills the assumptions. Throws on failure,
/// with the original formula string appearing in the error messages.
///
/// First, the JIT-free expression parser is tried (unless disabled via
/// ROOFIT_FORMULA_BACKEND=tformula). On any unsupported construct it silently
/// falls back to the TFormula backend, so genuinely invalid formulas produce
/// exactly the same errors as before.
/// \param[in] name Name of the calling object, used to name the engine and in error messages.
/// \param[in] processedFormula The formula string in the normalized `x[i]` dialect,
/// with `i` referring to the position in `varList`.
/// \param[in] origFormula The original formula string as given by the user, used in error messages.
/// \param[in] varList List of variables to be passed to the formula.
std::unique_ptr<RooFormulaEvaluator>
RooFormulaUtils::makeEvaluator(std::string const &name, std::string const &processedFormula,
                               std::string const &origFormula, RooArgList const &varList)
{
   oocxcoutD(static_cast<TObject *>(nullptr), InputArguments)
      << "RooFormula '" << name << "' will be compiled as "
      << "\n\t" << processedFormula << "\n  and used as"
      << "\n\t" << reconstructFormula(processedFormula, varList) << "\n  with the parameters " << varList << std::endl;

   const FormulaBackend backend = formulaBackend();

   if (backend != FormulaBackend::TFormulaOnly) {
      std::string parseError;
      if (auto program = RooFormulaParser::compile(processedFormula, varList.size(), &parseError)) {
         return std::make_unique<RooExprEvaluator>(std::move(program));
      }
      if (backend == FormulaBackend::AstOnly) {
         std::stringstream msg;
         msg << "RooFormula '" << name << "' could not be compiled by the RooFit expression parser (" << parseError
             << "), and ROOFIT_FORMULA_BACKEND=ast disables the TFormula fallback."
             << "\nInput:\n\t" << origFormula << "\nProcessed formula:\n\t" << processedFormula << std::endl;
         oocoutF(static_cast<TObject *>(nullptr), InputArguments) << msg.str();
         throw std::runtime_error(msg.str());
      }
      // Report the silent fallback once per process at INFO level (further
      // fallbacks are only visible on the debug stream, to avoid spamming).
      static std::once_flag fallbackNoticeFlag;
      std::call_once(fallbackNoticeFlag, [&] {
         oocoutI(static_cast<TObject *>(nullptr), InputArguments)
            << "RooFormula '" << name
            << "': expression not supported by the RooFit formula "
               "parser ("
            << parseError
            << "), falling back to the TFormula (cling JIT) backend. This notice is only "
               "printed once; see the ROOFIT_FORMULA_BACKEND environment variable and the InputArguments debug "
               "stream for details."
            << std::endl;
      });
      oocxcoutD(static_cast<TObject *>(nullptr), InputArguments)
         << "RooFormula '" << name << "': expression not supported by the RooFit expression parser (" << parseError
         << "), falling back to TFormula" << std::endl;
   }

   return std::make_unique<RooTFormulaEvaluator>(name.c_str(), processedFormula, origFormula, varList);
}

////////////////////////////////////////////////////////////////////////////////
/// Create the evaluation engine for an unprocessed formula expression, e.g. a
/// cut expression on a dataset, with `x[i]` in the engine referring to
/// `varList[i]`. Unused variables are not pruned. Throws if the expression is
/// invalid.
std::unique_ptr<RooFormulaEvaluator>
RooFormulaUtils::makeFormulaEvaluator(std::string const &name, std::string const &expression, RooArgList const &varList)
{
   return makeEvaluator(name, processFormula(expression, varList, name), expression, varList);
}

////////////////////////////////////////////////////////////////////////////////
/// Implementation of the formula constructors of RooFormulaVar and
/// RooGenericPdf: compile the expression currently held in `formExpr` and
/// initialize the owner's state, pruning the variables that the expression
/// doesn't use. Throws if the expression is invalid.
void RooFormulaUtils::initFormula(std::unique_ptr<RooFormulaEvaluator> &evaluator, TString &formExpr,
                                  RooAbsCollection &actualVars, RooArgList const &dependents, const char *name)
{
   const std::string processed = processFormula(formExpr.Data(), dependents, name);
   RooArgList usedVars;
   const std::string pruned = pruneAndReindexFormula(processed, dependents, usedVars);
   evaluator = makeEvaluator(name, pruned, formExpr.Data(), usedVars);
   actualVars.add(usedVars);
   formExpr = pruned.c_str();
}

////////////////////////////////////////////////////////////////////////////////
/// Return an owner's formula evaluation engine, creating it on the fly if it
/// doesn't exist yet (i.e. after being read from file). The expression is
/// normalized to the `x[i]` dialect in the process, as old files may store it
/// with name or ordinal references. Throws if the formula is invalid.
RooFormulaEvaluator &RooFormulaUtils::ensureEvaluator(std::unique_ptr<RooFormulaEvaluator> &evaluator,
                                                      TString &formExpr, RooArgList const &actualVars,
                                                      const char *name)
{
   if (!evaluator) {
      std::string processed = processFormula(formExpr.Data(), actualVars, name);
      evaluator = makeEvaluator(name, processed, formExpr.Data(), actualVars);
      formExpr = processed.c_str();
   }
   return *evaluator;
}

////////////////////////////////////////////////////////////////////////////////
/// Clone a formula evaluation engine, renaming the copy after the
/// possibly-different name of the new owner.
std::unique_ptr<RooFormulaEvaluator> RooFormulaUtils::cloneEvaluator(RooFormulaEvaluator const &other,
                                                                     const char *newName)
{
   std::unique_ptr<RooFormulaEvaluator> out = other.clone();
   out->setName(newName);
   return out;
}

////////////////////////////////////////////////////////////////////////////////
/// Evaluate a formula for the current values of the variables: all variables
/// are evaluated given the normalisation set, and then the formula is
/// evaluated with `x[i]` taking the value of the i-th variable.
double
RooFormulaUtils::evalFormula(RooFormulaEvaluator const &evaluator, RooAbsCollection const &vars, RooArgSet const *nset)
{
   std::vector<double> pars;
   pars.reserve(vars.size());
   for (RooAbsArg const *arg : vars) {
      if (arg->isCategory()) {
         auto const &cat = static_cast<RooAbsCategory const &>(*arg);
         pars.push_back(cat.getCurrentIndex());
      } else {
         auto const &real = static_cast<RooAbsReal const &>(*arg);
         pars.push_back(real.getVal(nset));
      }
   }

   return evaluator.eval(pars.data());
}

////////////////////////////////////////////////////////////////////////////////
/// Evaluate a formula for a batch of input values from the evaluation context,
/// with `x[i]` taking the values of the i-th variable in `actualVars`.
///
/// If every input is a single value (e.g. a formula of parameters only, the
/// HistFactory expression-NormFactor shape), the formula is evaluated once and
/// the result is broadcast. Otherwise, formulas on the JIT-free expression
/// backend are evaluated with RooBatchCompute::computeExprProgram(), which
/// applies one instruction across a chunk of RooBatchCompute::bufferSize
/// events at a time in vectorizable elementwise loops. When ROOT is built with
/// VDT, that path evaluates exp/log/sin/cos with the same fast vectorizable
/// implementations as the RooBatchCompute pdf kernels, so batch results can
/// differ from per-event scalar evaluation within the usual RooBatchCompute
/// batch-vs-scalar tolerance (relative ~5e-14, see the vectorisedPDFs tests),
/// and, because those approximations cover only their normal argument range,
/// the special values are not reproduced either (fast_log() of a non-positive
/// number returns a finite garbage value instead of -Inf or NaN). Without VDT
/// the batch results are bitwise identical to scalar evaluation. The TFormula
/// fallback backend evaluates with a scalar per-event loop as before.
void RooFormulaUtils::doEvalFormula(RooFormulaEvaluator const &evaluator, RooArgList const &actualVars,
                                    RooFit::EvalContext &ctx)
{
   std::span<double> output = ctx.output();

   // Every x[i] has an input span, because the evaluation engine refers to
   // the (pruned) list of actual dependents directly.
   const std::size_t nPars = actualVars.size();
   // Note: emplace_back() instead of assignment into a pre-sized vector,
   // because the custom std::span backport for C++ < 20 in ROOT/span.hxx is
   // not move-assignable.
   std::vector<std::span<const double>> inputSpans;
   inputSpans.reserve(nPars);
   bool allScalar = true;
   for (std::size_t i = 0; i < nPars; ++i) {
      inputSpans.emplace_back(ctx.at(static_cast<const RooAbsReal *>(&actualVars[i])));
      allScalar &= inputSpans.back().size() <= 1;
   }

   // All inputs are single values: evaluate once and broadcast.
   if (allScalar) {
      std::vector<double> pars(nPars);
      for (std::size_t j = 0; j < nPars; j++) {
         if (!inputSpans[j].empty()) {
            pars[j] = inputSpans[j][0];
         }
      }
      std::fill(output.begin(), output.end(), evaluator.eval(pars.data()));
      return;
   }

   // Chunked, vectorized evaluation of JIT-free expression programs.
   if (auto *expr = dynamic_cast<RooExprEvaluator const *>(&evaluator)) {
      if (expr->stackDepth() <= RooBatchCompute::maxExprProgramStackDepth) {
         // Load the RooBatchCompute CPU dispatch if no RooFit::Evaluator has
         // done so yet: doEvalFormula() can also be called directly, and the
         // dispatch pointer is null until the library is loaded (a cheap
         // no-op once initialized).
         RooBatchCompute::initCPU();
         RooBatchCompute::computeExprProgram({}, expr->code(), expr->stackDepth(), output,
                                             {inputSpans.data(), inputSpans.size()});
         return;
      }
   }

   // Scalar per-event loop: the TFormula fallback backend, and expression
   // programs too deep for the vector interpreter's fixed-size chunk stack.
   std::vector<double> pars(nPars);
   for (std::size_t i = 0; i < output.size(); ++i) {
      for (std::size_t j = 0; j < nPars; ++j) {
         pars[j] = inputSpans[j].size() > 1 ? inputSpans[j][i] : inputSpans[j][0];
      }
      output[i] = evaluator.eval(pars.data());
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Print info about a compiled formula to the given stream.
void RooFormulaUtils::printFormula(std::ostream &os, TString indent, std::string const &formula,
                                   RooArgList const &actualVars)
{
   os << indent << "--- RooFormula ---" << std::endl;
   os << indent << " Formula:        '" << formula << "'" << std::endl;
   os << indent << " Interpretation: '" << reconstructFormula(formula, actualVars) << "'" << std::endl;
   indent.Append("  ");
   os << indent << "Servers: " << actualVars << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/// Deep-clone a map of user-defined binnings.
RooFormulaUtils::BinningMap RooFormulaUtils::cloneBinnings(BinningMap const &binnings)
{
   BinningMap out;
   for (auto const &item : binnings) {
      out[item.first] = std::unique_ptr<RooAbsBinning>{item.second->clone()};
   }
   return out;
}

////////////////////////////////////////////////////////////////////////////////
/// Declare a binning in which `caller` is piecewise constant (flat); see
/// RooGenericPdf::setBinning() for details.
void RooFormulaUtils::setBinning(BinningMap &binnings, RooAbsReal const &caller, RooArgList const &actualVars,
                                 const char *formExpr, RooAbsRealLValue const &obs, RooAbsBinning const &binning,
                                 bool checkFlatness)
{
   // Match the observable to a formula variable by name, so that a same-named
   // stand-in for the actual server is accepted too.
   const int idx = actualVars.index(obs.GetName());
   if (idx < 0) {
      oocoutE(&caller, InputArguments) << caller.ClassName() << "::setBinning(" << caller.GetName()
                                       << ") the observable " << obs.GetName()
                                       << " is not one of the formula variables, nothing done." << std::endl;
      return;
   }

   if (checkFlatness) {
      // Sample the function by varying the actual formula variable (the server),
      // which may be a different object than `obs` if `obs` is just a same-named
      // stand-in: the function's value depends on the server, not on `obs`.
      if (auto *serverObs = dynamic_cast<RooAbsRealLValue *>(actualVars.at(idx))) {
         std::span<const double> boundaries{binning.array(), static_cast<std::size_t>(binning.numBoundaries())};
         if (!RooHelpers::isFunctionFlatInBins(caller, *serverObs, boundaries)) {
            oocoutE(&caller, InputArguments)
               << caller.ClassName() << "::setBinning(" << caller.GetName() << ") the expression \"" << formExpr
               << "\" is not flat within the given bins of " << obs.GetName()
               << ". The binning is not set. Pass checkFlatness=false to override this check." << std::endl;
            return;
         }
      }
   }

   // Key the binning by the observable's index in the formula variables (not
   // its name), so that it survives a renaming of the variable or a server
   // redirection.
   binnings[idx] = std::unique_ptr<RooAbsBinning>{binning.clone()};
}

////////////////////////////////////////////////////////////////////////////////
/// Return the binning declared with setBinning() for observable `obs` (which
/// is matched to a formula variable by name), or nullptr.
const RooAbsBinning *
RooFormulaUtils::getBinning(BinningMap const &binnings, RooArgList const &actualVars, RooAbsRealLValue const &obs)
{
   auto found = binnings.find(actualVars.index(obs.GetName()));
   return found != binnings.end() ? found->second.get() : nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Return true if a binning was declared for every observable in the
/// integration set `obs`.
bool RooFormulaUtils::isBinnedDistribution(BinningMap const &binnings, RooArgList const &actualVars,
                                           RooArgSet const &obs)
{
   if (obs.empty() || binnings.empty()) {
      return false;
   }
   for (RooAbsArg *o : obs) {
      const int idx = actualVars.index(o->GetName());
      // Observables that are not formula variables are ones the caller does
      // not depend on: the function is constant (hence trivially binned) in
      // them, so they must be ignored here. This matches the convention that
      // composite functions like RooProduct rely on, where each component's
      // isBinnedDistribution() is queried with the full observable set.
      if (idx < 0) {
         continue;
      }
      if (binnings.find(idx) == binnings.end()) {
         return false;
      }
   }
   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the boundaries of the declared binning that fall within [xlo, xhi],
/// or a null pointer if no binning was declared for this observable.
std::list<double> *RooFormulaUtils::binBoundaries(BinningMap const &binnings, RooArgList const &actualVars,
                                                  RooAbsRealLValue const &obs, double xlo, double xhi)
{
   auto found = binnings.find(actualVars.index(obs.GetName()));
   if (found == binnings.end()) {
      return nullptr;
   }
   const RooAbsBinning &binning = *found->second;
   auto hint = new std::list<double>;
   for (int i = 0; i < binning.numBoundaries(); ++i) {
      const double boundary = binning.array()[i];
      if (boundary >= xlo && boundary <= xhi) {
         hint->push_back(boundary);
      }
   }
   return hint;
}

////////////////////////////////////////////////////////////////////////////////
/// Return sampling hints that draw the piecewise-flat shape exactly, or a
/// null pointer if no binning was declared for this observable.
std::list<double> *RooFormulaUtils::plotSamplingHint(BinningMap const &binnings, RooArgList const &actualVars,
                                                     RooAbsRealLValue const &obs, double xlo, double xhi)
{
   const RooAbsBinning *binning = getBinning(binnings, actualVars, obs);
   if (!binning) {
      return nullptr;
   }
   return RooCurve::plotSamplingHintForBinBoundaries(
      {binning->array(), static_cast<std::size_t>(binning->numBoundaries())}, xlo, xhi);
}

/// \endcond
