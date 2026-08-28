/// \cond ROOFIT_INTERNAL

/*
 * Project: RooFit
 *
 * Copyright (c) 2026, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

/**
 * Recursive-descent parser that compiles a processed RooFormula expression
 * string into the postfix instruction sequence of RooExprEvaluator.
 *
 * The contract is: either the compiled program evaluates *bitwise identically*
 * to what TFormula/cling computes for the same string, or compile() fails and
 * the caller falls back to the TFormula backend. Consequently:
 *
 *  - The tokenizer owns all character-level concerns (numbers, multi-character
 *    operators, `x[i]` variables, `::`-qualified names); the parser only deals
 *    in tokens.
 *  - Operator precedence matches C++, which is what cling compiled. The one
 *    deliberate dialect difference is `^` (and `**`), which TFormula rewrites
 *    to pow()/TMath::Sq() *before* cling sees the string: it is
 *    right-associative exponentiation binding tighter than unary minus, whose
 *    right-hand side may carry one leading sign (see
 *    TFormula::HandleExponentiation).
 *  - cling's expression typing is tracked as double/int/bool: integer
 *    division like `1/2` or `(x>0)/2` truncates in cling, so such expressions
 *    are not supported here and fall back. Int-typed constant subexpressions
 *    are folded in int64 at parse time, and any intermediate leaving the
 *    int32 range falls back (cling's int arithmetic would wrap around); an
 *    integer literal too large for int32 (which is long or unsigned in C++,
 *    not int) falls back as well. min/max with mixed argument types
 *    (int/double or bool/int) does not compile in cling at all and is
 *    rejected, and sign()/TMath::Sign with a bool-typed first argument
 *    resolves to the generic template returning bool (not copysign) and is
 *    rejected too.
 *  - `%` on doubles does not compile in cling, so TFormula formulas using it
 *    are invalid today; it is not part of this grammar either.
 *  - Several textual TFormula constructs that are invalid or surprising today
 *    are kept out of the dialect so that they keep behaving as before (see
 *    the FallbackTriggers test): the `++` linear-combination separator, runs
 *    of three or more `-`, bare chained comparisons (cling compiles with
 *    -Wparentheses as an error), and `^` with a sign on a parenthesized
 *    exponent (TFormula's rewrite distributes the sign into the group).
 *  - `&&` and `||` do not short-circuit and `?:` evaluates both branches, so
 *    that the scalar and a future vectorized path behave identically. The
 *    selected/combined values are unchanged; this is only observable if
 *    floating-point exceptions are trapped.
 */

#include "RooFormulaParser.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace {

using Op = RooExprEvaluator::Op;
using Instr = RooExprEvaluator::Instr;
using Program = RooExprEvaluator::Program;

enum class Tok : std::uint8_t {
   Number,
   Var,
   Ident,
   LParen,
   RParen,
   Comma,
   Plus,
   Minus,
   Star,
   Slash,
   Caret,
   Lt,
   Le,
   Gt,
   Ge,
   Eq,
   Ne,
   AndAnd,
   OrOr,
   Not,
   Question,
   Colon,
   End
};

struct Token {
   Tok kind = Tok::End;
   double value = 0.0;         ///< Tok::Number
   long long intValue = 0;     ///< Tok::Number with isInt: the exact integer value
   bool isInt = false;         ///< Tok::Number: literal has integer type in C++
   std::uint32_t varIndex = 0; ///< Tok::Var
   std::string_view text;      ///< Tok::Ident / Tok::Number: raw spelling
};

bool isIdentStart(char c)
{
   return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
}

bool isIdentChar(char c)
{
   return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

class Tokenizer {
public:
   Tokenizer(std::string const &s, std::string &error) : _s{s}, _error{error} {}

   bool run(std::vector<Token> &out)
   {
      const std::size_t n = _s.size();
      std::size_t i = 0;
      while (i < n) {
         const char c = _s[i];
         if (std::isspace(static_cast<unsigned char>(c))) {
            ++i;
            continue;
         }
         if (std::isdigit(static_cast<unsigned char>(c)) ||
             (c == '.' && i + 1 < n && std::isdigit(static_cast<unsigned char>(_s[i + 1])))) {
            if (!lexNumber(i, out))
               return false;
            continue;
         }
         if (isIdentStart(c)) {
            if (!lexIdentOrVar(i, out))
               return false;
            continue;
         }
         if (!lexOperator(i, out))
            return false;
      }
      out.push_back(Token{});
      out.back().kind = Tok::End;
      return true;
   }

private:
   bool fail(std::string msg)
   {
      _error = std::move(msg);
      return false;
   }

   /// Lex one numeric literal, reproducing the C++ literal type: a literal is
   /// `int` unless it has a decimal point or a (well-formed) exponent. Octal
   /// (`010`) and hex (`0x64`) integer literals are handled by strtoll with
   /// base 0, exactly as cling reads them. Literal values are parsed with
   /// strtod/strtoll, never by hand, so they are correctly rounded like the
   /// compiler's.
   bool lexNumber(std::size_t &i, std::vector<Token> &out)
   {
      const std::size_t n = _s.size();
      const char *begin = _s.c_str() + i;
      char *end = nullptr;

      bool isFloat = false;
      const bool isHex = _s[i] == '0' && i + 1 < n && (_s[i + 1] == 'x' || _s[i + 1] == 'X');
      if (!isHex) {
         std::size_t j = i;
         while (j < n && std::isdigit(static_cast<unsigned char>(_s[j])))
            ++j;
         if (j < n && _s[j] == '.') {
            isFloat = true;
         } else if (j < n && (_s[j] == 'e' || _s[j] == 'E')) {
            std::size_t k = j + 1;
            if (k < n && (_s[k] == '+' || _s[k] == '-'))
               ++k;
            if (k < n && std::isdigit(static_cast<unsigned char>(_s[k])))
               isFloat = true;
            // else: something like "1e" -- fails the trailing-character check below
         }
      }

      Token tok;
      tok.kind = Tok::Number;
      errno = 0;
      if (isFloat) {
         tok.value = std::strtod(begin, &end);
         tok.isInt = false;
      } else {
         const long long v = std::strtoll(begin, &end, 0);
         if (errno == ERANGE)
            return fail("integer literal out of range");
         // An integer literal too large for int32 does not have type int in
         // C++ (a decimal one becomes long, a hex/octal one unsigned int),
         // which is not the int typing tracked here. Fall back.
         if (v > std::numeric_limits<int>::max())
            return fail("integer literal does not fit in int");
         tok.value = static_cast<double>(v);
         tok.intValue = v;
         tok.isInt = true;
      }
      if (end == begin)
         return fail("invalid numeric literal");
      const std::size_t len = end - begin;
      // A hex literal whose final digit is e/E followed by '+' or '-' forms
      // one single (invalid) pp-number in C++: cling rejects "0x1e+2" rather
      // than computing 0x1e + 2. TFormula strips whitespace before compiling,
      // so "0x1e + 2" is equally invalid. Keep such formulas failing.
      if (isHex && (end[-1] == 'e' || end[-1] == 'E')) {
         std::size_t j = i + len;
         while (j < n && std::isspace(static_cast<unsigned char>(_s[j])))
            ++j;
         if (j < n && (_s[j] == '+' || _s[j] == '-'))
            return fail("hex literal followed by an exponent-like sign is invalid in C++");
      }
      // Reject trailing characters that would make this an invalid or
      // differently-typed literal in C++ (e.g. "1e", "08", "1.5f", "1.2.3").
      if (i + len < n) {
         const char next = _s[i + len];
         if (isIdentChar(next) || next == '.')
            return fail("invalid numeric literal");
      }
      tok.text = std::string_view{begin, len};
      out.push_back(tok);
      i += len;
      return true;
   }

   /// Lex an identifier (including `::`-qualified names like TMath::Erf as one
   /// token), or an `x[i]` variable reference. Only the exact shape
   /// `x[<digits>]` is a variable; RooFormula::processFormula() guarantees it.
   bool lexIdentOrVar(std::size_t &i, std::vector<Token> &out)
   {
      const std::size_t n = _s.size();
      const std::size_t start = i;
      while (i < n && isIdentChar(_s[i]))
         ++i;
      // absorb `::name` qualifications (a single ':' is a ternary colon)
      while (i + 2 < n && _s[i] == ':' && _s[i + 1] == ':' && isIdentStart(_s[i + 2])) {
         i += 2;
         while (i < n && isIdentChar(_s[i]))
            ++i;
      }
      std::string_view name{_s.c_str() + start, i - start};

      if (name == "x" && i < n && _s[i] == '[') {
         std::size_t j = i + 1;
         std::uint32_t index = 0;
         if (j >= n || !std::isdigit(static_cast<unsigned char>(_s[j])))
            return fail("malformed variable reference");
         while (j < n && std::isdigit(static_cast<unsigned char>(_s[j]))) {
            index = index * 10 + (_s[j] - '0');
            if (index > 1000000)
               return fail("variable index out of range");
            ++j;
         }
         if (j >= n || _s[j] != ']')
            return fail("malformed variable reference");
         Token tok;
         tok.kind = Tok::Var;
         tok.varIndex = index;
         out.push_back(tok);
         i = j + 1;
         return true;
      }

      Token tok;
      tok.kind = Tok::Ident;
      tok.text = name;
      out.push_back(tok);
      return true;
   }

   /// Lex one operator or punctuation token, with maximal munch for the
   /// two-character operators. Anything not in the supported set (notably `%`,
   /// single `&`, `|`, `=`, brackets outside `x[i]`, string literals) fails.
   bool lexOperator(std::size_t &i, std::vector<Token> &out)
   {
      const std::size_t n = _s.size();
      const char c = _s[i];
      const char c2 = i + 1 < n ? _s[i + 1] : '\0';
      Tok kind;
      std::size_t len = 1;
      switch (c) {
      case '+':
         // A textually adjacent `++` is TFormula's linear-combination
         // separator (TLinearFitter syntax with one fit parameter per part),
         // not an addition. `+ +` with whitespace is ordinary addition.
         if (c2 == '+')
            return fail("'++' is TFormula's linear-combination separator");
         kind = Tok::Plus;
         break;
      case '-': {
         // TFormula rewrites a double negation `--` (also with whitespace in
         // between) so that cling accepts it, but a run of three or more `-`
         // survives as a `--` pre-decrement in the generated code, which does
         // not compile. Keep such formulas failing (fall back).
         std::size_t j = i + 1;
         int run = 1;
         while (j < n && (_s[j] == '-' || std::isspace(static_cast<unsigned char>(_s[j])))) {
            if (_s[j] == '-')
               ++run;
            ++j;
         }
         if (run >= 3)
            return fail("three or more consecutive '-' are invalid in TFormula");
         kind = Tok::Minus;
         break;
      }
      case '*':
         if (c2 == '*') { // TFormula rewrites `**` to `^`
            kind = Tok::Caret;
            len = 2;
         } else {
            kind = Tok::Star;
         }
         break;
      case '/': kind = Tok::Slash; break;
      case '^': kind = Tok::Caret; break;
      case '(': kind = Tok::LParen; break;
      case ')': kind = Tok::RParen; break;
      case ',': kind = Tok::Comma; break;
      case '?': kind = Tok::Question; break;
      case ':': kind = Tok::Colon; break;
      case '<':
         if (c2 == '=') {
            kind = Tok::Le;
            len = 2;
         } else {
            kind = Tok::Lt;
         }
         break;
      case '>':
         if (c2 == '=') {
            kind = Tok::Ge;
            len = 2;
         } else {
            kind = Tok::Gt;
         }
         break;
      case '!':
         if (c2 == '=') {
            kind = Tok::Ne;
            len = 2;
         } else {
            kind = Tok::Not;
         }
         break;
      case '=':
         if (c2 != '=')
            return fail("unsupported operator '='");
         kind = Tok::Eq;
         len = 2;
         break;
      case '&':
         if (c2 != '&')
            return fail("unsupported operator '&'");
         kind = Tok::AndAnd;
         len = 2;
         break;
      case '|':
         if (c2 != '|')
            return fail("unsupported operator '|'");
         kind = Tok::OrOr;
         len = 2;
         break;
      default: return fail(std::string{"unsupported character '"} + c + "'");
      }
      Token tok;
      tok.kind = kind;
      out.push_back(tok);
      i += len;
      return true;
   }

   std::string const &_s;
   std::string &_error;
};

/// Binary operator precedence, mirroring C++ exactly (what cling compiled).
/// Loosest binds first; `?:` sits below level 1 (handled in parseTernary) and
/// unary +/-/! above level 6 (handled in parseUnary), with `^`/`**`
/// exponentiation tighter still (handled in parsePower). All these operators
/// are left-associative. Bitwise `& | ^`(C++ meaning) are not part of the
/// dialect: `^` is exponentiation and single `&`/`|` fail to tokenize.
struct BinOpInfo {
   Tok tok;
   int prec;
   Op op;
};

// clang-format off
constexpr BinOpInfo gBinaryOps[] = {
   {Tok::OrOr,   1, Op::Or},
   {Tok::AndAnd, 2, Op::And},
   {Tok::Eq,     3, Op::EQ},
   {Tok::Ne,     3, Op::NE},
   {Tok::Lt,     4, Op::LT},
   {Tok::Le,     4, Op::LE},
   {Tok::Gt,     4, Op::GT},
   {Tok::Ge,     4, Op::GE},
   {Tok::Plus,   5, Op::Add},
   {Tok::Minus,  5, Op::Sub},
   {Tok::Star,   6, Op::Mul},
   {Tok::Slash,  6, Op::Div},
};
// clang-format on

BinOpInfo const *findBinOp(Tok kind)
{
   for (auto const &info : gBinaryOps) {
      if (info.tok == kind)
         return &info;
   }
   return nullptr;
}

class Parser {
public:
   Parser(std::vector<Token> const &tokens, unsigned int nVars, std::string &error)
      : _tokens{tokens}, _nVars{nVars}, _error{error}
   {
   }

   /// C++ typing of a subexpression (double vs int vs bool), tracked to
   /// reject constructs whose cling semantics double arithmetic cannot
   /// reproduce: truncating integer division, min/max with mixed argument
   /// types, and the bool-typed constructs with non-arithmetic behavior.
   struct ExprInfo {
      enum class Type : std::uint8_t {
         Double,
         Int,
         Bool
      };
      Type type = Type::Double;
      /// Int-typed constant subexpressions are folded in 64-bit arithmetic at
      /// parse time: cling evaluated them in (wrapping) int32 arithmetic, so
      /// any int-typed constant intermediate leaving the int32 range makes
      /// this evaluator's double arithmetic diverge ("100000*100000" is
      /// 1410065408 in cling, not 1e10) and must fall back. Non-constant
      /// int-typed intermediates (reachable through an int(x) cast or a
      /// promoted bool subexpression) are not tracked.
      bool isIntConst = false;
      long long intConstValue = 0;
      /// Whether the C++ type is an integral type (int or bool).
      bool isIntegral() const { return type != Type::Double; }
   };

   bool run(Program &prog)
   {
      _used.assign(_nVars, false);
      ExprInfo info;
      if (!parseTernary(info))
         return false;
      if (peek().kind != Tok::End)
         return fail("unexpected token after end of expression");

      // Compute the maximum evaluation stack depth. The switch is exhaustive
      // over ExprOp with no default case, so that adding an opcode without
      // extending the accounting is a compiler warning (-Wswitch); the
      // static_assert additionally breaks the build when the enum grows.
      static_assert(static_cast<int>(Op::Call4) == 23,
                    "ExprOp changed: update the stack-depth accounting switch in RooFormulaParser");
      int depth = 0;
      int maxDepth = 0;
      for (Instr const &ins : _code) {
         switch (ins.op) {
         case Op::Const:
         case Op::Var: ++depth; break;
         case Op::Neg:
         case Op::Not:
         case Op::Sq:
         case Op::IntNorm:
         case Op::Call1: break;
         case Op::Add:
         case Op::Sub:
         case Op::Mul:
         case Op::Div:
         case Op::LT:
         case Op::LE:
         case Op::GT:
         case Op::GE:
         case Op::EQ:
         case Op::NE:
         case Op::And:
         case Op::Or:
         case Op::Pow:
         case Op::Call2: --depth; break;
         case Op::Select:
         case Op::Call3: depth -= 2; break;
         case Op::Call4: depth -= 3; break;
         }
         maxDepth = std::max(maxDepth, depth);
      }
      if (maxDepth > static_cast<int>(RooExprEvaluator::kMaxStackDepth))
         return fail("expression too deep");

      prog.code = std::move(_code);
      prog.stackDepth = maxDepth;
      // Trim to the highest used index so that programs can be shared between
      // formulas with different (sufficiently long) variable lists.
      std::size_t lastUsed = 0;
      for (std::size_t i = 0; i < _used.size(); ++i) {
         if (_used[i])
            lastUsed = i + 1;
      }
      _used.resize(lastUsed);
      prog.usedVars = std::move(_used);
      return true;
   }

private:
   static constexpr int kMaxRecursionDepth = 128;

   struct DepthGuard {
      DepthGuard(int &d) : _d{d} { ++_d; }
      ~DepthGuard() { --_d; }
      int &_d;
   };

   static bool fitsInInt32(long long v)
   {
      return v >= std::numeric_limits<int>::min() && v <= std::numeric_limits<int>::max();
   }

   bool fail(std::string msg)
   {
      if (_error.empty())
         _error = std::move(msg);
      return false;
   }

   Token const &peek() const { return _tokens[_pos]; }
   Token const &next() { return _tokens[_pos++]; }

   void emit(Op op, std::uint32_t arg = 0, double konst = 0.0)
   {
      Instr ins;
      ins.op = op;
      ins.arg = arg;
      ins.konst = konst;
      _code.push_back(ins);
   }

   /// conditional-expression: right-associative, both branches always
   /// evaluated with the value of the active branch selected (Op::Select).
   bool parseTernary(ExprInfo &out)
   {
      DepthGuard guard{_depth};
      if (_depth > kMaxRecursionDepth)
         return fail("expression too deeply nested");
      if (!parseBinary(1, out))
         return false;
      if (peek().kind != Tok::Question)
         return true;
      next();
      ExprInfo left;
      ExprInfo right;
      if (!parseTernary(left))
         return false;
      if (peek().kind != Tok::Colon)
         return fail("expected ':' in conditional expression");
      next();
      if (!parseTernary(right))
         return false;
      emit(Op::Select);
      // The C++ type of `c ? a : b` is the common type of the branches.
      if (left.type == ExprInfo::Type::Double || right.type == ExprInfo::Type::Double) {
         out.type = ExprInfo::Type::Double;
      } else if (left.type == ExprInfo::Type::Bool && right.type == ExprInfo::Type::Bool) {
         out.type = ExprInfo::Type::Bool;
      } else {
         out.type = ExprInfo::Type::Int;
      }
      // Not a constant (the branch values themselves stay in int32 range).
      out.isIntConst = false;
      return true;
   }

   /// Precedence climbing over gBinaryOps.
   bool parseBinary(int minPrec, ExprInfo &out)
   {
      if (!parseUnary(out))
         return false;
      // Whether the expression accumulated so far is a bare (unparenthesized)
      // relational comparison: cling compiles TFormula code with clang's
      // -Wparentheses promoted to an error, so a chained comparison like
      // `a < b < c` is invalid in TFormula today (chained equality like
      // `a == b == c` is accepted, and parenthesized operands are fine).
      bool lhsIsBareRelational = false;
      while (true) {
         BinOpInfo const *info = findBinOp(peek().kind);
         if (!info || info->prec < minPrec)
            break;
         if (info->prec == 4 && lhsIsBareRelational)
            return fail("chained comparison is invalid in TFormula");
         lhsIsBareRelational = info->prec == 4;
         next();
         ExprInfo rhs;
         if (!parseBinary(info->prec + 1, rhs))
            return false;
         switch (info->op) {
         case Op::Add:
         case Op::Sub:
         case Op::Mul:
            // integral operands (bool promotes to int) give an int result
            if (out.isIntegral() && rhs.isIntegral()) {
               out.type = ExprInfo::Type::Int;
               // Fold int-typed constants in 64-bit arithmetic; leaving the
               // int32 range means cling's int arithmetic wrapped around,
               // which is not reproduced here. Fall back. (Operands are
               // within int32 range, so the int64 fold cannot overflow.)
               if (out.isIntConst && rhs.isIntConst) {
                  const long long l = out.intConstValue;
                  const long long r = rhs.intConstValue;
                  out.intConstValue = info->op == Op::Add ? l + r : info->op == Op::Sub ? l - r : l * r;
                  if (!fitsInInt32(out.intConstValue))
                     return fail("integer constant expression overflows int in cling");
               } else {
                  out.isIntConst = false;
               }
            } else {
               out.type = ExprInfo::Type::Double;
               out.isIntConst = false;
            }
            emit(info->op);
            // cling would compute `int * int` in integer arithmetic, where
            // e.g. (-1) * 0 is +0 and not the -0.0 of double arithmetic.
            if (out.type == ExprInfo::Type::Int && info->op == Op::Mul)
               emit(Op::IntNorm);
            break;
         case Op::Div:
            // `1/2` is a truncating integer division in cling. Not supported;
            // fall back to TFormula so the behavior is unchanged.
            if (out.isIntegral() && rhs.isIntegral())
               return fail("integer division has truncating semantics in TFormula/cling");
            out.type = ExprInfo::Type::Double;
            out.isIntConst = false;
            emit(info->op);
            break;
         default:
            // comparisons and logical operators: C++ result type is bool
            out.type = ExprInfo::Type::Bool;
            out.isIntConst = false;
            emit(info->op);
            break;
         }
      }
      return true;
   }

   /// unary-expression: prefix `+`, `-`, `!`.
   bool parseUnary(ExprInfo &out)
   {
      DepthGuard guard{_depth};
      if (_depth > kMaxRecursionDepth)
         return fail("expression too deeply nested");
      switch (peek().kind) {
      case Tok::Plus:
         next();
         if (!parseUnary(out))
            return false;
         // unary plus: no-op on the value, but bool promotes to int
         if (out.type == ExprInfo::Type::Bool)
            out.type = ExprInfo::Type::Int;
         return true;
      case Tok::Minus:
         next();
         if (!parseUnary(out))
            return false;
         emit(Op::Neg);
         if (out.isIntegral()) {
            emit(Op::IntNorm); // cling: -(int)0 is +0, not -0.0
            out.type = ExprInfo::Type::Int;
            if (out.isIntConst) {
               out.intConstValue = -out.intConstValue;
               if (!fitsInInt32(out.intConstValue)) // -(INT_MIN) overflows in cling
                  return fail("integer constant expression overflows int in cling");
            }
         }
         return true;
      case Tok::Not:
         next();
         if (!parseUnary(out))
            return false;
         emit(Op::Not);
         out.type = ExprInfo::Type::Bool;
         out.isIntConst = false;
         return true;
      default: return parsePower(out);
      }
   }

   /// Exponentiation via `^` (or `**`), reproducing TFormula's textual
   /// `a^b` -> `pow(a,b)` rewrite (TFormula::HandleExponentiation):
   /// right-associative, binding tighter than `*`, `/` and unary minus, with
   /// TFormula's special case `expr^2` -> `TMath::Sq(expr)` when the exponent
   /// is spelled exactly `2`.
   bool parsePower(ExprInfo &out)
   {
      const std::size_t startTok = _pos;
      if (!parsePrimary(out))
         return false;
      if (peek().kind != Tok::Caret)
         return true;
      // TFormula's textual operand scan runs through `,` and `:` (it only
      // stops at operators and parentheses), so a `^` directly adjacent to a
      // function-argument or ternary boundary produces invalid code today
      // (e.g. `pow(x,2^3)` or `c?a:b^2` do not compile). Keep those failing.
      if (startTok > 0 && (_tokens[startTok - 1].kind == Tok::Comma || _tokens[startTok - 1].kind == Tok::Colon)) {
         return fail("'^' operand adjacent to ',' or ':' is invalid in TFormula");
      }
      next();
      const std::size_t codeSizeBeforeRhs = _code.size();
      ExprInfo rhs;
      bool rhsIsLiteralTwo = false;
      if (!parsePowerRhs(rhs, rhsIsLiteralTwo))
         return false;
      if (peek().kind == Tok::Comma || peek().kind == Tok::Colon)
         return fail("'^' operand adjacent to ',' or ':' is invalid in TFormula");
      if (rhsIsLiteralTwo) {
         // TMath::Sq is unary: drop the emitted `Const 2` again.
         _code.resize(codeSizeBeforeRhs);
         emit(Op::Sq);
      } else {
         emit(Op::Pow);
      }
      out.type = ExprInfo::Type::Double; // pow() and TMath::Sq(Double_t) return double
      out.isIntConst = false;
      return true;
   }

   /// The right-hand side of `^`: an optional single sign, then a
   /// power-expression (making `^` right-associative, `x^-2^3` = pow(x,-(2^3))).
   bool parsePowerRhs(ExprInfo &out, bool &isLiteralTwo)
   {
      DepthGuard guard{_depth};
      if (_depth > kMaxRecursionDepth)
         return fail("expression too deeply nested");
      bool haveSign = false;
      bool negate = false;
      if (peek().kind == Tok::Plus) {
         next();
         haveSign = true;
      } else if (peek().kind == Tok::Minus) {
         next();
         haveSign = true;
         negate = true;
      }
      // TFormula's textual rewrite pushes an explicit sign into a
      // parenthesized exponent group, onto only its first term: `x^-(a+b)`
      // compiles as pow(x,-(a)+b) today. Do not reproduce that; fall back.
      if (haveSign && peek().kind == Tok::LParen)
         return fail("'^' with a sign on a parenthesized exponent is broken in TFormula");
      const std::size_t operandStart = _pos;
      if (!parsePower(out))
         return false;
      // TFormula turns `a^b` into TMath::Sq(a) only when the exponent is the
      // literal token `2` (not `2.0`, not `(2)`, not `+2`).
      isLiteralTwo = !haveSign && _pos == operandStart + 1 && _tokens[operandStart].kind == Tok::Number &&
                     _tokens[operandStart].text == "2";
      if (negate) {
         emit(Op::Neg);
         if (out.isIntegral()) {
            emit(Op::IntNorm);
            out.type = ExprInfo::Type::Int;
            if (out.isIntConst) {
               out.intConstValue = -out.intConstValue;
               if (!fitsInInt32(out.intConstValue))
                  return fail("integer constant expression overflows int in cling");
            }
         }
      }
      return true;
   }

   /// primary-expression: literal, `x[i]`, parenthesized expression, or an
   /// allow-listed function call (zero-argument constant calls are folded).
   bool parsePrimary(ExprInfo &out)
   {
      switch (peek().kind) {
      case Tok::Number: {
         Token const &tok = next();
         emit(Op::Const, 0, tok.value);
         out.type = tok.isInt ? ExprInfo::Type::Int : ExprInfo::Type::Double;
         out.isIntConst = tok.isInt;
         out.intConstValue = tok.intValue;
         return true;
      }
      case Tok::Var: {
         Token const &tok = next();
         if (tok.varIndex >= _nVars)
            return fail("formula references x[" + std::to_string(tok.varIndex) + "] but fewer variables were provided");
         _used[tok.varIndex] = true;
         emit(Op::Var, tok.varIndex);
         out.type = ExprInfo::Type::Double;
         out.isIntConst = false;
         return true;
      }
      case Tok::LParen: {
         next();
         if (!parseTernary(out))
            return false;
         if (peek().kind != Tok::RParen)
            return fail("expected ')'");
         next();
         return true;
      }
      case Tok::Ident: return parseCall(out);
      default: return fail("expected an expression");
      }
   }

   bool parseCall(ExprInfo &out)
   {
      const std::string name{next().text};
      if (peek().kind != Tok::LParen)
         return fail("unknown identifier '" + name + "'");
      next();
      unsigned int nArgs = 0;
      ExprInfo argInfo[4];
      if (peek().kind == Tok::RParen) {
         next();
      } else {
         while (true) {
            if (nArgs == 4)
               return fail("too many arguments in call to '" + name + "'");
            if (!parseTernary(argInfo[nArgs]))
               return false;
            ++nArgs;
            if (peek().kind == Tok::Comma) {
               next();
               continue;
            }
            if (peek().kind == Tok::RParen) {
               next();
               break;
            }
            return fail("expected ',' or ')' in call to '" + name + "'");
         }
      }

      std::uint32_t index = 0;
      RooFormulaFunctions::Entry const *entry = RooFormulaFunctions::find(name, nArgs, index);
      if (!entry) {
         return fail("unsupported function '" + name + "' with " + std::to_string(nArgs) + " argument(s)");
      }

      using RooFormulaFunctions::TypeRule;
      using Type = ExprInfo::Type;
      switch (entry->rule) {
      case TypeRule::Double: out.type = Type::Double; break;
      case TypeRule::SameAsFirstArg:
         // abs(bool) resolves to abs(int) in cling: bool promotes to int
         out.type = argInfo[0].type == Type::Bool ? Type::Int : argInfo[0].type;
         break;
      case TypeRule::Int: out.type = Type::Int; break;
      case TypeRule::Bool: out.type = Type::Bool; break;
      case TypeRule::Sign:
         // With a bool first argument, cling resolves TMath::Sign to the
         // generic template returning bool: Sign(true, -1.) is +1 there, not
         // the -1 of copysign. Fall back rather than reproducing that.
         if (argInfo[0].type == Type::Bool)
            return fail("'" + name + "' with a bool-typed first argument is not copysign in cling");
         out.type = argInfo[0].type;
         break;
      case TypeRule::MinMax:
         // e.g. std::min(x, 3) with double x and int 3 (or a bool/int mix)
         // does not compile in cling, so such formulas are invalid in
         // TFormula today. Keep it so.
         if (argInfo[0].type != argInfo[1].type)
            return fail("'" + name + "' with mixed argument types is invalid in TFormula");
         out.type = argInfo[0].type;
         break;
      }
      out.isIntConst = false; // call results are not constant-folded

      switch (nArgs) {
      case 0:
         // Zero-argument calls are the TMath constants: fold to a literal.
         emit(Op::Const, 0, entry->fn0());
         break;
      case 1: emit(Op::Call1, index); break;
      case 2: emit(Op::Call2, index); break;
      case 3: emit(Op::Call3, index); break;
      case 4: emit(Op::Call4, index); break;
      }
      if (out.isIntegral())
         emit(Op::IntNorm); // integer-valued calls cannot yield -0.0 in cling
      return true;
   }

   std::vector<Token> const &_tokens;
   unsigned int _nVars = 0;
   std::string &_error;
   std::size_t _pos = 0;
   int _depth = 0;
   std::vector<Instr> _code;
   std::vector<bool> _used;
};

std::shared_ptr<const Program> parseImpl(std::string const &formula, unsigned int nVars, std::string &error)
{
   std::vector<Token> tokens;
   if (!Tokenizer{formula, error}.run(tokens))
      return nullptr;
   auto prog = std::make_shared<Program>();
   if (!Parser{tokens, nVars, error}.run(*prog))
      return nullptr;
   prog->formula = formula;
   return prog;
}

} // namespace

std::shared_ptr<const RooExprEvaluator::Program>
RooFormulaParser::compile(std::string const &processedFormula, unsigned int nVars, std::string *error)
{
   // Process-wide registry of compiled programs, keyed on the processed
   // formula string: identical formulas (e.g. thousands of structurally equal
   // HistFactory expressions) share one immutable instruction vector. Like
   // TFormula's gClingFunctions cache, the registry grows without bound over
   // the process lifetime; entries are small (the instruction vector).
   // Only successful parses are cached: formulas destined for the TFormula
   // fallback are re-parsed on each construction, which is cheap compared to
   // the JIT compilation that follows.
   static std::mutex mutex;
   static std::unordered_map<std::string, std::shared_ptr<const Program>> registry;

   std::string errorBuffer;
   std::string &err = error ? *error : errorBuffer;

   {
      std::lock_guard<std::mutex> lock{mutex};
      auto it = registry.find(processedFormula);
      if (it != registry.end()) {
         if (it->second->usedVars.size() <= nVars)
            return it->second;
         err = "formula references more variables than provided";
         return nullptr;
      }
   }

   auto prog = parseImpl(processedFormula, nVars, err);
   if (!prog)
      return nullptr;

   std::lock_guard<std::mutex> lock{mutex};
   // If another thread compiled the same formula concurrently, share its copy.
   return registry.emplace(processedFormula, std::move(prog)).first->second;
}

/// \endcond
