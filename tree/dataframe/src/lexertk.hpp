/*
 *****************************************************************
 *                 Simple C++ Lexer Toolkit Library              *
 *                                                               *
 * Author: Arash Partow (2001)                                   *
 * URL: http://www.partow.net/programming/lexertk/index.html     *
 *                                                               *
 * Copyright notice:                                             *
 * Free use of the Simple C++ Lexer Toolkit Library is permitted *
 * under the guidelines and in accordance with the MIT License.  *
 * http://www.opensource.org/licenses/MIT                        *
 *                                                               *
 *                                                               *
 * The lexer will tokenize input against the following BNF:      *
 *                                                               *
 * expression ::= term { +|- term }                              *
 * term       ::= (symbol | factor) {operator symbol | factor}   *
 * factor     ::= symbol | ( '(' {-} expression ')' )            *
 * symbol     ::= number | gensymb | string                      *
 * gensymb    ::= alphabet {alphabet | digit}                    *
 * string     ::= '"' {alphabet | digit | operator } '"'         *
 * operator   ::= * | / | % | ^ | < | > | <= | >= | << | >> !=   *
 * alphabet   ::= a | b | .. | z | A | B | .. | Z                *
 * digit      ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 |  7 | 8 | 9         *
 * sign       ::= + | -                                          *
 * edef       ::= e | E                                          *
 * decimal    ::= {digit} (digit [.] | [.] digit) {digit}        *
 * exponent   ::= edef [sign] digit {digit}                      *
 * real       ::= [sign] decimal [exponent]                      *
 * integer    ::= [sign] {digit}                                 *
 * number     ::= real | integer                                 *
 *                                                               *
 *                                                               *
 * Note: This lexer has been taken from the ExprTk Library.      *
 *                                                               *
 *****************************************************************
 */

#ifndef INCLUDE_LEXERTK_HPP
#define INCLUDE_LEXERTK_HPP

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <exception>
#include <limits>
#include <map>
#include <set>
#include <stack>
#include <stdexcept>
#include <string>

namespace lexertk {

namespace details {

inline bool is_whitespace(const char c)
{
   return (' ' == c) || ('\n' == c) || ('\r' == c) || ('\t' == c) || ('\b' == c) || ('\v' == c) || ('\f' == c);
}

inline bool is_operator_char(const char c)
{
   return ('+' == c) || ('-' == c) || ('*' == c) || ('/' == c) || ('^' == c) || ('<' == c) || ('>' == c) ||
          ('=' == c) || (',' == c) || ('!' == c) || ('(' == c) || (')' == c) || ('[' == c) || (']' == c) ||
          ('{' == c) || ('}' == c) || ('%' == c) || (':' == c) || ('?' == c) || ('&' == c) || ('|' == c) || (';' == c);
}

inline bool is_letter(const char c)
{
   return (('a' <= c) && (c <= 'z')) || (('A' <= c) && (c <= 'Z'));
}

inline bool is_digit(const char c)
{
   return ('0' <= c) && (c <= '9');
}

inline bool is_letter_or_digit(const char c)
{
   return is_letter(c) || is_digit(c);
}

inline bool is_left_bracket(const char c)
{
   return ('(' == c) || ('[' == c) || ('{' == c);
}

inline bool is_right_bracket(const char c)
{
   return (')' == c) || (']' == c) || ('}' == c);
}

inline bool is_bracket(const char c)
{
   return is_left_bracket(c) || is_right_bracket(c);
}

inline bool is_sign(const char c)
{
   return ('+' == c) || ('-' == c);
}

inline bool is_invalid(const char c)
{
   return !is_whitespace(c) && !is_operator_char(c) && !is_letter(c) && !is_digit(c) && ('.' != c) && ('_' != c) &&
          ('$' != c) && ('~' != c) && ('\'' != c);
}

inline bool imatch(const char c1, const char c2)
{
   return std::tolower(c1) == std::tolower(c2);
}

inline bool imatch(const std::string &s1, const std::string &s2)
{
   if (s1.size() == s2.size()) {
      for (std::size_t i = 0; i < s1.size(); ++i) {
         if (std::tolower(s1[i]) != std::tolower(s2[i])) {
            return false;
         }
      }

      return true;
   }

   return false;
}

struct ilesscompare {
   inline bool operator()(const std::string &s1, const std::string &s2) const
   {
      const std::size_t length = std::min(s1.size(), s2.size());

      for (std::size_t i = 0; i < length; ++i) {
         if (std::tolower(s1[i]) > std::tolower(s2[i]))
            return false;
         else if (std::tolower(s1[i]) < std::tolower(s2[i]))
            return true;
      }

      return s1.size() < s2.size();
   }
};

inline void cleanup_escapes(std::string &s)
{
   typedef std::string::iterator str_itr_t;

   str_itr_t itr1 = s.begin();
   str_itr_t itr2 = s.begin();
   str_itr_t end = s.end();

   std::size_t removal_count = 0;

   while (end != itr1) {
      if ('\\' == (*itr1)) {
         ++removal_count;

         if (end == ++itr1)
            break;
         else if ('\\' != (*itr1)) {
            switch (*itr1) {
            case 'n': (*itr1) = '\n'; break;
            case 'r': (*itr1) = '\r'; break;
            case 't': (*itr1) = '\t'; break;
            }

            continue;
         }
      }

      if (itr1 != itr2) {
         (*itr2) = (*itr1);
      }

      ++itr1;
      ++itr2;
   }

   s.resize(s.size() - removal_count);
}
} // namespace details

struct token {

   enum token_type {
      e_none = 0,
      e_error = 1,
      e_err_symbol = 2,
      e_err_number = 3,
      e_err_string = 4,
      e_err_sfunc = 5,
      e_eof = 6,
      e_number = 7,
      e_symbol = 8,
      e_string = 9,
      e_assign = 10,
      e_shr = 11,
      e_shl = 12,
      e_lte = 13,
      e_ne = 14,
      e_gte = 15,
      e_lt = '<',
      e_gt = '>',
      e_eq = '=',
      e_rbracket = ')',
      e_lbracket = '(',
      e_rsqrbracket = ']',
      e_lsqrbracket = '[',
      e_rcrlbracket = '}',
      e_lcrlbracket = '{',
      e_comma = ',',
      e_add = '+',
      e_sub = '-',
      e_div = '/',
      e_mul = '*',
      e_mod = '%',
      e_pow = '^',
      e_colon = ':'
   };

   token() : type(e_none), value(""), position(std::numeric_limits<std::size_t>::max()) {}

   void clear()
   {
      type = e_none;
      value = "";
      position = std::numeric_limits<std::size_t>::max();
   }

   template <typename Iterator>
   inline token &
   set_operator(const token_type tt, const Iterator begin, const Iterator end, const Iterator base_begin = Iterator(0))
   {
      type = tt;
      value.assign(begin, end);
      if (base_begin)
         position = std::distance(base_begin, begin);
      return *this;
   }

   template <typename Iterator>
   inline token &set_symbol(const Iterator begin, const Iterator end, const Iterator base_begin = Iterator(0))
   {
      type = e_symbol;
      value.assign(begin, end);
      if (base_begin)
         position = std::distance(base_begin, begin);
      return *this;
   }

   template <typename Iterator>
   inline token &set_numeric(const Iterator begin, const Iterator end, const Iterator base_begin = Iterator(0))
   {
      type = e_number;
      value.assign(begin, end);
      if (base_begin)
         position = std::distance(base_begin, begin);
      return *this;
   }

   template <typename Iterator>
   inline token &set_string(const Iterator begin, const Iterator end, const Iterator base_begin = Iterator(0))
   {
      type = e_string;
      value.assign(begin, end);
      if (base_begin)
         position = std::distance(base_begin, begin);
      return *this;
   }

   inline token &set_string(const std::string &s, const std::size_t p)
   {
      type = e_string;
      value = s;
      position = p;
      return *this;
   }

   template <typename Iterator>
   inline token &
   set_error(const token_type et, const Iterator begin, const Iterator end, const Iterator base_begin = Iterator(0))
   {
      if ((e_error == et) || (e_err_symbol == et) || (e_err_number == et) || (e_err_string == et)) {
         type = e_error;
      } else
         type = e_error;

      value.assign(begin, end);

      if (base_begin)
         position = std::distance(base_begin, begin);

      return *this;
   }

   static inline std::string to_str(token_type t)
   {
      switch (t) {
      case e_none: return "NONE";
      case e_error: return "ERROR";
      case e_err_symbol: return "ERROR_SYMBOL";
      case e_err_number: return "ERROR_NUMBER";
      case e_err_string: return "ERROR_STRING";
      case e_eof: return "EOF";
      case e_number: return "NUMBER";
      case e_symbol: return "SYMBOL";
      case e_string: return "STRING";
      case e_assign: return ":=";
      case e_shr: return ">>";
      case e_shl: return "<<";
      case e_lte: return "<=";
      case e_ne: return "!=";
      case e_gte: return ">=";
      case e_lt: return "<";
      case e_gt: return ">";
      case e_eq: return "=";
      case e_rbracket: return ")";
      case e_lbracket: return "(";
      case e_rsqrbracket: return "]";
      case e_lsqrbracket: return "[";
      case e_rcrlbracket: return "}";
      case e_lcrlbracket: return "{";
      case e_comma: return ",";
      case e_add: return "+";
      case e_sub: return "-";
      case e_div: return "/";
      case e_mul: return "*";
      case e_mod: return "%";
      case e_pow: return "^";
      case e_colon: return ":";
      default: return "UNKNOWN";
      }
   }

   inline bool is_error() const
   {
      return ((e_error == type) || (e_err_symbol == type) || (e_err_number == type) || (e_err_string == type));
   }

   token_type type;
   std::string value;
   std::size_t position;
};

class generator {
public:
   typedef token token_t;
   typedef std::deque<token_t> token_list_t;
   typedef std::deque<token_t>::iterator token_list_itr_t;

   generator() : base_itr_(0), s_itr_(0), s_end_(0) { clear(); }

   inline void clear()
   {
      base_itr_ = 0;
      s_itr_ = 0;
      s_end_ = 0;
      token_list_.clear();
      token_itr_ = token_list_.end();
      store_token_itr_ = token_list_.end();
   }

   inline bool process(const std::string &str)
   {
      base_itr_ = str.data();
      s_itr_ = str.data();
      s_end_ = str.data() + str.size();

      eof_token_.set_operator(token_t::e_eof, s_end_, s_end_, base_itr_);
      token_list_.clear();

      while (!is_end(s_itr_)) {
         scan_token();

         if (token_list_.empty())
            return true;
         else if (token_list_.back().is_error()) {
            return false;
         }
      }
      return true;
   }

   inline bool empty() const { return token_list_.empty(); }

   inline std::size_t size() const { return token_list_.size(); }

   inline void begin()
   {
      token_itr_ = token_list_.begin();
      store_token_itr_ = token_list_.begin();
   }

   inline void store() { store_token_itr_ = token_itr_; }

   inline void restore() { token_itr_ = store_token_itr_; }

   inline token_t &next_token()
   {
      if (token_list_.end() != token_itr_) {
         return *token_itr_++;
      } else
         return eof_token_;
   }

   inline token_t &peek_next_token()
   {
      if (token_list_.end() != token_itr_) {
         return *token_itr_;
      } else
         return eof_token_;
   }

   inline token_t &operator[](const std::size_t &index)
   {
      if (index < token_list_.size())
         return token_list_[index];
      else
         return eof_token_;
   }

   inline token_t operator[](const std::size_t &index) const
   {
      if (index < token_list_.size())
         return token_list_[index];
      else
         return eof_token_;
   }

   inline bool finished() const { return (token_list_.end() == token_itr_); }

   inline std::string remaining() const
   {
      if (finished())
         return "";
      else if (token_list_.begin() != token_itr_)
         return std::string(base_itr_ + (token_itr_ - 1)->position, s_end_);
      else
         return std::string(base_itr_ + token_itr_->position, s_end_);
   }

private:
   inline bool is_end(const char *itr) { return (s_end_ == itr); }

   inline void skip_whitespace()
   {
      while (!is_end(s_itr_) && details::is_whitespace(*s_itr_)) {
         ++s_itr_;
      }
   }

   inline void skip_comments()
   {
      // The following comment styles are supported:
      // 1. // .... \n
      // 2. #  .... \n
      // 3. /* .... */
      struct test {
         static inline bool comment_start(const char c0, const char c1, int &mode, int &incr)
         {
            mode = 0;
            if ('#' == c0) {
               mode = 1;
               incr = 1;
            } else if ('/' == c0) {
               if ('/' == c1) {
                  mode = 1;
                  incr = 2;
               } else if ('*' == c1) {
                  mode = 2;
                  incr = 2;
               }
            }
            return (mode != 0);
         }

         static inline bool comment_end(const char c0, const char c1, const int mode)
         {
            return ((1 == mode) && ('\n' == c0)) || ((2 == mode) && ('*' == c0) && ('/' == c1));
         }
      };

      int mode = 0;
      int increment = 0;

      if (is_end(s_itr_) || is_end((s_itr_ + 1)))
         return;
      else if (!test::comment_start(*s_itr_, *(s_itr_ + 1), mode, increment))
         return;

      s_itr_ += increment;

      while (!is_end(s_itr_) && !test::comment_end(*s_itr_, *(s_itr_ + 1), mode)) {
         ++s_itr_;
      }

      if (!is_end(s_itr_)) {
         s_itr_ += mode;
         skip_whitespace();
         skip_comments();
      }
   }

   inline void scan_token()
   {
      skip_whitespace();

      skip_comments();

      if (is_end(s_itr_)) {
         return;
      } else if (details::is_operator_char(*s_itr_)) {
         scan_operator();
         return;
      } else if (details::is_letter(*s_itr_)) {
         scan_symbol();
         return;
      } else if (details::is_digit((*s_itr_)) || ('.' == (*s_itr_))) {
         scan_number();
         return;
      } else if ('\'' == (*s_itr_) || '\"' == (*s_itr_)) {
         scan_string(*s_itr_);
         return;
      } else {
         token_t t;
         t.set_error(token::e_error, s_itr_, s_itr_ + 2, base_itr_);
         token_list_.push_back(t);
         ++s_itr_;
      }
   }

   inline void scan_operator()
   {
      token_t t;

      if (!is_end(s_itr_ + 1)) {
         token_t::token_type ttype = token_t::e_none;

         char c0 = s_itr_[0];
         char c1 = s_itr_[1];

         if ((c0 == '<') && (c1 == '='))
            ttype = token_t::e_lte;
         else if ((c0 == '>') && (c1 == '='))
            ttype = token_t::e_gte;
         else if ((c0 == '<') && (c1 == '>'))
            ttype = token_t::e_ne;
         else if ((c0 == '!') && (c1 == '='))
            ttype = token_t::e_ne;
         else if ((c0 == '=') && (c1 == '='))
            ttype = token_t::e_eq;
         else if ((c0 == ':') && (c1 == '='))
            ttype = token_t::e_assign;
         else if ((c0 == '<') && (c1 == '<'))
            ttype = token_t::e_shl;
         else if ((c0 == '>') && (c1 == '>'))
            ttype = token_t::e_shr;

         if (token_t::e_none != ttype) {
            t.set_operator(ttype, s_itr_, s_itr_ + 2, base_itr_);
            token_list_.push_back(t);
            s_itr_ += 2;
            return;
         }
      }

      if ('<' == *s_itr_)
         t.set_operator(token_t::e_lt, s_itr_, s_itr_ + 1, base_itr_);
      else if ('>' == *s_itr_)
         t.set_operator(token_t::e_gt, s_itr_, s_itr_ + 1, base_itr_);
      else if (';' == *s_itr_)
         t.set_operator(token_t::e_eof, s_itr_, s_itr_ + 1, base_itr_);
      else if ('&' == *s_itr_)
         t.set_symbol(s_itr_, s_itr_ + 1, base_itr_);
      else if ('|' == *s_itr_)
         t.set_symbol(s_itr_, s_itr_ + 1, base_itr_);
      else
         t.set_operator(token_t::token_type(*s_itr_), s_itr_, s_itr_ + 1, base_itr_);

      token_list_.push_back(t);

      ++s_itr_;
   }

   inline void scan_symbol()
   {
      const char *begin = s_itr_;
         while (
             (!is_end(s_itr_)) &&
             (details::is_letter_or_digit(*s_itr_) || ((*s_itr_) == '_')))
         {
         ++s_itr_;
         }
         token_t t;
         t.set_symbol(begin,s_itr_,base_itr_);
         token_list_.push_back(t);
   }

   inline void scan_number()
   {
      /*
         Attempt to match a valid numeric value in one of the following formats:
         01. 123456
         02. 123.456
         03. 123.456e3
         04. 123.456E3
         05. 123.456e+3
         06. 123.456E+3
         07. 123.456e-3
         08. 123.456E-3
         09. .1234
         10. .1234e3
         11. .1234E+3
         12. .1234e+3
         13. .1234E-3
         14. .1234e-3
      */
      const char *begin = s_itr_;
      bool dot_found = false;
      bool e_found = false;
      bool post_e_sign_found = false;
      bool post_e_digit_found = false;
      token_t t;

      while (!is_end(s_itr_)) {
         if ('.' == (*s_itr_)) {
            if (dot_found) {
               t.set_error(token::e_err_number, begin, s_itr_, base_itr_);
               token_list_.push_back(t);

               return;
            }

            dot_found = true;
            ++s_itr_;

            continue;
         } else if (details::imatch('e', (*s_itr_))) {
            const char &c = *(s_itr_ + 1);

            if (is_end(s_itr_ + 1)) {
               t.set_error(token::e_err_number, begin, s_itr_, base_itr_);
               token_list_.push_back(t);

               return;
            } else if (('+' != c) && ('-' != c) && !details::is_digit(c)) {
               // Here we patch the original version in order to support
               // cases where we have a data member with a name starting with 'e'
               break;
               // Original version below:
               // t.set_error(token::e_err_number,begin,s_itr_,base_itr_);
               // token_list_.push_back(t);

               // return;
            }

            e_found = true;
            ++s_itr_;

            continue;
         } else if (e_found && details::is_sign(*s_itr_) && !post_e_digit_found) {
            if (post_e_sign_found) {
               t.set_error(token::e_err_number, begin, s_itr_, base_itr_);
               token_list_.push_back(t);

               return;
            }

            post_e_sign_found = true;
            ++s_itr_;

            continue;
         } else if (e_found && details::is_digit(*s_itr_)) {
            post_e_digit_found = true;
            ++s_itr_;

            continue;
         } else if (('.' != (*s_itr_)) && !details::is_digit(*s_itr_))
            break;
         else
            ++s_itr_;
      }

      t.set_numeric(begin, s_itr_, base_itr_);

      token_list_.push_back(t);

      return;
   }

   inline void scan_string(char quote)
   {
      const char *begin = s_itr_ + 1;

      token_t t;

      if (std::distance(s_itr_, s_end_) < 2) {
         t.set_error(token::e_err_string, s_itr_, s_end_, base_itr_);
         token_list_.push_back(t);

         return;
      }

      ++s_itr_;

      bool escaped_found = false;
      bool escaped = false;

      while (!is_end(s_itr_)) {
         if (!escaped && ('\\' == *s_itr_)) {
            escaped_found = true;
            escaped = true;
            ++s_itr_;

            continue;
         } else if (!escaped) {
            if (quote == *s_itr_)
               break;
         } else if (escaped)
            escaped = false;

         ++s_itr_;
      }

      if (is_end(s_itr_)) {
         t.set_error(token::e_err_string, begin, s_itr_, base_itr_);
         token_list_.push_back(t);

         return;
      }

      if (!escaped_found)
         t.set_string(begin, s_itr_, base_itr_);
      else {
         std::string parsed_string(begin, s_itr_);
         details::cleanup_escapes(parsed_string);
         t.set_string(parsed_string, std::distance(base_itr_, begin));
      }

      token_list_.push_back(t);
      ++s_itr_;

      return;
   }

private:
   token_list_t token_list_;
   token_list_itr_t token_itr_;
   token_list_itr_t store_token_itr_;
   token_t eof_token_;
   const char *base_itr_;
   const char *s_itr_;
   const char *s_end_;

   friend class token_scanner;
   friend class token_modifier;
   friend class token_inserter;
   friend class token_joiner;
};

class helper_interface {
public:
   virtual void init() {}
   virtual void reset() {}
   virtual bool result() { return true; }
   virtual std::size_t process(generator &) { return 0; }
   virtual ~helper_interface() {}
};

class token_scanner : public helper_interface {
public:
   virtual ~token_scanner() {}

   explicit token_scanner(const std::size_t &stride) : stride_(stride)
   {
      if (stride > 4) {
         throw std::invalid_argument("token_scanner() - Invalid stride value");
      }
   }

   inline std::size_t process(generator &g)
   {
      if (!g.token_list_.empty()) {
         for (std::size_t i = 0; i < (g.token_list_.size() - stride_ + 1); ++i) {
            token t;
            switch (stride_) {
            case 1: {
               const token &t0 = g.token_list_[i];

               if (!operator()(t0))
                  return i;
            } break;

            case 2: {
               const token &t0 = g.token_list_[i];
               const token &t1 = g.token_list_[i + 1];

               if (!operator()(t0, t1))
                  return i;
            } break;

            case 3: {
               const token &t0 = g.token_list_[i];
               const token &t1 = g.token_list_[i + 1];
               const token &t2 = g.token_list_[i + 2];

               if (!operator()(t0, t1, t2))
                  return i;
            } break;

            case 4: {
               const token &t0 = g.token_list_[i];
               const token &t1 = g.token_list_[i + 1];
               const token &t2 = g.token_list_[i + 2];
               const token &t3 = g.token_list_[i + 3];

               if (!operator()(t0, t1, t2, t3))
                  return i;
            } break;
            }
         }
      }

      return (g.token_list_.size() - stride_ + 1);
   }

   virtual bool operator()(const token &) { return false; }

   virtual bool operator()(const token &, const token &) { return false; }

   virtual bool operator()(const token &, const token &, const token &) { return false; }

   virtual bool operator()(const token &, const token &, const token &, const token &) { return false; }

private:
   std::size_t stride_;
};

class token_modifier : public helper_interface {
public:
   inline std::size_t process(generator &g)
   {
      std::size_t changes = 0;

      for (std::size_t i = 0; i < g.token_list_.size(); ++i) {
         if (modify(g.token_list_[i]))
            changes++;
      }

      return changes;
   }

   virtual bool modify(token &t) = 0;
};

class token_inserter : public helper_interface {
public:
   explicit token_inserter(const std::size_t &stride) : stride_(stride)
   {
      if (stride > 5) {
         throw std::invalid_argument("token_inserter() - Invalid stride value");
      }
   }

   inline std::size_t process(generator &g)
   {
      if (g.token_list_.empty())
         return 0;

      std::size_t changes = 0;

      for (std::size_t i = 0; i < (g.token_list_.size() - stride_ + 1); ++i) {
         token t;
         int insert_index = -1;

         switch (stride_) {
         case 1: insert_index = insert(g.token_list_[i], t); break;

         case 2: insert_index = insert(g.token_list_[i], g.token_list_[i + 1], t); break;

         case 3: insert_index = insert(g.token_list_[i], g.token_list_[i + 1], g.token_list_[i + 2], t); break;

         case 4:
            insert_index =
               insert(g.token_list_[i], g.token_list_[i + 1], g.token_list_[i + 2], g.token_list_[i + 3], t);
            break;

         case 5:
            insert_index = insert(g.token_list_[i], g.token_list_[i + 1], g.token_list_[i + 2], g.token_list_[i + 3],
                                  g.token_list_[i + 4], t);
            break;
         }

         if ((insert_index >= 0) && (insert_index <= (static_cast<int>(stride_) + 1))) {
            g.token_list_.insert(g.token_list_.begin() + (i + insert_index), t);
            changes++;
         }
      }

      return changes;
   }

   virtual inline int insert(const token &, token &) { return -1; }

   virtual inline int insert(const token &, const token &, token &) { return -1; }

   virtual inline int insert(const token &, const token &, const token &, token &) { return -1; }

   virtual inline int insert(const token &, const token &, const token &, const token &, token &) { return -1; }

   virtual inline int insert(const token &, const token &, const token &, const token &, const token &, token &)
   {
      return -1;
   }

private:
   std::size_t stride_;
};

class token_joiner : public helper_interface {
public:
   inline std::size_t process(generator &g)
   {
      if (g.token_list_.empty())
         return 0;

      std::size_t changes = 0;

      for (std::size_t i = 0; i < g.token_list_.size() - 1; ++i) {
         token t;

         if (join(g.token_list_[i], g.token_list_[i + 1], t)) {
            g.token_list_[i] = t;
            g.token_list_.erase(g.token_list_.begin() + (i + 1));

            ++changes;
         }
      }

      return changes;
   }

   virtual bool join(const token &, const token &, token &) = 0;
};

namespace helper {

inline void dump(lexertk::generator &generator)
{
   for (std::size_t i = 0; i < generator.size(); ++i) {
      lexertk::token t = generator[i];
      printf("Token[%02d] @ %03d  %6s  -->  '%s'\n", static_cast<unsigned int>(i),
             static_cast<unsigned int>(t.position), t.to_str(t.type).c_str(), t.value.c_str());
   }
}

class commutative_inserter : public token_inserter {
public:
   commutative_inserter() : lexertk::token_inserter(2) {}

   inline void ignore_symbol(const std::string &symbol) { ignore_set_.insert(symbol); }

   using token_inserter::insert;
   inline int insert(const lexertk::token &t0, const lexertk::token &t1, lexertk::token &new_token)
   {
      new_token.type = lexertk::token::e_mul;
      new_token.value = "*";
      new_token.position = t1.position;
      bool match = false;

      if (t0.type == lexertk::token::e_symbol) {
         if (ignore_set_.end() != ignore_set_.find(t0.value)) {
            return -1;
         } else if (!t0.value.empty() && ('$' == t0.value[0])) {
            return -1;
         }
      }

      if (t1.type == lexertk::token::e_symbol) {
         if (ignore_set_.end() != ignore_set_.find(t1.value)) {
            return -1;
         }
      }

      if ((t0.type == lexertk::token::e_number) && (t1.type == lexertk::token::e_symbol))
         match = true;
      else if ((t0.type == lexertk::token::e_number) && (t1.type == lexertk::token::e_lbracket))
         match = true;
      else if ((t0.type == lexertk::token::e_number) && (t1.type == lexertk::token::e_lcrlbracket))
         match = true;
      else if ((t0.type == lexertk::token::e_number) && (t1.type == lexertk::token::e_lsqrbracket))
         match = true;
      else if ((t0.type == lexertk::token::e_symbol) && (t1.type == lexertk::token::e_number))
         match = true;
      else if ((t0.type == lexertk::token::e_rbracket) && (t1.type == lexertk::token::e_number))
         match = true;
      else if ((t0.type == lexertk::token::e_rcrlbracket) && (t1.type == lexertk::token::e_number))
         match = true;
      else if ((t0.type == lexertk::token::e_rsqrbracket) && (t1.type == lexertk::token::e_number))
         match = true;
      else if ((t0.type == lexertk::token::e_rbracket) && (t1.type == lexertk::token::e_symbol))
         match = true;
      else if ((t0.type == lexertk::token::e_rcrlbracket) && (t1.type == lexertk::token::e_symbol))
         match = true;
      else if ((t0.type == lexertk::token::e_rsqrbracket) && (t1.type == lexertk::token::e_symbol))
         match = true;

      return (match) ? 1 : -1;
   }

private:
   std::set<std::string, details::ilesscompare> ignore_set_;
};

class operator_joiner : public token_joiner {
public:
   inline bool join(const lexertk::token &t0, const lexertk::token &t1, lexertk::token &t)
   {
      //': =' --> ':='
      if ((t0.type == lexertk::token::e_colon) && (t1.type == lexertk::token::e_eq)) {
         t.type = lexertk::token::e_assign;
         t.value = ":=";
         t.position = t0.position;

         return true;
      }
      //'> =' --> '>='
      else if ((t0.type == lexertk::token::e_gt) && (t1.type == lexertk::token::e_eq)) {
         t.type = lexertk::token::e_gte;
         t.value = ">=";
         t.position = t0.position;

         return true;
      }
      //'< =' --> '<='
      else if ((t0.type == lexertk::token::e_lt) && (t1.type == lexertk::token::e_eq)) {
         t.type = lexertk::token::e_lte;
         t.value = "<=";
         t.position = t0.position;

         return true;
      }
      //'= =' --> '=='
      else if ((t0.type == lexertk::token::e_eq) && (t1.type == lexertk::token::e_eq)) {
         t.type = lexertk::token::e_eq;
         t.value = "==";
         t.position = t0.position;

         return true;
      }
      //'! =' --> '!='
      else if ((static_cast<char>(t0.type) == '!') && (t1.type == lexertk::token::e_eq)) {
         t.type = lexertk::token::e_ne;
         t.value = "!=";
         t.position = t0.position;

         return true;
      }
      //'< >' --> '<>'
      else if ((t0.type == lexertk::token::e_lt) && (t1.type == lexertk::token::e_gt)) {
         t.type = lexertk::token::e_ne;
         t.value = "<>";
         t.position = t0.position;

         return true;
      } else
         return false;
   }
};

class bracket_checker : public token_scanner {
public:
   bracket_checker() : token_scanner(1), state_(true) {}

   bool result() { return state_ && stack_.empty(); }

   lexertk::token error_token() { return error_token_; }

   void reset()
   {
      // why? because msvc doesn't support swap properly.
      stack_ = std::stack<char>();
      state_ = true;
      error_token_.clear();
   }

   using token_scanner::operator();
   bool operator()(const lexertk::token &t)
   {
      if (!t.value.empty() && (lexertk::token::e_string != t.type) && (lexertk::token::e_symbol != t.type) &&
          details::is_bracket(t.value[0])) {
         char c = t.value[0];

         if (t.type == lexertk::token::e_lbracket)
            stack_.push(')');
         else if (t.type == lexertk::token::e_lcrlbracket)
            stack_.push('}');
         else if (t.type == lexertk::token::e_lsqrbracket)
            stack_.push(']');
         else if (details::is_right_bracket(c)) {
            if (stack_.empty()) {
               state_ = false;
               error_token_ = t;

               return false;
            } else if (c != stack_.top()) {
               state_ = false;
               error_token_ = t;

               return false;
            } else
               stack_.pop();
         }
      }

      return true;
   }

private:
   bool state_;
   std::stack<char> stack_;
   lexertk::token error_token_;
};

class symbol_replacer : public token_modifier {
private:
   typedef std::map<std::string, std::pair<std::string, token::token_type>, details::ilesscompare> replace_map_t;

public:
   bool remove(const std::string &target_symbol)
   {
      replace_map_t::iterator itr = replace_map_.find(target_symbol);

      if (replace_map_.end() == itr)
         return false;

      replace_map_.erase(itr);

      return true;
   }

   bool add_replace(const std::string &target_symbol, const std::string &replace_symbol,
                    const lexertk::token::token_type token_type = lexertk::token::e_symbol)
   {
      replace_map_t::iterator itr = replace_map_.find(target_symbol);

      if (replace_map_.end() != itr) {
         return false;
      }

      replace_map_[target_symbol] = std::make_pair(replace_symbol, token_type);

      return true;
   }

   void clear() { replace_map_.clear(); }

private:
   bool modify(lexertk::token &t)
   {
      if (lexertk::token::e_symbol == t.type) {
         if (replace_map_.empty())
            return false;

         replace_map_t::iterator itr = replace_map_.find(t.value);

         if (replace_map_.end() != itr) {
            t.value = itr->second.first;
            t.type = itr->second.second;

            return true;
         }
      }

      return false;
   }

   replace_map_t replace_map_;
};

class sequence_validator : public token_scanner {
private:
   typedef std::pair<lexertk::token::token_type, lexertk::token::token_type> token_pair_t;
   typedef std::set<token_pair_t> set_t;

public:
   sequence_validator() : lexertk::token_scanner(2)
   {
      add_invalid(lexertk::token::e_number, lexertk::token::e_number);
      add_invalid(lexertk::token::e_string, lexertk::token::e_string);
      add_invalid(lexertk::token::e_number, lexertk::token::e_string);
      add_invalid(lexertk::token::e_string, lexertk::token::e_number);
      add_invalid(lexertk::token::e_string, lexertk::token::e_colon);
      add_invalid(lexertk::token::e_colon, lexertk::token::e_string);
      add_invalid_set1(lexertk::token::e_assign);
      add_invalid_set1(lexertk::token::e_shr);
      add_invalid_set1(lexertk::token::e_shl);
      add_invalid_set1(lexertk::token::e_lte);
      add_invalid_set1(lexertk::token::e_ne);
      add_invalid_set1(lexertk::token::e_gte);
      add_invalid_set1(lexertk::token::e_lt);
      add_invalid_set1(lexertk::token::e_gt);
      add_invalid_set1(lexertk::token::e_eq);
      add_invalid_set1(lexertk::token::e_comma);
      add_invalid_set1(lexertk::token::e_add);
      add_invalid_set1(lexertk::token::e_sub);
      add_invalid_set1(lexertk::token::e_div);
      add_invalid_set1(lexertk::token::e_mul);
      add_invalid_set1(lexertk::token::e_mod);
      add_invalid_set1(lexertk::token::e_pow);
      add_invalid_set1(lexertk::token::e_colon);
   }

   bool result() { return error_list_.empty(); }

   using token_scanner::operator();
   bool operator()(const lexertk::token &t0, const lexertk::token &t1)
   {
      set_t::value_type p = std::make_pair(t0.type, t1.type);

      if (invalid_bracket_check(t0.type, t1.type)) {
         error_list_.push_back(std::make_pair(t0, t1));
      } else if (invalid_comb_.find(p) != invalid_comb_.end())
         error_list_.push_back(std::make_pair(t0, t1));

      return true;
   }

   std::size_t error_count() { return error_list_.size(); }

   std::pair<lexertk::token, lexertk::token> error(const std::size_t index)
   {
      if (index < error_list_.size()) {
         return error_list_[index];
      } else {
         static const lexertk::token error_token;
         return std::make_pair(error_token, error_token);
      }
   }

   void clear_errors() { error_list_.clear(); }

private:
   void add_invalid(lexertk::token::token_type base, lexertk::token::token_type t)
   {
      invalid_comb_.insert(std::make_pair(base, t));
   }

   void add_invalid_set1(lexertk::token::token_type t)
   {
      add_invalid(t, lexertk::token::e_assign);
      add_invalid(t, lexertk::token::e_shr);
      add_invalid(t, lexertk::token::e_shl);
      add_invalid(t, lexertk::token::e_lte);
      add_invalid(t, lexertk::token::e_ne);
      add_invalid(t, lexertk::token::e_gte);
      add_invalid(t, lexertk::token::e_lt);
      add_invalid(t, lexertk::token::e_gt);
      add_invalid(t, lexertk::token::e_eq);
      add_invalid(t, lexertk::token::e_comma);
      add_invalid(t, lexertk::token::e_div);
      add_invalid(t, lexertk::token::e_mul);
      add_invalid(t, lexertk::token::e_mod);
      add_invalid(t, lexertk::token::e_pow);
      add_invalid(t, lexertk::token::e_colon);
   }

   bool invalid_bracket_check(lexertk::token::token_type base, lexertk::token::token_type t)
   {
      if (details::is_right_bracket(static_cast<char>(base))) {
         switch (t) {
         case lexertk::token::e_string: return true;
         case lexertk::token::e_assign: return true;
         default: return false;
         }
      } else if (details::is_left_bracket(static_cast<char>(base))) {
         if (details::is_right_bracket(static_cast<char>(t)))
            return false;
         else if (details::is_left_bracket(static_cast<char>(t)))
            return false;
         else {
            switch (t) {
            case lexertk::token::e_number: return false;
            case lexertk::token::e_symbol: return false;
            case lexertk::token::e_string: return false;
            case lexertk::token::e_add: return false;
            case lexertk::token::e_sub: return false;
            case lexertk::token::e_colon: return false;
            default: return true;
            }
         }
      } else if (details::is_right_bracket(static_cast<char>(t))) {
         switch (base) {
         case lexertk::token::e_number: return false;
         case lexertk::token::e_symbol: return false;
         case lexertk::token::e_string: return false;
         case lexertk::token::e_eof: return false;
         case lexertk::token::e_colon: return false;
         default: return true;
         }
      } else if (details::is_left_bracket(static_cast<char>(t))) {
         switch (base) {
         case lexertk::token::e_rbracket: return true;
         case lexertk::token::e_rsqrbracket: return true;
         case lexertk::token::e_rcrlbracket: return true;
         default: return false;
         }
      }

      return false;
   }

   set_t invalid_comb_;
   std::deque<std::pair<lexertk::token, lexertk::token>> error_list_;
};

struct helper_assembly {
   inline bool register_scanner(lexertk::token_scanner *scanner)
   {
      if (token_scanner_list.end() != std::find(token_scanner_list.begin(), token_scanner_list.end(), scanner)) {
         return false;
      }

      token_scanner_list.push_back(scanner);

      return true;
   }

   inline bool register_modifier(lexertk::token_modifier *modifier)
   {
      if (token_modifier_list.end() != std::find(token_modifier_list.begin(), token_modifier_list.end(), modifier)) {
         return false;
      }

      token_modifier_list.push_back(modifier);

      return true;
   }

   inline bool register_joiner(lexertk::token_joiner *joiner)
   {
      if (token_joiner_list.end() != std::find(token_joiner_list.begin(), token_joiner_list.end(), joiner)) {
         return false;
      }

      token_joiner_list.push_back(joiner);

      return true;
   }

   inline bool register_inserter(lexertk::token_inserter *inserter)
   {
      if (token_inserter_list.end() != std::find(token_inserter_list.begin(), token_inserter_list.end(), inserter)) {
         return false;
      }

      token_inserter_list.push_back(inserter);

      return true;
   }

   inline bool run_modifiers(lexertk::generator &g)
   {
      error_token_modifier = reinterpret_cast<lexertk::token_modifier *>(0);

      for (std::size_t i = 0; i < token_modifier_list.size(); ++i) {
         lexertk::token_modifier &modifier = (*token_modifier_list[i]);

         modifier.reset();
         modifier.process(g);

         if (!modifier.result()) {
            error_token_modifier = token_modifier_list[i];

            return false;
         }
      }

      return true;
   }

   inline bool run_joiners(lexertk::generator &g)
   {
      error_token_joiner = reinterpret_cast<lexertk::token_joiner *>(0);

      for (std::size_t i = 0; i < token_joiner_list.size(); ++i) {
         lexertk::token_joiner &joiner = (*token_joiner_list[i]);

         joiner.reset();
         joiner.process(g);

         if (!joiner.result()) {
            error_token_joiner = token_joiner_list[i];

            return false;
         }
      }

      return true;
   }

   inline bool run_inserters(lexertk::generator &g)
   {
      error_token_inserter = reinterpret_cast<lexertk::token_inserter *>(0);

      for (std::size_t i = 0; i < token_inserter_list.size(); ++i) {
         lexertk::token_inserter &inserter = (*token_inserter_list[i]);

         inserter.reset();
         inserter.process(g);

         if (!inserter.result()) {
            error_token_inserter = token_inserter_list[i];

            return false;
         }
      }

      return true;
   }

   inline bool run_scanners(lexertk::generator &g)
   {
      error_token_scanner = reinterpret_cast<lexertk::token_scanner *>(0);

      for (std::size_t i = 0; i < token_scanner_list.size(); ++i) {
         lexertk::token_scanner &scanner = (*token_scanner_list[i]);

         scanner.reset();
         scanner.process(g);

         if (!scanner.result()) {
            error_token_scanner = token_scanner_list[i];

            return false;
         }
      }

      return true;
   }

   std::deque<lexertk::token_scanner *> token_scanner_list;
   std::deque<lexertk::token_modifier *> token_modifier_list;
   std::deque<lexertk::token_joiner *> token_joiner_list;
   std::deque<lexertk::token_inserter *> token_inserter_list;

   lexertk::token_scanner *error_token_scanner;
   lexertk::token_modifier *error_token_modifier;
   lexertk::token_joiner *error_token_joiner;
   lexertk::token_inserter *error_token_inserter;
};
} // namespace helper

class parser_helper {
public:
   typedef token token_t;
   typedef generator generator_t;

   inline bool init(const std::string &str)
   {
      if (!lexer_.process(str)) {
         return false;
      }

      lexer_.begin();

      next_token();

      return true;
   }

   inline generator_t &lexer() { return lexer_; }

   inline const generator_t &lexer() const { return lexer_; }

   inline void next_token() { current_token_ = lexer_.next_token(); }

   inline const token_t &current_token() const { return current_token_; }

   enum token_advance_mode { e_hold = 0, e_advance = 1 };

   inline void advance_token(const token_advance_mode mode)
   {
      if (e_advance == mode) {
         next_token();
      }
   }

   inline bool token_is(const token_t::token_type &ttype, const token_advance_mode mode = e_advance)
   {
      if (current_token().type != ttype) {
         return false;
      }

      advance_token(mode);

      return true;
   }

   inline bool
   token_is(const token_t::token_type &ttype, const std::string &value, const token_advance_mode mode = e_advance)
   {
      if ((current_token().type != ttype) || !details::imatch(value, current_token().value)) {
         return false;
      }

      advance_token(mode);

      return true;
   }

   inline bool
   token_is_then_assign(const token_t::token_type &ttype, std::string &token, const token_advance_mode mode = e_advance)
   {
      if (current_token_.type != ttype) {
         return false;
      }

      token = current_token_.value;

      advance_token(mode);

      return true;
   }

   template <typename Allocator, template <typename, typename> class Container>
   inline bool token_is_then_assign(const token_t::token_type &ttype, Container<std::string, Allocator> &token_list,
                                    const token_advance_mode mode = e_advance)
   {
      if (current_token_.type != ttype) {
         return false;
      }

      token_list.push_back(current_token_.value);

      advance_token(mode);

      return true;
   }

   inline bool peek_token_is(const token_t::token_type &ttype) { return (lexer_.peek_next_token().type == ttype); }

   inline bool peek_token_is(const std::string &s) { return (details::imatch(lexer_.peek_next_token().value, s)); }

private:
   generator_t lexer_;
   token_t current_token_;
};
} // namespace lexertk

#endif
