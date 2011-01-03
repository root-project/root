// @(#)root/reflex:$Id$
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2010, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef REFLEX_BUILD
# define REFLEX_BUILD
#endif

#include "Reflex/Tools.h"

#include "Reflex/Kernel.h"
#include "Reflex/Type.h"
#include "Reflex/internal/OwnedMember.h"
#include <cstring>

#if defined(__GNUC__)
# include <cxxabi.h>
#elif defined(__SUNPRO_CC)
# include <demangle.h>
#endif

using namespace Reflex;

//-------------------------------------------------------------------------------
static std::string
splitScopedName(const std::string& nam,
                bool returnScope,
                bool startFromLeft = false) {
   // Split a scoped name. If returnScope is true return the scope part otherwise
   // the base part. If startFromLeft is true, parse from left otherwise from the end.
   size_t pos = 0;
   size_t start = 0;

   pos = Tools::GetFirstScopePosition(nam, start);
   if (!startFromLeft) {
      // but we keep start!
      pos = Tools::GetBasePosition(nam);
   }

   if (pos == 0) { // There is no scope in the name.
      if (returnScope) {
         return "";
      }
      return nam;
   }

   if (returnScope) {
      return nam.substr(start, pos - 2 - start);
   }
   return nam.substr(pos);
} // splitScopedName


//-------------------------------------------------------------------------------
std::string
Tools::GetScopeName(const std::string& name,
                    bool startFromLeft /*= false*/) {
   // Get the scope of a name. Start either from the beginning (startfFromLeft=true) or end.
   return splitScopedName(name, true, startFromLeft);
}


//-------------------------------------------------------------------------------
std::string
Tools::GetBaseName(const std::string& name,
                   bool startFromLeft /*= false*/) {
   // Get the base of a name. Start either from the beginning (startFromLeft=true) or end.
   return splitScopedName(name, false, startFromLeft);
}


EFUNDAMENTALTYPE
Tools::FundamentalType(const Type& typ) {
   // Return an enum representing the fundamental type passed in.
   static const TypeBase* stb_Char = Type::ByName("char").ToTypeBase();
   static const TypeBase* stb_SigChar = Type::ByName("signed char").ToTypeBase();
   static const TypeBase* stb_ShoInt = Type::ByName("short int").ToTypeBase();
   static const TypeBase* stb_Int = Type::ByName("int").ToTypeBase();
   static const TypeBase* stb_LonInt = Type::ByName("long int").ToTypeBase();
   static const TypeBase* stb_UnsChar = Type::ByName("unsigned char").ToTypeBase();
   static const TypeBase* stb_UnsShoInt = Type::ByName("unsigned short int").ToTypeBase();
   static const TypeBase* stb_UnsInt = Type::ByName("unsigned int").ToTypeBase();
   static const TypeBase* stb_UnsLonInt = Type::ByName("unsigned long int").ToTypeBase();
   static const TypeBase* stb_Bool = Type::ByName("bool").ToTypeBase();
   static const TypeBase* stb_Float = Type::ByName("float").ToTypeBase();
   static const TypeBase* stb_Double = Type::ByName("double").ToTypeBase();
   static const TypeBase* stb_LonDouble = Type::ByName("long double").ToTypeBase();
   static const TypeBase* stb_Void = Type::ByName("void").ToTypeBase();
   static const TypeBase* stb_LonLong = Type::ByName("long long").ToTypeBase();
   static const TypeBase* stb_UnsLonLong = Type::ByName("unsigned long long").ToTypeBase();

   const TypeBase* tbType = typ.FinalType().ToTypeBase();

   if (tbType == stb_Int) {
      return kINT;
   }

   if (tbType == stb_Float) {
      return kFLOAT;
   }

   if (tbType == stb_Double) {
      return kDOUBLE;
   }

   if (tbType == stb_LonInt) {
      return kLONG_INT;
   }

   if (tbType == stb_Char) {
      return kCHAR;
   }

   if (tbType == stb_SigChar) {
      return kSIGNED_CHAR;
   }

   if (tbType == stb_ShoInt) {
      return kSHORT_INT;
   }

   if (tbType == stb_UnsChar) {
      return kUNSIGNED_CHAR;
   }

   if (tbType == stb_UnsShoInt) {
      return kUNSIGNED_SHORT_INT;
   }

   if (tbType == stb_UnsInt) {
      return kUNSIGNED_INT;
   }

   if (tbType == stb_UnsLonInt) {
      return kUNSIGNED_LONG_INT;
   }

   if (tbType == stb_Bool) {
      return kBOOL;
   }

   if (tbType == stb_LonDouble) {
      return kLONG_DOUBLE;
   }

   if (tbType == stb_Void) {
      return kVOID;
   }

   if (tbType == stb_LonLong) {
      return kLONGLONG;
   }

   if (tbType == stb_UnsLonLong) {
      return kULONGLONG;
   }
   return kNOTFUNDAMENTAL;
} // FundamentalType


//-------------------------------------------------------------------------------
std::string
Tools::BuildTypeName(Type& t,
                     unsigned int /* modifiers */) {
//-------------------------------------------------------------------------------
// Build a complete qualified type name.
   std::string mod = "";

   if (t.IsConstVolatile()) {
      mod = "const volatile";
   } else if (t.IsConst()) {
      mod = "const";
   } else if (t.IsVolatile()) {
      mod = "volatile";
   }

   std::string name = t.Name();

   if (t.IsPointer() || t.IsPointerToMember()) {
      name += " " + mod;
   } else { name = mod + " " + name; }

   if (t.IsReference()) {
      name += "&";
   }

   return name;
} // BuildTypeName


//-------------------------------------------------------------------------------
std::vector<std::string>
Tools::GenTemplateArgVec(const std::string& Name) {
//-------------------------------------------------------------------------------
// Return a vector of template arguments from a template type string.

   std::vector<std::string> vec;
   std::string tname;
   GetTemplateComponents(Name, tname, vec);
   return vec;
}


//-------------------------------------------------------------------------------
void
Tools::GetTemplateComponents(const std::string& name,
                             std::string& templatename,
                             std::vector<std::string>& args) {
   // Return the template name and a vector of template arguments.
   //
   // Note:  We must be careful of:
   //
   //       operator<,   operator>,
   //       operator<=,  operator>=,
   //       operator<<,  operator>>,
   //       operator<<=, operator>>=
   //       operator->,  operator->*,
   //       operator()
   //
   long pos = GetBasePosition(name);
   int bracket_depth = 0;
   int paren_depth = 0;
   long args_pos = 0;
   bool have_template = false;
   long len = name.size();

   for (long i = pos; !have_template && (i < len); ++i) {
      char c = name[i];

      if (c == '(') { // check for operator()
         if (i > 7) { // there is room for "operator"
            long j = i - 1;

            while (j && isspace(name[j])) {
               --j;
            }

            if ((j > 6) && (name.substr(j - 7, 8) == "operator")) { // possibly found operator()
               j = i + 1;

               while ((j < len) && isspace(name[j])) {
                  ++j;
               }

               if (j < len) {
                  if (name[j] == ')') {
                     i = j;
                     continue; // skip changing depth
                  }
               }
            }
         }
         ++paren_depth;
      } else if (c == ')') {
         --paren_depth;
      } else if (c == '<') { // check for operator<, operator<=, operator<<, and operator<<=
         if (i > 7) { // there is room for "operator"
            long j = i - 1;

            while (j && isspace(name[j])) {
               --j;
            }

            if ((j > 6) && (name.substr(j - 7, 8) == "operator")) { // found at least operator<
               j = i + 1;

               if (j < len) { // check for operator<=, operator<<, or operator<<=
                  if (name[j] == '=') { // operator<=
                     i = j;
                  } else if (name[j] == '<') { // we have operator<<, or operator<<=
                     i = j;
                     ++j;

                     if (j < len) {
                        if (name[j] == '=') { // we have operator<<=
                           i = j;
                        }
                     }
                  }
               }
               continue; // skip changing depth
            }
         }

         if (!paren_depth && !bracket_depth) { // We found the opening '<' of a set of template arguments.
            templatename = name.substr(0, i);
            have_template = true;
            args_pos = i;
            continue;
         }
         ++bracket_depth;
      } else if (c == '>') { // check for operator>, operator>=, operator>>, operator>>=, operator->, or operator->*
         if (i > 7) { // there is room for "operator"
            long j = i - 1;
            bool have_arrow = false;

            if (name[j] == '-') { // allow for operator->, or operator->*
               have_arrow = true;
               --j;
            }

            while (j && isspace(name[j])) {
               --j;
            }

            if ((j > 6) && (name.substr(j - 7, 8) == "operator")) { // found at least operator> or operator->
               j = i + 1;

               if (j < len) { // check for operator->*, operator>=, operator>>, or operator>>=
                  if (have_arrow && (name[j] == '*')) { // we have operator->*
                     i = j;
                  } else if (!have_arrow) {
                     if (name[j] == '=') { // we have operator>=
                        i = j;
                     } else if (name[j] == '>') { // we have operator>>, or operator>>=
                        i = j;
                        ++j;

                        if (j < len) {
                           if (name[j] == '=') { // we have operator>>=
                              i = j;
                           }
                        }
                     }
                  }
               }
               continue; // skip changing depth
            }
         }
         --bracket_depth;
      }
   }

   if (!have_template) {
      return;
   }
   long begin_arg = args_pos + 1;

   for (long i = args_pos; i < len; ++i) {
      char c = name[i];

      if (c == '(') { // check for operator()
         if (i > 7) { // there is room for "operator"
            long j = i - 1;

            while (j && isspace(name[j])) {
               --j;
            }

            if ((j > 6) && (name.substr(j - 7, 8) == "operator")) { // possibly found operator()
               j = i + 1;

               while ((j < len) && isspace(name[j])) {
                  ++j;
               }

               if (j < len) {
                  if (name[j] == ')') {
                     i = j;
                     continue; // skip changing depth
                  }
               }
            }
         }
         ++paren_depth;
      } else if (c == ')') {
         --paren_depth;
      } else if (c == '<') { // check for operator<, operator<=, operator<<, and operator<<=
         if (i > 7) { // there is room for "operator"
            long j = i - 1;

            while (j && isspace(name[j])) {
               --j;
            }

            if ((j > 6) && (name.substr(j - 7, 8) == "operator")) { // found at least operator<
               j = i + 1;

               if (j < len) { // check for operator<=, operator<<, or operator<<=
                  if (name[j] == '=') { // operator<=
                     i = j;
                  } else if (name[j] == '<') { // we have operator<<, or operator<<=
                     i = j;
                     ++j;

                     if (j < len) {
                        if (name[j] == '=') { // we have operator<<=
                           i = j;
                        }
                     }
                  }
               }
               continue; // skip changing depth
            }
         }
         ++bracket_depth;
      } else if (c == '>') { // check for operator>, operator>=, operator>>, operator>>=, operator->, or operator->*
         if (i > 7) { // there is room for "operator"
            long j = i - 1;
            bool have_arrow = false;

            if (name[j] == '-') { // allow for operator->, or operator->*
               have_arrow = true;
               --j;
            }

            while (j && isspace(name[j])) {
               --j;
            }

            if ((j > 6) && (name.substr(j - 7, 8) == "operator")) { // found at least operator> or operator->
               j = i + 1;

               if (j < len) { // check for operator->*, operator>=, operator>>, or operator>>=
                  if (have_arrow && (name[j] == '*')) { // we have operator->*
                     i = j;
                  } else if (!have_arrow) {
                     if (name[j] == '=') { // we have operator>=
                        i = j;
                     } else if (name[j] == '>') { // we have operator>>, or operator>>=
                        i = j;
                        ++j;

                        if (j < len) {
                           if (name[j] == '=') { // we have operator>>=
                              i = j;
                           }
                        }
                     }
                  }
               }
               continue; // skip changing depth
            }
         }
         --bracket_depth;

         if (!bracket_depth) { // We have reached the end of the template arguments;
            if (i - begin_arg) { // Be careful of MyTempl<>
               std::string tmp(name.substr(begin_arg, i - begin_arg));
               StringStrip(tmp);
               args.push_back(tmp);
            }
            return;
         }
      } else if (!paren_depth && (bracket_depth == 1) && (c == ',')) { // We have reached the end of an argument
         std::string tmp(name.substr(begin_arg, i - begin_arg));
         StringStrip(tmp);
         args.push_back(tmp);
         begin_arg = i + 1;
      }
   }
   // We cannot get here.
   return;
} // GetTemplateComponents


//-------------------------------------------------------------------------------
size_t
Tools::GetBasePosition(const std::string& name) {
//-------------------------------------------------------------------------------
// -- Get the position of the base part of a scoped name.
//
// Remove the template part of the name <...>,
// but we must be careful of:
//
//       operator<,   operator>,
//       operator<=,  operator>=,
//       operator<<,  operator>>,
//       operator<<=, operator>>=
//       operator->,  operator->*,
//       operator()
//
   int ab = 0; // angle brace depth
   int rb = 0; // right brace depth, actually parenthesis depth
   size_t pos = 0;

   for (long i = name.size() - 1; (i >= 0) && !pos; --i) {
      switch (name[i]) {
      case '>':
      {
         long j = i - 1;

         if (j > -1) {
            if ((name[j] == '-') || (name[j] == '>')) {
               --j;
            }
         }

         for ( ; (j > -1) && (name[j] == ' '); --j) {
         }

         if ((j > -1) && (name[j] == 'r') && ((j - 7) > -1)) {
            // -- We may have an operator name.
            if (name.substr(j - 7, 8) == "operator") {
               i = j - 7;
               break;
            }
         }
         ab++;
      }
      break;
      case '<':
      {
         long j = i - 1;

         if (j > -1) {
            if (name[j] == '<') {
               --j;
            }
         }

         for ( ; (j > -1) && (name[j] == ' '); --j) {
         }

         if ((j > -1) && (name[j] == 'r') && ((j - 7) > -1)) {
            // -- We may have an operator name.
            if (name.substr(j - 7, 8) == "operator") {
               i = j - 7;
               break;
            }
         }
      }
         ab--;
         break;
      case ')':
      {
         long j = i - 1;

         for ( ; (j > -1) && (name[j] == ' '); --j) {
         }

         if (j > -1) {
            if (name[j] == '(') {
               --j;

               for ( ; (j > -1) && (name[j] == ' '); --j) {
               }

               if ((j > -1) && (name[j] == 'r') && ((j - 7) > -1)) {
                  // -- We may have an operator name.
                  if (name.substr(j - 7, 8) == "operator") {
                     i = j - 7;
                     break;
                  }
               }
            }
         }
      }
         rb++;
         break;
      case '(':
      {
         long j = i - 1;

         for ( ; (j > -1) && (name[j] == ' '); --j) {
         }

         if ((j > -1) && (name[j] == 'r') && ((j - 7) > -1)) {
            // -- We may have an operator name.
            if (name.substr(j - 7, 8) == "operator") {
               i = j - 7;
               break;
            }
         }
      }
         rb--;
         break;
      case ':':

         if (!ab && !rb && i && (name[i - 1] == ':')) {
            pos = i + 1;
         }
         break;
      } // switch
   }
   return pos;
} // GetBasePosition


//-------------------------------------------------------------------------------
size_t
Tools::GetFirstScopePosition(const std::string& name, size_t& start) {
   // Get the position of the first scope of a scoped name.
   //
   // Note:  We must be careful of:
   //
   //       operator<,   operator>,
   //       operator<=,  operator>=,
   //       operator<<,  operator>>,
   //       operator<<=, operator>>=
   //       operator->,  operator->*,
   //       operator()
   //
   int bracket_depth = 0;
   int paren_depth = 0;
   long len = name.size();
   size_t scopePos = std::string::npos;
   start = 0;

   for (long i = 0; i < len; ++i) {
      char c = name[i];

      if (c == '(') { // check for operator()
         if (i > 7) { // there is room for "operator"
            long j = i - 1;

            while (j && isspace(name[j])) {
               --j;
            }

            if ((j > 6) && (name.substr(j - 7, 8) == "operator")) { // possibly found operator()
               j = i + 1;

               while ((j < len) && isspace(name[j])) {
                  ++j;
               }

               if (j < len) {
                  if (name[j] == ')') {
                     i = j;
                     continue; // skip changing depth
                  }
               }
            }
         }
         ++paren_depth;
      } else if (c == ')') {
         --paren_depth;
      } else if (c == '<') { // check for operator<, operator<=, operator<<, and operator<<=
         if (i > 7) { // there is room for "operator"
            long j = i - 1;

            while (j && isspace(name[j])) {
               --j;
            }

            if ((j > 6) && (name.substr(j - 7, 8) == "operator")) { // found at least operator<
               j = i + 1;

               if (j < len) { // check for operator<=, operator<<, or operator<<=
                  if (name[j] == '=') { // operator<=
                     i = j;
                  } else if (name[j] == '<') { // we have operator<<, or operator<<=
                     i = j;
                     ++j;

                     if (j < len) {
                        if (name[j] == '=') { // we have operator<<=
                           i = j;
                        }
                     }
                  }
               }
               continue; // skip changing depth
            }
         }
         ++bracket_depth;
      } else if (c == '>') { // check for operator>, operator>=, operator>>, operator>>=, operator->, or operator->*
         if (i > 7) { // there is room for "operator"
            long j = i - 1;
            bool have_arrow = false;

            if (name[j] == '-') { // allow for operator->, or operator->*
               have_arrow = true;
               --j;
            }

            while (j && isspace(name[j])) {
               --j;
            }

            if ((j > 6) && (name.substr(j - 7, 8) == "operator")) { // found at least operator> or operator->
               j = i + 1;

               if (j < len) { // check for operator->*, operator>=, operator>>, or operator>>=
                  if (have_arrow && (name[j] == '*')) { // we have operator->*
                     i = j;
                  } else if (!have_arrow) {
                     if (name[j] == '=') { // we have operator>=
                        i = j;
                     } else if (name[j] == '>') { // we have operator>>, or operator>>=
                        i = j;
                        ++j;

                        if (j < len) {
                           if (name[j] == '=') { // we have operator>>=
                              i = j;
                           }
                        }
                     }
                  }
               }
               continue; // skip changing depth
            }
         }
         --bracket_depth;
      } else if (!paren_depth && !bracket_depth){
         if (isspace(c)) {
            start = i + 1;
            scopePos = std::string::npos;
         } else if (scopePos == std::string::npos
                    && (c == ':') && ((i + 1) < len) && (name[i + 1] == ':')) {
            scopePos = i + 2;
         }
      }
   }
   if (scopePos == std::string::npos)
      scopePos = 0;

   return scopePos;
} // GetFirstScopePosition


//-------------------------------------------------------------------------------
bool
Tools::IsTemplated(const char* name) {
   // Check if the final component of a qualified-id has template arguments.
   //
   // Note:  We must be careful of:
   //
   //       operator<,   operator>,
   //       operator<=,  operator>=,
   //       operator<<,  operator>>,
   //       operator<<=, operator>>=
   //       operator->,  operator->*,
   //       operator()
   //
   long pos = GetBasePosition(std::string(name));
   int bracket_depth = 0;
   int paren_depth = 0;
   long len = std::strlen(name);

   for (long i = pos; i < len; ++i) {
      char c = name[i];

      if (c == '(') { // check for operator()
         if (i > 7) { // there is room for "operator"
            long j = i - 1;

            while (j && isspace(name[j])) {
               --j;
            }

            if ((j > 6) && !strncmp(name + j - 7, "operator", 8)) { // possibly found operator()
               j = i + 1;

               while ((j < len) && isspace(name[j])) {
                  ++j;
               }

               if (j < len) {
                  if (name[j] == ')') {
                     i = j;
                     continue; // skip changing depth
                  }
               }
            }
         }
         ++paren_depth;
      } else if (c == ')') {
         --paren_depth;
      } else if (c == '<') { // check for operator<, operator<=, operator<<, and operator<<=
         if (i > 7) { // there is room for "operator"
            long j = i - 1;

            while (j && isspace(name[j])) {
               --j;
            }

            if ((j > 6) && !strncmp(name + j - 7, "operator", 8)) { // found at least operator<
               j = i + 1;

               if (j < len) { // check for operator<=, operator<<, or operator<<=
                  if (name[j] == '=') { // operator<=
                     i = j;
                  } else if (name[j] == '<') { // we have operator<<, or operator<<=
                     i = j;
                     ++j;

                     if (j < len) {
                        if (name[j] == '=') { // we have operator<<=
                           i = j;
                        }
                     }
                  }
               }
               continue; // skip changing depth
            }
         }

         if (!paren_depth && !bracket_depth) {
            return true; // We found the opening '<' of a set of template arguments.
         }
         ++bracket_depth;
      } else if (c == '>') { // check for operator>, operator>=, operator>>, operator>>=, operator->, or operator->*
         if (i > 7) { // there is room for "operator"
            long j = i - 1;
            bool have_arrow = false;

            if (name[j] == '-') { // allow for operator->, or operator->*
               have_arrow = true;
               --j;
            }

            while (j && isspace(name[j])) {
               --j;
            }

            if ((j > 6) && !strncmp(name + j - 7, "operator", 8)) { // found at least operator> or operator->
               j = i + 1;

               if (j < len) { // check for operator->*, operator>=, operator>>, or operator>>=
                  if (have_arrow && (name[j] == '*')) { // we have operator->*
                     i = j;
                  } else if (!have_arrow) {
                     if (name[j] == '=') { // we have operator>=
                        i = j;
                     } else if (name[j] == '>') { // we have operator>>, or operator>>=
                        i = j;
                        ++j;

                        if (j < len) {
                           if (name[j] == '=') { // we have operator>>=
                              i = j;
                           }
                        }
                     }
                  }
               }
               continue; // skip changing depth
            }
         }
         --bracket_depth;
      }
   }
   return false;
} // IsTemplated


//-------------------------------------------------------------------------------
void
Tools::StringSplit(std::vector<std::string>& splitValues,
                   const std::string& str,
                   const std::string& delim) {
//-------------------------------------------------------------------------------
// Split a string by a delimiter and return it's vector of strings.
   if (!str.size()) {
      return;
   }

   std::string str2 = str;

   size_t pos = 0;

   while ((pos = str2.find_first_of(delim)) != std::string::npos) {
      std::string s = str2.substr(0, pos);
      StringStrip(s);
      splitValues.push_back(s);
      str2 = str2.substr(pos + delim.length());
   }

   StringStrip(str2);
   splitValues.push_back(str2);
} // StringSplit


//-------------------------------------------------------------------------------
std::string
Tools::StringVec2String(const std::vector<std::string>& vec) {
//-------------------------------------------------------------------------------
   std::string s = "";
   StdString_Iterator lastbutone = vec.end() - 1;

   for (StdString_Iterator it = vec.begin(); it != vec.end(); ++it) {
      s += *it;

      if (it != lastbutone) {
         s += ", ";
      }
   }
   return s;
}


//-------------------------------------------------------------------------------
std::string
Tools::Demangle(const std::type_info& ti) {
//-------------------------------------------------------------------------------
// Demangle a type_info object.
#if defined(_WIN32)
   static std::vector<std::string> keywords;

   if (0 == keywords.size()) {
      keywords.push_back("class ");
      keywords.push_back("struct ");
      keywords.push_back("enum ");
      keywords.push_back("union ");
      keywords.push_back("__cdecl");
   }
   std::string r = ti.name();

   for (size_t i = 0; i < keywords.size(); i++) {
      while (r.find(keywords[i]) != std::string::npos)
         r = r.replace(r.find(keywords[i]), keywords[i].size(), "");

      while (r.find(" *") != std::string::npos)
         r = r.replace(r.find(" *"), 2, "*");

      while (r.find(" &") != std::string::npos)
         r = r.replace(r.find(" &"), 2, "&");
   }
   return r;

#elif defined(__GNUC__)

   int status = 0;
#define RFLX_REMOVE_ADDITIONAL_POINTER false
   std::string mangled = ti.name();

   // if the At Name is string return the final string Name
   // abi::Demangle would return "std::string" instead
   if (mangled == "Ss") {
      return "std::basic_string<char>";
   }

# if __GNUC__ <= 3 && __GNUC_MINOR__ <= 3

   // Function types are not decoded at all. We are an extra 'P' to convert it to a pointer
   // and remove it at the end.
   if (mangled[0] == 'F') {
      mangled.insert(0, "P");
#undef RFLX_REMOVE_ADDITIONAL_POINTER
#define RFLX_REMOVE_ADDITIONAL_POINTER true
   }
# elif __GNUC__ >= 4

   // From gcc 4.0 on the fundamental types are not demangled anymore by the dynamic demangler
   if (mangled.length() == 1) {
      switch (mangled[0]) {
      case 'a': return "signed char";        break;
      case 'b': return "bool";               break;
      case 'c': return "char";               break;
      case 'd': return "double";             break;
      case 'e': return "long double";        break;
      case 'f': return "float";              break;
      case 'g': return "__float128";         break;
      case 'h': return "unsigned char";      break;
      case 'i': return "int";                break;
      case 'j': return "unsigned int";       break;
      //case 'k': return "";                   break;
      case 'l': return "long";               break;
      case 'm': return "unsigned long";      break;
      case 'n': return "__int128";           break;
      case 'o': return "unsigned __int128";  break;
      //case 'p': return "";                   break;
      //case 'q': return "";                   break;
      //case 'r': return "";                   break;
      case 's': return "short";              break;
      case 't': return "unsigned short";     break;
      //case 'u': return "";                   break;
      case 'v': return "void";               break;
      case 'w': return "wchar_t";            break;
      case 'x': return "long long";          break;
      case 'y': return "unsigned long long"; break;
      case 'z': return "...";                break;
      default:                               break;
      } // switch
   }
# endif
   char* c_demangled = abi::__cxa_demangle(mangled.c_str(), 0, 0, &status);

   if (status == -1) {
      throw RuntimeError("Memory allocation failure while demangling ");
   } else if (status == -2) {
      throw RuntimeError(std::string(mangled) + " is not a valid Name under the C++ ABI");
   } else if (status == -3) {
      throw RuntimeError(std::string("Failure while demangling ") + mangled +
                         ". One of the arguments is invalid ");
   } else {
      std::string demangled = c_demangled;
      free(c_demangled);

      if (RFLX_REMOVE_ADDITIONAL_POINTER) {
         demangled = demangled.replace(demangled.find("(*)"), 3, "");
      }

      while (demangled.find(", ") != std::string::npos) {
         demangled = demangled.replace(demangled.find(", "), 2, ",");
      }
      return demangled;
   }

#elif defined(__SUNPRO_CC)

   const char* mangled = ti.name();
   size_t buffer = 1024;
   char* c_demangled = new char[buffer];
   int ret = cplus_demangle(mangled, c_demangled, buffer);

   while (ret == -1) {
      buffer = buffer * 2;
      delete[] c_demangled;
      c_demangled = new char[buffer];
      ret = cplus_demangle(mangled, c_demangled, buffer);
   }

   if (ret == 1) {
      throw RuntimeError(std::string("Symbol ") + mangled + " not mangled correctly");
   } else {
      std::string demangled = Tools::NormalizeName(c_demangled);
      delete[] c_demangled;
      return demangled;
   }

#elif defined(__IBMCPP__)

   return Tools::NormalizeName(ti.name());

#endif
   return "";
} // Demangle


//-------------------------------------------------------------------------------
void
Tools::StringSplitPair(std::string& val1,
                       std::string& val2,
                       const std::string& str,
                       const std::string& delim) {
//-------------------------------------------------------------------------------
// Split a string by a delimiter into a pair and return them as val1 and val2.
   std::string str2 = str;
   size_t pos = str2.rfind(delim);

   if (pos != std::string::npos) {
      val1 = str2.substr(0, pos);
      val2 = str2.substr(pos + delim.length());
   } else {
      val1 = str2;
   }
   StringStrip(val1);
   StringStrip(val2);
}


//-------------------------------------------------------------------------------
void
Tools::StringStrip(std::string& str) {
//-------------------------------------------------------------------------------
// Strip spaces at the beginning and the end from a string.
   if (str.empty()) {
      return;
   }
   size_t sPos = 0;
   size_t ePos = str.length() - 1;

   while (ePos >= sPos && str[sPos] == ' ') {
      ++sPos;
   }

   while (ePos > sPos && str[ePos] == ' ') {
      --ePos;
   }

   if (ePos >= sPos) {
      str = str.substr(sPos, ePos + 1 - sPos);
   } else {
      str.clear(); // all spaces
   }
} // StringStrip


//-------------------------------------------------------------------------------
std::string
Tools::GetTemplateArguments(const char* name) {
   // Return the template arguments part of a templated type name.
   //
   // Note:  We must be careful of:
   //
   //       operator<,   operator>,
   //       operator<=,  operator>=,
   //       operator<<,  operator>>,
   //       operator<<=, operator>>=
   //       operator->,  operator->*,
   //       operator()
   //
   long pos = GetBasePosition(std::string(name));
   int bracket_depth = 0;
   int paren_depth = 0;
   long len = std::strlen(name);

   for (long i = pos; i < len; ++i) {
      char c = name[i];

      if (c == '(') { // check for operator()
         if (i > 7) { // there is room for "operator"
            long j = i - 1;

            while (j && isspace(name[j])) {
               --j;
            }

            if ((j > 6) && !strncmp(name + j - 7, "operator", 8)) { // possibly found operator()
               j = i + 1;

               while ((j < len) && isspace(name[j])) {
                  ++j;
               }

               if (j < len) {
                  if (name[j] == ')') {
                     i = j;
                     continue; // skip changing depth
                  }
               }
            }
         }
         ++paren_depth;
      } else if (c == ')') {
         --paren_depth;
      } else if (c == '<') { // check for operator<, operator<=, operator<<, and operator<<=
         if (i > 7) { // there is room for "operator"
            long j = i - 1;

            while (j && isspace(name[j])) {
               --j;
            }

            if ((j > 6) && !strncmp(name + j - 7, "operator", 8)) { // found at least operator<
               j = i + 1;

               if (j < len) { // check for operator<=, operator<<, or operator<<=
                  if (name[j] == '=') { // operator<=
                     i = j;
                  } else if (name[j] == '<') { // we have operator<<, or operator<<=
                     i = j;
                     ++j;

                     if (j < len) {
                        if (name[j] == '=') { // we have operator<<=
                           i = j;
                        }
                     }
                  }
               }
               continue; // skip changing depth
            }
         }

         if (!paren_depth && !bracket_depth) { // We found the opening '<' of a set of template arguments.
            return name + i;
         }
         ++bracket_depth;
      } else if (c == '>') { // check for operator>, operator>=, operator>>, operator>>=, operator->, or operator->*
         if (i > 7) { // there is room for "operator"
            long j = i - 1;
            bool have_arrow = false;

            if (name[j] == '-') { // allow for operator->, or operator->*
               have_arrow = true;
               --j;
            }

            while (j && isspace(name[j])) {
               --j;
            }

            if ((j > 6) && !strncmp(name + j - 7, "operator", 8)) { // found at least operator> or operator->
               j = i + 1;

               if (j < len) { // check for operator->*, operator>=, operator>>, or operator>>=
                  if (have_arrow && (name[j] == '*')) { // we have operator->*
                     i = j;
                  } else if (!have_arrow) {
                     if (name[j] == '=') { // we have operator>=
                        i = j;
                     } else if (name[j] == '>') { // we have operator>>, or operator>>=
                        i = j;
                        ++j;

                        if (j < len) {
                           if (name[j] == '=') { // we have operator>>=
                              i = j;
                           }
                        }
                     }
                  }
               }
               continue; // skip changing depth
            }
         }
         --bracket_depth;
      }
   }
   return std::string();
} // GetTemplateArguments


//-------------------------------------------------------------------------------
std::string
Tools::GetTemplateName(const char* name) {
   // Return the fully qualified scope name without template arguments.
   //
   // Note:  We must be careful of:
   //
   //       operator<,   operator>,
   //       operator<=,  operator>=,
   //       operator<<,  operator>>,
   //       operator<<=, operator>>=
   //       operator->,  operator->*,
   //       operator()
   //
   long base_pos = GetBasePosition(std::string(name));
   int bracket_depth = 0;
   int paren_depth = 0;
   long len = std::strlen(name);

   for (long i = base_pos; i < len; ++i) {
      char c = name[i];

      if (c == '(') { // check for operator()
         if (i > 7) { // there is room for "operator"
            long j = i - 1;

            while (j && isspace(name[j])) {
               --j;
            }

            if ((j > 6) && !strncmp(name + j - 7, "operator", 8)) { // possibly found operator()
               j = i + 1;

               while ((j < len) && isspace(name[j])) {
                  ++j;
               }

               if (j < len) {
                  if (name[j] == ')') {
                     i = j;
                     continue; // skip changing depth
                  }
               }
            }
         }
         ++paren_depth;
      } else if (c == ')') {
         --paren_depth;
      } else if (c == '<') { // check for operator<, operator<=, operator<<, and operator<<=
         if (i > 7) { // there is room for "operator"
            long j = i - 1;

            while (j && isspace(name[j])) {
               --j;
            }

            if ((j > 6) && !strncmp(name + j - 7, "operator", 8)) { // found at least operator<
               j = i + 1;

               if (j < len) { // check for operator<=, operator<<, or operator<<=
                  if (name[j] == '=') { // operator<=
                     i = j;
                  } else if (name[j] == '<') { // we have operator<<, or operator<<=
                     i = j;
                     ++j;

                     if (j < len) {
                        if (name[j] == '=') { // we have operator<<=
                           i = j;
                        }
                     }
                  }
               }
               continue; // skip changing depth
            }
         }

         if (!paren_depth && !bracket_depth) { // We found the opening '<' of a set of template arguments.
            // Remove any trailing spaces, we might be operator<< <int>
            long j = i - 1;

            while ((j >= base_pos) && isspace(name[j])) {
               --j;
            }
            return std::string(name, j + 1);
         }
         ++bracket_depth;
      } else if (c == '>') { // check for operator>, operator>=, operator>>, operator>>=, operator->, or operator->*
         if (i > 7) { // there is room for "operator"
            long j = i - 1;
            bool have_arrow = false;

            if (name[j] == '-') { // allow for operator->, or operator->*
               have_arrow = true;
               --j;
            }

            while (j && isspace(name[j])) {
               --j;
            }

            if ((j > 6) && !strncmp(name + j - 7, "operator", 8)) { // found at least operator> or operator->
               j = i + 1;

               if (j < len) { // check for operator->*, operator>=, operator>>, or operator>>=
                  if (have_arrow && (name[j] == '*')) { // we have operator->*
                     i = j;
                  } else if (!have_arrow) {
                     if (name[j] == '=') { // we have operator>=
                        i = j;
                     } else if (name[j] == '>') { // we have operator>>, or operator>>=
                        i = j;
                        ++j;

                        if (j < len) {
                           if (name[j] == '=') { // we have operator>>=
                              i = j;
                           }
                        }
                     }
                  }
               }
               continue; // skip changing depth
            }
         }
         --bracket_depth;
      }
   }
   return name;
} // GetTemplateName


//-------------------------------------------------------------------------------
bool
isalphanum(int i) {
//-------------------------------------------------------------------------------
// Return true if char is alpha or digit.
   return isalpha(i) || isdigit(i);
}


//-------------------------------------------------------------------------------
std::string
Tools::NormalizeName(const char* nam) {
//-------------------------------------------------------------------------------
// Normalize a type name.
   std::string norm_name;
   char prev = 0;

   for (size_t i = 0; nam[i] != 0; i++) {
      char curr = nam[i];

      if (curr == ' ') {
         char next = 0;

         while (nam[i] != 0 && (next = nam[i + 1]) == ' ') {
            ++i;
         }

         if (!isalphanum(prev) || !isalpha(next)) {
            continue; // continue on non-word boundaries
         }
      } else if ((curr == '>' && prev == '>') || (curr == '(' && prev != ')')) {
         norm_name += ' ';
      }
      norm_name += (prev = curr);
   }

   return norm_name;
} // NormalizeName


//-------------------------------------------------------------------------------
std::string
Tools::NormalizeName(const std::string& nam) {
//-------------------------------------------------------------------------------
   return Tools::NormalizeName(nam.c_str());
}
