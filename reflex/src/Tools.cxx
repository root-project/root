// @(#)root/reflex:$Name:  $:$Id: Tools.cxx,v 1.12 2006/05/31 22:03:52 roiser Exp $
// Author: Stefan Roiser 2004

// Copyright CERN, CH-1211 Geneva 23, 2004-2006, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#ifndef REFLEX_BUILD
#define REFLEX_BUILD
#endif

#include "Reflex/Tools.h"

#include "Reflex/Kernel.h"
#include "Reflex/Type.h"
#include "Reflex/Member.h"

#if defined(__GNUC__)
#include <cxxabi.h>
#elif defined(__SUNPRO_CC)
#include <demangle.h>
#endif


//-------------------------------------------------------------------------------
static std::string splitScopedName( const std::string & Name, 
                                    bool returnScope ) {
//-------------------------------------------------------------------------------
   size_t pos = ROOT::Reflex::Tools::GetBasePosition( Name ); 
   if ( pos != 0 ) {
      if ( returnScope )  return Name.substr(0,pos-2);
      else                return Name.substr(pos);
   }
   else {
      if ( returnScope )  return "";
      else                return Name;
   }
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::Tools::BuildTypeName( Type & t,
                                                unsigned int /* modifiers */ ) {
//-------------------------------------------------------------------------------
   std::string mod = "";
   if ( t.IsConstVolatile()) mod = "const volatile";
   else if ( t.IsConst())    mod = "const";
   else if ( t.IsVolatile()) mod = "volatile";

   std::string name = t.Name();

   if (t.IsPointer() || t.IsPointerToMember()) name += " " + mod;
   else                                        name = mod + " " + name;

   if ( t.IsReference()) name += "&";

   return name;
}


//-------------------------------------------------------------------------------
std::vector<std::string> ROOT::Reflex::Tools::GenTemplateArgVec( const std::string & Name ) {
//-------------------------------------------------------------------------------
   std::string tpl = "";
   std::vector<std::string> tl = std::vector<std::string>();
   unsigned long int pos1 = Name.find("<");
   unsigned long int pos2 = Name.rfind(">");
   if ( pos1 == std::string::npos && pos2 == std::string::npos ) return tl; 
   else tpl = Name.substr(pos1+1, pos2-pos1-1);
   if (tpl[tpl.size()-1] == ' ') {
      tpl = tpl.substr(0,tpl.size()-1);
   }
   unsigned int par = 0;
   std::string argName = "";
   for (unsigned int i = 0; i < tpl.length(); ++i) {
      char c = tpl[i];
      if ( c == ',' ) {
         if ( ! par ) {
            StringStrip(argName);
            tl.push_back(argName);
            argName = "";
         }
         else {
            argName += c;
         }
      }
      else {
         argName += c;
         if ( c == '<' ) ++par;
         else if ( c == '>' ) --par;
      }
   }
   if ( argName.length() ) {
      StringStrip(argName);
      tl.push_back(argName);
   }
   return tl;
}


//-------------------------------------------------------------------------------
size_t ROOT::Reflex::Tools::GetBasePosition( const std::string & name ) {
//-------------------------------------------------------------------------------
   // remove the template part of the name <...>
   int ab = 0;
   int rb = 0;
   int i = 0;
   size_t pos = 0;
   for ( i = name.size()-1; i >= 0; --i) {
      switch (name[i]) {
      case '>' : ab++; break;
      case '<' : ab--; break;
      case ')' : rb++; break;
      case '(' : rb--; break;
      case ':' : 
         if ( ab == 0 && rb == 0 && name[i-1] == ':' ) {
            pos = i + 1;
            break; 
         }
      default: continue;
      }
      if ( pos ) break;
   }
   return pos;
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::Tools::GetScopeName( const std::string & name ){
//-------------------------------------------------------------------------------
   return splitScopedName( name, true );
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::Tools::GetBaseName( const std::string & name ) {
//-------------------------------------------------------------------------------
   return splitScopedName( name, false );
}


//-------------------------------------------------------------------------------
bool ROOT::Reflex::Tools::IsTemplated(const char * name ) {
//-------------------------------------------------------------------------------
   for (size_t i = strlen(name)-1; i > 0; --i) {
      if (( name[i] == '>' ) && ( strchr(name,'<') != 0 )) return true;
      else if ( name[i] == ' ') break;
      else return false;
   }
   return false;
   /* alpha 
      size_t i = strlen(name)-1;
      while (name[i] == ' ') --i;
      if (( name[i] == '>' ) && ( strchr(name,'<') != 0 )) return true;
      return false;
   */
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Tools::StringSplit( std::vector < std::string > & splitValues, 
                                       const std::string & str,
                                       const std::string & delim ) {
//-------------------------------------------------------------------------------
   std::string str2 = str;
  
   size_t pos = 0;
  
   while (( pos = str2.find_first_of( delim )) != std::string::npos ) {
      std::string s = str2.substr(0, pos);
      StringStrip( s );
      if ( s.length()) { splitValues.push_back( s ); }
      str2 = str2.substr( pos + delim.length());
   }
  
   StringStrip( str2 );
   if ( str2.length()) { splitValues.push_back( str2 ); }
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::Tools::Demangle( const std::type_info & ti ) { 
//-------------------------------------------------------------------------------
#if defined(_WIN32)
   static std::vector<std::string> keywords;
   if ( 0 == keywords.size() ) {
      keywords.push_back("class ");
      keywords.push_back("struct ");
      keywords.push_back("enum ");
      keywords.push_back("union ");
      keywords.push_back("__cdecl");
   }
   std::string r = ti.name();
   for ( size_t i = 0; i < keywords.size(); i ++ ) {
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
   bool remove_additional_pointer = false;
   std::string  mangled = ti.name();

   // if the At Name is string return the final string Name 
   // abi::Demangle would return "std::string" instead
   if ( mangled == "Ss" ) return "std::basic_string<char>";

#if __GNUC__ <= 3 && __GNUC_MINOR__ <= 3
   // Function types are not decoded at all. We are an extra 'P' to convert it to a pointer
   // and remove it at the end.
   if ( mangled[0] == 'F' ) {
      mangled.insert(0,"P");
      remove_additional_pointer = true;
   }
#elif __GNUC__ >= 4
   // From gcc 4.0 on the fundamental types are not demangled anymore by the dynamic demangler
   if (mangled.length() == 1) {
      switch ( mangled[0] ) {
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
      }
   }
#endif
   char * c_demangled = abi::__cxa_demangle( mangled.c_str(), 0, 0, & status );
   if ( status == -1 ) {
      throw RuntimeError("Memory allocation failure while demangling ");
   }
   else if ( status == -2 ) {
      throw RuntimeError( std::string(mangled) + " is not a valid Name under the C++ ABI");
   }
   else if ( status == -3 ) {
      throw RuntimeError( std::string("Failure while demangling ") + mangled +
                          ". One of the arguments is invalid ");
   }
   else {
      std::string demangled = c_demangled;
      free( c_demangled );
      if ( remove_additional_pointer ) {
         demangled = demangled.replace(demangled.find("(*)"), 3, "");
      }
      while ( demangled.find(", ") != std::string::npos ) {
         demangled = demangled.replace(demangled.find(", "), 2, ",");
      }
      return demangled;
   }

#elif defined(__SUNPRO_CC)

   std::string mangled = ti.name();
   size_t buffer = 1024;
   char * c_demangled = new char(buffer);
   int ret = cplus_demangle( mangled.c_str(), c_demangled, buffer);
   while ( ret == -1 ) {
      buffer = buffer*2;
      delete c_demangled;
      c_demangled = new char(buffer);
      ret = cplus_demangle( mangled.c_str(), c_demangled, buffer);
   }
   if ( ret == 1 ) {
      throw RuntimeError(std::string("Symbol ") + mangled + " not mangled correctly");
   }
   else {
      std::string demangled = c_demangled;
      delete c_demangled;
      return demangled;
   }

#endif
   return "";
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Tools::StringSplitPair( std::string & val1,
                                           std::string & val2,
                                           const std::string & str,
                                           const std::string & delim ) { 
//-------------------------------------------------------------------------------
   std::string str2 = str;
   size_t pos = str2.rfind( delim );
   if ( pos != std::string::npos ) { 
      val1 = str2.substr( 0, pos ); 
      val2 = str2.substr( pos + delim.length());
   }
   else { 
      val1 = str2; 
   }
   StringStrip( val1 );
   StringStrip( val2 );
}


//-------------------------------------------------------------------------------
void ROOT::Reflex::Tools::StringStrip( std::string & str ) {
//-------------------------------------------------------------------------------
   size_t sPos = 0;
   size_t ePos = str.length();
   while ( str[sPos] == ' ' ) { ++sPos; }
   while ( str[ePos] == ' ' ) { --ePos; }
   str = str.substr( sPos, ePos - sPos );
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::Tools::GetTemplateArguments( const char * name ) {
//-------------------------------------------------------------------------------
   std::string baseName = GetBaseName(name);
   return baseName.substr(baseName.find('<'));
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::Tools::GetTemplateName( const char * name ) {
//-------------------------------------------------------------------------------
   std::string scopeName = GetScopeName( name );
   std::string baseName = GetBaseName( name );
   std::string templateName = baseName.substr(0, baseName.find('<'));
   if ( scopeName.length()) return scopeName + "::" + templateName;
  
   return templateName;
}


bool isalphanum(int i) {
   return isalpha(i) || isdigit(i);
}


//-------------------------------------------------------------------------------
std::string ROOT::Reflex::Tools::NormalizeName( const std::string & nam ) {
//-------------------------------------------------------------------------------

   std::string norm_name = "";
   //char lchar = ' ';
   unsigned int nlen = nam.length();
   unsigned int par = 0;
   bool first_word = true;
   char cprev = ' ';
   char cnext = ' ';

   for (unsigned int i = 0; i < nlen; ++i ) {

      switch (nam[i]) {

      case ' ':
         // are we at the beginning of the name
         //bool fist_pos = i ? false : true;

         // consume all spaces
         while (i < nlen && nam[i+1] == ' ') ++i;
      
         // if only white spaces at the beginning break or
         // spaces at the end of the name then break
         if ( first_word || ( i = nlen -1) ) break;

         // if the last pos and the next pos is char 
         // or we are at the closing of angular braces 
         // then insert a space and the next position
         cprev = nam[i-1];
         cnext = nam[i+1];
         if ( ( isalphanum(cprev) &&  isalpha(cnext) ) ||
              ( cprev == '>' && cnext == '>' ) ) norm_name += ' ' + nam[i+1];
         // otherwise only add the next pos
         else norm_name += nam[i+1];
         // increase counter by consumed pos
         ++i;
         break;

      case '(':
      case '<':
         ++par;
         norm_name += nam[i];
         ++i;
         break;

      case ')':
      case '>':
         --par;
         norm_name += nam[i];
         ++i;
         break;

      case '*':
      case '&':
         first_word = false;
         norm_name += nam[i];
         ++i;
         break;

      case 'c':
         norm_name += nam[i];
         ++i;
         break;

      case 'v':
         norm_name += nam[i];
         ++i;
         break;

      default:
         norm_name += nam[i];
         ++i;
         break;

      }

   }

   return norm_name;
}
