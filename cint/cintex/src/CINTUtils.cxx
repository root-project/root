// @(#)root/cintex:$Id$
// Author: Pere Mato 2005

// Copyright CERN, CH-1211 Geneva 23, 2004-2005, All rights reserved.
//
// Permission to use, copy, modify, and distribute this software for any
// purpose is hereby granted without fee, provided that this copyright and
// permissions notice appear in all copies and derivatives.
//
// This software is provided "as is" without express or implied warranty.

#include "Reflex/Reflex.h"
#include "Reflex/Type.h"
#include "CINTdefs.h"
#include "Api.h"


using namespace ROOT::Reflex;
using namespace std;

namespace ROOT { namespace Cintex {

   Indirection IndirectionGet(const Type& typ) {
      Type t(typ);
      int indir = 0;
      for ( ; t.IsTypedef(); t = t.ToType()) ;
      for ( ; t.IsPointer(); t = t.ToType()) indir++; 
      return Indirection(indir, t);
   }

   void CintType( const Type& typ, int& typenum, int& tagnum ) {
      Type t(typ);
      int indir = 0;
      while( t.IsTypedef()) t = t.ToType();
      for ( ; t.IsPointer(); t = t.ToType()) indir++; 

      CintTypeDesc dsc = CintType( t );
      

      typenum  = dsc.first + (indir > 0 ? 'A'-'a' : 0);
      tagnum   = -1;
      if ( dsc.first == 'u' ) {
         tagnum = ::G__defined_tagname(dsc.second.c_str(), 2);
         if ( tagnum == -1 ) {
            G__linked_taginfo taginfo;
            taginfo.tagnum  = -1;
            if ( t.IsClass() || t.IsStruct() ) taginfo.tagtype = 'c';
            else                               taginfo.tagtype = 'a';
            taginfo.tagname = dsc.second.c_str();
            G__get_linked_tagnum(&taginfo);
            tagnum = taginfo.tagnum;
         }
      }
   }

   CintTypeDesc CintType(const ROOT::Reflex::Type& typ)  {
      Type t(CleanType(typ));
      string nam = t.Name(SCOPED);

      if ( nam == "void" )
         return CintTypeDesc('y', "-");
      else if ( nam == "bool" )
         return CintTypeDesc('g', "-");
      else if ( nam == "char" )
         return CintTypeDesc('c', "-");
      else if ( nam == "signed char" )
         return CintTypeDesc('c', "-");
      else if ( nam == "unsigned char" )
         return CintTypeDesc('b', "-");
      else if ( nam == "short" )
         return CintTypeDesc('s', "-");
      else if ( nam == "short int" )
         return CintTypeDesc('s', "-");
      else if ( nam == "signed short" )
         return CintTypeDesc('s', "-");
      else if ( nam == "signed short int" )
         return CintTypeDesc('s', "-");
      else if ( nam == "unsigned short" )
         return CintTypeDesc('r', "-");
      else if ( nam == "short unsigned" )
         return CintTypeDesc('r', "-");
      else if ( nam == "unsigned short int" )
         return CintTypeDesc('r', "-");
      else if ( nam == "short unsigned int" )
         return CintTypeDesc('r', "-");
      else if ( nam == "int" )
         return CintTypeDesc('i', "-");
      else if ( nam == "signed" )
         return CintTypeDesc('i', "-");
      else if ( nam == "signed int" )
         return CintTypeDesc('i', "-");
      else if ( nam == "unsigned int" )
         return CintTypeDesc('h', "-");
      else if ( nam == "unsigned" )
         return CintTypeDesc('h', "-");
      else if ( nam == "long" )
         return CintTypeDesc('l', "-");
      else if ( nam == "long int" )
         return CintTypeDesc('l', "-");
      else if ( nam == "signed long" )
         return CintTypeDesc('l', "-");
      else if ( nam == "signed long int" )
         return CintTypeDesc('l', "-");
      else if ( nam == "long signed" )
         return CintTypeDesc('l', "-");
      else if ( nam == "long signed int" )
         return CintTypeDesc('l', "-");
      else if ( nam == "unsigned long" )
         return CintTypeDesc('k', "-");
      else if ( nam == "unsigned long int" )
         return CintTypeDesc('k', "-");
      else if ( nam == "long unsigned int" )
         return CintTypeDesc('k', "-");
      else if ( nam == "longlong" )
         return CintTypeDesc('n', "-");
      else if ( nam == "long long" )
         return CintTypeDesc('n', "-");
      else if ( nam == "long long int" )
         return CintTypeDesc('n', "-");
      else if ( nam == "long long signed int" )
         return CintTypeDesc('n', "-");
      else if ( nam == "long long signed" )
         return CintTypeDesc('n', "-");
      else if ( nam == "signed long long int" )
         return CintTypeDesc('n', "-");
      else if ( nam == "ulonglong" )
         return CintTypeDesc('m', "-");
      else if ( nam == "unsigned long long" )
         return CintTypeDesc('m', "-");
      else if ( nam == "long long unsigned" )
         return CintTypeDesc('m', "-");
      else if ( nam == "long long unsigned int" )
         return CintTypeDesc('m', "-");
      else if ( nam == "unsigned long long int" )
         return CintTypeDesc('m', "-");
      else if ( nam == "long double" )
         return CintTypeDesc('q', "-");
      else if ( nam == "double" )
         return CintTypeDesc('d', "-");
      else if ( nam == "double32" )
         return CintTypeDesc('d', "-");
      else if ( nam == "float" )
         return CintTypeDesc('f', "-");
      else if ( t.IsEnum() )
         return CintTypeDesc('i', "-");
      else if ( t.IsFunction() )
         return CintTypeDesc('y', "-");
      else if ( !t.IsFundamental() ) 
         return CintTypeDesc('u', CintName(t));
      else 
         return CintTypeDesc('-', CintName(t)); 
   }

   Type CleanType(const Type& typ)  {
      if ( typ )  {
         Type t( typ );
         while ( t.IsTypedef()   ) t = CleanType(t.ToType());
         while ( t.IsPointer()   ) t = CleanType(t.ToType());
         while ( t.IsArray()     ) t = CleanType(t.ToType());
         return t;
      }
      return typ;
   }

   bool IsTypeOf(Type& typ, const std::string& base_name)  {
      Type ref_typ = Type::ByName(base_name);
      if ( ref_typ )  {
         if ( typ == ref_typ || typ.HasBase(ref_typ) )  {
            return true;
         }
         return false;
      }
      return false;
      //throw std::runtime_error("Fatal error: Failed to retrieved dictionary for "+base_name);
   }

   bool IsSTLinternal(const std::string& nam)  {
      std::string sub = nam.substr(0,8); 
      std::string sub6 = nam.substr(0,6); 
      std::string sub9 = nam.substr(0,9); 
      if ( nam.empty() || sub6 =="std::_" || sub9 =="stdext::_" || nam.substr(0,12)=="__gnu_cxx::_")  {
         return true;
      }
      return false;
   }

   bool IsSTL(const std::string& nam)  {
      if ( ! IsSTLinternal(nam) )  {
         std::string sub = nam.substr(0,8); 
         bool stl = nam.substr(0,17)=="std::basic_string" ||
            sub=="std::str" || sub=="std::vec" || sub=="std::lis" ||
            sub=="std::set" || sub=="std::deq" || sub=="std::map" ||
            sub=="std::mul" || sub=="stdext::" || sub=="__gnu_cx";
         return stl;
      }
      return true;
   }
   bool IsSTLext(const std::string& nam)  {
      std::string sub = nam.substr(0,8); 
      return sub=="stdext::" || sub=="__gnu_cx";
   }

   // Conversion table: Note: order is important!
   static const char* s_normalize[][2] =  {
      {"  ",                     " " }
      ,{", ",                     "," }
      ,{" signed ",               " " }
      ,{",signed ",               "," }
      ,{"<signed ",               "<" }
      ,{"(signed ",               "(" }
      ,{"ulonglong",              "unsigned long long" }
      ,{"longlong",               "long long" }
      ,{"long unsigned",          "unsigned long" }
      ,{"short unsigned",         "unsigned short" }
      ,{"short int",              "short" }
      ,{"long int",               "long" }
      ,{"basic_string<char> ",    "string"}
      ,{"basic_string<char>",     "string"}
      ,{"basic_string<char,allocator<char> > ","string"}
      ,{"basic_string<char,allocator<char> >","string"}
      ,{"basic_string<char,char_traits<char>,allocator<char> > ","string"}
      ,{"basic_string<char,char_traits<char>,allocator<char> >","string"}
   };

   std::string CintName(const Type& typ)  {
      Type t(CleanType(typ));
      return CintName(t.Name(SCOPED));
   }
   std::string CintName(const std::string& full_nam)  {
      // was: else  {
      {
         size_t occ;
         std::string nam = full_nam; //(full) ? full_nam : cl->Name();
         std::string s = (nam.substr(0,2) == "::") ? nam.substr(2) : nam;

         /// Completely ignore namespace std
         while ( (occ=s.find("std::")) != std::string::npos )    {
            s.replace(occ, 5, "");
         }
         /// Completely ignore namespace std
         while ( (occ=s.find(", ")) != std::string::npos )    {
            s.replace(occ, 2, ",");
         }
         /// remove optional spaces to be conformant with CINT
         while ( (occ=s.find("* const")) != std::string::npos )    {
            if (!isalnum(s[occ + 7]))
               s.replace(occ, 7, "*const");
         }
         /// remove optional spaces to be conformant with CINT
         while ( (occ=s.find("& const")) != std::string::npos )    {
            if (!isalnum(s[occ + 7]))
               s.replace(occ, 7, "&const");
         }

         //
         // Perform naming normalization for primitives
         // since GCC-XML just generates anything.
         //
         // Let's hope this will not kick us back sometimes.
         //
         for(size_t i=0; i<sizeof(s_normalize)/sizeof(s_normalize[0]); ++i) {
            /// Normalize names
            while ( (occ=s.find(s_normalize[i][0])) != std::string::npos )    {
               s.replace(occ, strlen(s_normalize[i][0]), s_normalize[i][1]);
            }
         }
         if ( s.find('[') != std::string::npos ) {
            s = s.substr(0, s.find('['));
         }
         return s;
      }
   }

   int CintTag(const std::string& Name) {
      std::string n = CintName(Name);
      if ( n == "-" ) return -1;
      else            return G__search_tagname(n.c_str(),'c');
   }

   std::string CintSignature(const Member& func ) {
      // Argument signature is:
      // <type-char> <'object type name'> <typedef-name> <indirection> <default-value> <argument-Name>
      // i - 'Int_t' 0 0 option
      string signature;
      Type ft = func.TypeOf();
      for ( size_t p = 0 ; p < ft.FunctionParameterSize(); p++ ) {
         Type pt = ft.FunctionParameterAt(p);
         string arg_sig;
         Indirection  indir = IndirectionGet(pt);
         CintTypeDesc ctype = CintType(indir.second);
         if( indir.first == 0 ) arg_sig +=  ctype.first;
         else                   arg_sig += (ctype.first - ('a'-'A'));  // if pointer: 'f' -> 'F' etc.
         arg_sig += " ";
         //---Fill the true-type and eventually the typedef
         if( ctype.second == "-") {
            arg_sig += "-";
            if ( pt.IsTypedef() ) arg_sig += " '" + CintName(pt.Name(SCOPED)) + "' ";
            else                  arg_sig += " - ";                    
         }
         else {
            G__TypedefInfo tdef(ctype.second.c_str());
            int tagnum = G__defined_tagname(ctype.second.c_str(), 2);
            if ( tdef.IsValid() )    arg_sig += "'" + string(tdef.TrueName()) + "'";
            else if ( tagnum != -1 ) arg_sig += "'" + string(G__fulltagname(tagnum, 1)) + "'";
            else                     arg_sig += "'" + ctype.second + "'";  // object type name

            if ( pt.IsTypedef() || 
                 tdef.IsValid() ) arg_sig += " '" + CintName(pt.Name(SCOPED)) + "' ";
            else                  arg_sig += " - ";                    

         }
         // Assign indirection. First indirection already taken into account by uppercasing type
         if( indir.first == 0 || indir.first == 1 ) {
            if ( pt.IsReference() && pt.IsConst() )          arg_sig += "11";
            else if ( pt.IsReference() )                     arg_sig += "1";
            else if ( pt.IsConst() )                         arg_sig += "10";
            else                                             arg_sig += "0";
         } 
         else {
            arg_sig += char('0'+indir.first); // convert 2 -> '2', 3 ->'3' etc.
         } 
         arg_sig += " ";
         // Default value
         if ( func.FunctionParameterDefaultAt(p) != "" ) arg_sig += "'" + func.FunctionParameterDefaultAt(p) + "'";
         else                                  arg_sig += "-";
         arg_sig += " ";
         // Parameter Name
         if ( func.FunctionParameterNameAt(p) != "" ) arg_sig += func.FunctionParameterNameAt(p);
         else                               arg_sig += "-";
  
         signature += arg_sig;
         if ( p < ft.FunctionParameterSize() -1 ) signature += " ";
      }
      return signature;
   }

   void FillCintResult( G__value* result, const Type& typ, void* obj ) {
      CintTypeDesc ctype = CintType(typ);
      char t = ctype.first;
      if ( typ.IsPointer() ) t = (t - ('a'-'A'));
      result->type = t;
      switch( t ) {
      case 'y': G__setnull(result); break;
      case 'Y': Converter<long>::toCint          (result, obj); break;
      case 'g': Converter<bool>::toCint          (result, obj); break;
      case 'G': Converter<int>::toCint           (result, obj); break;
      case 'c': Converter<char>::toCint          (result, obj); break;
      case 'C': Converter<int>::toCint           (result, obj); break;
      case 'b': Converter<unsigned char>::toCint (result, obj); break;
      case 'B': Converter<int>::toCint           (result, obj); break;
      case 's': Converter<short>::toCint         (result, obj); break;
      case 'S': Converter<int>::toCint           (result, obj); break;
      case 'r': Converter<unsigned short>::toCint(result, obj); break;
      case 'R': Converter<int>::toCint           (result, obj); break;
      case 'i': Converter<int>::toCint           (result, obj); break;
      case 'I': Converter<int>::toCint           (result, obj); break;
      case 'h': Converter<unsigned int>::toCint  (result, obj); break;
      case 'H': Converter<int>::toCint           (result, obj); break;
      case 'l': Converter<long>::toCint          (result, obj); break;
      case 'L': Converter<int>::toCint           (result, obj); break;
      case 'k': Converter<unsigned long>::toCint (result, obj); break;
      case 'K': Converter<int>::toCint           (result, obj); break;
      case 'n': Converter<long long>::toCint     (result, obj); break;
      case 'N': Converter<int>::toCint           (result, obj); break;
      case 'm': Converter<unsigned long long>::toCint (result, obj); break;
      case 'M': Converter<int>::toCint           (result, obj); break;
      case 'f': Converter<float>::toCint         (result, obj); break;
      case 'F': Converter<int>::toCint           (result, obj); break;
      case 'd': Converter<double>::toCint        (result, obj); break;
      case 'D': Converter<int>::toCint           (result, obj); break;
      case 'q': Converter<long double>::toCint   (result, obj); break;
      case 'Q': Converter<int>::toCint           (result, obj); break;
      default:  
         result->obj.i = (long)obj;
         if( ! typ.IsPointer()) result->ref = (long)obj;
         else                   result->ref = 0;
         result->tagnum = G__search_tagname(ctype.second.c_str(),'c');
         break;
      }
   }


}}
