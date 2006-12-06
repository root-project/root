/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file v6_rflxutil.cxx
 ************************************************************************
 * Description:
 *  Utilities for Reflex migration / integration
 ************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#include "common.h"

#include "Reflex/Builder/TypeBuilder.h"
#include "Reflex/Tools.h"
#include <string>

namespace {
   static size_t GetReflexPropertyID() {
      static size_t reflexPropertyID = (size_t)-1;

      if (reflexPropertyID == (size_t)-1) {
         const char* propName = "Cint Properties";
         const G__RflxProperties rp;
         reflexPropertyID = Reflex::PropertyList::KeyByName(propName, true);
      }
      return reflexPropertyID;
   }
}

int Cint::Internal::G__get_type(const ::ROOT::Reflex::Type& in) {
   // FINAL for a typedef only remove the typedef layer!

   ::ROOT::Reflex::Type final = in.FinalType();

   if (in.IsPointerToMember() || in.ToType().IsPointerToMember() ||  final.IsPointerToMember()) return 'a';

   int pointerThusUppercase = final.IsPointer();
   pointerThusUppercase *= 'A' - 'a';
   ::ROOT::Reflex::Type raw = in.RawType();

   if (raw.IsFundamental()) {
      ::Reflex::EFUNDAMENTALTYPE fundamental = ::Reflex::Tools::FundamentalType(raw);
      char unsigned_flag = (fundamental == ::Reflex::kUNSIGNED_CHAR 
         || fundamental == ::Reflex::kUNSIGNED_SHORT_INT
         || fundamental == ::Reflex::kUNSIGNED_INT
         || fundamental == ::Reflex::kUNSIGNED_LONG_INT
         || fundamental == ::Reflex::kULONGLONG
         );
      switch (fundamental) {
         // NOTE: "raw unsigned" (type 'h') is gone - it's equivalent to unsigned int.
         case ::Reflex::kCHAR:
         case ::Reflex::kSIGNED_CHAR:
         case ::Reflex::kUNSIGNED_CHAR:
            return ((int) 'c') - unsigned_flag + pointerThusUppercase;
         case ::Reflex::kSHORT_INT:
         case ::Reflex::kUNSIGNED_SHORT_INT:
            return ((int) 's') - unsigned_flag + pointerThusUppercase;
         case ::Reflex::kINT:
         case ::Reflex::kUNSIGNED_INT:
            return ((int) 'i') - unsigned_flag + pointerThusUppercase;
         case ::Reflex::kLONG_INT:
         case ::Reflex::kUNSIGNED_LONG_INT:
            return ((int) 'l') - unsigned_flag + pointerThusUppercase;
         case ::Reflex::kLONGLONG:
         case ::Reflex::kULONGLONG:
            return ((int) 'n') - unsigned_flag + pointerThusUppercase;

         case ::Reflex::kBOOL:
            return ((int) 'g') + pointerThusUppercase;
         case ::Reflex::kVOID:
            return ((int) 'y') + pointerThusUppercase;
            
         case ::Reflex::kFLOAT:
            return ((int) 'f') + pointerThusUppercase;
         case ::Reflex::kDOUBLE:
            return ((int) 'd') + pointerThusUppercase;
         case ::Reflex::kLONG_DOUBLE:
            return ((int) 'q') + pointerThusUppercase;
         default:
            printf("G__gettype error: fundamental Reflex::Type %s %s is not taken into account!\n",
               raw.TypeTypeAsString().c_str(), raw.Name_c_str());
            return 0;
      } // switch fundamental
   }
   if (raw.Name() == "FILE") return ((int) 'e') + pointerThusUppercase;
   if (raw.IsClass()|| raw.IsStruct() || 
       raw.IsEnum() || raw.IsUnion()) 
       return ((int) 'u') + pointerThusUppercase;
   return 0;
}

int Cint::Internal::G__get_tagtype(const ::ROOT::Reflex::Type& in) 
{
   switch (in.RawType().TypeType()) {
      case ::ROOT::Reflex::CLASS: return 'c';
      case ::ROOT::Reflex::STRUCT: return 's';
      case ::ROOT::Reflex::ENUM: return 'e';
      case ::ROOT::Reflex::UNION: return 'u';
   }
   return 0;
}

int Cint::Internal::G__get_reftype(const ::ROOT::Reflex::Type& in) {
   char reftype = G__PARANORMAL; // aka =0

   ::ROOT::Reflex::Type current = in;
   bool isref = current.IsReference();
   while (!isref && current && current.IsTypedef()) {
      current = current.ToType();
      isref = current.IsReference();
   }
   int pointers = 0;
   do {
      if (current.IsPointer()) {
         ++pointers;
      }
      current = current.ToType();
   } while ( current );

   if (pointers>1) {
      reftype = pointers + isref * G__PARAREF;
   } else if (isref) {
      reftype = G__PARAREFERENCE;
   }
   return reftype;
}

G__RflxProperties *Cint::Internal::G__get_properties(const ::ROOT::Reflex::Type& in) {
   if (!in) return 0;
   size_t pid = GetReflexPropertyID();
   if (!in.Properties().HasProperty(pid)) 
      G__set_properties(in, G__RflxProperties());
   return ::ROOT::Reflex::any_cast<G__RflxProperties>( & in.Properties().PropertyValue(pid) );
}

G__RflxProperties *Cint::Internal::G__get_properties(const ::ROOT::Reflex::Scope& in) {
   if (!in) return 0;
   size_t pid = GetReflexPropertyID();
   if (!in.Properties().HasProperty(pid)) 
      G__set_properties(in, G__RflxProperties());
   return ::ROOT::Reflex::any_cast<G__RflxProperties>( & in.Properties().PropertyValue(pid) );
}

void Cint::Internal::G__set_properties(const ::ROOT::Reflex::Type& in, const G__RflxProperties& rp) {
   in.Properties().AddProperty(GetReflexPropertyID(), rp);
}

void Cint::Internal::G__set_properties(const ::ROOT::Reflex::Scope& in, const G__RflxProperties& rp) {
   in.Properties().AddProperty(GetReflexPropertyID(), rp);
}

G__SIGNEDCHAR_T Cint::Internal::G__get_isconst(const ::ROOT::Reflex::Type& in) {
   ::ROOT::Reflex::Type current = in;
   bool seen_first_pointer = false;
   bool last_const = 0;
   char isconst = 0;
   do {
      if ((current.IsPointer() || current.IsReference())) {
         if (!seen_first_pointer) {
            isconst |= current.IsConst() * G__PCONSTVAR;
            seen_first_pointer = true;
         }
         last_const = false;
      } else {
         last_const |= current.IsConst();
      }
      current = current.ToType();
   } while (current);

   if (last_const) isconst |= last_const * G__CONSTVAR;

   if (in.IsFunction()) {
      isconst  = in.ReturnType().IsConst() * G__CONSTVAR;
      isconst |= in.IsConst() * G__CONSTFUNC;
   }
   return isconst;
}

int Cint::Internal::G__get_nindex(const ::ROOT::Reflex::Type& in) 
{
   int nindex = 0;
   ::ROOT::Reflex::Type current = in;
   for (; current && current.IsArray(); ++nindex)
      current = current.ToType();
   return nindex;
}

std::vector<int> Cint::Internal::G__get_index(const ::ROOT::Reflex::Type& in) 
{
   ::ROOT::Reflex::Type current = in;
   if (current.IsTypedef()) current = in.FinalType();

   std::vector<int> index;
   for (; current && current.IsArray(); current = current.ToType()) {
      index.push_back(current.ArrayLength());
   }
   return index;
}



int Cint::Internal::G__get_typenum(const ::ROOT::Reflex::Type& in)
{
   if (!in.IsTypedef()) return -1;
   G__RflxProperties *prop = G__get_properties(in);
   return prop ? prop->typenum : -1;
}

int Cint::Internal::G__get_tagnum(const ::ROOT::Reflex::Type& in)
{
   G__RflxProperties *prop = G__get_properties(in);
   return prop ? prop->tagnum : -1;
}

int Cint::Internal::G__get_tagnum(const ::ROOT::Reflex::Scope& in)
{
   G__RflxProperties *prop = G__get_properties(in);
   return prop ? prop->tagnum : -1;
}

// Note: The return value must be by-reference, because this routine
//       is commonly used as a lvalue on the left-hand side of
//       an assignment expresssion.
::ROOT::Reflex::Type& Cint::Internal::G__value_typenum(G__value& gv) {
   return *((::ROOT::Reflex::Type*)&gv.buf_typenum);
}

::ROOT::Reflex::Type Cint::Internal::G__get_from_type(int type, int createpointer) 
{
   // We should NOT consider type q and a as pointer (ever).

   std::string name;
   switch(tolower(type)) {
      case 'b': name = "unsigned char"; break;
      case 'c': name = "char"; break;
      case 'r': name = "unsigned short"; break;
      case 's': name = "short"; break;
      case 'h': name = "unsigned int"; break;
      case 'i': name = "int"; break;
      case 'k': name = "unsigned long"; break;
      case 'l': name = "long"; break;
      case 'g': name = "bool"; break;
      case 'n': name = "long long"; break;
      case 'm': name = "unsigned long long"; break;
      case 'q': name = "long double"; break; 
      case 'f': name = "float"; break;
      case 'd': name = "double"; break;
#ifndef G__OLDIMPLEMENTATION2191
      case '1': 
#else
      case 'q': 
#endif
      case 'y': name = "void"; break;
      case 'e': name = "FILE"; break;
      case 'u': name = "enum";
         break;
      case 't':
#ifndef G__OLDIMPLEMENTATION2191
      case 'j':
#else
      case 'm':
#endif
      case 'p': name = "#define"; break;
      case 'o': name[0]='\0'; /* sprintf(name,""); */ break;
      case 'a': 
         /* G__ASSERT(isupper(type)); */
         //name = "G__p2memfunc";
         //type=tolower(type);
         return Reflex::PointerToMemberBuilder(Reflex::Dummy::Type(), Reflex::Dummy::Scope());
         break;
      default:  name = "(unknown)"; break;
   }

   createpointer &= isupper(type);
   if (createpointer && (type!='q' && type!='a')) {
      return ::ROOT::Reflex::PointerBuilder(::ROOT::Reflex::Type::ByName(name));
   } else {
      return ::ROOT::Reflex::Type::ByName(name);
   }
}

::ROOT::Reflex::Type Cint::Internal::G__modify_type(const ::ROOT::Reflex::Type& typein
                                    ,bool ispointer
                                    ,int reftype,int isconst
                                    ,int nindex, int *index)
{
   int ref = G__REF(reftype);
   reftype = G__PLVL(reftype);

   ::ROOT::Reflex::Type result = typein;

   if(isconst&G__CONSTVAR) result = ::ROOT::Reflex::Type( result, ::ROOT::Reflex::CONST, true );

   if (ispointer) result = ::ROOT::Reflex::PointerBuilder(result);

   if (nindex) {
      // CHECKME: Is this the right order?
      for(int i=nindex-1;i>=0;--i) {
         result = ::ROOT::Reflex::ArrayBuilder(result,index[i]);
      }
   }

   switch(reftype) {
    case G__PARANORMAL: break;
    case G__PARAREFERENCE:
       result = ::ROOT::Reflex::Type( result, ::ROOT::Reflex::REFERENCE, true );
       //strcpy(string+strlen(string),"&");
       break;
    case G__PARAP2P:
       // else strcpy(string+strlen(string),"*");
       result = ::ROOT::Reflex::PointerBuilder(result);
       break;
    case G__PARAP2P2P:
       // else strcpy(string+strlen(string),"**");
       result = ::ROOT::Reflex::PointerBuilder(result);
       result = ::ROOT::Reflex::PointerBuilder(result);
       break;
    default:
       if(reftype>10||reftype<0) break; /* workaround */
       for(int i=G__PARAP2P;i<=reftype;i++) {
          // strcpy(string+strlen(string),"*");
          result = ::ROOT::Reflex::PointerBuilder(result);
       }
       break;
   }
   //if(isconst&G__PCONSTVAR) strcpy(string+strlen(string)," const");
   if((isconst&G__PCONSTVAR) && (reftype>=G__PARAREFERENCE || ispointer))
      result = ::ROOT::Reflex::Type( result, ::ROOT::Reflex::CONST, true );

   if(ref)result = ::ROOT::Reflex::Type( result, ::ROOT::Reflex::REFERENCE, true);

   return result;
}

static void G__dumpreflex(const ::ROOT::Reflex::Scope& scope, int level) {
   for (::ROOT::Reflex::Scope_Iterator iscope = scope.SubScope_Begin();
      iscope != scope.SubScope_End(); ++iscope) {
         for (int i=0; i<level; ++i)
            printf(" ");
         printf("%s %s\n", iscope->IsClass()?"class":"scope", iscope->Name().c_str());
         G__dumpreflex(*iscope, level+1);
   }
   for (::ROOT::Reflex::Type_Iterator itype = scope.SubType_Begin();
      itype != scope.SubType_End(); ++itype) {
         if (itype->IsClass()) continue;
         for (int i=0; i<level; ++i)
            printf(" ");
         printf("%s %s\n", itype->TypeTypeAsString().c_str(), itype->Name().c_str());
   }
}

void Cint::Internal::G__dumpreflex() {
   ::G__dumpreflex(::ROOT::Reflex::Scope::GlobalScope(), 0);
}


::ROOT::Reflex::Type Cint::Internal::G__findInScope(const ::ROOT::Reflex::Scope& scope, const char* name)
{
   ::ROOT::Reflex::Type cl;
#ifdef __GNUC__
#else
#pragma message (FIXME("This needs to be in Reflex itself"))
#endif
   for (::ROOT::Reflex::Type_Iterator itype = scope.SubType_Begin();
      itype != scope.SubType_End(); ++itype) {
         if (itype->Name()==name) {
            cl = *itype;
            break;
         }
   }
   return cl;
}
