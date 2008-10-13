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
#include "Reflex/Builder/FunctionBuilder.h"
#include "Reflex/Tools.h"
#include "Dict.h"
#include <cassert>
#include <string>
#include <cstring>

using namespace std;

extern "C" void G__dump_reflex();
extern "C" void G__dump_reflex_atlevel(const ::Reflex::Scope scope, int level);
extern "C" void G__dump_reflex_function(const ::Reflex::Scope scope, int level);

//______________________________________________________________________________
size_t Cint::Internal::GetReflexPropertyID()
{
   // -- Internal use only, encodes static key for CINT.
   static size_t reflexPropertyID = (size_t) -1;

   if (reflexPropertyID == (size_t) -1) {
      const char* propName = "Cint Properties";
      const G__RflxProperties rp;
      reflexPropertyID = Reflex::PropertyList::KeyByName(propName, true);
   }
   return reflexPropertyID;
}

//______________________________________________________________________________
void Cint::Internal::G__get_cint5_type_tuple(const ::Reflex::Type in_type, char* out_type, int* out_tagnum, int* out_typenum, int* out_reftype, int* out_constvar)
{
   // Return the cint5 type tuple for a reflex type.
   //
   // Note: Some C++ types cannot be represented by a cint5 type tuple.
   //
   //--
   //
   //  Get the type part.
   //
   *out_type = G__get_type(in_type);
   //
   //  Get the tagnum part.
   //
   {
      //for (::Reflex::Type current = in_type; current; current = current.ToType()) {
      //   if (current.IsClass() || current.IsUnion() || current.IsEnum()) {
      //      *out_tagnum = G__get_properties(current)->tagnum;
      //      break;
      //   }
      //}
      *out_tagnum = -1;
      ::Reflex::Type ty = in_type.RawType();
      if (ty) {
         G__RflxProperties* prop = G__get_properties(ty);
         if (prop) {
            *out_tagnum = prop->tagnum;
         }
      }
   }
   //
   //  Get the typenum part.
   //
   //  Note: This is not right because cint5 can only have certain type nodes
   //        above a typedef, but this is good enough for now.
   //
   {
      *out_typenum = -1;
      for (::Reflex::Type current = in_type; current; current = current.ToType()) {
         if (current.IsTypedef()) {
            G__RflxProperties* prop = G__get_properties(current);
            if (prop) {
               *out_typenum = prop->typenum;
            }
            break;
         }
      }
   }
   //
   // Get CINT reftype for a type (PARANORMAL, PARAREFERENCE, etc.).
   //
   // Given this:
   //
   //      typedef char& cref;
   //      void f(cref a);
   //
   // cint v5 does not consider function parameter "a" to be a reference.
   //
   // Type in cint v5 is a five-tuple (type, tagnum, typenum, reftype=(ref,ptr-level), const)
   // and here it would be: ('c', -1, ('c', -1, -1, (ref, 0), 0), (notref,0), 0).
   //
   // Given this:
   //
   //      typedef char* ch;
   //      void f(ch**& a);
   //
   // cint v5 considers "a" to be: ('C', -1, ('C', -1, -1, (notref, 0), 0), (ref, 2), 0)
   //
   // Given this:
   //
   //      typedef char* cp;
   //      typedef cp*   cpp;
   //      void f(cpp**& a);
   //
   // cint v5 considers "a" to be: ('C', -1, ('C', -1, -1, (notref, 2), 0), (ref, 2), 0)
   //
   // Note that cint v5 cannot remember more than one level of typedef deep.
   //
   {
      bool isref = in_type.IsReference();
      ::Reflex::Type current = in_type; 
      for (; current && current.IsTypedef();) {
         current = current.ToType();
      }
      // Count pointer levels.
      int pointers = 0;
      for (; current && current.IsPointer(); current = current.ToType()) {
         ++pointers;
      }
      *out_reftype = G__PARANORMAL;
      if (pointers > 1) {
         *out_reftype = (isref * G__PARAREF) + pointers;
      } else if (isref) {
         *out_reftype = G__PARAREFERENCE;
      }
   }
   //
   //  Now get the constvar part.
   //
   //  Note: This is not right, the correct setting of G__PCONSTVAR depends
   //        on where the space characters were in the declaration in the
   //        source code.  We do not have that information available.
   //
   {
      *out_constvar = G__get_isconst(in_type);
      //if (in_type.IsFunction()) {
      //   if (in_type.IsConst()) {
      //      *out_constvar |= G__CONSTFUNC;
      //   }
      //   if (in_type.ReturnType().IsConst()) {
      //      *out_constvar |= G__CONSTVAR;
      //   }
      //}
      //else {
      //   //bool ptr_or_ref_seen = false;
      //   //bool accumulated_const = false;
      //   //for (::Reflex::Type current = in_type; current; current = current.ToType()) {
      //   //   accumulated_const = accumulated_const || current.IsConst();
      //   //   if (current.IsPointer() || current.IsReference()) {
      //   //      accumulated_const = false;
      //   //      if (!ptr_or_ref_seen) {
      //   //         ptr_or_ref_seen = true;
      //   //         if (current.IsConst()) {
      //   //            *out_constvar |= G__PCONSTVAR;
      //   //         }
      //   //      }
      //   //   }
      //   //}
      //   //if (accumulated_const) {
      //   //   *out_constvar |= G__CONSTVAR;
      //   //}
      //   if (in_type.IsConst() && in_type.IsPointer()) {
      //      *out_constvar |= G__PCONSTVAR;
      //   }
      //   if (in_type.RawType().IsConst()) {
      //      *out_constvar |= G__CONSTVAR;
      //   }
      //}
   }
}

//______________________________________________________________________________
int Cint::Internal::G__get_cint5_typenum(const ::Reflex::Type in_type)
{
   //  Get the cint5 typenum part of a cint7 type.
   //
   //  Note: This is not right because cint5 can only have certain type nodes
   //        above a typedef and technically we should allow only those and
   //        return an error if an invalid node is found, but this is good
   //        enough for now.
   //
   int ret_typenum = -1;
   for (::Reflex::Type current = in_type; current; current = current.ToType()) {
      if (current.IsTypedef()) {
         G__RflxProperties* prop = G__get_properties(current);
         if (prop) {
            ret_typenum = prop->typenum;
         }
         break;
      }
   }
   return ret_typenum;
}

//______________________________________________________________________________
extern "C" int G__value_get_type(G__value *buf)
{
   return Cint::Internal::G__get_type(*buf);
}

//______________________________________________________________________________
int Cint::Internal::G__get_type(const ::Reflex::Type in)
{
   // -- Get CINT type code for data type.
   // Note: Structures are all 'u'.
   if (!in) return 0;
   if (in.Name() == "macroInt$")    return 'p';
   if (in.Name() == "macroDouble$") return 'P';
   if (in.Name() == "autoInt$")     return 'o';
   if (in.Name() == "autoDouble$")  return 'O';
   if (in.Name() == "macro$") return 'j';
   if (in.Name() == "switchStart$")  return 'a';
   if (in.Name() == "switchDefault$")  return 'z';
   if (in.Name() == "codeBreak$")  return 'Z';
   if (in.Name() == "codeBreak$*")  return 'Z'; // This is actually a 'slot' for a not yet found special object
   if (in.Name() == "macroChar*$") return 'T';
   if (in.Name() == "defaultFunccall$") return G__DEFAULT_FUNCCALL;


   // FINAL for a typedef only remove the typedef layer!

   ::Reflex::Type final = in.FinalType();

   if (in.IsPointerToMember() || in.ToType().IsPointerToMember() ||  final.IsPointerToMember()) return 'a';

   // The type code for a variable in cint5 ignores any array part.
   for (; final.IsArray(); final = final.ToType()) {}

   int pointerThusUppercase = final.IsPointer();
   pointerThusUppercase *= 'A' - 'a';
   ::Reflex::Type raw = in.RawType();

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
         case ::Reflex::kVOID: {
            if (final.IsPointer()) {
               return 'Y';
            } else if (final.TypeType()==Reflex::FUNCTION || final.TypeType()==Reflex::FUNCTIONMEMBER || strstr(in.Name().c_str(),"(")) {
               return '1';
            } else {
               return 'y';
            };
            // return ((int) 'y') + pointerThusUppercase;
         }
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
   if (raw.IsEnum()) return ((int)'i') + pointerThusUppercase;
   if (raw.IsClass()|| raw.IsStruct() ||
       /* raw.IsEnum() || */ raw.IsUnion())
       return ((int) 'u') + pointerThusUppercase;
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__get_type(const G__value& in)
{
   // -- Get CINT type code for a G__value.
   int ret = G__get_type(G__value_typenum(in));
   return ret;
}

//______________________________________________________________________________
int Cint::Internal::G__get_tagtype(const ::Reflex::Type in)
{
   // -- Get CINT type code for a structure (c,s,e,u).
   switch (in.RawType().TypeType()) {
      case ::Reflex::TYPETEMPLATEINSTANCE: return 'c';
      case ::Reflex::CLASS: return 'c';
      case ::Reflex::STRUCT: return 's';
      case ::Reflex::ENUM: return 'e';
      case ::Reflex::UNION: return 'u';
      case ::Reflex::FUNCTION:
      case ::Reflex::ARRAY:
      case ::Reflex::FUNDAMENTAL:
      case ::Reflex::POINTER:
      case ::Reflex::POINTERTOMEMBER:
      case ::Reflex::TYPEDEF:
      case ::Reflex::MEMBERTEMPLATEINSTANCE:
      case ::Reflex::NAMESPACE:
      case ::Reflex::DATAMEMBER:
      case ::Reflex::FUNCTIONMEMBER:
      case ::Reflex::UNRESOLVED:
         break;
   }
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__get_tagtype(const ::Reflex::Scope in)
{
   // -- Get CINT type code for a structure (c,s,e,u).
   
   switch (in.ScopeType()) {
      case ::Reflex::TYPETEMPLATEINSTANCE: return 'c';
      case ::Reflex::CLASS: return 'c';
      case ::Reflex::STRUCT: return 's';
      case ::Reflex::ENUM: return 'e';
      case ::Reflex::UNION: return 'u';
      case ::Reflex::NAMESPACE: return 'n';
      case ::Reflex::FUNCTION:
      case ::Reflex::ARRAY:
      case ::Reflex::FUNDAMENTAL:
      case ::Reflex::POINTER:
      case ::Reflex::POINTERTOMEMBER:
      case ::Reflex::TYPEDEF:
      case ::Reflex::MEMBERTEMPLATEINSTANCE:
      case ::Reflex::DATAMEMBER:
      case ::Reflex::FUNCTIONMEMBER:
      case ::Reflex::UNRESOLVED:
         break;
   }
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__get_reftype(const ::Reflex::Type in)
{
   // -- Get CINT reftype for a type (PARANORMAL, PARAREFERENCE, etc.).
   //
   // Given this:
   //
   //      typedef char& cref;
   //      void f(cref a);
   //
   // cint v5 does not consider function parameter "a" to be a reference.
   //
   // Type in cint v5 is a five-tuple (type, tagnum, typenum, reftype=(ref,ptr-level), const)
   // and here it would be: ('c', -1, ('c', -1, -1, (ref, 0), 0), (notref,0), 0).
   //
   // Given this:
   //
   //      typedef char* ch;
   //      void f(ch**& a);
   //
   // cint v5 considers "a" to be: ('C', -1, ('C', -1, -1, (notref, 0), 0), (ref, 2), 0)
   //
   // Given this:
   //
   //      typedef char* cp;
   //      typedef cp*   cpp;
   //      void f(cpp**& a);
   //
   // cint v5 considers "a" to be: ('C', -1, ('C', -1, -1, (notref, 2), 0), (ref, 2), 0)
   //
   // Note that cint v5 cannot remember more than one level of typedef deep.
   //
   bool isref = in.IsReference();

   ::Reflex::Type current = in; 
   while (!isref && current && current.IsTypedef()) {
      current = current.ToType();
      isref = current.IsReference();
   }

   // Count pointer levels.
   int pointers = 0;
   for (
      ;
      current && current.IsPointer();
      current = current.ToType()
   ) {
      ++pointers;
   }
   char reftype = G__PARANORMAL;
   if (pointers > 1) {
      reftype = (isref * G__PARAREF) + pointers;
   } else if (isref) {
      reftype = G__PARAREFERENCE;
   }
   return reftype;
}

//______________________________________________________________________________
G__RflxProperties* Cint::Internal::G__get_properties(const ::Reflex::Type in)
{
   // -- Get REFLEX property list from a data type.
   if (!in) {
      return 0;
   }
   size_t pid = GetReflexPropertyID();
   if (!in.Properties().HasProperty(pid)) {
      G__set_properties(in, G__RflxProperties());
   }
   return ::Reflex::any_cast<G__RflxProperties>(&in.Properties().PropertyValue(pid));
}

//______________________________________________________________________________
G__RflxProperties* Cint::Internal::G__get_properties(const ::Reflex::Scope in)
{
   // -- Get REFLEX property list from a scope.
   if (!in) {
      return 0;
   }
   size_t pid = GetReflexPropertyID();
   if (!in.Properties().HasProperty(pid)) {
      G__set_properties(in, G__RflxProperties());
   }
   return ::Reflex::any_cast<G__RflxProperties>(&in.Properties().PropertyValue(pid));
}

//______________________________________________________________________________
G__RflxVarProperties* Cint::Internal::G__get_properties(const ::Reflex::Member in)
{
   // -- Get REFLEX property list from a class member.
   if (!in || !in.IsDataMember()) {
      return 0;
   }
   size_t pid = GetReflexPropertyID();
   if (!in.Properties().HasProperty(pid)) {
      G__set_properties(in, G__RflxVarProperties());
   }
   G__RflxVarProperties *res = ::Reflex::any_cast<G__RflxVarProperties>(&in.Properties().PropertyValue(pid));
   return res;
}

//______________________________________________________________________________
G__RflxFuncProperties* Cint::Internal::G__get_funcproperties(const ::Reflex::Member in)
{
   // -- Get REFLEX property list from a class member function.
   if (!in || !(in.IsFunctionMember() || in.IsTemplateInstance())) {
      //fprintf(stderr, "G__get_funcproperties(const ::Reflex::Member): Bad member '%s'  %s:%d\n", in.Name(::Reflex::SCOPED), __FILE__, __LINE__);
      //fprintf(stderr, "G__get_funcproperties(const ::Reflex::Member): valid: %d  %s:%d\n", (bool) in, __FILE__, __LINE__);
      //fprintf(stderr, "G__get_funcproperties(const ::Reflex::Member): IsFunctionMember: %d  %s:%d\n", in.IsFunctionMember(), __FILE__, __LINE__);
      //fprintf(stderr, "G__get_funcproperties(const ::Reflex::Member): IsTemplateInstance: %d  %s:%d\n", in.IsTemplateInstance(), __FILE__, __LINE__);
      return 0;
   }
   size_t pid = GetReflexPropertyID();
   if (!in.Properties().HasProperty(pid)) {
      G__set_properties(in, G__RflxFuncProperties());
   }
   // Stock in a temporary variable so that we can inspect it in a debugger.
   G__RflxFuncProperties *ret = ::Reflex::any_cast<G__RflxFuncProperties>(&in.Properties().PropertyValue(pid));
   return ret;
}

//______________________________________________________________________________
G__SIGNEDCHAR_T Cint::Internal::G__get_isconst(const ::Reflex::Type in)
{
   // -- Is data type const qualified?
   ::Reflex::Type current = in; // .FinalType(); is currently buggy

   bool seen_first_pointer = false;
   bool last_const = false;
   char isconst = '\0';

   if (in.IsFunction()) {
      isconst = in.IsConst() * G__CONSTFUNC;
      current = in.ReturnType();
   }

   do {
      if ((current.IsPointer())) { //  || current.IsReference())) {
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

   return isconst;
}

//______________________________________________________________________________
int Cint::Internal::G__get_nindex(const ::Reflex::Type passed_type)
{
   // -- Get dimensionality of a data type.

   // Start with the passed type.
   ::Reflex::Type ty = passed_type;

   int nindex = 0;
   for (; ty; ty = ty.ToType()) {
      // Skip typedefs.
      for (; ty.IsTypedef(); ty = ty.ToType()) {}
      if (!ty.IsArray()) {
         // -- Stop when we are no longer an array.
         break;
      }
      // Count each array dimension.
      ++nindex;
   }

   return nindex;
}

//______________________________________________________________________________
std::vector<int> Cint::Internal::G__get_index(const ::Reflex::Type passed_var)
{
   // -- Get CINT array maximum dimensions as a std::vector<int> for a data type.
   //

   // This will be our return value.
   std::vector<int> dimensions;

   // Start with the passed type.
   Reflex::Type var(passed_var);

   // Remove typedefs until we get to a real type.
   for (; var.IsTypedef(); var = var.ToType()) {}

   // Handle a scalar right away.
   if (!var.IsArray()) {
      // -- We are a scalar.
      return dimensions;
   }

   // Scan down the type chain and collect the array dimensions.
   int idx = 0;
   for (Reflex::Type ty = var; ty; ty = ty.ToType()) {
      // Skip typedefs.
      for (; ty.IsTypedef(); ty = ty.ToType()) {}
      if (!ty.IsArray()) {
         // -- Stop when we are no longer an array.
         break;
      }
      dimensions.push_back(ty.ArrayLength());
      ++idx;
   }

   // And fill out the rest with zeroes.
   //for (; idx < G__MAXVARDIM; ++idx) {
   //   dimensions.push_back(0);
   //}

   return dimensions;
}

#if 0 //PSRXXX// Will be useful later.
//______________________________________________________________________________
std::vector<int> Cint::Internal::G__get_varlabel_as_vector(const ::Reflex::Type& passed_var)
{
   // -- Get CINT varlabel as a std::vector<int> for a data type.
   //
   // varlabel is:
   //
   // for int a;
   //
   //      { 1, 0, 1, 0, ... }
   //
   // for int a[A][B][C];
   //
   //      { B*C, A*B*C, B, C, 1, 0, ... }
   //
   // for typedef int ARR[a][b][c]; ARR a[A][B][C];
   //
   //      { b*c*A*B*C, a*b*c*A*B*C, b, c, A, B, C, 1, 0, ...}
   //
   // for int (*ARRP[A][B][C])[x][y][z];
   //
   //      { B*C, A*B*C, B, C, 1, 0, x*y, x, y, 1, 0, ...}
   //
   // for typedef int ARR[a][b][c]; ARR (*vp[A][B][C])[x][y];
   //
   //      { b*c*A*B*C, a*b*c*A*B*C, b, c, A, B, C, 1, 0, x*y, x, y, 1, 0, ... }
   //

   // This will be our return value.
   std::vector<int> varlabel;

   // Start with the passed type.
   Reflex::Type var(passed_var);

   // Remove typedefs until we get to a real type.
   for (; var.IsTypedef(); var = var.ToType());

   // Handle a scalar right away.
   if (!var.IsArray()) {
      // -- We are a scalar.
      varlabel.push_back(1);
      varlabel.push_back(0);
      varlabel.push_back(1);
      if (var.IsPointer()) {
         varlabel.push_back(1);
         varlabel.push_back(1);
      }
      else {
         varlabel.push_back(0);
         varlabel.push_back(0);
      }
      for (int i = 5; i < G__MAXVARDIM; ++i) {
         varlabel.push_back(0);
      }
      return varlabel;
   }

   //
   // Handle the two special cases, first and second varlabel index.
   //

   // For the first index, we need the stride.
   {
      int stride = 1;
      Reflex::Type ty = var.ToType();
      for (; ty; ty = ty.ToType()) {
         // -- Multipy all but the first array dimension together.
         // Skip typedefs.
         for (; ty.IsTypedef(); ty = ty.ToType());
         if (!ty.IsArray()) {
            // -- Stop when we are no longer an array.
            break;
         }
         stride *= ty.ArrayLength();
      }
      varlabel.push_back(stride);
   }

   // For the second index we need the number of elements.
   {
      int number_of_elements = 1;
      for (Reflex::Type ty = var; ty; ty = ty.ToType()) {
         // -- Multipy all the array dimensions together.
         // Skip typedefs.
         for (; ty.IsTypedef(); ty = ty.ToType());
         if (!ty.IsArray()) {
            // -- Stop when we are no longer an array.
            break;
         }
         number_of_elements *= ty.ArrayLength();
      }
      varlabel.push_back(number_of_elements);
   }

   // For the rest of the indexes we just return the size of the dimension.
   {
      int idx = 0;
      for (Reflex::Type ty = var; ty; ty = ty.ToType()) {
         // -- Multipy all the array dimensions together.
         // Skip typedefs.
         for (; ty.IsTypedef(); ty = ty.ToType());
         if (!ty.IsArray()) {
            // -- Stop when we are no longer an array.
            break;
         }
         if (idx > 1) {
            varlabel.push_back(ty.ArrayLength());
         }
         ++idx;
      }
      // Now put on the end marker, which is a one followed by a zero.
      varlabel.push_back(1);
      ++idx;
      varlabel.push_back(0);
      ++idx;
      // And fill out the rest with zeroes.
      for (; idx < G__MAXVARDIM; ++idx) {
         varlabel.push_back(0);
      }
   }

   return varlabel;
}
#endif

//______________________________________________________________________________
int Cint::Internal::G__get_varlabel(const Reflex::Member var, int idx)
{
   // -- Get CINT varlabel for a class member.
   return G__get_varlabel(var.TypeOf(), idx);
}

//______________________________________________________________________________
int Cint::Internal::G__get_varlabel(const Reflex::Type passed_var, int idx)
{
   // -- Get CINT varlabel for a data type.
   //
   // varlabel is:
   //
   // for int a;
   //
   //      { 1, 0, 1, 0, ... }
   //
   // for int a[A][B][C];
   //
   //      { B*C, A*B*C, B, C, 1, 0, ... }
   //
   // for typedef int ARR[a][b][c]; ARR a[A][B][C];
   //
   //      { b*c*A*B*C, a*b*c*A*B*C, b, c, A, B, C, 1, 0, ...}
   //
   // for int (*ARRP[A][B][C])[x][y][z];
   //
   //      { B*C, A*B*C, B, C, 1, 0, x*y, x, y, 1, 0, ...}
   //
   // for typedef int ARR[a][b][c]; ARR (*vp[A][B][C])[x][y];
   //
   //      { b*c*A*B*C, a*b*c*A*B*C, b, c, A, B, C, 1, 0, x*y, x, y, 1, 0, ... }
   //

   // Validate passed index.
   if (idx < 0) {
      // -- Bad index, fail.
      // FIXME: We should give an error message here!
      return 0;
   }

   // Start with the passed type.
   Reflex::Type var(passed_var);

   // Remove typedefs until we get to a real type.
   for (; var.IsTypedef(); var = var.ToType()) {}

   // Handle a scalar right away.
   if (!var.IsArray()) {
      // -- We are a scalar.
      // FIXME: We should really give an error message if idx > 0.
      if (!idx || (idx == 2)) {
         return 1;
      }
      if (var.IsPointer() && ((idx == 3) || (idx == 4))) {
         return 1;
      }
      return 0;
   }

   // Handle the two special cases, first and second varlabel index.
   if (idx == 0) {
      // -- Need to return the stride.
      int stride = 1;
      Reflex::Type ty = var.ToType();
      for (; ty; ty = ty.ToType()) {
         // -- Multipy all but the first array dimension together.
         // Skip typedefs.
         for (; ty.IsTypedef(); ty = ty.ToType()) {}
         if (!ty.IsArray()) {
            // -- Stop when we are no longer an array.
            break;
         }
         stride *= ty.ArrayLength();
      }
      return stride;
   } else if (idx == 1) {
      // -- Need to return number of elements.
      int number_of_elements = 1;
      for (Reflex::Type ty = var; ty; ty = ty.ToType()) {
         // -- Multipy all the array dimensions together.
         // Skip typedefs.
         for (; ty.IsTypedef(); ty = ty.ToType()) {}
         if (!ty.IsArray()) {
            // -- Stop when we are no longer an array.
            break;
         }
         // Handle the unspecified length array flag specially.
         // FIXME: We could be more careful and insist that this be the last dimension in the type chain.
         if (ty.ArrayLength() == INT_MAX) {
            return INT_MAX;
         }
         number_of_elements *= ty.ArrayLength();
      }
      return number_of_elements;
   }

   //
   // Handle the non-special cases.
   //

   // Scan the type chain until we get to the dimension we want.
   int i = idx - 1;
   int size_of_dimension = 0;
   Reflex::Type ty = var;
   for (; ty.IsArray(); --i) {
      if (!i) {
         size_of_dimension = ty.ArrayLength();
         return size_of_dimension;
      }
      ty = ty.ToType();
      // Skip typedefs.
      for (; ty.IsTypedef(); ty = ty.ToType()) {}
   }
   if (!i) {
      // -- We are at the end marker, which is a one.
      return 1;
   }

   // Handle a scalar right away.
   if (ty.IsPointer() && ((i == 2) || (i == 3))) {
      return 1;
   }
   // -- Error, caller asked for a non-existant dimension.
   // FIXME: We need an error message here!
   return 0;
}

//______________________________________________________________________________
int Cint::Internal::G__get_typenum(const ::Reflex::Type in)
{
   // -- Get CINT typenum for a given typedef type.
   if (!in.IsTypedef()) {
      return -1;
   }
   G__RflxProperties* prop = G__get_properties(in);
   if (!prop) {
      return -1;
   }
   return prop->typenum;
}

//______________________________________________________________________________
int Cint::Internal::G__get_tagnum(const ::Reflex::Type in)
{
   // -- Get CINT tagnum for a given type.
   // Note: Only class types have a tagnum.
   G__RflxProperties* prop = G__get_properties(in.RawType());
   if (!prop) {
      return -1;
   }
   return prop->tagnum;
}

//______________________________________________________________________________
int Cint::Internal::G__get_tagnum(const ::Reflex::Scope in)
{
   // -- Get CINT tagnum for a given scope.
   // Note: Only class scopes have a tagnum.
   G__RflxProperties* prop = G__get_properties(in);
   if (!prop) {
      return -1;
   }
   return prop->tagnum;
}

//______________________________________________________________________________
::Reflex::Type& Cint::Internal::G__value_typenum(G__value& gv)
{
   // -- Get access to the buf_typenum field of a G__value.
   //
   // Note: The return value must be by-reference, because this routine
   //       is commonly used as a lvalue on the left-hand side of
   //       an assignment expresssion.
   return *((::Reflex::Type*) &gv.buf_typenum);
}

//______________________________________________________________________________
const ::Reflex::Type& Cint::Internal::G__value_typenum(const G__value& gv)
{
   return *((::Reflex::Type*)&gv.buf_typenum);
}

//______________________________________________________________________________
::Reflex::Type Cint::Internal::G__get_Type(int type, int tagnum, int typenum, int isconst)
{
   // -- Get a REFLEX Type from a CINT (type, tagnum, typenum) triple.
   // Note: The typenum is tried first, then tagnum, then type.
   Reflex::Type current;
   if (typenum != -1) {
      current = G__Dict::GetDict().GetTypedef(typenum);
   } else if (tagnum != -1) {
      current = G__Dict::GetDict().GetType(tagnum);
   }
   if (current) {
      if (isconst) current = Reflex::Type( current, Reflex::CONST, Reflex::Type::APPEND);
      if (type=='U') {
         return ::Reflex::PointerBuilder(current);
      } else {
         return current;
      }
   }
   return G__get_from_type(type, 1, isconst);
}

//______________________________________________________________________________
::Reflex::Type Cint::Internal::G__get_from_type(int type, int createpointer, int isconst /*=0*/)
{
   // -- Get a REFLEX Type from a CINT type.
   // Note: The typenum is tried first, then tagnum, then type.
   // We should NOT consider type q and a as pointer (ever).

   std::string name;
   switch (type) {
      /****************************************************
       * Automatic variable and macro
       *   p : macro int
       *   P : macro double
       *   o : auto int
       *   O : auto double
       *   a : switchStart (but also sometimes pointer to member function)
       *   z : switchDefault
       *   Z : codeBreak (continue, break, goto)
       ****************************************************/
   //case 'a': return ::Reflex::Type::ByName("switchStart$");
   case 'z': return ::Reflex::Type::ByName("switchDefault$");
   case 'Z': return ::Reflex::Type::ByName("codeBreak$");
   case 'p': return ::Reflex::Type::ByName("macroInt$");
   case 'j': return ::Reflex::Type::ByName("macro$");
   case 'P': return ::Reflex::Type::ByName("macroDouble$");
   case 'o': return ::Reflex::Type::ByName("autoInt$");
   case 'O': return ::Reflex::Type::ByName("autoDouble$");
   case 'T': return ::Reflex::Type::ByName("macroChar*$");
   case G__DEFAULT_FUNCCALL: return ::Reflex::Type::ByName("defaultFunccall$");
   }

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

   if (createpointer & !isupper(type)) {
      createpointer = 0;
   }
   Reflex::Type raw( ::Reflex::Type::ByName(name) );
   if (isconst & G__CONSTVAR) { 
      raw = Reflex::Type(raw,Reflex::CONST,Reflex::Type::APPEND);
   }
   if (createpointer /*&& (type != 'q') && (type != 'a')*/) {
      return ::Reflex::PointerBuilder(raw);
   } else {
      return raw;
   }
}

//______________________________________________________________________________
bool Cint::Internal::G__is_localstaticbody(const Reflex::Member var)
{
   // -- Is var is a static variable in an interpreted function body?

   // An interpreted function body has an invalid scope.
   return var.IsStatic() && !var.DeclaringScope();
}

//______________________________________________________________________________
int Cint::Internal::G__get_paran(const Reflex::Member var) {
   // -- Get the CINT paran for a REFLEX class member.
   // Note: This is zero if the member is not an array,
   //       otherwise it is the number of array dimensions.
   return G__get_nindex(var.TypeOf());
}

//______________________________________________________________________________
size_t Cint::Internal::G__get_bitfield_width(const Reflex::Member var)
{
   return G__get_properties(var)->bitfield_width;
}

//______________________________________________________________________________
size_t Cint::Internal::G__get_bitfield_start(const Reflex::Member var)
{
   return G__get_properties(var)->bitfield_start;
}

//______________________________________________________________________________
char*& Cint::Internal::G__get_offset(const ::Reflex::Member var) 
{
   return G__get_properties(var)->addressOffset;
}

//______________________________________________________________________________
Reflex::Type Cint::Internal::G__strip_array(const Reflex::Type typein)
{
   // Return the type obtains after stripping all array dimensions.

   Reflex::Type current( typein );
   Reflex::Type result( typein );
   while (current && (current.IsTypedef() || current.IsArray()))
   {
      if (current.IsArray()) {
         result = current.ToType();  // Advance prev only if we have an array.
      }
      current = current.ToType();
   }
   return result;
}

//______________________________________________________________________________
Reflex::Type Cint::Internal::G__strip_one_array(const Reflex::Type typein)
{
   // Remove one array node from the type.

   Reflex::Type current( typein );
   while (current && current.IsTypedef()) 
   {
      current = current.ToType();
   }
   assert( current.IsArray() );

   return current.ToType();
}

//______________________________________________________________________________
Reflex::Type Cint::Internal::G__deref(const Reflex::Type typein)
{
   // Return the type equivalent to the return type of the default unary operator*
   // of the type 'typein' (aka for float* returns float).

   Reflex::Type current( typein );
   while (current && current.IsTypedef()) 
   {
      current = current.ToType();
   }
   assert( current.IsPointer() );

   return current.ToType();
}

//______________________________________________________________________________
::Reflex::Type Cint::Internal::G__modify_type(const ::Reflex::Type typein, bool ispointer, int reftype, int isconst, int nindex, int* index)
{
   // -- Construct a new REFLEX Type based on a given type.
   int ref = G__REF(reftype);
   reftype = G__PLVL(reftype);
   ::Reflex::Type result = typein;
   if (isconst & G__CONSTVAR) {
     result = G__replace_rawtype( result,  ::Reflex::Type(result.RawType(), ::Reflex::CONST, Reflex::Type::APPEND) );
   }
   if (ispointer) {
      result = ::Reflex::PointerBuilder(result);
   }
   if (nindex) {
      // Build the array type chain in the reverse order
      // of the dimensions, starting from the right and
      // moving left towards the variable name.
      //
      // Note: This means that any unspecified length
      //       flag will be next-to-last in the chain,
      //       just before the type of the array elements.
      //
      // For example:
      //
      //      int a[2][3];
      //
      // gives:
      //
      //      a --> array[2] --> array[3] --> int
      //
      for (int i = nindex - 1; i >= 0; --i) {
         result = ::Reflex::ArrayBuilder(result, index[i]);
      }
   }
   switch (reftype) {
    case G__PARANORMAL:
       break;
    case G__PARAREFERENCE:
       result = ::Reflex::Type(result, ::Reflex::REFERENCE, Reflex::Type::APPEND);
       //strcpy(string+strlen(string),"&");
       break;
    case G__PARAP2P:
       // else strcpy(string+strlen(string),"*");
       result = ::Reflex::PointerBuilder(result);
       break;
    case G__PARAP2P2P:
       // else strcpy(string+strlen(string),"**");
       result = ::Reflex::PointerBuilder(result);
       result = ::Reflex::PointerBuilder(result);
       break;
    default:
       if ((reftype > 10) || (reftype < 0)) {
          // workaround
          break;
       }
       for (int i = G__PARAP2P; i <= reftype; ++i) {
          // strcpy(string+strlen(string),"*");
          result = ::Reflex::PointerBuilder(result);
       }
       break;
   }
   //if(isconst&G__PCONSTVAR) strcpy(string+strlen(string)," const");
   if ((isconst & G__PCONSTVAR) && ((reftype >= G__PARAREFERENCE) || ispointer)) {
      result = ::Reflex::Type(result, ::Reflex::CONST, Reflex::Type::APPEND);
   }
   if (ref) {
      result = ::Reflex::Type(result, ::Reflex::REFERENCE, Reflex::Type::APPEND);
   }
   return result;
}

//______________________________________________________________________________
extern "C" void G__dump_reflex()
{
   //return;
   ::G__dump_reflex_atlevel(::Reflex::Scope::GlobalScope(), 0);
}

//______________________________________________________________________________
extern "C" void G__dump_reflex_atlevel(const ::Reflex::Scope scope, int level)
{
   for ( // Dump data members.
      ::Reflex::Member_Iterator m = scope.DataMember_Begin();
      m != scope.DataMember_End();
      ++m
   ) { // Dump data members.
         // Indent.
         for (int i=0; i<(level*2); ++i) {
            fprintf(stderr, " ");
         }
         // Dump.
         fprintf(stderr, "data member: '%s' '%s'\n", m->Name(Reflex::SCOPED | Reflex::QUALIFIED).c_str(), m->TypeOf().Name(Reflex::SCOPED | Reflex::QUALIFIED).c_str());
   }
   G__dump_reflex_function(scope, level); // Dump member functions.
   for ( // Dump typdefs and enums.
      ::Reflex::Type_Iterator itype = scope.SubType_Begin();
      itype != scope.SubType_End();
      ++itype
   ) { // Dump typdefs and enums.
         // Skip classes and structs.  // FIXME: Should skip unions too!
         if (itype->IsClass()) {
            continue;
         }
         // Indent.
         for (int i=0; i<(level*2); ++i) {
            fprintf(stderr, " ");
         }
         // Dump.
         fprintf(stderr, "%s: '%s'\n", itype->TypeTypeAsString().c_str(), itype->Name(Reflex::SCOPED | Reflex::QUALIFIED).c_str());
   }
   for ( // Dump inner classes recursively.
      ::Reflex::Scope_Iterator iscope = scope.SubScope_Begin();
      iscope != scope.SubScope_End();
      ++iscope
   ) { // Dump inner classes recursively.
         // Indent.
         for (int i=0; i<(level*2); ++i) {
            fprintf(stderr, " ");
         }
         // Dump.
         fprintf(stderr, "%s: '%s'\n", iscope->ScopeTypeAsString().c_str(), iscope->Name(Reflex::SCOPED | Reflex::QUALIFIED).c_str());
         G__dump_reflex_atlevel(*iscope, level+1); // Recurse. // FIXME: This is a tail recursion!
   }
}

//______________________________________________________________________________
extern "C" void G__dump_reflex_function(const ::Reflex::Scope scope, int level)
{
   for (
      ::Reflex::Member_Iterator itype = scope.FunctionMember_Begin();
      itype != scope.FunctionMember_End();
      ++itype
   ) {
         // Indent.
         for (int i=0; i<(level*2); ++i) {
            fprintf(stderr, " ");
         }
         // Dump.
         fprintf(stderr, "function member: '%s' '%s'\n", itype->Name(Reflex::SCOPED | Reflex::QUALIFIED).c_str(), itype->TypeOf().Name(Reflex::SCOPED | Reflex::QUALIFIED).c_str());
   }
}

//______________________________________________________________________________
::Reflex::Type Cint::Internal::G__findInScope(const ::Reflex::Scope scope, const char* name)
{
   // -- Find a REFLEX Type in a REFLEX Scope by name.
   ::Reflex::Type cl;
#ifdef __GNUC__
#else
#pragma message (FIXME("This needs to be in Reflex itself"))
#endif
   for (
      ::Reflex::Type_Iterator itype = scope.SubType_Begin();
      itype != scope.SubType_End();
      ++itype
   ) {
      if (itype->Name() == name) {
         cl = *itype;
         break;
      }
   }
   return cl;
}

//______________________________________________________________________________
bool Cint::Internal::G__test_access(const ::Reflex::Member var, int access)
{
   // Return true if the member (var or function) is accessible
   // with the access level described by 'access'
   switch(access) {
   case G__PUBLIC: return var.IsPublic();
   case G__PROTECTED: return var.IsProtected();
   case G__PRIVATE: return var.IsPrivate();
   case G__GRANDPRIVATE: assert(0);
   case G__PUBLIC_PROTECTED_PRIVATE: return true;
   case G__PUBLIC_PROTECTED: return var.IsProtected() || var.IsPublic();
   }
   return false;
}

//______________________________________________________________________________
bool Cint::Internal::G__is_cppmacro(const ::Reflex::Member var)
{
  const Reflex::Type type = var.TypeOf();
  return (type.Name()  == "macroInt$") || (type.Name()  == "macroDouble$");
}

//______________________________________________________________________________
bool Cint::Internal::G__test_const(const ::Reflex::Type type, int what_const)
{
   switch (what_const) {
      case G__VARIABLE: {
         G__SIGNEDCHAR_T what = G__get_isconst(type);
         return what == what_const;
      }
      case G__CONSTVAR:
      case G__PCONSTVAR:
      case G__PCONSTCONSTVAR:
      case G__CONSTFUNC: {
         G__SIGNEDCHAR_T what = G__get_isconst(type);
         return what & what_const;
      }
      case G__STATICCONST: {
         assert(0); // return var.IsStatic() && var.TypeOf().IsConst();
      }
      case G__LOCKVAR:
         // case G__DYNCONST: aliased to something else!
         assert(0);  // return G__get_properties(var)->lock;
   }
   return false;
}

//______________________________________________________________________________
bool Cint::Internal::G__test_const(const ::Reflex::Member var, int what_const)
{
   switch(what_const){
   case G__VARIABLE:
      {
         G__SIGNEDCHAR_T what = G__get_isconst( var.TypeOf() );
         return what == what_const;
      }
   case G__CONSTVAR:
   case G__PCONSTVAR:
   case G__PCONSTCONSTVAR:
   case G__CONSTFUNC:
      {
         G__SIGNEDCHAR_T what = G__get_isconst( var.TypeOf() );
         return what & what_const;
      }
   case G__STATICCONST:
      {
         return var.IsStatic() && var.TypeOf().IsConst();
      }
   case G__LOCKVAR:
      // case G__DYNCONST: aliased to something else!
      return G__get_properties(var)->lock;
   }
   return false;
}

//______________________________________________________________________________
bool Cint::Internal::G__filescopeaccess(int filenum, int statictype)
{
   int store_filenum = filenum;
   int store_statictype = statictype;

   // Return true if filenum describe a file including the file defining
   // the static variable.
   if (filenum == statictype) {
      return true;
   }

   while (statictype >= 0) {
      statictype = G__srcfile[statictype].included_from;
      if (filenum == statictype) {
         return true;
      }
   }

   // Return true if the static variable is defined in any of the
   // file including the file describe by 'filenum'
   statictype = store_statictype;
   while (statictype >= 0) {
      filenum = store_filenum;
      if (filenum == statictype) {
         return true;
      }
      statictype = G__srcfile[statictype].included_from;
      while (filenum >= 0) {
         if (filenum == statictype) {
            return true;
         }
         filenum = G__srcfile[filenum].included_from;
      }
   }
   return false;
}

//______________________________________________________________________________
bool Cint::Internal::G__test_static(const ::Reflex::Member var, int what_static, int filenum)
{
   switch (what_static) {
      case G__AUTO:
         return !var.IsStatic() && !G__get_properties(var)->isCompiledGlobal;
      case G__LOCALSTATIC:
         return var.IsStatic();
      case G__LOCALSTATICBODY:
         if (var.IsStatic()) {
            ::Reflex::Scope d(var.DeclaringScope());
            if (!(d.IsClass() || d.IsEnum() || d.IsUnion() || d.IsNamespace())) {
               assert((d.Name()[d.Name().length()-1] == '$'));
               // The declaring scope is a block or function scope
               return true;
            }
         }
         return false;
      case G__AUTOARYDISCRETEOBJ:
         return G__get_properties(var)->statictype == G__AUTOARYDISCRETEOBJ;
      case G__COMPILEDGLOBAL:
         return G__get_properties(var)->isCompiledGlobal;
   }


   if (!what_static) {
      // Need to return true if any of the case above are true!

      if (!var.IsStatic()) {
         return true;
      }

      if (var.DeclaringScope().IsClass()) {
         return true;
      }

      int statictype = G__get_properties(var)->filenum;
      int store_filenum = filenum;
      int store_statictype = statictype;

      // Return true if filenum describe a file including the file defining
      // the static variable.
      if (filenum == statictype) {
         return true;
      }

      while (statictype >= 0) {
         statictype = G__srcfile[statictype].included_from;
         if (filenum == statictype) {
            return true;
         }
      }

      // Return true if the static variable is defined in any of the
      // file including the file describe by 'filenum'
      statictype = store_statictype;
      while (statictype >= 0) {
         filenum = store_filenum;
         if (filenum == statictype) {
            return true;
         }
         statictype = G__srcfile[statictype].included_from;
         while (filenum >= 0) {
            if (filenum == statictype) {
               return true;
            }
            filenum = G__srcfile[filenum].included_from;
         }
      }
   }

   return false;
}

//______________________________________________________________________________
Reflex::Type Cint::Internal::G__replace_rawtype(const Reflex::Type target, const Reflex::Type raw)
{
   // copy with modifiers
   if (target != target.ToType()) {
      // FIXME: This recursion can be turned into a loop.
      Reflex::Type out = G__replace_rawtype(target.ToType(), raw);
      if (target.IsTypedef()) {
         out = Reflex::TypedefTypeBuilder(target.Name().c_str(), out);
      }
      else if (target.IsPointer()) {
         out = Reflex::PointerBuilder(out, target.TypeInfo());
      }
      else if (target.IsPointerToMember()) {
         out = Reflex::PointerToMemberBuilder(out, target.PointerToMemberScope(), target.TypeInfo());
      }
      else if (target.IsArray()) {
         out = Reflex::ArrayBuilder(out, target.ArrayLength(), target.TypeInfo());
      }
      // FIXME: We are missing an else if clause here for Reflex::Function.
      //
      // FIXME: This does not properly handle a const volatile type.
      if (target.IsConst()) {
         out = Reflex::ConstBuilder(out);
      }
      if (target.IsVolatile()) {
         out = Reflex::VolatileBuilder(out);
      }
      // KEEP RAW'S! if (target.IsAbstract()) out = Reflex::Type(out, Reflex::ABSTRACT, Reflex::Type::APPEND);
      // KEEP RAW'S! if (target.IsVirtual()) out = Reflex::Type(out, Reflex::ABSTRACT, Reflex::Type::APPEND);
      // KEEP RAW'S! if (target.IsArtificial()) out = Reflex::Type(out, Reflex::ARTIFICIAL, Reflex::Type::APPEND);
      return out;
   }
   return raw;
}

//______________________________________________________________________________
void Cint::Internal::G__set_G__tagnum(const ::Reflex::Scope scope)
{
   if (scope) {
      G__tagnum = scope;
   }
   else {
      G__tagnum = ::Reflex::Scope::GlobalScope();
   }
}

//______________________________________________________________________________
void Cint::Internal::G__set_G__tagnum(const ::Reflex::Type type)
{
   ::Reflex::Scope s = type.RawType();
   G__set_G__tagnum(s);
}

//______________________________________________________________________________
void Cint::Internal::G__set_G__tagnum(const G__value& result)
{
   G__set_G__tagnum(G__value_typenum(result));
}

//______________________________________________________________________________
::Reflex::Member Cint::Internal::G__update_array_dimension(::Reflex::Member member, size_t nelem)
{
   // Modify the data member type to 'set' its dimensions.
   // This is used when parsing something like:
   //   MyClass myvar[][2] = {....};
   // At the point when we see the = we create type with is a pointer to.
   // We need to switch to an array of the 'right' dimension (nelem).

   ::Reflex::Scope varscope(member.DeclaringScope());
   G__RflxVarProperties prop = *G__get_properties(member);
   std::string name(member.Name());
   // FIXME: Remove this assert!
   assert(member.TypeOf().FinalType().IsPointer() || member.TypeOf().FinalType().IsArray());
   ::Reflex::Type newType(::Reflex::ArrayBuilder(member.TypeOf().FinalType().ToType(), nelem));
   size_t offset = member.Offset();
#ifdef __GNUC__
#else
#pragma message (FIXME("Should call ::Reflex::Member::Modifiers()"))
#endif
   int modifiers = 0;
   if (member.IsPublic()) {
      modifiers |= ::Reflex::PUBLIC;
   } else if (member.IsProtected()) {
      modifiers |= ::Reflex::PROTECTED;
   } else if (member.IsPrivate()) {
      modifiers |= ::Reflex::PRIVATE;
   }
   if (member.IsStatic()) {
      modifiers |= ::Reflex::STATIC;
   }
   if (member.IsConst()) {
      modifiers |= ::Reflex::CONST;
   }
   if (member.IsVolatile()) {
      modifiers |= ::Reflex::VOLATILE;
   }
   // etc...
   varscope.RemoveDataMember(member);
   varscope.AddDataMember(name.c_str(), newType, offset, modifiers);
   member = varscope.DataMemberByName(name);
   *G__get_properties(member) = prop;
   return member;
}

//______________________________________________________________________________
int ::Cint::Internal::G__get_access(const ::Reflex::Member mem) 
{
   if (mem.IsPublic()) return G__PUBLIC;
   else if(mem.IsProtected()) return G__PROTECTED;
   else if(mem.IsPrivate()) return G__PRIVATE;
   return 0;
}

//______________________________________________________________________________
int ::Cint::Internal::G__get_static(const ::Reflex::Member mem) 
{
   if (mem.IsDataMember() && G__get_properties(mem)->isCompiledGlobal) {
      return G__COMPILEDGLOBAL;
   }
   ::Reflex::Scope d( mem.DeclaringScope() );
   if (mem.IsStatic() && 
         !(/* d.IsClass() || d.IsEnum() || d.IsUnion() || */ d.IsNamespace() ) ) 
   {
      return G__LOCALSTATIC; // or G__LOCALSTATICBODY
   }
   if (!mem.IsStatic() && mem.TypeOf().FinalType().IsArray()) {
      return G__AUTOARYDISCRETEOBJ;
   }
   if (mem.IsStatic() && d.IsNamespace()) {
      int filenum = -1;
      if (mem.IsDataMember()) {
         filenum = G__get_properties(mem)->filenum;
      } else {
         filenum = G__get_funcproperties(mem)->filenum;
      }
      if (filenum != -1) {
         return filenum;
      }
   }
   return G__AUTO;
}

#include "fproto.h"
#include "common.h"

//______________________________________________________________________________
G__RflxStackProperties::~G__RflxStackProperties()
{
}

//______________________________________________________________________________
G__RflxProperties::~G__RflxProperties()
{
}

//______________________________________________________________________________
G__RflxVarProperties::~G__RflxVarProperties()
{
}

//______________________________________________________________________________
G__RflxFuncProperties::~G__RflxFuncProperties()
{
}

//______________________________________________________________________________
G__funcentry::~G__funcentry()
{
   G__funcentry::clear();
}

//______________________________________________________________________________
void G__funcentry::clear()
{
   if (bytecode) {
      Cint::Internal::G__free_bytecode(bytecode);
      bytecode = 0;
   }
   if (friendtag) {
      Cint::Internal::G__free_friendtag(friendtag);
      friendtag = 0;
   }
   // FIXME: Add deletion of defaults.
}

//______________________________________________________________________________
G__funcentry& G__funcentry::copy(const G__funcentry& orig) 
{
   // This function must NOT be made virtual
   p = orig.p;
   line_number = orig.line_number;
   filenum = orig.filenum;
   pos = orig.pos;
   ptradjust = orig.ptradjust;
#ifdef G__ASM_FUNC
   size = orig.size;
#endif
#ifdef G__TRUEP2F
   for_tp2f = orig.for_tp2f;
   if (for_tp2f.length()) {
      tp2f = (void*)for_tp2f.c_str();
   } else {
      tp2f = orig.tp2f;
   }
#endif
#ifdef G__ASM_WHOLEFUNC
   bytecode = orig.bytecode;
   bytecodestatus = orig.bytecodestatus;
#endif
   ansi = orig.ansi;
   busy = orig.busy;
#ifdef G__FRIEND
   friendtag = Cint::Internal::G__copy_friendtag(orig.friendtag);
#endif
   userparam = orig.userparam;
   vtblindex = orig.vtblindex;
   vtblbasetagnum = orig.vtblbasetagnum;
   para_default = orig.para_default;

   return *this;
}

//______________________________________________________________________________
Cint::Internal::G__BuilderInfo::G__BuilderInfo()
: fBaseoffset(0)
, fAccess(G__PUBLIC)
, fIsconst(0)
, fIsexplicit(0)
, fStaticalloc(0)
, fIsvirtual(-1)
, fIspurevirtual(0)
{
   // -- Used by: v6_newlink.cxx(G__memfunc_setup), and v6_ifunc.cxx(G__make_ifunctable).
}

//______________________________________________________________________________
std::string Cint::Internal::G__BuilderInfo::GetParamNames()
{
   // -- Internal, called only by G__BuilderInfo::Build().
   std::string result;
   for (names_t::const_iterator iter = fParams_name.begin(); iter != fParams_name.end(); ++iter) {
      if (iter != fParams_name.begin()) {
         result += ";";
      }
      result += iter->first;
      if (iter->second.length()) {
         result += "=" + iter->second;
      }
   }
   return result;
}

//______________________________________________________________________________
void Cint::Internal::G__BuilderInfo::ParseParameterLink(const char* paras)
{
   // -- Dictionary Interface, called by v6_newlink.cxx(G__memfunc_setup).
   int type;
   int tagnum;
   int reftype_const;
   ::Reflex::Type typenum;
   G__value* para_default = 0;
   char c_type[10];
   G__StrBuf tagname_sb(G__MAXNAME*6);
   char *tagname = tagname_sb;
   G__StrBuf type_name_sb(G__MAXNAME*6);
   char *type_name = type_name_sb;
   char c_reftype_const[10];
   G__StrBuf c_default_sb(G__MAXNAME*2);
   char *c_default = c_default_sb;
   G__StrBuf c_paraname_sb(G__MAXNAME*2);
   char *c_paraname = c_paraname_sb;
   int os = 0;
   int store_loadingDLL = G__loadingDLL;
   G__loadingDLL = 1;
   int store_var_type = G__var_type;
   char ch = paras[0];
   for (int ifn = 0; ch != '\0'; ++ifn) {
      G__separate_parameter(paras, &os, c_type);
      type = c_type[0];
      G__separate_parameter(paras, &os, tagname);
      if (tagname[0] == '-') {
         tagnum = -1;
      }
      else {
         // -- We have a tagname.
         // save G__p_ifunc, G__search_tagname might change it when autoloading
         ::Reflex::Scope current_G__p_ifunc = G__p_ifunc;
         // Let's make sure that the lookup in done in the global scope.
         ::Reflex::Scope store_def_tagnum = G__def_tagnum;
         G__def_tagnum = Reflex::Scope::GlobalScope();
         if (type == 'i') {
            // -- A parameter of type 'i' with a tagname is an enum.
            // Note: We must have this case because the reflex
            //       ClassBuilder will not replace a TypeBase
            //       with a ScopeBase.
            // Note: May create a dummy TypeBase which will
            //       be replaced later when we see the declaration.
            tagnum = G__search_tagname(tagname, 'e');
         }
         else {
            // -- Could be a class/struct/union/namespace
            // Note: May create a dummy ScopeBase which will
            //       be replaced later when we see the declaration.
            // FIXME: This is the only place where G__search_tagname
            //        is called with a type of zero.
            tagnum = G__search_tagname(tagname, isupper(type) ? 0xff : 0);
         }
         G__def_tagnum = store_def_tagnum;
         G__p_ifunc = current_G__p_ifunc;
      }
      G__separate_parameter(paras, &os, type_name);
      if (type_name[0] == '-') {
         typenum = ::Reflex::Type();
      }
      else {
         if (type_name[0] == '\'') {
            type_name[std::strlen(type_name)-1] = '\0';
            typenum = G__find_typedef(type_name + 1);
         }
         else {
            typenum = G__find_typedef(type_name);
         }
      }
      G__separate_parameter(paras, &os, c_reftype_const);
      reftype_const = std::atoi(c_reftype_const);
#ifndef G__OLDIMPLEMENTATION1861
      //if (typenum) {
      // NO - this is already taken into account when writing the dictionary
      //  reftype_const += G__newtype.isconst[typenum] * 10;
      //}
#endif
      G__separate_parameter(paras, &os, c_default);
      if ((c_default[0] == '-') && (c_default[1] == '\0')) {
         para_default = 0;
         c_default[0] = '\0';
      }
      else {
         para_default = (G__value*) - 1;
      }
      ch = G__separate_parameter(paras, &os, c_paraname);
      if (c_paraname[0] == '-') {
         c_paraname[0] = '\0';
      }
      AddParameter(ifn, type, tagnum, G__get_typenum(typenum), reftype_const, para_default, c_default, c_paraname);
   }
   G__var_type = store_var_type;
   G__loadingDLL = store_loadingDLL;
}

//______________________________________________________________________________
void Cint::Internal::G__BuilderInfo::AddParameter(int /* ifn */, int type, int numerical_tagnum, int numerical_typenum, int reftype_const, G__value* para_default, char* para_def, char* para_name)
{
   // -- Internal, called only by G__BuilderInfo::ParseParameterLink().
   ::Reflex::Type paramType;
   ::Reflex::Type typenum = G__Dict::GetDict().GetTypedef(numerical_typenum);
   ::Reflex::Type tagnum = G__Dict::GetDict().GetScope(numerical_tagnum);
   if (typenum) {
      paramType = typenum;
   } else if (tagnum) {
      paramType = tagnum;
   }
   else {
      paramType = G__get_from_type(type, 0);
   }
   int isconst = (reftype_const / 10) % 10;
   int reftype = reftype_const - (reftype_const / 10 % 10 * 10);
   paramType = G__modify_type(paramType, isupper(type), reftype, isconst, 0, 0);
   fParams_type.push_back(paramType);
   fDefault_vals.push_back(para_default);
   fParams_name.push_back(std::make_pair(para_name, para_def));
}

//______________________________________________________________________________
::Reflex::Member Cint::Internal::G__BuilderInfo::Build(const std::string& name)
{
   // -- Create the reflex database entries for function name.
   //
   //  Make the entry into the reflex database.
   //
   //  Note: The Reflex::TypeBase constructor explicitly does not
   //        add the created function type as a member of any Reflex::Scope.
   //
   ::Reflex::Type ftype = Reflex::FunctionTypeBuilder(fReturnType, fParams_type, typeid(::Reflex::UnknownType));
   //fprintf(stderr, "\nG__BuilderInfo::Build: processing function '%s::%s' type '%s'.\n", G__p_ifunc.Name(Reflex::SCOPED).c_str(), name.c_str(), ftype.Name(Reflex::SCOPED | Reflex::QUALIFIED).c_str());
   //if (!G__p_ifunc.Name().compare("FumiliMinimizer")) {
   //   fprintf(stderr, "G__BuilderInfo::Build: Abstract count for '%s' count: %d  for: '%s'\n", G__p_ifunc.Name(Reflex::SCOPED).c_str(), G__struct.isabstract[G__get_properties(G__tagdefining)->tagnum], name.c_str());
   //}
   //fprintf(stderr, "G__Builderinfo::Build: fIsconst: %d\n", fIsconst);
   //fprintf(stderr, "G__Builderinfo::Build: fIsexplicit: %d\n", fIsexplicit);
   //fprintf(stderr, "G__Builderinfo::Build: fStaticalloc: %d\n", (int) fStaticalloc);
   //fprintf(stderr, "G__Builderinfo::Build: fIsvirtual: %d\n", fIsvirtual);
   //fprintf(stderr, "G__Builderinfo::Build: fIspurevirtual: %d\n", fIspurevirtual);
   //for (
   //   std::vector<Reflex::Type>::iterator iter = fParams_type.begin();
   //   iter != fParams_type.end();
   //   ++iter
   //) {
   //   fprintf(stderr, "G__Builderinfo::Build: param type: %s\n", iter->Name(Reflex::SCOPED | Reflex::QUALIFIED).c_str());
   //}
   //fprintf(stderr, "G__Builderinfo::Build:\n");
   //
   //  Fetch any previously seen declaration of this function in the defining class or namespace.
   //
   //  Note: The name lookup is done without regard to the accessibility of the member.
   //
   //--
   //
   //  Build the reflex modifiers.  // FIXME: Name lookup should be done without regard to modifiers!
   //
   int modifiers = 0;
   if (fIsconst) {
      modifiers |= ::Reflex::CONST;
   }
   if (fIsexplicit) {
      modifiers |= ::Reflex::EXPLICIT;
   }
   if (fStaticalloc) {
      modifiers |= ::Reflex::STATIC;
   }
   if (fIsvirtual) {
      modifiers |= ::Reflex::VIRTUAL;
   }
   if (fIspurevirtual) {
      modifiers |= ::Reflex::ABSTRACT;
   }
   //
   //  Do the access part of the reflex modifiers.
   //
   //switch(access) {
   //   case G__PUBLIC:     modifiers |= ::Reflex::PUBLIC; break;
   //   case G__PROTECTED:  modifiers |= ::Reflex::PROTECTED; break;
   //   case G__PRIVATE:    modifiers |= ::Reflex::PRIVATE; break;
   //};
   ::Reflex::Type modftype(::Reflex::Type(ftype, modifiers));
   unsigned int modifiers_mask = ::Reflex::PUBLIC | ::Reflex::PROTECTED | ::Reflex::PRIVATE | ::Reflex::EXPLICIT | ::Reflex::STATIC | ::Reflex::VIRTUAL | ::Reflex::ABSTRACT | ::Reflex::CONSTRUCTOR | ::Reflex::DESTRUCTOR | ::Reflex::COPYCONSTRUCTOR;
   //fprintf(stderr, "G__BuilderInfo::Build: search for function member '%s' modftype '%s'\n", name.c_str(), modftype.Name(Reflex::SCOPED | Reflex::QUALIFIED).c_str());
   ::Reflex::Member m(G__p_ifunc.FunctionMemberByName(name, modftype, modifiers_mask));
   //if (m) {
   //   fprintf(stderr, "G__BuilderInfo::Build: search found member '%s'\n", m.Name(Reflex::SCOPED | Reflex::QUALIFIED).c_str());
   //}
   //else {
   //   fprintf(stderr, "G__BuilderInfo::Build: search failed.\n");
   //}
   //
   //  If we implement a pure virtual function from
   //  a base class, decrement the pure virtual count
   //  of our defining class.
   //
   //  Check if we should be virtual because we
   //  are declared virtual in a base class.
   //
   if ( // is a member function, and not previously declared
      (G__tagdefining && G__tagdefining.IsClass()) && // is a member function, and
      !m // not previous declared
   ) {
      struct G__inheritance* baseclass = G__struct.baseclass[G__get_tagnum(G__tagdefining)];
      for (int basen = 0; basen < baseclass->basen; ++basen) {
         G__incsetup_memfunc(baseclass->basetagnum[basen]);
         ::Reflex::Scope scope(G__Dict::GetDict().GetScope(baseclass->basetagnum[basen]));
         //fprintf(stderr, "G__BuilderInfo::Build: search base class '%s' for function member '%s' type '%s'\n", scope.Name(Reflex::SCOPED).c_str(), name.c_str(), modftype.Name(Reflex::SCOPED | Reflex::QUALIFIED).c_str());
         ::Reflex::Member base_m(scope.FunctionMemberByNameAndSignature(name, modftype, modifiers_mask));
         //if (base_m) {
         //   fprintf(stderr, "G__BuilderInfo::Build: search found member '%s'\n", base_m.Name(Reflex::SCOPED | Reflex::QUALIFIED).c_str());
         //}
         //else {
         //   fprintf(stderr, "G__BuilderInfo::Build: search failed.\n");
         //}
         if (base_m) {
            int tagnum_tagdefining  = G__get_tagnum(G__tagdefining);
            if (base_m.IsAbstract() && G__struct.isabstract[tagnum_tagdefining]) {
               //fprintf(stderr, "G__BuilderInfo::Build: 1 Decrement abstract count for '%s' count: %d -> %d\n", G__p_ifunc.Name(Reflex::SCOPED).c_str(), G__struct.isabstract[tagnum_tagdefining], G__struct.isabstract[tagnum_tagdefining] - 1);
               --G__struct.isabstract[tagnum_tagdefining]; // We now have one less pure virtual member function.
            }
            if (base_m.IsVirtual()) {
               fIsvirtual = 1; // FIXME: We need to update the reflex entry immediately, we may return in the next block.
               //fprintf(stderr, "G__Builderinfo::Build: changed fIsvirtual to: %d\n", fIsvirtual);
            }
            break;
         }
      }
   }
   //
   //  Return early if we are in the command line, or a temp file.
   //
   if (G__ifile.filenum >= G__nfile) { // in a command line or a temp file
      G__fprinterr(G__serr, "Limitation: Function can not be defined in a command line or a tempfile\n");
      G__genericerror("You need to write it in a source file");
      return m;
   }
   if (m) { // function was previously declared
      // -- Now handle the parts which go into the function property.
      //
      // Consume the friendtag, put it into the function properties friendtag list.
      if (fProp.entry.friendtag) {
         if (!G__get_funcproperties(m)->entry.friendtag) {
            G__get_funcproperties(m)->entry.friendtag = fProp.entry.friendtag;
         }
         else {
            struct G__friendtag* friendtag = G__get_funcproperties(m)->entry.friendtag;
            while (friendtag->next) {
               friendtag = friendtag->next;
            }
            friendtag->next = fProp.entry.friendtag;
         }
         fProp.entry.friendtag = 0;
      }
      // The old code was doing:
      // 
      //      - Warn if any default parameter is redefined.
      //      - Import ansi, explicit, return type, param char type
      //        and ref, and if there is a function body, change the parameter
      //        names, otherwise leave them as is.
      //      - Import concreteness (aka remove pure virtual).
      //      - Copy the 'entry' fields.
      //      - Avoid double counting pure virtual function.
      //
      if (
         (FILE*) fProp.entry.p && // we have file pointer to text of body, and
         (
            !G__def_struct_member || // not a class member, or
            (G__struct.iscpplink[G__get_tagnum(G__def_tagnum)] != G__CPPLINK) // is not precompiled
         )
      ) {
         // Should we issue an error for one of the following condition?
         // if (!m.IsExplicit() && isexplicit)
         // if (m.IsAbstract() != !ispurevirtual)
         // if (m.IsConst() != isconst)
         //m.SetTypeOf(fReturnType);
         if (fIspurevirtual & !m.IsAbstract()) {
            if (G__tagdefining) {
               //fprintf(stderr, "G__BuilderInfo::Build: 2 Decrement abstract count for '%s' count: %d -> %d\n", G__p_ifunc.Name(Reflex::SCOPED).c_str(), G__struct.isabstract[G__get_tagnum(G__tagdefining)], G__struct.isabstract[G__get_tagnum(G__tagdefining)] - 1);
               --G__struct.isabstract[G__get_tagnum(G__tagdefining)];
            }
         }
         struct G__friendtag* prev_friendtag = G__get_funcproperties(m)->entry.friendtag;
         G__get_funcproperties(m)->entry.friendtag = 0;
         std::vector<G__value*> prev_para_default = G__get_funcproperties(m)->entry.para_default;
         G__get_funcproperties(m)->entry.para_default.clear();
         G__get_funcproperties(m)->entry = fProp.entry;
         G__get_funcproperties(m)->entry.friendtag = prev_friendtag;
         G__get_funcproperties(m)->entry.para_default = prev_para_default;
#ifndef GNUC
#pragma message(FIXME("tp2f need to move from char* to be an opaque member::Id"))
#endif
         G__get_funcproperties(m)->entry.for_tp2f = m.Name();
         G__get_funcproperties(m)->entry.tp2f = (void*) G__get_funcproperties(m)->entry.for_tp2f.c_str();
         for (unsigned int i = 0; i < fParams_name.size(); ++i) {
            if (fParams_name[i].second.size()) {
               G__genericerror("Error: Redefinition of default argument");
            }
         }
         //m.UpdateFunctionParameterNames( GetParamNames().c_str() );
      }
      else {
         if (fIspurevirtual && G__tagdefining) {
            //fprintf(stderr, "G__BuilderInfo::Build: 3 Decrement abstract count for '%s' count: %d -> %d\n", G__p_ifunc.Name(Reflex::SCOPED).c_str(), G__struct.isabstract[G__get_tagnum(G__tagdefining)], G__struct.isabstract[G__get_tagnum(G__tagdefining)] - 1);
            --G__struct.isabstract[G__get_tagnum(G__tagdefining)];
         }
      }
      G__func_now = m;
   }
   else if ( // function not seen before, and ok to process
      (
         fProp.entry.p || // we have a file pointer to the function body (definition), or
         fProp.entry.ansi || // ansi-style declaration, or
         G__nonansi_func || // not ansi-style declaration, or // FIXME: What???  Combined with previous clause is always true???
         (G__globalcomp < G__NOLINK) || // is precompiled, or
         fProp.entry.friendtag // is a friend
      )
      // This block is skipped only when compicated template
      // instantiation is done during reading argument list
      // 'f(vector<int> &x) { }'
      //
      // with 1706, do not skip this block with template instantiation
      // in function argument. Do not know exactly why...
      //
      // && ((store_ifunc_tmp == G__p_ifunc) && (func_now == G__p_ifunc->allifunc))
#ifndef G__OLDIMPLEMENTATION2027
      // && (name.c_str()[0] != '~')
#endif // G__OLDIMPLEMENTATION2027
      // --
   ) {
      int modifiers = 0;
      if (fIsconst) {
         modifiers |= ::Reflex::CONST;
      }
      if (fIsexplicit) {
         modifiers |= ::Reflex::EXPLICIT;
      }
      if (fStaticalloc) {
         modifiers |= ::Reflex::STATIC;
      }
      if (fIsvirtual) {
         modifiers |= ::Reflex::VIRTUAL;
      }
      if (fIspurevirtual) {
         modifiers |= ::Reflex::ABSTRACT;
      }
      switch (fAccess) {
         case G__PUBLIC:
            modifiers |= ::Reflex::PUBLIC;
            break;
         case G__PROTECTED:
            modifiers |= ::Reflex::PROTECTED;
            break;
         case G__PRIVATE:
            modifiers |= ::Reflex::PRIVATE;
            break;
      };
      switch (G__get_tagtype(G__p_ifunc)) {
         case 'c':
         case 's':
            // -- Class or struct.
            //
            //  Do special processing for a constructor/destructor.
            //
            if (name[0] == '~') {
               modifiers |= ::Reflex::DESTRUCTOR;
            }
            else if (name == G__p_ifunc.Name()) { // FIXME: Is this correct for templated classes?
               modifiers |= ::Reflex::CONSTRUCTOR;
               if (ftype.FunctionParameterSize() == 1) {
                  // Get the type without any modifiers
                  // (i.e., reference and const) nor any typedefs.
                  Reflex::Type argtype(ftype.FunctionParameterAt(0).FinalType().ToTypeBase()->ThisType());
                  if (argtype == (Reflex::Type) G__p_ifunc) {
                     modifiers |= ::Reflex::COPYCONSTRUCTOR;
                  }
               }
            }
            G__get_properties(G__p_ifunc)->builder.Class().AddFunctionMember(ftype, name.c_str(), 0 /*stubFP*/, 0 /*stubCtx*/, GetParamNames().c_str(), modifiers);
            modftype = ::Reflex::Type(ftype, modifiers);
            m = G__p_ifunc.FunctionMemberByName(name, modftype, modifiers_mask);
            //fprintf(stderr, "G__BuilderInfo::Build: added member function '%s'.\n", m.Name(Reflex::SCOPED | Reflex::QUALIFIED).c_str());
            break;
         case 'n':
            // -- Namespace.
            {
               std::string fullname = G__p_ifunc.Name(::Reflex::SCOPED);
               if (fullname.size()) {
                  fullname += "::";
               }
               fullname += name;
               ::Reflex::FunctionBuilder funcBuilder(ftype, fullname.c_str(), 0 /*stubFP*/, 0 /*stubCtx*/, GetParamNames().c_str(), modifiers);
               m = funcBuilder.ToMember();
               break;
            }
         case 'u':
            // -- Union.
            //
            //  Do special processing for a constructor/destructor.
            //
            if (name[0] == '~') {
               modifiers |= ::Reflex::DESTRUCTOR;
            }
            else if (name == G__p_ifunc.Name()) { // FIXME: Is this correct for templated classes?
               modifiers |= ::Reflex::CONSTRUCTOR;
               if (ftype.FunctionParameterSize() == 1) {
                  // Get the type without any modifiers
                  // (i.e., reference and const) nor any typedefs.
                  Reflex::Type argtype(ftype.FunctionParameterAt(0).FinalType().ToTypeBase()->ThisType());
                  if (argtype == (Reflex::Type) G__p_ifunc) {
                     modifiers |= ::Reflex::COPYCONSTRUCTOR;
                  }
               }
            }
            G__get_properties(G__p_ifunc)->builder.Union().AddFunctionMember(ftype, name.c_str(), 0 /*stubFP*/, 0 /*stubCtx*/, GetParamNames().c_str(), modifiers);
            modftype = ::Reflex::Type(ftype, modifiers);
            m = G__p_ifunc.FunctionMemberByName(name, modftype, modifiers_mask);
            //fprintf(stderr, "G__BuilderInfo::Build: added member function '%s'.\n", m.Name(Reflex::SCOPED | Reflex::QUALIFIED).c_str());
            break;
         default:
            // -- Assume this means we want it in the global namespace.
            ::Reflex::FunctionBuilder funcBuilder(ftype, name.c_str(), 0 /*stubFP*/, 0 /*stubCtx*/, GetParamNames().c_str(), modifiers);
            m = (funcBuilder.ToMember());
            //int tag = G__get_tagtype( G__p_ifunc ) ;
            //fprintf(stderr,"value=%d\n",tag);
            //G__genericerror("Attempt to add a function to something that does not appear to be a scope");
      }
      //::Reflex::FunctionBuilder funcBuilder(ftype, fullname.c_str(), 0 /*stubFP*/, 0 /*stubCtx*/, GetParamNames().c_str(), modifiers);
      //m = funcBuilder.ToMember();
      if (!m) {
         fprintf(stderr, "G__BuilderInfo::Build: Something went wrong creating entry for '%s' in '%s'\n", name.c_str(), G__p_ifunc.Name(::Reflex::SCOPED | ::Reflex::QUALIFIED).c_str());
         G__dump_reflex_function(G__p_ifunc, 1);
         return m;
      }
      G__RflxFuncProperties* a = G__get_funcproperties(m);
      //fprintf(stderr, "G__BuilderInfo::Build: a: %p  %s:%d\n", a, __FILE__, __LINE__);
      //fprintf(stderr, "G__BuilderInfo::Build: fProp.entry.ptradjust: %ld  %s:%d\n", fProp.entry.ptradjust, __FILE__, __LINE__);
      *a = fProp;
      //fprintf(stderr, "G__BuilderInfo::Build: a->entry.ptradjust: %ld  %s:%d\n", a->entry.ptradjust, __FILE__, __LINE__);
      a->entry.para_default = fDefault_vals;
      if (!a->entry.tp2f) {
         a->entry.for_tp2f = m.Name();
         a->entry.tp2f = (void*) a->entry.for_tp2f.c_str();
      }
   }
   //else {
   //   fprintf(stderr, "G__BuilderInfo::Build: Skipping creation of function properties for '%s'  %s:%d\n", name.c_str(), __FILE__, __LINE__);
   //   fprintf(stderr, "G__BuilderInfo::Build:    fProp.entry.p: 0x%lx  %s:%d\n", (long) fProp.entry.p, __FILE__, __LINE__);
   //   fprintf(stderr, "G__BuilderInfo::Build:    fProp.entry.ansi: %d  %s:%d\n", fProp.entry.ansi, __FILE__, __LINE__);
   //   fprintf(stderr, "G__BuilderInfo::Build:    G__nonansi_func: %d  %s:%d\n", G__nonansi_func, __FILE__, __LINE__);
   //   fprintf(stderr, "G__BuilderInfo::Build:    G__globalcomp < G__NOLINK: %d %d  %s:%d\n", G__globalcomp, G__globalcomp < G__NOLINK, __FILE__, __LINE__);
   //   fprintf(stderr, "G__BuilderInfo::Build:    fProp.entry.friendtag: %d  %s:%d\n", fProp.entry.friendtag, __FILE__, __LINE__);
   //}
   return m;
}

