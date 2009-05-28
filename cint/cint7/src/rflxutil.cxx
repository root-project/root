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
//#include "Reflex/Builder/FunctionBuilder.h"
#include "Reflex/Tools.h"
#include "Dict.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <string>

using namespace std;

#include "Reflex/internal/MemberBase.h"
#include "Reflex/internal/TypeBase.h"

//
// Function Directory
//

// Cint internal functions.
namespace Cint {
namespace Internal {
size_t GetReflexPropertyID();
void G__get_cint5_type_tuple(const ::Reflex::Type in_type, char* out_type, int* out_tagnum, int* out_typenum, int* out_reftype, int* out_constvar);
void G__get_cint5_type_tuple_long(const ::Reflex::Type in_type, long* out_type, long* out_tagnum, long* out_typenum, long* out_reftype, long* out_constvar);
int G__get_cint5_typenum(const ::Reflex::Type in_type);
int G__get_tagtype(const ::Reflex::Type in);
int G__get_tagtype(const ::Reflex::Scope in);
int G__get_reftype(const ::Reflex::Type in);
G__RflxProperties* G__get_properties(const ::Reflex::Type in);
G__RflxProperties* G__get_properties(const ::Reflex::Scope in);
G__RflxVarProperties* G__get_properties(const ::Reflex::Member in);
G__RflxFuncProperties* G__get_funcproperties(const ::Reflex::Member in);
G__SIGNEDCHAR_T G__get_isconst(const ::Reflex::Type in);
int G__get_nindex(const ::Reflex::Type passed_type);
std::vector<int> G__get_index(const ::Reflex::Type passed_var);
//std::vector<int> G__get_varlabel_as_vector(const ::Reflex::Type passed_var);
int G__get_varlabel(const Reflex::Member var, int idx);
int G__get_varlabel(const Reflex::Type passed_var, int idx);
int G__get_typenum(const ::Reflex::Type in);
int G__get_tagnum(const ::Reflex::Type in);
int G__get_tagnum(const ::Reflex::Scope in);
::Reflex::Type G__get_from_type(int type, int createpointer, int isconst /*=0*/);
int G__get_paran(const Reflex::Member var);
size_t G__get_bitfield_width(const Reflex::Member var);
size_t G__get_bitfield_start(const Reflex::Member var);
Reflex::Type G__strip_array(const Reflex::Type typein);
Reflex::Type G__strip_one_array(const Reflex::Type typein);
Reflex::Type G__deref(const Reflex::Type typein);
::Reflex::Type G__modify_type(const ::Reflex::Type typein, bool ispointer, int reftype, int isconst, int nindex, int* index);
::Reflex::Type G__cint5_tuple_to_type(int type, int tagnum, int typenum, int reftype, int isconst);
::Reflex::Scope G__findInScope(const ::Reflex::Scope scope, const char* name);
bool G__test_access(const ::Reflex::Member var, int access);
bool G__is_cppmacro(const ::Reflex::Member var);
bool G__filescopeaccess(int filenum, int statictype);
Reflex::Type G__replace_rawtype(const Reflex::Type target, const Reflex::Type raw);
Reflex::Type G__apply_const_to_typedef(const Reflex::Type target);
void G__set_G__tagnum(const ::Reflex::Scope scope);
void G__set_G__tagnum(const ::Reflex::Type type);
void G__set_G__tagnum(const G__value& result);
::Reflex::Member G__update_array_dimension(::Reflex::Member member, size_t nelem);
int G__get_access(const ::Reflex::Member mem);
} // namespace Cint
} // namespace Internal

// C interface functions.
extern "C" {
int G__value_get_type(G__value *buf);
int G__value_get_tagnum(G__value *buf);
void G__dump_reflex();
void G__dump_reflex_atlevel(const ::Reflex::Scope scope, int level);
void G__dump_reflex_function(const ::Reflex::Scope scope, int level);
}

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
void Cint::Internal::G__get_cint5_type_tuple(const ::Reflex::Type in_type, char* out_type, int* out_tagnum, int* out_typenum, int* out_reftype, int* out_isconst)
{
   // Return the cint5 type tuple for a reflex type.
   //
   // Note: Some C++ types cannot be represented by a cint5 type tuple.
   //
   //--
   if (!in_type) {
      *out_type = '\0';
      *out_tagnum = -1;
      *out_typenum = -1;
      *out_reftype = 0;
      *out_isconst = G__CONSTVAR;
      return;
   }
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
            if (prop->tagnum) { // The global scope is tagnum 0 and should be reported as -1.
               *out_tagnum = prop->tagnum;
            }
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
      for (; current && current.IsArray();) {
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
      *out_isconst = G__get_isconst(in_type);
      //if (in_type.IsFunction()) {
      //   if (in_type.IsConst()) {
      //      *out_isconst |= G__CONSTFUNC;
      //   }
      //   if (in_type.ReturnType().IsConst()) {
      //      *out_isconst |= G__CONSTVAR;
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
      //   //            *out_isconst |= G__PCONSTVAR;
      //   //         }
      //   //      }
      //   //   }
      //   //}
      //   //if (accumulated_const) {
      //   //   *out_isconst |= G__CONSTVAR;
      //   //}
      //   if (in_type.IsConst() && in_type.IsPointer()) {
      //      *out_isconst |= G__PCONSTVAR;
      //   }
      //   if (in_type.RawType().IsConst()) {
      //      *out_isconst |= G__CONSTVAR;
      //   }
      //}
   }
}

//______________________________________________________________________________
void Cint::Internal::G__get_cint5_type_tuple_long(const ::Reflex::Type in_type, long* out_type, long* out_tagnum, long* out_typenum, long* out_reftype, long* out_isconst)
{
   char type = '\0';
   int tagnum = -1;
   int typenum = -1;
   int reftype = 0;
   int isconst = 0;
   G__get_cint5_type_tuple(in_type, &type, &tagnum, &typenum, &reftype, &isconst);
   *out_type = static_cast<long>(type);
   *out_tagnum = static_cast<long>(tagnum);
   *out_typenum = static_cast<long>(typenum);
   *out_reftype = static_cast<long>(reftype);
   *out_isconst = static_cast<long>(isconst);
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
   if (!in_type) {
      abort();
   }
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
extern "C" int G__value_get_tagnum(G__value *buf)
{
   return Cint::Internal::G__get_tagnum(G__value_typenum(*buf));
}

//______________________________________________________________________________
int Cint::Internal::G__get_tagtype(const ::Reflex::Type in)
{
   // Get CINT type code for a structure (c,s,e,u).
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
   // Get CINT type code for a structure (c,s,e,u).
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
   bool isref = in.IsReference();
   ::Reflex::Type current = in; 
   while (!isref && current && current.IsTypedef()) {
      current = current.ToType();
      isref = current.IsReference();
   }
   while (current && current.IsArray()) {
      current = current.ToType();
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
   // Get REFLEX property list from a data type.
   if (!in) {
      abort();
      //return 0;
   }
   static size_t pid = GetReflexPropertyID();
   G__RflxProperties* p = (G__RflxVarProperties*)(in.ToTypeBase()->Properties().PropertyValue(pid).Address());
   if (!p) {
      G__set_properties(in, G__RflxProperties());
      p  = (G__RflxProperties*)(in.ToTypeBase()->Properties().PropertyValue(pid).Address());
      if (!p) {
         abort();
      }
    }
    return p;
}

//______________________________________________________________________________
G__RflxProperties* Cint::Internal::G__get_properties(const ::Reflex::Scope in)
{
   // -- Get REFLEX property list from a scope.
   if (!in) {
      abort();
      //return 0;
   }
   static size_t pid = GetReflexPropertyID();
   G__RflxProperties* p = (G__RflxProperties*)(in.ToScopeBase()->Properties().PropertyValue(pid).Address());
   if (!p) {
      G__set_properties(in, G__RflxProperties());
      p  = (G__RflxProperties*)(in.ToScopeBase()->Properties().PropertyValue(pid).Address());
      if (!p) {
         abort();
      }
   }
   return p;
}

//______________________________________________________________________________
G__RflxVarProperties* Cint::Internal::G__get_properties(const ::Reflex::Member in)
{
   // -- Get REFLEX property list from a class member.
   if (!in || !in.IsDataMember()) {
      abort();
      //return 0;
   }
#if 0
   size_t pid = GetReflexPropertyID();
   if (!in.Properties().HasProperty(pid)) {
      G__set_properties(in, G__RflxVarProperties());
   }
   G__RflxVarProperties* p = ::Reflex::any_cast<G__RflxVarProperties>(&in.Properties().PropertyValue(pid));
   if (!p) {
      abort();
   }
#else
   static size_t pid = GetReflexPropertyID();
   G__RflxVarProperties* p = (G__RflxVarProperties*)(in.ToMemberBase()->Properties().PropertyValue(pid).Address());
   if (!p) {
      G__set_properties(in, G__RflxVarProperties());  
      p  = (G__RflxVarProperties*)(in.ToMemberBase()->Properties().PropertyValue(pid).Address());
      if (!p) {
         abort();
      }
   }
#endif
   return p;
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
      abort();
      //return 0;
   }
   static size_t pid = GetReflexPropertyID();
   G__RflxFuncProperties* p = (G__RflxFuncProperties*)(in.ToMemberBase()->Properties().PropertyValue(pid).Address());
   if (!p) {
      G__set_properties(in, G__RflxFuncProperties());
      p  = (G__RflxFuncProperties*)(in.ToMemberBase()->Properties().PropertyValue(pid).Address());
      if (!p) {
        abort();
      }
   }
   return p;
}

//______________________________________________________________________________
G__SIGNEDCHAR_T Cint::Internal::G__get_isconst(const ::Reflex::Type given_type)
{
   // Is data type const qualified?
   //
   // G__VARIABLE       0  UNUSED
   // G__CONSTVAR       1  const int x;
   // G__LOCKVAR        2  UNUSED
   // G__DYNCONST       2  const int x = MyFunc(); Initializer uses function call, cannot be initialized at prerun.
   // G__PCONSTVAR      4  int* const x;
   // G__PCONSTCONSTVAR 5  UNUSED
   // G__CONSTFUNC      8  int f() const; Only for function return variables, marks a const member function.
   // G__STATICCONST 0x10  UNUSED, constant literal
   //
   ::Reflex::Type current = given_type;
   char result = '\0';
   if (given_type.IsFunction()) {
      if (given_type.IsConst()) {
         result = (char) G__CONSTFUNC;
      }
      current = given_type.ReturnType();
   }
   if (!current) {
      return result;
   }
   //
   //  Loop over the type nodes from the outside
   //  in, which is the same as reading the C++ type
   //  declaration from right to left.
   //
   bool seen_first_pointer = false;
   bool last_const = false;
   for (; current; current = current.ToType()) {
      if ((current.IsPointer())) {
         if (!seen_first_pointer) {
            seen_first_pointer = true;
            result |= current.IsConst() * G__PCONSTVAR; // We have int* const x;
         }
         last_const = false; // Ignore any const's we may have seen up to now.
      } else {
         last_const |= current.IsConst(); // Accumulate const qualifiers, move them all to the left.
      }
   }
   if (last_const) { // We had a const qualifier after the last pointer node seen.
      result |= G__CONSTVAR; // We have const int x;
   }
   return result;
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
std::vector<int> Cint::Internal::G__get_varlabel_as_vector(const ::Reflex::Type passed_var)
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
   if (!in || !in.IsTypedef()) {
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
   if (!in) {
      return -1;
   }
   ::Reflex::Type ty = in.RawType();
   if (!ty) {
      return -1;
   }
   G__RflxProperties* prop = G__get_properties(ty);
   if (!prop) {
      return -1;
   }
   if (!prop->tagnum) { // Global scope is tagnum 0, and should be reported as -1.
      return -1;
   }
   return prop->tagnum;
}

//______________________________________________________________________________
int Cint::Internal::G__get_tagnum(const ::Reflex::Scope in)
{
   // -- Get CINT tagnum for a given scope.
   // Note: Only class scopes have a tagnum.
   if (!in) {
      return -1;
   }
   G__RflxProperties* prop = G__get_properties(in);
   if (!prop) {
      return -1;
   }
   if (!prop->tagnum) { // Global scope is tagnum 0, and should be reported as -1.
      return -1;
   }
   return prop->tagnum;
}

//______________________________________________________________________________
::Reflex::Type Cint::Internal::G__get_from_type(int type, int createpointer, int isconst /*=0*/)
{
   // -- Get a REFLEX Type from a CINT type.
   // Note: The typenum is tried first, then tagnum, then type.
   // We should NOT consider type q and a as pointer (ever).

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
   static Reflex::Type typeCache[] = {
      Reflex::PointerToMemberBuilder(Reflex::Type(), Reflex::Scope()), // 'a'
      Reflex::Type::ByName("unsigned char"), // 'b'
      Reflex::Type::ByName("char"), // 'c'
      Reflex::Type::ByName("double"), // 'd'
      Reflex::Type::ByName("FILE"), // 'e'
      Reflex::Type::ByName("float"), // 'f'
      Reflex::Type::ByName("bool"), // 'g'
      Reflex::Type::ByName("unsigned int"), // 'h'
      Reflex::Type::ByName("int"), // 'i'
      Reflex::Type(), // 'j' - macro$""
      Reflex::Type::ByName("unsigned long"), // 'k'
      Reflex::Type::ByName("long"), // 'l'
      Reflex::Type::ByName("unsigned long long"), // 'm'
      Reflex::Type::ByName("long long"), // 'n'
      Reflex::Type(), // 'o' - "autoInt$"
      Reflex::Type(), // 'p' - "macroInt$"
      Reflex::Type::ByName("long double"), // 'q'
      Reflex::Type::ByName("unsigned short"), // 'r'
      Reflex::Type::ByName("short"), // 's'
      Reflex::Type(), // 't' - "#define"
      Reflex::Type(), // 'u' - "enum"
      Reflex::Type(), // 'v'
      Reflex::Type(), // 'w'
      Reflex::Type(), // 'x'
      Reflex::Type::ByName("void"), // 'y'
      Reflex::Type(), // 'z' - "switchDefault$"
      Reflex::Type(), // '\0'
      Reflex::Type(), // \001 - "blockBreakContinueGoto$"
      Reflex::Type(), // G__DEFAULT_FUNCCALL = "defaultFunccall$"
#ifndef G__OLDIMPLEMENTATION2191
      Reflex::Type::ByName("void") // '1'
#endif
   };

   static std::vector<Reflex::Type> typePCache;
   static std::vector<Reflex::Type> typeCPCache;
   if (typePCache.empty()) {
      const size_t numtypes = sizeof(typeCache)/sizeof(Reflex::Type);
      typePCache.resize(numtypes);
      typeCPCache.resize(numtypes);
      for (size_t i = 0; i < numtypes; ++i) {
         switch (i + 'A') {
         case 'Z': 
            typePCache[i] = Reflex::Type(); // "codeBreak$"
            typeCPCache[i] = typePCache[i];
            break;
         case 'P':
            typePCache[i] = Reflex::Type(); // "macroDouble$"
            typeCPCache[i] = typePCache[i];
            break;
         case 'O':
            typePCache[i] = Reflex::Type(); // "autoDouble$"
            typeCPCache[i] = typePCache[i];
            break;
         case 'T':
            typePCache[i] = Reflex::Type(); // "macroChar*$"
            typeCPCache[i] = typePCache[i];
            break;
         case 'A':
            typePCache[i] = Reflex::Type();
            typeCPCache[i] = typePCache[i];
            break;
         default:
            if (typeCache[i]) {
               typePCache[i] = Reflex::PointerBuilder(typeCache[i]);
               typeCPCache[i] = Reflex::Type(typeCache[i], Reflex::CONST,
                                             Reflex::Type::APPEND);
               typeCPCache[i] = Reflex::PointerBuilder(typeCPCache[i]);
            }
            break;
         }
      }
   }

   Reflex::Type raw;
   if (type >= 'a' && type <= 'z' ) {
      raw = typeCache[type - 'a'];
      if (!raw) {
         // special macro, not yet set up
         switch (type) {
         case 'j':
            raw = typeCache['j' - 'a'] = Reflex::Type::ByName("macro$");
            break;
         case 'o':
            raw = typeCache['o' - 'a'] = Reflex::Type::ByName("autoInt$");
            break;
         case 'p':
            raw = typeCache['p' - 'a'] = Reflex::Type::ByName("macroInt$");
            break;
         case 't':
            raw = typeCache['t' - 'a'] = Reflex::Type::ByName("#define");
            break;
         case 'u':
            raw = typeCache['u' - 'a'] = Reflex::Type::ByName("enum");
            break;
         case 'z':
            raw = typeCache['z' - 'a'] = Reflex::Type::ByName("switchDefault$");
            break;
         case 'e':
            raw = typeCache['e' - 'a'] = Reflex::Type::ByName("FILE");
            break;            
         }
      }
      if (isconst & G__CONSTVAR) {
         raw = Reflex::Type(raw, Reflex::CONST, Reflex::Type::APPEND);
      }      
   } else if (type >= 'A' && type <= 'Z') {
      if (type=='E') {
         if (createpointer) {
            if (isconst & G__CONSTVAR) {
               raw = typeCPCache['E' - 'A'];            
               if (!raw) {
                  raw = typeCPCache['E' - 'A'] = Reflex::PointerBuilder(Reflex::Type(Reflex::Type::ByName("FILE"), Reflex::CONST, Reflex::Type::APPEND));
               }
            } else {
               raw = typePCache['E' - 'A'];
               if (!raw) {
                  raw = typePCache['E' - 'A'] = Reflex::PointerBuilder(Reflex::Type::ByName("FILE"));
               }
            }
         } else {
            raw = typeCache['E' - 'A'];
            if (!raw) {
               raw = typeCache['E' - 'A'] = Reflex::Type::ByName("FILE");
            }           
            if (isconst & G__CONSTVAR) {
               raw = Reflex::Type(raw, Reflex::CONST, Reflex::Type::APPEND);
            }
         }
         return raw;
      } else if (type == 'Z' || type == 'P' || type == 'O' || type == 'T' || type == 'A') {
         raw = typePCache[type - 'A'];
         if (!raw) {
            // special macro, not yet set up
            switch (type) {
            case 'Z':
               raw = typePCache['Z' - 'A'] = Reflex::Type::ByName("codeBreak$");
               break;
            case 'P':
               raw = typePCache['P' - 'A'] = Reflex::Type::ByName("macroDouble$");
               break;
            case 'O':
               raw = typePCache['O' - 'A'] = Reflex::Type::ByName("autoDouble$");
               break;
            case 'T':
               raw = typePCache['T' - 'A'] = Reflex::Type::ByName("macroChar*$");
               break;
            }
         }
         return raw;
      }
      if (createpointer) {
         if (isconst & G__CONSTVAR) { 
            raw = typeCPCache[type - 'A'];
         } else {
            raw = typePCache[type - 'A'];
         }
      } else {
         raw = typeCache[type - 'A'];
         if (isconst & G__CONSTVAR) {
            raw = Reflex::Type(raw, Reflex::CONST, Reflex::Type::APPEND);
         }
      }
   } else {
      switch (type) {
      case '\0':
         return typeCache['z' - 'a' + 1];
      case '\001':
         raw = typeCache['z' - 'a' + 2];
         if (!raw) {
            return typeCache['z' - 'a' + 2] = Reflex::Type::ByName("blockBreakContinueGoto$");
         }
         return raw;
      case G__DEFAULT_FUNCCALL:
         raw = typeCache['z' - 'a' + 3];
         if (!raw) {
            return typeCache['z' - 'a' + 3] = Reflex::Type::ByName("defaultFunccall$");
         }
         return raw;
#ifndef G__OLDIMPLEMENTATION2191
      case '1':
         return typeCache['z' - 'a' + 4];
#endif
      default: break;
      }
   }

   return raw;
}

#if 0
//______________________________________________________________________________
::Reflex::Type Cint::Internal::G__get_from_type(int type, int createpointer, int isconst /*=0*/)
{
   // Get a reflex type from a cint type.
   static ::Reflex::Type void_type(::Reflex::Type::ByTypeInfo(typeid(void)));
   static ::Reflex::Type bool_type(::Reflex::Type::ByTypeInfo(typeid(bool)));
   static ::Reflex::Type uchar_type(::Reflex::Type::ByTypeInfo(typeid(unsigned char)));
   static ::Reflex::Type char_type(::Reflex::Type::ByTypeInfo(typeid(char)));
   static ::Reflex::Type ushort_type(::Reflex::Type::ByTypeInfo(typeid(unsigned short)));
   static ::Reflex::Type short_type(::Reflex::Type::ByTypeInfo(typeid(short)));
   static ::Reflex::Type uint_type(::Reflex::Type::ByTypeInfo(typeid(unsigned int)));
   static ::Reflex::Type int_type(::Reflex::Type::ByTypeInfo(typeid(int)));
   static ::Reflex::Type ulong_type(::Reflex::Type::ByTypeInfo(typeid(unsigned long)));
   static ::Reflex::Type long_type(::Reflex::Type::ByTypeInfo(typeid(long)));
   static ::Reflex::Type ulonglong_type(::Reflex::Type::ByTypeInfo(typeid(unsigned long long)));
   static ::Reflex::Type longlong_type(::Reflex::Type::ByTypeInfo(typeid(long long)));
   static ::Reflex::Type float_type(::Reflex::Type::ByTypeInfo(typeid(float)));
   static ::Reflex::Type double_type(::Reflex::Type::ByTypeInfo(typeid(double)));
   static ::Reflex::Type longdouble_type(::Reflex::Type::ByTypeInfo(typeid(long double)));
   //static ::Reflex::Type switchStart(::Reflex::Type::ByName("switchStart$"));
   static ::Reflex::Type switchDefault(::Reflex::Type::ByName("switchDefault$"));
   static ::Reflex::Type rootSpecial(::Reflex::Type::ByName("rootSpecial$"));
   static ::Reflex::Type blockBreakContinueGoto(::Reflex::Type::ByName("blockBreakContinueGoto$"));
   static ::Reflex::Type macroInt(::Reflex::Type::ByName("macroInt$"));
   static ::Reflex::Type macro(::Reflex::Type::ByName("macro$"));
   static ::Reflex::Type macroDouble(::Reflex::Type::ByName("macroDouble$"));
   static ::Reflex::Type autoInt(::Reflex::Type::ByName("autoInt$"));
   static ::Reflex::Type autoDouble(::Reflex::Type::ByName("autoDouble$"));
   static ::Reflex::Type macroCharStar(::Reflex::Type::ByName("macroChar*$"));
   static ::Reflex::Type defaultFunccall(::Reflex::Type::ByName("defaultFunccall$"));
   static ::Reflex::Type ptr_to_mbr_type(::Reflex::PointerToMemberBuilder(::Reflex::Type(), ::Reflex::Scope()));
   switch (type) {
      case '\001': return blockBreakContinueGoto;
      case G__DEFAULT_FUNCCALL: return defaultFunccall;
      case 'O': return autoDouble;
      case 'P': return macroDouble;
      case 'T': return macroCharStar;
      case 'Z': return rootSpecial;
      case 'a': return ptr_to_mbr_type;
      //case 'a': return switchStart;
      case 'j': return macro;
      case 'o': return autoInt;
      case 'p': return macroInt;
      case 'z': return switchDefault;
   }
   ::Reflex::Type raw;
   switch (tolower(type)) {
      case '1': raw = void_type; break; // ptr to function
      //case 'a': // handled above
      case 'b': raw = uchar_type; break;
      case 'c': raw = char_type; break;
      case 'd': raw = double_type; break;
      //case 'e': // FILE
      case 'f': raw = float_type; break;
      case 'g': raw = bool_type; break;
      case 'h': raw = uint_type; break;
      case 'i': raw = int_type; break;
      //case 'j': // handled above
      case 'k': raw = ulong_type; break;
      case 'l': raw = long_type; break;
      case 'm': raw = ulonglong_type; break;
      case 'n': raw = longlong_type; break;
      //case 'o': // handled above
      //case 'p': // handled above
      case 'q': raw = longdouble_type; break;
      case 'r': raw = ushort_type; break;
      case 's': raw = short_type; break;
      //case 't': // obsolete, see below
      //case 'u': // class, enum, namespace, struct, union
      //case 'v': // unused
      //case 'w': // logic
      //case 'x': // unused
      case 'y': raw = void_type; break;
      //case 'z': // handled above
   }
   if (createpointer & !isupper(type)) {
      createpointer = 0;
   }
   if (isconst & G__CONSTVAR) { 
      raw = ::Reflex::Type(raw, Reflex::CONST, Reflex::Type::APPEND);
   }
   if (createpointer) {
      raw = ::Reflex::PointerBuilder(raw);
   }
   return raw;
}
#endif // 0

//______________________________________________________________________________
int Cint::Internal::G__get_paran(const Reflex::Member var)
{
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
   // -- Modify a reflex type by applying pointer, reference, const and array bounds.
   //
   // Note: ispointer and reftype are required to be consistent.
   //       If reftype is G__PARAP2P or greater, then ispointer
   //       must be set or we will do the wrong thing.
   //
   int ref = G__REF(reftype);
   reftype = G__PLVL(reftype);
   ::Reflex::Type result = typein; // Start with the given type.
   if (isconst & G__CONSTVAR) { // Add a const qualification to the end node of the chain.
      if (G__get_cint5_typenum(typein) == -1) { // No typedefs in chain.
         result = G__replace_rawtype(result, ::Reflex::Type(result.RawType(), ::Reflex::CONST, Reflex::Type::APPEND));
      }
      else {
         result = G__apply_const_to_typedef(result);
      }
   }
   if (ispointer) { // Apply the first level of pointers.
      result = ::Reflex::PointerBuilder(result);
   }
   switch (reftype) { // Apply the rest of the pointer levels, and the reference.
      case G__PARANORMAL:
         break;
      case G__PARAREFERENCE: // Cannot happen, we removed this above by using G__PLVL.
         //strcpy(string+strlen(string),"&");
         result = ::Reflex::Type(result, ::Reflex::REFERENCE, Reflex::Type::APPEND);
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
   if ((isconst & G__PCONSTVAR) && ((reftype >= G__PARAREFERENCE) || ispointer)) { // Apply const pointer qualifier.
      result = ::Reflex::Type(result, ::Reflex::CONST, Reflex::Type::APPEND);
   }
   if (ref) { // Apply reference.
      result = ::Reflex::Type(result, ::Reflex::REFERENCE, Reflex::Type::APPEND);
   }
   if (nindex) { // Now make an array.
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
      // That is: a is an array of two arrays of 3 ints.
      //
      for (int i = nindex - 1; i >= 0; --i) {
         result = ::Reflex::ArrayBuilder(result, index[i]);
      }
   }
   return result;
}

//______________________________________________________________________________
::Reflex::Type Cint::Internal::G__cint5_tuple_to_type(int type, int tagnum, int typenum, int reftype, int isconst)
{
   int ref = G__REF(reftype);
   reftype = G__PLVL(reftype);
   ::Reflex::Type result;
   if (typenum != -1) { // Typedef type.
      result = G__Dict::GetDict().GetTypedef(typenum);
      if (!result) {
         return result;
      }
      //
      //  Adjust the given qualifiers to subtract the
      //  parts which came from the typedef.
      //
      ::Reflex::Type final = result.FinalType();
      if (G__get_isconst(final) & G__CONSTVAR) {
         isconst &= ~G__CONSTVAR;
      }
      if (final.IsPointer() && isupper(type)) {
         switch (G__get_reftype(final)) {
            case G__PARANORMAL: // final is a one level pointer.
            case G__PARAREFERENCE: // *cannot happen*
               switch (reftype) {
                  case G__PARANORMAL: // given is one level pointer.
                  case G__PARAREFERENCE: // given is a reference to a pointer.
                     type = tolower(type);
                     break;
                  case G__PARAP2P: // given is a pointer to a pointer.
                     reftype = G__PARANORMAL;
                     break;
                  default: // given is a multi-level pointer.
                     --reftype;
                     break;
               }
               break;
            default: // final is a multi-level pointer.
               if (G__get_reftype(final) == reftype) { // given matches final
                  type = tolower(type);
                  reftype = G__PARANORMAL;
               }
               else if ((G__get_reftype(final) + 1) == reftype) { // given is one more than final
                  reftype = G__PARANORMAL;
               }
               else if (G__get_reftype(final) < reftype) { // given is two or more than final
                  reftype -= G__get_reftype(final);
               }
               break;
         }
      }
      //
      //  Now apply any given G__CONSTVAR qualifier.
      //
      if (isconst & G__CONSTVAR) {
         result = G__apply_const_to_typedef(result);
      }
   }
   else if (tagnum != -1) { // Class type.
      result = G__Dict::GetDict().GetType(tagnum);
      if (!result) {
         return result;
      }
      //
      //  Now apply any given G__CONSTVAR qualifier.
      //
      if (isconst & G__CONSTVAR) {
         result = ::Reflex::Type(result, ::Reflex::CONST, ::Reflex::Type::APPEND);
      }
   }
   else { // Fundamental type.
      result = G__get_from_type(type, 0, 0);
      if (!result) {
         return result;
      }
      //
      //  Now apply any given G__CONSTVAR qualifier.
      //
      if (isconst & G__CONSTVAR) {
         result = ::Reflex::Type(result, ::Reflex::CONST, ::Reflex::Type::APPEND);
      }
   }
   //
   //  Apply first pointer level.
   //
   if (isupper(type)) {
      result = ::Reflex::PointerBuilder(result);
   }
   switch (reftype) { // Apply the rest of the pointer levels, and the reference.
      case G__PARANORMAL:
         break;
      case G__PARAREFERENCE: // Cannot happen, we removed this above by using G__PLVL.
         result = ::Reflex::Type(result, ::Reflex::REFERENCE, Reflex::Type::APPEND);
         break;
      case G__PARAP2P:
         result = ::Reflex::PointerBuilder(result);
         break;
      case G__PARAP2P2P:
         result = ::Reflex::PointerBuilder(result);
         result = ::Reflex::PointerBuilder(result);
         break;
      default:
         if ((reftype > 10) || (reftype < 0)) { // Error conditions.
            break;
         }
         for (int i = G__PARAP2P; i <= reftype; ++i) {
            result = ::Reflex::PointerBuilder(result);
         }
         break;
   }
   if ((isconst & G__PCONSTVAR) && isupper(type)) { // Apply const pointer qualifier.
      result = ::Reflex::Type(result, ::Reflex::CONST, Reflex::Type::APPEND);
   }
   if (ref) { // Apply reference.
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
::Reflex::Scope Cint::Internal::G__findInScope(const ::Reflex::Scope scope, const char* name)
{
   // -- Find a REFLEX Scope in a REFLEX Scope by name.
   ::Reflex::Scope cl;
#ifdef __GNUC__
#else
#pragma message (FIXME("This needs to be in Reflex itself"))
#endif
   if (name==0 || name[0]==0) {
      return cl;
   }
   ::Reflex::Scope_Iterator end = scope.SubScope_End();
   for (
      ::Reflex::Scope_Iterator itype = scope.SubScope_Begin();
      itype != end;
      ++itype
   ) {
      const char *iname = itype->Name_c_str() + itype->ToScopeBase()->GetBasePosition();
      if ( iname[0]==name[0] && iname[1]==name[1] && 0==strcmp(iname,name) ) {
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
   const char *type_name = type.Name_c_str();
   return type_name[0]=='m' && ( 0==strcmp(type_name,"macroInt$") || 0==strcmp(type_name,"macroDouble$") );
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
Reflex::Type Cint::Internal::G__replace_rawtype(const Reflex::Type target, const Reflex::Type raw)
{
   // copy with modifiers
   if (target == target.ToType()) { // Terminate recursion when target is the invalid type.
      return raw;
   }
   Reflex::Type out = G__replace_rawtype(target.ToType(), raw); // Recurse to the bottom of the type chain.
   if (target.IsTypedef()) {
      out = Reflex::TypedefTypeBuilder(target.Name_c_str(), out);
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
   // Now copy the qualifiers. // FIXME: What about reference on the first node, do we lose that?
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

//______________________________________________________________________________
Reflex::Type Cint::Internal::G__apply_const_to_typedef(const Reflex::Type target)
{
   // Copy type chain, but make last typedef in chain const. // FIXME: We change all typedefs in chain, but cint5 can have only one for now.
   if (!target.ToType()) { // Terminate recursion at last node in chain.
      return target;
   }
   Reflex::Type out = G__apply_const_to_typedef(target.ToType()); // Recurse to the bottom of the type chain.
   if (target.IsTypedef()) {
      out = ::Reflex::Type(Reflex::TypedefTypeBuilder(target.Name_c_str(), out), ::Reflex::CONST, ::Reflex::Type::APPEND);
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
   // Now copy the qualifiers. // FIXME: What about reference on the first node, do we lose that?
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
   char* cint_offset = member.InterpreterOffset();
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
   member = varscope.AddDataMember(name.c_str(), newType, offset, modifiers, cint_offset);
   *G__get_properties(member) = prop;
   G__get_offset(member) = cint_offset;
   return member;
}

//______________________________________________________________________________
int ::Cint::Internal::G__get_access(const ::Reflex::Member mem)
{
   if (mem.IsPublic()) {
      return G__PUBLIC;
   }
   if (mem.IsProtected()) {
      return G__PROTECTED;
   }
   if (mem.IsPrivate()) {
      return G__PRIVATE;
   }
   // FIXME: Need an error message here!
   return 0;
}

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
   entry.p = (void*) 0xdeadbeef;
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
   int isconst = (reftype_const / 10) % 10;
   int reftype = reftype_const - (reftype_const / 10 % 10 * 10);

   ::Reflex::Type paramType( G__cint5_tuple_to_type(type, numerical_tagnum, numerical_typenum, reftype, isconst) );

   fParams_type.push_back(paramType);
   fDefault_vals.push_back(para_default);
   fParams_name.push_back(std::make_pair(para_name, para_def));
}

//______________________________________________________________________________
::Reflex::Member Cint::Internal::G__BuilderInfo::Build(const std::string name)
{
   // Create the reflex database entries for function name.
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
   ::Reflex::Type modftype = ::Reflex::Type(ftype, modifiers);
   unsigned int modifiers_mask = ::Reflex::PUBLIC | ::Reflex::PROTECTED | ::Reflex::PRIVATE | ::Reflex::EXPLICIT | ::Reflex::STATIC | ::Reflex::VIRTUAL | ::Reflex::ABSTRACT | ::Reflex::CONSTRUCTOR | ::Reflex::DESTRUCTOR | ::Reflex::COPYCONSTRUCTOR;
   //fprintf(stderr, "G__BuilderInfo::Build: search for function member '%s' modftype '%s'\n", name.c_str(), modftype.Name(Reflex::SCOPED | Reflex::QUALIFIED).c_str());
   ::Reflex::Member m = G__p_ifunc.FunctionMemberByNameAndSignature(name, modftype, modifiers_mask);
   //if (m) {
   //   fprintf(stderr, "G__BuilderInfo::Build: search found member '%s'\n", m.Name(Reflex::SCOPED | Reflex::QUALIFIED).c_str());
   //}
   //else {
   //   fprintf(stderr, "G__BuilderInfo::Build: search failed.\n");
   //}
   //bool has_reflex_stub = false;
   bool from_reflex_callback = false;
   if (m) {
      //
      //  Check for the existence of a reflex dictionary
      //  stub function pointer.  If it is there, then this
      //  function was entered by a reflex dictionary.
      //
      //if (m.StubFunction()) {
      //   has_reflex_stub = true;
      //}
      //
      //  Check for the existence of cint properties on
      //  this function.  If they are not there then cint
      //  does not know about this function so it must
      //  come from a reflex callback through cintex.
      //
      static size_t pid = GetReflexPropertyID();
      G__RflxFuncProperties* prop = (G__RflxFuncProperties*) m.ToMemberBase()->Properties().PropertyValue(pid).Address();
      if (!prop) {
         from_reflex_callback = true;
      }
   }
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
      (
         !m  || // not previous declared, or
         from_reflex_callback // cint does not yet know about this function
      )
   ) {
      struct G__inheritance* baseclass = G__struct.baseclass[G__get_tagnum(G__tagdefining)];
      for (size_t basen = 0; basen < baseclass->vec.size(); ++basen) {
         G__incsetup_memfunc(baseclass->vec[basen].basetagnum);
         ::Reflex::Scope scope = G__Dict::GetDict().GetScope(baseclass->vec[basen].basetagnum);
         //fprintf(stderr, "G__BuilderInfo::Build: search base class '%s' for function member '%s' type '%s'\n", scope.Name(Reflex::SCOPED).c_str(), name.c_str(), modftype.Name(Reflex::SCOPED | Reflex::QUALIFIED).c_str());
         ::Reflex::Member base_m = scope.FunctionMemberByNameAndSignature(name, modftype, modifiers_mask);
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
   if (m && !from_reflex_callback) { // function was previously declared
      //
      //  Now handle the parts which go into the function property.
      //
      G__RflxFuncProperties* prop = G__get_funcproperties(m);
#ifdef G__FRIEND
      //
      //  Consume the friendtag.
      //
      if (fProp.entry.friendtag) { // New decl has a friendtag list.
         if (!prop->entry.friendtag) { // Old dcl did not have one, copy over from new decl.
            prop->entry.friendtag = fProp.entry.friendtag;
            fProp.entry.friendtag = 0; // Transfer ownership.
         }
         else { // Old decl has a friendtag list too, append new decl list to old decl list.
            G__friendtag* friendtag = prop->entry.friendtag;
            while (friendtag->next) {
               friendtag = friendtag->next;
            }
            friendtag->next = fProp.entry.friendtag;
            fProp.entry.friendtag = 0; // Transfer ownership.
         }
      }
#endif // G__FRIEND
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
         fProp.entry.p && // we have a stub function pointer for compiled func, or file pointer to body for interpreted func
         (
            !G__def_struct_member || // not a class member, or
            (G__struct.iscpplink[G__get_tagnum(G__def_tagnum)] != G__CPPLINK) // is not precompiled
         )
      ) {
         // Replace pre-existing function decl attributes.
         //
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
         G__friendtag* prev_friendtag = prop->entry.friendtag;
         prop->entry.friendtag = 0;
         std::vector<G__value*> prev_para_default = prop->entry.para_default;
         prop->entry.para_default.clear();
         //
         //  Overwrite the old entry point info with the new info.
         prop->entry = fProp.entry;
         prop->entry.friendtag = prev_friendtag;
         prop->entry.para_default = prev_para_default;
         prop->entry.for_tp2f = m.Name_c_str();
         prop->entry.tp2f = (void*) prop->entry.for_tp2f.c_str();
         //for (unsigned int i = 0; i < fParams_name.size(); ++i) {
         //   if (fParams_name[i].second.size()) {
         //      G__genericerror("Error: Redefinition of default argument");
         //      G__fprinterr(G__serr, "Cint::Internal::G__BuilderInfo::Build: function: '%s' param: '%s' default value: '%d'  %s:%d\n", m.Name(::Reflex::SCOPED).c_str(), fParams_name[i].first.c_str(), fParams_name[i].second.c_str(), __FILE__, __LINE__);
         //   }
         //}
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
         fProp.entry.p || // we have a stub function pointer for compiled func, or file pointer to body for interpreted func, or
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
      // Reset modifier to take in consideration the change above (in particular) fIsvirtual.
      modifiers = 0;
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
         case 'c': // class
         case 's': // struct
         case 'u': // union
            {
               //
               //  Do special processing for a constructor/destructor.
               //
               if (name[0] == '~') { // destructor
                  modifiers |= ::Reflex::DESTRUCTOR;
               }
               else if (name == G__p_ifunc.Name()) { // constructor
                  modifiers |= ::Reflex::CONSTRUCTOR;
                  if (ftype.FunctionParameterSize() == 1) { // possible copy constructor
                     // Get the first parameter type without modifiers.
                     ::Reflex::Type first_param_type = ftype.FunctionParameterAt(0).FinalType().ToTypeBase()->ThisType();
                     if (first_param_type == (Reflex::Type) G__p_ifunc) { // we have a copy constructor
                        modifiers |= ::Reflex::COPYCONSTRUCTOR;
                     }
                  }
               }
               if (!from_reflex_callback) {
                  m = G__p_ifunc.AddFunctionMember(name.c_str(), ftype, (Reflex::StubFunction) 0, 0 /*stubCtx*/, GetParamNames().c_str(), modifiers);
                  //modftype = ::Reflex::Type(ftype, modifiers);
                  //m = G__p_ifunc.FunctionMemberByNameAndSignature(name, modftype, modifiers_mask);
                  //fprintf(stderr, "G__BuilderInfo::Build: added member function '%s'.\n", m.Name(Reflex::SCOPED | Reflex::QUALIFIED).c_str());
               }
            }
            break;
         case 'n': // Namespace.
            {
               if (!from_reflex_callback) {
                  //std::string fullname = G__p_ifunc.Name(::Reflex::SCOPED);
                  //if (fullname.size()) {
                  //   fullname += "::";
                  //}
                  //fullname += name;
                  //m = Reflex::FunctionBuilder(ftype, fullname.c_str(), 0 /*stubFP*/, 0 /*stubCtx*/, GetParamNames().c_str(), modifiers).EnableCallback(false).ToMember();
                  m = G__p_ifunc.AddFunctionMember(name.c_str(), ftype, (Reflex::StubFunction) 0, 0 /*stubCtx*/, GetParamNames().c_str(), modifiers);
                  //modftype = ::Reflex::Type(ftype, modifiers);
                  //m = G__p_ifunc.FunctionMemberByNameAndSignature(name, modftype, modifiers_mask);
                  //fprintf(stderr, "G__BuilderInfo::Build: added member function '%s'.\n", m.Name(Reflex::SCOPED | Reflex::QUALIFIED).c_str());
               }
               break;
            }
         default: // Assume this means we want it in the global namespace.
            {
               if (!from_reflex_callback) {
                  //m = Reflex::FunctionBuilder(ftype, name.c_str(), 0 /*stubFP*/, 0 /*stubCtx*/, GetParamNames().c_str(), modifiers).EnableCallback(false).ToMember();
                  m = ::Reflex::Scope::GlobalScope().AddFunctionMember(name.c_str(), ftype, (Reflex::StubFunction) 0, 0 /*stubCtx*/, GetParamNames().c_str(), modifiers);
                  //modftype = ::Reflex::Type(ftype, modifiers);
                  //m = G__p_ifunc.FunctionMemberByNameAndSignature(name, modftype, modifiers_mask);
                  //fprintf(stderr, "G__BuilderInfo::Build: added member function '%s'.\n", m.Name(Reflex::SCOPED | Reflex::QUALIFIED).c_str());
               }
               //int tag = G__get_tagtype( G__p_ifunc ) ;
               //fprintf(stderr,"value=%d\n",tag);
               //G__genericerror("Attempt to add a function to something that does not appear to be a scope");
            }
      }
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
         a->entry.for_tp2f = m.Name_c_str();
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

