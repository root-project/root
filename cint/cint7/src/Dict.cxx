/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file Class.cxx
 ************************************************************************
 * Description:
 *  Dictionary information interface
 ************************************************************************
 * Author                  Philippe Canal 
 * Copyright(c) 1995~2004  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "Api.h"
#include "common.h"
#include "Dict.h"

/*********************************************************************
* class G__ClassInfo
*********************************************************************/
///////////////////////////////////////////////////////////////////////////
Cint::Internal::G__Dict &Cint::Internal::G__Dict::GetDict() 
{
   // Return the current instance
   static G__Dict dict;
   return dict;
}

///////////////////////////////////////////////////////////////////////////
::Reflex::Type Cint::Internal::G__Dict::GetType(int tagnum)
{
   // Return the Reflex type (class or namespace) corresponding 
   // to the CINT tagnum.
   static ::Reflex::Type null_type;
   if (-1 == tagnum) return null_type;
   ::Reflex::Type t = mScopes[tagnum];
   if (t) return t;
   // In case of non scope entity:
   std::string fulltagname(G__fulltagname(tagnum,0));
   t = ::Reflex::Type::ByName(fulltagname); // We stored the dollar (at least for now)
   if (!t) {
#ifdef __GNUC__
#else
#pragma message (FIXME("This is very bad we need to fix the std lookup more globabaly"))
#endif
      fulltagname.insert(0, "std::");
      t = ::Reflex::Type::ByName(fulltagname); 
   }
   return t;
}

///////////////////////////////////////////////////////////////////////////
::Reflex::Type Cint::Internal::G__Dict::GetTypeFromId(size_t handle)
{
   // FIXME this should be made inline for performance.
   return Reflex::Type( reinterpret_cast< const Reflex::TypeName* >(handle));
}

///////////////////////////////////////////////////////////////////////////
::Reflex::Scope Cint::Internal::G__Dict::GetScope(int tagnum)
{
   // Return the Reflex type (class or namespace) corresponding 
   // to the CINT tagnum.
   if (-1 == tagnum) {
      return ::Reflex::Scope::GlobalScope();
   } else {
      return mScopes[tagnum];
   }
   std::string fulltagname(G__fulltagname(tagnum,0));
   ::Reflex::Scope s = ::Reflex::Scope::ByName(fulltagname);
   if (!s) {
      // Note: "This is very bad we need to fix the std lookup more globabaly"
      fulltagname.insert(0, "std::");
      s = ::Reflex::Type::ByName(fulltagname); 
   }
   return s;
}

////////////////////////////////////////////////////////////////////////
::Reflex::Type Cint::Internal::G__Dict::GetTypedef(int typenum)
{
   // Return the Reflex typedef  corresponding 
   // to the CINT typenum.
   if ((typenum < 0) ||
       (typenum >= ((int) mTypenums.size())) ||
       !mTypenums[typenum].IsTypedef())
      return ::Reflex::Dummy::Type();
   return mTypenums[typenum];
}

////////////////////////////////////////////////////////////////////////
::Reflex::Scope Cint::Internal::G__Dict::GetScope(const struct G__ifunc_table* funcnum)
{
   // Return the Reflex function corresponding 
   // to the funcnum.
   // FIXME this should be made inline for performance.
   return Reflex::Scope(reinterpret_cast<const Reflex::ScopeName*>(funcnum));
}

////////////////////////////////////////////////////////////////////////
::Reflex::Scope Cint::Internal::G__Dict::GetScope(const struct G__var_array* varnum)
{
   // Return the Reflex function corresponding 
   // to the funcnum.
   // FIXME this should be made inline for performance.
   return Reflex::Scope(reinterpret_cast<const Reflex::ScopeName*>(varnum));
}

///////////////////////////////////////////////////////////////////////////
::Reflex::Member Cint::Internal::G__Dict::GetFunction( const struct G__ifunc_table* funcnum,
                                                             int ifn ) {
  if (ifn == -2)
    return GetFunction((size_t) funcnum);
  if (ifn < 0)
     return Reflex::Dummy::Member();
  return GetScope(funcnum).FunctionMemberAt(ifn);
}

///////////////////////////////////////////////////////////////////////////
::Reflex::Member Cint::Internal::G__Dict::GetFunction( size_t handle ) 
{
   // FIXME this should be made inline for performance.
   return Reflex::Member(reinterpret_cast< const Reflex::MemberBase* >(handle));
}

///////////////////////////////////////////////////////////////////////////
::Reflex::Member Cint::Internal::G__Dict::GetDataMember( const struct G__var_array* varnum,
                                                               int ig15 ) {
  return GetScope(varnum).DataMemberAt(ig15);
}

///////////////////////////////////////////////////////////////////////////
::Reflex::Member Cint::Internal::G__Dict::GetDataMember( size_t handle ) 
{
   // FIXME this should be made inline for performance.
   return Reflex::Member(reinterpret_cast< const Reflex::MemberBase* >(handle));
}

///////////////////////////////////////////////////////////////////////////
bool Cint::Internal::G__Dict::RegisterScope(int tagnum, const ::Reflex::Scope &in) 
{
   // Register a Reflex Scope corresponding to a tagnum.
   // Return false is tagnum is out of range.
   
   if (tagnum<0 || tagnum>G__MAXSTRUCT) {
      return false;
   }
   mScopes[tagnum] = in;
   return true;
}

///////////////////////////////////////////////////////////////////////////
int Cint::Internal::G__Dict::Register(const ::Reflex::Type &in) 
{
   // Register a Reflex type and returns the assigned 
   // CINT number (used for emulating typenum and tagnum).
   mTypenums.push_back(in);
   //fprintf(stderr, "%06d, Registered type '%s'\n", mTypenums.size() - 1, in.Name().c_str());
   return mTypenums.size()-1;
}
