/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file DataMbr.h
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Copyright(c) 1995~1998  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#ifndef G__DATAMEMBER_H
#define G__DATAMEMBER_H

#include "Api.h"

namespace Cint {

/*********************************************************************
 * class G__DataMemberInfo
 *
 *
 *********************************************************************/
class
#ifndef __CINT__
G__EXPORT
#endif
G__DataMemberInfo {
   // -- Types.
public:
   enum error_code { VALID, NOT_INT, NOT_DEF, IS_PRIVATE, UNKNOWN };
   // -- Data Members.
private:
   Reflex::Scope m_scope;
   Reflex::Member m_datambr;
   int m_memberiter;
   // -- Backward compatibility
   G__TypeInfo  *m_typeinfo;
   G__ClassInfo *m_memberof;
   std::string m_name;

   // -- Member Functions.
public:
   ~G__DataMemberInfo();

   G__DataMemberInfo()
   : m_memberiter(-1), m_typeinfo(0), m_memberof(0)
   {
      Init();
   }

   G__DataMemberInfo(const G__DataMemberInfo& dmi)
   : m_scope(dmi.m_scope)
   , m_datambr(dmi.m_datambr)
   , m_memberiter(dmi.m_memberiter)
   , m_typeinfo(0)
   , m_memberof(0)
   {
   }

   G__DataMemberInfo(class G__ClassInfo& a) : m_typeinfo(0), m_memberof(0)
   {
      Init(a);
   }

   G__DataMemberInfo(const ::Reflex::Member& m)
   : m_datambr(m), m_typeinfo(0), m_memberof(0)
   {
   }

   G__DataMemberInfo& operator=(const G__DataMemberInfo& dmi)
   {
      m_datambr = dmi.m_datambr;
      return *this;
   }

   void Init();
   void Init(class G__ClassInfo& a);
   void Init(const ::Reflex::Scope& a);
   void Init(long handlinin, long indexin, G__ClassInfo* belongingclassin);

   size_t Handle()
   {
      return (size_t) m_datambr.Id();
   }

   int Index()
   {
      return 0;
   }

   const char* Name();
   const char* Title();

   Reflex::Type ReflexType();

   G__TypeInfo* Type();

   long Property();
   long Offset();
   int Bitfield();
   int ArrayDim();
   int MaxIndex(int dim);

   Reflex::Scope DeclaringScope();
   G__ClassInfo* MemberOf();

   void SetGlobalcomp(int globalcomp);
   int IsValid();
   int SetFilePos(const char* fname);
   int Next();
   int Prev();
   const char* ValidArrayIndex(int* errnum = 0, char** errstr = 0);
   const char* FileName();
   int LineNumber();
};

} // namespace Cint

using namespace Cint;
#endif
