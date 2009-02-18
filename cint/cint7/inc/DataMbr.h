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

class
#ifndef __CINT__
G__EXPORT
#endif // __CINT__
G__DataMemberInfo // Data member iterator for a class or namespace.
{
public: // -- Types.
   enum error_code
   {
      VALID, /* 0 */
      NOT_INT, /* 1 */
      NOT_DEF, /* 2 */
      IS_PRIVATE, /* 3 */
      UNKNOWN /* 4 */
   };

public: // -- Member Functions.
   ~G__DataMemberInfo(); // destructor
   G__DataMemberInfo(); // default constructor
   G__DataMemberInfo(const G__DataMemberInfo&); // copy constructor
   G__DataMemberInfo(G__ClassInfo&); // init with a specific scope
   //G__DataMemberInfo(const ::Reflex::Member); // init with a specific member  TODO: Can we use iterator after?
   G__DataMemberInfo& operator=(const G__DataMemberInfo&); // copy assignment operator
   void Init(); // default init
   void Init(G__ClassInfo&); // init with a specific scope
   //void Init(const ::Reflex::Scope); // init with a specific scope
   void Init(long handlinin, long indexin, G__ClassInfo* belongingclassin);
   size_t Handle(); // id number of current data member
   int Index(); // always returns zero
   const char* Name(); // name of current data member
   const char* Title(); // comment of current data member
   //::Reflex::Type ReflexType(); // reflex type of current data member
   G__TypeInfo* Type(); // datatype of current data member as a G__TypeInfo
   long Property(); // summary of properties of current data member
   long Offset(); // address of storage for current data member
   int Bitfield(); // bitfield width of current data member
   int ArrayDim(); // number of array bounds of current data member
   int MaxIndex(int dim); // array bound for a given dimension of current data member
   //::Reflex::Scope DeclaringScope(); // scope of current data member
   G__ClassInfo* MemberOf(); // scope of current data member as a G__ClassInfo
   void SetGlobalcomp(int globalcomp); // change globalcomp of current data member
   int IsValid(); // Internal iterator is pointing at a valid data member.
   int SetFilePos(const char* fname); // Set internal iterator based on a saved file position
   int Next(); // Move to next data member in scope.
   int Prev(); // Move to previous data member in scope.
   const char* ValidArrayIndex(int* errnum = 0, char** errstr = 0); // check the current data member comment for an array index expression
   const char* FileName(); // filename of data member pointed to by internal iterator
   int LineNumber(); // line number of data member pointed to by internal iterator

private: // -- Data Members.
   // FIXME: We are *not* binary compatible with cint5.
   ::Reflex::Scope m_scope; // declaring scope we represent
   ::Reflex::Member m_datambr; // current state of internal iterator, the current data member
   int m_memberiter; // current state of internal iterator, member index
   // -- Cached data.
   G__TypeInfo* m_typeinfo; // we own, cached object, reflex type of current data member as a G__TypeInfo
   G__ClassInfo* m_memberof; // we own, cached object, the declaring scope as a G__ClassInfo
};

} // namespace Cint

using namespace Cint;
#endif // G__DATAMEMBER_H
