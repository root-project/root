/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file Type.h
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Copyright(c) 1995~2003  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__TYPEINFOX_H
#define G__TYPEINFOX_H

#include "Api.h"

namespace Cint {

// FIXME: Warning, G__TypeReader in src/bc_type.h inherits from G__TypeInfo, we may need a virtual destructor.

class
#ifndef __CINT__
G__EXPORT
#endif // __CINT__
G__TypeInfo : public G__ClassInfo
{
   friend class G__DataMemberInfo;
   friend class G__MethodInfo;
   friend class G__MethodArgInfo;
public:
   virtual ~G__TypeInfo();
   //G__TypeInfo(const ::Reflex::Type in); // FIXME: Interface change!
   G__TypeInfo(const char* typenamein);
   G__TypeInfo();
   void Init(const char* typenamein);
   //void Init(const ::Reflex::Type in); // FIXME: Interface change!
#ifndef __MAKECINT__
   G__TypeInfo(G__value buf);
   void Init(G__value&);
   void Init(G__var_array*, int idx);
  G__TypeInfo(const G__TypeInfo&);
  G__TypeInfo& operator=(const G__TypeInfo&);
#endif // __MAKECINT__
   int operator==(const G__TypeInfo&);
   int operator!=(const G__TypeInfo&);
   const char* Name();
   const char* TrueName();
   int Size() const;
   long Property();
   int IsValid();
   void* New();
   int Typenum() const;
   int Type() const;
   int Reftype() const;
   int Isconst() const;
   //Reflex::Type ReflexType() const; // FIXME: Interface change!
   G__value Value() const;
private:
   int Next(); // prohibit use of next
protected:
   long fType;
   long fTypenum;
   long fReftype;
   long fIsconst;
};

} // namespace Cint

#endif // G__TYPEINFOX_H
