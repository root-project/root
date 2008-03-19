/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file Type.h
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Copyright(c) 1995~2007  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__TYPEINFOX_H
#define G__TYPEINFOX_H 

#include "Api.h"

namespace Cint {

/*********************************************************************
* class G__TypeInfo
* 
*********************************************************************/
class 
#ifndef __CINT__
G__EXPORT
#endif
// FIXME: Warning, G__TypeReader in src/bc_type.h inherits from G__TypeInfo, we may need a virtual destructor.
G__TypeInfo : public G__ClassInfo {
friend class G__DataMemberInfo;
friend class G__MethodInfo;
friend class G__MethodArgInfo;
public:
   ~G__TypeInfo();
   G__TypeInfo(const ::Reflex::Type &in);
   G__TypeInfo(const char* typenamein);
   G__TypeInfo();
   void Init(const char* typenamein);
   void Init(const ::Reflex::Type &in);
#ifndef __MAKECINT__
   G__TypeInfo(G__value buf);
   void Init(G__value& buf);
   void Init(struct G__var_array*, int);
#endif
   int operator==(const G__TypeInfo& a);
   int operator!=(const G__TypeInfo& a);
   const char* Name() ;
   const char* TrueName() ;
   int Size() const; 
   long Property();
   int IsValid();
   void* New();

   int Typenum() const;
   int Type() const;
   int Reftype() const;
   int Isconst() const;
   Reflex::Type ReflexType() const;

   G__value Value() const;
protected:
   ::Reflex::Type typenum;
   int typeiter;
private:
   int Next(); // prohibit use of next
};

} // namespace Cint

#endif
