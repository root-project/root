/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file Class.h
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Copyright(c) 1995~1998  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__CLASSINFO_H
#define G__CLASSINFO_H

#include "Api.h"

namespace Cint {

class G__MethodInfo;
class G__DataMemberInfo;
class G__FriendInfo;

class
#ifndef __CINT__
G__EXPORT
#endif // __CINT__
G__ClassInfo
{
public:
   virtual ~G__ClassInfo();
   G__ClassInfo();
   void Init();
   G__ClassInfo(const char* classname);
   void Init(const char* classname);
   G__ClassInfo(const G__value&);
   G__ClassInfo(int tagnumin);
   void Init(int tagnumin);
   int operator==(const G__ClassInfo&);
   int operator!=(const G__ClassInfo&);
   const char* Name();
   const char* Fullname();
   const char* Title();
   int Size();
   long Property();
   int NDataMembers();
   int NMethods();
   long IsBase(const char* classname);
   long IsBase(G__ClassInfo&);
   long Tagnum() const;
   //::Reflex::Scope ReflexScope(); // FIXME: Interface change!
   G__ClassInfo EnclosingClass();
   G__ClassInfo EnclosingSpace();
   G__friendtag* GetFriendInfo();
   void SetGlobalcomp(int globalcomp);
   void SetProtectedAccess(int protectedaccess);
   int IsValid();
   int IsLoaded();
   int SetFilePos(const char* fname);
   int Next();
   int Linkage();
   const char* FileName();
   int LineNumber();
   int IsTmplt();
   const char* TmpltName();
   const char* TmpltArg();
protected:
   long fTagnum;  // class,struct,union,enum key for cint dictionary
   long fClassProperty;  // cache value (expensive to get)
///////////////////////////////////////////////////////////////
// Following things are for ROOT
///////////////////////////////////////////////////////////////
public:
   void SetDefFile(char* deffilein);
   void SetDefLine(int deflinein);
   void SetImpFile(char* impfilein);
   void SetImpLine(int implinein);
   void SetVersion(int versionin);
   const char* DefFile();
   int DefLine();
   const char* ImpFile();
   int ImpLine();
   int Version();
   void* New();
   void* New(int n);
   void* New(void* arena);
   void* New(int n, void* arena);
   void Delete(void* p) const;
   void Destruct(void* p) const;
   void DeleteArray(void* ary, int dtorOnly = 0);
   int InstanceCount();
   void ResetInstanceCount();
   void IncInstanceCount();
   int HeapInstanceCount();
   void IncHeapInstanceCount();
   void ResetHeapInstanceCount();
   int RootFlag();
   //void SetDefaultConstructor(void* p2f);
   enum MatchMode {
      ExactMatch = 0,
      ConversionMatch = 1,
      ConversionMatchBytecode = 2
   };
   enum InheritanceMode {
      InThisScope = 0,
      WithInheritance = 1
   };
   G__InterfaceMethod GetInterfaceMethod(
      const char* fname, const char* arg, long* poffset,
      MatchMode mode = ConversionMatch,
      InheritanceMode imode = WithInheritance
   );
   G__MethodInfo GetMethod(
      const char* fname, const char* arg, long* poffset,
      MatchMode mode = ConversionMatch,
      InheritanceMode imode = WithInheritance
   );
   G__MethodInfo GetMethod(
      const char* fname, G__param* libp, long* poffset,
      MatchMode mode = ConversionMatch,
      InheritanceMode imode = WithInheritance
   );
   G__MethodInfo GetDefaultConstructor();
   G__MethodInfo GetCopyConstructor();
   G__MethodInfo GetDestructor();
   G__MethodInfo GetAssignOperator();
   G__MethodInfo AddMethod(const char* typenam, const char* fname, const char* arg, int isstatic = 0, int isvirtual = 0, void* methodAddress = 0);
   G__DataMemberInfo GetDataMember(const char* name, long* poffset);
   int HasMethod(const char* fname);
   int HasDataMember(const char* name);
   int HasDefaultConstructor();
private:
   void CheckValidRootInfo();
public:
   long ClassProperty();
   unsigned char FuncFlag();
};

class
#ifndef __CINT__
G__EXPORT
#endif
G__FriendInfo
{
public:
   G__FriendInfo(G__friendtag* pin = 0) : pfriendtag(NULL), cls()
   {
      Init(pin);
   }

   G__FriendInfo(const G__FriendInfo& x) : pfriendtag(x.pfriendtag), cls(x.cls)
   {
      Init(x.pfriendtag);
   }

   G__FriendInfo& operator=(const G__FriendInfo& x)
   {
      Init(x.pfriendtag);
      return *this;
   }

   void Init(G__friendtag* pin)
   {
      pfriendtag = pin;
      if (pfriendtag) {
         cls.Init(pfriendtag->tagnum);
      }
      else {
         cls.Init(-1);
      }
   }

   G__ClassInfo* FriendOf()
   {
      return &cls;
   }

   int Next()
   {
      if (pfriendtag) {
         pfriendtag = pfriendtag->next;
         Init(pfriendtag);
         return IsValid();
      }
      return 0;
   }

   int IsValid() {
      if (pfriendtag) {
         return 1;
      }
      return 0;
   }

private:
   G__friendtag* pfriendtag;
   G__ClassInfo cls;
};

} // namespace Cint

using namespace Cint;
#endif // G__CLASSINFO_H
