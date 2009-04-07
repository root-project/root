/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file Class.cxx
 ************************************************************************
 * Description:
 *  Extended Run Time Type Identification API
 ************************************************************************
 * Author                  Masaharu Goto
 * Copyright(c) 1995~2007  Masaharu Goto
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "Api.h"
#include "common.h"
#include "Dict.h"

using namespace Cint::Internal;
using namespace std;

//______________________________________________________________________________
Cint::G__ClassInfo::~G__ClassInfo()
{
}

//______________________________________________________________________________
Cint::G__ClassInfo::G__ClassInfo() : fTagnum(0), fClassProperty(0)
{
   Init();
}

//______________________________________________________________________________
void Cint::G__ClassInfo::Init()
{
   fTagnum = -1;
   fClassProperty = 0;
}

//______________________________________________________________________________
Cint::G__ClassInfo::G__ClassInfo(const char* classname) : fTagnum(0) , fClassProperty(0)
{
   Init(classname);
}

//______________________________________________________________________________
void Cint::G__ClassInfo::Init(const char* classname)
{
   fTagnum = G__defined_tagname(classname,1);
   fClassProperty = 0;
}

//______________________________________________________________________________
Cint::G__ClassInfo::G__ClassInfo(const G__value& value_for_type) : fTagnum(0) , fClassProperty(0)
{
   Init(G__get_tagnum(G__value_typenum(value_for_type).RawType()));
}

//______________________________________________________________________________
Cint::G__ClassInfo::G__ClassInfo(int tagnumin) : fTagnum(0) , fClassProperty(0)
{
   Init(tagnumin);
}

//______________________________________________________________________________
void Cint::G__ClassInfo::Init(int tagnumin)
{
   fTagnum = tagnumin;
   fClassProperty = 0;
}

//______________________________________________________________________________
int Cint::G__ClassInfo::operator==(const G__ClassInfo& a)
{
   return fTagnum == a.fTagnum;
}

//______________________________________________________________________________
int Cint::G__ClassInfo::operator!=(const G__ClassInfo& a)
{
   return fTagnum != a.fTagnum;
}

//______________________________________________________________________________
const char* Cint::G__ClassInfo::Name()
{
   if (IsValid()) {
      return G__struct.name[fTagnum];
   }
   return 0;
}

//______________________________________________________________________________
const char* Cint::G__ClassInfo::Fullname()
{
   static char G__buf[G__ONELINE];
   if (IsValid()) {
      strcpy(G__buf, G__fulltagname((int) fTagnum, 1));
#if defined(_MSC_VER) && (_MSC_VER < 1300) /*vc6*/
      char* ptr = strstr(G__buf, "long long");
      if (ptr) {
         memcpy(ptr, " __int64 ", strlen(" __int64 "));
      }
#endif // defined(_MSC_VER) && (_MSC_VER < 1300) /*vc6*/
      return G__buf; // FIXME: We are returning a pointer to a statically allocated buffer, this is not thread-safe.
   }
   return 0;
}

//______________________________________________________________________________
const char* Cint::G__ClassInfo::Title()
{
   static char buf[G__INFO_TITLELEN];
   buf[0] = '\0';
   if (IsValid()) {
      ::Reflex::Scope var = G__Dict::GetDict().GetScope(fTagnum);
      G__RflxProperties *prop = G__get_properties(var);
      if (prop) {
         G__getcomment(buf, &prop->comment, (int) fTagnum);
      }
      return buf; // FIXME: We are returning a pointer to a statically allocated buffer, this is not thread-safe.
   }
   return 0;
}

//______________________________________________________________________________
int Cint::G__ClassInfo::Size()
{
   if (IsValid()) {
      return G__struct.size[fTagnum];
   }
   return -1;
}

//______________________________________________________________________________
long Cint::G__ClassInfo::Property()
{
   if (fClassProperty) {
      return fClassProperty;
   }
   if (!IsValid()) {
      return 0L;
   }
   long property = 0L;
   switch (G__struct.type[fTagnum]) {
      case 'c':
         property |= G__BIT_ISCLASS;
         break;
      case 'e':
         property |= G__BIT_ISENUM;
         break;
      case 'n':
         property |= G__BIT_ISNAMESPACE;
         break;
      case 's':
         property |= G__BIT_ISSTRUCT;
         break;
      case 'u':
         property |= G__BIT_ISUNION;
         break;
   }
   if (G__struct.istypedefed[fTagnum]) {
      property |= G__BIT_ISTYPEDEF;
   }
   if (G__struct.isabstract[fTagnum]) {
      property |= G__BIT_ISABSTRACT;
   }
   switch (G__struct.iscpplink[fTagnum]) {
      case G__CPPLINK:
         property |= G__BIT_ISCPPCOMPILED;
         break;
      case G__CLINK:
         property |= G__BIT_ISCCOMPILED;
         break;
      case G__NOLINK:
         break;
      default:
         break;
   }
   fClassProperty = property;
   return property;
}

//______________________________________________________________________________
int Cint::G__ClassInfo::NDataMembers()
{
   if (IsValid()) {
      G__incsetup_memvar((int) fTagnum);
      return G__Dict::GetDict().GetScope(fTagnum).DataMemberSize();
   }
   return -1;
}

//______________________________________________________________________________
int Cint::G__ClassInfo::NMethods()
{
   if (IsValid()) {
      G__incsetup_memfunc((int) fTagnum);
      return G__Dict::GetDict().GetScope(fTagnum).FunctionMemberSize();
   }
   return -1;
}

//______________________________________________________________________________
long Cint::G__ClassInfo::IsBase(const char* classname)
{
   Cint::G__ClassInfo base(classname);
   return IsBase(base);
}

//______________________________________________________________________________
long Cint::G__ClassInfo::IsBase(G__ClassInfo& a)
{
   if (!IsValid()) {
      return 0L;
   }
   long isbase = 0L;
   G__inheritance* baseclass = G__struct.baseclass[fTagnum];
   for (size_t i = 0; i < baseclass->vec.size(); ++i) {
      if (a.Tagnum() != baseclass->vec[i].basetagnum) {
         continue;
      }
      switch (baseclass->vec[i].baseaccess) {
         case G__PUBLIC:
            isbase = G__BIT_ISPUBLIC;
            break;
         case G__PROTECTED:
            isbase = G__BIT_ISPROTECTED;
            break;
         case G__PRIVATE:
            isbase = G__BIT_ISPRIVATE;
            break;
         default:
            break;
      }
      if (baseclass->vec[i].property & G__ISDIRECTINHERIT) {
         isbase |= G__BIT_ISDIRECTINHERIT;
      }
      if (baseclass->vec[i].property & G__ISVIRTUALBASE) {
         isbase |= G__BIT_ISVIRTUALBASE;
      }
      break;
   }
   return isbase;
}

//______________________________________________________________________________
long Cint::G__ClassInfo::Tagnum() const
{
   return fTagnum;
}

#if 0
//______________________________________________________________________________
::Reflex::Scope Cint::G__ClassInfo::ReflexScope()
{
   return G__Dict::GetDict().GetScope(fTagnum);
}
#endif // 0

//______________________________________________________________________________
G__ClassInfo Cint::G__ClassInfo::EnclosingClass()
{
   if (IsValid()) {
      Cint::G__ClassInfo enclosingclass(G__struct.parent_tagnum[fTagnum]);
      return enclosingclass;
   }
   Cint::G__ClassInfo enclosingclass;
   return enclosingclass;
}

//______________________________________________________________________________
G__ClassInfo Cint::G__ClassInfo::EnclosingSpace()
{
   if (IsValid()) {
      int enclosing_tag = G__struct.parent_tagnum[fTagnum];
      while ((enclosing_tag > -1) && (G__struct.type[enclosing_tag] != 'n')) {
         enclosing_tag = G__struct.parent_tagnum[enclosing_tag];
      }
      Cint::G__ClassInfo enclosingclass(enclosing_tag);
      return enclosingclass;
   }
   Cint::G__ClassInfo enclosingclass;
   return enclosingclass;
}

//______________________________________________________________________________
G__friendtag* Cint::G__ClassInfo::GetFriendInfo()
{
   if (IsValid()) {
      return G__struct.friendtag[fTagnum];
   }
   return 0;
}

//______________________________________________________________________________
void Cint::G__ClassInfo::SetGlobalcomp(int globalcomp)
{
   if (IsValid()) {
      G__struct.globalcomp[fTagnum] = globalcomp;
   }
}

//______________________________________________________________________________
void Cint::G__ClassInfo::SetProtectedAccess(int protectedaccess)
{
   if (IsValid()) {
      G__struct.protectedaccess[fTagnum] = protectedaccess;
   }
}

//______________________________________________________________________________
int Cint::G__ClassInfo::IsValid()
{
   if ((fTagnum > -1) && (fTagnum < G__struct.alltag)) {
      return 1;
   }
   return 0;
}

//______________________________________________________________________________
int Cint::G__ClassInfo::IsLoaded()
{
   if (
      IsValid() &&
      (
         (G__struct.iscpplink[fTagnum] != G__NOLINK) ||
         (G__struct.filenum[fTagnum] != -1)
      )
   ) {
      return 1;
   }
   return 0;
}

//______________________________________________________________________________
int Cint::G__ClassInfo::SetFilePos(const char* fname)
{
   G__dictposition* dict = G__get_dictpos(const_cast<char*>(fname));
   if (!dict) {
      return 0;
   }
   fTagnum = dict->tagnum - 1;
   fClassProperty = 0;
   return 1;
}

//______________________________________________________________________________
int Cint::G__ClassInfo::Next()
{
   ++fTagnum;
   fClassProperty = 0;
   return IsValid();
}

//______________________________________________________________________________
int Cint::G__ClassInfo::Linkage()
{
   return G__struct.globalcomp[fTagnum];
}

//______________________________________________________________________________
const char* Cint::G__ClassInfo::FileName()
{
   if (!IsValid()) {
      return 0;
   }
   if (G__struct.filenum[fTagnum] != -1) {
      return G__srcfile[G__struct.filenum[fTagnum]].filename;
   }
   switch (G__struct.iscpplink[fTagnum]) {
      case G__CLINK:
         return "(C compiled)";
      case G__CPPLINK:
         return "(C++ compiled)";
   }
   return 0;
}

//______________________________________________________________________________
int Cint::G__ClassInfo::LineNumber()
{
   if (!IsValid()) {
      return -1;
   }
   switch (G__struct.iscpplink[fTagnum]) {
      case G__CLINK:
      case G__CPPLINK:
         return 0;
      case G__NOLINK:
         if (G__struct.filenum[fTagnum] != -1) {
            return G__struct.line_number[fTagnum];
         }
         return -1;
   }
   return -1;
}

//______________________________________________________________________________
int Cint::G__ClassInfo::IsTmplt()
{
   return IsValid() && strchr((char*) Name(), '<');
}

//______________________________________________________________________________
const char* Cint::G__ClassInfo::TmpltName()
{
   static char buf[G__ONELINE];
   if (!IsValid()) {
      return 0;
   }
   strcpy(buf, Name());
   char* p = strchr(buf, '<');
   if (p) {
      *p = 0;
   }
   return buf;
}

//______________________________________________________________________________
const char* Cint::G__ClassInfo::TmpltArg()
{
   static char buf[G__ONELINE];
   if (!IsValid()) {
      return 0;
   }
   char* p = strchr((char*) Name(), '<');
   if (!p) {
      return 0;
   }
   strcpy(buf, p + 1);
   p = strrchr(buf, '>');
   if (p) {
      *p = 0;
      while (isspace(*(--p))) {
         *p = 0;
      }
   }
   return buf;
}

//______________________________________________________________________________
void Cint::G__ClassInfo::SetDefFile(char* deffilein)
{
   if (IsValid()) {
      CheckValidRootInfo();
      G__struct.rootspecial[fTagnum]->deffile = deffilein;
   }
}

//______________________________________________________________________________
void Cint::G__ClassInfo::SetDefLine(int deflinein)
{
   if (IsValid()) {
      CheckValidRootInfo();
      G__struct.rootspecial[fTagnum]->defline = deflinein;
   }
}

//______________________________________________________________________________
void Cint::G__ClassInfo::SetImpFile(char* impfilein)
{
   if (IsValid()) {
      CheckValidRootInfo();
      G__struct.rootspecial[fTagnum]->impfile = impfilein;
   }
}

//______________________________________________________________________________
void Cint::G__ClassInfo::SetImpLine(int implinein)
{
   if (IsValid()) {
      CheckValidRootInfo();
      G__struct.rootspecial[fTagnum]->impline = implinein;
   }
}

//______________________________________________________________________________
void Cint::G__ClassInfo::SetVersion(int versionin)
{
   if (IsValid()) {
      CheckValidRootInfo();
      G__struct.rootspecial[fTagnum]->version = versionin;
   }
}

//______________________________________________________________________________
const char* Cint::G__ClassInfo::DefFile()
{
   if (IsValid()) {
      CheckValidRootInfo();
      return G__struct.rootspecial[fTagnum]->deffile;
   }
   return 0;
}

//______________________________________________________________________________
int Cint::G__ClassInfo::DefLine()
{
   if (IsValid()) {
      CheckValidRootInfo();
      return G__struct.rootspecial[fTagnum]->defline;
   }
   return -1;
}

//______________________________________________________________________________
const char* Cint::G__ClassInfo::ImpFile()
{
   if (IsValid()) {
      CheckValidRootInfo();
      return G__struct.rootspecial[fTagnum]->impfile;
   }
   return 0;
}

//______________________________________________________________________________
int Cint::G__ClassInfo::ImpLine()
{
   if (IsValid()) {
      CheckValidRootInfo();
      return G__struct.rootspecial[fTagnum]->impline;
   }
   return -1;
}

//______________________________________________________________________________
int Cint::G__ClassInfo::Version()
{
   if (IsValid()) {
      CheckValidRootInfo();
      return G__struct.rootspecial[fTagnum]->version;
   }
   return -1;
}

//______________________________________________________________________________
void* Cint::G__ClassInfo::New()
{
   if (!IsValid()) {
      return 0;
   }
   void* p = 0;
   G__value buf = G__null;
   if (!fClassProperty) {
      Property();
   }
   if (fClassProperty & G__BIT_ISCPPCOMPILED) { // C++ precompiled class,struct
      G__param para;
      para.paran = 0;
      if (!G__struct.rootspecial[fTagnum]) {
         CheckValidRootInfo();
      }
      G__InterfaceMethod defaultconstructor = (G__InterfaceMethod) G__struct.rootspecial[fTagnum]->defaultconstructor;
      if (defaultconstructor) {
         G__CurrentCall(G__DELETEFREE, this, &fTagnum);
         (*defaultconstructor)(&buf, 0, &para, 0);
         G__CurrentCall(G__NOP, 0, 0);
         p = (void*) G__int(buf);
      }
   }
   else if (fClassProperty & G__BIT_ISCCOMPILED) { // C precompiled class,struct
      p = new char[G__struct.size[fTagnum]];
   }
   else { // Interpreted class,struct
      p = new char[G__struct.size[fTagnum]];
      ::Reflex::Scope store_tagnum = G__tagnum;
      char* store_struct_offset = G__store_struct_offset;
      G__tagnum = G__Dict::GetDict().GetScope(fTagnum);
      G__store_struct_offset = (char*) p;
      G__StrBuf temp_sb(G__ONELINE);
      char* temp = temp_sb;
      sprintf(temp, "%s()", G__struct.name[fTagnum]);
      int known = 0;
      G__getfunction(temp, &known, G__CALLCONSTRUCTOR);
      G__store_struct_offset = store_struct_offset;
      G__tagnum = store_tagnum;
   }
   return p;
}

//______________________________________________________________________________
void* Cint::G__ClassInfo::New(int n)
{
   if (!IsValid() || (n <= 0)) {
      return 0;
   }
   void* p = 0;
   G__value buf = G__null;
   if (!fClassProperty) {
      Property();
   }
   if (fClassProperty & G__BIT_ISCPPCOMPILED) { // C++ precompiled class,struct
      G__param para;
      para.paran = 0;
      if (!G__struct.rootspecial[fTagnum]) {
         CheckValidRootInfo();
      }
      G__InterfaceMethod defaultconstructor = (G__InterfaceMethod) G__struct.rootspecial[fTagnum]->defaultconstructor;
      if (defaultconstructor) {
         if (n) {
            G__cpp_aryconstruct = n;
         }
         G__CurrentCall(G__DELETEFREE, this, &fTagnum);
         (*defaultconstructor)(&buf, 0, &para, 0);
         G__CurrentCall(G__NOP, 0, 0);
         G__cpp_aryconstruct = 0;
         p = (void*) G__int(buf);
         // Record that we have allocated an array, and how many
         // elements that array has, for use by the G__calldtor function.
         G__alloc_newarraylist(p, n);
      }
   }
   else if (fClassProperty & G__BIT_ISCCOMPILED) { // C precompiled class,struct
      p = new char[G__struct.size[fTagnum]*n];
   }
   else { // Interpreted class,struct
      p = new char[G__struct.size[fTagnum]*n];
      // Record that we have allocated an array, and how many
      // elements that array has, for use by the G__calldtor function.
      G__alloc_newarraylist(p, n);
      ::Reflex::Scope store_tagnum = G__tagnum;
      char* store_struct_offset = G__store_struct_offset;
      G__tagnum = G__Dict::GetDict().GetScope(fTagnum);
      G__store_struct_offset = (char*) p;
      //// Do it this way for an array cookie implementation.
      ////p = new char[(G__struct.size[fTagnum]*n)+(2*sizeof(int))];
      ////int* pp = (int*) p;
      ////pp[0] = G__struct.size[fTagnum];
      ////pp[1] = n;
      ////G__store_struct_offset = (long)(((char*)p) + (2*sizeof(int)));
      ////... at end adjust returned pointer address ...
      ////p = ((char*) p) + (2 * sizeof(int));
      G__StrBuf temp_sb(G__ONELINE);
      char* temp = temp_sb;
      sprintf(temp, "%s()", G__struct.name[fTagnum]);
      for (int i = 0; i < n; ++i) {
         int known = 0;
         G__getfunction(temp, &known, G__CALLCONSTRUCTOR);
         if (!known) {
            break;
         }
         G__store_struct_offset += G__struct.size[fTagnum];
      }
      G__store_struct_offset = store_struct_offset;
      G__tagnum = store_tagnum;
   }
   return p;
}

//______________________________________________________________________________
void* Cint::G__ClassInfo::New(void* arena)
{
   if (!IsValid()) {
      return 0;
   }
   void* p = 0;
   G__value buf = G__null;
   if (!fClassProperty) {
      Property();
   }
   if (fClassProperty & G__BIT_ISCPPCOMPILED) { // C++ precompiled class,struct
      G__param para;
      para.paran = 0;
      if (!G__struct.rootspecial[fTagnum]) {
         CheckValidRootInfo();
      }
      G__InterfaceMethod defaultconstructor = (G__InterfaceMethod) G__struct.rootspecial[fTagnum]->defaultconstructor;
      if (defaultconstructor) {
         G__setgvp((long) arena);
         G__CurrentCall(G__DELETEFREE, this, &fTagnum);
#ifdef G__ROOT
         G__exec_alloc_lock();
#endif // G__ROOT
         (*defaultconstructor)(&buf, 0, &para, 0);
#ifdef G__ROOT
         G__exec_alloc_unlock();
#endif // G__ROOT
         G__CurrentCall(G__NOP, 0, 0);
         G__setgvp((long) G__PVOID);
         p = (void*) G__int(buf);
      }
   }
   else if (fClassProperty & G__BIT_ISCCOMPILED) { // C precompiled class,struct
      p = arena;
   }
   else { // Interpreted class,struct
      p = arena;
      ::Reflex::Scope store_tagnum = G__tagnum;
      char* store_struct_offset = G__store_struct_offset;
      G__tagnum = G__Dict::GetDict().GetScope(fTagnum);
      G__store_struct_offset = (char*) p;
      G__StrBuf temp_sb(G__ONELINE);
      char* temp = temp_sb;
      sprintf(temp, "%s()", G__struct.name[fTagnum]);
      int known = 0;
      G__getfunction(temp, &known, G__CALLCONSTRUCTOR);
      G__store_struct_offset = store_struct_offset;
      G__tagnum = store_tagnum;
   }
   return p;
}

//______________________________________________________________________________
void* Cint::G__ClassInfo::New(int n, void* arena)
{
   if (!IsValid() || (n <= 0)) {
      return 0;
   }
   void* p = 0;
   G__value buf = G__null;
   if (!fClassProperty) {
      Property();
   }
   if (fClassProperty & G__BIT_ISCPPCOMPILED) { // C++ precompiled class,struct
      G__param para;
      para.paran = 0;
      if (!G__struct.rootspecial[fTagnum]) {
         CheckValidRootInfo();
      }
      G__InterfaceMethod defaultconstructor = (G__InterfaceMethod) G__struct.rootspecial[fTagnum]->defaultconstructor;
      if (defaultconstructor) {
         G__cpp_aryconstruct = n;
         G__setgvp((long) arena);
         G__CurrentCall(G__DELETEFREE, this, &fTagnum);
         (*defaultconstructor)(&buf, 0, &para, 0);
         G__CurrentCall(G__NOP, 0, 0);
         G__setgvp((long) G__PVOID);
         G__cpp_aryconstruct = 0;
         p = (void*) G__int(buf);
         // Record that we have allocated an array, and how many
         // elements that array has, for use by the G__calldtor function.
         G__alloc_newarraylist(p, n);
      }
   }
   else if (fClassProperty & G__BIT_ISCCOMPILED) { // C precompiled class,struct
      p = arena;
   }
   else { // Interpreted class,struct
      p = arena;
      // Record that we have allocated an array, and how many
      // elements that array has, for use by the delete[] operator.
      G__alloc_newarraylist(p, n);
      ::Reflex::Scope store_tagnum = G__tagnum;
      char* store_struct_offset = G__store_struct_offset;
      G__tagnum = G__Dict::GetDict().GetScope(fTagnum);
      G__store_struct_offset = (char*) p;
      //// Do it this way for an array cookie implementation.
      ////p = arena;
      ////int* pp = (int*) p;
      ////pp[0] = G__struct.size[fTagnum];
      ////pp[1] = n;
      ////G__store_struct_offset = (long)(((char*)p) + (2*sizeof(int)));
      ////... at end adjust returned pointer address ...
      ////p = ((char*) p) + (2 * sizeof(int));
      G__StrBuf temp_sb(G__ONELINE);
      char* temp = temp_sb;
      sprintf(temp, "%s()", G__struct.name[fTagnum]);
      for (int i = 0; i < n; ++i) {
         int known = 0;
         G__getfunction(temp, &known, G__CALLCONSTRUCTOR);
         if (!known) {
            break;
         }
         G__store_struct_offset += G__struct.size[fTagnum];
      }
      G__store_struct_offset = store_struct_offset;
      G__tagnum = store_tagnum;
   }
   return p;
}

//______________________________________________________________________________
void Cint::G__ClassInfo::Delete(void* p) const
{
   G__calldtor(p, G__Dict::GetDict().GetScope(fTagnum), 1);
}

//______________________________________________________________________________
void Cint::G__ClassInfo::Destruct(void* p) const
{
   G__calldtor(p, G__Dict::GetDict().GetScope(fTagnum), 0);
}

//______________________________________________________________________________
void Cint::G__ClassInfo::DeleteArray(void* ary, int dtorOnly /*= 0*/)
{
   // Array Destruction, with optional deletion.
   if (!IsValid()) {
      return;
   }
   if (!fClassProperty) {
      Property();
   }
   if (fClassProperty & G__BIT_ISCPPCOMPILED) { // C++ precompiled class,struct
      // Fetch the number of elements in the array that
      // we saved when we originally allocated it.
      G__cpp_aryconstruct = G__free_newarraylist(ary);
      if (dtorOnly) {
         Destruct(ary);
      }
      else {
         Delete(ary);
      }
      G__cpp_aryconstruct = 0;
   }
   else if (fClassProperty & G__BIT_ISCCOMPILED) { // C precompiled class,struct
      if (!dtorOnly) {
         free(ary);
      }
   }
   else { // Interpreted class,struct
      // Fetch the number of elements in the array that
      // we saved when we originally allocated it.
      int n = G__free_newarraylist(ary);
      int element_size = G__struct.size[fTagnum];
      //// Do it this way for an array cookie implementation.
      ////int* pp = (int*) ary;
      ////int n = pp[-1];
      ////int element_size = pp[-2];
      char* r = ((char*) ary) + ((n - 1) * element_size);
      int status = 0;
      for (int i = n; i > 0; --i) {
         status = G__calldtor(r, G__Dict::GetDict().GetScope(fTagnum), 0);
         // ???FIX ME:  What does status mean here?
         // if (!status) break;
         r -= element_size;
      }
      if (!dtorOnly) {
         free(ary);
      }
   }
   return;
}

//______________________________________________________________________________
int Cint::G__ClassInfo::InstanceCount()
{
   if (IsValid()) {
      CheckValidRootInfo();
      return G__struct.rootspecial[fTagnum]->instancecount;
   }
   return 0;
}

//______________________________________________________________________________
void Cint::G__ClassInfo::ResetInstanceCount()
{
   if (IsValid()) {
      CheckValidRootInfo();
      G__struct.rootspecial[fTagnum]->instancecount = 0;
   }
}

//______________________________________________________________________________
void Cint::G__ClassInfo::IncInstanceCount()
{
   if (IsValid()) {
      CheckValidRootInfo();
      G__struct.rootspecial[fTagnum]->instancecount += 1;
   }
}

//______________________________________________________________________________
int Cint::G__ClassInfo::HeapInstanceCount()
{
   if (IsValid()) {
      CheckValidRootInfo();
      return G__struct.rootspecial[fTagnum]->heapinstancecount;
   }
   return 0;
}

//______________________________________________________________________________
void Cint::G__ClassInfo::IncHeapInstanceCount()
{
   if (IsValid()) {
      CheckValidRootInfo();
      G__struct.rootspecial[fTagnum]->heapinstancecount += 1;
   }
}

//______________________________________________________________________________
void Cint::G__ClassInfo::ResetHeapInstanceCount()
{
   if (IsValid()) {
      CheckValidRootInfo();
      G__struct.rootspecial[fTagnum]->heapinstancecount = 0;
   }
}

//______________________________________________________________________________
int Cint::G__ClassInfo::RootFlag()
{
   return G__struct.rootflag[fTagnum];
}

//______________________________________________________________________________
G__InterfaceMethod Cint::G__ClassInfo::GetInterfaceMethod(const char* fname, const char* arg, long* poffset, MatchMode mode /*= ConversionMatch*/, InheritanceMode imode /*= WithInheritance*/)
{
   // Search for method.
   Reflex::Scope memscope = ::Reflex::Scope::GlobalScope();
   if (fTagnum != -1) {
      memscope = G__Dict::GetDict().GetScope(fTagnum);
   }
   char* funcname = (char*) fname;
   char* param = (char*) arg;
   long index = 0L;
   G__ifunc_table* methodHandle = G__get_methodhandle(funcname, param, (G__ifunc_table*) memscope.Id(), &index, poffset, (mode == ConversionMatch) ? 1 : 0, imode);
   if (!methodHandle) {
      return 0;
   }
   ::Reflex::Member mbr(reinterpret_cast<const ::Reflex::MemberBase*>(methodHandle));
   G__RflxFuncProperties* prop = G__get_funcproperties(mbr);
   if (prop->entry.size != -1) {
      return 0;
   }
   return (G__InterfaceMethod) prop->entry.p;
}

//______________________________________________________________________________
G__MethodInfo Cint::G__ClassInfo::GetMethod(const char* fname, const char* arg, long* poffset, MatchMode mode /*= ConversionMatch*/, InheritanceMode imode /*= WithInheritance*/)
{
   // Search for method.
   Reflex::Scope memscope = ::Reflex::Scope::GlobalScope();
   if (fTagnum != -1) {
      memscope = G__Dict::GetDict().GetScope(fTagnum);
   }
   char* funcname = (char*) fname;
   char* param = (char*) arg;
   int convmode = 0;
   switch (mode) {
      case ExactMatch:
         convmode = 0;
         break;
      case ConversionMatch:
         convmode = 1;
         break;
      case ConversionMatchBytecode:
         convmode = 2;
         break;
      default:
         convmode = 0;
         break;
   }
   long index = 0L;
   G__ifunc_table* methodHandle = G__get_methodhandle(funcname, param, (G__ifunc_table*) memscope.Id(), &index, poffset, convmode, (imode == WithInheritance) ? 1 : 0);
   G__MethodInfo method;
   method.Init((long) methodHandle, index, this);
   return method;
}

//______________________________________________________________________________
G__MethodInfo Cint::G__ClassInfo::GetMethod(const char* fname, G__param* libp, long* poffset, MatchMode mode /*= ConversionMatch*/, InheritanceMode imode /*= WithInheritance*/)
{
   // Search for method.
   char* funcname = (char*) fname;
   Reflex::Scope memscope = ::Reflex::Scope::GlobalScope();
   if (fTagnum != -1) {
      memscope = G__Dict::GetDict().GetScope(fTagnum);
   }
   long index = 0L;
   G__ifunc_table* methodHandle = G__get_methodhandle2(funcname, libp, (G__ifunc_table*) memscope.Id(), &index, poffset, (mode == ConversionMatch) ? 1 : 0, (imode == WithInheritance) ? 1 : 0);
   G__MethodInfo method;
   method.Init((long) methodHandle, index, this);
   return method;
}

//______________________________________________________________________________
G__MethodInfo Cint::G__ClassInfo::GetDefaultConstructor()
{
   // TODO, reserve location for default ctor for tune up
   char* fname = (char*) malloc(strlen(Name()) + 1);
   sprintf(fname, "%s", Name());
   long dmy = 0L;
   G__MethodInfo method = GetMethod(fname, "", &dmy, ExactMatch, InThisScope);
   free(fname);
   return method;
}

//______________________________________________________________________________
G__MethodInfo Cint::G__ClassInfo::GetCopyConstructor()
{
   // TODO, reserve location for copy ctor for tune up
   char* fname = (char*) malloc(strlen(Name()) + 1);
   sprintf(fname, "%s", Name());
   char* arg = (char*) malloc(strlen(Name()) + 10);
   sprintf(arg, "const %s&", Name());
   long dmy = 0L;
   G__MethodInfo method = GetMethod(fname, arg, &dmy, ExactMatch, InThisScope);
   free(arg);
   free(fname);
   return method;
}

//______________________________________________________________________________
G__MethodInfo Cint::G__ClassInfo::GetDestructor()
{
   // TODO, dtor location is already reserved, ready for tune up
   char* fname = (char*) malloc(strlen(Name()) + 2);
   sprintf(fname, "~%s", Name());
   long dmy = 0L;
   G__MethodInfo method = GetMethod(fname, "", &dmy, ExactMatch, InThisScope);
   free(fname);
   return method;
}

//______________________________________________________________________________
G__MethodInfo Cint::G__ClassInfo::GetAssignOperator()
{
   // TODO, reserve operator= location for tune up
   char* arg = (char*) malloc(strlen(Name()) + 10);
   sprintf(arg, "const %s&", Name());
   long dmy = 0L;
   G__MethodInfo method = GetMethod("operator=", arg, &dmy, ExactMatch, InThisScope);
   free(arg);
   return method;
}

//______________________________________________________________________________
Cint::G__MethodInfo Cint::G__ClassInfo::AddMethod(const char* /*typenam*/, const char* /*fname*/, const char* /*arg*/, int /*isstatic*/ /*= 0*/, int /*isvirtual*/ /*= 0*/, void* /*methodAddress*/ /*= 0*/)
{
   // FIXME: Needs to be implemented!!!
   Reflex::Scope memscope = ::Reflex::Scope::GlobalScope();
   if (fTagnum != -1) {
      memscope = G__Dict::GetDict().GetScope(fTagnum);
   }
   //long index = 0L;
   return G__MethodInfo();
}

//______________________________________________________________________________
G__DataMemberInfo Cint::G__ClassInfo::GetDataMember(const char* name, long* poffset)
{
   // Search for variable.
   int hash = 0;
   int temp= 0;
   G__hash(name, hash, temp)
   char* varname = (char*) name;
   *poffset = 0;
   ::Reflex::Scope var = Reflex::Scope::GlobalScope();
   if (fTagnum != -1) {
      var = G__Dict::GetDict().GetScope(fTagnum);
   }
   ::Reflex::Scope store_tagnum = G__tagnum;
   G__tagnum = G__Dict::GetDict().GetScope(fTagnum);
   long original = 0;
   int idx = 0;
   var = G__Dict::GetDict().GetScope(G__searchvariable(varname, hash, (G__var_array*) var.Id(), 0, poffset, &original, &idx, 0));
   G__tagnum = store_tagnum;
   G__DataMemberInfo datamember;
   datamember.Init((long) var.Id(), (long) idx, this);
   return datamember;
}

//______________________________________________________________________________
int Cint::G__ClassInfo::HasMethod(const char* fname)
{
   if (IsValid()) {
      G__incsetup_memfunc((int) fTagnum);
      if (G__Dict::GetDict().GetScope(fTagnum).FunctionMemberByName(std::string(fname))) {
         return 1;
      }
   }
   return 0;
}

//______________________________________________________________________________
int Cint::G__ClassInfo::HasDataMember(const char* name)
{
   if (IsValid()) {
      G__incsetup_memvar((int) fTagnum);
      if (G__Dict::GetDict().GetScope(fTagnum).DataMemberByName(std::string(name))) {
         return 1;
      }
   }
   return 0;
}

//______________________________________________________________________________
int Cint::G__ClassInfo::HasDefaultConstructor()
{
   if (IsValid()) {
      CheckValidRootInfo();
      return G__struct.rootspecial[fTagnum]->defaultconstructor != 0;
   }
   return 0;
}

//______________________________________________________________________________
void Cint::G__ClassInfo::CheckValidRootInfo()
{
   if (G__struct.rootspecial[fTagnum]) {
      return;
   }
   G__struct.rootspecial[fTagnum] = (G__RootSpecial*) malloc(sizeof(G__RootSpecial));
   G__struct.rootspecial[fTagnum]->deffile = 0;
   G__struct.rootspecial[fTagnum]->impfile = 0;
   G__struct.rootspecial[fTagnum]->defline = 0;
   G__struct.rootspecial[fTagnum]->impline = 0;
   G__struct.rootspecial[fTagnum]->version = 0;
   G__struct.rootspecial[fTagnum]->instancecount = 0;
   G__struct.rootspecial[fTagnum]->heapinstancecount = 0;
   long offset = 0L;
   G__struct.rootspecial[fTagnum]->defaultconstructor = (void*) GetInterfaceMethod(G__struct.name[fTagnum], "", &offset);
}

static long G__ClassInfo_MemberFunctionProperty(long& property, int fTagnum);
static long G__ClassInfo_BaseClassProperty(long& property, G__ClassInfo& classinfo);
static long G__ClassInfo_DataMemberProperty(long& property, int fTagnum);

//______________________________________________________________________________
long Cint::G__ClassInfo::ClassProperty()
{
   if (!IsValid()) {
      return 0L;
   }
   long property = 0L;
   switch (G__struct.type[fTagnum]) {
      case 'e':
      case 'u':
         return 0L;
      case 'c':
      case 's':
         property |= G__CLS_VALID;
   }
   if (G__struct.isabstract[fTagnum]) {
      property |= G__CLS_ISABSTRACT;
   }
   G__ClassInfo_MemberFunctionProperty(property, (int) fTagnum);
   G__ClassInfo_BaseClassProperty(property, *this);
   G__ClassInfo_DataMemberProperty(property, (int) fTagnum);
   return property;
}

//______________________________________________________________________________
static long G__ClassInfo_MemberFunctionProperty(long& property, int fTagnum)
{
   Reflex::Scope memscope = G__Dict::GetDict().GetScope(fTagnum);
   for (Reflex::Member_Iterator iter = memscope.FunctionMember_Begin(); iter != memscope.FunctionMember_End(); ++iter) {
      if (iter->IsConstructor()) {
         property |= G__CLS_HASEXPLICITCTOR;
         if (!iter->TypeOf().FunctionParameterSize()) {
            property |= G__CLS_HASDEFAULTCTOR;
         }
         if (iter->TypeOf().IsVirtual()) {
            property |= G__CLS_HASIMPLICITCTOR;
         }
      }
      if (iter->IsDestructor()) {
         property |= G__CLS_HASEXPLICITDTOR;
      }
      if (iter->Name() == "operator=") {
         property |= G__CLS_HASASSIGNOPR;
      }
      if (iter->IsVirtual()) {
         property |= G__CLS_HASVIRTUAL;
      }
   }
   return property;
}

//______________________________________________________________________________
static long G__ClassInfo_BaseClassProperty(long& property, G__ClassInfo& classinfo)
{
   G__BaseClassInfo baseinfo(classinfo);
   while (baseinfo.Next()) {
      long baseprop = baseinfo.ClassProperty();
      if (!(property & G__CLS_HASEXPLICITCTOR) && (baseprop & G__CLS_HASCTOR)) {
         property |= (G__CLS_HASIMPLICITCTOR | G__CLS_HASDEFAULTCTOR);
      }
      if (!(property & G__CLS_HASEXPLICITDTOR) && (baseprop & G__CLS_HASDTOR)) {
         property |= G__CLS_HASIMPLICITDTOR;
      }
      if (baseprop & G__CLS_HASVIRTUAL) {
         property |= G__CLS_HASVIRTUAL;
      }
   }
   return property;
}

//______________________________________________________________________________
static long G__ClassInfo_DataMemberProperty(long& property, int fTagnum)
{
   ::Reflex::Scope var = G__Dict::GetDict().GetScope(fTagnum);
   if (var) {
      for (::Reflex::Member_Iterator iter = var.DataMember_Begin(); iter != var.DataMember_End(); ++iter) {
         ::Reflex::Type memType = iter->TypeOf();
         if (memType.RawType().IsClass() && !(memType.IsPointer() || memType.IsReference())) {
            G__ClassInfo classinfo(::Cint::Internal::G__get_tagnum(memType));
            long baseprop = classinfo.ClassProperty();
            if (!(property & G__CLS_HASEXPLICITCTOR) && (baseprop & G__CLS_HASCTOR)) {
               property |= (G__CLS_HASIMPLICITCTOR | G__CLS_HASDEFAULTCTOR);
            }
            if (!(property & G__CLS_HASEXPLICITDTOR) && (baseprop & G__CLS_HASDTOR)) {
               property |= G__CLS_HASIMPLICITDTOR;
            }
         }
      }
   }
   return property;
}

//______________________________________________________________________________
unsigned char Cint::G__ClassInfo::FuncFlag()
{
   if (IsValid()) {
      return G__struct.funcs[fTagnum];
   }
   return 0;
}

