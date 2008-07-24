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

// Class header first.
//#include "Class.h"

// All other non-system headers.
#include "Api.h"
#include "common.h"
#include "Dict.h"

// Using declarations.
using namespace std;
using namespace Cint::Internal;

/*********************************************************************
* class Cint::G__ClassInfo
*********************************************************************/

///////////////////////////////////////////////////////////////////////////
Cint::G__ClassInfo::~G__ClassInfo()
{
}

///////////////////////////////////////////////////////////////////////////
Cint::G__ClassInfo::G__ClassInfo()
: tagnum(0)
, class_property(0)
{
   Init();
}

///////////////////////////////////////////////////////////////////////////
Cint::G__ClassInfo::G__ClassInfo(const G__value &value_for_type)
  : tagnum(0)
    , class_property(0)
{
   Init(G__get_tagnum(G__value_typenum(value_for_type).RawType()));
}

///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::Init()
{
  tagnum = -1;
  class_property = 0;
}

///////////////////////////////////////////////////////////////////////////
Cint::G__ClassInfo::G__ClassInfo(const char* classname)
: tagnum(0)
, class_property(0)
{
   Init(classname);
}

///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::Init(const char* classname)
{
   tagnum = G__defined_tagname(classname, 1);
   class_property = 0;
}

///////////////////////////////////////////////////////////////////////////
Cint::G__ClassInfo::G__ClassInfo(int tagnumin)
: tagnum(0)
, class_property(0)
{
   Init(tagnumin);
}

///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::Init(int tagnumin)
{
   tagnum = tagnumin;
   class_property = 0;
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::operator==(const G__ClassInfo& a)
{
   return(tagnum == a.tagnum);
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::operator!=(const G__ClassInfo& a)
{
   return(tagnum != a.tagnum);
}

///////////////////////////////////////////////////////////////////////////
const char* Cint::G__ClassInfo::Name()
{
   if (IsValid()) {
      return G__struct.name[tagnum];
   } else {
      return 0;
   }
}

///////////////////////////////////////////////////////////////////////////
const char* Cint::G__ClassInfo::Fullname()
{
   static char G__buf[G__ONELINE];
#ifdef __GNUC__
#else
#pragma message(FIXME("Get rid of static G__buf by proper lookup"))
#endif

   if (IsValid()) {
      strcpy(G__buf, G__fulltagname((int) tagnum, 1));
#if defined(_MSC_VER) && (_MSC_VER < 1300) /*vc6*/
      char* ptr = strstr(G__buf, "long long");
      if (ptr) {
         memcpy(ptr, " __int64 ", strlen(" __int64 "));
      }
#endif
      // FIXME: We are returning a pointer to a statically allocated
      //        buffer, this is not thread-safe.
      return G__buf;
   } else {
      return 0;
   }
}

///////////////////////////////////////////////////////////////////////////
const char* Cint::G__ClassInfo::Title()
{
   static char buf[G__INFO_TITLELEN];
   buf[0] = '\0';
   if (IsValid()) {
      G__getcomment(buf, &G__struct.comment[tagnum], (int) tagnum);
      // FIXME: We are returning a pointer to a statically allocated
      //        buffer, this is not thread-safe.
      return buf;
   } else {
      return 0;
   }
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::Size()
{
   if (IsValid()) {
      return G__struct.size[tagnum];
   } else {
      return -1;
   }
}

///////////////////////////////////////////////////////////////////////////
long Cint::G__ClassInfo::Property()
{
  if (class_property) {
     return class_property;
  }
  long property = 0L;
  if (IsValid()) {
    switch (G__struct.type[tagnum]) {
    case 'e': property |= G__BIT_ISENUM; break;
    case 'c': property |= G__BIT_ISCLASS; break;
    case 's': property |= G__BIT_ISSTRUCT; break;
    case 'u': property |= G__BIT_ISUNION; break;
    case 'n': property |= G__BIT_ISNAMESPACE; break;
    }
    if (G__struct.istypedefed[tagnum]) {
       property |= G__BIT_ISTYPEDEF;
    }
    if (G__struct.isabstract[tagnum]) {
       property |= G__BIT_ISABSTRACT;
    }
    switch (G__struct.iscpplink[tagnum]) {
    case G__CPPLINK: property |= G__BIT_ISCPPCOMPILED; break;
    case G__CLINK: property |= G__BIT_ISCCOMPILED; break;
    case G__NOLINK: break;
    default: break;
    }
    class_property = property;
  }
  return property;
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::NDataMembers()
{

   if (IsValid()) {
      G__incsetup_memvar((int)tagnum);
      return G__Dict::GetDict().GetScope(tagnum).DataMemberSize();
   }
   return -1;
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::NMethods()
{
   if (IsValid()) {
      G__incsetup_memfunc((int)tagnum);
      return G__Dict::GetDict().GetScope(tagnum).FunctionMemberSize();
   }
   return -1;
}

///////////////////////////////////////////////////////////////////////////
long Cint::G__ClassInfo::IsBase(const char* classname)
{
   Cint::G__ClassInfo base(classname);
   return IsBase(base);
}

///////////////////////////////////////////////////////////////////////////
long Cint::G__ClassInfo::IsBase(G__ClassInfo& a)
{
   long isbase = 0L;
   if (IsValid()) {
      G__inheritance* baseclass = G__struct.baseclass[tagnum];
      for (int i = 0; i < baseclass->basen; i++) {
         if (a.Tagnum() == baseclass->basetagnum[i]) {
            switch (baseclass->baseaccess[i]) {
               case G__PUBLIC: isbase = G__BIT_ISPUBLIC; break;
               case G__PROTECTED: isbase = G__BIT_ISPROTECTED; break;
               case G__PRIVATE: isbase = G__BIT_ISPRIVATE; break;
               default: break;
            }
            if (baseclass->property[i] & G__ISDIRECTINHERIT) {
               isbase |= G__BIT_ISDIRECTINHERIT;
            }
            if (baseclass->property[i] & G__ISVIRTUALBASE) {
               isbase |= G__BIT_ISVIRTUALBASE;
            }
            break;
         }
      }
   }
   return isbase;
}

///////////////////////////////////////////////////////////////////////////
long Cint::G__ClassInfo::Tagnum() const
{
   return tagnum;
}
///////////////////////////////////////////////////////////////////////////
::Reflex::Type Cint::G__ClassInfo::ReflexType()
{
   return G__Dict::GetDict().GetType(tagnum);
}
///////////////////////////////////////////////////////////////////////////
G__ClassInfo Cint::G__ClassInfo::EnclosingClass()
{
   if (IsValid()) {
      Cint::G__ClassInfo enclosingclass(G__struct.parent_tagnum[tagnum]);
      return enclosingclass;
   } else {
      Cint::G__ClassInfo enclosingclass;
      return enclosingclass;
   }
}

///////////////////////////////////////////////////////////////////////////
G__ClassInfo Cint::G__ClassInfo::EnclosingSpace()
{
   if (IsValid()) {
      int enclosed_tag = G__struct.parent_tagnum[tagnum];
      while ((enclosed_tag >= 0) && (G__struct.type[enclosed_tag] != 'n')) {
         enclosed_tag = G__struct.parent_tagnum[enclosed_tag];
      }
      Cint::G__ClassInfo enclosingclass(enclosed_tag);
      return enclosingclass;
   } else {
      Cint::G__ClassInfo enclosingclass;
      return enclosingclass;
   }
}

///////////////////////////////////////////////////////////////////////////
struct G__friendtag* Cint::G__ClassInfo::GetFriendInfo()
{
   if (IsValid()) {
      // FIXME: Are we thread-safe?
      return G__struct.friendtag[tagnum];
   }
   return 0;
}

///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::SetGlobalcomp(int globalcomp)
{
   if (IsValid()) {
      G__struct.globalcomp[tagnum] = globalcomp;
   }
}

///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::SetProtectedAccess(int protectedaccess)
{
   if (IsValid()) {
      G__struct.protectedaccess[tagnum] = protectedaccess;
   }
}

///////////////////////////////////////////////////////////////////////////
#ifdef G__OLDIMPLEMENTATION1218_YET
int Cint::G__ClassInfo::IsValid()
{
   if ((tagnum >= 0) && (tagnum < G__struct.alltag)) {
      return 1;
   }
   return 0;
}
#else
int Cint::G__ClassInfo::IsValid()
{
  if ((tagnum >= 0) && (tagnum < G__struct.alltag)) {
    return 1;
  }
  return 0;
}
#endif

///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::IsLoaded()
{
   if (IsValid() &&
       ((G__struct.iscpplink[tagnum] != G__NOLINK) ||
        (G__struct.filenum[tagnum] != -1)
       ))
   {
      return 1;
   }
   return 0;
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::SetFilePos(const char* fname)
{
   struct G__dictposition* dict = G__get_dictpos((char*) fname);
   if (!dict) {
      return 0;
   }
   tagnum = dict->tagnum - 1;
   class_property = 0;
   return 1;
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::Next()
{
   ++tagnum;
   class_property = 0;
   return IsValid();
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::Linkage()
{
   return G__struct.globalcomp[tagnum];
}

///////////////////////////////////////////////////////////////////////////
const char* Cint::G__ClassInfo::FileName()
{
   if (IsValid()) {
      if (G__struct.filenum[tagnum] != -1) {
         return G__srcfile[G__struct.filenum[tagnum]].filename;
      } else {
         switch(G__struct.iscpplink[tagnum]) {
         case G__CLINK: return "(C compiled)";
         case G__CPPLINK: return "(C++ compiled)";
         default: return 0;
         }
      }
   }
   return 0;
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::LineNumber()
{
   if (IsValid()) {
      switch (G__struct.iscpplink[tagnum]) {
         case G__CLINK: return 0;
         case G__CPPLINK: return 0;
         case G__NOLINK:
            if (G__struct.filenum[tagnum] != -1) {
               return G__struct.line_number[tagnum];
            }
            return -1;
         default: return -1;
      }
   }
   return -1;
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::IsTmplt()
{
   return (IsValid() && strchr((char*) Name(), '<'));
}

///////////////////////////////////////////////////////////////////////////
const char* Cint::G__ClassInfo::TmpltName()
{
   static char buf[G__ONELINE];
   if (IsValid()) {
      strcpy(buf, Name());
      char* p = strchr(buf, '<');
      if (p) {
         *p = 0;
      }
      return buf;
   }
   return 0;
}

///////////////////////////////////////////////////////////////////////////
const char* Cint::G__ClassInfo::TmpltArg()
{
   static char buf[G__ONELINE];
   if (IsValid()) {
     char* p = strchr((char*) Name(), '<');
     if (p) {
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
   }
   return 0;
}

///////////////////////////////////////////////////////////////////////////

/*********************************************************************
* ROOT project special requirements
*********************************************************************/

///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::SetDefFile(char *deffilein)
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->deffile = deffilein;
  }
}

///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::SetDefLine(int deflinein)
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->defline = deflinein;
  }
}

///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::SetImpFile(char *impfilein)
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->impfile = impfilein;
  }
}

///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::SetImpLine(int implinein)
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->impline = implinein;
  }
}

///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::SetVersion(int versionin)
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->version = versionin;
  }
}

///////////////////////////////////////////////////////////////////////////
const char* Cint::G__ClassInfo::DefFile()
{
  if(IsValid()) {
    CheckValidRootInfo();
    return(G__struct.rootspecial[tagnum]->deffile);
  }
  return 0;
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::DefLine()
{
  if(IsValid()) {
    CheckValidRootInfo();
    return(G__struct.rootspecial[tagnum]->defline);
  }
  return -1;
}

///////////////////////////////////////////////////////////////////////////
const char* Cint::G__ClassInfo::ImpFile()
{
  if(IsValid()) {
    CheckValidRootInfo();
    return(G__struct.rootspecial[tagnum]->impfile);
  }
  return 0;
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::ImpLine()
{
  if(IsValid()) {
    CheckValidRootInfo();
    return(G__struct.rootspecial[tagnum]->impline);
  }
  return -1;
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::Version()
{
  if(IsValid()) {
    CheckValidRootInfo();
    return(G__struct.rootspecial[tagnum]->version);
  }
  return -1;
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::InstanceCount()
{
  if(IsValid()) {
    CheckValidRootInfo();
    return(G__struct.rootspecial[tagnum]->instancecount);
  }
  return 0;
}

///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::ResetInstanceCount()
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->instancecount = 0;
  }
}

///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::IncInstanceCount()
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->instancecount += 1;
  }
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::HeapInstanceCount()
{
  if(IsValid()) {
    CheckValidRootInfo();
    return(G__struct.rootspecial[tagnum]->heapinstancecount);
  }
  return 0;
}

///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::ResetHeapInstanceCount()
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->heapinstancecount = 0;
  }
}

///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::IncHeapInstanceCount()
{
  if(IsValid()) {
    CheckValidRootInfo();
    G__struct.rootspecial[tagnum]->heapinstancecount += 1;
  }
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::RootFlag()
{
   return G__struct.rootflag[tagnum];
}

///////////////////////////////////////////////////////////////////////////
G__InterfaceMethod Cint::G__ClassInfo::GetInterfaceMethod(
  const char* fname, const char* arg, long* poffset,
  MatchMode mode, InheritanceMode imode)
{

  Reflex::Member ifunc;
  Reflex::Scope memscope;
  char *funcname;
  char *param;
  long index;

  /* Search for method */
  /* Instead of the G__ifunc_table we pass the first member scope Id */
  if (-1==tagnum) memscope = Reflex::Scope::GlobalScope();
  else            memscope = G__Dict::GetDict().GetScope(tagnum);
  funcname = (char*)fname;
  param = (char*)arg;
  struct G__ifunc_table* methodHandle = G__get_methodhandle(funcname,param,(G__ifunc_table*)memscope.Id(),&index,poffset
                                                             ,(mode==ConversionMatch)?1:0
                                                             ,imode
                                                             );
  ifunc = G__Dict::GetDict().GetScope(methodHandle).FunctionMemberAt(index);

  if (ifunc)
    return (G__InterfaceMethod) G__get_funcproperties(ifunc)->ifmethod;

  return (G__InterfaceMethod) 0;
}

///////////////////////////////////////////////////////////////////////////
G__MethodInfo Cint::G__ClassInfo::GetMethod(
  const char* fname, const char* arg, long* poffset,
  MatchMode mode, InheritanceMode imode)
{
  Reflex::Scope memscope;
  /* Search for method */
  if(-1==tagnum) memscope = Reflex::Scope::GlobalScope();
  else           memscope = G__Dict::GetDict().GetScope(tagnum);
  char* funcname = (char*) fname;
  char* param = (char*) arg;
  int convmode = 0;
  switch (mode) {
     case ExactMatch:              convmode = 0; break;
     case ConversionMatch:         convmode = 1; break;
     case ConversionMatchBytecode: convmode = 2; break;
     default:                      convmode = 0; break;
  }

  long index = 0L;
  struct G__ifunc_table* methodHandle = G__get_methodhandle(funcname,param,
     (G__ifunc_table*)memscope.Id(),&index,poffset,
     convmode,(imode==WithInheritance)?1:0);
  //memscope = G__Dict::GetDict().GetScope(methodHandle);
  //return G__MethodInfo(memscope.FunctionMemberAt(index));
  Reflex::MemberBase* mb = reinterpret_cast<Reflex::MemberBase*>(methodHandle);
  return G__MethodInfo(mb);
}

///////////////////////////////////////////////////////////////////////////
G__MethodInfo Cint::G__ClassInfo::GetMethod(
  const char* fname, struct G__param* libp, long* poffset,
  MatchMode mode, InheritanceMode imode)
{
  Reflex::Member ifunc;
  Reflex::Scope memscope;
  char *funcname = (char*)fname;
  long index = 0L;

  /* Search for method */
  if(-1==tagnum) memscope = Reflex::Scope::GlobalScope();
  else           memscope = G__Dict::GetDict().GetScope(tagnum);

  struct G__ifunc_table* methodHandle = G__get_methodhandle2(
     funcname, libp, (G__ifunc_table*)memscope.Id(), &index, poffset,
     (mode == ConversionMatch) ? 1 : 0, 
     (imode == WithInheritance) ? 1 : 0);
  memscope = G__Dict::GetDict().GetScope(methodHandle);

  return G__MethodInfo(memscope.FunctionMemberAt(index));
}

///////////////////////////////////////////////////////////////////////////
G__MethodInfo Cint::G__ClassInfo::GetDefaultConstructor()
{
  // TODO, reserve location for default ctor for tune up
  long dmy;
  G__MethodInfo method;
  char *fname= (char*)malloc(strlen(Name())+1);
  sprintf(fname,"%s",Name());
  method = GetMethod(fname,"",&dmy,ExactMatch,InThisScope);
  free((void*)fname);
  return method;
}

///////////////////////////////////////////////////////////////////////////
G__MethodInfo Cint::G__ClassInfo::GetCopyConstructor()
{
  // TODO, reserve location for copy ctor for tune up
  long dmy;
  G__MethodInfo method;
  char *fname= (char*)malloc(strlen(Name())+1);
  sprintf(fname,"%s",Name());
  char *arg= (char*)malloc(strlen(Name())+10);
  sprintf(arg,"const %s&",Name());
  method = GetMethod(fname,arg,&dmy,ExactMatch,InThisScope);
  free((void*)arg);
  free((void*)fname);
  return method;
}

///////////////////////////////////////////////////////////////////////////
G__MethodInfo Cint::G__ClassInfo::GetDestructor()
{
  // TODO, dtor location is already reserved, ready for tune up
  long dmy;
  G__MethodInfo method;
  char *fname= (char*)malloc(strlen(Name())+2);
  sprintf(fname,"~%s",Name());
  method = GetMethod(fname,"",&dmy,ExactMatch,InThisScope);
  free((void*)fname);
  return(method);
}

///////////////////////////////////////////////////////////////////////////
G__MethodInfo Cint::G__ClassInfo::GetAssignOperator()
{
  // TODO, reserve operator= location for tune up
  long dmy;
  G__MethodInfo method;
  char *arg= (char*)malloc(strlen(Name())+10);
  sprintf(arg,"const %s&",Name());
  method = GetMethod("operator=",arg,&dmy,ExactMatch,InThisScope);
  free((void*)arg);
  return method;
}

///////////////////////////////////////////////////////////////////////////
G__DataMemberInfo Cint::G__ClassInfo::GetDataMember(const char* name, long* poffset)
{
  char *varname;
  int hash;
  int temp;
  long original=0;
  int ig15;
  ::Reflex::Scope var;
  ::Reflex::Scope store_tagnum;

  /* search for variable */
  G__hash(name,hash,temp);
  varname=(char*)name;
  *poffset = 0;

  if (-1==tagnum) var = Reflex::Scope::GlobalScope();
  else            var = G__Dict::GetDict().GetScope(tagnum);
  
  store_tagnum=G__tagnum;
  G__tagnum = G__Dict::GetDict().GetScope(tagnum);
  var = G__Dict::GetDict().GetScope(G__searchvariable(varname,hash,(G__var_array*)var.Id(),(struct G__var_array*)NULL
                                                      ,poffset,&original,&ig15,0));
  G__tagnum=store_tagnum;
  return G__DataMemberInfo(var.DataMemberAt(ig15));
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::HasDefaultConstructor()
{
   if (IsValid()) {
      CheckValidRootInfo();
      return G__struct.rootspecial[tagnum]->defaultconstructor != 0;
   }
   return 0;
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::HasMethod(const char* fname)
{
   if (IsValid()) {
      G__incsetup_memfunc((int)tagnum);
      if (G__Dict::GetDict().GetScope(tagnum).FunctionMemberByName(std::string(fname))) return 1;
   }
   return 0;
}

///////////////////////////////////////////////////////////////////////////
int Cint::G__ClassInfo::HasDataMember(const char* name)
{
   if (IsValid()) {
      G__incsetup_memvar((int)tagnum);
      if (G__Dict::GetDict().GetScope(tagnum).DataMemberByName(std::string(name))) return 1;
   }
   return 0;
}

///////////////////////////////////////////////////////////////////////////
void* Cint::G__ClassInfo::New()
{
   if (IsValid()) {
      void* p;
      G__value buf = G__null;
      if (!class_property) {
         Property();
      }
      if (class_property & G__BIT_ISCPPCOMPILED) {
         // C++ precompiled class,struct
         struct G__param para;
         G__InterfaceMethod defaultconstructor;
         para.paran = 0;
         if (!G__struct.rootspecial[tagnum]) {
            CheckValidRootInfo();
         }
         defaultconstructor =
           (G__InterfaceMethod) G__struct.rootspecial[tagnum]->defaultconstructor;
         if (defaultconstructor) {
            G__CurrentCall(G__DELETEFREE, this, &tagnum);
            (*defaultconstructor)(&buf, 0, &para, 0);
            G__CurrentCall(G__NOP, 0, 0);
            p = (void*) G__int(buf);
         } else {
            p = 0;
         }
      } else if (class_property & G__BIT_ISCCOMPILED) {
         // C precompiled class,struct
         p = new char[G__struct.size[tagnum]];
      } else {
         // Interpreted class,struct
         char *store_struct_offset;
         ::Reflex::Scope store_tagnum;
         G__StrBuf temp_sb(G__ONELINE);
         char *temp = temp_sb;
         int known = 0;
         p = new char[G__struct.size[tagnum]];
         store_tagnum = G__tagnum;
         store_struct_offset = G__store_struct_offset;
         G__tagnum = G__Dict::GetDict().GetScope(tagnum);
         G__store_struct_offset = (char*) p;
         sprintf(temp, "%s()", G__struct.name[tagnum]);
         G__getfunction(temp,&known,G__CALLCONSTRUCTOR);
         G__store_struct_offset = store_struct_offset;
         G__tagnum = store_tagnum;
      }
      return p;
   }
   return 0;
}

///////////////////////////////////////////////////////////////////////////
void* Cint::G__ClassInfo::New(int n)
{
  if(IsValid() && n>0 ) {
    void *p;
    G__value buf=G__null;
    if (!class_property) Property();
    if(class_property&G__BIT_ISCPPCOMPILED) {
      // C++ precompiled class,struct
      struct G__param para;
      G__InterfaceMethod defaultconstructor;
      para.paran=0;
      if(!G__struct.rootspecial[tagnum]) CheckValidRootInfo();
      defaultconstructor
        =(G__InterfaceMethod)G__struct.rootspecial[tagnum]->defaultconstructor;
      if(defaultconstructor) {
        if(n) G__cpp_aryconstruct = n;
        G__CurrentCall(G__DELETEFREE, this, &tagnum);
        (*defaultconstructor)(&buf,(char*)NULL,&para,0);
        G__CurrentCall(G__NOP, 0, 0);
        G__cpp_aryconstruct = 0;
        p = (void*)G__int(buf);
        // Record that we have allocated an array, and how many
        // elements that array has, for use by the G__calldtor function.
        G__alloc_newarraylist( p, n);
      }
      else {
        p = (void*)NULL;
      }
    }
    else if(class_property&G__BIT_ISCCOMPILED) {
      // C precompiled class,struct
      p = new char[G__struct.size[tagnum]*n];
    }
    else {
      // Interpreted class,struct
      int i;
      char *store_struct_offset;
      ::Reflex::Scope store_tagnum;
      G__StrBuf temp_sb(G__ONELINE);
      char *temp = temp_sb;
      int known=0;
      p = new char[G__struct.size[tagnum]*n];
      // Record that we have allocated an array, and how many
      // elements that array has, for use by the G__calldtor function.
      G__alloc_newarraylist( p, n);
      store_tagnum = G__tagnum;
      store_struct_offset = G__store_struct_offset;
      G__tagnum = G__Dict::GetDict().GetScope(tagnum);
      G__store_struct_offset = (char*)p;
      //// Do it this way for an array cookie implementation.
      ////p = new char[(G__struct.size[tagnum]*n)+(2*sizeof(int))];
      ////int* pp = (int*) p;
      ////pp[0] = G__struct.size[tagnum];
      ////pp[1] = n;
      ////G__store_struct_offset = (long)(((char*)p) + (2*sizeof(int)));
      ////... at end adjust returned pointer address ...
      ////p = ((char*) p) + (2 * sizeof(int));
      sprintf(temp,"%s()",G__struct.name[tagnum]);
      for(i=0;i<n;i++) {
        G__getfunction(temp,&known,G__CALLCONSTRUCTOR);
        if(!known) break;
        G__store_struct_offset += G__struct.size[tagnum];
      }
      G__store_struct_offset = store_struct_offset;
      G__tagnum = store_tagnum;
    }
    return(p);
  }
  else {
    return((void*)NULL);
  }
}

///////////////////////////////////////////////////////////////////////////
void* Cint::G__ClassInfo::New(void *arena)
{
  if(IsValid()) {
    void *p;
    G__value buf=G__null;
    if (!class_property) Property();
    if(class_property&G__BIT_ISCPPCOMPILED) {
      // C++ precompiled class,struct
      struct G__param para;
      G__InterfaceMethod defaultconstructor;
      para.paran=0;
      if(!G__struct.rootspecial[tagnum]) CheckValidRootInfo();
      defaultconstructor
        =(G__InterfaceMethod)G__struct.rootspecial[tagnum]->defaultconstructor;
      if(defaultconstructor) {
        G__setgvp((long)arena);
        G__CurrentCall(G__DELETEFREE, this, &tagnum);
#ifdef G__ROOT
        G__exec_alloc_lock();
#endif
        (*defaultconstructor)(&buf,(char*)NULL,&para,0);
#ifdef G__ROOT
        G__exec_alloc_unlock();
#endif
        G__CurrentCall(G__NOP, 0, 0);
        G__setgvp((long)G__PVOID);
        p = (void*)G__int(buf);
      }
      else {
        p = (void*)NULL;
      }
    }
    else if(class_property&G__BIT_ISCCOMPILED) {
      // C precompiled class,struct
      p = arena;
    }
    else {
      // Interpreted class,struct
      char *store_struct_offset;
      ::Reflex::Scope store_tagnum;
      G__StrBuf temp_sb(G__ONELINE);
      char *temp = temp_sb;
      int known=0;
      p = arena;
      store_tagnum = G__tagnum;
      store_struct_offset = G__store_struct_offset;
      G__tagnum = G__Dict::GetDict().GetScope(tagnum);
      G__store_struct_offset = (char*)p;
      sprintf(temp,"%s()",G__struct.name[tagnum]);
      G__getfunction(temp,&known,G__CALLCONSTRUCTOR);
      G__store_struct_offset = store_struct_offset;
      G__tagnum = store_tagnum;
    }
    return(p);
  }
  else {
    return((void*)NULL);
  }
}

///////////////////////////////////////////////////////////////////////////
void* Cint::G__ClassInfo::New(int n, void *arena)
{
  if(IsValid() && (n > 0)) {
    void *p;
    G__value buf=G__null;
    if (!class_property) Property();
    if(class_property&G__BIT_ISCPPCOMPILED) {
      // C++ precompiled class,struct
      struct G__param para;
      G__InterfaceMethod defaultconstructor;
      para.paran=0;
      if(!G__struct.rootspecial[tagnum]) CheckValidRootInfo();
      defaultconstructor
        =(G__InterfaceMethod)G__struct.rootspecial[tagnum]->defaultconstructor;
      if(defaultconstructor) {
        G__cpp_aryconstruct = n;
        G__setgvp((long)arena);
        G__CurrentCall(G__DELETEFREE, this, &tagnum);
        (*defaultconstructor)(&buf,(char*)NULL,&para,0);
        G__CurrentCall(G__NOP, 0, 0);
        G__setgvp((long)G__PVOID);
        G__cpp_aryconstruct = 0;
        p = (void*)G__int(buf);
        // Record that we have allocated an array, and how many
        // elements that array has, for use by the G__calldtor function.
        G__alloc_newarraylist( p, n);
      }
      else {
        p = (void*)NULL;
      }
    }
    else if(class_property&G__BIT_ISCCOMPILED) {
      // C precompiled class,struct
      p = arena;
    }
    else {
      // Interpreted class,struct
      char *store_struct_offset;
      ::Reflex::Scope store_tagnum;
      G__StrBuf temp_sb(G__ONELINE);
      char *temp = temp_sb;
      int known=0;
      p = arena;
      // Record that we have allocated an array, and how many
      // elements that array has, for use by the delete[] operator.
      G__alloc_newarraylist( p, n);
      store_tagnum = G__tagnum;
      store_struct_offset = G__store_struct_offset;
      G__tagnum = G__Dict::GetDict().GetScope(tagnum);
      G__store_struct_offset = (char*)p;
      //// Do it this way for an array cookie implementation.
      ////p = arena;
      ////int* pp = (int*) p;
      ////pp[0] = G__struct.size[tagnum];
      ////pp[1] = n;
      ////G__store_struct_offset = (long)(((char*)p) + (2*sizeof(int)));
      ////... at end adjust returned pointer address ...
      ////p = ((char*) p) + (2 * sizeof(int));
      sprintf(temp,"%s()",G__struct.name[tagnum]);
      for (int i = 0; i < n; ++i) {
        G__getfunction(temp,&known,G__CALLCONSTRUCTOR);
        if (!known) break;
        G__store_struct_offset += G__struct.size[tagnum];
      }
      G__store_struct_offset = store_struct_offset;
      G__tagnum = store_tagnum;
    }
    return(p);
  }
  else {
    return((void*)NULL);
  }
}

///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::Delete(void* p) const
{
   G__calldtor(p, G__Dict::GetDict().GetScope(tagnum), 1);
}

///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::Destruct(void* p) const
{
   G__calldtor(p, G__Dict::GetDict().GetScope(tagnum), 0);
}

///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::DeleteArray(void* ary, int dtorOnly)
{
  // Array Destruction, with optional deletion.
  if (!IsValid()) {
     return;
  }
  if (!class_property) {
     Property();
  }
  if (class_property & G__BIT_ISCPPCOMPILED) {
    // C++ precompiled class,struct
    // Fetch the number of elements in the array that
    // we saved when we originally allocated it.
    G__cpp_aryconstruct = G__free_newarraylist(ary);
    if (dtorOnly) {
      Destruct(ary);
    } else {
      Delete(ary);
    }
    G__cpp_aryconstruct = 0;
  } else if (class_property & G__BIT_ISCCOMPILED) {
    // C precompiled class,struct
    if (!dtorOnly) {
      free(ary);
    }
  } else {
    // Interpreted class,struct
    // Fetch the number of elements in the array that
    // we saved when we originally allocated it.
    int n = G__free_newarraylist(ary);
    int element_size = G__struct.size[tagnum];
    //// Do it this way for an array cookie implementation.
    ////int* pp = (int*) ary;
    ////int n = pp[-1];
    ////int element_size = pp[-2];
    char* r = ((char*) ary) + ((n - 1) * element_size) ;
    int status = 0;
    for (int i = n; i > 0; --i) {
      status = G__calldtor(r, G__Dict::GetDict().GetScope(tagnum), 0);
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

///////////////////////////////////////////////////////////////////////////
void Cint::G__ClassInfo::CheckValidRootInfo()
{
  long offset;
  if(G__struct.rootspecial[tagnum]) return;

  G__struct.rootspecial[tagnum] =
    (struct G__RootSpecial*) malloc(sizeof(struct G__RootSpecial));
  G__struct.rootspecial[tagnum]->deffile = 0;
  G__struct.rootspecial[tagnum]->impfile = 0;
  G__struct.rootspecial[tagnum]->defline = 0;
  G__struct.rootspecial[tagnum]->impline = 0;
  G__struct.rootspecial[tagnum]->version = 0;
  G__struct.rootspecial[tagnum]->instancecount = 0;
  G__struct.rootspecial[tagnum]->heapinstancecount = 0;
  G__struct.rootspecial[tagnum]->defaultconstructor =
    (void*) GetInterfaceMethod(G__struct.name[tagnum], "", &offset);
}

///////////////////////////////////////////////////////////////////////////
static long G__ClassInfo_MemberFunctionProperty(long& property, int tagnum)
{
  Reflex::Scope memscope = G__Dict::GetDict().GetScope(tagnum);

  for ( Reflex::Member_Iterator it = memscope.FunctionMember_Begin(); it != memscope.FunctionMember_End(); ++it ) {

    if ( it->IsConstructor() ) {
	property |= G__CLS_HASEXPLICITCTOR;
      if ( it->TypeOf().FunctionParameterSize() == 0 ) property |= G__CLS_HASDEFAULTCTOR;
      if ( it->TypeOf().IsVirtual() ) property |= G__CLS_HASIMPLICITCTOR;
    }
    if ( it->IsDestructor() ) property |= G__CLS_HASEXPLICITDTOR;
    if ( it->Name() == "operator=" ) property |= G__CLS_HASASSIGNOPR;
    if ( it->IsVirtual() ) property |= G__CLS_HASVIRTUAL;

  }

  return property;
}

///////////////////////////////////////////////////////////////////////////
static long G__ClassInfo_BaseClassProperty(long& property, G__ClassInfo& classinfo)
{
  G__BaseClassInfo baseinfo(classinfo);
  while (baseinfo.Next()) {
    long baseprop = baseinfo.ClassProperty();
    if (((property & G__CLS_HASEXPLICITCTOR) == 0) && (baseprop & G__CLS_HASCTOR)) {
      property |= (G__CLS_HASIMPLICITCTOR | G__CLS_HASDEFAULTCTOR);
    }
    if (((property & G__CLS_HASEXPLICITDTOR) == 0) && (baseprop & G__CLS_HASDTOR)) {
      property |= G__CLS_HASIMPLICITDTOR;
    }
    if (baseprop & G__CLS_HASVIRTUAL) {
       property |= G__CLS_HASVIRTUAL;
    }
  }
  return property;
}

///////////////////////////////////////////////////////////////////////////
static long G__ClassInfo_DataMemberProperty(long& property, int tagnum)
{
  ::Reflex::Scope var = G__Dict::GetDict().GetScope(tagnum);
  if (var) {
    for ( ::Reflex::Member_Iterator it = var.DataMember_Begin(); it != var.DataMember_End(); ++it ) {
      ::Reflex::Type memType = it->TypeOf();
      if ( memType.RawType().IsClass() && ! ( memType.IsPointer() || memType.IsReference() )) {
        G__ClassInfo classinfo(::Cint::Internal::G__get_tagnum(memType));
        long baseprop = classinfo.ClassProperty();
        if(0==(property&G__CLS_HASEXPLICITCTOR) && (baseprop&G__CLS_HASCTOR))
          property |= (G__CLS_HASIMPLICITCTOR|G__CLS_HASDEFAULTCTOR);
        if(0==(property&G__CLS_HASEXPLICITDTOR) && (baseprop&G__CLS_HASDTOR))
          property |= G__CLS_HASIMPLICITDTOR;
      }
    }
  }
  return property;
}

///////////////////////////////////////////////////////////////////////////
long Cint::G__ClassInfo::ClassProperty()
{
   long property = 0L;
   if (IsValid()) {
     switch (G__struct.type[tagnum]) {
        case 'e':
        case 'u':
           return property;
        case 'c':
        case 's':
           property |= G__CLS_VALID;
     }
     if (G__struct.isabstract[tagnum]) {
        property |= G__CLS_ISABSTRACT;
     }
     G__ClassInfo_MemberFunctionProperty(property, (int) tagnum);
     G__ClassInfo_BaseClassProperty(property, *this);
     G__ClassInfo_DataMemberProperty(property, (int) tagnum);
     return property;
   }
   return 0L;
}

///////////////////////////////////////////////////////////////////////////
Cint::G__MethodInfo Cint::G__ClassInfo::AddMethod(
  const char* typenam, const char* fname,
  const char* arg, int isstatic, int isvirtual,
  void *methodAddress)
{
  Reflex::Member ifunc;
  Reflex::Scope memscope;
  long index;

  if(-1==tagnum) memscope = Reflex::Scope::GlobalScope();
  else           memscope = G__Dict::GetDict().GetScope(tagnum);


#if 0
  //////////////////////////////////////////////////
  // Add a method entry
  while(ifunc->next) ifunc=ifunc->next;
  index = ifunc->allifunc;
  if(ifunc->allifunc==G__MAXIFUNC) {
     ifunc->next=new G__ifunc_table;
    ifunc->next->allifunc=0;
    ifunc->next->next=(struct G__ifunc_table *)NULL;
    ifunc->next->page = ifunc->page+1;
    ifunc->next->tagnum = ifunc->tagnum;
    ifunc = ifunc->next;
    for(int ix=0;ix<G__MAXIFUNC;ix++) {
      ifunc->funcname[ix] = (char*)NULL;
      ifunc->userparam[ix] = 0;
    }
    index=0;
  }

  //////////////////////////////////////////////////
  // save function name
  G__savestring(&ifunc->funcname[index],(char*)fname);
  int tmp;
  G__hash(ifunc->funcname[index],ifunc->hash[index],tmp);
  ifunc->para_name[index][0]=(char*)NULL;

  //////////////////////////////////////////////////
  // save type information
  G__TypeInfo type(typenam);
  ifunc->type[index] =   type.Type();
  ifunc->p_typetable[index] =   type.ReflexType();
  ifunc->p_tagtable[index] = (short)type.Tagnum();
  ifunc->reftype[index] =   type.Reftype();
  ifunc->isconst[index] =   type.Isconst();

  // flags
  ifunc->staticalloc[index] = isstatic;
  ifunc->isvirtual[index] = isvirtual;
  ifunc->ispurevirtual[index] = 0;
  ifunc->access[index] = G__PUBLIC;

  // miscellaneous flags
  ifunc->isexplicit[index] = 0;
  ifunc->iscpp[index] = 1;
  ifunc->ansi[index] = 1;
  ifunc->busy[index] = 0;
  ifunc->friendtag[index] = (struct G__friendtag*)NULL;
  ifunc->globalcomp[index] = G__NOLINK;
  ifunc->comment[index].p.com = (char*)NULL;
  ifunc->comment[index].filenum = -1;


  ifunc->userparam[index] = (void*)NULL;
  ifunc->vtblindex[index] = -1;
  ifunc->vtblbasetagnum[index] = -1;

  //////////////////////////////////////////////////
  // set argument infomation
  if (strcmp(arg,"ellipsis")==0) {
     // mimick ellipsis
     ifunc->para_nu[index] = -1;
     ifunc->ansi[index] = 0;
  } else {
     char *argtype = (char*)arg;
     struct G__param para;
     G__argtype2param(argtype,&para);

     ifunc->para_nu[index] = para.paran;
     for(int i=0;i<para.paran;i++) {
        ifunc->para_type[index][i] = para.para[i].type;
        if(para.para[i].type!='d' && para.para[i].type!='f')
           ifunc->para_reftype[index][i] = para.para[i].obj.reftype.reftype;
        else
           ifunc->para_reftype[index][i] = G__PARANORMAL;
        ifunc->para_p_tagtable[index][i] = para.para[i].tagnum;
        ifunc->para_p_typetable[index][i] = G__value_typenum(para.para[i]);
        ifunc->para_name[index][i] = (char*)malloc(10);
        sprintf(ifunc->para_name[index][i],"G__p%d",i);
        ifunc->para_default[index][i] = (G__value*)NULL;
        ifunc->para_def[index][i] = (char*)NULL;
     }
  }

  //////////////////////////////////////////////////
  ifunc->pentry[index] = &ifunc->entry[index];
  //ifunc->entry[index].pos
 if (methodAddress) {
    ifunc->entry[index].p = methodAddress;
    ifunc->entry[index].line_number = -1;
    ifunc->entry[index].filenum = -1;
    ifunc->entry[index].size = -1;
    ifunc->entry[index].tp2f = (char*)NULL;
    ifunc->entry[index].bytecode = 0;
    ifunc->entry[index].bytecodestatus = G__BYTECODE_NOTYET;
 } else {
    ifunc->entry[index].p = G__srcfile[G__struct.filenum[tagnum]].fp;
    ifunc->entry[index].line_number=(-1==tagnum)?0:G__struct.line_number[tagnum];
    ifunc->entry[index].filenum=(-1==tagnum)?0:G__struct.filenum[tagnum];
    ifunc->entry[index].size = 1;
    ifunc->entry[index].tp2f = (char*)NULL;
    ifunc->entry[index].bytecode = 0;
    ifunc->entry[index].bytecodestatus = G__BYTECODE_NOTYET;
 }

  //////////////////////////////////////////////////
  ++ifunc->allifunc;

  /* Initialize method object */
  G__MethodInfo method;
<<<<<<< Class.cxx
  method.Init((long)ifunc,index,this);
  return(method);

#endif

  return G__MethodInfo(ifunc);
}

///////////////////////////////////////////////////////////////////////////
unsigned char Cint::G__ClassInfo::FuncFlag()
{
   return IsValid() ? G__struct.funcs[tagnum] : 0;
}

///////////////////////////////////////////////////////////////////////////
