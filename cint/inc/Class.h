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
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/


#ifndef G__CLASSINFO_H
#define G__CLASSINFO_H 


#include "Api.h"
class G__MethodInfo;
class G__DataMemberInfo;
#ifndef G__OLDIMPLEMENTATION1020
class G__FriendInfo;
#endif

/*********************************************************************
* class G__ClassInfo
*
* 
*********************************************************************/
class 
#ifndef __CINT__
G__EXPORT
#endif
G__ClassInfo {
 public:
  virtual ~G__ClassInfo() {}
  G__ClassInfo() { Init(); }
  void Init();
  G__ClassInfo(const char *classname) { Init(classname); } 
  void Init(const char *classname);
  G__ClassInfo(int tagnumin) { Init(tagnumin); } 
  void Init(int tagnumin);

  int operator==(const G__ClassInfo& a);
  int operator!=(const G__ClassInfo& a);

  const char *Name() ;
  const char *Fullname();
  const char *Title() ;
  int Size() ; 
  long Property();
  int NDataMembers();
  int NMethods();
  long IsBase(const char *classname);
  long IsBase(G__ClassInfo& a);
  long Tagnum() const { return(tagnum); }
  G__ClassInfo EnclosingClass() ;
  G__ClassInfo EnclosingSpace() ;
#ifndef G__OLDIMPLEMENTATION1020
  struct G__friendtag* GetFriendInfo(); 
#endif
  void SetGlobalcomp(int globalcomp);
#ifndef G__OLDIMPLEMENTATION1334
  void SetProtectedAccess(int protectedaccess);
#endif
#ifdef G__OLDIMPLEMENTATION1218_YET
  int IsValid() { return 0<=tagnum && tagnum<G__struct.alltag ? 1 : 0; }
#else
  int IsValid();
#endif
  int IsLoaded();
  int SetFilePos(const char *fname);
  int Next();
  int Linkage();

  const char *FileName() ;
  int LineNumber() ;

  int IsTmplt();
  const char* TmpltName();
  const char* TmpltArg();
  

 protected:
  long tagnum;  // class,struct,union,enum key for cint dictionary
#ifndef G__OLDIMPLEMENTATION1218
  long class_property;  // cache value (expensive to get)
#endif

#ifdef G__ROOTSPECIAL
 ///////////////////////////////////////////////////////////////
 // Following things have to be added for ROOT
 ///////////////////////////////////////////////////////////////
 public:
  void SetDefFile(char *deffilein);
  void SetDefLine(int deflinein);
  void SetImpFile(char *impfilein);
  void SetImpLine(int implinein);
  void SetVersion(int versionin);
  const char *DefFile();
  int DefLine();
  const char *ImpFile();
  int ImpLine();
  int Version();
  void *New();
  void *New(int n);
  void *New(void *arena);
#ifndef G__OLDIMPLEMENTATION2043
  void Delete(void* p) const ;
  void Destruct(void* p) const ;
#endif
  int InstanceCount(); 
  void ResetInstanceCount();
  void IncInstanceCount();
  int HeapInstanceCount();
  void IncHeapInstanceCount();
  void ResetHeapInstanceCount();
  int RootFlag();
  //void SetDefaultConstructor(void* p2f);
  enum MatchMode { ExactMatch=0, ConversionMatch=1, ConversionMatchBytecode=2};
  enum InheritanceMode { InThisScope=0, WithInheritance=1 };
  G__InterfaceMethod GetInterfaceMethod(const char *fname,const char *arg
					,long* poffset
					,MatchMode mode=ConversionMatch
                                        ,InheritanceMode imode=WithInheritance
					);
  G__MethodInfo GetMethod(const char *fname,const char *arg,long* poffset
			  ,MatchMode mode=ConversionMatch
                          ,InheritanceMode imode=WithInheritance
                          );
  G__MethodInfo GetMethod(const char *fname,struct G__param* libp,long* poffset
			  ,MatchMode mode=ConversionMatch
                          ,InheritanceMode imode=WithInheritance
                          );
#ifndef G__OLDIMPLEMENTATION2059
  G__MethodInfo GetDefaultConstructor();
  G__MethodInfo GetCopyConstructor();
  G__MethodInfo GetDestructor();
  G__MethodInfo GetAssignOperator();
#endif
#ifndef G__OLDIMPLEMENTATION2076
  G__MethodInfo AddMethod(const char* typenam,const char* fname,const char *arg
                         ,int isstatic=0,int isvirtual=0);
#endif
  G__DataMemberInfo GetDataMember(const char *name,long* poffset);
  int HasMethod(const char *fname);
  int HasDataMember(const char *name);
  int HasDefaultConstructor();

 private:
  void CheckValidRootInfo();
#endif /* ROOTSPECIAL */


#ifndef G__OLDIMPLEMENTATION644
 public:
  long ClassProperty();
#endif
#ifndef G__OLDIMPLEMENTATION2118
  unsigned char FuncFlag(); 
#endif

};


#ifndef G__OLDIMPLEMENTATION1020
/*********************************************************************
* class G__FriendInfo
*
* 
*********************************************************************/
class 
#ifndef __CINT__
G__EXPORT
#endif
G__FriendInfo {
 public:
  G__FriendInfo(struct G__friendtag *pin=0) { Init(pin); }
  void operator=(const G__FriendInfo& x) { Init(x.pfriendtag); }
  void Init(struct G__friendtag* pin) {
    pfriendtag = pin;
    if(pfriendtag) cls.Init(pfriendtag->tagnum); 
    else           cls.Init(-1);
  }
  G__ClassInfo* FriendOf() { return(&cls); }
  int Next() { 
    if(pfriendtag) {
      pfriendtag=pfriendtag->next; 
      Init(pfriendtag);
      return(IsValid());
    }
    else {
      return(0);
    }
  }
  int IsValid() { if(pfriendtag) return(1); else return(0); }
 private:
  G__friendtag *pfriendtag;
  G__ClassInfo cls;
};

#endif

#endif
