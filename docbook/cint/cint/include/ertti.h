/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/************************************************************************
* Extensive Run Time Type Identification class
*
* ERTTI is an original extention. This interpreted header file provides
* example program using ERTTI classes.
*
************************************************************************/
#ifndef G__ERTTI_H
#define G__ERTTI_H

#ifdef __CINT__
#include <typeinfo.h>
#endif

// setup ERTTI interface method
#ifndef G__API_H
#pragma setertti
#endif

using namespace Cint;

/************************************************************************
* translation between ANSI/ISO RTTI type_info and ERTTI G__ClassInfo
************************************************************************/
type_info::type_info(G__ClassInfo& ci) {
  type = 'u';
  tagnum=ci.Tagnum();
  typenum = -1;
  reftype = 0;
  size = ci.Size();
}

type_info::type_info(G__TypeInfo& ti) {
  type = ti.Type();
  tagnum=ti.Tagnum();
  typenum=ti.Typenum();
  long property=ti.Property();
  if(property&G__BIT_ISREFERENCE) reftype=1;
  else reftype=0;
  size = ti.Size();
  return(*this);
}

int type_info::Tagnum() { return(tagnum); }

G__ClassInfo::G__ClassInfo(type_info& ti) {
  Init(ti.Tagnum());
}

void G__ClassInfo::Init(type_info& ti) {
  Init(ti.Tagnum());
}

G__TypeInfo::G__TypeInfo(type_info& ti) {
  Init(ti.Tagnum());
}

void G__TypeInfo::Init(type_info& ti) {
  Init(ti.Tagnum());
  if(-1==Tagnum()) {
    Init(ti.name());
  }
}


/************************************************************************
* Extensive G__ERTTI class
************************************************************************/
class G__ERTTI {
 public:
  static void inheritance(char *classname);
  static void class(char *classname=NULL);
  static void function(char *classname=NULL);
  static void datamember(char *classname=NULL);
  static void typedef(char *typedefname=NULL);
 private:
  static void printenclosing(G__ClassInfo& classinfo);
};

/************************************************************************
* list up typedefs
************************************************************************/
void G__ERTTI::typedef(char *typedefname)
{
  if(typedefname) {
    G__TypedefInfo typedefinfo(typedefname);
    printf("typedef %s %s;\n",typedefinfo.TrueName(),typedefinfo.Name());
  }
  else {
    G__TypedefInfo typedefinfo;
    while(typedefinfo.Next()) {
      printf("typedef %s %s;\n",typedefinfo.TrueName(),typedefinfo.Name());
    }
  }
}

/************************************************************************
* list up classes
************************************************************************/
void G__ERTTI::class(char *classname)
{
  if(classname) {
    printf("CLASS INHERITANCE-----------------------------------------\n");
    inheritance(classname);
    printf("DATA MEMBER-----------------------------------------------\n");
    datamember(classname);
    printf("MEMBER FUNCTION-------------------------------------------\n");
    function(classname);
  }
  else {
    G__ClassInfo classinfo;
    while(classinfo.Next()) {
      inheritance(classinfo.Name());
    }
  }
}

/************************************************************************
* list up inheritance
************************************************************************/
void G__ERTTI::inheritance(char *classname)
{
  G__ClassInfo classinfo(classname);
  if(!classinfo.IsValid()) {
    fprintf(stderr,"class %s not found\n",classname);
    return;
  }

  printf("FILE:%-15s LINE:%-4d ",classinfo.FileName(),classinfo.LineNumber());

  if(classinfo.Property()&G__BIT_ISCLASS) printf("class ");
  if(classinfo.Property()&G__BIT_ISSTRUCT) printf("struct ");
  if(classinfo.Property()&G__BIT_ISUNION) printf("union ");
  if(classinfo.Property()&G__BIT_ISENUM) printf("enum ");
  printenclosing(classinfo);
  printf("%s",classinfo.Name());
  
  G__BaseClassInfo baseinfo(classinfo);
  while(baseinfo.Next()) {
    if(baseinfo.Property()&G__BIT_ISPUBLIC) printf(" public:");
    if(baseinfo.Property()&G__BIT_ISPROTECTED) printf(" protected:");
    if(baseinfo.Property()&G__BIT_ISPRIVATE) printf(" private:");
    printenclosing(baseinfo);
    printf("%s",baseinfo.Name());
  }
  printf(";\n");
}

/************************************************************************
* print enclosing classs
************************************************************************/
void G__ERTTI::printenclosing(G__ClassInfo& classinfo)
{
  G__ClassInfo enclosing;
  enclosing=classinfo.EnclosingClass();
  if(enclosing.IsValid()) {
    printenclosing(enclosing);
    printf("%s::",enclosing.Name());
  }
}

/************************************************************************
* list up functions
************************************************************************/
void G__ERTTI::function(char *classname)
{
  if(NULL==classname) {
    // global function information
    G__MethodInfo funcinfo;
  }
  else {
    // member function information
    G__ClassInfo classinfo(classname);
    G__MethodInfo funcinfo(classinfo);
  }

  G__MethodArgInfo arginfo;
  G__TypeInfo *typeinfo;
  int flag;
  char *argname;
  char *defaultvalue;

  while(funcinfo.Next()) {
    flag=0;
    
    printf("FILE:%-15s LINE:%-4d ",funcinfo.FileName(),funcinfo.LineNumber());
    // print property 
    if(funcinfo.Property()&G__BIT_ISPUBLIC) printf("public: ");
    if(funcinfo.Property()&G__BIT_ISPROTECTED) printf("protected: ");
    if(funcinfo.Property()&G__BIT_ISPRIVATE) printf("private: ");
    if(funcinfo.Property()&G__BIT_ISVIRTUAL) printf("virtual ");
    if(funcinfo.Property()&G__BIT_ISSTATIC) printf("static ");
    //if(funcinfo.Property()&G__BIT_ISCONSTANT) printf("const ");
    // type and name
    typeinfo=funcinfo.Type();
    printf("%s %s(",typeinfo->Name(),funcinfo.Name());
    // argument
    arginfo.Init(funcinfo);
    if(0==funcinfo.NArg()) printf("void");
    while(arginfo.Next()) {
      if(flag) printf(",");
      else flag=1;
      // Type of argument
      typeinfo=arginfo.Type();
      printf("%s",typeinfo->Name());
      // Name of argument
      argname=arginfo.Name();
      if(argname) printf(" %s",argname);
      // default parameter
      defaultvalue=arginfo.DefaultValue();
      if(defaultvalue) printf("=%s",defaultvalue);
    }
    printf(")");
    // property
    if(funcinfo.Property()&G__BIT_ISPUREVIRTUAL) printf("=0");
    printf(";\n");
  }
}

/************************************************************************
* list up data members
************************************************************************/
void G__ERTTI::datamember(char *classname)
{
  if(NULL==classname) {
    // global data information
    G__DataMemberInfo datainfo;
  }
  else {
    // data member information
    G__ClassInfo classinfo(classname);
    G__DataMemberInfo datainfo(classinfo);
  }

  G__TypeInfo *typeinfo;
  int i;
  char *filename;

  while(datainfo.Next()) {
    filename = datainfo.FileName();
    if(filename) printf("%-14s",filename);
    else         printf("%-14s","");
    // print property 
    if(datainfo.Property()&G__BIT_ISPUBLIC) printf("public: ");
    if(datainfo.Property()&G__BIT_ISPROTECTED) printf("protected: ");
    if(datainfo.Property()&G__BIT_ISPRIVATE) printf("private: ");
    if(datainfo.Property()&G__BIT_ISSTATIC) printf("static ");
    //if(datainfo.Property()&G__BIT_ISCONSTANT) printf("const ");
    // type and name
    typeinfo=datainfo.Type();
    printf("%s %s",typeinfo->Name(),datainfo.Name());
    // array dimention
    for(i=0;i<datainfo.ArrayDim();i++) printf("[%d]",datainfo.MaxIndex(i));
    printf(";\n");
  }
}

#endif
