// @(#)root/mac:$Name$:$Id$
// Author: Fons Rademakers   14/08/96
/********************************************************************
* G__MacSystem.h
********************************************************************/
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
extern "C" {
#define G__ANSIHEADER
#include "G__ci.h"
extern void G__cpp_setup_tagtableG__MacSystem();
extern void G__cpp_setup_inheritanceG__MacSystem();
extern void G__cpp_setup_typetableG__MacSystem();
extern void G__cpp_setup_memvarG__MacSystem();
extern void G__cpp_setup_globalG__MacSystem();
extern void G__cpp_setup_memfuncG__MacSystem();
extern void G__cpp_setup_funcG__MacSystem();
extern void G__set_cpp_environmentG__MacSystem();
}


#include "TROOT.h"
#include "TMemberInspector.h"
#include "TMacSystem.h"

#ifndef G__MEMFUNCBODY
extern "C" int G__TMacSystem_TMacSystem_2_0(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_Init_4_0(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_Hostname_5_0(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_Exit_6_0(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_Abort_7_0(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_MakeDirectory_8_0(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_OpenDirectory_9_0(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_FreeDirectory_0_1(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_GetDirEntry_1_1(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_ChangeDirectory_2_1(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_WorkingDirectory_3_1(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_BaseName_4_1(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_DirName_5_1(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_ConcatFileName_6_1(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_IsAbsoluteFileName_7_1(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_AccessPathName_8_1(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_Unlink_9_1(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_GetPathInfo_0_2(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_UnixPathName_1_2(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_Setenv_2_2(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_Getenv_3_2(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_DeclFileName_4_2(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_DeclFileLine_5_2(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_ImplFileName_6_2(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_ImplFileLine_7_2(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_Class_Version_8_2(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_IsA_9_2(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_ShowMembers_0_3(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_Dictionary_1_3(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_TMacSystem_2_3(G__value *result7,char *funcname,struct G__param *libp,int hash);
extern "C" int G__TMacSystem_wATMacSystem_3_3(G__value *result7,char *funcname,struct G__param *libp,int hash);
#endif

extern G__linked_taginfo G__G__MacSystemLN_TClass;
extern G__linked_taginfo G__G__MacSystemLN_TObject;
extern G__linked_taginfo G__G__MacSystemLN_TString;
extern G__linked_taginfo G__G__MacSystemLN_TNamed;
extern G__linked_taginfo G__G__MacSystemLN_TSystem;
extern G__linked_taginfo G__G__MacSystemLN_TMacSystem;
/********************************************************
* G__MacSystem.cc
********************************************************/

#ifdef G__MEMTEST
#undef malloc
#endif


extern "C" void G__set_cpp_environmentG__MacSystem() {
  G__add_compiledheader("TROOT.h");
  G__add_compiledheader("TMemberInspector.h");
  G__add_compiledheader("TMacSystem.h");
}
int G__cpp_dllrevG__MacSystem() { return(50911); }

/*********************************************************
* Member function Interface Method
*********************************************************/

/* TMacSystem */
extern "C" int G__TMacSystem_TMacSystem_2_0(G__value *result7,char *funcname,struct G__param *libp,int hash) {
   TMacSystem *p;
   if(G__getaryconstruct()) p=new TMacSystem[G__getaryconstruct()];
   else                    p=new TMacSystem;
      result7->obj.i = (long)p;
      result7->ref = (long)p;
      result7->type = 'u';
      result7->tagnum = G__get_linked_tagnum(&G__G__MacSystemLN_TMacSystem);
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_Init_4_0(G__value *result7,char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,98,(long)((TMacSystem*)(G__getstructoffset()))->Init());
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_Hostname_5_0(G__value *result7,char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,67,(long)((TMacSystem*)(G__getstructoffset()))->Hostname());
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_Exit_6_0(G__value *result7,char *funcname,struct G__param *libp,int hash) {
   switch(libp->paran) {
   case 2:
      G__setnull(result7);
      ((TMacSystem*)(G__getstructoffset()))->Exit((int)G__int(libp->para[0]),(Bool_t)G__int(libp->para[1]));
      break;
   case 1:
      G__setnull(result7);
      ((TMacSystem*)(G__getstructoffset()))->Exit((int)G__int(libp->para[0]));
      break;
   }
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_Abort_7_0(G__value *result7,char *funcname,struct G__param *libp,int hash) {
   switch(libp->paran) {
   case 1:
      G__setnull(result7);
      ((TMacSystem*)(G__getstructoffset()))->Abort((int)G__int(libp->para[0]));
      break;
   case 0:
      G__setnull(result7);
      ((TMacSystem*)(G__getstructoffset()))->Abort();
      break;
   }
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_MakeDirectory_8_0(G__value *result7,char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((TMacSystem*)(G__getstructoffset()))->MakeDirectory((const char*)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_OpenDirectory_9_0(G__value *result7,char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,89,(long)((TMacSystem*)(G__getstructoffset()))->OpenDirectory((const char*)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_FreeDirectory_0_1(G__value *result7,char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      ((TMacSystem*)(G__getstructoffset()))->FreeDirectory((void*)G__int(libp->para[0]));
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_GetDirEntry_1_1(G__value *result7,char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,67,(long)((TMacSystem*)(G__getstructoffset()))->GetDirEntry((void*)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_ChangeDirectory_2_1(G__value *result7,char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,98,(long)((TMacSystem*)(G__getstructoffset()))->ChangeDirectory((const char*)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_WorkingDirectory_3_1(G__value *result7,char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,67,(long)((TMacSystem*)(G__getstructoffset()))->WorkingDirectory());
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_BaseName_4_1(G__value *result7,char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,67,(long)((TMacSystem*)(G__getstructoffset()))->BaseName((const char*)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_DirName_5_1(G__value *result7,char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,67,(long)((TMacSystem*)(G__getstructoffset()))->DirName((const char*)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_ConcatFileName_6_1(G__value *result7,char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,67,(long)((TMacSystem*)(G__getstructoffset()))->ConcatFileName((const char*)G__int(libp->para[0]),(const char*)G__int(libp->para[1])));
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_IsAbsoluteFileName_7_1(G__value *result7,char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,98,(long)((TMacSystem*)(G__getstructoffset()))->IsAbsoluteFileName((const char*)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_AccessPathName_8_1(G__value *result7,char *funcname,struct G__param *libp,int hash) {
   switch(libp->paran) {
   case 2:
      G__letint(result7,98,(long)((TMacSystem*)(G__getstructoffset()))->AccessPathName((const char*)G__int(libp->para[0]),(EAccessMode)G__int(libp->para[1])));
      break;
   case 1:
      G__letint(result7,98,(long)((TMacSystem*)(G__getstructoffset()))->AccessPathName((const char*)G__int(libp->para[0])));
      break;
   }
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_Unlink_9_1(G__value *result7,char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((TMacSystem*)(G__getstructoffset()))->Unlink((const char*)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_GetPathInfo_0_2(G__value *result7,char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((TMacSystem*)(G__getstructoffset()))->GetPathInfo((const char*)G__int(libp->para[0]),(ULong_t*)G__int(libp->para[1])
,(ULong_t*)G__int(libp->para[2]),(ULong_t*)G__int(libp->para[3])
,(ULong_t*)G__int(libp->para[4])));
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_UnixPathName_1_2(G__value *result7,char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,67,(long)((TMacSystem*)(G__getstructoffset()))->UnixPathName((const char*)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_Setenv_2_2(G__value *result7,char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      ((TMacSystem*)(G__getstructoffset()))->Setenv((const char*)G__int(libp->para[0]),(const char*)G__int(libp->para[1]));
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_Getenv_3_2(G__value *result7,char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,67,(long)((TMacSystem*)(G__getstructoffset()))->Getenv((const char*)G__int(libp->para[0])));
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_DeclFileName_4_2(G__value *result7,char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,67,(long)((TMacSystem*)(G__getstructoffset()))->DeclFileName());
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_DeclFileLine_5_2(G__value *result7,char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((TMacSystem*)(G__getstructoffset()))->DeclFileLine());
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_ImplFileName_6_2(G__value *result7,char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,67,(long)((TMacSystem*)(G__getstructoffset()))->ImplFileName());
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_ImplFileLine_7_2(G__value *result7,char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,105,(long)((TMacSystem*)(G__getstructoffset()))->ImplFileLine());
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_Class_Version_8_2(G__value *result7,char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,115,(long)((TMacSystem*)(G__getstructoffset()))->Class_Version());
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_IsA_9_2(G__value *result7,char *funcname,struct G__param *libp,int hash) {
      G__letint(result7,85,(long)((TMacSystem*)(G__getstructoffset()))->IsA());
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_ShowMembers_0_3(G__value *result7,char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      ((TMacSystem*)(G__getstructoffset()))->ShowMembers(*(TMemberInspector*)libp->para[0].ref,(char*)G__int(libp->para[1]));
   return(1 || funcname || hash || result7 || libp) ;
}

extern "C" int G__TMacSystem_Dictionary_1_3(G__value *result7,char *funcname,struct G__param *libp,int hash) {
      G__setnull(result7);
      ((TMacSystem*)(G__getstructoffset()))->Dictionary();
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic copy constructor
extern "C" int G__TMacSystem_TMacSystem_2_3(G__value *result7,char *funcname,struct G__param *libp,int hash)
{
   TMacSystem *p;
   if(1!=libp->paran) ;
   p=new TMacSystem(*(TMacSystem*)G__int(libp->para[0]));
   result7->obj.i = (long)p;
   result7->ref = (long)p;
   result7->type = 'u';
   result7->tagnum = G__get_linked_tagnum(&G__G__MacSystemLN_TMacSystem);
   return(1 || funcname || hash || result7 || libp) ;
}

// automatic destructor
extern "C" int G__TMacSystem_wATMacSystem_3_3(G__value *result7,char *funcname,struct G__param *libp,int hash) {
   if(G__getaryconstruct())
     if(G__PVOID==G__getgvp())
       delete[] (TMacSystem *)(G__getstructoffset());
     else
       for(int i=G__getaryconstruct()-1;i>=0;i--)
         delete (TMacSystem *)((G__getstructoffset())+sizeof(TMacSystem)*i);
   else  delete (TMacSystem *)(G__getstructoffset());
   return(1 || funcname || hash || result7 || libp) ;
}


/* Setting up global function */

/*********************************************************
* Member function Stub
*********************************************************/

/* Setting up global function stub */

/*********************************************************
* Get size of pointer to member function
*********************************************************/
class G__Sizep2memfuncG__MacSystem {
 public:
  G__Sizep2memfuncG__MacSystem() {p=&G__Sizep2memfuncG__MacSystem::sizep2memfunc;}
    size_t sizep2memfunc() { return(sizeof(p)); }
  private:
    size_t (G__Sizep2memfuncG__MacSystem::*p)();
};

size_t G__get_sizep2memfuncG__MacSystem()
{
  G__Sizep2memfuncG__MacSystem a;
  G__setsizep2memfunc((int)a.sizep2memfunc());
  return((size_t)a.sizep2memfunc());
}


/*********************************************************
* Inheritance information setup/
*********************************************************/
extern "C" void G__cpp_setup_inheritanceG__MacSystem() {

   /* Setting up class inheritance */
   if(0==G__getnumbaseclass(G__get_linked_tagnum(&G__G__MacSystemLN_TMacSystem))) {
     TMacSystem *derived;
     derived=(TMacSystem*)0x1000;
     {
       TSystem *pbase=(TSystem*)derived;
       G__inheritance_setup(G__get_linked_tagnum(&G__G__MacSystemLN_TMacSystem),G__get_linked_tagnum(&G__G__MacSystemLN_TSystem),(long)pbase-(long)derived,1,1);
     }
     {
       TNamed *pbase=(TNamed*)derived;
       G__inheritance_setup(G__get_linked_tagnum(&G__G__MacSystemLN_TMacSystem),G__get_linked_tagnum(&G__G__MacSystemLN_TNamed),(long)pbase-(long)derived,1,0);
     }
     {
       TObject *pbase=(TObject*)derived;
       G__inheritance_setup(G__get_linked_tagnum(&G__G__MacSystemLN_TMacSystem),G__get_linked_tagnum(&G__G__MacSystemLN_TObject),(long)pbase-(long)derived,1,0);
     }
   }
}

/*********************************************************
* typedef information setup/
*********************************************************/
extern "C" void G__cpp_setup_typetableG__MacSystem() {

   /* Setting up typedef entry */
   G__search_typename("Char_t",99,-1,0);
   G__setnewtype(-1,"Signed Character 1 byte",0);
   G__search_typename("UChar_t",98,-1,0);
   G__setnewtype(-1,"Unsigned Character 1 byte",0);
   G__search_typename("Short_t",115,-1,0);
   G__setnewtype(-1,"Signed Short integer 2 bytes",0);
   G__search_typename("UShort_t",114,-1,0);
   G__setnewtype(-1,"Unsigned Short integer 2 bytes",0);
   G__search_typename("Int_t",105,-1,0);
   G__setnewtype(-1,"Signed integer 4 bytes",0);
   G__search_typename("UInt_t",104,-1,0);
   G__setnewtype(-1,"Unsigned integer 4 bytes",0);
   G__search_typename("Seek_t",105,-1,0);
   G__setnewtype(-1,"File pointer",0);
   G__search_typename("Long_t",108,-1,0);
   G__setnewtype(-1,"Signed long integer 8 bytes",0);
   G__search_typename("ULong_t",107,-1,0);
   G__setnewtype(-1,"Unsigned long integer 8 bytes",0);
   G__search_typename("Float_t",102,-1,0);
   G__setnewtype(-1,"Float 4 bytes",0);
   G__search_typename("Double_t",100,-1,0);
   G__setnewtype(-1,"Float 8 bytes",0);
   G__search_typename("Text_t",99,-1,0);
   G__setnewtype(-1,"General string",0);
   G__search_typename("Bool_t",98,-1,0);
   G__setnewtype(-1,"Boolean (0=false, 1=true)",0);
   G__search_typename("Byte_t",98,-1,0);
   G__setnewtype(-1,"Byte (8 bits)",0);
   G__search_typename("Version_t",115,-1,0);
   G__setnewtype(-1,"Class version identifier",0);
   G__search_typename("Option_t",99,-1,0);
   G__setnewtype(-1,"Option string",0);
   G__search_typename("Ssiz_t",105,-1,0);
   G__setnewtype(-1,"String size",0);
   G__search_typename("VoidFuncPtr_t",89,-1,0);
   G__setnewtype(-1,"pointer to void function",0);
   G__search_typename("FreeHookFun_t",89,-1,0);
   G__setnewtype(-1,NULL,0);
   G__search_typename("Axis_t",102,-1,0);
   G__setnewtype(-1,"Axis values type",0);
   G__search_typename("Stat_t",100,-1,0);
   G__setnewtype(-1,"Statistics type",0);
}

/*********************************************************
* Data Member information setup/
*********************************************************/

   /* Setting up class,struct,union tag member variable */

   /* TMacSystem */
static void G__setup_memvarTMacSystem(void) {
   G__tag_memvar_setup(G__get_linked_tagnum(&G__G__MacSystemLN_TMacSystem));
   { TMacSystem *p; p=(TMacSystem*)0x1000;
   G__memvar_setup((void*)NULL,117,0,0,G__get_linked_tagnum(&G__G__MacSystemLN_TString),-1,-1,2,"fHostname=",0,"Hostname");
   G__memvar_setup((void*)NULL,98,0,0,-1,G__defined_typename("Bool_t"),-1,2,"fHasWaitNextEvent=",0,"Set in ctor, only used in TMacApplication::Run()");
   G__memvar_setup((void*)NULL,117,0,0,G__get_linked_tagnum(&G__G__MacSystemLN_TString),-1,-1,2,"fWdPath=",0,"Current working directory");
   G__memvar_setup((void*)(&TMacSystem::fgDebug),98,0,0,-1,G__defined_typename("Bool_t"),-2,1,"fgDebug=",0,(char*)NULL);
   G__memvar_setup((void*)(&TMacSystem::fgIsA),85,0,0,G__get_linked_tagnum(&G__G__MacSystemLN_TClass),-1,-2,1,"fgIsA=",0,(char*)NULL);
   }
   G__tag_memvar_reset();
}

extern "C" void G__cpp_setup_memvarG__MacSystem() {
}
/***********************************************************
************************************************************
************************************************************
************************************************************
************************************************************
************************************************************
************************************************************
***********************************************************/

/*********************************************************
* Member function information setup for each class
*********************************************************/
static void G__setup_memfuncTMacSystem(void) {
   /* TMacSystem */
   G__tag_memfunc_setup(G__get_linked_tagnum(&G__G__MacSystemLN_TMacSystem));
   G__memfunc_setup("SetupMenu",934,(G__InterfaceMethod)NULL,121,-1,-1,0,0,1,2,0,"",(char*)NULL);
   G__memfunc_setup("VeryFirstInit",1346,(G__InterfaceMethod)NULL,121,-1,-1,0,0,1,2,0,"",(char*)NULL);
   G__memfunc_setup("TMacSystem",1002,G__TMacSystem_TMacSystem_2_0,105,G__get_linked_tagnum(&G__G__MacSystemLN_TMacSystem),-1,0,0,1,1,0,"",(char*)NULL);
   G__memfunc_setup("Init",404,G__TMacSystem_Init_4_0,98,-1,G__defined_typename("Bool_t"),0,0,1,1,0,"",(char*)NULL);
   G__memfunc_setup("Hostname",831,G__TMacSystem_Hostname_5_0,67,-1,-1,0,0,1,1,1,"",(char*)NULL);
   G__memfunc_setup("Exit",410,G__TMacSystem_Exit_6_0,121,-1,-1,0,2,1,1,0,
"i - - 0 - code b - 'Bool_t' 0 kTRUE mode",(char*)NULL);
   G__memfunc_setup("Abort",504,G__TMacSystem_Abort_7_0,121,-1,-1,0,1,1,1,0,"i - - 0 0 code",(char*)NULL);
   G__memfunc_setup("MakeDirectory",1331,G__TMacSystem_MakeDirectory_8_0,105,-1,-1,0,1,1,1,0,"C - - 0 - name",(char*)NULL);
   G__memfunc_setup("OpenDirectory",1351,G__TMacSystem_OpenDirectory_9_0,89,-1,-1,0,1,1,1,0,"C - - 0 - name",(char*)NULL);
   G__memfunc_setup("FreeDirectory",1335,G__TMacSystem_FreeDirectory_0_1,121,-1,-1,0,1,1,1,0,"Y - - 0 - dirp",(char*)NULL);
   G__memfunc_setup("GetDirEntry",1105,G__TMacSystem_GetDirEntry_1_1,67,-1,-1,0,1,1,1,1,"Y - - 0 - dirp",(char*)NULL);
   G__memfunc_setup("ChangeDirectory",1531,G__TMacSystem_ChangeDirectory_2_1,98,-1,G__defined_typename("Bool_t"),0,1,1,1,0,"C - - 0 - path",(char*)NULL);
   G__memfunc_setup("WorkingDirectory",1686,G__TMacSystem_WorkingDirectory_3_1,67,-1,-1,0,0,1,1,1,"",(char*)NULL);
   G__memfunc_setup("BaseName",764,G__TMacSystem_BaseName_4_1,67,-1,-1,0,1,1,1,1,"C - - 0 - pathname",(char*)NULL);
   G__memfunc_setup("DirName",672,G__TMacSystem_DirName_5_1,67,-1,-1,0,1,1,1,1,"C - - 0 - pathname",(char*)NULL);
   G__memfunc_setup("ConcatFileName",1369,G__TMacSystem_ConcatFileName_6_1,67,-1,-1,0,2,1,1,1,
"C - - 0 - dir C - - 0 - name",(char*)NULL);
   G__memfunc_setup("IsAbsoluteFileName",1788,G__TMacSystem_IsAbsoluteFileName_7_1,98,-1,G__defined_typename("Bool_t"),0,1,1,1,0,"C - - 0 - dir",(char*)NULL);
   G__memfunc_setup("AccessPathName",1376,G__TMacSystem_AccessPathName_8_1,98,-1,G__defined_typename("Bool_t"),0,2,1,1,0,
"C - - 0 - path i EAccessMode - 0 kFileExists mode",(char*)NULL);
   G__memfunc_setup("Unlink",625,G__TMacSystem_Unlink_9_1,105,-1,-1,0,1,1,1,0,"C - - 0 - name",(char*)NULL);
   G__memfunc_setup("GetPathInfo",1081,G__TMacSystem_GetPathInfo_0_2,105,-1,-1,0,5,1,1,0,
"C - - 0 - path K - 'ULong_t' 0 - id "
"K - 'ULong_t' 0 - size K - 'ULong_t' 0 - flags "
"K - 'ULong_t' 0 - modtime",(char*)NULL);
   G__memfunc_setup("UnixPathName",1202,G__TMacSystem_UnixPathName_1_2,67,-1,-1,0,1,1,1,1,"C - - 0 - unixpathname",(char*)NULL);
   G__memfunc_setup("Setenv",629,G__TMacSystem_Setenv_2_2,121,-1,-1,0,2,1,1,0,
"C - - 0 - name C - - 0 - value","set environment variable name to value");
   G__memfunc_setup("Getenv",617,G__TMacSystem_Getenv_3_2,67,-1,-1,0,1,1,1,1,"C - - 0 - env",(char*)NULL);
   G__memfunc_setup("DeclFileName",1145,G__TMacSystem_DeclFileName_4_2,67,-1,-1,0,0,1,1,1,"",(char*)NULL);
   G__memfunc_setup("DeclFileLine",1152,G__TMacSystem_DeclFileLine_5_2,105,-1,-1,0,0,1,1,0,"",(char*)NULL);
   G__memfunc_setup("ImplFileName",1171,G__TMacSystem_ImplFileName_6_2,67,-1,-1,0,0,1,1,1,"",(char*)NULL);
   G__memfunc_setup("ImplFileLine",1178,G__TMacSystem_ImplFileLine_7_2,105,-1,-1,0,0,1,1,0,"",(char*)NULL);
   G__memfunc_setup("Class_Version",1339,G__TMacSystem_Class_Version_8_2,115,-1,G__defined_typename("Version_t"),0,0,1,1,0,"",(char*)NULL);
   G__memfunc_setup("IsA",253,G__TMacSystem_IsA_9_2,85,G__get_linked_tagnum(&G__G__MacSystemLN_TClass),-1,0,0,1,1,8,"",(char*)NULL);
   G__memfunc_setup("ShowMembers",1132,G__TMacSystem_ShowMembers_0_3,121,-1,-1,0,2,1,1,0,
"u TMemberInspector - 1 - insp C - - 0 - parent",(char*)NULL);
   G__memfunc_setup("Dictionary",1046,G__TMacSystem_Dictionary_1_3,121,-1,-1,0,0,1,1,0,"",(char*)NULL);
   // automatic copy constructor
   G__memfunc_setup("TMacSystem",1002,G__TMacSystem_TMacSystem_2_3,(int)('i'),G__get_linked_tagnum(&G__G__MacSystemLN_TMacSystem),-1,0,1,1,1,0,"u TMacSystem - 1 - -",(char*)NULL);
   // automatic destructor
   G__memfunc_setup("~TMacSystem",1128,G__TMacSystem_wATMacSystem_3_3,(int)('y'),-1,-1,0,0,1,1,0,"",(char*)NULL);
   G__tag_memfunc_reset();
}


/*********************************************************
* Member function information setup
*********************************************************/
extern "C" void G__cpp_setup_memfuncG__MacSystem() {
}

/*********************************************************
* Global variable information setup for each class
*********************************************************/
extern "C" void G__cpp_setup_globalG__MacSystem() {

   /* Setting up global variables */
   G__resetplocal();


   G__resetglobalenv();
}

/*********************************************************
* Global function information setup for each class
*********************************************************/
extern "C" void G__cpp_setup_funcG__MacSystem() {
   G__lastifuncposition();


   G__resetifuncposition();
}

/*********************************************************
* Class,struct,union,enum tag information setup
*********************************************************/
/* Setup class/struct taginfo */
G__linked_taginfo G__G__MacSystemLN_TClass = { "TClass" , 99 , -1 };
G__linked_taginfo G__G__MacSystemLN_TObject = { "TObject" , 99 , -1 };
G__linked_taginfo G__G__MacSystemLN_TString = { "TString" , 99 , -1 };
G__linked_taginfo G__G__MacSystemLN_TNamed = { "TNamed" , 99 , -1 };
G__linked_taginfo G__G__MacSystemLN_TSystem = { "TSystem" , 99 , -1 };
G__linked_taginfo G__G__MacSystemLN_TMacSystem = { "TMacSystem" , 99 , -1 };

extern "C" void G__cpp_setup_tagtableG__MacSystem() {

   /* Setting up class,struct,union tag entry */
   G__tagtable_setup(G__get_linked_tagnum(&G__G__MacSystemLN_TMacSystem),sizeof(TMacSystem),-1,0,"Interface to MacOS services",G__setup_memvarTMacSystem,G__setup_memfuncTMacSystem);
}
extern "C" void G__cpp_setupG__MacSystem() {
  G__check_setup_version(50911,"G__cpp_setupG__MacSystem()");
  G__set_cpp_environmentG__MacSystem();
  G__cpp_setup_tagtableG__MacSystem();

  G__cpp_setup_inheritanceG__MacSystem();

  G__cpp_setup_typetableG__MacSystem();

  G__cpp_setup_memvarG__MacSystem();

  G__cpp_setup_memfuncG__MacSystem();
  G__cpp_setup_globalG__MacSystem();
  G__cpp_setup_funcG__MacSystem();

   if(0==G__getsizep2memfunc()) G__get_sizep2memfuncG__MacSystem();
  return;
}
class G__cpp_setup_initG__MacSystem {
  public:
    G__cpp_setup_initG__MacSystem() { G__add_setup_func("G__MacSystem",&G__cpp_setupG__MacSystem); }
   ~G__cpp_setup_initG__MacSystem() { G__remove_setup_func("G__MacSystem"); }
};
static G__cpp_setup_initG__MacSystem G__cpp_setup_initializer;

//
// File generated by RootCint at Wed Aug 14 11:51:50 1996.
// Do NOT change. Changes will be lost next time file is generated
//

#include "TError.h"

//______________________________________________________________________________
void TMacSystem::ShowMembers(TMemberInspector &insp, char *parent)
{
   // Inspect the data members of an object of class TMacSystem.

   TClass *cl  = gROOT->GetClass("TMacSystem");
   Int_t   ncp = strlen(parent);
   fHostname.ShowMembers(insp, strcat(parent,"fHostname.")); parent[ncp] = 0;
   insp.Inspect(cl, parent, "fHasWaitNextEvent", &fHasWaitNextEvent);
   fWdPath.ShowMembers(insp, strcat(parent,"fWdPath.")); parent[ncp] = 0;
   TSystem::ShowMembers(insp, parent);
}

