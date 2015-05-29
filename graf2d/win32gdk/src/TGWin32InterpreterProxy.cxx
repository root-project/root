// @(#)root/meta:$Id: TGWin32InterpreterProxy.cxx 38517 2011-03-18 20:20:16Z pcanal $
// Author: Valeriy Onuchin  15/11/2003


/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGWin32InterpreterProxy                                              //
//                                                                      //
// This class defines thread-safe interface to a command line           //
// interpreter (CINT).                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TGWin32ProxyDefs.h"
#include "TGWin32InterpreterProxy.h"
#include "TROOT.h"
#include "TGWin32.h"

#if ROOT_VERSION_CODE < ROOT_VERSION(6,00,00)

////////////////////////////////////////////////////////////////////////////////
//______________________________________________________________________________
TInterpreter *TGWin32InterpreterProxy::RealObject()
{
   // returns TCint object

   return gROOT->GetInterpreter();
}

RETURN_PROXY_OBJECT(Interpreter)
VOID_METHOD_ARG1(Interpreter,AddIncludePath,const char*,path,1)
RETURN_METHOD_ARG1(Interpreter,Int_t,AutoLoad,const char *,classname)
VOID_METHOD_ARG0(Interpreter,ClearFileBusy,1)
VOID_METHOD_ARG0(Interpreter,ClearStack,1)
RETURN_METHOD_ARG1(Interpreter,Bool_t,Declare,const char*, code)
VOID_METHOD_ARG0(Interpreter,EndOfLineAction,1)
VOID_METHOD_ARG0(Interpreter,EnableAutoLoading,1)
VOID_METHOD_ARG0(Interpreter,Initialize,0)
// Does not support references:
//VOID_METHOD_ARG3(Interpreter,InspectMembers,TMemberInspector&, insp, void*, obj, const TClass*,cl,1)
void _NAME4_(p2,Interpreter,InspectMembers,insp)(void *in)
{
   struct tmp {
      TMemberInspector *par1; void *par2; const TClass *par3;
   };
   tmp *p = (tmp*)in;
   _NAME3_(TGWin32,Interpreter,Proxy)::RealObject()->InspectMembers(*p->par1,p->par2,p->par3);
}

void TGWin32InterpreterProxy::InspectMembers(TMemberInspector& insp, void* obj, const TClass* cl)
{
   DEBUG_PROFILE_PROXY_START(InspectMembers)
   struct tmp {
      TMemberInspector *par1; void *par2; const TClass *par3;
      tmp(TMemberInspector *par1,void *par2,const TClass *par3):par1(par1),par2(par2),par3(par3) {}
   };
   fParam = new tmp(&insp,obj,cl);
   fCallBack = &_NAME4_(p2,Interpreter,InspectMembers,insp);
   Bool_t batch = ForwardCallBack(1);
   //   insp = ((tmp*)fParam)->par1;
   obj = ((tmp*)fParam)->par2;
   cl = ((tmp*)fParam)->par3;
   if (!batch) delete fParam;
   DEBUG_PROFILE_PROXY_STOP(InspectMembers)
}

RETURN_METHOD_ARG3(Interpreter,TClass*,GenerateTClass,const char *,classname,Bool_t,emulation,Bool_t,silent);
RETURN_METHOD_ARG2(Interpreter,TClass*,GenerateTClass,ClassInfo_t *,classinfo,Bool_t,silent); 
RETURN_METHOD_ARG3(Interpreter,Int_t,GenerateDictionary,const char*,classes,const char*,headers,const char*,options); 
RETURN_METHOD_ARG0(Interpreter,char*,GetPrompt)
RETURN_METHOD_ARG0(Interpreter,const char*,GetSharedLibs)
RETURN_METHOD_ARG0(Interpreter,const char*,GetIncludePath)
RETURN_METHOD_ARG2(Interpreter,Int_t,Load,const char*,filenam,Bool_t,system)
RETURN_METHOD_ARG1(Interpreter,Int_t,LoadLibraryMap,const char*,rootmapfile)
RETURN_METHOD_ARG0(Interpreter,Int_t,RescanLibraryMap)
RETURN_METHOD_ARG0(Interpreter,Int_t,ReloadAllSharedLibraryMaps)
RETURN_METHOD_ARG0(Interpreter,Int_t,UnloadAllSharedLibraryMaps)
RETURN_METHOD_ARG1(Interpreter,Int_t,UnloadLibraryMap,const char*,library)
VOID_METHOD_ARG2(Interpreter,LoadMacro,const char*,filename,TInterpreter::EErrorCode*,error,1)
RETURN_METHOD_ARG2(Interpreter,Long_t,ProcessLine,const char*,line,TInterpreter::EErrorCode*,error)
RETURN_METHOD_ARG2(Interpreter,Long_t,ProcessLineSynch,const char*,line,TInterpreter::EErrorCode*,error)
VOID_METHOD_ARG0(Interpreter,PrintIntro,1)
typedef const char* (*GetlineFunc_t)(const char* prompt);
typedef void (*HistaddFunc_t)(const char* line);
RETURN_METHOD_ARG2(Interpreter,Int_t,SetClassSharedLibs,const char*,cls,const char*,libs);
VOID_METHOD_ARG2(Interpreter,SetGetline,GetlineFunc_t, getlineFunc,\
                 HistaddFunc_t, histaddFunc, 1)
VOID_METHOD_ARG0(Interpreter,Reset,1)
VOID_METHOD_ARG0(Interpreter,ResetAll,1)
VOID_METHOD_ARG0(Interpreter,ResetGlobals,1)
VOID_METHOD_ARG1(Interpreter,ResetGlobalVar,void*,obj,1)
VOID_METHOD_ARG0(Interpreter,RewindDictionary,1)
RETURN_METHOD_ARG1(Interpreter,Int_t,DeleteGlobal,void*,obj)
RETURN_METHOD_ARG1(Interpreter,Int_t,DeleteVariable,const char*,name)
VOID_METHOD_ARG0(Interpreter,SaveContext,1)
VOID_METHOD_ARG0(Interpreter,SaveGlobalsContext,1)
VOID_METHOD_ARG0_LOCK(Interpreter,UpdateListOfGlobals)
VOID_METHOD_ARG0_LOCK(Interpreter,UpdateListOfGlobalFunctions)
VOID_METHOD_ARG0_LOCK(Interpreter,UpdateListOfTypes)
VOID_METHOD_ARG2_LOCK(Interpreter,SetClassInfo,TClass*,cl,Bool_t,reload)
RETURN_METHOD_ARG3(Interpreter,Bool_t,CheckClassInfo,const char*,name,Bool_t,autoload,Bool_t,isClassOrNamespaceOnly)
RETURN_METHOD_ARG1(Interpreter,Bool_t,CheckClassTemplate,const char*,name)
RETURN_METHOD_ARG2(Interpreter,Long_t,Calc,const char*,line,TInterpreter::EErrorCode*,error)
VOID_METHOD_ARG1_LOCK(Interpreter,CreateListOfBaseClasses,TClass*,cl)
VOID_METHOD_ARG1_LOCK(Interpreter,CreateListOfDataMembers,TClass*,cl)
VOID_METHOD_ARG1_LOCK(Interpreter,CreateListOfMethods,TClass*,cl)
VOID_METHOD_ARG1_LOCK(Interpreter,CreateListOfMethodArgs,TFunction*,m)
VOID_METHOD_ARG1_LOCK(Interpreter,UpdateListOfMethods,TClass*,cl)
RETURN_METHOD_ARG3(Interpreter,TString,GetMangledName,TClass*,cl,const char*,method,const char*,params)
RETURN_METHOD_ARG3(Interpreter,TString,GetMangledNameWithPrototype,TClass*,cl,const char*,method,const char*,proto)
RETURN_METHOD_ARG3(Interpreter,void*,GetInterfaceMethod,TClass*,cl,const char*,method,const char*,params)
RETURN_METHOD_ARG3(Interpreter,void*,GetInterfaceMethodWithPrototype,TClass*,cl,const char*,method,const char*,proto)
RETURN_METHOD_ARG1(Interpreter,const char*,GetClassSharedLibs,const char*,s)
RETURN_METHOD_ARG1(Interpreter,const char*,GetSharedLibDeps,const char*,s)
RETURN_METHOD_ARG2(Interpreter,const char*,GetInterpreterTypeName,const char*,s,Bool_t,full)
VOID_METHOD_ARG3(Interpreter,Execute,const char*,function,const char*,params,int*,error,1)
VOID_METHOD_ARG5(Interpreter,Execute,TObject*,obj,TClass*,cl,const char*,method,const char*,params,int*,error,1)
VOID_METHOD_ARG5(Interpreter,Execute,TObject*,object,TClass*,cl,TMethod*,method,TObjArray*,params,int*,error,1)
RETURN_METHOD_ARG2(Interpreter,Long_t,ExecuteMacro,const char*,filename,TInterpreter::EErrorCode*,error)
RETURN_METHOD_ARG1(Interpreter,Bool_t,SetErrorMessages,Bool_t,enable)
VOID_METHOD_ARG1(Interpreter,SetProcessLineLock,Bool_t,lock,1)
RETURN_METHOD_ARG1(Interpreter,const char*,TypeName,const char*,s)
//Bool_t TGWin32InterpreterProxy::CheckClassInfo(const char* name) { return RealObject()->CheckClassInfo(name); }

#endif
