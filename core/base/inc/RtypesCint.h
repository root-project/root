/* @(#)root/base:$Id$*/

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RtypesCint
#define ROOT_RtypesCint

#if !defined(__CINT__) || defined(__MAKECINT)
#error This header file can only be used in interpreted code.
#endif

// The definition of ClassDef is used only for files loaded via the interpreter

#define ClassDef(name,id) \
private: \
   static TClass *fgIsA; \
public: \
 static TClass *Class() { return fgIsA ? fgIsA : (fgIsA = TClass::GetClass(#name)); } \
   static const char *Class_Name() { return #name; }       \
   static Version_t Class_Version() { return id; } \
   static void Dictionary(); \
   virtual TClass *IsA() const { return name::Class(); } \
   virtual void ShowMembers(TMemberInspector &insp, char *parent) { \
      Class()->InterpretedShowMembers(this, insp, parent); }        \
   virtual void Streamer(TBuffer &b); \
   void StreamerNVirtual(TBuffer &b) { name::Streamer(b); } \
   static const char *DeclFileName() { return __FILE__; } \
   static const char *ImplFileName(); \
   static int ImplFileLine(); \
   static int DeclFileLine() { return __LINE__; }

#define ClassDefT(name,id) ClassDef(name,id)

// Obsolete macros
#define ClassDefT2(name,Tmpl)
#define ClassDef2T2(name,Tmpl1,Tmpl2)
#define ClassImp2T(name,Tmpl1,Tmpl2) templateClassImp(name)
#define ClassDef3T2(name,Tmpl1,Tmpl2,Tmpl3)
#define ClassImp3T(name,Tmpl1,Tmpl2,Tmpl3) templateClassImp(name)

// Macro not useful (yet?) in CINT
#define ClassImp(X)
#define ClassImpUnique(name,key)
#define RootClassVersion(name,VersionNumber)

#endif
