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

//////////////////////////////////////////////////////////////////////////
//
// Wrapper class around the replacement for the gPad pointer.
//
//////////////////////////////////////////////////////////////////////////
class TVirtualPad;
struct TPadThreadLocal
{
   operator TVirtualPad*() { return TVirtualPad::Pad(); }
   operator bool() { return 0!=TVirtualPad::Pad(); }

   // Emulate the pointer behavior
   TVirtualPad*  operator->() { return TVirtualPad::Pad(); }
   TVirtualPad*& operator=(TVirtualPad *other) { return (TVirtualPad::Pad() = other); }
   bool operator!=(const TVirtualPad *other) const { return (TVirtualPad::Pad() != other); }
   bool operator!=(TVirtualPad *other) const { return (TVirtualPad::Pad() != other); }
   bool operator==(const TVirtualPad *other) const { return (TVirtualPad::Pad() == other); }
   bool operator==(TVirtualPad *other) const { return (TVirtualPad::Pad() == other); }
};

// Pretty printing routine for CINT
int G__ateval(TPadThreadLocal &) {
   TVirtualPad *pad  = TVirtualPad::Pad();
   if (pad) {
      printf("(class TVirtualPad*)%p\n",(void*)pad);
   } else {
      printf("(class TVirtualPad*)0x0\n",(void*)pad);
   }      
   return 1;
}

TPadThreadLocal gPad;


//////////////////////////////////////////////////////////////////////////
//
// Wrapper class around the replacement for the gDirectory pointer.
//
//////////////////////////////////////////////////////////////////////////
class TDirectory;
class TFile;
struct TDirectoryThreadLocal
{
   operator TDirectory*() { return TDirectory::CurrentDirectory(); }
   operator bool() { return 0!=TDirectory::CurrentDirectory(); }
   
   // This is needed to support (TFile*)gDirectory
   operator TFile*() { return (TFile*)(TDirectory::CurrentDirectory()); }
   // The following 2 are needed to remove ambiguity introduced by the operator TFile*
   operator void*() { return TDirectory::CurrentDirectory(); }

   // Emulate the pointer behavior
   TDirectory*  operator->() { return TDirectory::CurrentDirectory(); }
   TDirectory*& operator=(TDirectory *other) { return (TDirectory::CurrentDirectory() = other); }
   bool operator!=(const TDirectory *other) const { return (TDirectory::CurrentDirectory() != other); }
   bool operator!=(TDirectory *other) const { return (TDirectory::CurrentDirectory() != other); }
   bool operator==(const TDirectory *other) const { return (TDirectory::CurrentDirectory() == other); }
   bool operator==(TDirectory *other) const { return (TDirectory::CurrentDirectory() == other); }
};

// Pretty printing routine for CINT
int G__ateval(TDirectoryThreadLocal &) {
   TDirectory *dir  = TDirectory::CurrentDirectory();
   if (dir) {
      printf("(class TDirectory*)%p\n",(void*)dir);
   } else {
      printf("(class TDirectory*)0x0\n",(void*)dir);
   }      
   return 1;
}

TDirectoryThreadLocal gDirectory;


//////////////////////////////////////////////////////////////////////////
//
// Wrapper class around the replacement for the gFile pointer.
//
//////////////////////////////////////////////////////////////////////////
class TFile;
struct TFileThreadLocal
{
   operator TFile*() { return TFile::CurrentFile(); }
   operator bool() { return 0!=TFile::CurrentFile(); }
   
   // Emulate the pointer behavior
   TFile*  operator->() { return TFile::CurrentFile(); }
   TFile*& operator=(TFile *other) { return (TFile::CurrentFile() = other); }
   bool operator!=(const TFile *other) const { return (TFile::CurrentFile() != other); }
   bool operator!=(TFile *other) const { return (TFile::CurrentFile() != other); }
   bool operator==(const TFile *other) const { return (TFile::CurrentFile() == other); }
   bool operator==(TFile *other) const { return (TFile::CurrentFile() == other); }
};

// Pretty printing routine for CINT
int G__ateval(TFileThreadLocal &) {
   TFile *dir  = TFile::CurrentFile();
   if (dir) {
      printf("(class TFile*)%p\n",(void*)dir);
   } else {
      printf("(class TFile*)0x0\n",(void*)dir);
   }      
   return 1;
}

TFileThreadLocal gFile;


//////////////////////////////////////////////////////////////////////////
//
// Wrapper class around the replacement for the gInterpreter pointer.
//
//////////////////////////////////////////////////////////////////////////
class TInterpreter;
struct TInterpreterWrapper
{
   operator TInterpreter*() { return TInterpreter::Instance(); }
   operator bool() { return 0!=TInterpreter::Instance(); }
   
   // Emulate the pointer behavior
   TInterpreter*  operator->() { return TInterpreter::Instance(); }
   TInterpreter*& operator=(TInterpreter *other) { return (TInterpreter::Instance() = other); }
   bool operator!=(const TInterpreter *other) const { return (TInterpreter::Instance() != other); }
   bool operator!=(TInterpreter *other) const { return (TInterpreter::Instance() != other); }
   bool operator==(const TInterpreter *other) const { return (TInterpreter::Instance() == other); }
   bool operator==(TInterpreter *other) const { return (TInterpreter::Instance() == other); }
};

// Pretty printing routine for CINT
int G__ateval(TInterpreterWrapper &) {
   TInterpreter *interpreter  = TInterpreter::Instance();
   if (interpreter) {
      printf("(class TInterpreter*)%p\n",(void*)interpreter);
   } else {
      printf("(class TInterpreter*)0x0\n",(void*)interpreter);
   }      
   return 1;
}

TInterpreterWrapper gInterpreter;


//////////////////////////////////////////////////////////////////////////
//
// Wrapper class around the replacement for the gVirtualX pointer.
//
//////////////////////////////////////////////////////////////////////////
class TVirtualX;
struct TVirtualXWrapper
{
   operator TVirtualX*() { return TVirtualX::Instance(); }
   operator bool() { return 0!=TVirtualX::Instance(); }
   
   // Emulate the pointer behavior
   TVirtualX*  operator->() { return TVirtualX::Instance(); }
   TVirtualX*& operator=(TVirtualX *other) { return (TVirtualX::Instance() = other); }
   bool operator!=(const TVirtualX *other) const { return (TVirtualX::Instance() != other); }
   bool operator!=(TVirtualX *other) const { return (TVirtualX::Instance() != other); }
   bool operator==(const TVirtualX *other) const { return (TVirtualX::Instance() == other); }
   bool operator==(TVirtualX *other) const { return (TVirtualX::Instance() == other); }
};

// Pretty printing routine for CINT
int G__ateval(TVirtualXWrapper &) {
   TVirtualX *VirtualX  = TDirectory::Instance();
   if (VirtualX) {
      printf("(class TVirtualX*)%p\n",(void*)VirtualX);
   } else {
      printf("(class TVirtualX*)0x0\n",(void*)VirtualX);
   }      
   return 1;
}

TVirtualXWrapper gVirtualX;

#endif
