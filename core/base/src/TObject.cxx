// @(#)root/base:$Id$
// Author: Rene Brun   26/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TObject
\ingroup Base

Mother of all ROOT objects.

The TObject class provides default behaviour and protocol for all
objects in the ROOT system. It provides protocol for object I/O,
error handling, sorting, inspection, printing, drawing, etc.
Every object which inherits from TObject can be stored in the
ROOT collection classes.

TObject's bits can be used as flags, bits 0 - 13 and 24-31 are
reserved as  global bits while bits 14 - 23 can be used in different
class hierarchies (watch out for overlaps).

\Note
   Class inheriting directly or indirectly from TObject should not use
   `= default` for any of the constructors.
   The default implementation for a constructor can sometime do 'more' than we
   expect (and still being standard compliant).  On some platforms it will reset
   all the data member of the class including its base class's member before the
   actual execution of the base class constructor.
   `TObject`'s implementation of the `IsOnHeap` bit requires the memory occupied
   by `TObject::fUniqueID` to *not* be reset between the execution of `TObject::operator new`
   and the `TObject` constructor (Finding the magic pattern there is how we can determine
   that the object was allocated on the heap).
*/

#include <cstring>
#if !defined(WIN32) && !defined(__MWERKS__) && !defined(R__SOLARIS)
#include <strings.h>
#endif
#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>

#include "Varargs.h"
#include "snprintf.h"
#include "TObject.h"
#include "TBuffer.h"
#include "TClass.h"
#include "TGuiFactory.h"
#include "TMethod.h"
#include "TROOT.h"
#include "TError.h"
#include "TObjectTable.h"
#include "TVirtualPad.h"
#include "TInterpreter.h"
#include "TMemberInspector.h"
#include "TRefTable.h"
#include "TProcessID.h"

Longptr_t TObject::fgDtorOnly = 0;
Bool_t TObject::fgObjectStat = kTRUE;

ClassImp(TObject);

#if defined(__clang__) || defined (__GNUC__)
# define ATTRIBUTE_NO_SANITIZE_ADDRESS __attribute__((no_sanitize_address))
#else
# define ATTRIBUTE_NO_SANITIZE_ADDRESS
#endif

namespace ROOT {
namespace Internal {

// Return true if delete changes/poisons/taints the memory.
//
// Detect whether operator delete taints the memory. If it does, we can not rely
// on TestBit(kNotDeleted) to check if the memory has been deleted (but in case,
// like TClonesArray, where we know the destructor will be called but not operator
// delete, we can still use it to detect the cases where the destructor was called.

ATTRIBUTE_NO_SANITIZE_ADDRESS
bool DeleteChangesMemoryImpl()
{
   static constexpr UInt_t kGoldenUUID = 0x00000021;
   static constexpr UInt_t kGoldenbits = 0x03000000;

   TObject *o = new TObject;
   o->SetUniqueID(kGoldenUUID);
   UInt_t *o_fuid = &(o->fUniqueID);
   UInt_t *o_fbits = &(o->fBits);

   if (*o_fuid != kGoldenUUID) {
      Error("CheckingDeleteSideEffects",
            "fUniqueID is not as expected, we got 0x%.8x instead of 0x%.8x",
            *o_fuid, kGoldenUUID);
   }
   if (*o_fbits != kGoldenbits) {
      Error("CheckingDeleteSideEffects",
            "fBits is not as expected, we got 0x%.8x instead of 0x%.8x",
            *o_fbits, kGoldenbits);
   }
   if (gDebug >= 9) {
      unsigned char *oc = reinterpret_cast<unsigned char *>(o); // for address calculations
      unsigned char references[sizeof(TObject)];
      memcpy(references, oc, sizeof(TObject));

      // The effective part of this code (the else statement is just that without
      // any of the debug statement)
      delete o;

      // Not using the error logger, as there routine is meant to be called
      // during library initialization/loading.
      fprintf(stderr,
              "DEBUG: Checking before and after delete the content of a TObject with uniqueID 0x21\n");
      for(size_t i = 0; i < sizeof(TObject); i += 4) {
        fprintf(stderr, "DEBUG: 0x%.8x vs 0x%.8x\n", *(int*)(references +i), *(int*)(oc + i));
      }
   } else
      delete o;  // the 'if' part is that surrounded by the debug code.

   // Intentionally accessing the deleted memory to check whether it has been changed as
   // a consequence (side effect) of executing operator delete.  If there no change, we
   // can guess this is always the case and we can rely on the changes to fBits made
   // by ~TObject to detect use-after-delete error (and print a message rather than
   // stop the program with a segmentation fault)
#if defined(_MSC_VER) && defined(__SANITIZE_ADDRESS__)
   // on Windows, even __declspec(no_sanitize_address) does not prevent catching
   // heap-use-after-free errorswhen using the /fsanitize=address compiler flag
   // so don't even try
   return true;
#endif
   if ( *o_fbits != 0x01000000 ) {
      // operator delete tainted the memory, we can not rely on TestBit(kNotDeleted)
      return true;
   }
   return false;
}

bool DeleteChangesMemory()
{
   static const bool value = DeleteChangesMemoryImpl();
   if (gDebug >= 9)
      DeleteChangesMemoryImpl(); // To allow for printing the debug info
   return value;
}

}} // ROOT::Detail

////////////////////////////////////////////////////////////////////////////////
/// Copy this to obj.

void TObject::Copy(TObject &obj) const
{
   obj.fUniqueID = fUniqueID;   // when really unique don't copy
   if (obj.IsOnHeap()) {        // test uses fBits so don't move next line
      obj.fBits  = fBits;
      obj.fBits |= kIsOnHeap;
   } else {
      obj.fBits  = fBits;
      obj.fBits &= ~kIsOnHeap;
   }
   obj.fBits &= ~kIsReferenced;
   obj.fBits &= ~kCanDelete;
}

////////////////////////////////////////////////////////////////////////////////
/// TObject destructor. Removes object from all canvases and object browsers
/// if observer bit is on and remove from the global object table.

TObject::~TObject()
{
   // if (!TestBit(kNotDeleted))
   //    Fatal("~TObject", "object deleted twice");

   ROOT::CallRecursiveRemoveIfNeeded(*this);

   fBits &= ~kNotDeleted;

   if (fgObjectStat && gObjectTable) gObjectTable->RemoveQuietly(this);
}

////////////////////////////////////////////////////////////////////////////////
/// Private helper function which will dispatch to
/// TObjectTable::AddObj.
/// Included here to avoid circular dependency between header files.

void TObject::AddToTObjectTable(TObject *op)
{
   TObjectTable::AddObj(op);
}

////////////////////////////////////////////////////////////////////////////////
/// Append graphics object to current pad. In case no current pad is set
/// yet, create a default canvas with the name "c1".

void TObject::AppendPad(Option_t *option)
{
   if (!gPad)
      gROOT->MakeDefCanvas();

   if (!gPad->IsEditable())
      return;

   gPad->Add(this, option);
}

////////////////////////////////////////////////////////////////////////////////
/// Browse object. May be overridden for another default action

void TObject::Browse(TBrowser *b)
{
   //Inspect();
   TClass::AutoBrowse(this,b);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns name of class to which the object belongs.

const char *TObject::ClassName() const
{
   return IsA()->GetName();
}

////////////////////////////////////////////////////////////////////////////////
/// Make a clone of an object using the Streamer facility.
/// If the object derives from TNamed, this function is called
/// by TNamed::Clone. TNamed::Clone uses the optional argument to set
/// a new name to the newly created object.
///
/// If the object class has a DirectoryAutoAdd function, it will be
/// called at the end of the function with the parameter gDirectory.
/// This usually means that the object will be appended to the current
/// ROOT directory.

TObject *TObject::Clone(const char *) const
{
   if (gDirectory) {
     return gDirectory->CloneObject(this);
   } else {
     // Some of the streamer (eg. roofit's) expect(ed?) a valid gDirectory during streaming.
     return gROOT->CloneObject(this);
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Compare abstract method. Must be overridden if a class wants to be able
/// to compare itself with other objects. Must return -1 if this is smaller
/// than obj, 0 if objects are equal and 1 if this is larger than obj.

Int_t TObject::Compare(const TObject *) const
{
   AbstractMethod("Compare");
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Delete this object. Typically called as a command via the interpreter.
/// Normally use "delete" operator when object has been allocated on the heap.

void TObject::Delete(Option_t *)
{
   if (IsOnHeap()) {
      // Delete object from CINT symbol table so it can not be used anymore.
      // CINT object are always on the heap.
      gInterpreter->DeleteGlobal(this);

      delete this;
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Computes distance from point (px,py) to the object.
/// This member function must be implemented for each graphics primitive.
/// This default function returns a big number (999999).

Int_t TObject::DistancetoPrimitive(Int_t, Int_t)
{
   // AbstractMethod("DistancetoPrimitive");
   return 999999;
}

////////////////////////////////////////////////////////////////////////////////
/// Default Draw method for all objects

void TObject::Draw(Option_t *option)
{
   AppendPad(option);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw class inheritance tree of the class to which this object belongs.
/// If a class B inherits from a class A, description of B is drawn
/// on the right side of description of A.
/// Member functions overridden by B are shown in class A with a blue line
/// crossing-out the corresponding member function.
/// The following picture is the class inheritance tree of class TPaveLabel:
///
/// \image html base_object.png

void TObject::DrawClass() const
{
   IsA()->Draw();
}

////////////////////////////////////////////////////////////////////////////////
/// Draw a clone of this object in the current selected pad with:
/// `gROOT->SetSelectedPad(c1)`.
/// If pad was not selected - `gPad` will be used.

TObject *TObject::DrawClone(Option_t *option) const
{
   TVirtualPad::TContext ctxt(true);
   auto pad = gROOT->GetSelectedPad();
   if (pad)
      pad->cd();

   TObject *newobj = Clone();
   if (!newobj)
      return nullptr;

   if (!option || !*option)
      option = GetDrawOption();

   if (pad) {
      pad->Add(newobj, option);
      pad->Update();
   } else {
      newobj->Draw(option);
   }

   return newobj;
}

////////////////////////////////////////////////////////////////////////////////
/// Dump contents of object on stdout.
/// Using the information in the object dictionary (class TClass)
/// each data member is interpreted.
/// If a data member is a pointer, the pointer value is printed
///
/// The following output is the Dump of a TArrow object:
/// ~~~ {.cpp}
///   fAngle                   0           Arrow opening angle (degrees)
///   fArrowSize               0.2         Arrow Size
///   fOption.*fData
///   fX1                      0.1         X of 1st point
///   fY1                      0.15        Y of 1st point
///   fX2                      0.67        X of 2nd point
///   fY2                      0.83        Y of 2nd point
///   fUniqueID                0           object unique identifier
///   fBits                    50331648    bit field status word
///   fLineColor               1           line color
///   fLineStyle               1           line style
///   fLineWidth               1           line width
///   fFillColor               19          fill area color
///   fFillStyle               1001        fill area style
/// ~~~

void TObject::Dump() const
{
   // Get the actual address of the object.
   const void *actual = IsA()->DynamicCast(TObject::Class(),this,kFALSE);
   IsA()->Dump(actual);
}

////////////////////////////////////////////////////////////////////////////////
/// Execute method on this object with the given parameter string, e.g.
/// "3.14,1,\"text\"".

void TObject::Execute(const char *method, const char *params, Int_t *error)
{
   if (!IsA()) return;

   Bool_t must_cleanup = TestBit(kMustCleanup);

   gInterpreter->Execute(this, IsA(), method, params, error);

   if (gPad && must_cleanup) gPad->Modified();
}

////////////////////////////////////////////////////////////////////////////////
/// Execute method on this object with parameters stored in the TObjArray.
/// The TObjArray should contain an argv vector like:
/// ~~~ {.cpp}
///  argv[0] ... argv[n] = the list of TObjString parameters
/// ~~~

void TObject::Execute(TMethod *method, TObjArray *params, Int_t *error)
{
   if (!IsA()) return;

   Bool_t must_cleanup = TestBit(kMustCleanup);

   gInterpreter->Execute(this, IsA(), method, params, error);

   if (gPad && must_cleanup) gPad->Modified();
}


////////////////////////////////////////////////////////////////////////////////
/// Execute action corresponding to an event at (px,py). This method
/// must be overridden if an object can react to graphics events.

void TObject::ExecuteEvent(Int_t, Int_t, Int_t)
{
   // AbstractMethod("ExecuteEvent");
}

////////////////////////////////////////////////////////////////////////////////
/// Must be redefined in derived classes.
/// This function is typically used with TCollections, but can also be used
/// to find an object by name inside this object.

TObject *TObject::FindObject(const char *) const
{
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Must be redefined in derived classes.
/// This function is typically used with TCollections, but can also be used
/// to find an object inside this object.

TObject *TObject::FindObject(const TObject *) const
{
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Get option used by the graphics system to draw this object.
/// Note that before calling object.GetDrawOption(), you must
/// have called object.Draw(..) before in the current pad.

Option_t *TObject::GetDrawOption() const
{
   if (!gPad) return "";

   TListIter next(gPad->GetListOfPrimitives());
   while (auto obj = next()) {
      if (obj == this)
         return next.GetOption();
   }
   return "";
}

////////////////////////////////////////////////////////////////////////////////
/// Returns name of object. This default method returns the class name.
/// Classes that give objects a name should override this method.

const char *TObject::GetName() const
{
   return IsA()->GetName();
}

////////////////////////////////////////////////////////////////////////////////
/// Returns mime type name of object. Used by the TBrowser (via TGMimeTypes
/// class). Override for class of which you would like to have different
/// icons for objects of the same class.

const char *TObject::GetIconName() const
{
   return nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the unique object id.

UInt_t TObject::GetUniqueID() const
{
   return fUniqueID;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns string containing info about the object at position (px,py).
/// This method is typically overridden by classes of which the objects
/// can report peculiarities for different positions.
/// Returned string will be re-used (lock in MT environment).

char *TObject::GetObjectInfo(Int_t px, Int_t py) const
{
   if (!gPad) return (char*)"";
   static char info[64];
   Float_t x = gPad->AbsPixeltoX(px);
   Float_t y = gPad->AbsPixeltoY(py);
   snprintf(info,64,"x=%g, y=%g",gPad->PadtoX(x),gPad->PadtoY(y));
   return info;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns title of object. This default method returns the class title
/// (i.e. description). Classes that give objects a title should override
/// this method.

const char *TObject::GetTitle() const
{
   return IsA()->GetTitle();
}


////////////////////////////////////////////////////////////////////////////////
/// Execute action in response of a timer timing out. This method
/// must be overridden if an object has to react to timers.

Bool_t TObject::HandleTimer(TTimer *)
{
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Return hash value for this object.
///
/// Note: If this routine is overloaded in a derived class, this derived class
/// should also add
/// ~~~ {.cpp}
///    ROOT::CallRecursiveRemoveIfNeeded(*this)
/// ~~~
/// Otherwise, when RecursiveRemove is called (by ~TObject or example) for this
/// type of object, the transversal of THashList and THashTable containers will
/// will have to be done without call Hash (and hence be linear rather than
/// logarithmic complexity).  You will also see warnings like
/// ~~~
/// Error in <ROOT::Internal::TCheckHashRecursiveRemoveConsistency::CheckRecursiveRemove>: The class SomeName overrides TObject::Hash but does not call TROOT::RecursiveRemove in its destructor.
/// ~~~
///

ULong_t TObject::Hash() const
{
   //return (ULong_t) this >> 2;
   const void *ptr = this;
   return TString::Hash(&ptr, sizeof(void*));
}

////////////////////////////////////////////////////////////////////////////////
/// Returns kTRUE if object inherits from class "classname".

Bool_t TObject::InheritsFrom(const char *classname) const
{
   return IsA()->InheritsFrom(classname);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns kTRUE if object inherits from TClass cl.

Bool_t TObject::InheritsFrom(const TClass *cl) const
{
   return IsA()->InheritsFrom(cl);
}

////////////////////////////////////////////////////////////////////////////////
/// Dump contents of this object in a graphics canvas.
/// Same action as Dump but in a graphical form.
/// In addition pointers to other objects can be followed.
///
/// The following picture is the Inspect of a histogram object:
/// \image html base_inspect.png

void TObject::Inspect() const
{
   gGuiFactory->CreateInspectorImp(this, 400, 200);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns kTRUE in case object contains browsable objects (like containers
/// or lists of other objects).

Bool_t TObject::IsFolder() const
{
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// Default equal comparison (objects are equal if they have the same
/// address in memory). More complicated classes might want to override
/// this function.

Bool_t TObject::IsEqual(const TObject *obj) const
{
   return obj == this;
}

////////////////////////////////////////////////////////////////////////////////
/// The ls function lists the contents of a class on stdout. Ls output
/// is typically much less verbose then Dump().

void TObject::ls(Option_t *option) const
{
   TROOT::IndentLevel();
   std::cout <<"OBJ: " << IsA()->GetName() << "\t" << GetName() << "\t" << GetTitle() << " : ";
   std::cout << Int_t(TestBit(kCanDelete));
   if (option && strstr(option,"noaddr")==nullptr) {
      std::cout <<" at: "<< this ;
   }
   std::cout << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/// This method must be overridden to handle object notification (the base implementation is no-op).
///
/// Different objects in ROOT use the `Notify` method for different purposes, in coordination
/// with other objects that call this method at the appropriate time.
///
/// For example, `TLeaf` uses it to load class information; `TBranchRef` to load contents of
/// referenced branches `TBranchRef`; most notably, based on `Notify`, `TChain` implements a
/// callback mechanism to inform interested parties when it switches to a new sub-tree.
Bool_t TObject::Notify()
{
   return kFALSE;
}

////////////////////////////////////////////////////////////////////////////////
/// This method must be overridden if a class wants to paint itself.
/// The difference between Paint() and Draw() is that when a object
/// draws itself it is added to the display list of the pad in
/// which it is drawn (and automatically redrawn whenever the pad is
/// redrawn). While paint just draws the object without adding it to
/// the pad display list.

void TObject::Paint(Option_t *)
{
   // AbstractMethod("Paint");
}

////////////////////////////////////////////////////////////////////////////////
/// Pop on object drawn in a pad to the top of the display list. I.e. it
/// will be drawn last and on top of all other primitives.

void TObject::Pop()
{
   if (!gPad || !gPad->GetListOfPrimitives())
      return;

   if (this == gPad->GetListOfPrimitives()->Last())
      return;

   TListIter next(gPad->GetListOfPrimitives());
   while (auto obj = next())
      if (obj == this) {
         TString opt = next.GetOption();
         gPad->Remove(this, kFALSE); // do not issue modified by remove
         gPad->Add(this, opt.Data());
         return;
      }
}

////////////////////////////////////////////////////////////////////////////////
/// This method must be overridden when a class wants to print itself.

void TObject::Print(Option_t *) const
{
   std::cout <<"OBJ: " << IsA()->GetName() << "\t" << GetName() << "\t" << GetTitle() << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/// Read contents of object with specified name from the current directory.
/// First the key with the given name is searched in the current directory,
/// next the key buffer is deserialized into the object.
/// The object must have been created before via the default constructor.
/// See TObject::Write().

Int_t TObject::Read(const char *name)
{
   if (gDirectory)
      return gDirectory->ReadTObject(this,name);
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Recursively remove this object from a list. Typically implemented
/// by classes that can contain multiple references to a same object.

void TObject::RecursiveRemove(TObject *)
{
}


////////////////////////////////////////////////////////////////////////////////
/// Save this object in the file specified by filename.
///
/// - if "filename" contains ".root" the object is saved in filename as root
///   binary file.
///
/// - if "filename" contains ".xml"  the object is saved in filename as a xml
///   ascii file.
///
/// - if "filename" contains ".cc" the object is saved in filename as C code
///   independant from ROOT. The code is generated via SavePrimitive().
///   Specific code should be implemented in each object to handle this
///   option. Like in TF1::SavePrimitive().
///
/// - otherwise the object is written to filename as a CINT/C++ script. The
///   C++ code to rebuild this object is generated via SavePrimitive(). The
///   "option" parameter is passed to SavePrimitive. By default it is an empty
///   string. It can be used to specify the Draw option in the code generated
///   by SavePrimitive.
///
///  The function is available via the object context menu.

void TObject::SaveAs(const char *filename, Option_t *option) const
{
   //==============Save object as a root file===================================
   if (filename && strstr(filename,".root")) {
      if (gDirectory) gDirectory->SaveObjectAs(this,filename,"");
      return;
   }

   //==============Save object as a XML file====================================
   if (filename && strstr(filename,".xml")) {
      if (gDirectory) gDirectory->SaveObjectAs(this,filename,"");
      return;
   }

   //==============Save object as a JSON file================================
   if (filename && strstr(filename,".json")) {
      if (gDirectory) gDirectory->SaveObjectAs(this,filename,option);
      return;
   }

   //==============Save object as a C, ROOT independant, file===================
   if (filename && strstr(filename,".cc")) {
      TString fname;
      if (filename && strlen(filename) > 0) {
         fname = filename;
      } else {
         fname.Form("%s.cc", GetName());
      }
      std::ofstream out;
      out.open(fname.Data(), std::ios::out);
      if (!out.good ()) {
         Error("SaveAs", "cannot open file: %s", fname.Data());
         return;
      }
      ((TObject*)this)->SavePrimitive(out,"cc");
      out.close();
      Info("SaveAs", "cc file: %s has been generated", fname.Data());
      return;
   }

   //==============Save as a C++ CINT file======================================
   TString fname;
   if (filename && strlen(filename) > 0) {
      fname = filename;
   } else {
      fname.Form("%s.C", GetName());
   }
   std::ofstream out;
   out.open(fname.Data(), std::ios::out);
   if (!out.good ()) {
      Error("SaveAs", "cannot open file: %s", fname.Data());
      return;
   }
   out <<"{"<<std::endl;
   out <<"//========= Macro generated from object: "<<GetName()<<"/"<<GetTitle()<<std::endl;
   out <<"//========= by ROOT version"<<gROOT->GetVersion()<<std::endl;
   ((TObject*)this)->SavePrimitive(out,option);
   out <<"}"<<std::endl;
   out.close();
   Info("SaveAs", "C++ Macro file: %s has been generated", fname.Data());
}

////////////////////////////////////////////////////////////////////////////////
/// Save object constructor in the output stream "out".
/// Can be used as first statement when implementing SavePrimitive() method for the object

void TObject::SavePrimitiveConstructor(std::ostream &out, TClass *cl, const char *variable_name, const char *constructor_agrs, Bool_t empty_line)
{
   if (empty_line)
      out << "   \n";

   out << "   ";
   if (!gROOT->ClassSaved(cl))
      out << cl->GetName() << " *";
   out << variable_name << " = new " << cl->GetName() << "(" << constructor_agrs << ");\n";
}


////////////////////////////////////////////////////////////////////////////////
/// Save array in the output stream "out" as vector.
/// Create unique variable name based on prefix value
/// Returns name of vector which can be used in constructor or in other places of C++ code

TString TObject::SavePrimitiveVector(std::ostream &out, const char *prefix, Int_t len, Double_t *arr, Bool_t empty_line)
{
   thread_local int vectid = 0;

   TString vectame = TString::Format("%s_vect%d", prefix, vectid++);

   if (empty_line)
      out << "   \n";

   out << "   std::vector<Double_t> " << vectame;
   if (len > 0) {
      const auto old_precision{out.precision()};
      constexpr auto max_precision{std::numeric_limits<double>::digits10 + 1};
      out << std::setprecision(max_precision);
      Bool_t use_new_lines = len > 15;

      out << "{";
      for (Int_t i = 0; i < len; i++) {
         out << (((i % 10 == 0) && use_new_lines) ? "\n      " : " ") << arr[i];
         if (i < len - 1)
            out << ",";
      }
      out << (use_new_lines ? "\n   }" : " }");

      out << std::setprecision(old_precision);
   }
   out << ";\n";
   return vectame;
}

////////////////////////////////////////////////////////////////////////////////
/// Save invocation of primitive Draw() method
/// Skipped if option contains "nodraw" string

void TObject::SavePrimitiveDraw(std::ostream &out, const char *variable_name, Option_t *option)
{
   if (!option || !strstr(option, "nodraw")) {
      out << "   " << variable_name << "->Draw(";
      if (option && *option)
         out << "\"" << TString(option).ReplaceSpecialCppChars() << "\"";
      out << ");\n";
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Save a primitive as a C++ statement(s) on output stream "out".

void TObject::SavePrimitive(std::ostream &out, Option_t * /*= ""*/)
{
   out << "//Primitive: " << GetName() << "/" << GetTitle()
       <<". You must implement " << ClassName() << "::SavePrimitive" << std::endl;
}

////////////////////////////////////////////////////////////////////////////////
/// Set drawing option for object. This option only affects
/// the drawing style and is stored in the option field of the
/// TObjOptLink supporting a TPad's primitive list (TList).
/// Note that it does not make sense to call object.SetDrawOption(option)
/// before having called object.Draw().

void TObject::SetDrawOption(Option_t *option)
{
   if (!gPad || !option) return;

   TListIter next(gPad->GetListOfPrimitives());
   delete gPad->FindObject("Tframe");
   while (auto obj = next())
      if (obj == this) {
         next.SetOption(option);
         return;
      }
}

////////////////////////////////////////////////////////////////////////////////
/// Set or unset the user status bits as specified in f.

void TObject::SetBit(UInt_t f, Bool_t set)
{
   if (set)
      SetBit(f);
   else
      ResetBit(f);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the unique object id.

void TObject::SetUniqueID(UInt_t uid)
{
   fUniqueID = uid;
}

////////////////////////////////////////////////////////////////////////////////
/// Set current style settings in this object
/// This function is called when either TCanvas::UseCurrentStyle
/// or TROOT::ForceStyle have been invoked.

void TObject::UseCurrentStyle()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Write this object to the current directory.
/// The data structure corresponding to this object is serialized.
/// The corresponding buffer is written to the current directory
/// with an associated key with name "name".
///
/// Writing an object to a file involves the following steps:
///
///  - Creation of a support TKey object in the current directory.
///    The TKey object creates a TBuffer object.
///
///  - The TBuffer object is filled via the class::Streamer function.
///
///  - If the file is compressed (default) a second buffer is created to
///    hold the compressed buffer.
///
///  - Reservation of the corresponding space in the file by looking
///    in the TFree list of free blocks of the file.
///
///  - The buffer is written to the file.
///
///  Bufsize can be given to force a given buffer size to write this object.
///  By default, the buffersize will be taken from the average buffer size
///  of all objects written to the current file so far.
///
///  If a name is specified, it will be the name of the key.
///  If name is not given, the name of the key will be the name as returned
///  by GetName().
///
///  The option can be a combination of: kSingleKey, kOverwrite or kWriteDelete
///  Using the kOverwrite option a previous key with the same name is
///  overwritten. The previous key is deleted before writing the new object.
///  Using the kWriteDelete option a previous key with the same name is
///  deleted only after the new object has been written. This option
///  is safer than kOverwrite but it is slower.
///  NOTE: Neither kOverwrite nor kWriteDelete reduces the size of a TFile--
///  the space is simply freed up to be overwritten; in the case of a TTree,
///  it is more complicated. If one opens a TTree, appends some entries,
///  then writes it out, the behaviour is effectively the same. If, however,
///  one creates a new TTree and writes it out in this way,
///  only the metadata is replaced, effectively making the old data invisible
///  without deleting it. TTree::Delete() can be used to mark all disk space
///  occupied by a TTree as free before overwriting its metadata this way.
///  The kSingleKey option is only used by TCollection::Write() to write
///  a container with a single key instead of each object in the container
///  with its own key.
///
///  An object is read from the file into memory via TKey::Read() or
///  via TObject::Read().
///
///  The function returns the total number of bytes written to the file.
///  It returns 0 if the object cannot be written.

Int_t TObject::Write(const char *name, Int_t option, Int_t bufsize) const
{
   if (R__unlikely(option & kOnlyPrepStep))
      return 0;

   TString opt = "";
   if (option & kSingleKey)   opt += "SingleKey";
   if (option & kOverwrite)   opt += "OverWrite";
   if (option & kWriteDelete) opt += "WriteDelete";

   if (gDirectory)
      return gDirectory->WriteTObject(this,name,opt.Data(),bufsize);

   const char *objname = name ? name : GetName();
   Error("Write","The current directory (gDirectory) is null. The object (%s) has not been written.",objname);
   return 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Write this object to the current directory. For more see the
/// const version of this method.

Int_t TObject::Write(const char *name, Int_t option, Int_t bufsize)
{
   return ((const TObject*)this)->Write(name, option, bufsize);
}

////////////////////////////////////////////////////////////////////////////////
/// Stream an object of class TObject.

void TObject::Streamer(TBuffer &R__b)
{
   if (IsA()->CanIgnoreTObjectStreamer()) return;
   UShort_t pidf;
   if (R__b.IsReading()) {
      R__b.SkipVersion(); // Version_t R__v = R__b.ReadVersion(); if (R__v) { }
      R__b >> fUniqueID;
      const UInt_t isonheap = fBits & kIsOnHeap; // Record how this instance was actually allocated.
      R__b >> fBits;
      fBits |= isonheap | kNotDeleted;  // by definition de-serialized object are not yet deleted.
      if (TestBit(kIsReferenced)) {
         //if the object is referenced, we must read its old address
         //and store it in the ProcessID map in gROOT
         R__b >> pidf;
         pidf += R__b.GetPidOffset();
         TProcessID *pid = R__b.ReadProcessID(pidf);
         if (pid) {
            UInt_t gpid = pid->GetUniqueID();
            if (gpid>=0xff) {
               fUniqueID = fUniqueID | 0xff000000;
            } else {
               fUniqueID = ( fUniqueID & 0xffffff) + (gpid<<24);
            }
            pid->PutObjectWithID(this);
         }
      }
   } else {
      R__b.WriteVersion(TObject::IsA());
      // Can not read TFile.h here and avoid going through the interpreter by
      // simply hard-coding this value.
      // This **must** be equal to TFile::k630forwardCompatibility
      constexpr int TFile__k630forwardCompatibility = BIT(2);
      const auto parent = R__b.GetParent();
      if (!TestBit(kIsReferenced)) {
         R__b << fUniqueID;
         if (R__unlikely(parent && parent->TestBit(TFile__k630forwardCompatibility)))
            R__b << fBits;
         else
            R__b << (fBits & (~kIsOnHeap & ~kNotDeleted));
      } else {
         //if the object is referenced, we must save its address/file_pid
         UInt_t uid = fUniqueID & 0xffffff;
         R__b << uid;
         if (R__unlikely(parent && parent->TestBit(TFile__k630forwardCompatibility)))
            R__b << fBits;
         else
            R__b << (fBits & (~kIsOnHeap & ~kNotDeleted));
         TProcessID *pid = TProcessID::GetProcessWithUID(fUniqueID,this);
         //add uid to the TRefTable if there is one
         TRefTable *table = TRefTable::GetRefTable();
         if(table) table->Add(uid, pid);
         pidf = R__b.WriteProcessID(pid);
         R__b << pidf;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Interface to ErrorHandler (protected).

void TObject::DoError(int level, const char *location, const char *fmt, va_list va) const
{
   const char *classname = "UnknownClass";
   if (TROOT::Initialized())
      classname = ClassName();

   ::ErrorHandler(level, Form("%s::%s", classname, location), fmt, va);
}

////////////////////////////////////////////////////////////////////////////////
/// Issue info message. Use "location" to specify the method where the
/// warning occurred. Accepts standard printf formatting arguments.

void TObject::Info(const char *location, const char *va_(fmt), ...) const
{
   va_list ap;
   va_start(ap, va_(fmt));
   DoError(kInfo, location, va_(fmt), ap);
   va_end(ap);
}

////////////////////////////////////////////////////////////////////////////////
/// Issue warning message. Use "location" to specify the method where the
/// warning occurred. Accepts standard printf formatting arguments.

void TObject::Warning(const char *location, const char *va_(fmt), ...) const
{
   va_list ap;
   va_start(ap, va_(fmt));
   DoError(kWarning, location, va_(fmt), ap);
   va_end(ap);
   if (TROOT::Initialized())
      gROOT->Message(1001, this);
}

////////////////////////////////////////////////////////////////////////////////
/// Issue error message. Use "location" to specify the method where the
/// error occurred. Accepts standard printf formatting arguments.

void TObject::Error(const char *location, const char *va_(fmt), ...) const
{
   va_list ap;
   va_start(ap, va_(fmt));
   DoError(kError, location, va_(fmt), ap);
   va_end(ap);
   if (TROOT::Initialized())
      gROOT->Message(1002, this);
}

////////////////////////////////////////////////////////////////////////////////
/// Issue system error message. Use "location" to specify the method where
/// the system error occurred. Accepts standard printf formatting arguments.

void TObject::SysError(const char *location, const char *va_(fmt), ...) const
{
   va_list ap;
   va_start(ap, va_(fmt));
   DoError(kSysError, location, va_(fmt), ap);
   va_end(ap);
   if (TROOT::Initialized())
      gROOT->Message(1003, this);
}

////////////////////////////////////////////////////////////////////////////////
/// Issue fatal error message. Use "location" to specify the method where the
/// fatal error occurred. Accepts standard printf formatting arguments.

void TObject::Fatal(const char *location, const char *va_(fmt), ...) const
{
   va_list ap;
   va_start(ap, va_(fmt));
   DoError(kFatal, location, va_(fmt), ap);
   va_end(ap);
   if (TROOT::Initialized())
      gROOT->Message(1004, this);
}

////////////////////////////////////////////////////////////////////////////////
/// Call this function within a function that you don't want to define as
/// purely virtual, in order not to force all users deriving from that class to
/// implement that maybe (on their side) unused function; but at the same time,
/// emit a run-time warning if they try to call it, telling that it is not
/// implemented in the derived class: action must thus be taken on the user side
/// to override it. In other word, this method acts as a "runtime purely virtual"
/// warning instead of a "compiler purely virtual" error.
/// \warning This interface is a legacy function that is no longer recommended
/// to be used by new development code.
/// \note The name "AbstractMethod" does not imply that it's an abstract method
/// in the strict C++ sense.

void TObject::AbstractMethod(const char *method) const
{
   Warning(method, "this method must be overridden!");
}

////////////////////////////////////////////////////////////////////////////////
/// Use this method to signal that a method (defined in a base class)
/// may not be called in a derived class (in principle against good
/// design since a child class should not provide less functionality
/// than its parent, however, sometimes it is necessary).

void TObject::MayNotUse(const char *method) const
{
   Warning(method, "may not use this method");
}

////////////////////////////////////////////////////////////////////////////////
/// Use this method to declare a method obsolete. Specify as of which version
/// the method is obsolete and as from which version it will be removed.

void TObject::Obsolete(const char *method, const char *asOfVers, const char *removedFromVers) const
{
   const char *classname = "UnknownClass";
   if (TROOT::Initialized())
      classname = ClassName();

   ::Obsolete(Form("%s::%s", classname, method), asOfVers, removedFromVers);
}

////////////////////////////////////////////////////////////////////////////////
/// Get status of object stat flag.

Bool_t TObject::GetObjectStat()
{
   return fgObjectStat;
}
////////////////////////////////////////////////////////////////////////////////
/// Turn on/off tracking of objects in the TObjectTable.

void TObject::SetObjectStat(Bool_t stat)
{
   fgObjectStat = stat;
}

////////////////////////////////////////////////////////////////////////////////
/// Return destructor only flag

Longptr_t TObject::GetDtorOnly()
{
   return fgDtorOnly;
}

////////////////////////////////////////////////////////////////////////////////
/// Set destructor only flag

void TObject::SetDtorOnly(void *obj)
{
   fgDtorOnly = (Longptr_t) obj;
}

////////////////////////////////////////////////////////////////////////////////
/// Operator delete

void TObject::operator delete(void *ptr)
{
   if ((Longptr_t) ptr != fgDtorOnly)
      TStorage::ObjectDealloc(ptr);
   else
      fgDtorOnly = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Operator delete []

void TObject::operator delete[](void *ptr)
{
   if ((Longptr_t) ptr != fgDtorOnly)
      TStorage::ObjectDealloc(ptr);
   else
      fgDtorOnly = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Operator delete for sized deallocation.

void TObject::operator delete(void *ptr, size_t size)
{
   if ((Longptr_t) ptr != fgDtorOnly)
      TStorage::ObjectDealloc(ptr, size);
   else
      fgDtorOnly = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Operator delete [] for sized deallocation.

void TObject::operator delete[](void *ptr, size_t size)
{
   if ((Longptr_t) ptr != fgDtorOnly)
      TStorage::ObjectDealloc(ptr, size);
   else
      fgDtorOnly = 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Print value overload

std::string cling::printValue(TObject *val)
{
   std::ostringstream strm;
   strm << "Name: " << val->GetName() << " Title: " << val->GetTitle();
   return strm.str();
}

////////////////////////////////////////////////////////////////////////////////
/// Only called by placement new when throwing an exception.

void TObject::operator delete(void *ptr, void *vp)
{
   TStorage::ObjectDealloc(ptr, vp);
}

////////////////////////////////////////////////////////////////////////////////
/// Only called by placement new[] when throwing an exception.

void TObject::operator delete[](void *ptr, void *vp)
{
   TStorage::ObjectDealloc(ptr, vp);
}
