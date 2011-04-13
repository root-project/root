// @(#)root/base:$Id$
// Author: Rene Brun   26/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TObject                                                              //
//                                                                      //
// Mother of all ROOT objects.                                          //
//                                                                      //
// The TObject class provides default behaviour and protocol for all    //
// objects in the ROOT system. It provides protocol for object I/O,     //
// error handling, sorting, inspection, printing, drawing, etc.         //
// Every object which inherits from TObject can be stored in the        //
// ROOT collection classes.                                             //
// TObject's bits can be used as flags, bits 0 - 13 and 24-31 are       //
// reserved as  global bits while bits 14 - 23 can be used in different //
// class hierarchies (watch out for overlaps).                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <string.h>
#if !defined(WIN32) && !defined(__MWERKS__) && !defined(R__SOLARIS)
#include <strings.h>
#endif
#include <stdlib.h>
#include <stdio.h>

#include "Varargs.h"
#include "Riostream.h"
#include "TObject.h"
#include "TClass.h"
#include "TGuiFactory.h"
#include "TMethod.h"
#include "TROOT.h"
#include "TError.h"
#include "TObjectTable.h"
#include "TVirtualPad.h"
#include "TInterpreter.h"
#include "TMemberInspector.h"
#include "TObjString.h"
#include "TRefTable.h"
#include "TProcessID.h"

class TDumpMembers : public TMemberInspector {
   // Implemented in TClass.cxx
public:
   TDumpMembers() { }
   void Inspect(TClass *cl, const char *parent, const char *name, const void *addr);
};

Long_t TObject::fgDtorOnly = 0;
Bool_t TObject::fgObjectStat = kTRUE;

ClassImp(TObject)

//______________________________________________________________________________
TObject::TObject() : fUniqueID(0), fBits(kNotDeleted)
{
   // TObject constructor. It sets the two data words of TObject to their
   // initial values. The unique ID is set to 0 and the status word is
   // set depending if the object is created on the stack or allocated
   // on the heap. Depending on the ROOT environment variable "Root.MemStat"
   // (see TEnv) the object is added to the global TObjectTable for
   // bookkeeping.

   if (TStorage::IsOnHeap(this))
      fBits |= kIsOnHeap;

   if (fgObjectStat) TObjectTable::AddObj(this);
}

//______________________________________________________________________________
TObject::TObject(const TObject &obj)
{
   // TObject copy ctor.

   fUniqueID = obj.fUniqueID;  // when really unique don't copy
   fBits     = obj.fBits;

   if (TStorage::IsOnHeap(this))
      fBits |= kIsOnHeap;
   else
      fBits &= ~kIsOnHeap;

   fBits &= ~kIsReferenced;
   fBits &= ~kCanDelete;

   if (fgObjectStat) TObjectTable::AddObj(this);
}

//______________________________________________________________________________
TObject& TObject::operator=(const TObject &rhs)
{
   // TObject assignment operator.

   if (this != &rhs) {
      fUniqueID = rhs.fUniqueID;  // when really unique don't copy
      if (IsOnHeap()) {           // test uses fBits so don't move next line
         fBits  = rhs.fBits;
         fBits |= kIsOnHeap;
      } else {
         fBits  = rhs.fBits;
         fBits &= ~kIsOnHeap;
      }
      fBits &= ~kIsReferenced;
      fBits &= ~kCanDelete;
   }
   return *this;
}

//______________________________________________________________________________
void TObject::Copy(TObject &obj) const
{
   // Copy this to obj.

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

//______________________________________________________________________________
TObject::~TObject()
{
   // TObject destructor. Removes object from all canvases and object browsers
   // if observer bit is on and remove from the global object table.

   // if (!TestBit(kNotDeleted))
   //    Fatal("~TObject", "object deleted twice");

   if (gROOT) {
      if (gROOT->MustClean()) {
         if (gROOT == this) return;
         if (TestBit(kMustCleanup)) {
            gROOT->GetListOfCleanups()->RecursiveRemove(this);
         }
      }
   }

   fBits &= ~kNotDeleted;

   if (fgObjectStat && gObjectTable) gObjectTable->Remove(this);
}

//______________________________________________________________________________
void TObject::AppendPad(Option_t *option)
{
   // Append graphics object to current pad. In case no current pad is set
   // yet, create a default canvas with the name "c1".

   if (!gPad) {
      gROOT->MakeDefCanvas();
   }
   if (!gPad->IsEditable()) return;
   SetBit(kMustCleanup);
   gPad->GetListOfPrimitives()->Add(this,option);
   gPad->Modified(kTRUE);
}

//______________________________________________________________________________
void TObject::Browse(TBrowser *b)
{
   // Browse object. May be overridden for another default action

   //Inspect();
   TClass::AutoBrowse(this,b);
}

//______________________________________________________________________________
const char *TObject::ClassName() const
{
   // Returns name of class to which the object belongs.

   return IsA()->GetName();
}

//______________________________________________________________________________
TObject *TObject::Clone(const char *) const
{
   // Make a clone of an object using the Streamer facility.
   // If the object derives from TNamed, this function is called
   // by TNamed::Clone. TNamed::Clone uses the optional argument to set
   // a new name to the newly created object.
   //
   // If the object class has a DirectoryAutoAdd function, it will be
   // called at the end of the function with the parameter gDirectory.
   // This usually means that the object will be appended to the current
   // ROOT directory.

   if (gDirectory) return gDirectory->CloneObject(this);
   else            return 0;
}

//______________________________________________________________________________
Int_t TObject::Compare(const TObject *) const
{
   // Compare abstract method. Must be overridden if a class wants to be able
   // to compare itself with other objects. Must return -1 if this is smaller
   // than obj, 0 if objects are equal and 1 if this is larger than obj.

   AbstractMethod("Compare");
   return 0;
}

//______________________________________________________________________________
void TObject::Delete(Option_t *)
{
   // Delete this object. Typically called as a command via the interpreter.
   // Normally use "delete" operator when object has been allocated on the heap.

   if (IsOnHeap()) {
      // Delete object from CINT symbol table so it can not be used anymore.
      // CINT object are always on the heap.
      gInterpreter->DeleteGlobal(this);

      delete this;
   }
}


//______________________________________________________________________________
Int_t TObject::DistancetoPrimitive(Int_t, Int_t)
{
   // Computes distance from point (px,py) to the object.
   // This member function must be implemented for each graphics primitive.
   // This default function returns a big number (999999).

   // AbstractMethod("DistancetoPrimitive");
   return 999999;
}

//______________________________________________________________________________
void TObject::Draw(Option_t *option)
{
   // Default Draw method for all objects

   AppendPad(option);
}

//______________________________________________________________________________
void TObject::DrawClass() const
{
   // Draw class inheritance tree of the class to which this object belongs.
   // If a class B inherits from a class A, description of B is drawn
   // on the right side of description of A.
   // Member functions overridden by B are shown in class A with a blue line
   // crossing-out the corresponding member function.
   // The following picture is the class inheritance tree of class TPaveLabel:
   //Begin_Html
   /*
   <img src="gif/drawclass.gif">
   */
   //End_Html

   IsA()->Draw();
}

//______________________________________________________________________________
TObject *TObject::DrawClone(Option_t *option) const
{
   // Draw a clone of this object in the current pad

   TVirtualPad *pad    = gROOT->GetSelectedPad();
   TVirtualPad *padsav = gPad;
   if (pad) pad->cd();

   TObject *newobj = Clone();
   if (!newobj) return 0;
   if (pad) {
      if (strlen(option)) pad->GetListOfPrimitives()->Add(newobj,option);
      else                pad->GetListOfPrimitives()->Add(newobj,GetDrawOption());
      pad->Modified(kTRUE);
      pad->Update();
      if (padsav) padsav->cd();
      return newobj;
   }
   if (strlen(option))  newobj->Draw(option);
   else                 newobj->Draw(GetDrawOption());
   if (padsav) padsav->cd();

   return newobj;
}

//______________________________________________________________________________
void TObject::Dump() const
{
   // Dump contents of object on stdout.
   // Using the information in the object dictionary (class TClass)
   // each data member is interpreted.
   // If a data member is a pointer, the pointer value is printed
   //
   // The following output is the Dump of a TArrow object:
   //   fAngle                   0           Arrow opening angle (degrees)
   //   fArrowSize               0.2         Arrow Size
   //   fOption.*fData
   //   fX1                      0.1         X of 1st point
   //   fY1                      0.15        Y of 1st point
   //   fX2                      0.67        X of 2nd point
   //   fY2                      0.83        Y of 2nd point
   //   fUniqueID                0           object unique identifier
   //   fBits                    50331648    bit field status word
   //   fLineColor               1           line color
   //   fLineStyle               1           line style
   //   fLineWidth               1           line width
   //   fFillColor               19          fill area color
   //   fFillStyle               1001        fill area style

   if (sizeof(this) == 4)
      Printf("==> Dumping object at: 0x%08lx, name=%s, class=%s\n",(Long_t)this,GetName(),ClassName());
   else
      Printf("==> Dumping object at: 0x%016lx, name=%s, class=%s\n",(Long_t)this,GetName(),ClassName());

   TDumpMembers dm;
   ((TObject*)this)->ShowMembers(dm);
}

//______________________________________________________________________________
void TObject::Execute(const char *method, const char *params, Int_t *error)
{
   // Execute method on this object with the given parameter string, e.g.
   // "3.14,1,\"text\"".

   if (!IsA()) return;

   Bool_t must_cleanup = TestBit(kMustCleanup);

   gInterpreter->Execute(this, IsA(), method, params, error);

   if (gPad && must_cleanup) gPad->Modified();
}

//______________________________________________________________________________
void TObject::Execute(TMethod *method, TObjArray *params, Int_t *error)
{
   // Execute method on this object with parameters stored in the TObjArray.
   // The TObjArray should contain an argv vector like:
   //
   //  argv[0] ... argv[n] = the list of TObjString parameters

   if (!IsA()) return;

   Bool_t must_cleanup = TestBit(kMustCleanup);

   gInterpreter->Execute(this, IsA(), method, params, error);

   if (gPad && must_cleanup) gPad->Modified();
}


//______________________________________________________________________________
void TObject::ExecuteEvent(Int_t, Int_t, Int_t)
{
   // Execute action corresponding to an event at (px,py). This method
   // must be overridden if an object can react to graphics events.

   // AbstractMethod("ExecuteEvent");
}

//______________________________________________________________________________
TObject *TObject::FindObject(const char *) const
{
   // Must be redefined in derived classes.
   // This function is typycally used with TCollections, but can also be used
   // to find an object by name inside this object.

   return 0;
}

//______________________________________________________________________________
TObject *TObject::FindObject(const TObject *) const
{
   // Must be redefined in derived classes.
   // This function is typycally used with TCollections, but can also be used
   // to find an object inside this object.

   return 0;
}

//______________________________________________________________________________
Option_t *TObject::GetDrawOption() const
{
   // Get option used by the graphics system to draw this object.
   // Note that before calling object.GetDrawOption(), you must
   // have called object.Draw(..) before in the current pad.

   if (!gPad) return "";

   TListIter next(gPad->GetListOfPrimitives());
   TObject *obj;
   while ((obj = next())) {
      if (obj == this) return next.GetOption();
   }
   return "";
}

//______________________________________________________________________________
const char *TObject::GetName() const
{
   // Returns name of object. This default method returns the class name.
   // Classes that give objects a name should override this method.

   return IsA()->GetName();
}

//______________________________________________________________________________
const char *TObject::GetIconName() const
{
   // Returns mime type name of object. Used by the TBrowser (via TGMimeTypes
   // class). Override for class of which you would like to have different
   // icons for objects of the same class.

   return 0;
}

//______________________________________________________________________________
UInt_t TObject::GetUniqueID() const
{
   // Return the unique object id.

   return fUniqueID;
}

//______________________________________________________________________________
char *TObject::GetObjectInfo(Int_t px, Int_t py) const
{
   // Returns string containing info about the object at position (px,py).
   // This method is typically overridden by classes of which the objects
   // can report peculiarities for different positions.
   // Returned string will be re-used (lock in MT environment).

   if (!gPad) return (char*)"";
   static char info[64];
   Float_t x = gPad->AbsPixeltoX(px);
   Float_t y = gPad->AbsPixeltoY(py);
   snprintf(info,64,"x=%g, y=%g",gPad->PadtoX(x),gPad->PadtoY(y));
   return info;
}

//______________________________________________________________________________
const char *TObject::GetTitle() const
{
   // Returns title of object. This default method returns the class title
   // (i.e. description). Classes that give objects a title should override
   // this method.

   return IsA()->GetTitle();
}


//______________________________________________________________________________
Bool_t TObject::HandleTimer(TTimer *)
{
   // Execute action in response of a timer timing out. This method
   // must be overridden if an object has to react to timers.

   return kFALSE;
}

//______________________________________________________________________________
ULong_t TObject::Hash() const
{
   // Return hash value for this object.

   //return (ULong_t) this >> 2;
   const void *ptr = this;
   return TString::Hash(&ptr, sizeof(void*));
}

//______________________________________________________________________________
Bool_t TObject::InheritsFrom(const char *classname) const
{
   // Returns kTRUE if object inherits from class "classname".

   return IsA()->InheritsFrom(classname);
}

//______________________________________________________________________________
Bool_t TObject::InheritsFrom(const TClass *cl) const
{
   // Returns kTRUE if object inherits from TClass cl.

   return IsA()->InheritsFrom(cl);
}

//______________________________________________________________________________
void TObject::Inspect() const
{
   // Dump contents of this object in a graphics canvas.
   // Same action as Dump but in a graphical form.
   // In addition pointers to other objects can be followed.
   //
   // The following picture is the Inspect of a histogram object:
   //Begin_Html
   /*
   <img src="gif/hpxinspect.gif">
   */
   //End_Html

   gGuiFactory->CreateInspectorImp(this, 400, 200);
}

//______________________________________________________________________________
Bool_t TObject::IsFolder() const
{
   // Returns kTRUE in case object contains browsable objects (like containers
   // or lists of other objects).

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TObject::IsEqual(const TObject *obj) const
{
   // Default equal comparison (objects are equal if they have the same
   // address in memory). More complicated classes might want to override
   // this function.

   return obj == this;
}

//______________________________________________________________________________
void TObject::ls(Option_t *option) const
{
   // The ls function lists the contents of a class on stdout. Ls output
   // is typically much less verbose then Dump().

   TROOT::IndentLevel();
   cout <<"OBJ: " << IsA()->GetName() << "\t" << GetName() << "\t" << GetTitle() << " : ";
   cout << Int_t(TestBit(kCanDelete));  
   if (strstr(option,"noaddr")==0) {
      cout <<" at: "<< this ;
   }
   cout << endl;
}

//______________________________________________________________________________
Bool_t TObject::Notify()
{
   // This method must be overridden to handle object notifcation.

   return kFALSE;
}

//______________________________________________________________________________
void TObject::Paint(Option_t *)
{
   // This method must be overridden if a class wants to paint itself.
   // The difference between Paint() and Draw() is that when a object
   // draws itself it is added to the display list of the pad in
   // which it is drawn (and automatically redrawn whenever the pad is
   // redrawn). While paint just draws the object without adding it to
   // the pad display list.

   // AbstractMethod("Paint");
}

//______________________________________________________________________________
void TObject::Pop()
{
   // Pop on object drawn in a pad to the top of the display list. I.e. it
   // will be drawn last and on top of all other primitives.

   if (!gPad) return;

   if (this == gPad->GetListOfPrimitives()->Last()) return;

   TListIter next(gPad->GetListOfPrimitives());
   TObject *obj;
   while ((obj = next()))
      if (obj == this) {
         char *opt = StrDup(next.GetOption());
         gPad->GetListOfPrimitives()->Remove((TObject*)this);
         gPad->GetListOfPrimitives()->AddLast(this, opt);
         gPad->Modified();
         delete [] opt;
         return;
      }
}

//______________________________________________________________________________
void TObject::Print(Option_t *) const
{
   // This method must be overridden when a class wants to print itself.

   cout <<"OBJ: " << IsA()->GetName() << "\t" << GetName() << "\t" << GetTitle() << endl;
}

//______________________________________________________________________________
Int_t TObject::Read(const char *name)
{
   // Read contents of object with specified name from the current directory.
   // First the key with the given name is searched in the current directory,
   // next the key buffer is deserialized into the object.
   // The object must have been created before via the default constructor.
   // See TObject::Write().

   if (gDirectory) return gDirectory->ReadTObject(this,name);
   else            return 0;
}

//______________________________________________________________________________
void TObject::RecursiveRemove(TObject *)
{
   // Recursively remove this object from a list. Typically implemented
   // by classes that can contain mulitple references to a same object.

}


//______________________________________________________________________________
void TObject::SaveAs(const char *filename, Option_t *option) const
{
   // Save this object in the file specified by filename.
   // if (filename contains ".root" the object is saved in filename (root binary file)
   // if (filename contains ".xml"  the object is saved in filename (xml ascii file)
   //  otherwise the object is written to filename as a CINT/C++ script.
   //  The C++ code to rebuild this object is generated via SavePrimitive().
   //  The option parameter is passed to SavePrimitive. By default it is
   //  en empty string. It can be used to specify the Draw option
   //  in the code generated by SavePrimitive.
   // The function is available via the object context menu.

   //==============Save object as a root file===============================
   if (filename && strstr(filename,".root")) {
      if (gDirectory) gDirectory->SaveObjectAs(this,filename,"");
      return;
   }

   //==============Save object as a XML file================================
   if (filename && strstr(filename,".xml")) {
      if (gDirectory) gDirectory->SaveObjectAs(this,filename,"");
      return;
   }

   //==============Save as a C++ CINT file================================
   char *fname = 0;
   if (filename && strlen(filename) > 0) {
      fname = (char*)filename;
   } else {
      fname = Form("%s.C", GetName());
   }
   ofstream out;
   out.open(fname, ios::out);
   if (!out.good ()) {
      Error("SaveAs", "cannot open file: %s", fname);
      return;
   }
   out <<"{"<<endl;
   out <<"//========= Macro generated from object: "<<GetName()<<"/"<<GetTitle()<<endl;
   out <<"//========= by ROOT version"<<gROOT->GetVersion()<<endl;
   ((TObject*)this)->SavePrimitive(out,option);
   out <<"}"<<endl;
   out.close();
   Info("SaveAs", "C++ Macro file: %s has been generated", fname);
}

//______________________________________________________________________________
void TObject::SavePrimitive(ostream &out, Option_t * /*= ""*/)
{
   // Save a primitive as a C++ statement(s) on output stream "out".

   out << "//Primitive: " << GetName() << "/" << GetTitle()
       <<". You must implement " << ClassName() << "::SavePrimitive" << endl;
}

//______________________________________________________________________________
void TObject::SetDrawOption(Option_t *option)
{
   // Set drawing option for object. This option only affects
   // the drawing style and is stored in the option field of the
   // TObjOptLink supporting a TPad's primitive list (TList).
   // Note that it does not make sense to call object.SetDrawOption(option)
   // before having called object.Draw().

   if (!gPad || !option) return;

   TListIter next(gPad->GetListOfPrimitives());
   delete gPad->FindObject("Tframe");
   TObject *obj;
   while ((obj = next()))
      if (obj == this) {
         next.SetOption(option);
         return;
      }
}

//______________________________________________________________________________
void TObject::SetBit(UInt_t f, Bool_t set)
{
   // Set or unset the user status bits as specified in f.

   if (set)
      SetBit(f);
   else
      ResetBit(f);
}

//______________________________________________________________________________
void TObject::SetUniqueID(UInt_t uid)
{
   // Set the unique object id.

   fUniqueID = uid;
}

//______________________________________________________________________________
void TObject::UseCurrentStyle()
{
   // Set current style settings in this object
   // This function is called when either TCanvas::UseCurrentStyle
   // or TROOT::ForceStyle have been invoked.
}

//______________________________________________________________________________
Int_t TObject::Write(const char *name, Int_t option, Int_t bufsize) const
{
   // Write this object to the current directory.
   // The data structure corresponding to this object is serialized.
   // The corresponding buffer is written to the current directory
   // with an associated key with name "name".
   //
   // Writing an object to a file involves the following steps:
   //
   //  -Creation of a support TKey object in the current directory.
   //   The TKey object creates a TBuffer object.
   //
   //  -The TBuffer object is filled via the class::Streamer function.
   //
   //  -If the file is compressed (default) a second buffer is created to
   //   hold the compressed buffer.
   //
   //  -Reservation of the corresponding space in the file by looking
   //   in the TFree list of free blocks of the file.
   //
   //  -The buffer is written to the file.
   //
   //  Bufsize can be given to force a given buffer size to write this object.
   //  By default, the buffersize will be taken from the average buffer size
   //  of all objects written to the current file so far.
   //
   //  If a name is specified, it will be the name of the key.
   //  If name is not given, the name of the key will be the name as returned
   //  by GetName().
   //
   //  The option can be a combination of:
   //    kSingleKey, kOverwrite or kWriteDelete
   //  Using the kOverwrite option a previous key with the same name is
   //  overwritten. The previous key is deleted before writing the new object.
   //  Using the kWriteDelete option a previous key with the same name is
   //  deleted only after the new object has been written. This option
   //  is safer than kOverwrite but it is slower.
   //  The kSingleKey option is only used by TCollection::Write() to write
   //  a container with a single key instead of each object in the container
   //  with its own key.
   //
   //  An object is read from the file into memory via TKey::Read() or
   //  via TObject::Read().
   //
   //  The function returns the total number of bytes written to the file.
   //  It returns 0 if the object cannot be written.

   TString opt = "";
   if (option & kSingleKey)   opt += "SingleKey";
   if (option & kOverwrite)   opt += "OverWrite";
   if (option & kWriteDelete) opt += "WriteDelete";

   if (gDirectory) return gDirectory->WriteTObject(this,name,opt.Data(),bufsize);
   else {
      const char *objname = "no name specified";
      if (name) objname = name;
      else objname = GetName();
      Error("Write","The current directory (gDirectory) is null. The object (%s) has not been written.",objname);
      return 0;
   }
}

//______________________________________________________________________________
Int_t TObject::Write(const char *name, Int_t option, Int_t bufsize)
{
   // Write this object to the current directory. For more see the
   // const version of this method.

   return ((const TObject*)this)->Write(name, option, bufsize);
}

//______________________________________________________________________________
void TObject::Streamer(TBuffer &R__b)
{
   // Stream an object of class TObject.

   if (IsA()->CanIgnoreTObjectStreamer()) return;
   UShort_t pidf;
   if (R__b.IsReading()) {
      R__b.SkipVersion(); // Version_t R__v = R__b.ReadVersion(); if (R__v) { }
      R__b >> fUniqueID;
      R__b >> fBits;
      fBits |= kIsOnHeap;  // by definition de-serialized object is on heap
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
      if (!TestBit(kIsReferenced)) {
         R__b << fUniqueID;
         R__b << fBits;
      } else {
         //if the object is referenced, we must save its address/file_pid
         UInt_t uid = fUniqueID & 0xffffff;
         R__b << uid;
         R__b << fBits;
         TProcessID *pid = TProcessID::GetProcessWithUID(fUniqueID,this);
         //add uid to the TRefTable if there is one
         TRefTable *table = TRefTable::GetRefTable();
         if(table) table->Add(uid, pid);
         pidf = R__b.WriteProcessID(pid);
         R__b << pidf;
      }
   }
}

//______________________________________________________________________________

//---- error handling ----------------------------------------------------------

//______________________________________________________________________________
void TObject::DoError(int level, const char *location, const char *fmt, va_list va) const
{
   // Interface to ErrorHandler (protected).

   const char *classname = "UnknownClass";
   if (TROOT::Initialized())
      classname = ClassName();

   ::ErrorHandler(level, Form("%s::%s", classname, location), fmt, va);
}

//______________________________________________________________________________
void TObject::Info(const char *location, const char *va_(fmt), ...) const
{
   // Issue info message. Use "location" to specify the method where the
   // warning occured. Accepts standard printf formatting arguments.

   va_list ap;
   va_start(ap, va_(fmt));
   DoError(kInfo, location, va_(fmt), ap);
   va_end(ap);
}

//______________________________________________________________________________
void TObject::Warning(const char *location, const char *va_(fmt), ...) const
{
   // Issue warning message. Use "location" to specify the method where the
   // warning occured. Accepts standard printf formatting arguments.

   va_list ap;
   va_start(ap, va_(fmt));
   DoError(kWarning, location, va_(fmt), ap);
   va_end(ap);
   if (TROOT::Initialized())
      gROOT->Message(1001, this);
}

//______________________________________________________________________________
void TObject::Error(const char *location, const char *va_(fmt), ...) const
{
   // Issue error message. Use "location" to specify the method where the
   // error occured. Accepts standard printf formatting arguments.

   va_list ap;
   va_start(ap, va_(fmt));
   DoError(kError, location, va_(fmt), ap);
   va_end(ap);
   if (TROOT::Initialized())
      gROOT->Message(1002, this);
}

//______________________________________________________________________________
void TObject::SysError(const char *location, const char *va_(fmt), ...) const
{
   // Issue system error message. Use "location" to specify the method where
   // the system error occured. Accepts standard printf formatting arguments.

   va_list ap;
   va_start(ap, va_(fmt));
   DoError(kSysError, location, va_(fmt), ap);
   va_end(ap);
   if (TROOT::Initialized())
      gROOT->Message(1003, this);
}

//______________________________________________________________________________
void TObject::Fatal(const char *location, const char *va_(fmt), ...) const
{
   // Issue fatal error message. Use "location" to specify the method where the
   // fatal error occured. Accepts standard printf formatting arguments.

   va_list ap;
   va_start(ap, va_(fmt));
   DoError(kFatal, location, va_(fmt), ap);
   va_end(ap);
   if (TROOT::Initialized())
      gROOT->Message(1004, this);
}

//______________________________________________________________________________
void TObject::AbstractMethod(const char *method) const
{
   // Use this method to implement an "abstract" method that you don't
   // want to leave purely abstract.

   Warning(method, "this method must be overridden!");
}

//______________________________________________________________________________
void TObject::MayNotUse(const char *method) const
{
   // Use this method to signal that a method (defined in a base class)
   // may not be called in a derived class (in principle against good
   // design since a child class should not provide less functionality
   // than its parent, however, sometimes it is necessary).

   Warning(method, "may not use this method");
}

//______________________________________________________________________________
void TObject::Obsolete(const char *method, const char *asOfVers, const char *removedFromVers) const
{
   // Use this method to declare a method obsolete. Specify as of which version
   // the method is obsolete and as from which version it will be removed.
   
   const char *classname = "UnknownClass";
   if (TROOT::Initialized())
      classname = ClassName();
   
   ::Obsolete(Form("%s::%s", classname, method), asOfVers, removedFromVers);
}



//----------------- Static data members access ---------------------------------

//______________________________________________________________________________
Bool_t TObject::GetObjectStat()
{
   // Get status of object stat flag.

   return fgObjectStat;
}
//______________________________________________________________________________
void TObject::SetObjectStat(Bool_t stat)
{
   // Turn on/off tracking of objects in the TObjectTable.

   fgObjectStat = stat;
}

//______________________________________________________________________________
Long_t TObject::GetDtorOnly()
{
   //return destructor only flag
   return fgDtorOnly;
}

//______________________________________________________________________________
void TObject::SetDtorOnly(void *obj)
{
   //set destructor only flag
   fgDtorOnly = (Long_t) obj;
}

//______________________________________________________________________________
void TObject::operator delete(void *ptr)
{
   //operator delete
   if ((Long_t) ptr != fgDtorOnly)
      TStorage::ObjectDealloc(ptr);
   else
      fgDtorOnly = 0;
}

//______________________________________________________________________________
void TObject::operator delete[](void *ptr)
{
   //operator delete []
   if ((Long_t) ptr != fgDtorOnly)
      TStorage::ObjectDealloc(ptr);
   else
      fgDtorOnly = 0;
}

#ifdef R__PLACEMENTDELETE
//______________________________________________________________________________
void TObject::operator delete(void *ptr, void *vp)
{
   // Only called by placement new when throwing an exception.

   TStorage::ObjectDealloc(ptr, vp);
}

//______________________________________________________________________________
void TObject::operator delete[](void *ptr, void *vp)
{
   // Only called by placement new[] when throwing an exception.

   TStorage::ObjectDealloc(ptr, vp);
}
#endif
