// @(#)root/base:$Name:  $:$Id: TObject.cxx,v 1.35 2002/02/07 07:31:50 brun Exp $
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
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <string.h>
#if !defined(WIN32) && !defined(__MWERKS__) && !defined(R__SOLARIS)
#include <strings.h>
#endif
#include <stdlib.h>
#include <stdio.h>

#include "Riostream.h"
#include "TObject.h"
#include "TFile.h"
#include "TDirectory.h"
#include "TClass.h"
#include "TGuiFactory.h"
#include "TKey.h"
#include "TDataMember.h"
#include "TMethod.h"
#include "TDataType.h"
#include "TRealData.h"
#include "TROOT.h"
#include "TError.h"
#include "TObjectTable.h"
#include "TVirtualPad.h"
#include "TInterpreter.h"
#include "TMemberInspector.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TDatime.h"
#include "TProcessID.h"
#include "TMath.h"


Long_t TObject::fgDtorOnly = 0;
Bool_t TObject::fgObjectStat = kTRUE;


class TDumpMembers : public TMemberInspector {

public:
   TDumpMembers() { }
   void Inspect(TClass *cl, const char *parent, const char *name, const void *addr);
};

//______________________________________________________________________________
void TDumpMembers::Inspect(TClass *cl, const char *pname, const char *mname, const void *add)
{
   // Print value of member mname.
   //
   // This method is called by the ShowMembers() method for each
   // data member when object.Dump() is invoked.
   //
   //    cl    is the pointer to the current class
   //    pname is the parent name (in case of composed objects)
   //    mname is the data member name
   //    add   is the data member address

   const Int_t kvalue = 30;
   const Int_t ktitle = 42;
   const Int_t kline  = 1024;
   Int_t cdate = 0;
   Int_t ctime = 0;
   UInt_t *cdatime = 0;
   char line[kline];
   TDataMember *member = cl->GetDataMember(mname);
   if (!member) return;
   TDataType *membertype = member->GetDataType();
   Bool_t isdate = kFALSE;
   if (strcmp(member->GetName(),"fDatime") == 0 && strcmp(member->GetTypeName(),"UInt_t") == 0) {
      isdate = kTRUE;
   }

   Int_t i;
   for (i = 0;i < kline; i++) line[i] = ' ';
   line[kline-1] = 0;
   sprintf(line,"%s%s ",pname,mname);
   i = strlen(line); line[i] = ' ';

   // Encode data value or pointer value
   char *pointer = (char*)add;
   char **ppointer = (char**)(pointer);

   if (member->IsaPointer()) {
      char **p3pointer = (char**)(*ppointer);
      if (!p3pointer)
         sprintf(&line[kvalue],"->0");
      else if (!member->IsBasic())
         sprintf(&line[kvalue],"->%lx ", (Long_t)p3pointer);
      else if (membertype) {
         if (!strcmp(membertype->GetTypeName(), "char")) {
            i = strlen(*ppointer);
            if (kvalue+i >= kline) i=kline-kvalue;
            strncpy(&line[kvalue],*ppointer,i);
            line[kvalue+i] = 0;
         } else {
            strcpy(&line[kvalue], membertype->AsString(p3pointer));
         }
      } else if (!strcmp(member->GetFullTypeName(), "char*") ||
                 !strcmp(member->GetFullTypeName(), "const char*")) {
         i = strlen(*ppointer);
         if (kvalue+i >= kline) i=kline-kvalue;
         strncpy(&line[kvalue],*ppointer,i);
         line[kvalue+i] = 0;
      } else {
         sprintf(&line[kvalue],"->%lx ", (Long_t)p3pointer);
      }
   } else if (membertype)
       if (isdate) {
          cdatime = (UInt_t*)pointer;
          TDatime::GetDateTime(cdatime[0],cdate,ctime);
          sprintf(&line[kvalue],"%d/%d",cdate,ctime);
       } else {
         strcpy(&line[kvalue], membertype->AsString(pointer));
       }
   else
      sprintf(&line[kvalue],"->%lx ", (Long_t)pointer);

   // Encode data member title
   if (isdate == kFALSE && strcmp(member->GetFullTypeName(), "char*") &&
       strcmp(member->GetFullTypeName(), "const char*")) {
      i = strlen(&line[0]); line[i] = ' ';
      Int_t lentit = strlen(member->GetTitle());
      if (lentit > 250-ktitle) lentit = 250-ktitle;
      strncpy(&line[ktitle],member->GetTitle(),lentit);
      line[ktitle+lentit] = 0;
   }
   Printf("%s", line);
}


ClassImp(TObject)

//______________________________________________________________________________
TObject::TObject()
{
   // TObject constructor. It sets the two data words of TObject to their
   // initial values. The unique ID is set to 0 and the status word is
   // set depending if the object is created on the stack or allocated
   // on the heap. Of the status word the high 8 bits are reserved for
   // system usage and the low 24 bits are user settable. Depending on
   // the ROOT environment variable "Root.MemStat" (see TEnv.h) the object
   // is added to the global TObjectTable for bookkeeping.

   fUniqueID = 0;
   fBits     = kNotDeleted;
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
   }
   return *this;
}

//______________________________________________________________________________
void TObject::Copy(TObject &obj)
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
      if (!gROOT->GetMakeDefCanvas()) return;
      (gROOT->GetMakeDefCanvas())();
   }
   if (!gPad->IsEditable()) return;
   SetBit(kMustCleanup);
   gPad->GetListOfPrimitives()->Add(this,option);
   gPad->Modified(kTRUE);
}

//______________________________________________________________________________
void TObject::Browse(TBrowser *b)
{
   // Browse object. May be overridden for  other default action

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
   // a new name to the new created object.

   // if no default ctor return immediately (error issued by New())
   TObject *newobj = (TObject *)IsA()->New();
   if (!newobj) return 0;

   //create a buffer where the object will be streamed
   TFile *filsav = gFile;
   gFile = 0;
   const Int_t bufsize = 10000;
   TBuffer *buffer = new TBuffer(TBuffer::kWrite,bufsize);
   buffer->MapObject(this);  //register obj in map to handle self reference
   ((TObject*)this)->Streamer(*buffer);

   // read new object from buffer
   buffer->SetReadMode();
   buffer->ResetMap();
   buffer->SetBufferOffset(0);
   buffer->MapObject(newobj);  //register obj in map to handle self reference
   newobj->Streamer(*buffer);
   newobj->ResetBit(kIsReferenced);
   gFile = filsav;

   delete buffer;
   return newobj;
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

   TObject *newobj = Clone();
   if (!newobj) return 0;
   TVirtualPad *pad = gROOT->GetSelectedPad();
   if (pad) {
      if (strlen(option)) pad->GetListOfPrimitives()->Add(newobj,option);
      else                pad->GetListOfPrimitives()->Add(newobj,GetDrawOption());
      pad->Modified(kTRUE);
      pad->Update();
      return newobj;
   }
   if (strlen(option))  newobj->Draw(option);
   else                 newobj->Draw(GetDrawOption());
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

   char parent[256];
   parent[0] = 0;
   TDumpMembers dm;
   ((TObject*)this)->ShowMembers(dm, parent);
}

//______________________________________________________________________________
void TObject::Execute(const char *method, const char *params, int* error)
{
   // Execute method on this object with the given parameter string, e.g.
   // "3.14,1,\"text\"".

   if (!IsA()) return;

   gInterpreter->Execute(this, IsA(), method, params, error);

   if (gPad && TestBit(kMustCleanup)) gPad->Modified();
}

//______________________________________________________________________________
void TObject::Execute(TMethod *method, TObjArray *params, int* error)
{
   // Execute method on this object with parameters stored in the TObjArray.
   // The TObjArray should contain an argv vector like:
   //
   //  argv[0] ... argv[n] = the list of TObjString parameters

   if (!IsA()) return;

   gInterpreter->Execute(this, IsA(), method, params, error);

   if (gPad && TestBit(kMustCleanup)) gPad->Modified();
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
// must be redefined in derived classes
// this function is typycally used with TCollections, but can also be used
// to find an object by name inside this object

   return 0;
}

//______________________________________________________________________________
TObject *TObject::FindObject(const TObject *) const
{
// must be redefined in derived classes
// this function is typycally used with TCollections, but can also be used
// to find an object inside this object

   return 0;
}

//______________________________________________________________________________
Option_t *TObject::GetDrawOption() const
{
   // Get option used by the graphics system to draw this object.

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
   sprintf(info,"x=%.3g, y=%.3g",gPad->PadtoX(x),gPad->PadtoY(y));
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
   return TMath::Hash(&ptr, sizeof(void*));
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

#ifdef WIN32
#ifdef GDK_WIN32
   gROOT->ProcessLine(Form("TInspectCanvas::Inspector((TObject *)0x%lx);",(Long_t)this));
#else
   gGuiFactory->CreateInspectorImp(this, 400, 200);
#endif
#else
   gROOT->ProcessLine(Form("TInspectCanvas::Inspector((TObject *)0x%lx);",(Long_t)this));
#endif
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
void TObject::ls(Option_t *) const
{
   // The ls function lists the contents of a class on stdout. Ls output
   // is typically much less verbose then Dump().

   TROOT::IndentLevel();
   cout <<"OBJ: " << IsA()->GetName() << "\t" << GetName() << "\t" << GetTitle() << " : "
        << Int_t(TestBit(kCanDelete)) << endl;
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

   if (!gFile) { Error("Read","No file open"); return 0; }
   TKey *key = (TKey*)gDirectory->GetListOfKeys()->FindObject(name);
   if (!key)   { Error("Read","Key not found"); return 0; }
   return key->Read(this);
}

//______________________________________________________________________________
void TObject::RecursiveRemove(TObject *)
{
   // Recursively remove this object from a list. Typically implemented
   // by classes that can contain mulitple references to a same object.

}

//______________________________________________________________________________
void TObject::SavePrimitive(ofstream &out, Option_t *)
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
Int_t TObject::Write(const char *name, Int_t option, Int_t bufsize)
{
   // Write this object to the current directory
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
   //    kSingleKey and kOverwrite
   //  Using the kOverwrite option a previous key with the same name is
   //  overwritten.
   //  The kSingleKey option is only used by TCollection::Write() to write
   //  a container with a single key instead of each object in the container
   //  with its own key.
   //
   //  An object is read from the file into memory via TKey::Read() or
   //  via TObject::Read().

   if (!gFile) {
      Error("Write","No file open");
      return 0;
   }
   if (!gFile->IsWritable()) {
      Error("Write","File %s is not writable", gFile->GetName());
      return 0;
   }

   TKey *key;
   Int_t bsize = bufsize;
   if (!bsize) bsize = gFile->GetBestBuffer();

   const char *oname;
   if (name && *name)
      oname = name;
   else
      oname = GetName();

   // Remove trailing blanks in object name
   Int_t nch = strlen(oname);
   char *newName = 0;
   if (oname[nch-1] == ' ') {
      newName = new char[nch+1];
      strcpy(newName,oname);
      for (Int_t i=0;i<nch;i++) {
         if (newName[nch-i-1] != ' ') break;
         newName[nch-i-1] = 0;
      }
      oname = newName;
   }

   if ((option & kOverwrite)) {
      key = (TKey*)gDirectory->GetListOfKeys()->FindObject(oname);
      if (key) {
         key->Delete();
         delete key;
      }
   }
   key = new TKey(this, oname, bsize);
   if (newName) delete [] newName;

   if (!key->GetSeekKey()) {
      gDirectory->GetListOfKeys()->Remove(key);
      delete key;
      return 0;
   }
   gFile->SumBuffer(key->GetObjlen());
   return key->WriteFile(0);
}

//______________________________________________________________________________
void TObject::Streamer(TBuffer &R__b)
{
   // Stream an object of class TObject.

   if (IsA()->CanIgnoreTObjectStreamer()) return;
   UShort_t pidf;
   TFile *file = (TFile*)R__b.GetParent();
   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(); if (R__v) { }
      R__b >> fUniqueID;
      R__b >> fBits;
      fBits |= kIsOnHeap;  // by definition de-serialized object is on heap
      //if the object is referenced, we must read its old address
      //and store it in the ProcessID map in gROOT
      if (!TestBit(kIsReferenced)) return;
      R__b >> pidf;
      TProcessID *pid = TProcessID::ReadProcessID(pidf,file);
      if (pid) pid->PutObjectWithID(this);
   } else {
      R__b.WriteVersion(TObject::IsA());
      R__b << fUniqueID;
      R__b << fBits;
      //if the object is referenced, we must save its address/file_pid
      if (!TestBit(kIsReferenced)) return;
      pidf = 0;
      R__b << pidf;
   }
}

//______________________________________________________________________________

//---- error handling ----------------------------------------------------------

//______________________________________________________________________________
void TObject::DoError(int level, const char *location, const char *fmt, va_list va) const
{
   // Interface to ErrorHandler (protected).

   ::ErrorHandler(level, Form("%s::%s", ClassName(), location), fmt, va);
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
   gROOT->Message(1001,(TObject*)this);
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
   gROOT->Message(1002,(TObject*)this);
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
   gROOT->Message(1003,(TObject*)this);
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
   gROOT->Message(1004,(TObject*)this);
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
   return fgDtorOnly;
}

//______________________________________________________________________________
void TObject::SetDtorOnly(void *obj)
{
   fgDtorOnly = (Long_t) obj;
}

//______________________________________________________________________________
void TObject::operator delete(void *ptr)
{
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
#endif
