// @(#)root/base:$Name:  $:$Id: TObject.h,v 1.9 2001/02/03 15:42:55 rdm Exp $
// Author: Rene Brun   26/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TObject
#define ROOT_TObject


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

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#ifndef ROOT_Varargs
#include "Varargs.h"
#endif
#ifndef ROOT_TStorage
#include "TStorage.h"
#endif

#ifdef WIN32
#undef RemoveDirectory
#endif

#if defined(R__ANSISTREAM)
#include <iosfwd>
using namespace std;
#elif R__MWERKS
template <class charT> class ios_traits;
template <class charT, class traits> class basic_ofstream;
typedef basic_ofstream<char, ios_traits<char> > ofstream;
#else
class ofstream;
#endif

class TList;
class TBrowser;
class TBuffer;
class TObjArray;
class TMethod;
class TTimer;


//----- Global bits (can be set for any object and should not be reused).
//----- Bits 0 - 13 are reserved as global bits. Bits 14 - 23 can be used
//----- in different class hierarchies (make sure there is no overlap in
//----- any given hierarchy).
enum EObjBits {
   kCanDelete        = BIT(0),   // if object in a list can be deleted
   kMustCleanup      = BIT(3),   // if object destructor must call RecursiveRemove()
   kObjInCanvas      = BIT(3),   // for backward compatibility only, use kMustCleanup
   kCannotPick       = BIT(6),   // if object in a pad cannot be picked
   kInvalidObject    = BIT(13)   // if object ctor succeeded but object should not be used
};


class TObject {

private:
   UInt_t         fUniqueID;   //object unique identifier
   UInt_t         fBits;       //bit field status word

   static Long_t  fgDtorOnly;    //object for which to call dtor only (i.e. no delete)
   static Bool_t  fgObjectStat;  //if true keep track of objects in TObjectTable

protected:
   void MakeZombie() { fBits |= kZombie; }
   void DoError(int level, const char *location, const char *fmt, va_list va) const;

public:
   //----- Private bits, clients can only test but not change them
   enum {
      kIsOnHeap      = 0x01000000,    // object is on heap
      kNotDeleted    = 0x02000000,    // object has not been deleted
      kZombie        = 0x04000000,    // object ctor failed
      kBitMask       = 0x00ffffff
   };

   //----- Write() options
   enum {
      kSingleKey     = BIT(0),        // write collection with single key
      kOverwrite     = BIT(1)         // overwrite existing object with same name
   };

   TObject();
   TObject(const TObject &object);
   TObject &operator=(const TObject &rhs);
   virtual ~TObject();

   void                AppendPad(Option_t *option="");
   virtual void        Browse(TBrowser *b);
   virtual const char *ClassName() const;
   virtual void        Clear(Option_t * /*option*/ ="") { }
   virtual TObject    *Clone(const char *newname="") const;
   virtual Int_t       Compare(const TObject *obj) const;
   virtual void        Copy(TObject &object);
   virtual void        Delete(Option_t *option=""); // *MENU*
   virtual Int_t       DistancetoPrimitive(Int_t px, Int_t py);
   virtual void        Draw(Option_t *option="");
   virtual void        DrawClass() const; // *MENU*
   virtual void        DrawClone(Option_t *option="") const; // *MENU*
   virtual void        Dump() const; // *MENU*
   virtual void        Execute(const char *method,  const char *params);
   virtual void        Execute(TMethod *method, TObjArray *params);
   virtual void        ExecuteEvent(Int_t event, Int_t px, Int_t py);
   virtual TObject    *FindObject(const char *name) const;
   virtual TObject    *FindObject(const TObject *obj) const;
   virtual Option_t   *GetDrawOption() const;
   virtual UInt_t      GetUniqueID() const;
   virtual const char *GetName() const;
   virtual const char *GetIconName() const;
   virtual Option_t   *GetOption() const { return ""; }
   virtual char       *GetObjectInfo(Int_t px, Int_t py) const;
   virtual const char *GetTitle() const;
   virtual Bool_t      HandleTimer(TTimer *timer);
   virtual ULong_t     Hash() const;
   virtual Bool_t      InheritsFrom(const char *classname) const;
   virtual Bool_t      InheritsFrom(const TClass *cl) const;
   virtual void        Inspect() const; // *MENU*
   virtual Bool_t      IsFolder() const;
   virtual Bool_t      IsEqual(const TObject *obj) const;
   virtual Bool_t      IsSortable() const { return kFALSE; }
           Bool_t      IsOnHeap() const { return TestBit(kIsOnHeap); }
           Bool_t      IsZombie() const { return TestBit(kZombie); }
   virtual Bool_t      Notify();
   virtual void        ls(Option_t *option="") const;
   virtual void        Paint(Option_t *option="");
   virtual void        Pop();
   virtual void        Print(Option_t *option="") const;
   virtual Int_t       Read(const char *name);
   virtual void        RecursiveRemove(TObject *obj);
   virtual void        SavePrimitive(ofstream &out, Option_t *option);
   virtual void        SetDrawOption(Option_t *option="");  // *MENU*
   virtual void        SetUniqueID(UInt_t uid);
   virtual void        UseCurrentStyle();
   virtual Int_t       Write(const char *name=0, Int_t option=0, Int_t bufsize=0);

   //----- operators
   void    *operator new(size_t sz) { return TStorage::ObjectAlloc(sz); }
   void    *operator new(size_t sz, void *vp) { return TStorage::ObjectAlloc(sz, vp); }
   void     operator delete(void *ptr);
#ifdef R__PLACEMENTDELETE
   void     operator delete(void *ptr, void *vp);
#endif

   //----- bit manipulation
   void     SetBit(UInt_t f, Bool_t set);
   void     SetBit(UInt_t f) { fBits |= f & kBitMask; }
   void     ResetBit(UInt_t f) { fBits &= ~(f & kBitMask); }
   Bool_t   TestBit(UInt_t f) const { return (Bool_t) ((fBits & f) != 0); }
   void     InvertBit(UInt_t f) { fBits ^= f & kBitMask; }

   //---- error handling
   void     Warning(const char *method, const char *msgfmt, ...) const;
   void     Error(const char *method, const char *msgfmt, ...) const;
   void     SysError(const char *method, const char *msgfmt, ...) const;
   void     Fatal(const char *method, const char *msgfmt, ...) const;

   void     AbstractMethod(const char *method) const;
   void     MayNotUse(const char *method) const;

   //---- static functions
   static Long_t    GetDtorOnly();
   static void      SetDtorOnly(void *obj);
   static Bool_t    GetObjectStat();
   static void      SetObjectStat(Bool_t stat);

   ClassDef(TObject,1)  //Basic ROOT object
};

#ifndef ROOT_TBuffer
#include "TBuffer.h"
#endif

#endif
