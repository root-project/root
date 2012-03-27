// @(#)root/table:$Id$
// Author: Valery Fine(fine@bnl.gov)   01/03/2001

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2001 [BNL] Brookhaven National Laboratory.              *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFileIter
#define ROOT_TFileIter

///////////////////////////////////////////////////////////////////////////
//                                                                       //
// Class to iterate (read / write ) the events written to /from TFile.   //
//                                                                       //
//  - set the current internal cursor directly by different means        //
//  - set the current cursor to the "next" position if available         //
//  - gain extra information of the TKey object at the current position  //
//  - read TObject object from the TFile defined by TKey at the current  //
//         position                                                      //
//                                                                       //
//  - Read "next" object from the file                                   //
//  - n-th object from the file                                          //
//  - object that is in n object on the file                             //
//  - read current object                                                //
//  - return the name of the key of the current object                   //
//  - return the current position                                        //
//  - set the current position by the absolute position number           //
//  - set the current position by relative position number               //
//  - get the number of keys in the file                                 //
//                                                                       //
// The event is supposed to assign an unique ID in form of               //
//                                                                       //
// TKey name ::= event Id ::= eventName "." run_number "." event_number  //
//                                                                       //
//                                                                       //
// and stored as the TKey name of the object written                     //
//                                                                       //
//  author Valeri Fine                                                   //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TString
#include "TString.h"
#endif
#ifndef ROOT_TIterator
#include "TIterator.h"
#endif
#ifndef ROOT_TList
#include "TList.h"
#endif
#ifndef ROOT_TFile
#include "TFile.h"
#endif


class TFileIter : public TListIter {

private:

   TFileIter *fNestedIterator;   //! The inner TFidrectory interator;

   virtual TIterator &operator=(const TIterator &) { return *this; }
   virtual Bool_t operator!=(const TIterator &it) const { return TListIter::operator!=(it);}
   
protected:
   TDirectory   *fRootFile;       // TDirectory/TFile to be iterated over
   TString  fEventName;           // current key name
   UInt_t   fRunNumber;           // current "run number"
   UInt_t   fEventNumber;         // current "event number"
   Int_t    fCursorPosition;      // the position of the current key in the sorted TKey list
   Bool_t   fOwnTFile;            // Bit whether this classs creates TFile on its own to delete

   void     Initialize();
   TObject *ReadObj(const TKey *key) const;
   TKey    *NextEventKey(UInt_t eventNumber=UInt_t(-1), UInt_t runNumber=UInt_t(-1), const char *name="*");

public:

   TFileIter(const char *name, Option_t *option = "",
             const char *ftitle = "", Int_t compress = 1,
             Int_t netopt = 0);
   TFileIter(TFile *file=0);
   TFileIter(TDirectory *directory);
   TFileIter(const TFileIter &);
   virtual ~TFileIter();
// --- draft !!!     virtual Int_t Copy(TFile *destFile);
   Int_t   CurrentCursorPosition() const;
   virtual const TFile *GetTFile() const;
   virtual const TDirectory *GetTDirectory() const;
   static  TString MapName(const char *name, const char *localSystemKey = 0
                                            , const char *mountedFileSystemKey = 0);
   static  const char *GetResourceName();
   static  const char *GetDefaultMapFileName();
   static  const char *GetLocalFileNameKey();
   static  const char *GetForeignFileSystemKey();
   static  void  PurgeKeys(TList *listOfKeys);
   virtual Bool_t      IsOpen() const;
   virtual TObject    *NextEventGet(UInt_t eventNumber=UInt_t(-1), UInt_t runNumber=UInt_t(-1), const char *name="*");
   virtual Int_t       NextEventPut(TObject *obj, UInt_t eventNum, UInt_t runNumber, const char *name=0);
   void                SetCursorPosition(Int_t cursorPosition);
   void                SetCursorPosition(const char *keyNameToFind);
   Int_t               GetObjlen() const;
   virtual Int_t       TotalKeys() const;
   virtual TKey       *SkipObjects(Int_t  nSkip=1);
   virtual TObject    *GetObject() const;
   virtual Int_t       GetDepth() const;

   TKey               *GetCurrentKey() const;
   const char         *GetKeyName() const;

   TFileIter &operator=(Int_t cursorPosition);
   TFileIter &operator=(const char *keyNameToFind);
   TFileIter &operator+=(Int_t shift);
   TFileIter &operator-=(Int_t shift);
   TFileIter &operator++();
   TFileIter &operator--();

   TObject *operator*() const;
   operator const char *() const;
   operator const TFile *() const;
   operator const TDirectory *() const;
   operator int () const;
   int operator==(const char *name) const;
   int operator!=(const char *name) const;

public:  // abstract TIterator methods implementations:

   virtual TObject *Next();
   virtual TObject *Next(Int_t  nSkip);
   virtual void Reset();
   virtual void Rewind();
   TObject *operator()(Int_t  nSkip);
   TObject *operator()();

   ClassDef(TFileIter,0) // TFile class iterator
};

//__________________________________________________________________________
inline const char *TFileIter::GetResourceName()        {return "ForeignFileMap";}
//__________________________________________________________________________
inline const char *TFileIter::GetDefaultMapFileName()  {return "io.config";}
//__________________________________________________________________________
inline const char *TFileIter::GetLocalFileNameKey()    {return "LocalFileSystem";}
//__________________________________________________________________________
inline const char *TFileIter::GetForeignFileSystemKey(){return "MountedFileSystem";}

//__________________________________________________________________________
inline Int_t TFileIter::CurrentCursorPosition() const
{
   // return the current
   return fNestedIterator ? fNestedIterator->CurrentCursorPosition() : fCursorPosition;
}

//__________________________________________________________________________
inline const TFile *TFileIter::GetTFile() const { return GetTDirectory()->GetFile(); }
//__________________________________________________________________________
inline const TDirectory *TFileIter::GetTDirectory() const
{ return fNestedIterator ? fNestedIterator->GetTDirectory() : fRootFile; }

//__________________________________________________________________________
inline TObject *TFileIter::Next()
{
   // Make 1 step over the file objects and returns its pointer
   // or 0, if there is no object left in the container
   return Next(1);
}

//__________________________________________________________________________
inline void TFileIter::Rewind() 
{
   // Alias for "Reset" method
   Reset();
}
//__________________________________________________________________________
inline void  TFileIter::SetCursorPosition(Int_t cursorPosition)
{
   // Make <cursorPosition> steps (>0 - forward) over the file
   // objects to skip it
   if (fNestedIterator) 
      fNestedIterator->SetCursorPosition(cursorPosition);
   else 
      SkipObjects(cursorPosition - fCursorPosition);
}

//__________________________________________________________________________
inline TFileIter &TFileIter::operator=(const char *keyNameToFind)
{
   // Iterate unless the name of the object matches <keyNameToFind>
   SetCursorPosition(keyNameToFind); return *this;}

//__________________________________________________________________________
inline TFileIter &TFileIter::operator=(Int_t cursorPosition)
{
  // Iterate over <cursorPosition>
  SetCursorPosition(cursorPosition);
  return *this;
}
//__________________________________________________________________________
inline TFileIter::operator const TDirectory *() const
{ return GetTDirectory();  }

//__________________________________________________________________________
inline TFileIter::operator const TFile *() const
{ return GetTFile (); }
//__________________________________________________________________________
inline TFileIter &TFileIter::operator+=(Int_t shift)
{ SkipObjects(shift); return *this;}
//__________________________________________________________________________
inline TFileIter &TFileIter::operator-=(Int_t shift)
{ return operator+=(-shift);}
//__________________________________________________________________________
inline TFileIter &TFileIter::operator++()
{ SkipObjects( 1); return *this;}
//__________________________________________________________________________
inline TFileIter &TFileIter::operator--()
{ SkipObjects(-1); return *this;}
//__________________________________________________________________________
inline TObject *TFileIter::operator*() const
{ return GetObject();}
//__________________________________________________________________________
inline TFileIter::operator int () const
{ return CurrentCursorPosition(); }
//__________________________________________________________________________
inline TFileIter::operator const char *() const
{
   // return the current key name
   return GetKeyName();
}
//__________________________________________________________________________
inline int TFileIter::operator==(const char *name) const
{ return name ? !strcmp(name,GetKeyName()):0;}

//__________________________________________________________________________
inline int TFileIter::operator!=(const char *name) const
{ return !(operator==(name)); }

//__________________________________________________________________________
inline TObject *TFileIter::operator()(){ return Next(); }
//__________________________________________________________________________
inline TObject *TFileIter::operator()(Int_t  nSkip){ return Next(nSkip);}

#endif
