// $Id: TFileIter.h,v 1.1 2001/03/02 21:25:09 fine Exp $
// Author: Valery Fine(fine@bnl.gov)   01/03/2001
// Copyright(c) 2001 [BNL] Brookhaven National Laboratory, Valeri Fine  (fine@bnl.gov). All right reserved",
//
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

#include "TString.h"
#include "TIterator.h"
#include "TList.h"
#include "TFile.h"

class TFileIter : public TListIter {

  private:

    TFile      *fFileBackUp;       //! temporary data-members 
    TDirectory *fDirectoryBackUp;  //! to save/restore TFile/TDirectory global scope

  protected:
    TFile   *fRootFile;            // Tfile to be iterated over
    TString  fEventName;           // current key name
    UInt_t   fRunNumber;           // current "run number"
    UInt_t   fEventNumber;         // current "event number"  
    Int_t    fCursorPosition;      // the position of the current key in the sorted TKey list
    Bool_t   fOwnTFile;            // Bit whether this classs creates TFile on its own to delete
  
    TObject *ReadObj(const TKey *key) const;

  protected:
    void     Initialize();
    void     SaveFileScope();
    void     RestoreFileScope();
    TKey    *NextEventKey(UInt_t eventNumber=UInt_t(-1), UInt_t runNumber=UInt_t(-1), const char *name="*");

  public:

    TFileIter(const char *name, Option_t *option = "",
              const char *ftitle = "", Int_t compress = 1,
              Int_t netopt = 0);
    TFileIter(TFile *file=0);
    virtual ~TFileIter();
    Int_t   CurrentCursorPosition() const;
    virtual const TFile *GetTFile() const;
    virtual TObject *NextEventGet(UInt_t eventNumber=UInt_t(-1), UInt_t runNumber=UInt_t(-1), const char *name="*");
    virtual Int_t    NextEventPut(TObject *obj, UInt_t eventNum, UInt_t runNumber, const char *name=0);
            void     SetCursorPosition(Int_t cursorPosition);
            void     SetCursorPosition(const char *keyNameToFind);
            Int_t    GetObjlen() const;
    virtual Int_t    TotalKeys() const;
    virtual TObject *SkipObjects(Int_t  nSkip=1);
    virtual TObject *GetObject() const;

    TKey    *GetCurrentKey() const;
    const char *GetKeyName() const;

    TFileIter &operator=(Int_t cursorPosition);
    TFileIter &operator=(const char *keyNameToFind);
    TFileIter &operator+=(Int_t shift);
    TFileIter &operator-=(Int_t shift);
    TFileIter &operator++();
    TFileIter &operator--();

    TObject *operator*() const;
    operator const char *() const;
    operator const TFile *() const;
    operator int () const;
    int operator==(const char *name) const;
    int operator!=(const char *name) const;

  public:  // abstract TIterator methods implementations:

    virtual TObject *Next();
    virtual TObject *Next(Int_t  nSkip);
    virtual void Reset();
    TObject *operator()(Int_t  nSkip);
    TObject *operator()();

    ClassDef(TFileIter,0) // TFile class iterator
};

//__________________________________________________________________________
inline Int_t TFileIter::CurrentCursorPosition() const
{ return fCursorPosition;}

//__________________________________________________________________________
inline const TFile *TFileIter::GetTFile() const { return fRootFile; }

//__________________________________________________________________________
inline TObject *TFileIter::Next()
{ return Next(1); }

//__________________________________________________________________________
inline void  TFileIter::SetCursorPosition(Int_t cursorPosition)
{ SkipObjects(cursorPosition - fCursorPosition); }

//__________________________________________________________________________
inline TFileIter &TFileIter::operator=(const char *keyNameToFind)
{ SetCursorPosition(keyNameToFind); return *this;}

//__________________________________________________________________________
inline TFileIter &TFileIter::operator=(Int_t cursorPosition)
{ 
  SetCursorPosition(cursorPosition);
  return *this;
}
//__________________________________________________________________________
inline TFileIter::operator const TFile *() const
{ return GetTFile (); }
//__________________________________________________________________________
inline TFileIter &TFileIter::operator+=(Int_t shift) 
{ SkipObjects(shift); return *this;}
//__________________________________________________________________________
inline TFileIter &TFileIter::operator-=(Int_t shift) 
{ SkipObjects(-shift); return *this;}
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
{ return GetKeyName();}
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

//__________________________________________________________________________
inline void  TFileIter::SaveFileScope() 
{ fFileBackUp = gFile; fDirectoryBackUp = gDirectory; }

//__________________________________________________________________________
inline void TFileIter::RestoreFileScope()
{  gFile = fFileBackUp; gDirectory = fDirectoryBackUp; }

#endif
