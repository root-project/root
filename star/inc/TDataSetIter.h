// @(#)root/star:$Name:  $:$Id: TDataSetIter.h,v 1.1.1.1 2000/05/16 17:00:49 rdm Exp $
// Author: Valery Fine(fine@mail.cern.ch)   03/07/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDataSetIter
#define ROOT_TDataSetIter


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// $Id: TDataSetIter.h,v 1.1.1.1 2000/05/16 17:00:49 rdm Exp $
//
// TDataSetIter                                                         //
//                                                                      //
// Iterator of TDataSet lists.                                          //
//                                                                      //
// Provides "standard" features of the TIter class for TDataSet object  //
//                             and                                      //
// allows navigating TDataSet structure using the custom "directory"    //
//    notation (see TDataSet::Find(const Char *path) method)            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TList.h"
#include "TNamed.h"

#include "TDataSet.h"

class TDataSetIter : public TObject{
protected:
   TIter           *fNext;            // "standard" ROOT iterator for containers
   TIter           *fNextSet[100];    // the list of the TList iterators to bypass the whole dataset
   Int_t            fDepth;           // the current depth of the passing
   Int_t            fMaxDepth;        // the max depth of the passing (=1 by default)

   TDataSet       *fDataSet;          // Pointer to the last selected TDataSet
   TDataSet       *fRootDataSet;      // Pointer to the root TDataSet
   TDataSet       *fWorkingDataSet;   // Pointer to the working TDataSet
   TDataSet       *NextDataSet(TIter &next);
   TDataSet       *NextDataSet(Int_t nDataSet);
   TDataSet       *GetNullSet();      // return a fake pointer == -1 casted to (TDataSet *)

   static TDataSet *fNullDataSet;
public:
  TDataSetIter(TDataSet *l=0, Int_t depth=1, Bool_t dir=kIterForward);
  TDataSetIter(TDataSet *l, Bool_t dir);
  virtual         ~TDataSetIter();

  virtual TDataSet    *Add(TDataSet *set){return Add(set,(TDataSet *)0);}
  virtual TDataSet    *Add(TDataSet *set, const Char_t *path);
  virtual TDataSet    *Add(TDataSet *set, TDataSet *dataset);

  virtual TDataSet    *Cd(const Char_t *dirname);
  virtual TDataSet    *Cd(TDataSet *ds);
  virtual TDataSet    *operator()( EDataSetPass mode=kContinue ) {return  Next(mode);}
  virtual TDataSet    *operator()(const Char_t *path) { return Find(path); }
  virtual TDataSet    *operator[](const Char_t *path);
  virtual Int_t          GetDepth() const {return fDepth;}
  virtual TDataSet    *Cwd() const {return fWorkingDataSet;}
  virtual TDataSet    *Dir(Char_t *dirname);
  virtual Int_t          Du() const;            // summarize dataset usage
  virtual Int_t          Df() const {return 0;} // report number of free "table" blocks.

  virtual TDataSet    *Find(const Char_t *path, TDataSet *rootset=0,Bool_t mkdir=kFALSE);
  virtual TDataSet    *FindByPath(const Char_t *path, TDataSet *rootset=0,Bool_t mkdir=kFALSE);
  virtual TDataSet    *FindDataSet(const Char_t *name,const Char_t *path="",Option_t *opt="");
  virtual TDataSet    *FindByName(const Char_t *name,const Char_t *path="",Option_t *opt="");
  virtual TDataSet    *FindDataSet(TDataSet *set,const Char_t *path,Option_t *opt="");
  virtual Int_t          Flag(UInt_t flag=kMark,EBitOpt reset=kSet){return Flag((TDataSet *)0,flag,reset);}
  virtual Int_t          Flag(const Char_t *path,UInt_t flag=kMark,EBitOpt reset=kSet);
  virtual Int_t          Flag(TDataSet *dataset,UInt_t flag=kMark,EBitOpt reset=kSet);

  virtual TDataSet    *Ls(const Char_t *dirname="",Option_t *opt="");
  virtual TDataSet    *Ls(const Char_t *dirname,Int_t depth);
  virtual TDataSet    *ls(const Char_t *dirname="",Option_t *opt="")   {return Ls(dirname,opt);}
  virtual TDataSet    *ls(const Char_t *dirname,Int_t depth){return Ls(dirname,depth);}
  virtual TDataSet    *Mkdir(const Char_t *dirname);
  virtual TDataSet    *Md(const Char_t *dirname)                       {return Mkdir(dirname);}
  virtual TString        Path(const Char_t *path)                        {TDataSet *set = Find(path); return set ? TString (""):set->Path();}
  virtual TString        Path() {return fWorkingDataSet ? TString ("") : fWorkingDataSet->Path();}
  virtual TDataSet    *Pwd(Option_t *opt="") const                     {if (Cwd()) Cwd()->ls(opt); return Cwd();}
  virtual Int_t          Rmdir(TDataSet *dataset,Option_t *option="");
  virtual Int_t          Rmdir(const Char_t *dirname,Option_t *option=""){return Rmdir(Find(dirname),option);}
  virtual Int_t          Rd(const Char_t *dirname,Option_t *option="")   {return Rmdir(Find(dirname),option);}

  virtual TDataSet    *Shunt(TDataSet *set){return Shunt(set,(TDataSet *)0);}
  virtual TDataSet    *Shunt(TDataSet *set, const Char_t *path);
  virtual TDataSet    *Shunt(TDataSet *set, TDataSet *dataset);

  virtual TDataSet    *Next( EDataSetPass mode=kContinue);
  virtual TDataSet    *Next(const Char_t *path, TDataSet *rootset=0,Bool_t mkdir=kFALSE){return Find(path,rootset,mkdir);}
  virtual void        Notify(TDataSet *dataset);
   const  Option_t   *GetOption() const                                      { return fNext ? fNext->GetOption():0; }
  virtual void        Reset(TDataSet *l=0,Int_t depth=0);
  virtual TDataSet   *operator *() const ;
  ClassDef(TDataSetIter,0) // class-iterator to navigate TDataSet structure
};

inline TDataSet *TDataSetIter::operator *() const { return fDataSet == fNullDataSet ? fWorkingDataSet : fDataSet; }
inline TDataSet *TDataSetIter::GetNullSet() { return (TDataSet *)fNullDataSet; } // return a fake pointer == -1 casted to (TDataSet *)
#endif

