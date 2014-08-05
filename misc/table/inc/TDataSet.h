// @(#)root/table:$Id$
// Author: Valery Fine(fine@mail.cern.ch)   03/07/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TDataSet
#define ROOT_TDataSet


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TDataSet                                                             //
//                                                                      //
// TDataSet class is a base class to implement the directory-like       //
// data structures and maintain it via TDataSetIter class iterator      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TList.h"
#include "TObjArray.h"

#include "TNamed.h"
#include "TNode.h"

class TDataSetIter;
class TBrowser;

class TDataSet : public TNamed
{
friend class TDataSetIter;
public:
   // The control codes to navigate the TDataSet structure via TDataSet::Pass method

   typedef enum {
         kContinue,  // continue passing
         kPrune,     // stop passing of the current branch but continue with the next one if any
         kStop,      // break passing
         kUp,        // break passing, return to the previous level, then continue
         kStruct,    // work with structural links only
         kAll,       // work with all links
         kRefs,      // work with refs links only
         kMarked     // work with marked links only
    } EDataSetPass;

private:
   void operator=(const TDataSet &){}
// --   Int_t IncCnt(){ fCnt++; return fCnt;}
// --   Int_t DecCnt(){ fCnt--; return fCnt;}
// --   Int_t Cnt()   { return fCnt;}
protected:
   static TDataSet    *fgMainSet; // pointer the main dataset;
   TDataSet           *fParent;   // pointer to mother of the directory
   TSeqCollection     *fList;     // List of the the the objects included into this dataset
// --     Int_t               fCnt;      // reference counter.
   virtual void SetMother(TObject *mother) {SetParent((TDataSet*)mother);}
   TDataSet(const char *name,const char *title):
   TNamed(name,title),fParent(0),fList(0){} // to support TDictionary
   void AddMain(TDataSet *set);
   static EDataSetPass SortIt(TDataSet *ds);
   static EDataSetPass SortIt(TDataSet *ds,void *user);
   TDataSet *GetRealParent();
   void MakeCollection();


public:

   //----- dataset flags
   enum ESetBits {
      kMark        = BIT(22)   // if object is marked
     ,kArray       = BIT(20)   // if object has TObjArray inside
   };

   enum EBitOpt {
               kSet   = kTRUE,
               kReset = kFALSE
             };

   TDataSet(const char *name="", TDataSet *parent=0,  Bool_t arrayFlag = kFALSE);
   TDataSet(const TDataSet &src,EDataSetPass iopt=kAll);
   TDataSet(TNode &src);
   virtual ~TDataSet();
   virtual void         Add(TDataSet *dataset);
   virtual void         AddAt(TDataSet *dataset,Int_t idx=0);
   virtual void         AddAtAndExpand(TDataSet *dataset, Int_t idx=0);
   virtual void         AddFirst(TDataSet *dataset);
   virtual void         AddLast(TDataSet *dataset);
   TDataSet            *At(Int_t idx) const;
   virtual void         Browse(TBrowser *b);
   virtual TObject     *Clone(const char *newname="") const;
   virtual void         Delete(Option_t *opt="");
   virtual TDataSet    *Find(const char *path) const;
   virtual TDataSet    *FindByName(const char *name,const char *path="",Option_t *opt="") const;
   virtual TDataSet    *FindByPath(const char *path) const;
   virtual TDataSet    *FindByTitle(const char *title,const char *path="",Option_t *opt="") const;
   TObject             *FindObject(const char *name) const {return FindByName(name);}
   TObject             *FindObject(const TObject *o)  const { return TObject::FindObject(o);}
   virtual TDataSet    *First() const;
   TObjArray           *GetObjArray() const { return (TObjArray *)fList; }
   virtual TSeqCollection *GetCollection() const { return (TSeqCollection *)fList; }
   TList               *GetList()   const { return (TList *)fList; }
   virtual Int_t        GetListSize() const;
   static  TDataSet    *GetMainSet();
   TObject             *GetMother() const { return (TObject*)GetParent();}
   virtual TObject     *GetObject() const;
   virtual TDataSet    *GetParent() const { return fParent;}
   virtual Long_t       HasData() const {return 0;}    // Check whether this dataset has extra "data-members"
   virtual TDataSet    *Instance() const;
   static  TDataSet    *instance();
   virtual TString      Path() const;                  // return the "full" path of this dataset
   virtual EDataSetPass Pass(EDataSetPass ( *callback)(TDataSet *),Int_t depth=0);
   virtual EDataSetPass Pass(EDataSetPass ( *callback)(TDataSet *,void*),void *user,Int_t depth=0);
   virtual void         PrintContents(Option_t *opt="") const;
   virtual Int_t        Purge(Option_t *opt="");
   virtual void         Remove(TDataSet *set);
   virtual TDataSet    *RemoveAt(Int_t idx);
   virtual void         SetMother(TDataSet *parent=0){SetParent(parent);};
   virtual void         SetObject(TObject *obj);
   virtual void         SetParent(TDataSet *parent=0);
   virtual void         SetWrite();
   virtual void         Shunt(TDataSet *newParent=0);
   virtual void         Sort();                        //Sort objects in lexical order
   virtual Bool_t       IsEmpty() const;
   virtual Bool_t       IsFolder() const {return kTRUE;}
   virtual Bool_t       IsMarked() const ;
   virtual Bool_t       IsThisDir(const char *dirname,int len=-1,int ignorecase=0) const ;
   virtual TDataSet    *Last() const;
   virtual void         ls(Option_t *option="")  const;      // Option "*" means print all levels
   virtual void         ls(Int_t depth)  const;              // Print the "depth" levels of this datatset
   void                 Mark();                              // *MENU*
   void                 UnMark();                            // *MENU*
   void                 MarkAll();                           // *MENU*
   void                 UnMarkAll();                         // *MENU*
   void                 InvertAllMarks();                    // *MENU*
   void                 Mark(UInt_t flag,EBitOpt reset=kSet);
   virtual TDataSet    *Next() const;
   virtual TDataSet    *Prev() const;
   virtual void         Update();                            // Update dataset
   virtual void         Update(TDataSet *set,UInt_t opt=0);// Update this dataset with the new one
   virtual Int_t        Write(const char *name=0, Int_t option=0, Int_t bufsize=0);
   virtual Int_t        Write(const char *name=0, Int_t option=0, Int_t bufsize=0) const;
   ClassDef(TDataSet,1) // The base class to create the hierarchical data structures
};

inline void        TDataSet::Add(TDataSet *dataset){ AddLast(dataset); }
inline TDataSet   *TDataSet::At(Int_t idx) const {return fList ? (TDataSet *)fList->At(idx) : 0;  }
inline Int_t       TDataSet::GetListSize() const {return (fList) ? fList->GetSize():0;}
inline TDataSet   *TDataSet::instance() { return new TDataSet();}
inline Bool_t      TDataSet::IsMarked() const { return TestBit(kMark); }
inline void        TDataSet::Mark(UInt_t flag,EBitOpt reset){ SetBit(flag,reset); }
inline void        TDataSet::Mark()     { Mark(kMark,kSet); }
inline void        TDataSet::UnMark()   { Mark(kMark,kReset); }


#endif
