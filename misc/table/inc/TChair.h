// @(#)root/table:$Id$
// Author: Valery Fine(fine@bnl.gov)   13/03/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TChair
#define ROOT_TChair

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TChair                                                              //
//                                                                      //
//  It is a base class to create a custom interface for TTable objects  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TTable.h"

class TChair : public TDataSet {

protected:

   TTable  *fTable;     // the "TTable" object this object is pthe proxy for
   ULong_t  fLastIndx;  // index pof the last used  table row;
   void    *fLastRow;   // pointer to the last used table row; fLastRow = table[fLastIndx]
   
         TTable *GetThisTable()       {return fTable; }
   const TTable *GetThisTable() const {return fTable; }
   static void  *GetOffset(const void *base,ULong_t offset) { return (void  *)((Char_t *)base + offset);}
   TChair() : fTable(0), fLastIndx(0), fLastRow(0) { ; }

public:

   TChair(TTable *table) : fTable(table),fLastIndx(0),fLastRow(0) { ; }
   TChair(const TChair &org) : TDataSet(org) {assert(0);}
//   TChair     &operator=(const TChair &rhs){ assert(0); return rhs;}
   virtual    ~TChair(){;}

   virtual     void       Adopt(Int_t n, void *array){GetThisTable()->Adopt(n,array);}
   virtual     void       AddAt(TDataSet *dataset,Int_t idx);
   virtual     void       AddAt(const void *c, Int_t i){GetThisTable()->AddAt(c,i);}
              const void *At(Int_t i) const {return GetThisTable()->At(i);}
   virtual     void       Browse(TBrowser *b){GetThisTable()->Browse(b);}
   virtual     void       CopySet(TChair &chair){GetThisTable()->CopySet(*chair.GetThisTable());}
               Int_t      CopyRows(const TChair *srcChair, Int_t srcRow=0, Int_t dstRow=0, Int_t nRows=0, Bool_t expand=kFALSE)
                          {return GetThisTable()->CopyRows(srcChair->GetThisTable(),srcRow,dstRow,nRows,expand);}
   virtual     void       Draw(Option_t *opt){GetThisTable()->Draw(opt);}
   virtual     TH1       *Draw(TCut varexp, TCut selection, Option_t *option="",
                          Int_t nentries=1000000000, Int_t firstentry=0)
                          {return GetThisTable()->Draw(varexp,selection,option,nentries,firstentry);}
   virtual     TH1       *Draw(const char *varexp, const char *selection, Option_t *option="",
                               Int_t nentries=1000000000, Int_t firstentry=0) {
                           return GetThisTable()->Draw(varexp,selection,option,nentries,firstentry);}
   virtual     Char_t    *GetArray() const    {return (Char_t *)GetThisTable()->GetArray();}
   virtual     TClass    *GetRowClass() const {return GetThisTable()->GetRowClass();}
   virtual     Long_t     GetNRows() const    {return GetThisTable()->GetNRows();}
   virtual     Long_t     GetRowSize() const  {return GetThisTable()->GetRowSize();}
   virtual     Long_t     GetTableSize() const{return GetThisTable()->GetTableSize();}
               const TTable  *Table() const {return fTable; }
   virtual     TTableDescriptor *GetRowDescriptors()   const {return GetThisTable()->GetRowDescriptors();}
   virtual     const Char_t       *GetType()             const {return GetThisTable()->GetType();}
   virtual     void       Fit(const char *formula ,const char *varexp, const char *selection="",Option_t *option="",Option_t *goption="",
                              Int_t nentries=1000000000, Int_t firstentry=0) {
                           GetThisTable()->Fit(formula,varexp,selection,option,goption,nentries,firstentry);}
   virtual     Long_t     HasData() const  { return GetThisTable()->HasData();}
   virtual     Bool_t     IsFolder() const { return GetThisTable()->IsFolder();}
   virtual     void       ls(Option_t *option="") const {GetThisTable()->ls(option);}
   virtual     void       ls(Int_t deep) const  {GetThisTable()->ls(deep);}
               Int_t      NaN()           {return GetThisTable()->NaN();}
   virtual     Char_t    *MakeExpression(const Char_t *expressions[],Int_t nExpressions)
                         {return GetThisTable()->MakeExpression(expressions,nExpressions);}
   virtual     Char_t    *Print(Char_t *buf,Int_t n) const { return GetThisTable()->Print(buf, n);}
   virtual     void       Print(Option_t *opt="")    const {GetThisTable()->Print(opt);}
   virtual  const Char_t *Print(Int_t row, Int_t rownumber=10,
                                const Char_t *colfirst="",const Char_t *collast="") const {
                           return GetThisTable()->Print(row,rownumber,colfirst,collast); }
   virtual  const Char_t *PrintHeader() const {return GetThisTable()->PrintHeader();}
   virtual  Int_t         Purge(Option_t *opt="")    {return GetThisTable()->Purge(opt);}

               void      *ReAllocate(Int_t newsize) { return GetThisTable()->ReAllocate(newsize); }
               void      *ReAllocate()              { return GetThisTable()->ReAllocate(); }
   virtual     void       SavePrimitive(ostream &out, Option_t *option="") {GetThisTable()->SavePrimitive(out,option);}

   virtual     void       Set(Int_t n)                                   {GetThisTable()->Set(n);}
   virtual     void       Set(Int_t n, Char_t *array)                    {GetThisTable()->Set(n,array);}
   virtual     void       SetNRows(Int_t n)                              {GetThisTable()->SetNRows(n);}
   virtual     void       Reset(Int_t c=0)                               {GetThisTable()->Reset(c) ;}
   virtual     void       Update()                                       {GetThisTable()->Update();}
   virtual     void       Update(TDataSet *set, UInt_t opt=0)            {GetThisTable()->Update(set,opt);}
               void      *operator[](Int_t i);
              const void *operator[](Int_t i) const;

   ClassDef(TChair,0)  // A base class to provide a user custom interface to TTable class objects
};

inline void  TChair::AddAt(TDataSet *dataset,Int_t idx)
{TDataSet::AddAt(dataset,idx);}

inline void *TChair::operator[](Int_t i)
{

//   if (!GetThisTable()->BoundsOk("TChair::operator[]", i))
//      i = 0;
    return (void *)((char *)GetArray()+i*GetRowSize());
}

inline const void *TChair::operator[](Int_t i) const
{
//   if (!GetThisTable()->BoundsOk("TChair::operator[]", i))
//      i = 0;
    return (const void *)((char *)GetArray()+i*GetRowSize());
}

// $Log: TChair.h,v $
// Revision 1.5  2006/07/03 16:10:46  brun
// from Axel:
// Change the signature of SavePrimitive from
//
//   void SavePrimitive(ofstream &out, Option_t *option);
// to
//   void SavePrimitive(ostream &out, Option_t *option = "");
//
// With this change one can do, eg
//    myhist.SavePrimitive(std::cout);
//
// WARNING: do rm -f tree/src/*.o
//
// Revision 1.4  2005/04/25 17:23:29  brun
// From Valeri Fine:
//
//   TChair.h:
//      - Make the "fTable" data-member to be "protected" (it was "private")
//        to facilitate the class reuse (thanks Y.Fisyak)
//
//   TColumnView.cxx:
//      - extra protection against of zero gPad
//
//   TPad.cxx
//     - initialize the "fPadView3D" data-member
//      (causes the crash within "table" package occasionally)
//
// Revision 1.3  2003/01/27 20:41:36  brun
// New version of the Table package by Valeri Fine.
// New classes TIndexTable TResponseIterator TResponseTable TTableMap
//
// Revision 1.1.1.2  2002/12/02 21:57:31  fisyak
// *** empty log message ***
//
// Revision 1.2  2002/12/02 18:50:05  rdm
// mega patch to remove almost all compiler warnings on MacOS X where the
// compiler is by default in pedantic mode (LHCb also like to use this option).
// The following issues have been fixed:
// - removal of unused arguments
// - comparison between signed and unsigned integers
// - not calling of base class copy ctor in copy ctor's
// To be done, the TGeo classes where we get still many hundred warnings of
// the above nature. List forwarded to Andrei.
//
// Revision 1.1  2002/05/27 16:26:59  rdm
// rename star to table.
//
// Revision 1.9  2001/02/07 08:18:15  brun
//
// New version of the STAR classes compiling with no warnings.
//
// Revision 1.1.1.3  2001/01/22 12:59:34  fisyak
// *** empty log message ***
//
// Revision 1.8  2001/01/19 07:22:54  brun
// A few changes in the STAR classes to remove some compiler warnings.
//
// Revision 1.2  2001/01/14 01:26:54  fine
// New implementation TTable::SavePrimitive and AsString
//
// Revision 1.1.1.2  2000/12/18 21:05:26  fisyak
// *** empty log message ***
//
// Revision 1.7  2000/12/13 15:13:53  brun
//       W A R N I N G   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//      ==================================================================
// A very long list of changes in this pre-release of version 3.00.
// We have modified the signature of many functions (in particular TObject)
// to introduce more constness in the system.
// You must change your code if your class derives from TObject and uses
// one of the modified functions such as ls, Print, Compare, Hash, etc.
// The modified functions in TObject have the following signature:
//    virtual TObject    *Clone() const;
//    virtual Int_t       Compare(const TObject *obj) const;
//    virtual void        Delete(Option_t *option=""); // *MENU*
//    virtual void        DrawClass() const; // *MENU*
//    virtual void        DrawClone(Option_t *option="") const; // *MENU*
//    virtual void        Dump() const; // *MENU*
//    virtual TObject    *FindObject(const TObject *obj) const;
//    virtual char       *GetObjectInfo(Int_t px, Int_t py) const;
//    virtual ULong_t     Hash() const;
//    virtual void        Inspect() const; // *MENU*
//    virtual Bool_t      IsEqual(const TObject *obj) const;
//    virtual void        ls(Option_t *option="") const;
//    virtual void        Print(Option_t *option="") const;
//
// A similar operation has been done with classes such as TH1, TVirtualPad,
// TTree, etc.
//
// Revision 1.6  2000/12/11 09:52:24  brun
// Functions ls declared const like in the base class
//
// Revision 1.5  2000/09/29 07:15:30  brun
// Remove unused function ReadGenericArray
//
// Revision 1.4  2000/09/05 09:21:24  brun
// The following headers and classes have been modified to take into account;
//   - the new signature of IsFolder (now const)
//   - the new TObject::FindObject
//   - the fact that the static functions of TObject have been moved to TROOT.
//
// Revision 1.3  2000/08/09 08:41:22  brun
// Import new versions of the STAR classes from Valery Fine
//
// Revision 1.4  2000/08/05 19:01:59  fisyak
// Merge
//
// Revision 1.3  2000/06/05 21:22:01  fisyak
// mergre with Rene's corrections
//
// Revision 1.1.1.2  2000/06/05 12:44:33  fisyak
// *** empty log message ***
//
// Revision 1.2  2000/06/05 08:01:03  brun
// Merge with valery's version
//
// Revision 1.2  2000/06/02 14:51:37  fine
// new helper class to browse tables has been introduced
//
// Revision 1.1.1.1  2000/05/19 12:46:09  fisyak
// CVS version of root 2.24.05 without history
//
// Revision 1.1.1.1  2000/05/16 17:00:49  rdm
// Initial import of ROOT into CVS
//
// Revision 1.1  2000/03/09 21:57:03  fine
// TChair class to be moved to ROOT later
//
// Revision 1.1  2000/02/28 03:42:24  fine
// New base class to provide a custom interface to the TTable objects
//

#endif
