// @(#)root/cont:$Name:  $:$Id: TRefTable.h,v 1.4 2005/10/25 22:11:58 pcanal Exp $
// Author: Rene Brun   17/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRefTable
#define ROOT_TRefTable


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRefTable                                                            //
//                                                                      //
// A TRefTable maintains the association between a referenced object    //
// and the parent object supporting this referenced object.             //
// The parent object is typically a branch of a TTree.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TExMap
#include "TExMap.h"
#endif
#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif

class TRefTable : public TObject {

protected:
   Int_t             fSize;       //allocated size of array fParentIDs
   Int_t             fN;          //current maximum number of IDs in array fParentIDs
   Int_t            *fParentIDs;  //[fSize] array of Parent IDs
   Int_t             fParentID;   //current parent ID in fParents (latest call to SetParent)
   UInt_t            fUID;        //!Current uid (set by TRef::GetObject)
   TObjArray        *fParents;    //array of Parent objects  (eg TTree branch) holding the referenced objects
   TObject          *fOwner;      //Object owning this TRefTable
   static TRefTable *fgRefTable;  //Pointer to current TRefTable

public:

   TRefTable();
   TRefTable(TObject *owner, Int_t size);
   virtual ~TRefTable();
   virtual Int_t      Add(Int_t uid);
   virtual void       Clear(Option_t * /*option*/ ="");
   virtual Int_t      Expand(Int_t newsize);
   virtual void       FillBuffer(TBuffer &b);
   static TRefTable  *GetRefTable();
   Int_t              GetSize() const {return fSize;}
   Int_t              GetN() const {return fN;}
   TObject           *GetOwner() const {return fOwner;}
   TObject           *GetParent(Int_t uid) const;
   TObjArray         *GetParents() const {return fParents;}
   UInt_t             GetUID() const {return fUID;}
   virtual Bool_t     Notify();
   virtual void       ReadBuffer(TBuffer &b);
   virtual void       Reset(Option_t * /* option */ ="");
   virtual Int_t      SetParent(const TObject *parent);
   static  void       SetRefTable(TRefTable *table);
   virtual void       SetUID(UInt_t uid) {fUID=uid;}

   ClassDef(TRefTable,2)  //Table of referenced objects during an I/O operation
};

#endif
