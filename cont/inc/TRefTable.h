// @(#)root/cont:$Name:  $:$Id: TRefTable.h,v 1.1 2004/08/20 14:46:36 brun Exp $
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
   Int_t         fSize;      //allocated size of array fParentIDs
   Int_t         fN;         //current maximum number of IDs in array fParentIDs
   Int_t        *fParentIDs; //[fSize] array of Parent IDs
   Int_t         fParentID;  //current parent ID in fParents (latest call to SetParent)
   TObjArray    *fParents;   //array of Parent objects  (eg TTree branch) holding the referenced objects 

public:

   TRefTable();
   TRefTable(Int_t size);
   virtual ~TRefTable();
   virtual Int_t      Add(Int_t uid);
   virtual void       Clear(Option_t * /*option*/ ="");
   virtual Int_t      Expand(Int_t newsize);
   virtual void       FillBuffer(TBuffer &b);
   Int_t              GetSize() const {return fSize;}
   Int_t              GetN() const {return fN;}
   TObjArray         *GetParents() const {return fParents;}
   TObject           *GetParent(Int_t uid) const;
   virtual void       ReadBuffer(TBuffer &b);
   virtual Int_t      SetParent(const TObject *parent);

   ClassDef(TRefTable,1)  //Table of referenced objects during an I/O operation
};

#endif
