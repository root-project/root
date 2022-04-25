// @(#)root/cont:$Id$
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


#include "TObject.h"

#include <string>
#include <vector>

class TObjArray;
class TProcessID;

class TRefTable : public TObject {

protected:
   Int_t             fNumPIDs;    //!number of known ProcessIDs
   Int_t            *fAllocSize;  //![fNumPIDs] allocated size of array fParentIDs for each ProcessID
   Int_t            *fN;          //![fNumPIDs] current maximum number of IDs in array fParentIDs for each ProcessID
   Int_t           **fParentIDs;  //![fNumPIDs][fAllocSize] array of Parent IDs
   Int_t             fParentID;   //!current parent ID in fParents (latest call to SetParent)
   Int_t             fDefaultSize;//!default size for a new PID array
   UInt_t            fUID;        //!Current uid (set by TRef::GetObject)
   TProcessID       *fUIDContext; //!TProcessID the current uid is referring to
   Int_t             fSize;       //dummy for backward compatibility
   TObjArray        *fParents;    //array of Parent objects  (eg TTree branch) holding the referenced objects
   TObject          *fOwner;      //Object owning this TRefTable
   std::vector<std::string> fProcessGUIDs; // UUIDs of TProcessIDs used in fParentIDs
   std::vector<Int_t> fMapPIDtoInternal;   //! cache of pid to index in fProcessGUIDs
   static TRefTable *fgRefTable;  //Pointer to current TRefTable

   Int_t              AddInternalIdxForPID(TProcessID* procid);
   virtual Int_t      ExpandForIID(Int_t iid, Int_t newsize);
   void               ExpandPIDs(Int_t numpids);
   Int_t              FindPIDGUID(const char* guid) const;
   Int_t              GetInternalIdxForPID(TProcessID* procid) const;
   Int_t              GetInternalIdxForPID(Int_t pid) const;

public:

   enum EStatusBits {
      kHaveWarnedReadingOld = BIT(14)
   };

   TRefTable();
   TRefTable(TObject *owner, Int_t size);
   virtual ~TRefTable();
   virtual Int_t      Add(Int_t uid, TProcessID* context = nullptr);
   void               Clear(Option_t * /*option*/ ="") override;
   virtual Int_t      Expand(Int_t pid, Int_t newsize);
   virtual void       FillBuffer(TBuffer &b);
   static TRefTable  *GetRefTable();
   Int_t              GetNumPIDs() const {return fNumPIDs;}
   Int_t              GetSize(Int_t pid) const {return fAllocSize[GetInternalIdxForPID(pid)];}
   Int_t              GetN(Int_t pid) const {return fN[GetInternalIdxForPID(pid)];}
   TObject           *GetOwner() const {return fOwner;}
   TObject           *GetParent(Int_t uid, TProcessID* context = 0) const;
   TObjArray         *GetParents() const {return fParents;}
   UInt_t             GetUID() const {return fUID;}
   TProcessID        *GetUIDContext() const {return fUIDContext;}
   Bool_t             Notify() override;
   virtual void       ReadBuffer(TBuffer &b);
   virtual void       Reset(Option_t * /* option */ ="");
   virtual Int_t      SetParent(const TObject* parent, Int_t branchID);
   static  void       SetRefTable(TRefTable *table);
   virtual void       SetUID(UInt_t uid, TProcessID* context = 0) {fUID=uid; fUIDContext = context;}

   ClassDefOverride(TRefTable,3)  //Table of referenced objects during an I/O operation
};

#endif
