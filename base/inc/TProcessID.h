// @(#)root/cont:$Name:  $:$Id: TProcessID.h,v 1.3 2001/12/02 15:11:32 brun Exp $
// Author: Rene Brun   28/09/2001

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProcessID
#define ROOT_TProcessID


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProcessID                                                           //
//                                                                      //
// Process Identifier object                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif

class TFile;

class TProcessID : public TNamed {

protected:
   Int_t              fCount;     //!Reference count to this object (from TFile)
   TObjArray         *fObjects;   //!Array pointing to the referenced objects
   
   static TProcessID *fgPID;      //Pointer to current session ProcessID
   static TObjArray  *fgPIDs;     //Table of ProcessIDs
   static UInt_t      fgNumber;   //Referenced objects count
  
  public:
   TProcessID();
   TProcessID(const TProcessID &ref);
   virtual ~TProcessID();
   Int_t            DecrementCount();
   Int_t            IncrementCount();
   Int_t            GetCount() const {return fCount;}
   TObjArray       *GetObjects() const {return fObjects;}
   TObject         *GetObjectWithID(UInt_t uid);
   void             PutObjectWithID(TObject *obj, UInt_t uid=0);
   virtual void     RecursiveRemove(TObject *obj);
   
   static TProcessID  *AddProcessID();
   static UInt_t       AssignID(TObject *obj);
   static void         Cleanup();
   static TProcessID  *ReadProcessID(UShort_t pidf , TFile *file);
   static UShort_t     WriteProcessID(TProcessID *pid , TFile *file);
   static TProcessID  *GetProcessID(UShort_t pid);
   static TProcessID  *GetSessionProcessID();
   static  UInt_t      GetObjectCount();
   static  void        SetObjectCount(UInt_t number);
         
   ClassDef(TProcessID,1)  //Process Unique Identifier in time and space
};

#endif
