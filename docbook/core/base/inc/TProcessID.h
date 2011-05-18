// @(#)root/cont:$Id$
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

class TExMap;

class TProcessID : public TNamed {

private:
   TProcessID(const TProcessID &ref);            // TProcessID are not copiable.
   TProcessID& operator=(const TProcessID &ref); // TProcessID are not copiable.

protected:
   Int_t              fCount;     //!Reference count to this object (from TFile)
   TObjArray         *fObjects;   //!Array pointing to the referenced objects
   
   static TProcessID *fgPID;      //Pointer to current session ProcessID
   static TObjArray  *fgPIDs;     //Table of ProcessIDs
   static TExMap     *fgObjPIDs;  //Table pointer to pids
   static UInt_t      fgNumber;   //Referenced objects count
  
public:
   TProcessID();
   virtual ~TProcessID();
   void             CheckInit();
   virtual void     Clear(Option_t *option="");
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
   static UInt_t       GetNProcessIDs();
   static TProcessID  *GetPID();
   static TObjArray   *GetPIDs();
   static TProcessID  *GetProcessID(UShort_t pid);
   static TProcessID  *GetProcessWithUID(const TObject *obj);
   static TProcessID  *GetProcessWithUID(UInt_t uid,const void *obj);
   static TProcessID  *GetSessionProcessID();
   static  UInt_t      GetObjectCount();
   static  Bool_t      IsValid(TProcessID *pid);
   static  void        SetObjectCount(UInt_t number);
         
   ClassDef(TProcessID,1)  //Process Unique Identifier in time and space
};

#endif
