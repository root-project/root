// @(#)root/cont:$Name:  $:$Id: TProcessID.h,v 1.1 2001/10/01 10:29:08 brun Exp $
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
   Int_t       fCount;          //!Reference count to this object (from TFile)
   TObjArray  *fObjects;        //!Array pointing to the referenced objects
   
public:
   TProcessID();
   TProcessID(UShort_t pid);
   TProcessID(const TProcessID &ref);
   virtual ~TProcessID();
   Int_t            DecrementCount();
   Int_t            IncrementCount();
   Int_t            GetCount() const {return fCount;}
   TObjArray       *GetObjects() const {return fObjects;}
   TObject         *GetObjectWithID(UInt_t uid);
   void             PutObjectWithID(TObject *obj, UInt_t uid=0);
   virtual void     RecursiveRemove(TObject *obj);
   
   static TProcessID  *ReadProcessID(UShort_t pidf , TFile *file);
      
   ClassDef(TProcessID,1)  //Process Unique Identifier in time and space
};

#endif
