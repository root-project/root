// @(#)root/cont:$Name:  $:$Id: TProcessID.h,v 1.4 2001/09/28 07:54:00 brun Exp $
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
#ifndef ROOT_TMap
#include "TExMap.h"
#endif

class TFile;

class TProcessID : public TNamed {

protected:
   Int_t     fCount;          //!Reference count to this object (from TFile)
   TExMap     *fMap;          //!Pointer to the map descriptor
   
public:
   TProcessID();
   TProcessID(Int_t pid);
   TProcessID(const TProcessID &ref);
   virtual ~TProcessID();
   Int_t            DecrementCount();
   Int_t            IncrementCount();
   Int_t            GetCount() const {return fCount;}
   TExMap          *GetMap() const {return fMap;}
   TObject         *GetObjectWithID(Long_t uid);
   void             PutObjectWithID(Long_t uid, TObject *obj);
   virtual void     RecursiveRemove(TObject *obj);
   
   static TProcessID  *ReadProcessID(Int_t pidf , TFile *file);
      
   ClassDef(TProcessID,1)  //Process Unique Identifier in time and space
};

#endif
