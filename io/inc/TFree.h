// @(#)root/io:$Name:  $:$Id: TFree.h,v 1.7 2005/11/21 11:17:18 rdm Exp $
// Author: Rene Brun   28/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFree
#define ROOT_TFree


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFree                                                                //
//                                                                      //
// Description of free segments on a file.                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif


class TFree : public TObject {

protected:
   Long64_t        fFirst;            //First free word of segment
   Long64_t        fLast;             //Last free word of segment

public:
   TFree();
   TFree(TList *lfree, Long64_t first, Long64_t last);
   virtual ~TFree();
           TFree    *AddFree(TList *lfree, Long64_t first, Long64_t last);
   virtual void      FillBuffer(char *&buffer);
           TFree    *GetBestFree(TList *lfree, Int_t nbytes);
           Long64_t  GetFirst() const {return fFirst;}
           Long64_t  GetLast() const {return fLast;}
   virtual void      ReadBuffer(char *&buffer);
           void      SetFirst(Long64_t first) {fFirst=first;}
           void      SetLast(Long64_t last) {fLast=last;}
           Int_t     Sizeof() const;

   ClassDef(TFree,1);  //Description of free segments on a file
};

#endif
