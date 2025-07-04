// @(#)root/io:$Id$
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

#include "TObject.h"


class TFree : public TObject {
   /// Prevent gap coalescing from exceeding the 1 GiB TKey limit
   /// (and the 2 GiB limit given by the fact that the gap size is stored as a signed 4 byte integer).
   /// The limit has a safety buffer until the actual 1 GiB limit to avoid possible corner cases very close to the
   /// 1 GiB limit.
   static constexpr Int_t kMaxGapSize = 1000 * 1000 * 1000;

protected:
   Long64_t        fFirst;            ///<First free word of segment
   Long64_t        fLast;             ///<Last free word of segment

public:
   TFree();
   TFree(TList *lfree, Long64_t first, Long64_t last);
   ~TFree() override;
           TFree    *AddFree(TList *lfree, Long64_t first, Long64_t last);
   virtual void      FillBuffer(char *&buffer);
           TFree    *GetBestFree(TList *lfree, Int_t nbytes);
           Long64_t  GetFirst() const {return fFirst;}
           Long64_t  GetLast() const {return fLast;}
           void      ls(Option_t * = "") const override;
   virtual void      ReadBuffer(char *&buffer);
           void      SetFirst(Long64_t first) {fFirst=first;}
           void      SetLast(Long64_t last) {fLast=last;}
           Int_t     Sizeof() const;

   ClassDefOverride(TFree,1);  //Description of free segments on a file
};

#endif
