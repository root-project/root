// @(#)root/tree:$Name:  $:$Id: TTreeFilePrefetch.h,v 1.4 2006/06/14 13:15:55 brun Exp $
// Author: Rene Brun   04/06/2006

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreeFilePrefetch
#define ROOT_TTreeFilePrefetch


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTreeFilePrefetch                                                    //
//                                                                      //
// Specialization of TFilePrefetch for a TTree                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TFilePrefetch
#include "TFilePrefetch.h"
#endif

class TTree;
class TBranch;

class TTreeFilePrefetch : public TFilePrefetch {

protected:
   Long64_t        fEntryMin;    //! first entry in the cache
   Long64_t        fEntryMax;    //! last entry in the cache
   Long64_t        fEntryNext;   //! next entry number where cache must be filled
   Int_t           fNbranches;   //! Number of branches in the cache
   TBranch       **fBranches;    //! [fNbranches] List of branches to be stored in the cache
   Bool_t          fIsLearning;  //! true if cache is in learning mode
   static Double_t fgLearnRatio; //fraction of entries used for learning mode

protected:
   TTreeFilePrefetch(const TTreeFilePrefetch &);            //this class cannot be copied
   TTreeFilePrefetch& operator=(const TTreeFilePrefetch &);

public:
   TTreeFilePrefetch();
   TTreeFilePrefetch(TTree *tree, Int_t buffersize=0);
   virtual ~TTreeFilePrefetch();
   void                AddBranch(TBranch *b);
   void                Clear(Option_t *option="");
   static Double_t     GetLearnRatio();
   Bool_t              FillBuffer();
   TTree              *GetTree() const;
   Bool_t              IsLearning() const {return fIsLearning;}
   virtual Bool_t      ReadBuffer(char *buf, Long64_t pos, Int_t len);
   void                SetEntryRange(Long64_t emin,   Long64_t emax);
   static void         SetLearnRatio(Double_t ratio=0.01);
        
   ClassDef(TTreeFilePrefetch,1)  //Specialization of TFilePrefetch for a TTree 
};

#endif
