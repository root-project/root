// @(#)root/tree:$Name:  $:$Id: TTreeFilePrefetch.h,v 1.1 2006/06/05 20:30:27 brun Exp $
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

class TTreeFilePrefetch : public TFilePrefetch {

protected:
   TTree        *fTree;       //!pointer to the TTree
   Long64_t      fEntryMax;   //last entry in the cache
   
protected:
   TTreeFilePrefetch(const TTreeFilePrefetch &);            //this class cannot be copied
   TTreeFilePrefetch& operator=(const TTreeFilePrefetch &);

public:
   TTreeFilePrefetch();
   TTreeFilePrefetch(TTree *tree, Int_t buffersize=0);
   virtual ~TTreeFilePrefetch();
   TTree              *GetTree() const {return fTree;}
   virtual Bool_t      ReadBuffer(char *buf, Long64_t pos, Int_t len);
   Bool_t              Register(Long64_t offset);
   void                SetEntryMax(Long64_t emax);
   void                SetTree(TTree *tree);
        
   ClassDef(TTreeFilePrefetch,1)  //Specialization of TFilePrefetch for a TTree 
};

#endif
