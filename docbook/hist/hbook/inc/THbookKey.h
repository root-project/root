// @(#)root/hbook:$Id$
// Author: Rene Brun   20/02/2002

/*************************************************************************
 * Copyright (C) 1995-2002, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_THbookKey
#define ROOT_THbookKey


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THbookKey                                                            //
//                                                                      //
// Hbook id descriptor                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_THbookFile
#include "THbookFile.h"
#endif

class THbookKey : public TNamed {

protected:
   THbookFile    *fDirectory;   //!pointer to the Hbook file
   Int_t          fID;          //hbook identifier

public:
   THbookKey() : fDirectory(0),fID(0) {;}
   THbookKey(Int_t id, THbookFile *file);
   virtual ~THbookKey();
   virtual void      Browse(TBrowser *b);
   Bool_t            IsFolder() const;

   ClassDef(THbookKey,1)  //Hbook id descriptor
};

#endif
