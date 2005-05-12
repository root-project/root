// @(#)root/net:$Name:  $:$Id: TAlienCollection.h,v 1.3 2004/11/01 17:38:08 jgrosseo Exp $
// Author: Andreas-Joachim Peters 2005-05-09

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGridCollection
#define ROOT_TGridCollection

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGridCollection                                                      //
//                                                                      //
// Class which manages collection files on the Grid.                    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TObject
#include "TObject.h"
#endif


class TGridCollection : public TObject {
public:
   TGridCollection() { }
   virtual ~TGridCollection() { }

   ClassDef(TGridCollection,0)  // ABC managing collection of files on the Grid
};

#endif
