// Author: Roel Aaij 30/08/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGTable.h"
#include "TClass.h"
#include "TGWindow.h"
#include "TGResourcePool.h"
#include "Riostream.h"
#include "TSystem.h"
#include "TImage.h"
#include "TEnv.h"
#include "TGToolTip.h"
#include "TGPicture.h"
#include "TRandom3.h"
#include "TTreeTableInterface.h"
#include "TGTreeTable.h"

ClassImp(TGTreeTable)

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGTreeTable                                                          //
//                                                                      //
// TGTreeTable is a TGTable that owns it's own interface, it            //
// can be used to view a TTree. If an expression is given to the        //
// constuctor, it will be used to define the columns. A selection can   //
// also be given. This selection is applied to the TTree as a           //
// TEntryList. See the documentation of TGTable for more information    //
//                                                                      //
// The interface is accesible after the creation through the            //
// GetInterface() method.                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TGTreeTable::TGTreeTable(TGWindow *p, Int_t id, TTree *tree,
                         const char *expression, const char *selection,
                         const char *option, UInt_t nrows, UInt_t ncolumns)
   : TGTable(p, id, 0, nrows, ncolumns)
{
   // TGTreeTable constructor.

   TTreeTableInterface *iface = new TTreeTableInterface(tree, expression,
                                                        selection, option);
   SetInterface(iface, nrows, ncolumns);
   Update();
}

//______________________________________________________________________________
TGTreeTable::~TGTreeTable()
{
   // TGTreeTable destructor.

   //FIXME this causes a double delete segfault, why???
//    delete fInterface;
}

