// @(#)root/net:$Id$
// Author: Fons Rademakers   25/11/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSQLRow                                                              //
//                                                                      //
// Abstract base class defining interface to a row of a SQL query       //
// result. Objects of this class are created by TSQLResult methods.     //
//                                                                      //
// Related classes are TSQLServer and TSQLResult.                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSQLRow.h"

ClassImp(TSQLRow);
