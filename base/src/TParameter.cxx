// @(#)root/base:$Name:  $:$Id: TObject.cxx,v 1.61 2004/06/04 16:28:30 brun Exp $
// Author: Maarten Ballintijn   21/06/2004

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TParameter<AParamType>                                               //
//                                                                      //
// Named parameter, streamable and storable.                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TParameter.h"

// explicit template instantiation of the versions specified in LinkDef.h
template class TParameter<Double_t>;
template class TParameter<Long_t>;


templateClassImp(TParameter)
