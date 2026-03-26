// @(#)root/geom:$Id$
// Author : Andrei Gheata 10/06/02
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CLING__

// Third-party BVH headers
#include <bvh2_third_party.h>
#pragma extra_include "bvh2_third_party.h";
#pragma link C++ struct bvh::v2::Bvh < bvh::v2::Node < float, 3, sizeof(float) * CHAR_BIT, 4>> + ;
#pragma link C++ struct bvh::v2::Node < float, 3, sizeof(float) * CHAR_BIT, 4> + ;
#pragma link C++ struct bvh::v2::Index < sizeof(float) * CHAR_BIT, 4> + ;

#endif
