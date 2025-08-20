/// \cond ROOFIT_INTERNAL

/*
 * Project: RooFit
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_RefCountList_h
#define RooFit_RefCountList_h

#include <RooLinkedList.h>

class RooRefCountList : public RooLinkedList {
public:
   RooRefCountList() {}
   ClassDefOverride(RooRefCountList, 1) // RooLinkedList alias for backwards compatibility
};

#endif

/// \endcond
