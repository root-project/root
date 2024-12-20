/*
 * Project: xRooFit
 * Author:
 *   Will Buttinger, RAL 2022
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "Config.h"

#ifdef XROOFIT_USE_PRAGMA_ONCE
#pragma once
#endif
#if !defined(XROOFIT_XROOHYPOSPACE_H) || defined(XROOFIT_USE_PRAGMA_ONCE)
#ifndef XROOFIT_USE_PRAGMA_ONCE
#define XROOFIT_XROOHYPOSPACE_H
#endif

#include "xRooNLLVar.h"

BEGIN_XROOFIT_NAMESPACE

class xRooHypoSpace : public xRooNLLVar::xRooHypoSpace {
public:
   using xRooNLLVar::xRooHypoSpace::xRooHypoSpace;
   ClassDef(xRooHypoSpace, 0)
};

END_XROOFIT_NAMESPACE

#endif // include guard
