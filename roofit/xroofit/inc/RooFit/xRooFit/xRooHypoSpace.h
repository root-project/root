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

#ifndef XROOFIT_NAMESPACE
#pragma once
#endif
#if !defined(XROOFIT_XROOHYPOSPACE_H) || !defined(XROOFIT_NAMESPACE)
#define XROOFIT_XROOHYPOSPACE_H

#include "xRooNLLVar.h"

#ifdef XROOFIT_NAMESPACE
namespace XROOFIT_NAMESPACE {
#endif

class xRooHypoSpace : public xRooNLLVar::xRooHypoSpace {

   ClassDef(xRooHypoSpace, 0)
};

#ifdef XROOFIT_NAMESPACE
}
#endif

#endif // include guard