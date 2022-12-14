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

#pragma once

#include "xRooNLLVar.h"

#ifdef XROOFIT_NAMESPACE
namespace XROOFIT_NAMESPACE {
#endif

class xRooHypoSpace : public xRooNLLVar::xRooHypoSpace {


    ClassDef(xRooHypoSpace,0)
};

#ifdef XROOFIT_NAMESPACE
}
#endif