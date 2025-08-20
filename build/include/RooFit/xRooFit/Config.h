/*
 * Project: RooFit
 * Author:
 *   Will Buttinger, RAL 2022
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef xRooFit_Config_h
#define xRooFit_Config_h

// ROOT configuration: all of xRooFit is placed into a detail namespace
#define XROOFIT_NAMESPACE ROOT::Experimental::XRooFit

// Define XROOFIT_USE_PRAGMA_ONCE if you want to use "pragma once" instead of
// header guards
// # define XROOFIT_USE_PRAGMA_ONCE

#ifdef XROOFIT_NAMESPACE
#define BEGIN_XROOFIT_NAMESPACE namespace XROOFIT_NAMESPACE {
#define END_XROOFIT_NAMESPACE } // namespace XROOFIT_NAMESPACE
#else
#define BEGIN_XROOFIT_NAMESPACE
#define END_XROOFIT_NAMESPACE
#endif

#endif
