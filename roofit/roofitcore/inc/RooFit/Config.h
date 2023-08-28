/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN 2023
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_Config_h
#define RooFit_Config_h

// Define ROOFIT_MEMORY_SAFE_INTERFACES to change RooFit interfaces to be
// memory safe.
//#define ROOFIT_MEMORY_SAFE_INTERFACES

// The memory safe interfaces mode implies that all RooFit::OwningPtr<T> are
// std::unique_ptr<T>.
#ifdef ROOFIT_MEMORY_SAFE_INTERFACES
#ifndef ROOFIT_OWNING_PTR_IS_UNIQUE_PTR
#define ROOFIT_OWNING_PTR_IS_UNIQUE_PTR
#endif
#endif

#include <memory>

namespace RooFit {

/// An alias for raw pointers for indicating that the return type of a RooFit
/// function is an owning pointer that must be deleted by the caller. For
/// RooFit developers, it can be very useful to make this an alias to
/// std::unique_ptr<T>, in order to check that your code has no memory
/// problems. Changing this alias is equivalent to forcing all code immediately
/// wraps the result of functions returning a RooFit::OwningPtr<T> in a
/// std::unique_ptr<T>.
template <typename T>
#ifdef ROOFIT_OWNING_PTR_IS_UNIQUE_PTR
using OwningPtr = std::unique_ptr<T>;
#else
using OwningPtr = T *;
#endif

namespace Detail {

/// Internal helper to turn a std::unique_ptr<T> into an OwningPtr.
template <typename T>
OwningPtr<T> owningPtr(std::unique_ptr<T> &&ptr)
{
#ifdef ROOFIT_OWNING_PTR_IS_UNIQUE_PTR
   return std::move(ptr);
#else
   return ptr.release();
#endif
}

} // namespace Detail

} // namespace RooFit

#endif
