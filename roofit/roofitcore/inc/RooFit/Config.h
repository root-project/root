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

namespace RooFit {

/// An alias for raw pointers for indicating that the return type of a RooFit
/// function is an owning pointer that must be deleted by the caller. For
/// RooFit developers, it can be very useful to make this an alias to
/// std::unique_ptr<T>, in order to check that your code has no memory
/// problems. Changing this alias is equivalent to forcing all code immediately
/// wraps the result of functions returning a RooFit::OwningPtr<T> in a
/// std::unique_ptr<T>.
template<typename T>
using OwningPtr = T*;
//using OwningPtr = std::unique_ptr<T>;

}

#endif
