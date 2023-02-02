/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN, Jan 2022
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef roofit_roofitcore_RooStringView_h
#define roofit_roofitcore_RooStringView_h

#include <ROOT/RStringView.hxx>
#include <TString.h>

#include <string>

/// The RooStringView is a wrapper around a C-syle string that can also be
/// constructed from a `std::string` or a Tstring. As such, it serves as a
/// drop-in replacement for `const char*` in public RooFit interfaces, keeping
/// the possibility to pass a C-style string without copying but also accepting
/// a `std::string`.

class RooStringView {
public:
   RooStringView(const char *str) : _cstr{str} {}
   RooStringView(TString const &str) : _cstr{str} {}
   RooStringView(std::string const &str) : _cstr{str.c_str()} {}
   // If the string is a temporary, we have to store it ourselves, otherwise the C-style string would be invalid.
   RooStringView(std::string &&str) : _strp{std::make_shared<std::string>(std::move(str))}, _cstr{_strp->c_str()} {}
   operator const char *() { return _cstr; }
   operator std::string_view() { return _cstr; }

private:
   std::shared_ptr<std::string> _strp;
   const char *_cstr;
};

#endif
