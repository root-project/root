/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// Author: Vincenzo Eduardo Padulano (CERN), 11/2025

#ifndef ROOT_IO_UTILS
#define ROOT_IO_UTILS

#include <optional>
#include <string>

namespace ROOT::Internal {

/// \brief Get extended attribute value from path
/// \param path Path to the file to check
/// \param xattr Extended attribute to evaluate
/// \return The string containing the extended attribute value if found, std::nullopt otherwise
std::optional<std::string> GetXAttrVal(const char *path, const char *xattr);

/// \brief Redirects the input URL to the equivalent XRootD path on EOS
/// \param inputUrl The input URL to redirect
/// \return The redirected URL in case of successful redirection, std::nullopt otherwise
std::optional<std::string> GetEOSRedirectedXRootURL(const char *inputURL);
} // namespace ROOT::Internal

#endif
