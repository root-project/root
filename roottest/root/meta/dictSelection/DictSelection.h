// Dear emacs, this is -*- c++ -*-
/*
This test has been provided by Attila Krasznahorkay.
*/
#ifndef DICTRULES_DICTSELECTION_H
#define DICTRULES_DICTSELECTION_H

// Get the active ROOT version:
#include <RVersion.h>

// Include the correct header:
#if ROOT_VERSION_CODE < ROOT_VERSION( 5, 99, 00 )
#   include <Reflex/Builder/DictSelection.h>
#else
#   include <RootMetaSelection.h>
#endif // ROOT 5

#if ROOT_VERSION_CODE < ROOT_VERSION( 5, 19, 0 )

// Definitions for *really* old ROOT versions:
#define ROOT_SELECTION_NS ROOT::Reflex::Selection
#define ENTER_ROOT_SELECTION_NS                                \
   namespace ROOT { namespace Reflex { namespace Selection {
#define EXIT_ROOT_SELECTION_NS }}}

#elif ROOT_VERSION_CODE < ROOT_VERSION( 5, 99, 0 )

// Definitions for ROOT 5:
#define ROOT_SELECTION_NS Reflex::Selection
#define ENTER_ROOT_SELECTION_NS                 \
   namespace Reflex { namespace Selection {
#define EXIT_ROOT_SELECTION_NS }}

#else

// Definitions for ROOT 6:
#define ROOT_SELECTION_NS ROOT::Meta::Selection
#define ENTER_ROOT_SELECTION_NS                             \
   namespace ROOT { namespace Meta { namespace Selection {
#define EXIT_ROOT_SELECTION_NS }}}

#endif // ROOT_VERSION

#endif // DICTRULES_DICTSELECTION_H
