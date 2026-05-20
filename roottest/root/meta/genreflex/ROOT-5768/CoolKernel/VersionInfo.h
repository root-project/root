// $Id: VersionInfo.h,v 1.12.2.7 2013-05-07 15:09:38 avalassi Exp $
#ifndef COOLKERNEL_VERSIONINFO_H
#define COOLKERNEL_VERSIONINFO_H 1

// Explicitly disable the COOL300 extensions
#undef COOL300    // COOL 2.x.x

// This switch is now hardcoded in the two branches of the code
// tagged as COOL-preview and COOL_2_8-patches (bug #92204).
//#define COOL290 1 // COOL 2.9.x
#undef COOL290      // COOL 2.8.x

// COOL_VERSIONINFO_RELEASE is #defined in API as of COOL 2.8.4 (sr #111706)
// COOL_VERSIONINFO_RELEASE_x are #defined as of COOL 2.8.15
// Note that the former is defined within quotes, the latter are not!

#ifdef COOL290
//---------------------------------------------------------------------------
// COOL-preview (COOL 2.9.x releases)
// Disable all extensions (do not allow explicit -D to enable them)
//---------------------------------------------------------------------------
#define COOL_VERSIONINFO_RELEASE_MAJOR 2
#define COOL_VERSIONINFO_RELEASE_MINOR 9
#define COOL_VERSIONINFO_RELEASE_PATCH 0
#define COOL_VERSIONINFO_RELEASE "2.9.0"
#define COOL290CO 1 // API fixes for Coverity (bugs #95363 and #95823)
#define COOL290EX 1 // API fixes in inlined Exception method (bug #68061)
#define COOL290VP 1 // API extension for vector payload (task #10335)
//---------------------------------------------------------------------------
#else
//---------------------------------------------------------------------------
// COOL_2_8-patches (COOL 2.8.x releases)
// Disable all extensions (do not allow explicit -D to enable them)
//---------------------------------------------------------------------------
#define COOL_VERSIONINFO_RELEASE_MAJOR 2
#define COOL_VERSIONINFO_RELEASE_MINOR 8
#define COOL_VERSIONINFO_RELEASE_PATCH 19
#define COOL_VERSIONINFO_RELEASE "2.8.19"
#undef COOL290CO // Do undef (do not leave the option to -D this explicitly)
#undef COOL290EX // Do undef (do not leave the option to -D this explicitly)
#undef COOL290VP // Do undef (do not leave the option to -D this explicitly)
//---------------------------------------------------------------------------
#endif

// Drop support for TimingReport as of COOL 2.8.15 (task #31638)
#undef COOL_ENABLE_TIMING_REPORT

// Define a portable macro for c++11 in COOL (only for COOL290 and ROOT6!)
#ifdef COOL290
#if defined(__GXX_EXPERIMENTAL_CXX0X) || __cplusplus >= 201103L
#include "RVersion.h"
#if ROOT_VERSION_CODE >= ROOT_VERSION(5,99,0)
#define COOL_HAS_CPP11 1
#else
// NB c++11 extensions are always disabled for ROOT5 (bug #103302)!
#undef COOL_HAS_CPP11
#endif
#else
// NB c++11 extensions are disabled if the compiler does not support c++11!
#undef COOL_HAS_CPP11
#endif
#else
// NB c++11 extensions are always disabled in the COOL28x API!
#undef COOL_HAS_CPP11
#endif

#endif // COOLKERNEL_VERSIONINFO_H
