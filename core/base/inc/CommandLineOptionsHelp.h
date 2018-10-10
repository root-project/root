/* @(#)root/base:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_CommandLineOptionsHelp
#define ROOT_CommandLineOptionsHelp


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// CommandLineOptionsHelp                                               //
//                                                                      //
// Help text displayed by TApplication and rootx.cxx.                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

/// Help for command line options.
///
///  - b : run in batch mode without graphics
///  - x : exit on exception
///  - e expression: request execution of the given C++ expression.
///  - n : do not execute logon and logoff macros as specified in .rootrc
///  - q : exit after processing command line macro files
///  - l : do not show splash screen
///
/// The last three options are only relevant in conjunction with TRint.
/// The following help and info arguments are supported:
///
///  - ?       : print usage
///  - h       : print usage
///  - -help   : print usage
///  - config  : print ./configure options
///  - memstat : run with memory usage monitoring
///
/// In addition to the above options the arguments that are not options,
/// i.e. they don't start with - or + are treated as follows (and also removed
/// from the argument array):
///
///  - `<dir>`       is considered the desired working directory and available
///                  via WorkingDirectory(), if more than one dir is specified the
///                  first one will prevail
///  - `<file>`      if the file exists its added to the InputFiles() list
///  - `<file>.root` are considered ROOT files and added to the InputFiles() list,
///                  the file may be a remote file url
///  - `<macro>.C`   are considered ROOT macros and also added to the InputFiles() list
///
/// In TRint we set the working directory to the `<dir>`, the ROOT files are
/// connected, and the macros are executed. If your main TApplication is not
/// TRint you have to decide yourself what to do with these options.
/// All specified arguments (also the ones removed) can always be retrieved
/// via the TApplication::Argv() method.

constexpr static const char kCommandLineOptionsHelp[] = R"RAW(
Usage: %s [-l] [-b] [-n] [-q] [dir] [[file:]data.root] [file1.C ... fileN.C]
Options:
  -b : run in batch mode without graphics
  -x : exit on exception
  -e expression: request execution of the given C++ expression
  -n : do not execute logon and logoff macros as specified in .rootrc
  -q : exit after processing command line macro files
  -l : do not show splash screen
  -t : enable thread-safety and implicit multi-threading (IMT)
 --web: display graphics in a default web browser
 --web=<browser>: display graphics in specified web browser
 --notebook : execute ROOT notebook
 dir : if dir is a valid directory cd to it before executing

  -?      : print usage
  -h      : print usage
  --help  : print usage
  -config : print ./configure options
  -memstat : run with memory usage monitoring

)RAW";

#endif
