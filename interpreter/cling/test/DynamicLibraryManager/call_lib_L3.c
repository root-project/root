/*------------------------------------------------------------------------------
// CLING - the C++ LLVM-based InterpreterG :)
//
// This file is dual-licensed: you can choose to license it under the University
// of Illinois Open Source License or the GNU Lesser General Public License. See
// LICENSE.TXT for details.
//----------------------------------------------------------------------------*/

// RUN: true
// Used as library source by callale_lib_L3.C, etc.
extern int cling_testlibrary_function();

CLING_EXPORT int cling_testlibrary_function3() {
  return cling_testlibrary_function() + 3;
}
