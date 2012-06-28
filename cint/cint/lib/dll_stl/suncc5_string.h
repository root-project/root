/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifndef G__SUNCC5_STRING_H
#define G__SUNCC5_STRING_H

#ifdef __CINT__
#define _RWSTDExport
#endif

#if (__SUNPRO_CC>=1280)
//#define _RWSTD_COMPILE_INSTANTIATE
namespace __rwstd {
#ifdef _RWSTD_LOCALIZED_ERRORS
  const unsigned int _RWSTDExport __rwse_InvalidSizeParam=0;
  const unsigned int _RWSTDExport __rwse_PosBeyondEndOfString=0;
  const unsigned int _RWSTDExport __rwse_ResultLenInvalid=0;
  const unsigned int _RWSTDExport __rwse_StringIndexOutOfRange=0;
  const unsigned int _RWSTDExport __rwse_UnexpectedNullPtr=0;
#else
  const char * __rwse_InvalidSizeParam=0;
  const char * __rwse_PosBeyondEndOfString=0;
  const char * __rwse_ResultLenInvalid=0;
  const char * __rwse_StringIndexOutOfRange=0;
  const char * __rwse_UnexpectedNullPtr=0;
#endif
}
#endif

#endif
