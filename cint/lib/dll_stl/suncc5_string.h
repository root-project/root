#ifndef G__SUNCC5_STRING_H
#define G__SUNCC5_STRING_H

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
  const char _RWSTDExportFunc(*) __rwse_InvalidSizeParam=0;
  const char _RWSTDExportFunc(*) __rwse_PosBeyondEndOfString=0;
  const char _RWSTDExportFunc(*) __rwse_ResultLenInvalid=0;
  const char _RWSTDExportFunc(*) __rwse_StringIndexOutOfRange=0;
  const char _RWSTDExportFunc(*) __rwse_UnexpectedNullPtr=0;
#endif
}
#endif

#endif
