/*

  SEHMAP.H - Map old-style structured exception handling to correct names.

  The mapping of structured exception handling statements from {try, except,
  finally, leave} to their proper names (prefaced by "__") has been removed
  from win32.mak.  This header is provided solely for compatibility with
  source code that used the older convention.

*/


#ifndef __cplusplus
#undef try
#undef except
#undef finally
#undef leave
#define try     __try
#define except  __except
#define finally __finally
#define leave   __leave
#endif
