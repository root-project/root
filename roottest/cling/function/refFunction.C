#if defined(__MAKECLING__) || !defined(__ICLING__)
#include "refClasses.cxx"
#endif

zz refFunction(const yy& arg="def-xx")
{
  printf("xx -- arg='%s'\n", arg.Data());
  return arg;
}
