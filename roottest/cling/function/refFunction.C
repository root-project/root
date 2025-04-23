#if defined(__MAKECINT__) || !defined(__CINT__)
#include "refClasses.cxx"
#endif

zz refFunction(const yy& arg="def-xx")
{
  printf("xx -- arg='%s'\n", arg.Data());
  return arg;
}
