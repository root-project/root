/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifndef P2F_H
#define P2F_H

class ExecP2F {
 public:
  void SetP2F(void* p);
  int DoP2F(char* s,double d);
 // private:
  void *p2f;
  int mode;
  char *fname;
};


/* example of pointer to interface method */
#ifdef __CINT__
int InterfaceMethod(char* s, double d) {
  int result;
  result = sprintf(s,"InterfaceMethod %g\n",d);
  return(result);
}
#else
#define InterfaceMethod(s,d)                   \
   sprintf(s,"InterfaceMethod %g\n",d)
#endif

/* example of pointer to compiled function */
extern int CompiledFunc(char* s,double d); 

#endif
