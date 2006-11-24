/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include <string>
#include <stdio.h>

class A {
  public:
    A(): mString("default") {}
    A(const A &s): mString(s.mString) {}
    A(const char *s): mString(s) {}
    virtual ~A() {}

    A operator +(const char *s)
      { A r = mString.c_str(); r.mString += s; 
        return r; }

    A& operator=(const A& s) {
       // CINT's default operator= is not appropriate in this case (it does memcpy)
       mString = s.mString;
       return *this;
    }

    const char *val() const { return mString.c_str(); }
    operator const char*() const { return mString.c_str(); }

  private:
    std::string mString;

    //ClassDef(A, 1)
};

void f(const char *a,const char *b) {
  printf("%s. %s.\n",a,b);
}


