/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifndef T1247
#define T1247

class A1 {
 private:
  A1(const A1& x) { }
 public:
  A1() { }
  friend class A2;
  void f(int x) { }
};

class A2 {
 private:
  A2() { }
 public:
  void f(int x) { }
};

class A3 {
#if defined(__ICC) && __ICC==800
 protected:
#else
 private:
#endif
  ~A3() { }
 public:
  friend class A2;
  void f(int x) { }
};

class A4 {
 private:
  A4& operator=(const A4& x) { return(*this); }
 public:
  void f(int x) { }
};

class C1 {
 protected:
  C1(const C1& x) { }
 public:
  C1() { }
  void f(int x) { }
};

class C2 {
 protected:
  C2() { }
 public:
  void f(int x) { }
};

class C3 {
 protected:
  ~C3() { }
 public:
  void f(int x) { }
};

class C4 {
 protected:
  C4& operator=(const C4& x) { return(*this); }
 public:
  void f(int x) { }
};
#endif
