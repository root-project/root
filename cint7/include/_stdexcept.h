/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifndef G__STDEXCEPT_H
#define G__STDEXCEPT_H

#include <exception>
#include <string>

//namespace std {

class logic_error : public exception {
 protected:
  string arg;
 public:
  explicit logic_error(const string& what_arg) {arg = what_arg;}
};

class domain_error : public logic_error {
 public:
  explicit domain_error(const string& what_arg) : logic_error(what_arg) { }
};

class invalid_argument : public logic_error {
 public:
  explicit invalid_argument(const string& what_arg) : logic_error(what_arg) { }
};

class length_error : public logic_error {
 public:
  explicit length_error(const string& what_arg) : logic_error(what_arg) { }
};

class out_of_range : public logic_error {
 public:
  explicit out_of_range(const string& what_arg) : logic_error(what_arg) { }
};

class runtime_error : public logic_error {
 public:
  explicit runtime_error(const string& what_arg) : logic_error(what_arg) { }
};

class range_error : public logic_error {
 public:
  explicit range_error(const string& what_arg) : logic_error(what_arg) { }
};

class overflow_error : public logic_error {
 public:
  explicit overflow_error(const string& what_arg) : logic_error(what_arg) { }
};

class underflow_error : public logic_error {
 public:
  explicit underflow_error(const string& what_arg) : logic_error(what_arg) { }
};


#ifdef __MAKECINT__

#pragma link off class exception;
#pragma link off class bad_exception;
#pragma link off function set_unexpected;
#pragma link off function unexpected;
#pragma link off function set_terminate;
#pragma link off function terminate;
#pragma link off function uncaught_exception;
#pragma link off typedef bool;

#endif

//}

#endif
