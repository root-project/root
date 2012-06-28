/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
// demo/exception/ehdemo.h
// This source has to be compiled. Refer to setup

#include <exception>
#include <string>

////////////////////////////////////////////////////////////////
// Derive your exception class from std::exception. Then Cint
// handles your exception by virtual function call
////////////////////////////////////////////////////////////////
class my_exception 
#ifdef STD
: public std::exception 
#else
: public exception 
#endif
{
  string msg;
 public:
  my_exception(const char* x) : msg(x) { }
  my_exception() : msg("This is my exception") { }
  my_exception(const my_exception& x) : msg(x.msg)  { }
  my_exception& operator=(const my_exception& x) { msg=x.msg;return(*this); }
  virtual const char* what() const throw(){ return(msg.c_str()); }
  virtual ~my_exception() throw() { }
};

////////////////////////////////////////////////////////////////
// If you do not derive from std::exception, Cint catches it but
// can not distinguish it
////////////////////////////////////////////////////////////////
class unhandled_exception {
 public:
  unhandled_exception() { }
  const char* what() const { return("this is not reported"); }
  ~unhandled_exception() { }
};

////////////////////////////////////////////////////////////////
// This compiled function is called by the interpreter
////////////////////////////////////////////////////////////////
void test(char* name) {
  if(!name)    throw my_exception("NULL pointer");
  if(!name[0]) throw my_exception("Empty string");
  if(strcmp(name,"error")==0) 
#ifdef STD
    throw std::exception();
#else
    throw exception();
#endif
  if(strcmp(name,"unknown")==0) throw unhandled_exception();
  printf("%s\n",name);
}
