/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifndef G__EXCEPTION_H
#define G__EXCEPTION_H

//namespace std {

class exception;
class bad_exception;
typedef void (*unexpected_handler)();
unexpected_handler set_unexpected(unexpected_handler f) /* throw() */ ;
void unexpected();
typedef void (*terminate_handler)();
terminate_handler set_terminate(terminate_handler f) /* throw() */ ;
void terminate();
bool uncaught_exception();

/////////////////////////////////////////////////////////////////////////
class exception {
 public:
  exception() /* throw() */ { msg=0; }
  exception(const exception& x) /* throw() */ {
    if(x.msg) {
      msg = new char[strlen(x.msg)+1];
      strcpy(msg,x.msg);
    }
    else msg = 0;
  }
  exception& operator=(const exception& x) /* throw() */ {
    delete[] msg;
    if(x.msg) {
      msg = new char[strlen(x.msg)+1];
      strcpy(msg,x.msg);
    }
    else msg = 0;
  }
  virtual ~exception() /* throw() */ { delete[] msg; }
  virtual const char* what() const /* throw() */{return(msg);}

  exception(const char* msgin) { 
    msg = new char[strlen(msgin)+1];
    strcpy(msg,msgin);
  }
 private:
  char* msg;
};

/////////////////////////////////////////////////////////////////////////
class bad_exception : public exception {
 public:
  bad_exception() /* throw() */ {}
  bad_exception(const bad_exception&) /* throw() */ {}
  bad_exception& operator=(const bad_exception&) /* throw() */ {}
  virtual ~bad_exception() /* throw() */ {}
  virtual const char* what() const /* throw() */ {return("Unknown bad_exception");} 
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
