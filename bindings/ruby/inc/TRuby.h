// @(#)root/ruby:$Id$
// Author:  Elias Athanasopoulos, May 2004

#ifndef ROOT_TRuby
#define ROOT_TRuby

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TRuby {

private:
  static Bool_t Initialize();
public:
  virtual ~TRuby() { }
  // execute a Ruby statement (e.g. "require 'ruby'")
  static void Exec(const char *cmd);

  // evaluate a Ruby expression (e.g. "1+1")
  static TObject *Eval(const char *expr);

  // bind a ROOT object with, at the ruby side, the name "label"
  static Bool_t Bind(TObject *obj, const char *label);

  // enter an interactive ruby session (exit with ^D)
  static void Prompt();

  ClassDef(TRuby,0)   //Ruby/ROOT interface
};

#endif
