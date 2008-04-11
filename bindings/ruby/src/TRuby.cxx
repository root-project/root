// @(#)root/ruby:$Id$
// Author:  Elias Athanasopoulos, May 2004
//
// Interface for the Ruby shell.
//
// (c) 2004 - Elias Athanasopoulos  <elathan@phys.uoa.gr>
//
//

#include "TRuby.h"

#include "TROOT.h"
#include "TSystem.h"

#include "ruby.h"

ClassImp(TRuby)

extern VALUE cTObject;

bool TRuby::Initialize()
{
    static int IsInitialized = 0;

    if (!IsInitialized)
      {
        ruby_init();
        IsInitialized = 1;
      }

    return true;
}

void TRuby::Exec(const char *cmd)
{
    int state = 0;

    TRuby::Initialize();
    rb_eval_string_protect(cmd, &state);

    /* Print error if needed.  */
    if (state) rb_eval_string("puts $!");
}

TObject *TRuby::Eval(const char* expr)
{
    TObject *res;
    int state = 0;

    TRuby::Initialize();
    VALUE ret = rb_eval_string_protect(expr, &state);

    /* Print error if needed.  */
    if (state)
      {
        rb_eval_string("puts $!");
        return (TObject*)(0);
      }

    if (NIL_P(ret)) return (TObject*)0;

    /* Return the instance pointer if it is a ROOT
     * object.
     */
    VALUE ptr = rb_iv_get(ret, "__rr__");
    if (!NIL_P(ptr))
      {
        Data_Get_Struct(rb_iv_get(ret, "__rr__"), TObject, res);
        return res;
      }

    return (TObject*)0;
}

bool TRuby::Bind(TObject *obj, const char *label)
{
    VALUE *v = ALLOC(VALUE);

    *v = rb_class_new_instance (0, 0, cTObject);

    rb_iv_set(*v, "__rr__", Data_Wrap_Struct (cTObject, 0, 0, obj));
    rb_define_variable(label, v);

    return true;
}

void TRuby::Prompt()
{
    gSystem->Exec("irb");
}
