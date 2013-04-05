// @(#)root/ruby:$Id$
// Author:  Elias Athanasopoulos, May 2004

/*  ruby-root
 *  
 *  Elias Athanasopoulos            <elathan@phys.uoa.gr>
 *  George Tzanakos (Supervisor)    <tzanakos@cc.uoa.gr>
 *    
 *  University of Athens 
 *  Department of Physics  
 *  HEPA Lab
 *  (http://daedalus.phys.uoa.gr)
 *  (c) 2003, 2004
 */

#ifndef rr_common_h
#define rr_common_h

#include "ruby.h"

#include "TObject.h"
#include "TList.h"
#include "TArrayC.h"
#include "TArrayS.h"
#include "TArrayI.h"
#include "TArrayL.h"
#include "TArrayF.h"
#include "TArrayD.h"
#include "TSeqCollection.h"

#include "CallFunc.h"

/* FIXME: This is from SWIG. */
#ifdef __cplusplus
#  ifndef RUBY_METHOD_FUNC	/* These definitions should work for Ruby 1.4.6 */
#    define VALUEFUNC(f) ((VALUE (*)()) f)
#    define VOIDFUNC(f)  ((void (*)()) f)
#  else
#    ifndef ANYARGS		/* These definitions should work for Ruby 1.6 */
#      define VALUEFUNC(f) ((VALUE (*)()) f)
#      define VOIDFUNC(f)  ((RUBY_DATA_FUNC) f)
#    else /* These definitions should work for Ruby 1.7 */
#      define VALUEFUNC(f) ((VALUE (*)(ANYARGS)) f)
#      define VOIDFUNC(f)  ((RUBY_DATA_FUNC) f)
#    endif
#  endif
#else
#  define VALUEFUNC(f) (f)
#  define VOIDFUNC(f) (f)
#endif

/* some usefull macros */

#define RRNEW(obj, type) obj = rb_class_new_instance (0, NULL, type)

#define RRGRAB(fromobj, type, toobj)                            \
    type *toobj;                                                \
    Data_Get_Struct (rb_iv_get (fromobj, "__rr__"), type, toobj)
 
#define RRCALL(obj, type)                                       \
    type *v;                                                    \
    Data_Get_Struct(rb_iv_get (obj, "__rr__"), type, v); ((type *)(v)) 

#define RRCALL2(obj, type, ptr)                                 \
    type *v;                                                    \
    Data_Get_Struct(rb_iv_get (obj, "__rr__"), type, v); ptr = v 

#define RRMODCALL(obj, modtype, convfunc)                       \
    modtype *v;                                                 \
    Data_Get_Struct (rb_iv_get (obj, "__rr__"), modtype, v);    \
    convfunc ((void**)&v, obj); v
    
#define RRMODCALL2(obj, modtype, convfunc, ptr)                 \
    modtype *v;                                                 \
    Data_Get_Struct (rb_iv_get (obj, "__rr__"), modtype, v);    \
    convfunc ((void**)&v, obj); ptr = v


#define RRSTRING(v) (TYPE(v) == T_STRING)
#define RRINT(v) (TYPE(v) == T_FIXNUM)
#define RRFLOAT(v) ((TYPE(v) == T_FLOAT) || (TYPE(v) == T_FIXNUM)) 
#define RRARRAY(v, kind) (TYPE(v) == T_ARRAY && kind(rb_ary_entry(v, 0)))
#define RRDATA(v) (TYPE(v) == T_OBJECT)
#define RRFUNC(v) (TYPE(v) == T_SYMBOL)
#define RRVOID(v) (v)

extern VALUE cTObject;

VALUE rr_bool (Bool_t q);

VALUE rr_ary_new (TList *l);
VALUE rr_arrayc_new (const TArrayC *a);
VALUE rr_arrays_new (const TArrayS *a);
VALUE rr_arrayi_new (const TArrayI *a);
VALUE rr_arrayl_new (const TArrayL *a);
VALUE rr_arrayf_new (const TArrayF *a);
VALUE rr_arrayd_new (const TArrayD *a);
VALUE rr_seqcollection_new (TSeqCollection *sc);

/* mod convertions */
void rr_tattfill_conv(void **ptr, VALUE klass);
void rr_tattline_conv(void **ptr, VALUE klass);
void rr_tattmarker_conv(void **ptr, VALUE klass);
void rr_tattpad_conv(void **ptr, VALUE klass);
void rr_tatttext_conv(void **ptr, VALUE klass);
void rr_tattaxis_conv(void **ptr, VALUE klass);


/* Map user defined C functions to Ruby methods.  */
struct rr_fcn_info {
    ID id;
    char *name;
};

/* TF1 user defined methods  */

double rr_ctf1_fcn (double *, double *);
void rr_register_ctf1_fcn (char *name, ID id);

/* TF2 user defined methods  */

double rr_ctf2_fcn (double *, double *);
void rr_register_ctf2_fcn (char *name, ID id);

/* Dynamic ruby-root specific.  */

struct drr_func_entry {
  G__CallFunc *func;
  G__ClassInfo *klass;
  char *name;
  char *cproto;
  int rtype;
};

struct drr_func_cache {
  struct drr_func_entry *entry;
  struct drr_func_cache *next;
  struct drr_func_cache *last;
};

/* Function cache.  */
struct drr_func_cache * drr_func_cache_init(struct drr_func_entry *entry);
void drr_func_cache_push (struct drr_func_cache *cache, struct drr_func_entry *entry);
struct drr_func_entry * drr_func_cache_find (struct drr_func_cache *cache, char *name);
void drr_func_entry_free (struct drr_func_entry *entry);

static VALUE drr_generic_method(int argc, VALUE argv[], VALUE self);
static VALUE drr_method_missing(int argc, VALUE argv[], VALUE self);
        
#endif
