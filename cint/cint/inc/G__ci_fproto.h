/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * CINT header file G__ci_fproto.h
 ************************************************************************
 * Description:
 *  C/C++ interpreter header file for API function prototypes
 ************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifndef G__CI_FPROTO_INCLUDE
#define G__CI_FPROTO_INCLUDE

#if defined(__clang__) && !defined(__ICC)
# if __has_warning("-Wreturn-type-c-linkage")
/* 'G__getfunction' has C-linkage specified, but returns
   user-defined type 'G__value' which is incompatible with C
*/
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif

G__DECL_API(0, unsigned long, G__uint, (G__value buf))
G__DECL_API(1, int, G__fgetline, (char *string))
G__DECL_API(2, int, G__load, (char *commandfile))
/* G__DECL_API(???, float, G__float, (G__value buf))*/
G__DECL_API(221, int, G__globalsetup,(void))
G__DECL_API(3, int, G__call_setup_funcs, (void))
G__DECL_API(4, void, G__reset_setup_funcs, (void))
G__DECL_API(5, G__CONST char*, G__cint_version, (void))
G__DECL_API(6, void, G__init_garbagecollection, (void))
G__DECL_API(7, int, G__garbagecollection, (void))
G__DECL_API(8, void, G__add_alloctable, (void* allocedmem,int type,int tagnum))
G__DECL_API(9, int, G__del_alloctable, (void* allocmem))
G__DECL_API(10, int, G__add_refcount, (void* allocedmem,void** storedmem))
G__DECL_API(11, int, G__del_refcount, (void* allocedmem,void** storedmem))
G__DECL_API(12, int, G__disp_garbagecollection, (FILE* fout))
G__DECL_API(13, struct G__ifunc_table*, G__get_methodhandle, (const char *funcname,const char *argtype
                                           ,struct G__ifunc_table *p_ifunc
                                           ,long *pifn,long *poffset
                                           ,int withConversion
                                           ,int withInheritance))
G__DECL_API(14, struct G__ifunc_table*, G__get_methodhandle2, (char *funcname
                                           ,struct G__param* libp
                                           ,struct G__ifunc_table *p_ifunc
                                           ,long *pifn,long *poffset
                                           ,int withConversion
                                           ,int withInheritance))
G__DECL_API(15, struct G__var_array*, G__searchvariable, (char *varname,int varhash
                                       ,struct G__var_array *varlocal
                                       ,struct G__var_array *varglobal
                                       ,long *pG__struct_offset
                                       ,long *pstore_struct_offset
                                       ,int *pig15
                                       ,int isdecl))

G__DECL_API(16, struct G__ifunc_table*, G__p2f2funchandle, (void* p2f,struct G__ifunc_table* p_ifunc,int* pindex))
G__DECL_API(17, char*, G__p2f2funcname, (void *p2f))
G__DECL_API(18, int, G__isinterpretedp2f, (void* p2f))
G__DECL_API(19, int, G__compile_bytecode, (struct G__ifunc_table* ifunc,int index_in_ifunc))


G__DECL_API(20, void, G__va_arg_setalign, (int n))
G__DECL_API(21, void, G__va_arg_copyvalue, (int t,void* p,G__value* pval,int objsize))

/*************************************************************************
* ROOT script compiler
*************************************************************************/
G__DECL_API(22, void, G__Set_RTLD_NOW, (void))
G__DECL_API(23, void, G__Set_RTLD_LAZY, (void))
G__DECL_API(24, void, G__RegisterScriptCompiler, (int(*p2f)(G__CONST char*,G__CONST char*)))
/*************************************************************************
* Pointer to function evaluation function
*************************************************************************/

/*************************************************************************
* G__atpause, G__aterror API
*************************************************************************/

/*************************************************************************
* interface method setup functions
*************************************************************************/

G__DECL_API(25, int, G__defined_tagname, (G__CONST char* tagname,int noerror))
G__DECL_API(26, struct G__Definedtemplateclass *,G__defined_templateclass, (G__CONST char *name))

G__DECL_API(27, int, G__deleteglobal, (void* p))
G__DECL_API(269, int, G__resetglobalvar, (void* p))
G__DECL_API(28, int, G__deletevariable, (G__CONST char* varname))
G__DECL_API(29, int, G__optimizemode, (int optimizemode))
G__DECL_API(30, int, G__getoptimizemode, (void))
G__DECL_API(31, G__value, G__string2type_body, (G__CONST char *typenamin,int noerror))
G__DECL_API(32, G__value, G__string2type, (G__CONST char *typenamin))
G__DECL_API(33, void*, G__findsym, (G__CONST char *fname))

G__DECL_API(34, int, G__IsInMacro, (void))
G__DECL_API(35, void, G__storerewindposition, (void))
G__DECL_API(36, void, G__rewinddictionary, (void))
G__DECL_API(37, void, G__SetCriticalSectionEnv, (int (*issamethread)()
                              ,void (*storelockthread)()
                              ,void (*entercs)()
                              ,void (*leavecs)()))

G__DECL_API(38, void, G__storelasterror, (void))


G__DECL_API(39, void, G__set_smartunload, (int smartunload))

G__DECL_API(40, void, G__set_autoloading, (int (* /*p2f*/) (char*)))

G__DECL_API(41, void, G__set_class_autoloading_callback, (int (* /*p2f*/) (char*, char*)))
G__DECL_API(42, void, G__set_class_autoloading_table, (char* classname, char* libname))
G__DECL_API(259, char*, G__get_class_autoloading_table, (char* classname))
G__DECL_API(43, int, G__set_class_autoloading, (int newvalue))

#ifdef G__NEVER
G__DECL_API(44, void*, G__operator_new, (size_t size,void* p))
G__DECL_API(45, void*, G__operator_new_ary, (size_t size,void* p))
G__DECL_API(46, void, G__operator_delete, (void *p))
G__DECL_API(47, void, G__operator_delete_ary, (void *p))
#endif

G__DECL_API(48, int, G__getexitcode, (void))
G__DECL_API(49, int, G__get_return, (int *exitval))

#ifndef G__OLDIMPLEMENTATION1485
G__DECL_API(50, int, G__fprinterr, (FILE* fp,const char* fmt,...))
G__DECL_API(51, int, G__fputerr, (int c))
#else
#define G__fprinterr  fprintf
#endif

G__DECL_API(52, void, G__SetUseCINTSYSDIR, (int UseCINTSYSDIR))
G__DECL_API(53, void, G__SetCINTSYSDIR, (char* cintsysdir))
G__DECL_API(54, void, G__set_eolcallback, (void* eolcallback))
G__DECL_API(55, G__parse_hook_t*, G__set_beforeparse_hook, (G__parse_hook_t* hook))
G__DECL_API(56, void, G__set_ioctortype_handler, (int (* /*p2f*/) (const char*)))
G__DECL_API(57, void, G__SetCatchException, (int mode))

#ifdef G__ASM_WHOLEFUNC
/**************************************************************************
* Interface method to run bytecode function
**************************************************************************/
G__DECL_API(58, int, G__exec_bytecode, (G__value *result7,G__CONST char *funcname,struct G__param *libp,int hash))
#endif


/**************************************************************************
 * Exported Cint API functions
 **************************************************************************/
#ifdef G__WIN32
G__DECL_API(59, int, G__fprintf, (FILE* fp,const char* fmt,...))
G__DECL_API(60, int, G__setmasksignal, (int))
#endif
G__DECL_API(61, void, G__settemplevel, (int val))
G__DECL_API(62, void, G__clearstack, (void))
G__DECL_API(63, int, G__lasterror, (void))
G__DECL_API(64, void, G__reset_lasterror, (void))
G__DECL_API(65, int, G__gettempfilenum, (void))
G__DECL_API(66, void, G__LockCpp, (void))
G__DECL_API(67, void, G__set_sym_underscore, (int x))
G__DECL_API(68, int, G__get_sym_underscore, (void))
G__DECL_API(69, void*, G__get_errmsgcallback, (void))
G__DECL_API(70, void, G__mask_errmsg, (char* msg))
G__DECL_API(71, int, G__main, (int argc,char **argv))
G__DECL_API(72, void, G__setothermain, (int othermain))
G__DECL_API(73, void, G__exit, (int rtn))
G__DECL_API(74, int, G__getnumbaseclass, (int tagnum))
G__DECL_API(75, void, G__setnewtype, (char globalcomp,G__CONST char* comment,int nindex))
G__DECL_API(76, void, G__setnewtypeindex, (int j,int type_index))
G__DECL_API(77, void, G__resetplocal, (void))
G__DECL_API(78, long, G__getgvp, (void))
G__DECL_API(79, void, G__resetglobalenv, (void))
G__DECL_API(80, void, G__lastifuncposition, (void))
G__DECL_API(81, void, G__resetifuncposition, (void))
G__DECL_API(82, void, G__setnull, (G__value* result))
G__DECL_API(83, long, G__getstructoffset, (void))
G__DECL_API(84, int, G__getaryconstruct, (void))
G__DECL_API(85, long, G__gettempbufpointer, (void))
G__DECL_API(86, void, G__setsizep2memfunc, (int sizep2memfunc))
G__DECL_API(87, int, G__getsizep2memfunc, (void))
G__DECL_API(88, int, G__get_linked_tagnum, (G__linked_taginfo *p))
G__DECL_API(257, int, G__get_linked_tagnum_fwd, (G__linked_taginfo *p))
G__DECL_API(89, int, G__tagtable_setup, (int tagnum,int size,int cpplink,int isabstract,G__CONST char *comment,G__incsetup setup_memvar,G__incsetup setup_memfunc))
G__DECL_API(90, int, G__search_tagname, (G__CONST char *tagname,int type))
G__DECL_API(91, int, G__search_typename, (G__CONST char *typenamein,int typein,int tagnum,int reftype))
G__DECL_API(92, int, G__defined_typename, (G__CONST char* typenamein))
G__DECL_API(93, int, G__tag_memvar_setup, (int tagnum))
G__DECL_API(94, int, G__memvar_setup, (void *p,int type,int reftype,int constvar,int tagnum,int typenum,int statictype,int var_access,G__CONST char *expr,int definemacro,G__CONST char *comment))
G__DECL_API(95, int, G__tag_memvar_reset, (void))
G__DECL_API(96, int, G__tag_memfunc_setup, (int tagnum))

#ifdef G__TRUEP2F
G__DECL_API(97, int, G__memfunc_setup, (G__CONST char *funcname,int hash,G__InterfaceMethod funcp,int type
,int tagnum,int typenum,int reftype,int para_nu,int ansi,int var_access,int isconst,G__CONST char *paras,G__CONST char *comment,void* tp2f,int isvirtual))
#else /* G__TRUEP2F */
G__DECL_API(98, int, G__memfunc_setup, (G__CONST char *funcname,int hash,G__InterfaceMethod funcp,int type
,int tagnum,int typenum,int reftype,int para_nu,int ansi,int var_access,int isconst,G__CONST char *paras,G__CONST char *comment))
#endif /* G__TRUEP2F */

G__DECL_API(99, int, G__memfunc_next, (void))
G__DECL_API(101, int, G__tag_memfunc_reset, (void))
G__DECL_API(100, int, G__value_get_type, (G__value* buf))
G__DECL_API(262, int, G__value_get_tagnum, (G__value* buf))
G__DECL_API(102, void, G__letint, (G__value *buf,int type,long value))
G__DECL_API(103, void, G__letdouble, (G__value *buf,int type,double value))
G__DECL_API(104, void, G__store_tempobject, (G__value reg))
G__DECL_API(105, int, G__inheritance_setup, (int tagnum,int basetagnum,long baseoffset,int baseaccess,int property))
G__DECL_API(106, void, G__add_compiledheader, (G__CONST char *headerfile))
G__DECL_API(107, void, G__add_ipath, (G__CONST char *ipath))
G__DECL_API(108, int, G__delete_ipath, (G__CONST char *ipath))
G__DECL_API(109, struct G__includepath*, G__getipathentry, ())
G__DECL_API(110, void, G__add_macro, (G__CONST char *macro))
G__DECL_API(111, void, G__check_setup_version, (int version,G__CONST char *func))
G__DECL_API(112, long, G__int, (G__value buf))
G__DECL_API(113, long, G__int_cast, (G__value buf))
G__DECL_API(114, double, G__double, (G__value buf))
G__DECL_API(115, G__value, G__calc, (G__CONST char *expr))
G__DECL_API(116, int , G__loadfile, (G__CONST char* filename))
G__DECL_API(117, int , G__unloadfile, (G__CONST char* filename))
G__DECL_API(258, int , G__setfilecontext, (G__CONST char* filename, struct G__input_file* ifile))
G__DECL_API(118, int, G__init_cint, (G__CONST char* command))
G__DECL_API(119, void, G__scratch_all, (void))
G__DECL_API(120, void, G__setdouble, (G__value *pbuf,double d,void* pd,int type,int tagnum,int typenum,int reftype))
G__DECL_API(121, void, G__setint, (G__value *pbuf,long d,void* pd,int type,int tagnum,int typenum,int reftype))
G__DECL_API(122, void, G__stubstoreenv, (struct G__StoreEnv *env,void* p,int tagnum))
G__DECL_API(123, void, G__stubrestoreenv, (struct G__StoreEnv *env))
G__DECL_API(124, int, G__getstream, (const char *source,int *isrc,char *string,const char *endmark))
G__DECL_API(125, char*, G__type2string, (int type,int tagnum,int typenum,int reftype,int isconst))
G__DECL_API(240, void, G__set_typenum,(G__value* val, const char* type))
G__DECL_API(252, void, G__set_type,(G__value* val, char* type))
G__DECL_API(253, void, G__set_tagnum,(G__value* val, int tagnum))
G__DECL_API(126, void, G__alloc_tempobject, (int tagnum,int typenum))
G__DECL_API(251, void, G__alloc_tempobject_val, (G__value* val))
G__DECL_API(127, void, G__set_p2fsetup, (void (*p2f)()))
G__DECL_API(128, void, G__free_p2fsetup, (void))
G__DECL_API(129, int, G__genericerror, (G__CONST char *message))
G__DECL_API(130, char*, G__tmpnam, (char* name))
G__DECL_API(131, int, G__setTMPDIR, (char* badname))
G__DECL_API(132, void, G__setPrerun, (int prerun))
G__DECL_API(133, int, G__readline, (FILE *fp,char *line,char *argbuf,int *argn,char *arg[]))
G__DECL_API(134, int, G__getFuncNow, (void))
G__DECL_API(135, FILE*, G__getIfileFp, (void))
G__DECL_API(136, void, G__incIfileLineNumber, (void))
G__DECL_API(137, struct G__input_file*, G__get_ifile, (void))
G__DECL_API(138, void, G__setReturn, (int rtn))
G__DECL_API(139, int, G__getPrerun, (void))
G__DECL_API(140, short, G__getDispsource, (void))
G__DECL_API(141, FILE*, G__getSerr, (void))
G__DECL_API(142, int, G__getIsMain, (void))
G__DECL_API(143, void, G__setIsMain, (int ismain))
G__DECL_API(144, void, G__setStep, (int step))
G__DECL_API(145, int, G__getStepTrace, (void))
G__DECL_API(146, void, G__setDebug, (int dbg))
G__DECL_API(147, int, G__getDebugTrace, (void))
G__DECL_API(148, void, G__set_asm_noverflow, (int novfl))
G__DECL_API(149, int, G__get_no_exec, (void))
G__DECL_API(150, int, G__get_no_exec_compile, (void))
G__DECL_API(151, void, G__setdebugcond, (void))
G__DECL_API(152, int, G__init_process_cmd, (void))
G__DECL_API(153, int, G__process_cmd, (char *line,char *prompt,int *more,int *err,G__value *rslt))
G__DECL_API(154, int, G__pause, (void))
G__DECL_API(155, char*, G__input, (const char* prompt))
G__DECL_API(156, int, G__split, (char *line,char *string,int *argc,char **argv))
G__DECL_API(157, int, G__getIfileLineNumber, (void))
G__DECL_API(158, void, G__addpragma, (char* comname, void (* /*p2f*/) (char*)) )
G__DECL_API(159, void, G__add_setup_func, (G__CONST char *libname, G__incsetup func))
G__DECL_API(160, void, G__remove_setup_func, (G__CONST char *libname))
G__DECL_API(161, void, G__setgvp, (long gvp))
G__DECL_API(162, void, G__set_stdio_handle, (FILE* s_out,FILE* s_err,FILE* s_in))
G__DECL_API(163, void, G__setautoconsole, (int autoconsole))
G__DECL_API(164, int, G__AllocConsole, (void))
G__DECL_API(165, int, G__FreeConsole, (void))
G__DECL_API(166, int, G__getcintready, (void))
G__DECL_API(167, int, G__security_recover, (FILE* fout))
G__DECL_API(168, void, G__breakkey, (int signame))
G__DECL_API(169, int, G__stepmode, (int stepmode))
G__DECL_API(170, int, G__tracemode, (int tracemode))
G__DECL_API(171, int, G__setbreakpoint, (const char *breakline,const char *breakfile))
G__DECL_API(172, int, G__getstepmode, (void))
G__DECL_API(173, int, G__gettracemode, (void))
G__DECL_API(174, int, G__printlinenum, (void))
G__DECL_API(175, int, G__search_typename2, (G__CONST char *typenamein,int typein,int tagnum,int reftype,int parent_tagnum))
G__DECL_API(176, void, G__set_atpause, (void (*p2f)()))
G__DECL_API(177, void, G__set_aterror, (void (*p2f)()))
G__DECL_API(178, void, G__p2f_void_void, (void* p2f))
G__DECL_API(179, int, G__setglobalcomp, (int globalcomp))
G__DECL_API(180, const char*, G__getmakeinfo, (const char *item))
G__DECL_API(181, const char*, G__getmakeinfo1, (const char *item))
G__DECL_API(182, int, G__get_security_error, (void))
G__DECL_API(183, char*, G__map_cpp_name, (const char *in))
G__DECL_API(184, char*, G__Charref, (G__value *buf))
G__DECL_API(185, short*, G__Shortref, (G__value *buf))
G__DECL_API(186, int*, G__Intref, (G__value *buf))
G__DECL_API(187, long*, G__Longref, (G__value *buf))
G__DECL_API(188, unsigned char*, G__UCharref, (G__value *buf))
#ifdef G__BOOL4BYTE
G__DECL_API(189, int*, G__Boolref, (G__value *buf))
#else /* G__BOOL4BYTE */
G__DECL_API(189, unsigned char*, G__Boolref, (G__value *buf))
#endif /* G__BOOL4BYTE */
G__DECL_API(190, unsigned short*, G__UShortref, (G__value *buf))
G__DECL_API(191, unsigned int*, G__UIntref, (G__value *buf))
G__DECL_API(192, unsigned long*, G__ULongref, (G__value *buf))
G__DECL_API(193, float*, G__Floatref, (G__value *buf))
G__DECL_API(194, double*, G__Doubleref, (G__value *buf))
G__DECL_API(195, int, G__loadsystemfile, (G__CONST char* filename))
G__DECL_API(196, void, G__set_ignoreinclude, (G__IgnoreInclude ignoreinclude))
G__DECL_API(197, G__value, G__exec_tempfile_fp, (FILE *fp))
G__DECL_API(198, G__value, G__exec_tempfile, (G__CONST char *file))
G__DECL_API(199, G__value, G__exec_text, (G__CONST char *unnamedmacro))
G__DECL_API(200, char*, G__exec_text_str, (G__CONST char *unnamedmacro,char* result))
G__DECL_API(201, char*, G__lasterror_filename, (void))
G__DECL_API(202, int, G__lasterror_linenum, (void))
G__DECL_API(203, void, G__va_arg_put, (G__va_arg_buf* pbuf,struct G__param* libp,int n))

#ifndef G__OLDIMPLEMENTATION1546
G__DECL_API(204, const char*, G__load_text, (G__CONST char *namedmacro))
G__DECL_API(205, void, G__set_emergencycallback, (void (*p2f)()))
#endif
#ifndef G__OLDIMPLEMENTATION1485
G__DECL_API(206, void, G__set_errmsgcallback, (void* p))
#endif
G__DECL_API(207, void, G__letLonglong, (G__value* buf,int type,G__int64 value))
G__DECL_API(208, void, G__letULonglong, (G__value* buf,int type,G__uint64 value))
G__DECL_API(209, void, G__letLongdouble, (G__value* buf,int type,long double value))
G__DECL_API(210, G__int64, G__Longlong, (G__value buf)) 
G__DECL_API(211, G__uint64, G__ULonglong, (G__value buf))
G__DECL_API(212, long double, G__Longdouble, (G__value buf))
G__DECL_API(213, G__int64*, G__Longlongref, (G__value *buf))
G__DECL_API(214, G__uint64*, G__ULonglongref, (G__value *buf))
G__DECL_API(215, long double*, G__Longdoubleref, (G__value *buf))

G__DECL_API(216, int, G__clearfilebusy, (int))
G__DECL_API(217, int, G__close_inputfiles, (void))
G__DECL_API(218, int, G__const_resetnoerror, (void))
G__DECL_API(219, int, G__const_setnoerror, (void))
G__DECL_API(220, int, G__const_whatnoerror, (void))
G__DECL_API(260, void, G__enable_wrappers, (int set))
G__DECL_API(261, int, G__wrappers_enabled, ())
G__DECL_API(222, void, G__scratch_globals_upto, (struct G__dictposition *dictpos))
G__DECL_API(223, int, G__scratch_upto, (struct G__dictposition *dictpos))
G__DECL_API(224, void, G__store_dictposition, (struct G__dictposition* dictpos))
#ifdef G__WIN32
G__DECL_API(225, int, G__printf, (const char* fmt,...))
#endif
G__DECL_API(226, void, G__free_tempobject, (void))
G__DECL_API(227, int, G__display_class, (FILE *fout, char *name, int base, int start))
G__DECL_API(228, int, G__display_includepath, (FILE *fout))
G__DECL_API(229, void, G__set_alloclockfunc, (void(*)()))
G__DECL_API(230, void, G__set_allocunlockfunc, (void(*)()))
#ifdef G__TRUEP2F
G__DECL_API(231, int, G__usermemfunc_setup, (char *funcname,int hash,int (*funcp)(),int type,
                         int tagnum,int typenum,int reftype,
                         int para_nu,int ansi,int accessin,int isconst,
                         char *paras, char *comment
                         ,void *truep2f,int isvirtual
                         ,void *userparam))
#else
G__DECL_API(232, int, G__usermemfunc_setup, (char *funcname,int hash,int (*funcp)(),int type,
                         int tagnum,int typenum,int reftype,
                         int para_nu,int ansi,int accessin,int isconst,
                         char *paras, char *comment
                         ,void *userparam))
#endif
G__DECL_API(233, char, *G__fulltagname, (int tagnum,int mask_dollar))
G__DECL_API(234, void, G__loadlonglong, (int* ptag,int* ptype,int which))
G__DECL_API(235, long, G__isanybase, (int basetagnum,int derivedtagnum,long pobject))
G__DECL_API(236, int, G__pop_tempobject, (void))
G__DECL_API(263, int, G__pop_tempobject_nodel, (void))
G__DECL_API(237, const char*, G__stripfilename, (const char* filename))

/***********************************************************************
 * Native long long support
 ***********************************************************************/
G__DECL_API(238, G__int64, G__expr_strtoll, (const char *nptr,char **endptr, register int base))
G__DECL_API(239, G__uint64, G__expr_strtoull, (const char *nptr, char **endptr, register int base))

G__DECL_API(250, int, G__check_drange, (int p,double low,double up,double d,G__value *result7,const char *funcname))
G__DECL_API(241, int, G__check_lrange, (int p,long low,long up,long l,G__value *result7,const char *funcname))
G__DECL_API(242, int, G__check_type, (int p,int t1,int t2,G__value *para,G__value *result7,const char *funcname))
G__DECL_API(243, int, G__check_nonull, (int p,int t,G__value *para,G__value *result7,const char *funcname))
G__DECL_API(244, void, G__printerror, (const char *funcname,int ipara,int paran))
#ifdef G__SECURITY
G__DECL_API(245, int, G__security_handle,(G__UINT32 category))
#endif
G__DECL_API(246, void, G__CurrentCall,(int, void*, long*))
G__DECL_API(247, G__value, G__getfunction, (G__CONST char *item,int *known3,int memfunc_flag))

G__DECL_API(248, int, G__sizeof, (G__value *object))
G__DECL_API(249, void, G__exec_alloc_lock, ())
G__DECL_API(254, void, G__exec_alloc_unlock, ())
#ifdef _WIN32
G__DECL_API(255, FILE*, FOpenAndSleep, (const char *filename, const char *mode))
#endif
G__DECL_API(256, void, G__letbool, (G__value* buf,int type,long value))
G__DECL_API(264, int, G__Lsizeof, (const char* type_name_in))
G__DECL_API(265, int, G__GetCatchException, ())
#ifdef G__TRUEP2F
G__DECL_API(266, int, G__memfunc_setup2, (G__CONST char *funcname,int hash,G__CONST char *mangled_name, \
                                          G__InterfaceMethod funcp,int type,int tagnum,int typenum, \
                                          int reftype,int para_nu,int ansi,int access,int isconst, \
                                          G__CONST char *paras,G__CONST char *comment,void* tp2f,int isvirtual))
#else /* G__TRUEP2F */
G__DECL_API(266, int, G__memfunc_setup2, (G__CONST char *funcname,int hash,G__CONST char *mangled_name, \
                                          G__InterfaceMethod funcp,int type,int tagnum,int typenum, \
                                          int reftype,int para_nu,int ansi,int access,int isconst, \
                                          G__CONST char *paras,G__CONST char *comment))
#endif /* G__TRUEP2F */
G__DECL_API(267, int, G__memfunc_para_setup, (int ifn,int type,int tagnum,int typenum,int reftype, \
                                              G__value *para_default,char *para_def,char *para_name))

G__DECL_API(268, struct G__ifunc_table_internal*, G__get_ifunc_internal, (struct G__ifunc_table* iref))

#define G__NUMBER_OF_API_FUNCTIONS 270
G__DUMMYTOCHECKFORDUPLICATES(G__NUMBER_OF_API_FUNCTIONS)

#endif /* G__CI_FPROTO_INCLUDE */
