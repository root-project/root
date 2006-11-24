/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file fproto.h
 ************************************************************************
 * Description:
 *  K&R style function prototype
 ************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__FPROTO_H
#define G__FPROTO_H


#ifndef __CINT__

#ifdef __cplusplus
extern "C" {
#endif
/* G__cfunc.c */
int G__compiled_func(G__value *result7,char *funcname,struct G__param *libp,int hash);
void G__list_sut(FILE *fp);

/* G__setup.c */
extern int G__dll_revision(void);
extern int G__globalsetup(void);
extern int G__revision(FILE *fp);
extern int G__list_global(FILE *fp);
extern int G__list_struct(FILE *fp);

/* G__setup.c */
extern int G__list_stub(FILE *fp);

/* in src/xxx.c */
/* struct G__var_array *G__rawvarentry(char *name,int hash,int *ig15,struct G__var_array *memvar); */
/* int G__split(char *line,char *string,int *argc,char **argv); */
int G__readsimpleline(FILE *fp,char *line);
int G__cmparray(short array1[],short array2[],int num,short mask);
void G__setarray(short array[],int num,short mask,char *mode);
int G__graph(double *xdata,double *ydata,int ndata,char *title,int mode);
int G__storeobject(G__value *buf1,G__value *buf2);
int G__scanobject(G__value *buf1);
int G__dumpobject(char *file,void *buf,int size);
int G__loadobject(char *file,void *buf,int size);
long G__what_type(char *name,char *type,char *tagname,char *typenamein);
int G__textprocessing(FILE *fp);
#ifdef G__REGEXP
int G__matchregex(char *pattern,char *string);
#endif
#ifdef G__REGEXP1
int G__matchregex(char *pattern,char *string);
#endif
void G__castclass(G__value *result3,int tagnum,int castflag,int *ptype,int reftype);
G__value G__castvalue(char *casttype,G__value result3);
void G__asm_cast(int type,G__value *buf,int tagnum,int reftype);
  /* void G__setdebugcond(void); */
int G__findposition(char *string,struct G__input_file view,int *pline,int *pfnum);
int G__findfuncposition(char *func,int *pline,int *pfnum);
int G__beforelargestep(char *statement,int *piout,int *plargestep);
void G__afterlargestep(int *plargestep);
void G__EOFfgetc(void);
void G__BREAKfgetc(void);
void G__DISPNfgetc(void);
void G__DISPfgetc(int c);
void G__lockedvariable(char *item);
int G__lock_variable(char *varname);
int G__unlock_variable(char *varname);
G__value G__interactivereturn(void);
void G__set_tracemode(char *name);
void G__del_tracemode(char *name);
void G__set_classbreak(char *name);
void G__del_classbreak(char *name);
void G__setclassdebugcond(int tagnum,int brkflag);
int G__get_newname(char *new_name);
int G__unsignedintegral(int *pspaceflag,int *piout,int mparen);
void G__define_var(int tagnum,int typenum);
int G__initary(char *new_name);
struct G__var_array* G__initmemvar(int tagnum,int* pindex,G__value *pbuf);
struct G__var_array* G__incmemvar(struct G__var_array* memvar,int* pindex,G__value *pbuf);
int G__initstruct(char *new_name);
int G__ignoreinit(char *new_name);
int G__listfunc(FILE *fp,int access,char* fname,struct G__ifunc_table *ifunc);
int G__listfunc_pretty(FILE *fp,int access,char* fname,struct G__ifunc_table *ifunc,char friendlyStyle);
int G__showstack(FILE *fout);
void G__display_note(void);
int G__display_proto(FILE *fout,char *string);
int G__display_proto_pretty(FILE *fout,char *string,char friendlyStyle);
int G__display_newtypes(FILE *fout,char *string);
int G__display_typedef(FILE *fout,char *name,int startin);
int G__display_template(FILE *fout,char *name);
int G__display_macro(FILE *fout,char *name);
int G__display_string(FILE *fout);
int G__display_files(FILE *fout);
int G__pr(FILE *fout,struct G__input_file view);
int G__dump_tracecoverage(FILE *fout);
int G__objectmonitor(FILE *fout,long pobject,int tagnum,char *addspace);
int G__varmonitor(FILE *fout,struct G__var_array *var,char *index,char *addspace,long offset);
int G__pushdumpinput(FILE *fp,int exflag);
int G__popdumpinput(void);
int G__dumpinput(char *line);
char *G__xdumpinput(char *prompt);
  /* void G__scratch_all(void); */
int G__free_ifunc_table(struct G__ifunc_table *ifunc);
int G__free_member_table(struct G__var_array *mem);
int G__free_ipath(struct G__includepath *ipath);
int G__isfilebusy(int ifn);
void G__free_preprocessfilekey(struct G__Preprocessfilekey *pkey);
int G__free_ifunc_table_upto(struct G__ifunc_table *ifunc,struct G__ifunc_table *dictpos,int ifn);
int G__free_string_upto(struct G__ConstStringList *conststringpos);
int G__free_typedef_upto(int typenum);
int G__free_struct_upto(int tagnum);
int G__destroy_upto(struct G__var_array *var,int global,struct G__var_array *dictpos,int ig15);
void G__close_inputfiles_upto(int nfile);
void G__destroy(struct G__var_array *var,int global);
int G__call_atexit(void);
int G__close_inputfiles(void);
int G__interpretexit(void);
void G__nosupport(char *name);
void G__malloc_error(char *varname);
void G__arrayindexerror(int ig15,struct G__var_array *var,char *item,int p_inc);
int G__asm_execerr(char *message,int num);
int G__assign_error(char *item,G__value *pbuf);
int G__reference_error(char *item);
int G__warnundefined(char *item);
int G__unexpectedEOF(char *message);
int G__shl_load(char *shlfile);
int G__shl_load_error(char *shlname,char *message);
int G__getvariable_error(char *item);
int G__referencetypeerror(char *new_name);
int G__err_pointer2pointer(char *item);
int G__syntaxerror(char *expr);
void G__setDLLflag(int flag);
void G__setInitFunc(char *initfunc);
int G__assignmenterror(char *item);
int G__parenthesiserror(char *expression,char *funcname);
int G__commenterror(void);
int G__changeconsterror(char *item,char *categ);
  /* int G__printlinenum(void); */
int G__autocc(void);
int G__init_readline(void);
int G__using_namespace(void);

#ifdef G__FRIEND
int G__parse_friend(int *piout,int *pspaceflag,int mparen);
#else
int G__friendignore(int *piout,int *pspaceflag,int mparen);
#endif
int G__externignore(int *piout,int *pspaceflag,int mparen);
int G__handleEOF(char *statement,int mparen,int single_quote,int double_quote);
void G__printerror(char *funcname,int ipara,int paran);
int G__pounderror(void);
int G__missingsemicolumn(char *item);
G__value G__calc_internal(char *exprwithspace);
G__value G__getexpr(char *expression);
G__value G__getprod(char *expression1);
G__value G__getpower(char *expression2);
G__value G__getitem(char *item);
int G__getoperator(int newoperator,int oldoperator);
int G__testandor(int lresult,char *rexpression,int operator2);
int G__test(char *expression2);
int G__btest(int operator2,G__value lresult,G__value rresult);
int G__fgetspace(void);
int G__fgetvarname(char *string,char *endmark);
int G__fgetname(char *string,char *endmark);
int G__getname(char* source,int* isrc,char *string,char *endmark);
int G__fdumpstream(char *string,char *endmark);
int G__fgetstream(char *string,char *endmark);
int G__fignorestream(char *endmark);
int G__ignorestream(char *string,int* isrc,char *endmark);
int G__fgetstream_new(char *string,char *endmark);
void G__fignoreline(void);
void G__fsetcomment(struct G__comment_info *pcomment);
int G__fgetc(void);
long G__op1_operator_detail(int opr,G__value *val);
long G__op2_operator_detail(int opr,G__value *lval,G__value *rval);
int G__explicit_fundamental_typeconv(char* funcname,int hash,struct G__param *libp,G__value *presult3);
int G__special_func(G__value *result7,char *funcname,struct G__param *libp,int hash);
int G__library_func(G__value *result7,char *funcname,struct G__param *libp,int hash);
char *G__charformatter(int ifmt,struct G__param *libp,char *result);
int G__istypename(char *temp);
char* G__savestring(char** pbuf,char* name);
void G__make_ifunctable(char *funcheader);
int G__readansiproto(struct G__ifunc_table *ifunc,int func_now);
int G__interpret_func(G__value *result7,char *funcname,struct G__param *libp,int hash,struct G__ifunc_table *p_ifunc,int funcmatch,int memfunc_flag);
struct G__ifunc_table *G__ifunc_exist(struct G__ifunc_table *ifunc_now,int allifunc,struct G__ifunc_table *ifunc,int *piexist,int mask);
struct G__ifunc_table *G__ifunc_ambiguous(struct G__ifunc_table *ifunc_now,int allifunc,struct G__ifunc_table *ifunc,int *piexist,int derivedtagnum);
void G__inheritclass(int to_tagnum,int from_tagnum,char baseaccess);
int G__baseconstructorwp(void);
int G__baseconstructor(int n,struct G__baseparam *pbaseparam);
int G__basedestructor(void);
int G__basedestructrc(struct G__var_array *mem);
#ifdef G__VIRTUALBASE
int G__publicinheritance(G__value *val1,G__value *val2);
int G__ispublicbase(int basetagnum,int derivedtagnum,long pobject);
long G__getvirtualbaseoffset(long pobject,int tagnum,struct G__inheritance *baseclass,int basen);
#else
int G__ispublicbase(int basetagnum,int derivedtagnum);
int G__isanybase(int basetagnum,int derivedtagnum);
#endif
int G__find_virtualoffset(int virtualtag);
  /* int G__main(int argc,char **argv); */
int G__init_globals(void);
void G__set_stdio(void);
int G__cintrevision(FILE *fp);
  /* char *G__input(char *prompt); */
char *G__strrstr(char *string1,char *string2);
void G__set_sym_underscore(int x);

  /* void G__breakkey(int signame); */
void G__floatexception(int signame);
void G__segmentviolation(int signame);
void G__outofmemory(int signame);
void G__buserror(int signame);
void G__errorexit(int signame);

void G__killproc(int signame);
void G__timeout(int signame);

void G__fsigabrt(int);
void G__fsigfpe(int);
void G__fsigill(int);
void G__fsigint(int);
void G__fsigsegv(int);
void G__fsigterm(int);
#ifdef SIGHUP
void G__fsighup(int);
#endif
#ifdef SIGQUIT
void G__fsigquit(int);
#endif
#ifdef SIGTSTP
void G__fsigtstp(int);
#endif
#ifdef SIGTTIN
void G__fsigttin(int);
#endif
#ifdef SIGTTOU
void G__fsigttou(int);
#endif
#ifdef SIGALRM
void G__fsigalrm(int);
#endif
#ifdef SIGUSR1
void G__fsigusr1(int);
#endif
#ifdef SIGUSR2
void G__fsigusr2(int);
#endif

int G__errorprompt(char *nameoferror);
int G__call_interruptfunc(char *func);

int G__include_file(void);
int G__getcintsysdir(void);
int G__preprocessor(char *outname,char *inname,int cppflag,char *macros,char *undeflist,char *ppopt,char *includepath);
int G__difffile(char *file1,char *file2);
int G__copyfile(FILE *to,FILE *from);

int G__matchfilename(int i1,char* filename);
int G__cleardictfile(int flag);

void G__openmfp(void);
int G__closemfp(void);
void G__define(void);
int G__handle_as_typedef(char *oldtype,char *newtype);
void G__createmacro(char *new_name,char *initvalue);
G__value G__execfuncmacro(char *item,int *done);
int G__transfuncmacro(char *item,struct G__Deffuncmacro *deffuncmacro,struct G__Callfuncmacro *callfuncmacro,fpos_t call_pos,char *p,int nobraces,int nosemic);
int G__replacefuncmacro(char *item,struct G__Callfuncmacro *callfuncmacro,struct G__Charlist *callpara,struct G__Charlist *defpara,FILE *def_fp,fpos_t def_pos,int nobraces,int nosemic);
int G__execfuncmacro_noexec(char* macroname);
int G__maybe_finish_macro(void);
int G__argsubstitute(char *symbol,struct G__Charlist *callpara,struct G__Charlist *defpara);
int G__createfuncmacro(char *new_name);
int G__getparameterlist(char *paralist,struct G__Charlist *charlist);
int G__freedeffuncmacro(struct G__Deffuncmacro *deffuncmacro);
int G__freecallfuncmacro(struct G__Callfuncmacro *callfuncmacro);
int G__freecharlist(struct G__Charlist *charlist);
long G__malloc(int n,int bsize,char *item);
void *G__TEST_Malloc(size_t size);
void *G__TEST_Calloc(size_t n,size_t bsize);
void G__TEST_Free(void *p);
void *G__TEST_Realloc(void *p,size_t size);
int G__memanalysis(void);
int G__memresult(void);
void G__DUMMY_Free(void *p);
void *G__TEST_fopen(char *fname,char *mode);
int G__TEST_fclose(FILE *p);
G__value G__new_operator(char *expression);
int G__getarrayindex(char *indexlist);
void G__delete_operator(char *expression,int isarray);
int G__alloc_newarraylist(long point,int pinc);
int G__free_newarraylist(long point);
int G__handle_delete(int *piout,char *statement);
int G__call_cppfunc(G__value *result7,struct G__param *libp,struct G__ifunc_table *ifunc,int ifn);
void G__gen_cppheader(char *headerfile);
void G__gen_clink(void);
void G__gen_cpplink(void);
void G__clink_header(FILE *fp);
void G__cpplink_header(FILE *fp);
void G__cpplink_linked_taginfo(FILE* fp,FILE* hfp);
char *G__get_link_tagname(int tagnum);
/* char *G__map_cpp_name(char *in); */
char *G__map_cpp_funcname(int tagnum,char *funcname,int ifn,int page);
void G__set_globalcomp(char *mode,char *linkfilename,char* dllid);
int G__ishidingfunc(struct G__ifunc_table *fentry,struct G__ifunc_table *fthis,int ifn);
void G__cppif_memfunc(FILE *fp,FILE *hfp);
void G__cppif_func(FILE *fp,FILE *hfp);
void G__cppif_dummyfuncname(FILE *fp);
void G__cppif_genconstructor(FILE *fp,FILE *hfp,int tagnum,int ifn,struct G__ifunc_table *ifunc);
int G__isprivateconstructor(int tagnum,int iscopy);
void G__cppif_gendefault(FILE *fp,FILE *hfp,int tagnum,int ifn,struct G__ifunc_table *ifunc,int isconstructor,int iscopyconstructor,int isdestructor,int isassignmentoperator,int isnonpublicnew);
void G__cppif_genfunc(FILE *fp,FILE *hfp,int tagnum,int ifn,struct G__ifunc_table *ifunc);
int G__cppif_returntype(FILE *fp,int ifn,struct G__ifunc_table *ifunc,char *endoffunc);
void G__cppif_paratype(FILE *fp,int ifn,struct G__ifunc_table *ifunc,int k);
void G__cpplink_tagtable(FILE *pfp,FILE *hfp);
#ifdef G__VIRTUALBASE
int G__iosrdstate(G__value *pios);
void G__cppif_inheritance(FILE *pfp);
#endif
void G__cpplink_inheritance(FILE *pfp);
void G__cpplink_typetable(FILE *pfp,FILE *hfp);
void G__cpplink_memvar(FILE *pfp);
void G__cpplink_memfunc(FILE *pfp);
void G__cpplink_global(FILE *pfp);
void G__cpplink_func(FILE *pfp);
void G__incsetup_memvar(int tagnum);
void G__incsetup_memfunc(int tagnum);


int G__separate_parameter(char *original,int *pos,char *param);
int G__parse_parameter_link(char *paras);
int G__cppif_p2memfunc(FILE *fp);
int G__set_sizep2memfunc(FILE *fp);
int G__getcommentstring(char *buf,int tagnum,struct G__comment_info *pcomment);
void G__bstore(int operatorin,G__value expressionin,G__value *defined);
void G__doubleassignbyref(G__value *defined,double val);
void G__intassignbyref(G__value *defined,G__int64 val);
int G__scopeoperator(char *name,int *phash,long *pstruct_offset,int *ptagnum);
int G__label_access_scope(char *statement,int *piout,int *pspaceflag,int *pmparen);
int G__cmp(G__value buf1,G__value buf2);
int G__getunaryop(char unaryop,char *expression,char *buf,G__value *preg);
int G__overloadopr(int operatorin,G__value expressionin,G__value *defined);
int G__parenthesisovldobj(G__value *result3,G__value *result,char *realname,struct G__param *libp,int flag);
int G__parenthesisovld(G__value *result3,char *funcname,struct G__param *libp,int flag);
int G__tryindexopr(G__value *result7,G__value *para,int paran,int ig25);
int G__exec_delete(char *statement,int *piout,int *pspaceflag,int isarray,int mparen);
int G__exec_function(char *statement,int *pc,int *piout,int *plargestep,G__value *presult);
int G__keyword_anytime_5(char *statement);
int G__keyword_anytime_6(char *statement);
int G__keyword_anytime_7(char *statement);
int G__keyword_exec_6(char *statement,int *piout,int *pspaceflag,int mparen);
int G__setline(char *statement,int c,int *piout);
int G__skip_comment(void);
int G__pp_command(void);
void G__pp_skip(int elifskip);
int G__pp_if(void);
int G__defined_macro(char *macro);
int G__pp_ifdef(int def);
void G__pp_undef(void);
G__value G__exec_do(void);
G__value G__return_value(char *statement);
void G__free_tempobject(void);
G__value G__alloc_tempstring(char *string);
G__value G__exec_switch(void);
G__value G__exec_if(void);
G__value G__exec_loop(char *forinit,char *condition,int naction,char **foraction);
G__value G__exec_while(void);
G__value G__exec_for(void);
G__value G__exec_else_if(void);
G__value G__exec_statement(void);
int G__readpointer2function(char *new_name,char *pvar_type);
int G__search_gotolabel(char *label,fpos_t *pfpos,int line,int *pmparen);
int G__update_stdio(void);
  /* int G__pause(void); */
int G__setaccess(char *statement,int iout);
int G__class_conversion_operator(int tagnum,G__value *presult,char* ttt);
int G__fundamental_conversion_operator(int type,int tagnum,int typenum,int reftype,int constvar,G__value *presult,char* ttt);
int G__asm_gen_stvar(long G__struct_offset,int ig15,int paran,struct G__var_array *var,char *item,long store_struct_offset,int var_type,G__value *presult);
int G__exec_asm(int start,int stack,G__value *presult,long localmem);
int G__asm_test_E(int *a,int *b);
int G__asm_test_N(int *a,int *b);
int G__asm_test_GE(int *a,int *b);
int G__asm_test_LE(int *a,int *b);
int G__asm_test_g(int *a,int *b);
int G__asm_test_l(int *a,int *b);
long G__asm_gettest(int op,long *inst);
int G__asm_optimize(int *start);
int G__asm_optimize3(int *start);
int G__inc_cp_asm(int cp_inc,int dt_dec);
int G__clear_asm(void);
int G__asm_clear(void);
void G__gen_addstros(int addstros);
void G__suspendbytecode(void);
void G__resetbytecode(void);
void G__resumebytecode(int store_asm_noverflow);
void G__abortbytecode(void);
int G__asm_putint(int i);
G__value G__getreserved(char *item,void** ptr,void** ppdict);
/*
G__value G__getreserved(char *item,void *ptr);
G__value G__getreserved(char *item);
*/
int G__dasm(FILE *fout,int isthrow);
G__value G__getrsvd(int i);
int G__read_setmode(int *pmode);
int G__pragma(void);
int G__execpragma(char *comname,char *args);
#ifndef G__VMS
void G__freepragma(struct G__AppPragma *paddpragma);
#endif
void G__free_bytecode(struct G__bytecodefunc *bytecode);
void G__asm_gen_strip_quotation(G__value *pval);
int G__security_handle(G__UINT32 category);
void G__asm_get_strip_quotation(G__value *pval);
G__value G__strip_quotation(char *string);
char *G__charaddquote(char *string,char c);
G__value G__strip_singlequotation(char *string);
char *G__add_quotation(char *string,char *temp);
char *G__tocharexpr(char *result7);
char *G__string(G__value buf,char *temp);
char *G__quotedstring(char *buf,char *result);
char *G__logicstring(G__value buf,int dig,char *result);
int G__revprint(FILE *fp);
int G__dump_header(char *outfile);
void G__listshlfunc(FILE *fout);
void G__listshl(FILE *G__temp);
int G__free_shl_upto(int allsl);
G__value G__pointer2func(G__value* obj_p2f,char *parameter0,char *parameter1,int *known3);
char *G__search_func(char *funcname,G__value *buf);
char *G__search_next_member(char *text,int state);
int G__Loffsetof(char *tagname,char *memname);
int G__Lsizeof(char *typenamein);
long *G__typeid(char *typenamein);
void G__getcomment(char *buf,struct G__comment_info *pcomment,int tagnum);
void G__getcommenttypedef(char *buf,struct G__comment_info *pcomment,int typenum);
long G__get_classinfo(char *item,int tagnum);
long G__get_variableinfo(char *item,long *phandle,long *pindex,int tagnum);
long G__get_functioninfo(char *item,long *phandle,long *pindex,int tagnum);
int G__get_envtagnum(void);
int G__isenclosingclass(int enclosingtagnum,int env_tagnum);
int G__isenclosingclassbase(int enclosingtagnum,int env_tagnum);
char* G__find_first_scope_operator(char* name);
char* G__find_last_scope_operator(char* name);
#ifndef G__OLDIMPLEMENTATION1560
int G__checkset_charlist(char *tname,struct G__Charlist *pcall_para,int narg,int ftype);
#endif
void G__define_struct(char type);
G__value G__classassign(long pdest,int tagnum,G__value result);
int G__cattemplatearg(char *tagname,struct G__Charlist *charlist);
char *G__catparam(struct G__param *libp,int catn,char *connect);
int G__fgetname_template(char *string,char *endmark);
int G__fgetstream_newtemplate(char *string,char *endmark);
int G__fgetstream_template(char *string,char *endmark);
int G__fgetstream_spaces(char *string,char *endmark);
int G__getstream_template(char *source,int *isrc,char *string,char *endmark);
void G__IntList_init(struct G__IntList *body,long iin,struct G__IntList *prev);
struct G__IntList* G__IntList_new(long iin,struct G__IntList *prev);
void G__IntList_add(struct G__IntList *body,long iin);
void G__IntList_addunique(struct G__IntList *body,long iin);
void G__IntList_delete(struct G__IntList *body);
struct G__IntList* G__IntList_find(struct G__IntList *body,long iin);
void G__IntList_free(struct G__IntList *body);
struct G__Templatearg *G__read_formal_templatearg(void);
int G__createtemplatememfunc(char *new_name);
int G__createtemplateclass(char *new_name,struct G__Templatearg *targ,int isforwarddecl);
#ifndef G__OLDIMPLEMENTATION1560
struct G__Definetemplatefunc *G__defined_templatefunc(char *name);
#endif
struct G__Definetemplatefunc *G__defined_templatememfunc(char *name);
void G__declare_template(void);
int G__gettemplatearglist(char *paralist,struct G__Charlist *charlist,struct G__Templatearg *def_para,int *pnpara,int parent_tagnum);
int G__instantiate_templateclass(char *tagname,int noerror);
void G__replacetemplate(char *templatename,char *tagname,struct G__Charlist *callpara,FILE *def_fp,int line,int filenum,fpos_t *pdef_pos,struct G__Templatearg *def_para,int isclasstemplate,int npara,int parent_tagnum);
int G__templatesubstitute(char *symbol,struct G__Charlist *callpara,struct G__Templatearg *defpara,char *templatename,char *tagname,int c,int npara,int isnew);
void G__freedeftemplateclass(struct G__Definedtemplateclass *deftmpclass);
void G__freetemplatememfunc(struct G__Definedtemplatememfunc *memfunctmplt);
char *G__gettemplatearg(int n,struct G__Templatearg *def_para);
void G__freetemplatearg(struct G__Templatearg *def_para);
void G__freetemplatefunc(struct G__Definetemplatefunc *deftmpfunc);
struct G__funclist* G__add_templatefunc(char *funcnamein,struct G__param *libp,int hash,struct G__funclist *funclist,struct G__ifunc_table *p_ifunc,int isrecursive);
struct G__funclist* G__funclist_add(struct G__funclist *last,struct G__ifunc_table *ifunc,int ifn,int rate);
void G__funclist_delete(struct G__funclist *body);
int G__templatefunc(G__value *result,char *funcname,struct G__param *libp,int hash,int funcmatch);
int G__matchtemplatefunc(struct G__Definetemplatefunc *deftmpfunc,struct G__param *libp,struct G__Charlist *pcall_para,int funcmatch);
int G__createtemplatefunc(char *funcname,struct G__Templatearg *targ,int line_number,fpos_t *ppos);
void G__define_type(void);
int G__defined_type(char *typenamein,int len);
char *G__valuemonitor(G__value buf,char *temp);
char *G__access2string(int caccess);
char *G__tagtype2string(int tagtype);
#ifndef G__OLDIMPLEMENTATION1560
char* G__rename_templatefunc(char* funcname,int isrealloc);
char *G__fulltypename(int typenum);
#endif
int G__val2pointer(G__value *result7);
char *G__getbase(unsigned int expression,int base,int digit,char *result1);
int G__getdigit(unsigned int number);
G__value G__checkBase(char *string,int *known4);
int G__isfloat(char *string,int *type);
int G__isoperator(int c);
int G__isexponent(char *expression4,int lenexpr);
int G__isvalue(char *temp);
int G__isdouble(G__value buf);
/* float G__float(G__value buf); */
G__value G__tovalue(G__value p);
G__value G__toXvalue(G__value result,int var_type);
G__value G__letVvalue(G__value *p,G__value result);
G__value G__letPvalue(G__value *p,G__value result);
G__value G__letvalue(G__value *p,G__value result);
G__value G__letvariable(char *item,G__value expression,struct G__var_array *varglobal,struct G__var_array *varlocal);
G__value G__getvariable(char *item,int *known2,struct G__var_array *varglobal,struct G__var_array *varlocal);
G__value G__getstructmem(int store_var_type,char *varname,char *membername,char *tagname,int *known2,struct G__var_array *varglobal,int objptr);
G__value G__letstructmem(int store_var_type,char *varname,char *membername,char *tagname,struct G__var_array *varglobal,G__value expression,int objptr);
void G__letstruct(G__value *result,int p_inc,struct G__var_array *var,int ig15,char *item,int paran,long G__struct_offset);
void G__letstructp(G__value result,long G__struct_offset,int ig15,int p_inc,struct G__var_array *var,int paran,char *item,G__value *para,int pp_inc);
void G__returnvartype(G__value* presult,struct G__var_array *var,int ig15,int paran);
G__value G__allocvariable(G__value result,G__value para[],struct G__var_array *varglobal,struct G__var_array *varlocal,int paran,int varhash,char *item,char *varname,int parameter00);
struct G__var_array *G__getvarentry(char *varname,int varhash,int *pi,struct G__var_array *varglobal,struct G__var_array *varlocal);
int G__getthis(G__value *result7,char *varname,char *item);
void G__letpointer2memfunc(struct G__var_array *var,int paran,int ig15,char *item,int p_inc,G__value *presult,long G__struct_offset);
void G__letautomatic(struct G__var_array *var,int ig15,long G__struct_offset,int p_inc,G__value result);
void G__display_classkeyword(FILE *fout,char *classnamein,char *keyword,int base);
#ifdef G__FRIEND
int G__isfriend(int tagnum);
#endif
void G__set_c_environment(void);
void G__specify_link(int link_stub);

long G__new_ClassInfo(char *classname);
long G__get_ClassInfo(int item,long tagnum,char *buf);
long G__get_BaseClassInfo(int item,long tagnum,long basep,char *buf);
long G__get_DataMemberInfo(int item,long tagnum,long *handle,long *index,char *buf);
long G__get_MethodInfo(int item,long tagnum,long *handle,long *index,char *buf);
long G__get_MethodArgInfo(int item,long tagnum,long handle,long index,long *argn,char *buf);

#ifdef G__SECURITY
int G__check_drange(int p,double low,double up,double d,G__value *result7,char *funcname);
int G__check_lrange(int p,long low,long up,long l,G__value *result7,char *funcname);
int G__check_type(int p,int t1,int t2,G__value *para,G__value *result7,char *funcname);
int G__check_nonull(int p,int t,G__value *para,G__value *result7,char *funcname);
G__UINT32 G__getsecuritycode(char *string);
#endif
void G__cpp_setupG__stream(void);
void G__cpp_setupG__API(void);
void G__c_setupG__stdstrct(void);
int G__setautoccnames(void);
int G__appendautocc(FILE *fp);
int G__isautoccupdate(void);
void G__free_friendtag(struct G__friendtag *friendtag);

int G__free_exceptionbuffer(void);
int G__exec_try(char* statement);
int G__exec_throw(char* statement);
int G__ignore_catch(void);
int G__exec_catch(char* statement);


void G__cppstub_memfunc(FILE* fp);
void G__cppstub_func(FILE* fp);
void G__set_stubflags(struct G__dictposition *dictpos);

void G__set_DLLflag(void);
void G__setPROJNAME(char* proj);

#ifdef G__ERROR_HANDLE
void G__error_handle(int signame);
#endif

#ifdef G__MULTIBYTE
int G__CodingSystem(int c);
#endif

extern void G__more_col(int len);
extern int G__more(FILE* fp,char *msg);
extern int G__more_pause(FILE* fp,int len);
extern void G__redirect_on(void);
extern void G__redirect_off(void);

void G__init_jumptable_bytecode(void);
void G__add_label_bytecode(char *label);
void G__add_jump_bytecode(char *label);
void G__resolve_jumptable_bytecode(void);

extern void G__LockCriticalSection(void);
extern void G__UnlockCriticalSection(void);

extern void G__asm_tovalue_p2p(G__value *result);
extern void G__asm_tovalue_p2p2p(G__value *result);
extern void G__asm_tovalue_p2p2p2(G__value *result);
extern void G__asm_tovalue_LL(G__value *result);
extern void G__asm_tovalue_ULL(G__value *result);
extern void G__asm_tovalue_LD(G__value *result);
extern void G__asm_tovalue_B(G__value *result);
extern void G__asm_tovalue_C(G__value *result);
extern void G__asm_tovalue_R(G__value *result);
extern void G__asm_tovalue_S(G__value *result);
extern void G__asm_tovalue_H(G__value *result);
extern void G__asm_tovalue_I(G__value *result);
extern void G__asm_tovalue_K(G__value *result);
extern void G__asm_tovalue_L(G__value *result);
extern void G__asm_tovalue_F(G__value *result);
extern void G__asm_tovalue_D(G__value *result);
extern void G__asm_tovalue_U(G__value *result);

extern int G__gen_linksystem(char* headerfile);

extern void G__smart_shl_unload(int allsl);
extern void G__smart_unload(int ifn);

extern struct G__dictposition* G__get_dictpos(char* fname);

void G__specify_extra_include(void);
void G__gen_extra_include(void);

struct G__ConstStringList* G__AddConstStringList(struct G__ConstStringList* current,char* str,int islen);
void G__DeleteConstStringList(struct G__ConstStringList* current);

int G__ReadInputMode(void);

#ifndef G__OLDIMPLEMENTATION1825
char* G__setiparseobject(G__value* result,char *str);
#endif

#ifdef G__SHMGLOBAL
void* G__shmmalloc(int size);
void* G__shmcalloc(int atomsize,int num);
void G__initshm(void);
#endif

#ifdef G__BORLAND
int G__gettempfilenum(void);
void G__redirect_on(void);
void G__redirect_off(void);
#endif

int G__DLL_direct_globalfunc(G__value *result7
				   ,G__CONST char *funcname
				   ,struct G__param *libp,int hash);
void* G__SetShlHandle(char* filename);
void G__ResetShlHandle();
void* G__FindSymbol(struct G__ifunc_table *ifunc,int ifn);
void* G__GetShlHandle();
int G__GetShlFilenum();

int G__loadfile_tmpfile(FILE *fp);

#ifndef G__OLDIMPLEMENTATION2030
int G__callfunc0(G__value *result,struct G__ifunc_table *ifunc,int ifn,struct G__param* libp,void* p,int funcmatch);
int G__calldtor(void* p,int tagnum,int isheap);
#endif

void G__init_replacesymbol();
void G__add_replacesymbol(const char* s1,const char* s2);
const char* G__replacesymbol(const char* s);
int G__display_replacesymbol(FILE *fout,const char* name);

void G__asm_storebytecodefunc(struct G__ifunc_table *ifunc,int ifn,struct G__var_array *var,G__value *pstack,int sp,long *pinst,int instsize);

void G__push_autoobjectstack(void *p,int tagnum,int num
			           ,int scopelevel,int isheap);
void G__delete_autoobjectstack(int scopelevel);

int G__LD_IFUNC_optimize(struct G__ifunc_table* ifunc,int ifn ,long *inst,int pc);

int G__bc_compile_function(struct G__ifunc_table *ifunc,int iexist);
int G__bc_throw_compile_error();
int G__bc_throw_runtime_error();
int G__bc_objassignment(G__value *plresult ,G__value *prresult);

int G__bc_exec_virtual_bytecode(G__value *result7
			,char *funcname        // vtagnum
			,struct G__param *libp
			,int hash              // vtblindex
			);
int G__bc_exec_normal_bytecode(G__value *result7
			,char *funcname        // ifunc
			,struct G__param *libp
			,int hash              // ifn
			);
int G__bc_exec_ctor_bytecode(G__value *result7
			,char *funcname        // ifunc
			,struct G__param *libp
			,int hash              // ifn
			);
int G__bc_exec_ctorary_bytecode(G__value *result7
			,char *funcname        // ifunc
			,struct G__param *libp
			,int hash              // ifn, n
			);
int G__bc_exec_dtorary_bytecode(G__value *result7
			,char *funcname        // ifunc
			,struct G__param *libp
			,int hash              // ifn, n
			);
void G__bc_struct(int tagnum);

void G__bc_delete_vtbl(int tagnum);
void G__bc_disp_vtbl(FILE* fp,int tagnum);

G__value G__bc_new_operator(const char *expression);
void G__bc_delete_operator(const char *expression,int isarray);

int G__bc_exec_try_bytecode(int start,int stack,G__value *presult,long localmem);
int G__bc_exec_throw_bytecode(G__value* pval);
int G__bc_exec_typematch_bytecode(G__value* catchtype,G__value* excptobj);
int G__Isvalidassignment_val(G__value* ltype,int varparan,int lparan,int lvar_type,G__value* rtype);
int G__bc_conversion(G__value *result,struct G__var_array* var,int ig15
			   ,int var_type,int paran);

int G__bc_assignment(struct G__var_array *var,int ig15,int paran
			   ,int var_type,G__value *prresult
			   ,long struct_offset,long store_struct_offset
			   ,G__value *ppara);

int G__bc_setdebugview(int i,struct G__view *pview);
int G__bc_showstack(FILE* fp);
void G__bc_setlinenum(int line);

void G__bc_Baseclassctor_vbase(int tagnum);

void G__bc_VIRTUALADDSTROS(int tagnum,struct G__inheritance* baseclas,int basen);
void G__bc_cancel_VIRTUALADDSTROS();

void G__bc_REWINDSTACK(int n);

int G__bc_casejump(void* p,int val);

G__value G__alloc_exceptionbuffer(int tagnum);

void G__argtype2param(char *argtype,struct G__param *libp);

void G__letbool(G__value *buf,int type,long value);
long G__bool(G__value buf);


void G__letLonglong(G__value* buf,int type,G__int64 value);
void G__letULonglong(G__value* buf,int type,G__uint64 value);
void G__letLongdouble(G__value* buf,int type,long double value);
G__int64 G__Longlong(G__value buf); /* used to be int */
G__uint64 G__ULonglong(G__value buf); /* used to be int */
long double G__Longdouble(G__value buf); /* used to be int */

void G__display_purevirtualfunc(int tagnum);

#ifndef G__OLDIMPLEMENTATION2226
void G__setmemtestbreak(int n,int m);
#endif

void G__clear_errordictpos();

#ifdef __cplusplus
} // extern "C"
#endif

#else /* __CINT__ */

#define G__fprinterr  fprintf

#endif /* __CINT__ */


#endif /* G__FPROTO_H */


/*
 * Local Variables:
 * c-tab-always-indent:nil
 * c-indent-level:2
 * c-continued-statement-offset:2
 * c-brace-offset:-2
 * c-brace-imaginary-offset:0
 * c-argdecl-indent:0
 * c-label-offset:-2
 * compile-command:"make -k"
 * End:
 */
