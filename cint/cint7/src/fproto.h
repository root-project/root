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

#include "Reflex/Type.h"

extern "C" struct stat;

extern "C" {
   /* G__cfunc.c */
   int G__compiled_func(G__value *result7, char *funcname,struct G__param *libp, int hash, 
      int no_exec_compile, G__value* tempbuf_obj, FILE* intp_sout, FILE* intp_sin);
   void G__list_sut(FILE *fp);
   /* defined Class.cxx used in G__exec_asm.h */
   void G__exec_alloc_lock();
   void G__exec_alloc_unlock();
   void G__dump_reflex();
   void G__dump_reflex_atlevel(const ::Reflex::Scope scope, int level);
   void G__dump_reflex_function(const ::Reflex::Scope scope, int level);
}

namespace Cint {
   // Future public interfaces
   int G__compile_function_bytecode(const ::Reflex::Member &ifunc);

   namespace Internal {

/* G__setup.c */
extern int G__dll_revision(void);
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
G__value G__castvalue_bc(char* casttype, G__value result3, int bc);
G__value G__castvalue(char* casttype, G__value result3);
void G__this_adjustment(const Reflex::Member ifunc);
void G__asm_cast(G__value* buf, const Reflex::Type totype);
  /* void G__setdebugcond(void); */
int G__findposition(char *string,struct G__input_file view,int *pline,int *pfnum);
int G__findfuncposition(char *func,int *pline,int *pfnum);
int G__beforelargestep(char *statement,int *piout,int *plargestep);
void G__afterlargestep(int *plargestep);
void G__EOFfgetc(void);
void G__BREAKfgetc(void);
void G__DISPNfgetc(void);
void G__DISPfgetc(int c);
void G__lockedvariable(const char *item);
int G__lock_variable(const char *varname);
int G__unlock_variable(const char *varname);
G__value G__interactivereturn(void);
void G__set_tracemode(const char *name);
void G__del_tracemode(const char *name);
void G__set_classbreak(const char *name);
void G__del_classbreak(const char *name);
void G__setclassdebugcond(int tagnum,int brkflag);
void G__define_var(int tagnum,::Reflex::Type typenum);
int G__listfunc(FILE *fp,int access,const char* fname = 0,const ::Reflex::Scope &ifunc = ::Reflex::Scope());
int G__listfunc_pretty(FILE* fp, int access, const char* fname, const ::Reflex::Scope ifunc, char friendlyStyle);
int G__showstack(FILE *fout);
void G__display_note(void);
int G__display_proto(FILE *fout,const char *string);
int G__display_proto_pretty(FILE *fout,const char *string,char friendlyStyle);
int G__display_newtypes(FILE *fout,const char *string);
int G__display_typedef(FILE *fout,const char *name,int startin);
int G__display_template(FILE *fout,const char *name);
int G__display_macro(FILE *fout,const char *name);
int G__display_string(FILE *fout);
int G__display_files(FILE *fout);
int G__pr(FILE *fout,struct G__input_file view);
int G__dump_tracecoverage(FILE *fout);
int G__objectmonitor(FILE *fout,char *pobject,const ::Reflex::Type &tagnum,const char *addspace);
int G__varmonitor(FILE* fout, const ::Reflex::Scope scope, const char* mbrname, const char* addspace, long offset);
int G__pushdumpinput(FILE *fp,int exflag);
int G__popdumpinput(void);
int G__dumpinput(char *line);
char *G__xdumpinput(const char *prompt);
int G__free_ifunc_table(::Reflex::Scope& ifunc);
int G__isfilebusy(int ifn);
int G__destroy_upto(::Reflex::Scope& scope, int global, int index);
int G__call_atexit(void);
int G__interpretexit(void);
void G__nosupport(const char *name);
void G__malloc_error(const char *varname);
void G__arrayindexerror(const ::Reflex::Member &var,const char *item,int p_inc);
int G__asm_execerr(const char *message,int num);
int G__assign_error(const char *item,G__value *pbuf);
int G__reference_error(const char *item);
int G__warnundefined(const char *item);
int G__unexpectedEOF(const char *message);
int G__shl_load(char *shlfile);
int G__shl_load_error(const char *shlname,const char *message);
int G__getvariable_error(const char *item);
int G__referencetypeerror(const char *new_name);
int G__syntaxerror(const char *expr);
void G__setDLLflag(int flag);
void G__setCINTLIBNAME(const char * cintlib);
void G__setInitFunc(char *initfunc);
int G__parenthesiserror(const char *expression, const char *funcname);
int G__commenterror(void);
int G__changeconsterror(const char* item, const char* categ);
  /* int G__printlinenum(void); */
int G__autocc(void);
int G__init_readline(void);
int G__using_namespace(void);
void G__initcxx();

int G__pounderror(void);
int G__missingsemicolumn(const char *item);
G__value G__calc_internal(char *exprwithspace);
G__value G__getexpr(const char *expression);
G__value G__getprod(char *expression1);
G__value G__getitem(const char *item);
int G__testandor(int lresult,char *rexpression,int operator2);
int G__test(char *expression2);
int G__btest(int operator2,G__value lresult,G__value rresult);
int G__fgetspace(void);
int G__fgetspace_peek(void);
int G__fgetvarname(char *string,const char *endmark);
int G__fgetname(char *string,const char *endmark);
int G__getname(char* source,int* isrc,char *string,const char *endmark);
int G__fdumpstream(char *string,const char *endmark);
int G__fgetstream(char *string,const char *endmark);
void G__fgetstream_peek(char* string, int nchars);
int G__fignorestream(const char *endmark);
int G__ignorestream(char *string,int* isrc,const char *endmark);
int G__fgetstream_new(char *string,const char *endmark);
void G__fignoreline(void);
void G__fignoreline_peek(void);
void G__fsetcomment(Reflex::Scope &scope);
void G__fsetcomment(struct G__comment_info *pcomment);
int G__fgetc(void);
int G__fgetc_for_peek(void);
long G__op1_operator_detail(int opr,G__value *val);
long G__op2_operator_detail(int opr,G__value *lval,G__value *rval);
int G__explicit_fundamental_typeconv(char* funcname,int hash,struct G__param *libp,G__value *presult3);
int G__special_func(G__value *result7,char *funcname,struct G__param *libp,int hash);
int G__library_func(G__value *result7,char *funcname,struct G__param *libp,int hash);
char *G__charformatter(int ifmt,struct G__param *libp,char *result);
int G__istypename(char *temp);
char* G__savestring(char** pbuf,char* name);
void G__make_ifunctable(char *funcheader);
int G__interpret_func(G__value *result7,const char *funcname,struct G__param *libp,int hash,const ::Reflex::Scope p_ifunc,int funcmatch,int memfunc_flag);
int G__interpret_func(G__value *result7,struct G__param *libp,int hash,const ::Reflex::Member func,int funcmatch,int memfunc_flag);
void G__rate_parameter_match(G__param* libp, const ::Reflex::Member func, G__funclist* funclist, int recursive);
int G__function_signature_match(const Reflex::Member func1, const Reflex::Member func2, bool check_return_type, int /*matchmode*/,int* nref);
::Reflex::Member G__ifunc_exist(::Reflex::Member ifunc_now, const ::Reflex::Scope ifunc, bool check_return_type);
::Reflex::Member G__ifunc_ambiguous(const ::Reflex::Member &ifunc_now,const ::Reflex::Scope &ifunc,const ::Reflex::Type &derivedtagnum);
int G__method_inbase(const Reflex::Member mbr);
void G__inheritclass(int to_tagnum,int from_tagnum,char baseaccess);
int G__baseconstructorwp(void);
int G__baseconstructor(int n,struct G__baseparam *pbaseparam);
int G__basedestructor(void);
int G__basedestructrc(const ::Reflex::Type &mem);
#ifdef G__VIRTUALBASE
int G__publicinheritance(G__value *val1,G__value *val2);
long G__ispublicbase(const ::Reflex::Scope &basetagnum,const ::Reflex::Scope &derivedtagnum,void *pobject);
long G__ispublicbase(const ::Reflex::Type &basetagnum,const ::Reflex::Type &derivedtagnum,void *pobject);
long G__ispublicbase(int basetagnum,int derivedtagnum,void *pobject);
long G__getvirtualbaseoffset(void* pobject,int tagnum,struct G__inheritance *baseclass,int basen);
#else
int G__ispublicbase(int basetagnum,int derivedtagnum);
#endif
long G__find_virtualoffset(int virtualtag);
  /* int G__main(int argc,char **argv); */
int G__init_globals(void);
void G__set_stdio(void);
int G__cintrevision(FILE *fp);
  /* char *G__input(char *prompt); */
const char *G__strrstr(const char *string1,const char *string2);
char *G__strrstr(char *string1,const char *string2);

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

int G__errorprompt(const char *nameoferror);
int G__call_interruptfunc(char *func);

int G__include_file(void);
int G__getcintsysdir(void);
int G__preprocessor(char *outname,char *inname,int cppflag,char *macros,char *undeflist,char *ppopt,char *includepath);
int G__difffile(char *file1,char *file2);
int G__copyfile(FILE *to,FILE *from);

int G__statfilename(const char *filename, struct stat* buf);
int G__matchfilename(int i1,const char* filename);
int G__cleardictfile(int flag);

void G__openmfp(void);
int G__closemfp(void);
void G__define(void);
int G__handle_as_typedef(char *oldtype,char *newtype);
void G__createmacro(const char *new_name,char *initvalue);
G__value G__execfuncmacro(const char *item,int *done);
int G__transfuncmacro(const char *item,struct G__Deffuncmacro *deffuncmacro,struct G__Callfuncmacro *callfuncmacro,fpos_t call_pos,char *p,int nobraces,int nosemic);
int G__replacefuncmacro(const char *item,struct G__Callfuncmacro *callfuncmacro,struct G__Charlist *callpara,struct G__Charlist *defpara,FILE *def_fp,fpos_t def_pos,int nobraces,int nosemic);
int G__execfuncmacro_noexec(const char* macroname);
int G__maybe_finish_macro(void);
int G__argsubstitute(char *symbol,struct G__Charlist *callpara,struct G__Charlist *defpara);
int G__createfuncmacro(const char *new_name);
int G__getparameterlist(char *paralist,struct G__Charlist *charlist);
int G__freedeffuncmacro(struct G__Deffuncmacro *deffuncmacro);
int G__freecharlist(struct G__Charlist *charlist);
void *G__malloc(int n,int bsize,const char *item);
void G__set_static_varname(const char*);
void *G__TEST_Malloc(size_t size);
void *G__TEST_Calloc(size_t n,size_t bsize);
void G__TEST_Free(void *p);
void *G__TEST_Realloc(void *p,size_t size);
int G__memanalysis(void);
int G__memresult(void);
void G__DUMMY_Free(void *p);
void *G__TEST_fopen(char *fname,char *mode);
int G__TEST_fclose(FILE *p);
G__value G__new_operator(const char *expression);
int G__getarrayindex(const char *indexlist);
void G__delete_operator(const char *expression,int isarray);
int G__alloc_newarraylist(void* point,int pinc);
int G__free_newarraylist(void* point);
int G__handle_delete(int *piout,char *statement);
void G__gen_cppheader(char *headerfile);
void G__gen_clink(void);
void G__gen_cpplink(void);
void G__clink_header(FILE *fp);
void G__cpplink_header(FILE *fp);
void G__cpplink_linked_taginfo(FILE* fp,FILE* hfp);
char *G__get_link_tagname(int tagnum);
/* char *G__map_cpp_name(char *in); */
char *G__map_cpp_funcname(int tagnum,char *funcname,long ifn,int page);
void G__set_globalcomp(const char *mode,const char *linkfilename,const char* dllid);
int G__ishidingfunc(struct G__ifunc_table *fentry,struct G__ifunc_table *fthis,int ifn);
void G__cppif_memfunc(FILE *fp,FILE *hfp);
void G__cppif_func(FILE *fp,FILE *hfp);
void G__cppif_dummyfuncname(FILE *fp);
void G__cppif_genconstructor(FILE *fp,FILE *hfp,int tagnum,const ::Reflex::Member &ifunc);
int G__isprivateconstructor(int tagnum,int iscopy);
void G__cppif_gendefault(FILE *fp,FILE *hfp,int tagnum,int isconstructor,int iscopyconstructor,int isdestructor,int isassignmentoperator,int isnonpublicnew);
void G__cppif_genfunc(FILE *fp,FILE *hfp,int tagnum,const ::Reflex::Member &ifunc);
int G__cppif_returntype(FILE *fp,const ::Reflex::Member &ifunc,char *endoffunc);
void G__cppif_paratype(FILE *fp, const ::Reflex::Member &ifunc, int k);
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
void G__incsetup_memvar(::Reflex::Scope &tagnum);
void G__incsetup_memfunc(::Reflex::Scope &tagnum);
void G__incsetup_memvar(::Reflex::Type &tagnum);
void G__incsetup_memfunc(::Reflex::Type &tagnum);
int G__compiled_func_cxx(G__value *result7, char *funcname,struct G__param *libp, int hash);


int G__separate_parameter(const char *original,int *pos,char *param);
int G__parse_parameter_link(char *paras);
int G__cppif_p2memfunc(FILE *fp);
int G__set_sizep2memfunc(FILE *fp);
int G__getcommentstring(char* buf, int tagnum, struct G__comment_info* pcomment);
void G__bstore(int operatorin,G__value expressionin,G__value *defined);
void G__doubleassignbyref(G__value *defined,double val);
void G__intassignbyref(G__value *defined,G__int64 val);
int G__scopeoperator(char *name,int *phash,char **pstruct_offset,int *ptagnum);
int G__cmp(G__value buf1,G__value buf2);
int G__getunaryop(char unaryop,char *expression,char *buf,G__value *preg);
int G__overloadopr(int operatorin,G__value expressionin,G__value *defined);
int G__parenthesisovldobj(G__value *result3,G__value *result,const char *realname,struct G__param *libp,int flag);
int G__parenthesisovld(G__value *result3,const char *funcname,struct G__param *libp,int flag);
int G__tryindexopr(G__value *result7,G__value *para,int paran,int ig25);
int G__skip_comment(void);
int G__skip_comment_peek(void);
int G__pp_command(void);
void G__pp_skip(int elifskip);
int G__pp_if(void);
int G__defined_macro(const char *macro);
int G__pp_ifdef(int def);
G__value G__alloc_tempstring(char *string);
G__value G__exec_statement(int *mparen);
int G__update_stdio(void);
void G__set_history_size(int s);
  /* int G__pause(void); */
int G__setaccess(char *statement,int iout);
 int G__class_conversion_operator(const ::Reflex::Type &tagnum,G__value *presult);
int G__fundamental_conversion_operator(int type,int tagnum,::Reflex::Type typenum,int reftype,int constvar,G__value *presult);
long G__asm_gettest(int op,long *inst);
int G__asm_optimize(int *start);
int G__asm_optimize3(int *start);
int G__inc_cp_asm(int cp_inc,int dt_dec);
int G__clear_asm(void);
int G__asm_clear(void);
void G__gen_addstros(long addstros);
void G__suspendbytecode(void);
void G__resetbytecode(void);
void G__resumebytecode(int store_asm_noverflow);
void G__abortbytecode(void);
int G__asm_putint(int i);
G__value G__getreserved(const char *item,void** ptr,void** ppdict);
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
void G__asm_get_strip_quotation(G__value *pval);
G__value G__strip_quotation(const char *string);
char *G__charaddquote(char *string,char c);
G__value G__strip_singlequotation(const char *string);
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
bool G__search_func(char *funcname,G__value *buf);
char *G__search_next_member(char *text,int state);
long G__Loffsetof(char *tagname,char *memname);
long *G__typeid(char *typenamein);
void G__getcomment(char* buf, int tagnum);
void G__getcomment(char* buf, Reflex::Scope scope);
void G__getcomment(char* buf, struct G__comment_info* pcomment, int tagnum);
void G__getcommenttypedef(char *buf,struct G__comment_info *pcomment,::Reflex::Type typenum);
long G__get_classinfo(char *item,int tagnum);
long G__get_variableinfo(char *item,long *phandle,long *pindex,int tagnum);
long G__get_functioninfo(char *item,long *phandle,long *pindex,int tagnum);
::Reflex::Scope G__get_envtagnum(void);
int G__isenclosingclass(const ::Reflex::Scope enclosingtagnum, const ::Reflex::Scope env_tagnum);
int G__isenclosingclassbase(int enclosingtagnum, int env_tagnum);
int G__isenclosingclassbase(const ::Reflex::Scope enclosingtagnum, const ::Reflex::Scope env_tagnum);
char* G__find_first_scope_operator(char* name);
char* G__find_last_scope_operator(char* name);
void G__define_struct(char type);
void G__create_global_namespace();
void G__create_bytecode_arena();
G__value G__classassign(char *pdest,const ::Reflex::Type &tagnum,G__value result);
int G__fgetname_template(char *string,const char *endmark);
int G__fgetstream_newtemplate(char *string,const char *endmark);
int G__fgetstream_template(char *string,const char *endmark);
int G__fgetstream_spaces(char *string,const char *endmark);
int G__getstream_template(const char *source,int *isrc,char *string,const char *endmark);
struct G__Definetemplatefunc* G__defined_templatefunc(const char* name);
struct G__Definetemplatefunc* G__defined_templatememfunc(const char* name);
void G__declare_template();
int G__createtemplateclass(char* new_name, G__Templatearg* targ, int isforwarddecl);
int G__instantiate_templateclass(char* tagnamein, int noerror);
void G__freedeftemplateclass(G__Definedtemplateclass* deftmpclass);
char* G__gettemplatearg(int n, G__Templatearg* def_para);
void G__freetemplatefunc(G__Definetemplatefunc* deftmpfunc);
struct G__funclist* G__add_templatefunc(const char* funcnamein, G__param* libp, int hash, G__funclist* funclist, const ::Reflex::Scope p_ifunc, int isrecursive);
struct G__funclist* G__funclist_add(struct G__funclist* last, const ::Reflex::Member ifunc, int ifn, int rate);
void G__funclist_delete(struct G__funclist* body);
int G__templatefunc(G__value* result, const char* funcname, G__param* libp, int hash, int funcmatch);
void G__define_type(void);
int G__defined_type(char *typenamein,int len);
char *G__valuemonitor(G__value buf,char *temp);
const char *G__access2string(int caccess);
const char *G__tagtype2string(int tagtype);
bool G__rename_templatefunc(std::string &funcname);
const std::string G__fulltypename(::Reflex::Type typenum);
int G__val2pointer(G__value *result7);
char *G__getbase(unsigned int expression,int base,int digit,char *result1);
int G__getdigit(unsigned int number);
G__value G__checkBase(const char *string,int *known4);
int G__isfloat(const char *string,int *type);
int G__isoperator(int c);
int G__isexponent(const char *expression4,int lenexpr);
int G__isvalue(const char *temp);
int G__isdouble(G__value buf);

/* float G__float(G__value buf); */
G__value G__tovalue(G__value p);
G__value G__toXvalue(G__value result,int var_type);
G__value G__letVvalue(G__value *p,G__value result);
G__value G__letPvalue(G__value *p,G__value result);
G__value G__letvalue(G__value *p,G__value result);
G__value G__letvariable(const char* item, G__value expression, const ::Reflex::Scope varglobal, const ::Reflex::Scope varlocal, Reflex::Member& output_var);
G__value G__letvariable(const char* item, G__value expression, const ::Reflex::Scope varglobal, const ::Reflex::Scope varlocal);
G__value G__getvariable(char *item,int *known2,const ::Reflex::Scope &varglobal,const ::Reflex::Scope &varlocal);
G__value G__getstructmem(int store_var_type,char *varname,char *membername,char *tagname,int *known2,const ::Reflex::Scope &varglobal,int objptr);
G__value G__letstructmem(int store_var_type,const char *varname,char *membername,char *tagname,const ::Reflex::Scope &varglobal,G__value expression,int objptr,Reflex::Member &output_var);
void G__letstruct(G__value *result,int p_inc,const ::Reflex::Member &var,const char *item,int paran,char *G__struct_offset);
void G__letstructp(G__value result,char *G__struct_offset,int p_inc,const ::Reflex::Member &var,int paran,const char *item,G__value *para,int pp_inc);
void G__returnvartype(G__value* presult,const ::Reflex::Member &var,int paran);
::Reflex::Member G__getvarentry(const char *varname,int varhash,const ::Reflex::Scope &varglobal,const ::Reflex::Scope &varlocal);
int G__getthis(G__value *result7,const char *varname,const char *item);
void G__letpointer2memfunc(const ::Reflex::Member &var,int paran,const char *item,int p_inc,G__value *presult,char *G__struct_offset);
void G__letautomatic(const ::Reflex::Member &var,char *G__struct_offset,int p_inc,G__value result);
::Reflex::Member G__add_scopemember(::Reflex::Scope envvar, const char* varname, const ::Reflex::Type type, int reflex_modifiers, size_t reflex_offset, char* cint_offset, int access, int statictype);
void G__display_classkeyword(FILE *fout,const char *classnamein,const char *keyword,int base);
void G__display_tempobject(const char* action);
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
G__UINT32 G__getsecuritycode(char *string);
#endif

extern "C" void G__cpp_setupG__stream(void);
extern "C" void G__cpp_setupG__API(void);
extern "C" void G__c_setupG__stdstrct(void);

int G__setautoccnames(void);
int G__appendautocc(FILE *fp);
int G__isautoccupdate(void);
struct G__friendtag* G__new_friendtag(int tagnum);
struct G__friendtag* G__copy_friendtag(const G__friendtag* orig);
void G__free_friendtag(G__friendtag* friendtag);

int G__free_exceptionbuffer(void);
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

// Note: The return type must be by-reference,
//       this routine is used as a lvalue.
#if defined(__GNUC__) && (__GNUC__ > 4 || ((__GNUC__ == 4) && (__GNUC_MINOR__ > 1)))
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif // __GNUC__ && __GNUC__ > 3 && __GNUC_MINOR__ > 1
inline ::Reflex::Type& G__value_typenum(G__value& gv) { return *(::Reflex::Type*) &gv.buf_typenum; }
inline const ::Reflex::Type& G__value_typenum(const G__value& gv) { return *(::Reflex::Type*) &gv.buf_typenum; }

extern void G__more_col(int len);
extern int G__more(FILE* fp,const char *msg);
extern int G__more_pause(FILE* fp,int len);
extern void G__redirect_on(void);
extern void G__redirect_off(void);

void G__init_jumptable_bytecode(void);
void G__add_label_bytecode(char *label);
void G__add_jump_bytecode(char *label);
void G__resolve_jumptable_bytecode(void);

extern void G__LockCriticalSection(void);
extern void G__UnlockCriticalSection(void);

extern void G__CMP2_equal(G__value*,G__value*);

extern int G__gen_linksystem(char* headerfile);

extern void G__smart_shl_unload(int allsl);
extern void G__smart_unload(int ifn);

extern struct G__dictposition* G__get_dictpos(char* fname);

void G__specify_extra_include(void);
void G__gen_extra_include(void);

struct G__ConstStringList* G__AddConstStringList(struct G__ConstStringList* current,char* str,int islen);
void G__DeleteConstStringList(struct G__ConstStringList* current);

int G__ReadInputMode(void);

char* G__setiparseobject(G__value* result,char *str);

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
void* G__FindSymbol(const ::Reflex::Member &ifunc);
void* G__GetShlHandle();
int G__GetShlFilenum();
void* G__FindSym(const char *filename,const char *funcname);

int G__register_sharedlib(const char *libname);
int G__unregister_sharedlib(const char *libname);
int G__RegisterLibrary (void (*func) ());
int G__UnregisterLibrary (void (*func) ());
      
int G__loadfile_tmpfile(FILE *fp);

int G__class_autoloading(int *tagnum);

#ifndef G__OLDIMPLEMENTATION2030
int G__callfunc0(G__value* result, const Reflex::Member func, struct G__param* libp, void* p, int funcmatch);
int G__calldtor(void* p, const Reflex::Scope tagnum, int isheap);
#endif // G__OLDIMPLEMENTATION2030

void G__init_replacesymbol();
void G__add_replacesymbol(const char* s1,const char* s2);
const char* G__replacesymbol(const char* s);
int G__display_replacesymbol(FILE *fout,const char* name);

void G__asm_storebytecodefunc(const Reflex::Member& func,const Reflex::Scope& frame,G__value *pstack,int sp,long *pinst,int instsize);

G__value G__alloc_exceptionbuffer(int tagnum);

void G__argtype2param(const char *argtype,struct G__param *libp);

long G__bool(G__value buf);

void G__display_purevirtualfunc(int tagnum);

#ifndef G__OLDIMPLEMENTATION2226
void G__setmemtestbreak(int n,int m);
#endif

void G__clear_errordictpos();
void G__setcopyflag(int flag);

void G__get_cint5_type_tuple(const ::Reflex::Type in_type, char* out_type, int* out_tagnum, int* out_typenum, int* out_reftype, int* out_constvar);
void G__get_cint5_type_tuple_long(const ::Reflex::Type in_type, long* out_type, long* out_tagnum, long* out_typenum, long* out_reftype, long* out_constvar);
int G__get_cint5_typenum(const ::Reflex::Type in_type);
inline int G__get_type(const ::Reflex::Type in) { return in.RepresType(); }
inline int G__get_type(const G__value in) { return G__value_typenum(in).RepresType(); }
int G__get_tagtype(const ::Reflex::Type in);
int G__get_tagtype(const ::Reflex::Scope in);
int G__get_reftype(const ::Reflex::Type in);
G__SIGNEDCHAR_T G__get_isconst(const ::Reflex::Type in);
int G__get_tagnum(const ::Reflex::Type in);
int G__get_tagnum(const ::Reflex::Scope in);
int G__get_typenum(const ::Reflex::Type in);
int G__get_nindex(const ::Reflex::Type in);
std::vector<int> G__get_index(const ::Reflex::Type in);
int G__get_varlabel(const ::Reflex::Type in, int idx);
int G__get_varlabel(const ::Reflex::Member in, int idx);
int G__get_paran(const Reflex::Member var);

void G__get_stack_varname(std::string &output,const char *varname,const ::Reflex::Member &m,int tagnum);

::G__RflxProperties* G__get_properties(const ::Reflex::Type in);
::G__RflxProperties* G__get_properties(const ::Reflex::Scope in);
::G__RflxVarProperties* G__get_properties(const ::Reflex::Member in);
::G__RflxFuncProperties* G__get_funcproperties(const ::Reflex::Member in);
size_t GetReflexPropertyID();
template< class T, class Prop >
void G__set_properties(const T& in, const Prop& rp) {
   in.Properties().AddProperty(GetReflexPropertyID(), rp);
}
size_t G__get_bitfield_width(const ::Reflex::Member in);
size_t G__get_bitfield_start(const ::Reflex::Member in);

int G__sizeof_deref(const G__value*);
::Reflex::Type G__strip_array(const ::Reflex::Type typein);
::Reflex::Type G__strip_one_array(const Reflex::Type typein);
::Reflex::Type G__deref(const ::Reflex::Type typein);
::Reflex::Type G__modify_type(const ::Reflex::Type typein
                                    ,bool ispointer
                                    ,int reftype,int isconst
                                    ,int nindex, int *index);
::Reflex::Type G__cint5_tuple_to_type(int type, int tagnum, int typenum, int reftype, int isconst);
::Reflex::Member G__update_array_dimension(::Reflex::Member member, size_t nelem );
::Reflex::Type G__get_from_type(int type, int createpointer, int isconst = 0);
::Reflex::Type G__find_type(const char *type_name, int errorflag, int templateflag);
::Reflex::Type G__find_typedef(const char*,int noerror = 0);
::Reflex::Scope G__findInScope(const ::Reflex::Scope scope, const char* name);
::Reflex::Type G__declare_typedef(const char *typenamein,
                                        int typein,int tagnum,int reftype,
                                        int isconst, int globalcomp,
                                        int parent_tagnum,
                                        bool pointer_fix);
::Reflex::Member G__find_variable(const char* varname, int varhash, const ::Reflex::Scope varlocal, const ::Reflex::Scope varglobal, char** pG__struct_offset, char** pstore_struct_offset, int* pig15, int isdecl);

bool G__test_access(const ::Reflex::Member var, int access);
bool G__is_cppmacro(const ::Reflex::Member var);
bool G__filescopeaccess(int filenum, int statictype);
int G__get_access(const ::Reflex::Member mem);
inline char*& G__get_offset(const ::Reflex::Member& mbr) { return mbr.InterpreterOffset(); }
Reflex::Type G__replace_rawtype(const Reflex::Type target, const Reflex::Type raw);
Reflex::Type G__apply_const_to_typedef(const Reflex::Type target);

void G__set_G__tagnum(const ::Reflex::Scope);
void G__set_G__tagnum(const ::Reflex::Type);
void G__set_G__tagnum(const G__value&);


#if defined(G__WIN32) && (!defined(G__SYMANTEC)) && defined(G__CINTBODY)
/* ON562 , this used to be test for G__SPECIALSTDIO */
/************************************************************************
* Dummy I/O function for all Win32 based application
************************************************************************/
#ifdef printf
#undef printf
#endif
#ifdef fprintf
#undef fprintf
#endif
#ifdef fputc
#undef fputc
#endif
#ifdef putc
#undef putc
#endif
#ifdef putchar
#undef putchar
#endif
#ifdef fputs
#undef fputs
#endif
#ifdef puts
#undef puts
#endif
#ifdef fgets
#undef fgets
#endif
#ifdef gets
#undef gets
#endif
#ifdef tmpfile
#undef tmpfile
#endif
#define printf  G__printf
#define fprintf G__fprintf
#define fputc   ::Cint::Internal::G__fputc
#define putc    ::Cint::Internal::G__fputc
#define putchar ::Cint::Internal::G__putchar
#define fputs   ::Cint::Internal::G__fputs
#define puts    ::Cint::Internal::G__puts
#define fgets   ::Cint::Internal::G__fgets
#define gets    ::Cint::Internal::G__gets
#define system  ::Cint::Internal::G__system
#define tmpfile ::Cint::Internal::G__tmpfile

int G__fputc(int character,FILE *fp);
int G__putchar(int character);
int G__fputs(char *string,FILE *fp);
int G__puts(char *string);
char *G__fgets(char *string,int n,FILE *fp);
char *G__gets(char *buffer);
int G__system(char *com);
FILE *G__tmpfile();
const char* G__tmpfilenam();
      

#ifdef G__SPECIALSTDIO

/* THIS IS AN OLD WILDC++ IMPLEMENTATION */
/* signal causes problem in Windows-95 with Tcl/Tk */
#define signal(sigid,p2f)  NULL
#define alarm(time)        NULL

#else /* G__SPECIALSTDIO */

#ifdef signal
#undef signal
#endif
#define signal ::Cint::Internal::G__signal
#define alarm(time)        NULL
typedef void (*G__signaltype)(int,void (*)(int));
G__signaltype G__signal(int sgnl,void (*f)(int));

#endif /* G__SPECIALSTDIO */

#endif /* WIN32 !SYMANTEC CINTBODY*/
/**************************************************************************
* end of specialstdio or win32
**************************************************************************/

   } // namespace Internal
} // namespace Cint

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
