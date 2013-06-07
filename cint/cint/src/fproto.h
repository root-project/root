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
namespace Cint {
   class G__DataMemberHandle;
}
class G__FastAllocString;
struct G__AppPragma;

extern "C" {
#endif
void psrxxx_dump_gvars();
/* G__cfunc.c */
int G__compiled_func(G__value *result7,const char *funcname,struct G__param *libp,int hash);
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
G__value G__castvalue(char *casttype,G__value result3);
G__value G__castvalue_bc(char *casttype,G__value result3, int i);
void* G__dynamiccast(int totagnum, int fromtagnum, void* addr);
void G__this_adjustment(struct G__ifunc_table_internal *ifunc, int ifn);
void G__asm_cast(int type,G__value *buf,int tagnum,int reftype);
  /* void G__setdebugcond(void); */
int G__findposition(const char *string,struct G__input_file* view,int *pline,int *pfnum);
int G__beforelargestep(char *statement,int *piout,int *plargestep);
void G__afterlargestep(int *plargestep);
void G__EOFfgetc(void);
void G__BREAKfgetc(void);
void G__DISPNfgetc(void);
void G__DISPfgetc(int c);
int G__lock_variable(const char *varname);
int G__unlock_variable(const char *varname);
G__value G__interactivereturn(void);
void G__set_tracemode(char *name);
void G__del_tracemode(char *name);
void G__set_classbreak(char *name);
void G__del_classbreak(char *name);
void G__setclassdebugcond(int tagnum,int brkflag);
void G__define_var(int tagnum,int typenum);
struct G__var_array* G__initmemvar(int tagnum,int* pindex,G__value *pbuf);
struct G__var_array* G__incmemvar(struct G__var_array* memvar,int* pindex,G__value *pbuf);
int G__listfunc(FILE *fp,int access,const char* fname,struct G__ifunc_table *ifunc);
int G__listfunc_pretty(FILE *fp,int access,const char* fname,struct G__ifunc_table *ifunc,char friendlyStyle);
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
#ifdef __cplusplus
int G__pr(FILE *fout,const struct G__input_file &view);
#endif
int G__dump_tracecoverage(FILE *fout);
int G__objectmonitor(FILE *fout,long pobject,int tagnum,const char *addspace);
int G__varmonitor(FILE *fout,struct G__var_array *var,const char *index,const char *addspace,long offset);
int G__pushdumpinput(FILE *fp,short int exflag);
int G__popdumpinput(void);
int G__dumpinput(const char *line);
char *G__xdumpinput(const char *prompt);
int G__free_ifunc_table(struct G__ifunc_table_internal *ifunc);
int G__isfilebusy(int ifn);
int G__destroy_upto(struct G__var_array *var,int global,struct G__var_array *dictpos,int ig15);
int G__call_atexit(void);
int G__interpretexit(void);
void G__nosupport(const char* name);
void G__malloc_error(const char* varname);
void G__arrayindexerror(int varid, struct G__var_array* var, const char* name, int index);
#ifdef G__ASM
int G__asm_execerr(const char* message, int num);
#endif // G__ASM
int G__assign_using_null_pointer_error(const char* item);
int G__assign_error(const char* item, G__value* pbuf);
int G__reference_error(const char* item);
int G__warnundefined(const char* item);
int G__unexpectedEOF(const char* message);
int G__shl_load_error(const char* shlname, const char* message);
int G__getvariable_error(const char* item);
int G__referencetypeerror(const char* new_name);
int G__syntaxerror(const char* expr);
int G__parenthesiserror(const char* expression, const char* funcname);
int G__commenterror();
int G__changeconsterror(const char* item, const char* categ);
int G__pounderror();
int G__missingsemicolumn(const char* item);
void G__printerror(const char* funcname, int ipara, int paran);
#ifdef G__SECURITY
int G__check_drange(int p, double low, double up, double d, G__value* result7, const char* funcname);
int G__check_lrange(int p, long low, long up, long l, G__value* result7, const char* funcname);
int G__check_type(int p, int t1, int t2, G__value* para, G__value* result7, const char* funcname);
int G__check_nonull(int p, int t, G__value* para, G__value* result7, const char* funcname);
#endif // G__SECURITY
int G__shl_load(char *shlfile);
void G__setDLLflag(int flag);
void G__setInitFunc(char *initfunc);
int G__autocc(void);
int G__init_readline(void);
int G__using_namespace(void);
G__value G__calc_internal(const char *exprwithspace);
G__value G__getexpr(const char *expression);
G__value G__getprod(char *expression1);
G__value G__getpower(const char *expression2);
G__value G__getitem(const char *item);
long G__testandor(int lresult,const char *rexpression,int operator2);
long G__test(const char *expression2);
long G__btest(int operator2,G__value lresult,G__value rresult);
int G__fgetspace(void);
int G__fgetspace_peek(void);
#ifdef __cplusplus
} // extern "C"
char* G__setiparseobject(G__value* result,G__FastAllocString &str);
int G__fgetvarname(G__FastAllocString& string, size_t offset, const char *endmark);
int G__fgetname(G__FastAllocString& string, size_t offset, const char *endmark);
int G__getname(const char* source,int* isrc,G__FastAllocString& string,const char *endmark);
int G__fgetstream(G__FastAllocString& string, size_t offset, const char *endmark);
void G__fgetstream_peek(G__FastAllocString& string, int nchars);
int G__fgetstream_new(G__FastAllocString& string, size_t offset, const char *endmark);
int G__fdumpstream(G__FastAllocString& string, size_t offset, const char *endmark);
int G__getcommentstring(G__FastAllocString& buf,int tagnum,struct G__comment_info *pcomment);
extern "C" {
#endif
int G__ignorestream(const char *string,int* isrc,const char *endmark);
int G__fignorestream(const char *endmark);
void G__fignoreline(void);
void G__fignoreline_peek(void);
void G__fsetcomment(struct G__comment_info *pcomment);
int G__fgetc(void);
long G__op1_operator_detail(int opr,G__value *val);
long G__op2_operator_detail(int opr,G__value *lval,G__value *rval);
int G__explicit_fundamental_typeconv(char* funcname,int hash,struct G__param *libp,G__value *presult3);
int G__special_func(G__value *result7,char *funcname,struct G__param *libp,int hash);
int G__library_func(G__value *result7,char *funcname,struct G__param *libp,int hash);
char *G__charformatter(int ifmt,struct G__param *libp,char *result,size_t result_length);
int G__istypename(char *temp);
char* G__savestring(char** pbuf,char* name);
struct G__ifunc_table* G__get_ifunc_ref(struct G__ifunc_table_internal*);
void G__reset_ifunc_refs_for_tagnum(int tagnum);
int G__interpret_func(G__value *result7,const char *funcname,struct G__param *libp,int hash,struct G__ifunc_table_internal *p_ifunc,int funcmatch,int memfunc_flag);
struct G__ifunc_table_internal *G__ifunc_exist(struct G__ifunc_table_internal *ifunc_now,int allifunc,struct G__ifunc_table_internal *ifunc,int *piexist,int mask);
struct G__ifunc_table_internal *G__ifunc_ambiguous(struct G__ifunc_table_internal *ifunc_now,int allifunc,struct G__ifunc_table_internal *ifunc,int *piexist,int derivedtagnum);
int G__method_inbase(int ifn, struct G__ifunc_table_internal *ifunc);
void G__inheritclass(int to_tagnum,int from_tagnum,char baseaccess);
int G__baseconstructorwp(void);
int G__baseconstructor(int n,struct G__baseparam *pbaseparam);
int G__basedestructor(void);
int G__basedestructrc(struct G__var_array *mem);
#ifdef G__VIRTUALBASE
long G__publicinheritance(G__value *val1,G__value *val2);
long G__ispublicbase(int basetagnum,int derivedtagnum,long pobject);
long G__getvirtualbaseoffset(long pobject,int tagnum,struct G__inheritance *baseclass,int basen);
#else
long G__ispublicbase(int basetagnum,int derivedtagnum);
long G__isanybase(int basetagnum,int derivedtagnum);
#endif
#ifdef G__VIRTUALBASE
long G__find_virtualoffset(long virtualtag, long pobject);
#else
long G__find_virtualoffset(long virtualtag);
#endif
  /* int G__main(int argc,char **argv); */
int G__init_globals(void);
void G__set_stdio(void);
int G__cintrevision(FILE *fp);
  /* char *G__input(char *prompt); */
const char *G__strrstr(const char *string1,const char *string2);
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

int G__errorprompt(const char *nameoferror);
int G__call_interruptfunc(char *func);

int G__include_file(void);
int G__getcintsysdir(void);
int G__preprocessor(char *outname,const char *inname,int cppflag,const char *macros,const char *undeflist,const char *ppopt,const char *includepath);
int G__difffile(char *file1,char *file2);
int G__copyfile(FILE *to,FILE *from);

#ifdef __cplusplus
struct stat;
} // extern "C" 
int G__statfilename(const char *filename, struct stat* buf, G__FastAllocString* fullPath);
extern "C" {
#endif
int G__matchfilename(int i1,const char* filename);
int G__cleardictfile(int flag);

void G__openmfp(void);
int G__closemfp(void);
void G__define(void);
G__value G__execfuncmacro(const char *item,int *done);
int G__execfuncmacro_noexec(const char* macroname);
int G__maybe_finish_macro(void);
int G__freedeffuncmacro(struct G__Deffuncmacro *deffuncmacro);
int G__freecharlist(struct G__Charlist *charlist);
long G__malloc(int n,int bsize,const char *item);
void *G__TEST_Malloc(size_t size);
void *G__TEST_Calloc(size_t n,size_t bsize);
void G__TEST_Free(void *p);
void *G__TEST_Realloc(void *p,size_t size);
int G__memanalysis(void);
int G__memresult(void);
void G__DUMMY_Free(void *p);
void *G__TEST_fopen(const char *fname,const char *mode);
int G__TEST_fclose(FILE *p);
G__value G__new_operator(const char *expression);
int G__getarrayindex(const char *indexlist);
void G__delete_operator(char *expression,int isarray);
int G__alloc_newarraylist(long point,int pinc);
int G__free_newarraylist(long point);
int G__call_cppfunc(G__value *result7,struct G__param *libp,struct G__ifunc_table_internal *ifunc,int ifn);
void G__gen_cppheader(char *headerfile);
void G__gen_clink(void);
void G__gen_cpplink(void);
void G__clink_header(FILE *fp);
void G__cpplink_header(FILE *fp);
void G__cpplink_linked_taginfo(FILE* fp,FILE* hfp);
char *G__get_link_tagname(int tagnum);
/* char *G__map_cpp_name(char *in); */
char *G__map_cpp_funcname(int tagnum,const char *funcname,int ifn,int page);
void G__set_globalcomp(const char *mode,const char *linkfilename,const char* dllid);
int G__ishidingfunc(struct G__ifunc_table_internal *fentry,struct G__ifunc_table_internal *fthis,int ifn);
void G__cppif_memfunc(FILE *fp,FILE *hfp);
void G__cppif_func(FILE *fp,FILE *hfp);
void G__cppif_dummyfuncname(FILE *fp);
void G__cppif_genconstructor(FILE *fp,FILE *hfp,int tagnum,int ifn,struct G__ifunc_table_internal *ifunc);
int G__isprivateconstructor(int tagnum,int iscopy);
void G__cppif_gendefault(FILE *fp,FILE *hfp,int tagnum,int ifn,struct G__ifunc_table_internal *ifunc,int isconstructor,int iscopyconstructor,int isdestructor,int isassignmentoperator,int isnonpublicnew);
void G__cppif_genfunc(FILE *fp,FILE *hfp,int tagnum,int ifn,struct G__ifunc_table_internal *ifunc);
int G__cppif_paratype(FILE *fp,int ifn,struct G__ifunc_table_internal *ifunc,int k);
void G__cpplink_tagtable(FILE *pfp,FILE *hfp);
#ifdef G__VIRTUALBASE
long G__iosrdstate(G__value *pios);
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

// 04-07-07
// (stub-less calls)
void* G__get_funcptr(struct G__ifunc_table_internal *ifunc, int ifn);
int   G__defined_typename_noerror(const char *type_name, int noerror);
G__value G__string2type_noerror(const char *typenamin, int noerror);
struct G__ifunc_table *G__get_methodhandle_noerror(const char *funcname,const char *argtype
                                           ,struct G__ifunc_table *p_ifunc
                                           ,long *pifn,long *poffset
                                           ,int withConversion
                                           ,int withInheritance
                                           ,int noerror,int isconst);
#ifdef G__NOSTUBS
int   G__stub_method_calling (G__value *result7, struct G__param *libp, struct G__ifunc_table_internal *ifunc, int ifn);
void  G__register_class(const char *libname, const char *clstr);

struct G__ifunc_table_internal *G__get_methodhandle4(char *funcname
                                           ,struct G__param* libp
                                           ,struct G__ifunc_table_internal *p_ifunc
                                           ,long *pifn,long *poffset
                                           ,int withConversion
                                           ,int withInheritance,int isconst);
#endif //G__NOSTUBS

void G__setnewtype_settypeum(int typenum);
int G__parse_parameter_link(char *paras);
int G__cppif_p2memfunc(FILE *fp);
int G__set_sizep2memfunc(FILE *fp);
void G__bstore(int operatorin,G__value expressionin,G__value *defined);
void G__doubleassignbyref(G__value *defined,double val);
void G__intassignbyref(G__value *defined,G__int64 val);
int G__scopeoperator(char *name,int *phash,long *pstruct_offset,int *ptagnum);
int G__cmp(G__value buf1,G__value buf2);
int G__getunaryop(const char unaryop,const char *expression,char *buf,G__value *preg);
int G__overloadopr(int operatorin,G__value expressionin,G__value *defined);
int G__parenthesisovldobj(G__value *result3,G__value *result,const char *realname,struct G__param *libp,int flag);
int G__parenthesisovld(G__value *result3,char *funcname,struct G__param *libp,int flag);
int G__tryindexopr(G__value *result7,G__value *para,int paran,int ig25);
int G__skip_comment(void);
int G__skip_comment_peek(void);
int G__pp_command(void);
void G__pp_skip(int elifskip);
int G__pp_if(void);
int G__defined_macro(const char *macro);
int G__pp_ifdef(int def);
void G__free_tempobject(void);
G__value G__exec_statement(int* mparen);
int G__update_stdio(void);
  /* int G__pause(void); */
int G__setaccess(char *statement,int iout);
int G__class_conversion_operator(int tagnum,G__value *presult,char* ttt);
int G__fundamental_conversion_operator(int type,int tagnum,int typenum,int reftype,int constvar,G__value *presult,char* ttt);
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
void G__free_bytecode(struct G__bytecodefunc *bytecode);
void G__asm_gen_strip_quotation(G__value *pval);
int G__security_handle(G__UINT32 category);
void G__asm_get_strip_quotation(G__value *pval);
G__value G__strip_quotation(const char *string);
G__value G__strip_singlequotation(char *string);
char *G__quotedstring(char *buf,char *result);
char *G__logicstring(G__value buf,int dig,char *result);
int G__revprint(FILE *fp);
int G__dump_header(char *outfile);
void G__listshlfunc(FILE *fout);
void G__listshl(FILE *G__temp);
int G__free_shl_upto(short allsl);
G__value G__pointer2func(G__value* obj_p2f,char *parameter0,char *parameter1,int *known3);
char *G__search_func(const char *funcname,G__value *buf);
char *G__search_next_member(const char *text,int state);
int G__Loffsetof(const char *tagname,const char *memname);
long *G__typeid(const char *typenamein);
void G__getcomment(char *buf,struct G__comment_info *pcomment,int tagnum);
void G__getcommenttypedef(char *buf,struct G__comment_info *pcomment,int typenum);
long G__get_classinfo(const char *item,int tagnum);
long G__get_variableinfo(const char *item,long *phandle,long *pindex,int tagnum);
long G__get_functioninfo(const char *item,long *phandle,long *pindex,int tagnum);
int G__get_envtagnum(void);
int G__isenclosingclass(int enclosingtagnum,int env_tagnum);
int G__isenclosingclassbase(int enclosingtagnum,int env_tagnum);
const char* G__find_first_scope_operator(const char* name);
const char* G__find_last_scope_operator(const char* name);
int G__checkset_charlist(char *tname,struct G__Charlist *pcall_para,int narg,int ftype);
int G__class_autoloading(int* tagnum);
void G__define_struct(char type);
G__value G__classassign(long pdest,int tagnum,G__value result);
char *G__catparam(struct G__param *libp,int catn,const char *connect);
#ifdef __cplusplus
} // extern "C"
void G__make_ifunctable(G__FastAllocString &funcheader);
G__FastAllocString &G__charaddquote(G__FastAllocString &string,char c);
int G__readline_FastAlloc(FILE* fp, G__FastAllocString& line, G__FastAllocString& argbuf,
                          int* argn, char* arg[]);
int G__separate_parameter(const char *original,int *pos,G__FastAllocString& param);
int G__cppif_returntype(FILE *fp,int ifn,struct G__ifunc_table_internal *ifunc,G__FastAllocString& endoffunc);
char *G__string(G__value buf, G__FastAllocString& temp);
char *G__add_quotation(const char* string,G__FastAllocString& temp);
int G__cattemplatearg(G__FastAllocString& tagname,struct G__Charlist *charlist);
int G__fgetname_template(G__FastAllocString& string, size_t offset, const char *endmark);
int G__fgetstream_newtemplate(G__FastAllocString& string, size_t offset, const char *endmark);
int G__fgetstream_template(G__FastAllocString& string, size_t offset, const char *endmark);
int G__fgetstream_spaces(G__FastAllocString& string, size_t offset, const char *endmark);
int G__getstream_template(const char *source,int *isrc,G__FastAllocString& string, size_t offset, const char *endmark);
char* G__rename_templatefunc(G__FastAllocString& funcname);
int G__templatesubstitute(G__FastAllocString& symbol,struct G__Charlist *callpara,struct G__Templatearg *defpara,const char *templatename,const char *tagname,int c,int npara,int isnew);
char *G__valuemonitor(G__value buf,G__FastAllocString& temp);
int G__execpragma(const char *comname,char *args);
void G__freepragma(G__AppPragma *paddpragma);
extern "C" {
#endif
void G__IntList_init(struct G__IntList *body,long iin,struct G__IntList *prev);
struct G__IntList* G__IntList_new(long iin,struct G__IntList *prev);
void G__IntList_add(struct G__IntList *body,long iin);
void G__IntList_addunique(struct G__IntList *body,long iin);
void G__IntList_delete(struct G__IntList *body);
struct G__IntList* G__IntList_find(struct G__IntList *body,long iin);
void G__IntList_free(struct G__IntList *body);
struct G__Templatearg *G__read_formal_templatearg(void);
int G__createtemplatememfunc(const char *new_name);
int G__createtemplateclass(const char *new_name,struct G__Templatearg *targ,int isforwarddecl);
struct G__Definetemplatefunc *G__defined_templatefunc(const char *name);
struct G__Definetemplatefunc *G__defined_templatememfunc(const char *name);
void G__declare_template(void);
int G__gettemplatearglist(const char *paralist,struct G__Charlist *charlist,struct G__Templatearg *def_para,int *pnpara,int parent_tagnum);
int G__instantiate_templateclass(const char *tagname,int noerror);
void G__replacetemplate(const char* templatename,const char *tagname,struct G__Charlist *callpara,FILE *def_fp,int line,int filenum,fpos_t *pdef_pos,struct G__Templatearg *def_para,int isclasstemplate,int npara,int parent_tagnum);
void G__freedeftemplateclass(struct G__Definedtemplateclass *deftmpclass);
void G__freetemplatememfunc(struct G__Definedtemplatememfunc *memfunctmplt);
char *G__gettemplatearg(int n,struct G__Templatearg *def_para);
void G__freetemplatearg(struct G__Templatearg *def_para);
void G__freetemplatefunc(struct G__Definetemplatefunc *deftmpfunc);
struct G__funclist* G__add_templatefunc(const char *funcnamein,struct G__param *libp,int hash,struct G__funclist *funclist,struct G__ifunc_table_internal *p_ifunc,int isrecursive);
struct G__funclist* G__funclist_add(struct G__funclist *last,struct G__ifunc_table_internal *ifunc,int ifn,int rate);
void G__funclist_delete(struct G__funclist *body);
int G__templatefunc(G__value *result,const char *funcname,struct G__param *libp,int hash,int funcmatch);
int G__matchtemplatefunc(struct G__Definetemplatefunc *deftmpfunc,struct G__param *libp,struct G__Charlist *pcall_para,int funcmatch);
int G__createtemplatefunc(char *funcname,struct G__Templatearg *targ,int line_number,fpos_t *ppos);
void G__define_type(void);
const char *G__access2string(int caccess);
const char *G__tagtype2string(int tagtype);
const char *G__fulltypename(int typenum);
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
G__value G__getvariable(char *item,int *known2,struct G__var_array *varglobal,struct G__var_array *varlocal);
void G__letstruct(G__value *result,int p_inc,struct G__var_array *var,int ig15,const char *item,int paran,long G__struct_offset);
void G__letstructp(G__value result,long G__struct_offset,int ig15,int p_inc,struct G__var_array *var,int paran,const char *item,G__value *para,int pp_inc);
void G__returnvartype(G__value* presult,struct G__var_array *var,int ig15,int paran);
struct G__var_array *G__getvarentry(const char *varname,int varhash,int *pi,struct G__var_array *varglobal,struct G__var_array *varlocal);
int G__getthis(G__value *result7,const char *varname,const char *item);
void G__letpointer2memfunc(struct G__var_array *var,int paran,int ig15,const char *item,int p_inc,G__value *presult,long G__struct_offset);
void G__letautomatic(struct G__var_array *var,int ig15,long G__struct_offset,int p_inc,G__value result);
void G__display_classkeyword(FILE *fout,const char *classnamein,const char *keyword,int base);
#ifdef G__FRIEND
int G__isfriend(int tagnum);
#endif
void G__set_c_environment(void);
void G__specify_link(int link_stub);

long G__new_ClassInfo(const char *classname);
long G__get_ClassInfo(int item,long tagnum,const char *buf);
long G__get_BaseClassInfo(int item,long tagnum,long basep,const char *buf);
long G__get_DataMemberInfo(int item,long tagnum,long *handle,long *index,const char *buf);
long G__get_MethodInfo(int item,long tagnum,long *handle,long *index,const char *buf);
long G__get_MethodArgInfo(int item,long tagnum,long handle,long index,long *argn,const char *buf);

#ifdef G__SECURITY
G__UINT32 G__getsecuritycode(const char *string);
#endif
void G__cpp_setupG__stream(void);
void G__cpp_setupG__API(void);
void G__c_setupG__stdstrct(void);
int G__setautoccnames(void);
int G__appendautocc(FILE *fp);
int G__isautoccupdate(void);

int G__free_exceptionbuffer(void);

#ifdef __cplusplus
} // extern "C"

int G__exec_catch(G__FastAllocString& statement);

extern "C" {
#endif

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

extern int G__gen_linksystem(const char* headerfile);

extern void G__smart_shl_unload(int allsl);
extern void G__smart_unload(int ifn);

extern struct G__dictposition* G__get_dictpos(char* fname);

void G__specify_extra_include(void);
void G__gen_extra_include(void);

struct G__ConstStringList* G__AddConstStringList(struct G__ConstStringList* current,char* str,int islen);
void G__DeleteConstStringList(struct G__ConstStringList* current);

int G__ReadInputMode(void);

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
void* G__FindSymbol(struct G__ifunc_table_internal *ifunc,int ifn);
void* G__GetShlHandle();
int G__GetShlFilenum();

int G__loadfile_tmpfile(FILE *fp);

int G__callfunc0(G__value *result,struct G__ifunc_table *ifunc,int ifn,struct G__param* libp,void* p,int funcmatch);
int G__calldtor(void* p,int tagnum,int isheap);

void G__init_replacesymbol();
void G__add_replacesymbol(const char* s1,const char* s2);
const char* G__replacesymbol(const char* s);
int G__display_replacesymbol(FILE *fout,const char* name);

void G__asm_storebytecodefunc(struct G__ifunc_table_internal *ifunc,int ifn,struct G__var_array *var,G__value *pstack,int sp,long *pinst,int instsize);

void G__push_autoobjectstack(void *p,int tagnum,int num
			           ,int scopelevel,int isheap);
void G__delete_autoobjectstack(int scopelevel);

int G__LD_IFUNC_optimize(struct G__ifunc_table_internal* ifunc,int ifn ,long *inst,int pc);

int G__bc_compile_function(struct G__ifunc_table_internal *ifunc,int iexist);
int G__bc_objassignment(G__value *plresult ,G__value *prresult);

int G__bc_exec_virtual_bytecode(G__value *result7
                        ,char *funcname        /* vtagnum */
                        ,struct G__param *libp
                        ,int hash              /* vtblindex */
			);
int G__bc_exec_normal_bytecode(G__value *result7
			,char *funcname        /* ifunc */
			,struct G__param *libp
			,int hash              /* ifn */
			);
int G__bc_exec_ctor_bytecode(G__value *result7
			,char *funcname        /* ifunc */
			,struct G__param *libp
			,int hash              /* ifn */
			);
int G__bc_exec_ctorary_bytecode(G__value *result7
			,char *funcname        /* ifunc */
			,struct G__param *libp
			,int hash              /* ifn, n */
			);
int G__bc_exec_dtorary_bytecode(G__value *result7
			,char *funcname        /* ifunc */
			,struct G__param *libp
			,int hash              /* ifn, n */
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

void G__argtype2param(const char *argtype,struct G__param *libp, int noerror, int* error);

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

int G__register_sharedlib(const char *libname);
int G__unregister_sharedlib(const char *libname);
void *G__RegisterLibrary (void (*func) ());
void *G__UnregisterLibrary (void (*func) ());
  
#ifdef __cplusplus
} // extern "C"

G__value G__getstructmem(int store_var_type,G__FastAllocString& varname,char *membername,int memnamesize,char *tagname,int *known2,struct G__var_array *varglobal,int objptr);
G__value G__letvariable(G__FastAllocString &item,G__value expression,struct G__var_array *varglobal,struct G__var_array *varlocal);
G__value G__letvariable(G__FastAllocString &item,G__value expression,struct G__var_array *varglobal,struct G__var_array *varlocal,Cint::G__DataMemberHandle &member);
G__value G__letstructmem(int store_var_type,G__FastAllocString& varname,int membernameoffset,
                         G__FastAllocString& result7,char* tagname,
                         struct G__var_array *varglobal,G__value expression,int objptr,
                         Cint::G__DataMemberHandle &member);

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
