/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file fproto.h
 ************************************************************************
 * Description:
 *  K&R style function prototype
 ************************************************************************
 * Copyright(c) 1995~2004  Masaharu Goto 
 *
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/

#ifndef G__FPROTO_H
#define G__FPROTO_H


#ifndef __CINT__

#ifdef __cplusplus

extern "C" {

#endif

#if 0
/* Just for experimenting Windows Server OS tmpfile patch */
FILE* G__dmy_tmpfile() ;
#define tmpfile G__dmy_tmpfile
#endif


/* G__cfunc.c */
extern int G__compiled_func G__P((G__value *result7,char *funcname,struct G__param *libp,int hash));
extern void G__list_sut G__P((FILE *fp));

/* G__setup.c */
extern int G__dll_revision G__P((void));
extern int G__globalsetup G__P((void));
extern int G__revision G__P((FILE *fp));
extern int G__list_global G__P((FILE *fp));
extern int G__list_struct G__P((FILE *fp));

/* G__setup.c */
extern int G__list_stub G__P((FILE *fp));

/* in src/xxx.c */
/* struct G__var_array *G__rawvarentry G__P((char *name,int hash,int *ig15,struct G__var_array *memvar)); */
/* int G__split G__P((char *line,char *string,int *argc,char **argv)); */
int G__readsimpleline G__P((FILE *fp,char *line));
#ifdef G__OLDIMPLEMENTATION451
int G__readline G__P((FILE *fp,char *line,char *argbuf,int *argn,char **arg));
#endif
int G__cmparray G__P((short array1[],short array2[],int num,short mask));
void G__setarray G__P((short array[],int num,short mask,char *mode));
int G__graph G__P((double *xdata,double *ydata,int ndata,char *title,int mode));
int G__storeobject G__P((G__value *buf1,G__value *buf2));
int G__scanobject G__P((G__value *buf1));
int G__dumpobject G__P((char *file,void *buf,int size));
int G__loadobject G__P((char *file,void *buf,int size));
long G__what_type G__P((char *name,char *type,char *tagname,char *typenamein));
int G__textprocessing G__P((FILE *fp));
#ifdef G__REGEXP
int G__matchregex G__P((char *pattern,char *string));
#endif
#ifdef G__REGEXP1
int G__matchregex G__P((char *pattern,char *string));
#endif
#ifndef G__OLDIMPLEMENTATION1397
void G__castclass G__P((G__value *result3,int tagnum,int castflag,int *ptype,int reftype));
#else
void G__castclass G__P((G__value *result3,int tagnum,int castflag,int *ptype));
#endif
G__value G__castvalue G__P((char *casttype,G__value result3));
void G__asm_cast G__P((int type,G__value *buf,int tagnum,int reftype));
  /* void G__setdebugcond G__P((void)); */
int G__findposition G__P((char *string,struct G__input_file view,int *pline,int *pfnum));
int G__findfuncposition G__P((char *func,int *pline,int *pfnum));
int G__beforelargestep G__P((char *statement,int *piout,int *plargestep));
void G__afterlargestep G__P((int *plargestep));
void G__EOFfgetc G__P((void));
void G__BREAKfgetc G__P((void));
void G__DISPNfgetc G__P((void));
void G__DISPfgetc G__P((int c));
void G__lockedvariable G__P((char *item));
int G__lock_variable G__P((char *varname));
int G__unlock_variable G__P((char *varname));
G__value G__interactivereturn G__P((void));
void G__set_tracemode G__P((char *name));
void G__del_tracemode G__P((char *name));
void G__set_classbreak G__P((char *name));
void G__del_classbreak G__P((char *name));
void G__setclassdebugcond G__P((int tagnum,int brkflag));
int G__get_newname G__P((char *new_name));
int G__unsignedintegral G__P((int *pspaceflag,int *piout,int mparen));
void G__define_var G__P((int tagnum,int typenum));
int G__initary G__P((char *new_name));
struct G__var_array* G__initmemvar G__P((int tagnum,int* pindex,G__value *pbuf));
struct G__var_array* G__incmemvar G__P((struct G__var_array* memvar,int* pindex,G__value *pbuf));
int G__initstruct G__P((char *new_name));
int G__ignoreinit G__P((char *new_name));
int G__listfunc G__P((FILE *fp,int access,char* fname,struct G__ifunc_table *ifunc));
int G__showstack G__P((FILE *fout));
void G__display_note G__P((void));
int G__display_proto G__P((FILE *fout,char *string));
int G__display_newtypes G__P((FILE *fout,char *string));
int G__display_class G__P((FILE *fout,char *name,int base,int start));
int G__display_typedef G__P((FILE *fout,char *name,int startin));
int G__display_template G__P((FILE *fout,char *name));
int G__display_includepath G__P((FILE *fout));
int G__display_macro G__P((FILE *fout,char *name));
int G__display_string G__P((FILE *fout));
int G__display_files G__P((FILE *fout));
int G__pr G__P((FILE *fout,struct G__input_file view));
int G__dump_tracecoverage G__P((FILE *fout));
#ifndef G__OLDIMPLEMENTATION444
int G__objectmonitor G__P((FILE *fout,long pobject,int tagnum,char *addspace));
#endif
int G__varmonitor G__P((FILE *fout,struct G__var_array *var,char *index,char *addspace,long offset));
int G__pushdumpinput G__P((FILE *fp,int exflag));
int G__popdumpinput G__P((void));
int G__dumpinput G__P((char *line));
char *G__xdumpinput G__P((char *prompt));
  /* void G__scratch_all G__P((void)); */
int G__free_ifunc_table G__P((struct G__ifunc_table *ifunc));
int G__free_member_table G__P((struct G__var_array *mem));
int G__free_ipath G__P((struct G__includepath *ipath));
int G__isfilebusy G__P((int ifn));
void G__free_preprocessfilekey G__P((struct G__Preprocessfilekey *pkey));
void G__store_dictposition G__P((struct G__dictposition* dictpos));
void G__scratch_upto G__P((struct G__dictposition *dictpos));
#ifndef G__FONS21
void G__scratch_globals_upto G__P((struct G__dictposition *dictpos));
#endif
int G__free_ifunc_table_upto G__P((struct G__ifunc_table *ifunc,struct G__ifunc_table *dictpos,int ifn));
int G__free_string_upto G__P((struct G__ConstStringList *conststringpos));
int G__free_typedef_upto G__P((int typenum));
int G__free_struct_upto G__P((int tagnum));
int G__destroy_upto G__P((struct G__var_array *var,int global,struct G__var_array *dictpos,int ig15));
void G__close_inputfiles_upto G__P((int nfile));
void G__destroy G__P((struct G__var_array *var,int global));
void G__exit G__P((int rtn));
int G__call_atexit G__P((void));
int G__close_inputfiles G__P((void));
int G__interpretexit G__P((void));
void G__nosupport G__P((char *name));
void G__malloc_error G__P((char *varname));
void G__arrayindexerror G__P((int ig15,struct G__var_array *var,char *item,int p_inc));
int G__asm_execerr G__P((char *message,int num));
int G__assign_error G__P((char *item,G__value *pbuf));
int G__reference_error G__P((char *item));
int G__warnundefined G__P((char *item));
int G__unexpectedEOF G__P((char *message));
int G__shl_load G__P((char *shlfile));
int G__shl_load_error G__P((char *shlname,char *message));
int G__getvariable_error G__P((char *item));
int G__referencetypeerror G__P((char *new_name));
int G__err_pointer2pointer G__P((char *item));
int G__syntaxerror G__P((char *expr));
void G__setDLLflag G__P((int flag));
void G__setInitFunc G__P((char *initfunc));
int G__assignmenterror G__P((char *item));
int G__parenthesiserror G__P((char *expression,char *funcname));
int G__commenterror G__P((void));
int G__changeconsterror G__P((char *item,char *categ));
  /* int G__printlinenum G__P((void)); */
int G__autocc G__P((void));
int G__init_readline G__P((void));
int G__using_namespace G__P((void));

#ifdef G__FRIEND
int G__parse_friend G__P((int *piout,int *pspaceflag,int mparen));
#else
int G__friendignore G__P((int *piout,int *pspaceflag,int mparen));
#endif
int G__externignore G__P((int *piout,int *pspaceflag,int mparen));
int G__handleEOF G__P((char *statement,int mparen,int single_quote,int double_quote));
void G__printerror G__P((char *funcname,int ipara,int paran));
int G__pounderror G__P((void));
int G__missingsemicolumn G__P((char *item));
G__value G__calc_internal G__P((char *exprwithspace));
G__value G__getexpr G__P((char *expression));
G__value G__getprod G__P((char *expression1));
G__value G__getpower G__P((char *expression2));
G__value G__getitem G__P((char *item));
int G__getoperator G__P((int newoperator,int oldoperator));
int G__testandor G__P((int lresult,char *rexpression,int operator2));
int G__test G__P((char *expression2));
int G__btest G__P((int operator2,G__value lresult,G__value rresult));
int G__fgetspace G__P((void));
int G__fgetvarname G__P((char *string,char *endmark));
int G__fgetname G__P((char *string,char *endmark));
int G__getname G__P((char* source,int* isrc,char *string,char *endmark));
int G__fdumpstream G__P((char *string,char *endmark));
int G__fgetstream G__P((char *string,char *endmark));
int G__fignorestream G__P((char *endmark));
int G__ignorestream G__P((char *string,int* isrc,char *endmark));
int G__fgetstream_new G__P((char *string,char *endmark));
void G__fignoreline G__P((void));
void G__fsetcomment G__P((struct G__comment_info *pcomment));
int G__fgetc G__P((void));
long G__op1_operator_detail G__P((int opr,G__value *val));
long G__op2_operator_detail G__P((int opr,G__value *lval,G__value *rval));
int G__explicit_fundamental_typeconv G__P((char* funcname,int hash,struct G__param *libp,G__value *presult3));
G__value G__getfunction G__P((char *item,int *known3,int memfunc_flag));
int G__special_func G__P((G__value *result7,char *funcname,struct G__param *libp,int hash));
int G__library_func G__P((G__value *result7,char *funcname,struct G__param *libp,int hash));
char *G__charformatter G__P((int ifmt,struct G__param *libp,char *result));
int G__istypename G__P((char *temp));
#ifndef G__OLDIMPLEMENTATION1543
char* G__savestring G__P((char** pbuf,char* name));
#endif
void G__make_ifunctable G__P((char *funcheader));
int G__readansiproto G__P((struct G__ifunc_table *ifunc,int func_now));
int G__interpret_func G__P((G__value *result7,char *funcname,struct G__param *libp,int hash,struct G__ifunc_table *p_ifunc,int funcmatch,int memfunc_flag));
struct G__ifunc_table *G__ifunc_exist G__P((struct G__ifunc_table *ifunc_now,int allifunc,struct G__ifunc_table *ifunc,int *piexist,int mask));
struct G__ifunc_table *G__ifunc_ambiguous G__P((struct G__ifunc_table *ifunc_now,int allifunc,struct G__ifunc_table *ifunc,int *piexist,int derivedtagnum));
void G__inheritclass G__P((int to_tagnum,int from_tagnum,char baseaccess));
int G__baseconstructorwp G__P((void));
#ifndef G__OLDIMPLEMENTATION1870
int G__baseconstructor G__P((int n,struct G__baseparam *pbaseparam));
#else
int G__baseconstructor G__P((int n,struct G__baseparam *pbaseparam));
#endif
int G__basedestructor G__P((void));
int G__basedestructrc G__P((struct G__var_array *mem));
#ifdef G__VIRTUALBASE
int G__publicinheritance G__P((G__value *val1,G__value *val2));
int G__ispublicbase G__P((int basetagnum,int derivedtagnum,long pobject));
int G__isanybase G__P((int basetagnum,int derivedtagnum,long pobject));
long G__getvirtualbaseoffset G__P((long pobject,int tagnum,struct G__inheritance *baseclass,int basen));
#else
int G__ispublicbase G__P((int basetagnum,int derivedtagnum));
int G__isanybase G__P((int basetagnum,int derivedtagnum));
#endif
int G__find_virtualoffset G__P((int virtualtag));
  /* int G__main G__P((int argc,char **argv)); */
int G__init_globals G__P((void));
void G__set_stdio G__P((void));
int G__cintrevision G__P((FILE *fp));
  /* char *G__input G__P((char *prompt)); */
char *G__strrstr G__P((char *string1,char *string2));
void G__set_sym_underscore G__P((int x));

  /* void G__breakkey G__P((int signame)); */
void G__floatexception G__P((int signame));
void G__segmentviolation G__P((int signame));
void G__outofmemory G__P((int signame));
void G__buserror G__P((int signame));
#ifndef G__OLDIMPLEMENTATION1946
void G__errorexit G__P((int signame));
#endif

void G__killproc G__P((int signame));
void G__timeout G__P((int signame));

void G__fsigabrt G__P((void));
void G__fsigfpe G__P((void));
void G__fsigill G__P((void));
void G__fsigint G__P((void));
void G__fsigsegv G__P((void));
void G__fsigterm G__P((void));
#ifdef SIGHUP
void G__fsighup G__P((void));
#endif
#ifdef SIGQUIT
void G__fsigquit G__P((void));
#endif
#ifdef SIGTSTP
void G__fsigtstp G__P((void));
#endif
#ifdef SIGTTIN
void G__fsigttin G__P((void));
#endif
#ifdef SIGTTOU
void G__fsigttou G__P((void));
#endif
#ifdef SIGALRM
void G__fsigalrm G__P((void));
#endif
#ifdef SIGUSR1
void G__fsigusr1 G__P((void));
#endif
#ifdef SIGUSR2
void G__fsigusr2 G__P((void));
#endif

int G__errorprompt G__P((char *nameoferror));
int G__call_interruptfunc G__P((char *func));

int G__include_file G__P((void));
int G__getcintsysdir G__P((void));
int G__preprocessor G__P((char *outname,char *inname,int cppflag,char *macros,char *undeflist,char *ppopt,char *includepath));
int G__difffile G__P((char *file1,char *file2));
int G__copyfile G__P((FILE *to,FILE *from));

#ifndef G__OLDIMPLEMENTATION1196
int G__matchfilename G__P((int i1,char* filename));
#endif
char* G__stripfilename G__P((char* filename));
#ifndef G__OLDIMPLEMENTATION1197
int G__cleardictfile G__P((int flag));
#endif

void G__openmfp G__P((void));
int G__closemfp G__P((void));
void G__define G__P((void));
int G__handle_as_typedef G__P((char *oldtype,char *newtype));
#ifndef G__OLDIMPLEMENTATION1062
void G__createmacro G__P((char *new_name,char *initvalue,int nowrapper));
#else
void G__createmacro G__P((char *new_name,char *initvalue));
#endif
G__value G__execfuncmacro G__P((char *item,int *done));
#ifndef G__OLDIMPLEMENTATION942
int G__transfuncmacro G__P((char *item,struct G__Deffuncmacro *deffuncmacro,struct G__Callfuncmacro *callfuncmacro,fpos_t call_pos,char *p,int nobraces,int nosemic));
int G__replacefuncmacro G__P((char *item,struct G__Callfuncmacro *callfuncmacro,struct G__Charlist *callpara,struct G__Charlist *defpara,FILE *def_fp,fpos_t def_pos,int nobraces,int nosemic));
int G__execfuncmacro_noexec G__P((char* macroname));
int G__maybe_finish_macro G__P((void));
#else
int G__transfuncmacro G__P((char *item,struct G__Deffuncmacro *deffuncmacro,struct G__Callfuncmacro *callfuncmacro,fpos_t call_pos,char *p));
int G__replacefuncmacro G__P((char *item,struct G__Callfuncmacro *callfuncmacro,struct G__Charlist *callpara,struct G__Charlist *defpara,FILE *def_fp,fpos_t def_pos));
#endif
#ifndef G__OLDIMPLEMENTATION1062
int G__execvarmacro_noexec G__P((char* macroname));
#endif
int G__argsubstitute G__P((char *symbol,struct G__Charlist *callpara,struct G__Charlist *defpara));
int G__createfuncmacro G__P((char *new_name));
int G__getparameterlist G__P((char *paralist,struct G__Charlist *charlist));
int G__freedeffuncmacro G__P((struct G__Deffuncmacro *deffuncmacro));
int G__freecallfuncmacro G__P((struct G__Callfuncmacro *callfuncmacro));
int G__freecharlist G__P((struct G__Charlist *charlist));
long G__malloc G__P((int n,int bsize,char *item));
void *G__TEST_Malloc G__P((size_t size));
void *G__TEST_Calloc G__P((size_t n,size_t bsize));
void G__TEST_Free G__P((void *p));
void *G__TEST_Realloc G__P((void *p,size_t size));
int G__memanalysis G__P((void));
int G__memresult G__P((void));
void G__DUMMY_Free G__P((void *p));
void *G__TEST_fopen G__P((char *fname,char *mode));
int G__TEST_fclose G__P((FILE *p));
G__value G__new_operator G__P((char *expression));
int G__getarrayindex G__P((char *indexlist));
void G__delete_operator G__P((char *expression,int isarray));
int G__alloc_newarraylist G__P((long point,int pinc));
int G__free_newarraylist G__P((long point));
int G__handle_delete G__P((int *piout,char *statement));
int G__call_cppfunc G__P((G__value *result7,struct G__param *libp,struct G__ifunc_table *ifunc,int ifn));
void G__gen_cppheader G__P((char *headerfile));
void G__gen_clink G__P((void));
void G__gen_cpplink G__P((void));
void G__clink_header G__P((FILE *fp));
void G__cpplink_header G__P((FILE *fp));
void G__cpplink_linked_taginfo G__P((FILE* fp,FILE* hfp));
#ifdef G__OLDIMPLEMENTATION408
int G__get_linked_tagnum G__P((G__linked_taginfo *p));
#endif
char *G__get_link_tagname G__P((int tagnum));
/* char *G__map_cpp_name G__P((char *in)); */
char *G__map_cpp_funcname G__P((int tagnum,char *funcname,int ifn,int page));
void G__set_globalcomp G__P((char *mode,char *linkfilename,char* dllid));
#ifdef G__OLDIMPLEMENTATION408
void G__add_compiledheader G__P((char *headerfile));
void G__add_macro G__P((char *macro));
void G__add_ipath G__P((char *path));
#endif
int G__ishidingfunc G__P((struct G__ifunc_table *fentry,struct G__ifunc_table *fthis,int ifn));
void G__cppif_memfunc G__P((FILE *fp,FILE *hfp));
void G__cppif_func G__P((FILE *fp,FILE *hfp));
void G__cppif_dummyfuncname G__P((FILE *fp));
void G__cppif_genconstructor G__P((FILE *fp,FILE *hfp,int tagnum,int ifn,struct G__ifunc_table *ifunc));
int G__isprivateconstructor G__P((int tagnum,int iscopy));
void G__cppif_gendefault G__P((FILE *fp,FILE *hfp,int tagnum,int ifn,struct G__ifunc_table *ifunc,int isconstructor,int iscopyconstructor,int isdestructor,int isassignmentoperator,int isnonpublicnew));
void G__cppif_genfunc G__P((FILE *fp,FILE *hfp,int tagnum,int ifn,struct G__ifunc_table *ifunc));
int G__cppif_returntype G__P((FILE *fp,int ifn,struct G__ifunc_table *ifunc,char *endoffunc));
void G__cppif_paratype G__P((FILE *fp,int ifn,struct G__ifunc_table *ifunc,int k));
void G__cpplink_tagtable G__P((FILE *pfp,FILE *hfp));
#ifdef G__VIRTUALBASE
int G__iosrdstate G__P((G__value *pios));
void G__cppif_inheritance G__P((FILE *pfp));
#endif
void G__cpplink_inheritance G__P((FILE *pfp));
void G__cpplink_typetable G__P((FILE *pfp,FILE *hfp));
void G__cpplink_memvar G__P((FILE *pfp));
void G__cpplink_memfunc G__P((FILE *pfp));
void G__cpplink_global G__P((FILE *pfp));
void G__cpplink_func G__P((FILE *pfp));
void G__incsetup_memvar G__P((int tagnum));
void G__incsetup_memfunc G__P((int tagnum));
#ifdef G__OLDIMPLEMENTATION408
int G__tagtable_setup G__P((char *name,int type,int size,int cpplink,int isabstract,char *comment));
#endif
#ifdef G__OLDIMPLEMENTATION408
int G__inheritance_setup G__P((int tagnum,int basetagnum,long baseoffset,int baseaccess,int directinherit));
#endif

#ifdef G__OLDIMPLEMENTATION408
int G__tag_memvar_setup G__P((int tagnum));
int G__memvar_setup G__P((void *p,int type,int constvar,int tagnum,int typenum,int statictype,int accessin,int isinherit,char *expr,int definemacro,char *comment));
int G__tag_memvar_reset G__P((void));
int G__tag_memfunc_setup G__P((int tagnum));
#ifdef G__TRUEP2F
int G__memfunc_setup G__P((char *funcname,int hash,int (*funcp)(),int type,int tagnum,int typenum,int reftype,int para_nu,int ansi,int accessin,int isconst,char *paras,char *comment,void* truep2f));
#else
int G__memfunc_setup G__P((char *funcname,int hash,int (*funcp)(),int type,int tagnum,int typenum,int reftype,int para_nu,int ansi,int accessin,int isconst,char *paras,char *comment));
#endif
int G__memfunc_para_setup G__P((int ifn,int type,int tagnum,int typenum,int reftype,G__value *para_default,char *para_def,char *para_name));
int G__memfunc_next G__P((void));
int G__tag_memfunc_reset G__P((void));
#endif

int G__separate_parameter G__P((char *original,int *pos,char *param));
int G__parse_parameter_link G__P((char *paras));
int G__cppif_p2memfunc G__P((FILE *fp));
int G__set_sizep2memfunc G__P((FILE *fp));
int G__getcommentstring G__P((char *buf,int tagnum,struct G__comment_info *pcomment));
void G__bstore G__P((int operatorin,G__value expressionin,G__value *defined));
void G__doubleassignbyref G__P((G__value *defined,double val));
void G__intassignbyref G__P((G__value *defined,long val));
int G__scopeoperator G__P((char *name,int *phash,long *pstruct_offset,int *ptagnum));
int G__label_access_scope G__P((char *statement,int *piout,int *pspaceflag,int *pmparen));
int G__cmp G__P((G__value buf1,G__value buf2));
int G__getunaryop G__P((char unaryop,char *expression,char *buf,G__value *preg));
int G__overloadopr G__P((int operatorin,G__value expressionin,G__value *defined));
#ifndef G__OLDIMPLEMENTATION1871
int G__parenthesisovldobj G__P((G__value *result3,G__value *result,char *realname,struct G__param *libp,int flag));
#endif
int G__parenthesisovld G__P((G__value *result3,char *funcname,struct G__param *libp,int flag));
int G__tryindexopr G__P((G__value *result7,G__value *para,int paran,int ig25));
int G__exec_delete G__P((char *statement,int *piout,int *pspaceflag,int isarray,int mparen));
int G__exec_function G__P((char *statement,int *pc,int *piout,int *plargestep,G__value *presult));
int G__keyword_anytime_5 G__P((char *statement));
int G__keyword_anytime_6 G__P((char *statement));
int G__keyword_anytime_7 G__P((char *statement));
int G__keyword_exec_6 G__P((char *statement,int *piout,int *pspaceflag,int mparen));
int G__setline G__P((char *statement,int c,int *piout));
int G__skip_comment G__P((void));
int G__pp_command G__P((void));
void G__pp_skip G__P((int elifskip));
int G__pp_if G__P((void));
int G__defined_macro G__P((char *macro));
int G__pp_ifdef G__P((int def));
void G__pp_undef G__P((void));
G__value G__exec_do G__P((void));
G__value G__return_value G__P((char *statement));
void G__free_tempobject G__P((void));
G__value G__alloc_tempstring G__P((char *string));
#ifdef G__OLDIMPLEMENTATION408
void G__store_tempobject G__P((G__value reg));
#endif
int G__pop_tempobject G__P((void));
G__value G__exec_switch G__P((void));
G__value G__exec_if G__P((void));
G__value G__exec_loop G__P((char *forinit,char *condition,int naction,char **foraction));
G__value G__exec_while G__P((void));
G__value G__exec_for G__P((void));
G__value G__exec_else_if G__P((void));
G__value G__exec_statement G__P((void));
int G__readpointer2function G__P((char *new_name,char *pvar_type));
int G__search_gotolabel G__P((char *label,fpos_t *pfpos,int line,int *pmparen));
  /* int G__pause G__P((void)); */
int G__setaccess G__P((char *statement,int iout));
int G__class_conversion_operator G__P((int tagnum,G__value *presult,char* ttt));
int G__fundamental_conversion_operator G__P((int type,int tagnum,int typenum,int reftype,int constvar,G__value *presult,char* ttt));
#ifndef G__OLDIMPLEMENTATION2089
int G__asm_gen_stvar G__P((long G__struct_offset,int ig15,int paran,struct G__var_array *var,char *item,long store_struct_offset,int var_type,G__value *presult));
#else
int G__asm_gen_stvar G__P((long G__struct_offset,int ig15,int paran,struct G__var_array *var,char *item,long store_struct_offset,int var_type));
#endif
int G__exec_asm G__P((int start,int stack,G__value *presult,long localmem));
int G__asm_test_E G__P((int *a,int *b));
int G__asm_test_N G__P((int *a,int *b));
int G__asm_test_GE G__P((int *a,int *b));
int G__asm_test_LE G__P((int *a,int *b));
int G__asm_test_g G__P((int *a,int *b));
int G__asm_test_l G__P((int *a,int *b));
long G__asm_gettest G__P((int op,long *inst));
int G__asm_optimize G__P((int *start));
int G__asm_optimize3 G__P((int *start));
int G__inc_cp_asm G__P((int cp_inc,int dt_dec));
int G__clear_asm G__P((void));
int G__asm_clear G__P((void));
#ifndef G__OLDIMPLEMENTATION1158
void G__gen_addstros G__P((int addstros));
#endif
#ifndef G__OLDIMPLEMENTATION1164
void G__suspendbytecode G__P((void));
void G__resetbytecode G__P((void));
void G__resumebytecode G__P((int store_asm_noverflow));
#endif
void G__abortbytecode G__P((void));
int G__asm_putint G__P((int i));
G__value G__getreserved G__P((char *item,void** ptr,void** ppdict));
/*
G__value G__getreserved G__P((char *item,void *ptr));
G__value G__getreserved G__P((char *item));
*/
int G__dasm G__P((FILE *fout,int isthrow));
G__value G__getrsvd G__P((int i));
int G__read_setmode G__P((int *pmode));
int G__pragma G__P((void));
int G__execpragma G__P((char *comname,char *args));
#ifndef G__VMS
void G__freepragma G__P((struct G__AppPragma *paddpragma));
#endif
void G__free_bytecode G__P((struct G__bytecodefunc *bytecode));
void G__asm_gen_strip_quotation G__P((G__value *pval));
int G__security_handle G__P((G__UINT32 category));
void G__asm_get_strip_quotation G__P((G__value *pval));
G__value G__strip_quotation G__P((char *string));
char *G__charaddquote G__P((char *string,char c));
G__value G__strip_singlequotation G__P((char *string));
char *G__add_quotation G__P((char *string,char *temp));
char *G__tocharexpr G__P((char *result7));
char *G__string G__P((G__value buf,char *temp));
char *G__quotedstring G__P((char *buf,char *result));
char *G__logicstring G__P((G__value buf,int dig,char *result));
int G__revprint G__P((FILE *fp));
int G__dump_header G__P((char *outfile));
void G__listshlfunc G__P((FILE *fout));
void G__listshl G__P((FILE *G__temp));
int G__free_shl_upto G__P((int allsl));
G__value G__pointer2func G__P((G__value* obj_p2f,char *parameter0,char *parameter1,int *known3));
char *G__search_func G__P((char *funcname,G__value *buf));
char *G__search_next_member G__P((char *text,int state));
int G__sizeof G__P((G__value *object));
int G__Loffsetof G__P((char *tagname,char *memname));
int G__Lsizeof G__P((char *typenamein));
long *G__typeid G__P((char *typenamein));
void G__getcomment G__P((char *buf,struct G__comment_info *pcomment,int tagnum));
void G__getcommenttypedef G__P((char *buf,struct G__comment_info *pcomment,int typenum));
long G__get_classinfo G__P((char *item,int tagnum));
long G__get_variableinfo G__P((char *item,long *phandle,long *pindex,int tagnum));
long G__get_functioninfo G__P((char *item,long *phandle,long *pindex,int tagnum));
int G__get_envtagnum G__P((void));
int G__isenclosingclass G__P((int enclosingtagnum,int env_tagnum));
int G__isenclosingclassbase G__P((int enclosingtagnum,int env_tagnum));
#ifndef G__OLDIMPLEMENTATION671
char* G__find_first_scope_operator G__P((char* name));
char* G__find_last_scope_operator G__P((char* name));
#endif
#ifdef G__OLDIMPLEMENTATION408
int G__defined_tagname G__P((char *tagname,int noerror));
int G__search_tagname G__P((char *tagname,int type));
#endif
#ifndef G__OLDIMPLEMENTATION1560
int G__checkset_charlist G__P((char *tname,struct G__Charlist *pcall_para,int narg,int ftype));
#endif
void G__define_struct G__P((char type));
G__value G__classassign G__P((long pdest,int tagnum,G__value result));
int G__cattemplatearg G__P((char *tagname,struct G__Charlist *charlist));
char *G__catparam G__P((struct G__param *libp,int catn,char *connect));
int G__fgetname_template G__P((char *string,char *endmark));
int G__fgetstream_newtemplate G__P((char *string,char *endmark));
int G__fgetstream_template G__P((char *string,char *endmark));
int G__fgetstream_spaces G__P((char *string,char *endmark));
int G__getstream_template G__P((char *source,int *isrc,char *string,char *endmark));
#ifndef G__OLDIMPLEMENTATION691
void G__IntList_init G__P((struct G__IntList *body,long iin,struct G__IntList *prev));
struct G__IntList* G__IntList_new G__P((long iin,struct G__IntList *prev));
void G__IntList_add G__P((struct G__IntList *body,long iin));
void G__IntList_addunique G__P((struct G__IntList *body,long iin));
void G__IntList_delete G__P((struct G__IntList *body));
struct G__IntList* G__IntList_find G__P((struct G__IntList *body,long iin));
void G__IntList_free G__P((struct G__IntList *body));
#endif
struct G__Templatearg *G__read_formal_templatearg G__P((void));
int G__createtemplatememfunc G__P((char *new_name));
#ifndef G__OLDIMPLEMENTATION691
int G__createtemplateclass G__P((char *new_name,struct G__Templatearg *targ,int isforwarddecl));
#else
int G__createtemplateclass G__P((char *new_name,struct G__Templatearg *targ));
#endif
struct G__Definedtemplateclass *G__defined_templateclass G__P((char *name));
#ifndef G__OLDIMPLEMENTATION1560
struct G__Definetemplatefunc *G__defined_templatefunc G__P((char *name));
#endif
#ifndef G__OLDIMPLEMENTATION1611
struct G__Definetemplatefunc *G__defined_templatememfunc G__P((char *name));
#endif
void G__declare_template G__P((void));
#ifndef G__OLDIMPLEMENTATION1780
int G__gettemplatearglist G__P((char *paralist,struct G__Charlist *charlist,struct G__Templatearg *def_para,int *pnpara,int parent_tagnum));
#else
int G__gettemplatearglist G__P((char *paralist,struct G__Charlist *charlist,struct G__Templatearg *def_para,int *pnpara));
#endif
int G__instantiate_templateclass G__P((char *tagname));
void G__replacetemplate G__P((char *templatename,char *tagname,struct G__Charlist *callpara,FILE *def_fp,int line,int filenum,fpos_t *pdef_pos,struct G__Templatearg *def_para,int isclasstemplate,int npara,int parent_tagnum));
int G__templatesubstitute G__P((char *symbol,struct G__Charlist *callpara,struct G__Templatearg *defpara,char *templatename,char *tagname,int c,int npara,int isnew));
void G__freedeftemplateclass G__P((struct G__Definedtemplateclass *deftmpclass));
void G__freetemplatememfunc G__P((struct G__Definedtemplatememfunc *memfunctmplt));
char *G__gettemplatearg G__P((int n,struct G__Templatearg *def_para));
void G__freetemplatearg G__P((struct G__Templatearg *def_para));
void G__freetemplatefunc G__P((struct G__Definetemplatefunc *deftmpfunc));
#ifndef G__OLDIMPLEMENTATION1727
struct G__funclist* G__add_templatefunc G__P((char *funcnamein,struct G__param *libp,int hash,struct G__funclist *funclist,struct G__ifunc_table *p_ifunc,int isrecursive));
struct G__funclist* G__funclist_add G__P((struct G__funclist *last,struct G__ifunc_table *ifunc,int ifn,int rate));
void G__funclist_delete(struct G__funclist *body);
#endif
int G__templatefunc G__P((G__value *result,char *funcname,struct G__param *libp,int hash,int funcmatch));
int G__matchtemplatefunc G__P((struct G__Definetemplatefunc *deftmpfunc,struct G__param *libp,struct G__Charlist *pcall_para,int funcmatch));
int G__createtemplatefunc G__P((char *funcname,struct G__Templatearg *targ,int line_number,fpos_t *ppos));
void G__define_type G__P((void));
#ifdef G__OLDIMPLEMENTATION408
int G__defined_typename G__P((char *typename));
int G__search_typename G__P((char *typenamein,int typein,int tagnum,int reftype));
#endif
int G__defined_type G__P((char *typenamein,int len));
char *G__valuemonitor G__P((G__value buf,char *temp));
char *G__access2string G__P((int caccess));
char *G__tagtype2string G__P((int tagtype));
#ifndef G__OLDIMPLEMENTATION1560
char* G__rename_templatefunc G__P((char* funcname,int isrealloc));
char *G__fulltypename G__P((int typenum));
#endif
char *G__fulltagname G__P((int tagnum,int mask_dollar));
int G__val2pointer G__P((G__value *result7));
char *G__getbase G__P((unsigned int expression,int base,int digit,char *result1));
int G__getdigit G__P((unsigned int number));
G__value G__checkBase G__P((char *string,int *known4));
int G__isfloat G__P((char *string,int *type));
int G__isoperator G__P((int c));
int G__isexponent G__P((char *expression4,int lenexpr));
int G__isvalue G__P((char *temp));
#ifdef G__OLDIMPLEMENTATION408
void G__letdouble G__P((G__value *buf,int type,double value));
void G__letint G__P((G__value *buf,int type,long value));
#endif
int G__isdouble G__P((G__value buf));
/* float G__float G__P((G__value buf)); */
G__value G__tovalue G__P((G__value p));
#ifndef G__OLDIMPLEMENTATION695
G__value G__toXvalue G__P((G__value result,int var_type));
#endif
G__value G__letVvalue G__P((G__value *p,G__value result));
G__value G__letPvalue G__P((G__value *p,G__value result));
G__value G__letvalue G__P((G__value *p,G__value result));
G__value G__letvariable G__P((char *item,G__value expression,struct G__var_array *varglobal,struct G__var_array *varlocal));
G__value G__getvariable G__P((char *item,int *known2,struct G__var_array *varglobal,struct G__var_array *varlocal));
#ifndef G__OLDIMPLEMENTATION1013
G__value G__getstructmem G__P((int store_var_type,char *varname,char *membername,char *tagname,int *known2,struct G__var_array *varglobal,int objptr));
G__value G__letstructmem G__P((int store_var_type,char *varname,char *membername,char *tagname,struct G__var_array *varglobal,G__value expression,int objptr));
#else
G__value G__getstructmem G__P((int store_var_type,char *varname,char *membername,char *tagname,int *known2,struct G__var_array *varglobal));
G__value G__letstructmem G__P((int store_var_type,char *varname,char *membername,char *tagname,struct G__var_array *varglobal,G__value expression));
#endif
void G__letstruct G__P((G__value *result,int p_inc,struct G__var_array *var,int ig15,char *item,int paran,long G__struct_offset));
void G__letstructp G__P((G__value result,long G__struct_offset,int ig15,int p_inc,struct G__var_array *var,int paran,char *item,G__value *para,int pp_inc));
void G__returnvartype G__P((G__value* presult,struct G__var_array *var,int ig15,int paran));
G__value G__allocvariable G__P((G__value result,G__value para[],struct G__var_array *varglobal,struct G__var_array *varlocal,int paran,int varhash,char *item,char *varname,int parameter00));
struct G__var_array *G__getvarentry G__P((char *varname,int varhash,int *pi,struct G__var_array *varglobal,struct G__var_array *varlocal));
int G__getthis G__P((G__value *result7,char *varname,char *item));
void G__letpointer2memfunc G__P((struct G__var_array *var,int paran,int ig15,char *item,int p_inc,G__value *presult,long G__struct_offset));
void G__letautomatic G__P((struct G__var_array *var,int ig15,long G__struct_offset,int p_inc,G__value result));
void G__display_classkeyword G__P((FILE *fout,char *classnamein,char *keyword,int base));
#ifdef G__FRIEND
int G__isfriend G__P((int tagnum));
#endif
void G__set_c_environment G__P((void));
void G__specify_link G__P((int link_stub));

long G__new_ClassInfo G__P((char *classname));
long G__get_ClassInfo G__P((int item,long tagnum,char *buf));
long G__get_BaseClassInfo G__P((int item,long tagnum,long basep,char *buf));
long G__get_DataMemberInfo G__P((int item,long tagnum,long *handle,long *index,char *buf)) ;
long G__get_MethodInfo G__P((int item,long tagnum,long *handle,long *index,char *buf));
long G__get_MethodArgInfo G__P((int item,long tagnum,long handle,long index,long *argn,char *buf));

#ifdef G__SECURITY
int G__check_drange G__P((int p,double low,double up,double d,G__value *result7,char *funcname));
int G__check_lrange G__P((int p,long low,long up,long l,G__value *result7,char *funcname));
#ifndef G__OLDIMPLEMENTATION575
int G__check_type G__P((int p,int t1,int t2,G__value *para,G__value *result7,char *funcname));
int G__check_nonull G__P((int p,int t,G__value *para,G__value *result7,char *funcname));
#else
int G__check_nonull G__P((int p,long l,G__value *result7,char *funcname));
#endif
G__UINT32 G__getsecuritycode G__P((char *string));
#endif
void G__cpp_setupG__stream G__P((void));
void G__cpp_setupG__API G__P((void));
void G__c_setupG__stdstrct G__P((void));
int G__setautoccnames G__P((void));
int G__appendautocc G__P((FILE *fp));
int G__isautoccupdate G__P((void));
void G__free_friendtag G__P((struct G__friendtag *friendtag));

int G__free_exceptionbuffer G__P((void));
int G__exec_try G__P((char* statement));
int G__exec_throw G__P((char* statement));
int G__ignore_catch G__P((void));
int G__exec_catch G__P((char* statement));


void G__cppstub_memfunc G__P((FILE* fp));
void G__cppstub_func G__P((FILE* fp));
void G__set_stubflags G__P((struct G__dictposition *dictpos));

void G__set_DLLflag G__P((void));
void G__setPROJNAME G__P((char* proj));

#ifdef G__ERROR_HANDLE
void G__error_handle G__P((int signame));
#endif

#ifdef G__MULTIBYTE
int G__CodingSystem G__P((int c));
#endif

#ifndef G__OLDIMPLEMENTATION472
extern int G__SetGlobalcomp G__P((char *funcname,char *param,int globalcomp));
#endif
#ifndef G__OLDIMPLEMENTATION1781
extern int G__ForceBytecodecompilation G__P((char *funcname,char *param));
#endif

#ifndef G__OLDIMPLEMENTATION640
extern void G__more_col G__P((int len));
extern int G__more G__P((FILE* fp,char *msg));
extern int G__more_pause G__P((FILE* fp,int len));
extern void G__redirect_on G__P((void));
extern void G__redirect_off G__P((void));
#endif

#ifndef G__OLDIMPLEMENTATION842
void G__init_jumptable_bytecode G__P((void));
void G__add_label_bytecode G__P((char *label));
void G__add_jump_bytecode G__P((char *label));
void G__resolve_jumptable_bytecode G__P((void));
#endif

extern void G__LockCriticalSection G__P((void));
extern void G__UnlockCriticalSection G__P((void));

#ifndef G__OLDIMPLEMENTATION1401
extern void G__asm_tovalue_p2p G__P((G__value *result));
extern void G__asm_tovalue_p2p2p G__P((G__value *result));
extern void G__asm_tovalue_p2p2p2 G__P((G__value *result));
extern void G__asm_tovalue_LL G__P((G__value *result));
extern void G__asm_tovalue_ULL G__P((G__value *result));
extern void G__asm_tovalue_LD G__P((G__value *result));
extern void G__asm_tovalue_B G__P((G__value *result));
extern void G__asm_tovalue_C G__P((G__value *result));
extern void G__asm_tovalue_R G__P((G__value *result));
extern void G__asm_tovalue_S G__P((G__value *result));
extern void G__asm_tovalue_H G__P((G__value *result));
extern void G__asm_tovalue_I G__P((G__value *result));
extern void G__asm_tovalue_K G__P((G__value *result));
extern void G__asm_tovalue_L G__P((G__value *result));
extern void G__asm_tovalue_F G__P((G__value *result));
extern void G__asm_tovalue_D G__P((G__value *result));
extern void G__asm_tovalue_U G__P((G__value *result));
#endif

#ifdef G__EXCEPTIONWRAPPER
/*********************************************************************
* G__ExceptionWrapper
*********************************************************************/
extern int G__ExceptionWrapper G__P((G__InterfaceMethod funcp
				     ,G__value* result7
				     ,char* funcname
				     ,struct G__param *libp
				     ,int hash));
#endif

#ifndef G__OLDIMPLEMENTATION1271
extern int G__gen_linksystem G__P((char* headerfile));
#endif

#ifndef G__OLDIMPLEMENTATION1273
extern void G__smart_shl_unload G__P((int allsl));
extern void G__smart_unload G__P((int ifn));
#endif

#ifndef G__OLDIMPLEMENTATION873
extern struct G__dictposition* G__get_dictpos G__P((char* fname));
#endif

#ifndef G__PHILIPPE30
void G__specify_extra_include G__P((void)) ;
void G__gen_extra_include G__P((void)) ;
#endif

#ifndef G__OLDIMPLEMENTATION1451
struct G__ConstStringList* G__AddConstStringList G__P((struct G__ConstStringList* current,char* str,int islen));
void G__DeleteConstStringList G__P((struct G__ConstStringList* current));
#endif

#ifndef G__OLDIMPLEMENTATION1649
#endif

#ifndef G__OLDIMPLEMENTATION1795
int G__ReadInputMode G__P((void));
#endif

#ifndef G__OLDIMPLEMENTATION1825
char* G__setiparseobject G__P((G__value* result,char *str));
#endif

#ifdef G__SHMGLOBAL
void* G__shmmalloc G__P((int size));
void* G__shmcalloc G__P((int atomsize,int num));
void G__initshm G__P((void));
#endif

#ifdef G__BORLAND
int G__gettempfilenum G__P((void));
void G__redirect_on G__P((void));
void G__redirect_off G__P((void));
#endif

#ifndef G__OLDIMPLEMENTATION1836
void G__loadlonglong G__P((int* ptag,int* ptype,int which));
#endif

#ifndef G__OLDIMPLEMENTATION1908
int G__DLL_direct_globalfunc G__P((G__value *result7
				   ,G__CONST char *funcname
				   ,struct G__param *libp,int hash)) ;
void* G__SetShlHandle G__P((char* filename));
void G__ResetShlHandle G__P(());
void* G__FindSymbol G__P((struct G__ifunc_table *ifunc,int ifn));
void* G__GetShlHandle G__P(());
#ifndef G__OLDIMPLEMENTATION2012
int G__GetShlFilenum G__P(());
#endif
#endif

#ifndef G__OLDIMPLEMENTATION1919
int G__loadfile_tmpfile G__P((FILE *fp));
#endif

#ifndef G__OLDIMPLEMENTATION2030
int G__callfunc0 G__P((G__value *result,struct G__ifunc_table *ifunc,int ifn,struct G__param* libp,void* p,int funcmatch));
int G__calldtor G__P((void* p,int tagnum,int isheap));
#endif

#ifndef G__OLDIMPLEMENTATION2034
void G__init_replacesymbol G__P(());
void G__add_replacesymbol G__P((const char* s1,const char* s2));
const char* G__replacesymbol G__P((const char* s));
int G__display_replacesymbol G__P((FILE *fout,const char* name));
#endif

void G__asm_storebytecodefunc G__P((struct G__ifunc_table *ifunc,int ifn,struct G__var_array *var,G__value *pstack,int sp,long *pinst,int instsize));

#ifndef G__OLDIMPLEMENTATION2042
void G__push_autoobjectstack G__P((void *p,int tagnum,int num
			           ,int scopelevel,int isheap)) ;
void G__delete_autoobjectstack G__P((int scopelevel)) ;
#endif

#ifndef G__OLDIMPLEMENTATION2066
int G__LD_IFUNC_optimize G__P((struct G__ifunc_table* ifunc,int ifn ,long *inst,int pc));
#endif

#ifndef G__OLDIMPLEMENTATION2067
int G__bc_compile_function G__P((struct G__ifunc_table *ifunc,int iexist));
#endif
#ifndef G__OLDIMPLEMENTATION2117
int G__bc_throw_compile_error();
int G__bc_throw_runtime_error();
#endif

#ifndef G__OLDIMPLEMENTATION2074
int G__bc_exec_virtual_bytecode G__P((G__value *result7
			,char *funcname        // vtagnum
			,struct G__param *libp
			,int hash              // vtblindex
			)) ;
int G__bc_exec_normal_bytecode G__P((G__value *result7
			,char *funcname        // ifunc
			,struct G__param *libp
			,int hash              // ifn
			)) ;
int G__bc_exec_ctor_bytecode G__P((G__value *result7
			,char *funcname        // ifunc
			,struct G__param *libp
			,int hash              // ifn
			)) ;
int G__bc_exec_ctorary_bytecode G__P((G__value *result7
			,char *funcname        // ifunc
			,struct G__param *libp
			,int hash              // ifn, n
			)) ;
int G__bc_exec_dtorary_bytecode G__P((G__value *result7
			,char *funcname        // ifunc
			,struct G__param *libp
			,int hash              // ifn, n
			)) ;
#endif
#ifndef G__OLDIMPLEMENTATION2075
void G__bc_struct G__P((int tagnum)) ;
#endif

#ifndef G__OLDIMPLEMENTATION2084
void G__bc_delete_vtbl G__P((int tagnum)) ;
#endif
#ifndef G__OLDIMPLEMENTATION2162
void G__bc_disp_vtbl G__P((FILE* fp,int tagnum)) ;
#endif

#ifndef G__OLDIMPLEMENTATION2087
G__value G__bc_new_operator G__P((const char *expression)) ;
#endif
void G__bc_delete_operator G__P((const char *expression,int isarray)) ;

#ifndef G__OLDIMPLEMENTATION2109
int G__bc_exec_try_bytecode G__P((int start,int stack,G__value *presult,long localmem)) ;
int G__bc_exec_throw_bytecode G__P((G__value* pval));
int G__bc_exec_typematch_bytecode G__P((G__value* catchtype,G__value* excptobj));
int G__Isvalidassignment_val G__P((G__value* ltype,int varparan,int lparan,int lvar_type,G__value* rtype));
int G__bc_conversion G__P((G__value *result,struct G__var_array* var,int ig15
			   ,int var_type,int paran)) ;
#endif

#ifndef G__OLDIMPLEMENTATION2182
int G__bc_assignment G__P((struct G__var_array *var,int ig15,int paran
			   ,int var_type,G__value *prresult
			   ,long struct_offset,long store_struct_offset));
#endif

#ifndef G__OLDIMPLEMENTATION2136
int G__bc_setdebugview G__P((int i,struct G__view *pview));
int G__bc_showstack G__P((FILE* fp));
void G__bc_setlinenum G__P((int line));
#endif

#ifndef G__OLDIMPLEMENTATION2150
void G__bc_Baseclassctor_vbase G__P((int tagnum));
#endif

#ifndef G__OLDIMPLEMENTATION2152
void G__bc_VIRTUALADDSTROS G__P((int tagnum,struct G__inheritance* baseclas,int basen));
void G__bc_cancel_VIRTUALADDSTROS();
#endif

#ifndef G__OLDIMPLEMENTATION2154
void G__bc_REWINDSTACK G__P((int n)) ;
#endif

#ifndef G__OLDIMPLEMENTATION2160
int G__bc_casejump G__P((void* p,int val)) ;
#endif

G__value G__alloc_exceptionbuffer G__P((int tagnum));

void G__argtype2param G__P((char *argtype,struct G__param *libp));

void G__letbool G__P((G__value *buf,int type,long value));
long G__bool G__P((G__value buf));


#ifndef G__OLDIMPLEMENTATION2189
void G__letLonglong G__P((G__value* buf,int type,G__int64 value));
void G__letULonglong G__P((G__value* buf,int type,G__uint64 value));
void G__letLongdouble G__P((G__value* buf,int type,long double value));
G__int64 G__Longlong G__P((G__value buf)); /* used to be int */
G__uint64 G__ULonglong G__P((G__value buf)); /* used to be int */
long double G__Longdouble G__P((G__value buf)); /* used to be int */
#endif

#ifndef G__OLDIMPLEMENTATION2221
void G__display_purevirtualfunc G__P((int tagnum));
#endif

#ifndef G__OLDIMPLEMENTATION2226
void G__setmemtestbreak G__P((int n,int m));
#endif

#ifndef G__OLDIMPLEMENTATION2227
void G__clear_errordictpos();
#endif

#ifdef __cplusplus
}
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
