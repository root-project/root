/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file struct.c
 ************************************************************************
 * Description:
 *  Struct, class, enum, union handling
 ************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto 
 *
 * Permission to use, copy, modify and distribute this software and its 
 * documentation for any purpose is hereby granted without fee,
 * provided that the above copyright notice appear in all copies and
 * that both that copyright notice and this permission notice appear
 * in supporting documentation.  The author makes no
 * representations about the suitability of this software for any
 * purpose.  It is provided "as is" without express or implied warranty.
 ************************************************************************/

#include "common.h"


#ifndef G__OLDIMPLEMENTATION1087
/******************************************************************
* G__check_semicolumn_after_classdef
******************************************************************/
static int G__check_semicolumn_after_classdef(isclassdef) 
int isclassdef;
{
  char checkbuf[G__ONELINE];
  int store_linenum = G__ifile.line_number;
  int store_c;
  int errflag=0;
  fpos_t store_pos;
  fgetpos(G__ifile.fp,&store_pos);
  G__disp_mask=1000;
  
  store_c = G__fgetname(checkbuf,";,(");
#ifndef G__OLDIMPLEMENTATION1187
  if(isspace(store_c) && '*'!=checkbuf[0] && 0==strchr(checkbuf,'[')) {
#else
  if(isspace(store_c) && '*'!=checkbuf[0]) {  /* ')' fixed 30/Jul/1999 */
#endif
    char checkbuf2[G__ONELINE];
    store_c = G__fgetname(checkbuf2,";,(");
    if(isalnum(checkbuf2[0])) errflag=1;
  }
  
  G__disp_mask=0;
  fsetpos(G__ifile.fp,&store_pos);
  G__ifile.line_number = store_linenum;
  if(errflag || (isclassdef&&'('==store_c)) {
    G__genericerror("Error: ';' missing after class/struct/enum declaration");
    return(1);
  }
  return(0);
}
#endif

#ifndef G__OLDIMPLEMENTATION613
/******************************************************************
* int G__using_namespace()
*
*  using  namespace [ns_name];  using directive   -> inheritance
*  using  [scope]::[member];    using declaration -> reference object
*        ^
*
* Note: using directive appears in global scope is not implemented yet
******************************************************************/
int G__using_namespace()
{
  int result=0;
  char buf[G__ONELINE];
  int c;

  /* check if using directive or declaration */
  c = G__fgetname_template(buf,";");

  if(strcmp(buf,"namespace")==0) {
    /*************************************************************
    * using directive, treat as inheritance
    *************************************************************/
    int basetagnum,envtagnum;
    c=G__fgetstream_template(buf,";");
#ifndef G__STD_NAMESPACE /* ON676 */
    if(';'==c && strcmp(buf,"std")==0
#ifndef G__OLDIMPLEMENTATION1285
       && G__ignore_stdnamespace
#endif
       ) return 1;
#endif
    basetagnum = G__defined_tagname(buf,2);
#ifndef G__OLDIMPLEMENTATION1732
    if(-1==basetagnum) {
      G__fprinterr(G__serr,"Error: namespace %s is not defined",buf);
      G__genericerror((char*)NULL);
      return(0);
    }
#endif
    if(G__def_struct_member) {
      /* using directive in other namespace or class/struct */
      envtagnum=G__get_envtagnum();
      if(0<=envtagnum) {
        int* pbasen;
        struct G__inheritance *base=G__struct.baseclass[envtagnum];
        pbasen = &base->basen;
	if(*pbasen<G__MAXBASE) {
	  base->basetagnum[*pbasen]=basetagnum;
	  base->baseoffset[*pbasen]=0;
	  base->baseaccess[*pbasen]=G__PUBLIC;
#ifndef G__OLDIMPLEMENTATION1057
	  base->property[*pbasen]=0;
#endif
	  ++(*pbasen);
	}
	else {
	  G__genericerror("Limitation: too many using directives");
	}
      }
    }
    else {
      /* using directive in global scope, to be implemented 
       * 1. global scope has baseclass information
       * 2. G__searchvariable() looks for global scope baseclass
       */
#ifndef G__OLDIMPLEMENTATION2167
       /* first check whether we already have this directive in 
          memory */
       int j;
       int found;
       found = 0;
       for(j=0; j<G__globalusingnamespace.basen; ++j) {
          struct G__inheritance *base = &G__globalusingnamespace;
          if ( base->basetagnum[j] == basetagnum ) {
             found = 1;
             break;
          }
       }
       if (!found) {
#ifndef G__OLDIMPLEMENTATION686
          if(G__globalusingnamespace.basen<G__MAXBASE) {
             struct G__inheritance *base = &G__globalusingnamespace;
             int* pbasen = &base->basen;
             base->basetagnum[*pbasen]=basetagnum;
             base->baseoffset[*pbasen]=0;
             base->baseaccess[*pbasen]=G__PUBLIC;
             ++(*pbasen);
#ifdef G__OLDIMPLEMENTATION1060
             G__fprinterr(G__serr,"Warning: using directive in global scope, not completely supported");
             G__printlinenum();
#endif
          }
          else {
             G__genericerror("Limitation: too many using directives in global scope");
          }
#else
          G__genericerror("Limitation: using directive can be only used in template/class/struct scope");
#endif
       }
       result=1;
#else /* 2167 */
#ifndef G__OLDIMPLEMENTATION686
      if(G__globalusingnamespace.basen<G__MAXBASE) {
        struct G__inheritance *base = &G__globalusingnamespace;
        int* pbasen = &base->basen;
	base->basetagnum[*pbasen]=basetagnum;
	base->baseoffset[*pbasen]=0;
	base->baseaccess[*pbasen]=G__PUBLIC;
	++(*pbasen);
#ifdef G__OLDIMPLEMENTATION1060
	G__fprinterr(G__serr,"Warning: using directive in global scope, not completely supported");
	G__printlinenum();
#endif
      }
      else {
	G__genericerror("Limitation: too many using directives in global scope");
      }
#else
      G__genericerror("Limitation: using directive can be only used in template/class/struct scope");
#endif
      result=1;
#endif /* 2167 */
    }
  }

  else {
    /*************************************************************
    * using declaration, treat as reference object
    *************************************************************/
    struct G__var_array *var;
    int ig15,hash;
    long struct_offset , store_struct_offset;
    /* G__value val; */
    G__hash(buf,hash,ig15);
    var = G__searchvariable(buf,hash,G__p_local,&G__global
			    ,&struct_offset,&store_struct_offset,&ig15,0);
    if(var) {
      int store_globalvarpointer;
      /* int store_reftype; */
      struct G__var_array *avar;
      int aig15,ahash;
      long astruct_offset , astore_struct_offset;
      char varname[G__ONELINE];
      char* pc;
      pc = strrchr(buf,':');
      if(!pc) pc = buf;
      else ++pc;
      strcpy(varname,buf);

      /* allocate variable table */
      store_globalvarpointer = G__globalvarpointer;
      G__globalvarpointer = var->p[ig15];
      G__letvariable(varname,G__null,&G__global,G__p_local);
      G__globalvarpointer = store_globalvarpointer;

      /* search allocated table entry */
      G__hash(varname,ahash,aig15);
      avar = G__searchvariable(varname,ahash,G__p_local,&G__global
			      ,&astruct_offset,&astore_struct_offset,&aig15,0);

      /* copy variable information */
#ifndef G__OLDIMPLEMENTATION1569
      if(avar && ( avar!=var || aig15!= ig15 ) ) {
#else
      if(avar) {
#endif
	int ii;
#ifndef G__OLDIMPLEMENTATION1543
	G__savestring(&avar->varnamebuf[aig15],var->varnamebuf[ig15]);
#else
	strcpy(avar->varnamebuf[aig15],var->varnamebuf[ig15]);
#endif
	avar->hash[aig15]=var->hash[ig15];
	for(ii=0;ii<G__MAXVARDIM;ii++) {
	  avar->varlabel[aig15][ii]=var->varlabel[ig15][ii];
	}
	avar->paran[aig15]=var->paran[ig15];
	avar->bitfield[aig15]=var->bitfield[ig15];
	avar->type[aig15]=var->type[ig15];
	avar->constvar[aig15]=var->constvar[ig15];
	avar->p_tagtable[aig15]=var->p_tagtable[ig15];
	avar->p_typetable[aig15]=var->p_typetable[ig15];
	avar->statictype[aig15]=var->statictype[ig15];
	avar->reftype[aig15]=var->reftype[ig15];
	avar->globalcomp[aig15]=G__COMPILEDGLOBAL;
	avar->comment[aig15]=var->comment[ig15]; /* questionable */
      }
    }
#ifndef G__OLDIMPLEMENTATION1687
    else {
      int tagnum = G__defined_tagname(buf,1);
      if(-1!=tagnum) {
	/* using scope::classname; to be implemented 
	*  Now, G__tagtable is not ready */
      }
      else result=1;
    }
#else
    else result=1;
#endif
  }

  return(result);
}
#endif


/******************************************************************
* int G__get_envtagnum()
*
******************************************************************/
int G__get_envtagnum()
{
  int env_tagnum;
  if(-1!=G__def_tagnum) {
#ifdef G__OLDIMPLEMENTATION765
    G__ASSERT(G__ASM_FUNC_NOP==G__asm_wholefunction);
#endif
    /* In case of enclosed class definition, G__tagdefining remains
     * as enclosing class identity, while G__def_tagnum changes to
     * enclosed class identity. For finding environment scope, we
     * must use G__tagdefining */
    if(G__tagdefining!=G__def_tagnum) env_tagnum=G__tagdefining;
    else                              env_tagnum=G__def_tagnum;
  }
  else if(G__exec_memberfunc) env_tagnum = G__memberfunc_tagnum;
  /* else if(-1!=G__func_now)    env_tagnum = -2-G__func_now; */
  else                        env_tagnum = -1;
  return(env_tagnum);
}

/******************************************************************
* int G__isenclosingclass()
*
******************************************************************/
int G__isenclosingclass(enclosingtagnum,env_tagnum)
int enclosingtagnum;
int env_tagnum;
{
  int tagnum;
#ifndef G__OLDIMPLEMENTATION1565
  if(0>env_tagnum || 0>enclosingtagnum) return(0);
#else
  if(0>env_tagnum) return(0);
#endif
  tagnum = G__struct.parent_tagnum[env_tagnum];
  while(-1!=tagnum) {
    if(tagnum==enclosingtagnum) return(1);
    tagnum = G__struct.parent_tagnum[tagnum];
  }
  return(0);
}

#ifndef G__OLDIMPLEMENTATION949
/******************************************************************
* int G__isenclosingclassbase()
*
******************************************************************/
int G__isenclosingclassbase(enclosingtagnum,env_tagnum)
int enclosingtagnum;
int env_tagnum;
{
  int tagnum;
#ifndef G__OLDIMPLEMENTATION1565
  if(0>env_tagnum || 0>enclosingtagnum) return(0);
#else
  if(0>env_tagnum) return(0);
#endif
  tagnum = G__struct.parent_tagnum[env_tagnum];
  while(-1!=tagnum) {
    if (-1 != G__isanybase (enclosingtagnum, tagnum, G__STATICRESOLUTION)) 
      return 1;
    if(tagnum==enclosingtagnum) return(1);
    tagnum = G__struct.parent_tagnum[tagnum];
  }
  return(0);
}
#endif

#ifndef G__OLDIMPLEMENTATION671
 /******************************************************************
* char* G__find_first_scope_operator(name) by Scott Snyder 1997/10/17
*
* Return a pointer to the first scope operator in name.
* Only those at the outermost level of template nesting are considered.
******************************************************************/
char* G__find_first_scope_operator (name)
char* name;
{
  char* p = name;
  int single_quote = 0;
  int double_quote = 0;
  int nest = 0;

  while (*p != '\0') {

    char c = *p;

    if (0 == single_quote && 0 == double_quote) {
      if (c == '<')
        ++nest;
      else if (nest > 0 && c == '>')
        --nest;
      else if (nest == 0 && c == ':' && *(p+1) == ':')
        return p;
    }

    if('\''==c && 0==double_quote) 
      single_quote = single_quote ^ 1 ;
    else if('"'==c && 0==single_quote) 
      double_quote = double_quote ^ 1 ;

    ++p;
  }

  return 0;
}

/******************************************************************
* char* G__find_last_scope_operator(name)   by Scott Snyder 1997/10/17
*
* Return a pointer to the last scope operator in name.
* Only those at the outermost level of template nesting are considered.
******************************************************************/
char* G__find_last_scope_operator (name)
char* name;
{
  char* p = name + strlen (name) - 1;
  int single_quote = 0;
  int double_quote = 0;
  int nest = 0;

  while (p > name) {

    char c = *p;

    if (0 == single_quote && 0 == double_quote) {
      if (c == '>')
        ++nest;
      else if (nest > 0 && c == '<')
        --nest;
      else if (nest == 0 && c == ':' && *(p-1) == ':')
        return p-1;
    }

    if('\''==c && 0==double_quote) 
      single_quote = single_quote ^ 1 ;
    else if('"'==c && 0==single_quote) 
      double_quote = double_quote ^ 1 ;

    --p;
  }

  return 0;
}
#endif /* ON671 */

#ifndef G__OLDIMPLEMENTATION2014
/******************************************************************
 * G__set_class_autoloading
 ******************************************************************/
#define G__CLASS_AUTOLOAD  'a'
static int G__enable_autoloading=1;
int (*G__p_class_autoloading) G__P((char*,char*));

#ifndef G__OLDIMPLEMENTATION2097
/************************************************************************
* G__set_class_autloading
************************************************************************/
int G__set_class_autoloading(newvalue)
int newvalue;
{  
  int oldvalue =  G__enable_autoloading;
  G__enable_autoloading = newvalue;
  return oldvalue;
}
#endif

/************************************************************************
* G__set_class_autoloading_callback
************************************************************************/
void G__set_class_autoloading_callback(p2f)
int (*p2f) G__P((char*,char*));
{
  G__p_class_autoloading = p2f;
}

/******************************************************************
 * G__set_class_autoloading_table
 ******************************************************************/
void G__set_class_autoloading_table(classname,libname)
char* classname;
char* libname;
{
  int tagnum;
  G__enable_autoloading = 0;
  tagnum = G__search_tagname(classname,G__CLASS_AUTOLOAD);
  if(G__struct.libname[tagnum]) {
    free((void*)G__struct.libname[tagnum]);
  }
  G__struct.libname[tagnum]=(char*)malloc(strlen(libname)+1);
  strcpy(G__struct.libname[tagnum],libname);
  G__enable_autoloading = 1;
}

/******************************************************************
 * G__class_autoloading
 ******************************************************************/
int G__class_autoloading(tagnum) 
int tagnum;
{
  char* libname;
  if(tagnum<0 || !G__enable_autoloading) return(0);
  if(G__CLASS_AUTOLOAD==G__struct.type[tagnum]) {
    libname = G__struct.libname[tagnum];
    /* G__struct.type[tagnum]=0; */
    if(G__p_class_autoloading) {
#ifndef G__OLDIMPLEMENTATION2048
      int res;
      G__enable_autoloading = 0;
      res = (*G__p_class_autoloading)(G__struct.name[tagnum],libname);
      G__enable_autoloading = 1;
      return(res);
#else
      return((*G__p_class_autoloading)(G__struct.name[tagnum],libname));
#endif
    }
    else if(libname) {
      G__enable_autoloading = 0;
      if(G__LOADFILE_SUCCESS<=G__loadfile(libname)) {
	G__enable_autoloading = 1;
	return(1);
      }
      else {
	G__struct.type[tagnum]=G__CLASS_AUTOLOAD;
	G__enable_autoloading = 1;
	return(-1);
      }
    }
    else return(0);
  }
  return(0);
}
#endif

/******************************************************************
* int G__defined_tagname(tagname,noerror)
*
* Description:
*   Scan tagname table and return tagnum. If not match, error message
*  is shown and -1 will be returned. 
*  If non zero value is given to second argument 'noerror', error 
*  message will be suppressed.
*
*  noerror = 0   if not found try to instantiate template class
*                if template is not found, display error
*          = 1   if not found try to instantiate template class
*                no error messages if template is not found
*          = 2   if not found just return without trying template
*
* CAUTION:
*  If template class with constant argument is given to this function,
* tagname argument may be modified like below.
*    A<int,5*2> => A<int,10>
* This may cause unexpected side-effect.
******************************************************************/
int G__defined_tagname(tagname,noerror)
char *tagname;
int noerror;
{
#ifdef G__OLDIMPLEMENTATION1593
  static int boolflag=0;
#endif
  int i,len;
  char *p;
  char temp[G__LONGLINE];
  char atom_tagname[G__LONGLINE];
  int env_tagnum;
  int store_var_type;

  /* handles X<X<int>> as X<X<int> > */
  while((char*)NULL!=(p=strstr(tagname,">>"))) {
    ++p;
    strcpy(temp,p);
    *p = ' ';
    ++p;
    strcpy(p,temp);
  }

#ifndef G__OLDIMPLEMENTATION1683
  /* handles X<int > as X<int> */
  p = tagname;
  while((char*)NULL!=(p=strstr(p," >"))) {
    if('>' != *(p-1)) {
      strcpy(temp,p+1);
      strcpy(p,temp);
    }
    ++p;
  }
  /* handles X <int> as X<int> */
  p = tagname;
  while((char*)NULL!=(p=strstr(p," <"))) {
    strcpy(temp,p+1);
    strcpy(p,temp);
    ++p;
  }
  /* handles X<int>  as X<int> */
  p = tagname;
  while((char*)NULL!=(p=strstr(p,"> "))) {
    if(strncmp(p,"> >",3)==0) {
      p+=2;
    }
    else {
      strcpy(temp,p+2);
      strcpy(p+1,temp);
      ++p;
    }
  }
  /* handles X< int> as X<int> */
  p = tagname;
  while((char*)NULL!=(p=strstr(p,"< "))) {
    strcpy(temp,p+2);
    strcpy(p+1,temp);
    ++p;
  }
#endif

#ifndef G__OLDIMPLEMENTATION1683
  /* handle X<const const Y> */
  p = strstr(tagname,"const const ");
  while(p) {
    char *p1= (p+=6);
    char *p2=p+6;
    while(*p2) *p1++ = *p2++;
    *p1 = 0;
    p = strstr(p,"const const ");
  }
#endif

  if(isspace(tagname[0])) strcpy(temp,tagname+1);
  else strcpy(temp,tagname);
  p = G__find_last_scope_operator (temp);

  if(p) {
    strcpy(atom_tagname,p+2);
    *p='\0';
    if(p==temp) env_tagnum = -1;  /* global scope */
#ifndef G__STD_NAMESPACE /* ON667 */
    else if (strcmp (temp, "std") == 0
#ifndef G__OLDIMPLEMENTATION1285
	     && G__ignore_stdnamespace
#endif
	     ) {
      env_tagnum = -1;
      tagname += 5;
    }
#endif
#ifndef G__OLDIMPLEMENTATION1550
    else {
      env_tagnum = G__defined_tagname(temp,noerror);
      if(-1==env_tagnum) return(-1);
    }
#else
    else env_tagnum = G__defined_tagname(temp,0);
#endif
  }
  else {
    strcpy(atom_tagname,temp);
    env_tagnum = G__get_envtagnum();
  }


  /* Search for old tagname */
  len=strlen(atom_tagname);
#ifndef G__OLDIMPLEMENTATION1786
 try_again:
#else
#ifndef G__OLDIMPLEMENTATION1109
  if(0==len) {
    strcpy(atom_tagname,"$");
    len=1;
  }
#endif
#endif

#ifndef G__OLDIMPLEMENTATION1697
  for(i=G__struct.alltag-1;i>=0;i--) {
    if(len==G__struct.hash[i]&&strcmp(atom_tagname,G__struct.name[i])==0&&
       (((char*)NULL==p&&-1==G__struct.parent_tagnum[i])||
	env_tagnum==G__struct.parent_tagnum[i])) {
#ifndef G__OLDIMPLEMENTATION2014
      G__class_autoloading(i);
#endif
      return(i);
    }
  }
#endif

#ifndef G__OLDIMPLEMENTATION1786
  if(0==len) {
    strcpy(atom_tagname,"$");
    len=1;
    goto try_again;
  }
#endif

  for(i=G__struct.alltag-1;i>=0;i--) {
    if(len==G__struct.hash[i]&&strcmp(atom_tagname,G__struct.name[i])==0&&
#ifndef G__OLDIMPLEMENTATION1010
       (((char*)NULL==p&&-1==G__struct.parent_tagnum[i])||
#else
       (-1==G__struct.parent_tagnum[i]||
#endif
	env_tagnum==G__struct.parent_tagnum[i]||
#ifdef G__VIRTUALBASE
	-1!=G__isanybase(G__struct.parent_tagnum[i],env_tagnum
			 ,G__STATICRESOLUTION)||
#else
	-1!=G__isanybase(G__struct.parent_tagnum[i],env_tagnum)||
#endif
	G__isenclosingclass(G__struct.parent_tagnum[i],env_tagnum)
#ifndef G__OLDIMPLEMENTATION1057
	||G__isenclosingclassbase(G__struct.parent_tagnum[i],env_tagnum)
#endif
#ifndef G__OLDIMPLEMENTATION1010
	||((char*)NULL==p&&G__tmplt_def_tagnum==G__struct.parent_tagnum[i])
#else
	||G__tmplt_def_tagnum==G__struct.parent_tagnum[i]
#endif
#ifdef G__VIRTUALBASE
	||-1!=G__isanybase(G__struct.parent_tagnum[i],G__tmplt_def_tagnum
			   ,G__STATICRESOLUTION)
#else
	||-1!=G__isanybase(G__struct.parent_tagnum[i],G__tmplt_def_tagnum)
#endif
	||G__isenclosingclass(G__struct.parent_tagnum[i],G__tmplt_def_tagnum)
#ifndef G__OLDIMPLEMENTATION1057
	||G__isenclosingclassbase(G__struct.parent_tagnum[i],G__tmplt_def_tagnum)
#endif
	)) {
#ifndef G__OLDIMPLEMENTATION2014
      G__class_autoloading(i);
#endif
      return(i);
    }
  }

  /* if tagname not found, try instantiating class template */
#ifndef G__OLDIMPLEMENTATION848
  len=strlen(tagname);
  if('>'==tagname[len-1] && noerror<2 && (len<2||'-'!=tagname[len-2])) { 
#else
  if('>'==tagname[strlen(tagname)-1] && noerror<2) { 
     /* (0==G__struct.name[G__struct.alltag] ||
      strcmp(tagname,G__struct.name[G__struct.alltag]))) { */
#endif
#ifndef G__OLDIMPLEMENTATION1854
    if(G__loadingDLL) {
      G__fprinterr(G__serr,
		   "Error: '%s' Incomplete template resolution in shared library"
		   ,tagname);
      G__genericerror((char*)NULL);
      G__fprinterr(G__serr, "Add following line in header for making dictionary\n");
      G__fprinterr(G__serr, "   #pragma link C++ class %s;\n" ,tagname);
      G__exit(-1);
      return(-1);
    }
#endif
    /* CAUTION: tagname may be modified in following function */
    i=G__instantiate_templateclass(tagname);
    return(i);
  }
  else if(noerror<2) {
    struct G__Definedtemplateclass *deftmplt=G__defined_templateclass(tagname);
    if(deftmplt 
#ifndef G__OLDIMPLEMENTATION1046
       && deftmplt->def_para
#endif
       && deftmplt->def_para->default_parameter) {
      i=G__instantiate_templateclass(tagname);
      return(i);
    }
  }

#ifndef G__OLDIMPLEMENTATION884 
  /* THIS PART(884) MAY NOT BE NEEDED ANY MORE BY FIX 1604 */
  if(strcmp(tagname,"bool")==0) {
    if(
#ifndef G__OLDIMPLEMENTATION1593
       0==G__boolflag
#else
       0==boolflag
#endif
       ) {
#ifndef G__OLDIMPLEMENTATION913
      long store_globalvarpointer=G__globalvarpointer;
      int store_tagdefining=G__tagdefining;
      int store_def_struct_member=G__def_struct_member;
      int store_def_tagnum=G__def_tagnum;
      int store_tagnum=G__tagnum;
      int store_cpp=G__cpp;
#ifndef G__OLDIMPLEMENTATION1584
      int store_globalcomp = G__globalcomp;
#endif
      struct G__ifunc_table *store_ifunc = G__p_ifunc;
      G__cpp=0;
      G__globalvarpointer=G__PVOID;
      G__tagdefining = -1;
      G__def_struct_member=0;
      G__def_tagnum = -1;
      G__tagnum = -1;
      G__p_ifunc = &G__ifunc;
#endif
#ifndef G__OLDIMPLEMENTATION1593
      G__boolflag=1;
#else
      boolflag=1;
#endif
      G__loadfile("bool.h");
      i=G__defined_tagname(tagname,noerror);
#ifndef G__OLDIMPLEMENTATION1584
      G__globalcomp = store_globalcomp;
#endif
#ifndef G__OLDIMPLEMENTATION913
      G__cpp=store_cpp;
      G__globalvarpointer=store_globalvarpointer;
      G__tagdefining = store_tagdefining;
      G__def_struct_member=store_def_struct_member;
      G__def_tagnum = store_def_tagnum;
      G__tagnum = store_tagnum;
      G__p_ifunc = store_ifunc;
#endif
      return(i);
    }
  }
#endif

#ifndef G__OLDIMPLEMENTATION1735
  /* search for typename */
  store_var_type = G__var_type;
  i=G__defined_typename(tagname);
  G__var_type=store_var_type;
  if(-1!=i) {
    i=G__newtype.tagnum[i];
    if(-1!=i) {
#ifndef G__OLDIMPLEMENTATION2014
      G__class_autoloading(i);
#endif
      return(i);
    }
  }
#ifndef G__OLDIMPLEMENTATION957
  {
    int i2=0,cx;
      while((cx=tagname[i2++])) if(G__isoperator(cx)) return(-1);
  }
#endif
  /* not found */
  if(noerror==0) {
    G__fprinterr(G__serr,
	    "Error: class,struct,union or type %s not defined " ,tagname);
    G__genericerror((char*)NULL);
  }
#else /* 1735 */
  /* not found */
  if(noerror==0) {
#ifndef G__OLDIMPLEMENTATION957
    int i2=0,cx;
#endif
    store_var_type = G__var_type;
    i=G__defined_typename(tagname);
    G__var_type=store_var_type;
    if(-1!=i) {
      i=G__newtype.tagnum[i];
      if(-1!=i) {
#ifndef G__OLDIMPLEMENTATION2014
	G__class_autoloading(i);
#endif
	return(i);
      }
    }
#ifndef G__OLDIMPLEMENTATION957
    while((cx=tagname[i2++])) if(G__isoperator(cx)) return(-1);
#endif
    G__fprinterr(G__serr,
	    "Error: class,struct,union or type %s not defined " ,tagname);
    G__genericerror((char*)NULL);
  }
#endif /* 1735 */
  return(-1);

}

/******************************************************************
* int G__search_tagname(tagname,type)
*
* Description:
*   Scan tagname table and return tagnum. If not match, create
*  new tag type.
*
******************************************************************/
int G__search_tagname(tagname,type)
char *tagname;
int type;
{
  int i ,len;
  char *p;
#ifndef G__OLDIMPLEMENTATION1823
  char buf[G__BUFLEN*2];
  char buf2[G__BUFLEN*2];
  char *temp=buf;
  char *atom_tagname=buf2;
#else
  char temp[G__LONGLINE];
  char atom_tagname[G__LONGLINE];
#endif
  /* int parent_tagnum; */
#ifndef G__OLDIMPLEMENTATION1789
  int envtagnum= -1;
#endif
#ifndef G__OLDIMPLEMENTATION1206
  int isstructdecl = isupper(type);
  type = tolower(type);
#endif

  /* Search for old tagname */
  i = G__defined_tagname(tagname,2);

#ifndef G__OLDIMPLEMENTATION1823
  if(strlen(tagname)>G__BUFLEN*2-10) {
    temp = (char*)malloc(strlen(tagname)+10);
    atom_tagname = (char*)malloc(strlen(tagname)+10);
  }
#endif


#ifndef G__OLDIMPLEMENTATION1789
  p = G__strrstr(tagname,"::");
  if(p
#ifndef G__OLDIMPLEMENTATION1790
     && !strchr(p,'>')
#endif
     ) {
    strcpy(atom_tagname,tagname);
    p=G__strrstr(atom_tagname,"::");
    *p=0;
    envtagnum = G__defined_tagname(atom_tagname,1);
  }
  else {
    envtagnum = G__get_envtagnum();
  }
#endif
  
  /* if new tagname, initialize tag table */
  if(-1==i
#ifndef G__OLDIMPLEMENTATION1206
#ifndef G__OLDIMPLEMENTATION1789
     || (envtagnum != G__struct.parent_tagnum[i] && isstructdecl)
#else
     || (G__get_envtagnum() != G__struct.parent_tagnum[i] && isstructdecl)
#endif
#endif
     ) {

    i=G__struct.alltag;
    
    if(i==G__MAXSTRUCT) {
      G__fprinterr(G__serr,
	      "Limitation: Number of struct/union tag exceed %d FILE:%s LINE:%d\nFatal error, exit program. Increase G__MAXSTRUCT in G__ci.h and recompile %s\n"
	      ,G__MAXSTRUCT
	      ,G__ifile.name
	      ,G__ifile.line_number
	      ,G__nam);
      
      G__eof=1;
#ifndef G__OLDIMPLEMENTATION1823
      if(buf!=temp) free((void*)temp);
      if(buf2!=atom_tagname) free((void*)atom_tagname);
#endif
      return(-1);
    }

    strcpy(temp,tagname);
#ifndef G__OLDIMPLEMENTATION671
    p=G__find_last_scope_operator(temp);
#else
    p=G__strrstr(temp,"::");
#endif
    if(p) {
      strcpy(atom_tagname,p+2);
      *p = '\0';
#ifndef G__STD_NAMESPACE /* ON667 */
      if (strcmp (temp, "std") == 0
#ifndef G__OLDIMPLEMENTATION1285
	  && G__ignore_stdnamespace
#endif
	  ) G__struct.parent_tagnum[i] = -1;
      else G__struct.parent_tagnum[i] = G__defined_tagname(temp,0);
#else
      G__struct.parent_tagnum[i] = G__defined_tagname(temp,0);
#endif
    }
    else {
#ifndef G__OLDIMPLEMENTATION1446
#ifndef G__OLDIMPLEMENTATION1487
      int env_tagnum;
      if(-1!=G__def_tagnum) {
	if(G__tagdefining!=G__def_tagnum) env_tagnum=G__tagdefining;
	else                              env_tagnum=G__def_tagnum;
      }
      else env_tagnum = -1;
      G__struct.parent_tagnum[i]=env_tagnum;
#else
      if(G__iscpp) {
	int env_tagnum;
	if(-1!=G__def_tagnum) {
	  if(G__tagdefining!=G__def_tagnum) env_tagnum=G__tagdefining;
	  else                              env_tagnum=G__def_tagnum;
	}
	else env_tagnum = -1;
	G__struct.parent_tagnum[i]=env_tagnum;
      }
      else         G__struct.parent_tagnum[i]= -1;
#endif
#else
      if(G__iscpp) G__struct.parent_tagnum[i]=G__get_envtagnum();
      else         G__struct.parent_tagnum[i]= -1;
#endif
      strcpy(atom_tagname,temp);
    }

    if(strncmp("G__NONAME",atom_tagname,9)==0) {
      atom_tagname[0]='\0';
      len=0;
    }
    else {
      len=strlen(atom_tagname);
    }
    
#ifndef G__OLDIMPLEMENTATION1749
    G__struct.userparam[i]=0;
#endif
    G__struct.name[i]=malloc((size_t)(len+1));
    strcpy(G__struct.name[i],atom_tagname);
    G__struct.hash[i]=len;
    
    G__struct.size[i]=0;
    G__struct.type[i]=type; /* 's' struct ,'u' union ,'e' enum , 'c' class */
    
    /***********************************************************
     * Allocate and initialize member variable table 
     ************************************************************/
    G__struct.memvar[i] = (struct G__var_array *)malloc(sizeof(struct G__var_array));
#ifdef G__OLDIMPLEMENTATION1776_YET
    memset(G__struct.memvar[i],0,sizeof(struct G__var_array));
#endif
#ifndef G__OLDIMPLEMENTATION2038
    G__struct.memvar[i]->enclosing_scope = (struct G__var_array*)NULL;
    G__struct.memvar[i]->inner_scope = (struct G__var_array**)NULL;
#endif
#ifndef G__OLDIMPLEMENTATION1866
    G__struct.memvar[i]->ifunc = (struct G__ifunc_table*)NULL;
#endif
    G__struct.memvar[i]->varlabel[0][0]=0;
    G__struct.memvar[i]->paran[0]=0;
    G__struct.memvar[i]->allvar=0;
    G__struct.memvar[i]->next = NULL;
    G__struct.memvar[i]->tagnum = i;
#ifndef G__OLDIMPLEMENTATION1543
    { 
      int ix;
      for(ix=0;ix<G__MEMDEPTH;ix++) {
	G__struct.memvar[i]->varnamebuf[ix]=(char*)NULL;
#ifndef G__OLDIMPLEMENTATION1776
	G__struct.memvar[i]->p[ix] = 0;
#endif
      }
    }
#endif
    
    /***********************************************************
     * Allocate and initialize member function table list
     ***********************************************************/
    G__struct.memfunc[i] = (struct G__ifunc_table *)malloc(sizeof(struct G__ifunc_table));
    G__struct.memfunc[i]->allifunc = 0;
    G__struct.memfunc[i]->next = (struct G__ifunc_table *)NULL;
    G__struct.memfunc[i]->page = 0;
#ifdef G__NEWINHERIT
    G__struct.memfunc[i]->tagnum = i;
#endif
#ifndef G__OLDIMPLEMENTATION1543
    {
      int ix;
      for(ix=0;ix<G__MAXIFUNC;ix++) {
	G__struct.memfunc[i]->funcname[ix]=(char*)NULL;
#ifndef G__OLDIMPLEMENTATION1706
	G__struct.memfunc[i]->override_ifunc[ix]=(struct G__ifunc_table*)NULL;
	G__struct.memfunc[i]->override_ifn[ix] = 0;
	G__struct.memfunc[i]->masking_ifunc[ix] = (struct G__ifunc_table*)NULL;
	G__struct.memfunc[i]->masking_ifn[ix] = 0;
#endif
      }
    }
#endif
#ifndef G__OLDIMPLEMENTATION2027
    /* reserve the first entry for dtor */
    G__struct.memfunc[i]->hash[0]=0;
    G__struct.memfunc[i]->funcname[0]=(char*)malloc(2);
    G__struct.memfunc[i]->funcname[0][0]=0;
    G__struct.memfunc[i]->para_nu[0]=0;
    G__struct.memfunc[i]->pentry[0] = &G__struct.memfunc[i]->entry[0];
    G__struct.memfunc[i]->pentry[0]->bytecode=(struct G__bytecodefunc*)NULL;
    G__struct.memfunc[i]->friendtag[0]=(struct G__friendtag*)NULL;
#ifndef G__OLDIMPLEMENTATION2039
    G__struct.memfunc[i]->pentry[0]->size = 0; 
    G__struct.memfunc[i]->pentry[0]->filenum = 0; 
    G__struct.memfunc[i]->pentry[0]->line_number = 0; 
#ifndef G__OLDIMPLEMENTATION2200
    G__struct.memfunc[i]->pentry[0]->bytecodestatus = G__BYTECODE_NOTYET;
#endif
    G__struct.memfunc[i]->ispurevirtual[0] = 0;
    G__struct.memfunc[i]->access[0] = G__PUBLIC;
    G__struct.memfunc[i]->ansi[0] = 1; 
    G__struct.memfunc[i]->isconst[0] = 0; 
    G__struct.memfunc[i]->reftype[0] = 0; 
    G__struct.memfunc[i]->type[0] = 0; 
    G__struct.memfunc[i]->p_tagtable[0] = -1; 
    G__struct.memfunc[i]->p_typetable[0] = -1; 
    G__struct.memfunc[i]->staticalloc[0] = 0; 
#ifndef G__OLDIMPLEMENTATION2050
    G__struct.memfunc[i]->busy[0] = 0; 
#endif
#endif

#ifndef G__OLDIMPLEMENTATION2200
#ifdef G__FONS_COMMENT
    G__struct.memfunc[i]->comment[0].filenum = -1;
#endif
#endif

    G__struct.memfunc[i]->allifunc = 1;
#endif

    /***********************************************************
     * Allocate and initialize class inheritance table
     ***********************************************************/
    G__struct.baseclass[i] = (struct G__inheritance *)malloc(sizeof(struct G__inheritance));
    G__struct.baseclass[i]->basen=0;
    
    /***********************************************************
     * Initialize iden information for virtual function
     ***********************************************************/
    G__struct.virtual_offset[i] = -1; /* -1 means no virtual function */
    
    G__struct.isabstract[i]=0;
    
#ifndef G__OLDIMPLEMENTATION1700
    G__struct.globalcomp[i] = G__default_link?G__globalcomp:G__NOLINK;
#else
    G__struct.globalcomp[i] = G__globalcomp;
#endif
    G__struct.iscpplink[i] = 0;
#ifndef G__OLDIMPLEMENTATION1334
    G__struct.protectedaccess[i] = 0;
#endif

    G__struct.line_number[i] = -1;
    G__struct.filenum[i] = -1;

    G__struct.istypedefed[i] = 0;

    G__struct.funcs[i] = 0;

    G__struct.istrace[i] = 0;
    G__struct.isbreak[i] = 0;

#ifdef G__FRIEND
    G__struct.friendtag[i] = (struct G__friendtag*)NULL;
#endif

#ifdef G__FONS_COMMENT
    G__struct.comment[i].p.com = (char*)NULL;
#ifdef G__OLDIMPLEMENTATION469
    G__struct.comment[i].p.pos = (fpos_t)0;
#endif
    G__struct.comment[i].filenum = -1;
#endif

    G__struct.incsetup_memvar[i] = (G__incsetup)NULL;
    G__struct.incsetup_memfunc[i] = (G__incsetup)NULL;
#ifdef G__ROOTSPECIAL
    G__struct.rootflag[i] = 0;
    G__struct.rootspecial[i] = (struct G__RootSpecial*)NULL;
#endif

#ifndef G__OLDIMPLEMENTATION1238
    G__struct.isctor[i] = 0;
#endif

#ifndef G__OLDIMPLEMENTATION1503
    G__struct.defaulttypenum[i] = -1;
#endif
#ifndef G__OLDIMPLEMENTATION2073
    G__struct.vtable[i]= (void*)NULL;
#endif

    G__struct.alltag++;
  }
  else if(0==G__struct.type[i] 
#ifndef G__OLDIMPLEMENTATION2014
	  || 'a'==G__struct.type[i]
#endif
	  ) {
#ifndef G__OLDIMPLEMENTATION2015
    switch(type) {
    case 'c':
    case 's':
    case 'e':
    case 'n':
    case 'a':
      G__struct.type[i]=type; 
      break;
    default:
      if('a'!=G__struct.type[i]) G__struct.type[i]=type-1;
      break;
    }
#else
    G__struct.type[i]=type; 
#endif
  }

  /* return tag table number */
#ifndef G__OLDIMPLEMENTATION1823
  if(buf!=temp) free((void*)temp);
  if(buf2!=atom_tagname) free((void*)atom_tagname);
#endif
  return(i);
}

/******************************************************************
* G__alloc_var_array()
******************************************************************/
struct G__var_array* G__alloc_var_array(var,pig15)
struct G__var_array* var;
int *pig15;
{
  if(var->allvar<G__MEMDEPTH) {
    *pig15=var->allvar;
  }
  else {
    var->next = (struct G__var_array *)malloc(sizeof(struct G__var_array)) ;
#ifdef G__OLDIMPLEMENTATION1776_YET
    memset(var->next,0,sizeof(struct G__var_array));
#endif
#ifndef G__OLDIMPLEMENTATION2038
    var->next->enclosing_scope = (struct G__var_array*)NULL;
    var->next->inner_scope = (struct G__var_array**)NULL;
#endif
#ifndef G__OLDIMPLEMENTATION1866
    var->next->ifunc = (struct G__ifunc_table*)NULL;
#endif
    var->next->tagnum=var->tagnum;
    var = var->next;
    var->varlabel[0][0]=0;
    var->paran[0]=0;
    var->next=NULL;
    var->allvar=0;
#ifndef G__OLDIMPLEMENTATION1543
    { 
      int ix;
      for(ix=0;ix<G__MEMDEPTH;ix++) {
	var->varnamebuf[ix]=(char*)NULL;
#ifndef G__OLDIMPLEMENTATION1776
	var->p[ix] = 0;
#endif
      }
    }
#endif
    *pig15=0;
  }
  return(var);
}
/******************************************************************
* G__copy_unionmember()
******************************************************************/
static void G__copy_unionmember(var,ig15,envvar,envig15,offset,access
				,statictype)
struct G__var_array* var;
int ig15;
struct G__var_array* envvar;
int envig15;
long offset;
int access;
int statictype;
{
  int i;
  envvar->p[envig15]=offset;
#ifndef G__OLDIMPLEMENTATION1543
  G__savestring(&envvar->varnamebuf[envig15],var->varnamebuf[ig15]);
#else
  strcpy(envvar->varnamebuf[envig15],var->varnamebuf[ig15]);
#endif
  envvar->hash[envig15]=var->hash[ig15];
  for(i=0;i<G__MAXVARDIM;i++) 
    envvar->varlabel[envig15][i]=var->varlabel[ig15][i];
  envvar->paran[envig15]=var->paran[ig15];
  envvar->bitfield[envig15]=var->bitfield[ig15];
  envvar->type[envig15]=var->type[ig15];
  envvar->constvar[envig15]=var->constvar[ig15];
  envvar->p_tagtable[envig15]=var->p_tagtable[ig15];
  envvar->p_typetable[envig15]=var->p_typetable[ig15];
  envvar->statictype[envig15]=statictype;
  envvar->reftype[envig15]=var->reftype[ig15];
  envvar->access[envig15]=access;
  envvar->globalcomp[envig15]=var->globalcomp[ig15];
#ifndef G__PHILIPPE0
  /* Let's also copy the comment information */
  envvar->comment[envig15].p.com=var->comment[ig15].p.com;
  envvar->comment[envig15].p.pos=var->comment[ig15].p.pos;
  envvar->comment[envig15].filenum = var->comment[ig15].filenum;
#else
  envvar->comment[ig15].p.com=(char*)NULL;
  envvar->comment[ig15].filenum = -1;
#endif
}

/******************************************************************
* G__add_anonymousunion()
******************************************************************/
static void G__add_anonymousunion(tagnum,def_struct_member,envtagnum)
int tagnum;
int def_struct_member;
int envtagnum;
{
  int envig15;
  int ig15;
  struct G__var_array *var;
  struct G__var_array *envvar;
  long offset;
  int access;
  int statictype=G__AUTO;
  var = G__struct.memvar[tagnum];
  if(def_struct_member) {
    /* anonymous union as class/struct member */
    envvar = G__struct.memvar[envtagnum];
    while(envvar->next) envvar=envvar->next;
    envvar=G__alloc_var_array(envvar,&envig15);
#ifndef G__OLDIMPLEMENTATION1747
    if(0==envig15) access=G__access;
    else access=envvar->access[envig15-1];
#else
    if(0==envig15) access=G__PUBLIC;
    else access=envvar->access[envig15-1];
#endif

    offset=G__malloc(1,G__struct.size[tagnum],"");
    while(var) {
      for(ig15=0;ig15<var->allvar;ig15++) {
	envvar=G__alloc_var_array(envvar,&envig15);
	G__copy_unionmember(var,ig15,envvar,envig15,offset,access,statictype);
	++envvar->allvar;
      }
      var=var->next;
    }
  }
  else {
    /* variable body as global or local variable */
    if(G__p_local) envvar= G__p_local;
    else {
      envvar= &G__global;
#ifndef G__OLDIMPLEMENTATION419
      statictype=G__ifile.filenum; /* file scope static */
#endif
    }
    while(envvar->next) envvar=envvar->next;
    envvar=G__alloc_var_array(envvar,&envig15);
    access=G__PUBLIC;

    offset=G__malloc(1,G__struct.size[tagnum],"");
    while(var) {
      for(ig15=0;ig15<var->allvar;ig15++) {
	envvar=G__alloc_var_array(envvar,&envig15);
	G__copy_unionmember(var,ig15,envvar,envig15,offset,access,statictype);
	statictype=G__COMPILEDGLOBAL;
	++envvar->allvar;
      }
      var=var->next;
    }
  }
}

/******************************************************************
* G__define_struct(type)
*
* [struct|union|enum] tagname { member } item ;
* [struct|union|enum]         { member } item ;
* [struct|union|enum] tagname            item ;
* [struct|union|enum] tagname { member }      ;
*
******************************************************************/
void G__define_struct(type)
/* struct G__input_file *fin; */
char type;
{
  /* fpos_t rewind_fpos; */
  int c;
  char tagname[G__LONGLINE],category[10],memname[G__ONELINE],val[G__ONELINE];
  int /* itag=0, */ mparen,store_tagnum ,store_def_struct_member=0;
  struct G__var_array *store_local;
  /* char store_tagname[G__LONGLINE]; */
  G__value enumval;
  
  int tagdefining;
  int store_access;
  char basename[G__LONGLINE];
  int *pbasen;
  struct G__inheritance *baseclass;
  int baseaccess;
  int newdecl;
  /* int lenheader; */
  int store_static_alloc;
  int len;
  int ispointer=0;
  int store_prerun;
  int store_def_tagnum;
  int isvirtualbase=0;
#ifndef G__OLDIMPLEMENTATION1087
  int isclassdef=0;
#endif
  
#ifdef G__ASM
#ifdef G__ASM_DBG
  if(G__asm_dbg&&G__asm_noverflow)
    G__fprinterr(G__serr,"LOOP COMPILE ABORTED FILE:%s LINE:%d\n"
	    ,G__ifile.name
	    ,G__ifile.line_number);
#endif
  G__abortbytecode();
#endif
  
  /*
   * [struct|union|enum]   tagname  { member }  item ;
   *                    ^
   * read tagname
   */
  /* fgetpos(G__ifile.fp,&rewind_fpos); */
  
#ifndef G__OLDIMPLEMENTATION730
  c=G__fgetname_template(tagname,"{:;=&");
#else
  c=G__fgetname_template(tagname,"{:;=");
#endif

#ifndef G__OLDIMPLEMENTATION1038
  if(strlen(tagname)>=G__LONGLINE) {
    G__fprinterr(G__serr,"Limitation: class name too long. Must be < %d"
	    ,G__LONGLINE);
    G__genericerror((char*)NULL);
  }
#endif

 doitagain:
  
  /*
   * [struct|union|enum]   tagname{ member }  item ;
   *                               ^
   *                     OR
   * [struct|union|enum]          { member }  item ;
   *                               ^
   * push back before '{' and fgetpos 
   */
  if(c=='{') {
    fseek(G__ifile.fp,-1,SEEK_CUR);
    if(G__dispsource) G__disp_mask=1;
  }
  
  /*
   * [struct|union|enum]   tagname   { member }  item ;
   *                               ^
   *                     OR
   * [struct|union|enum]   tagname     item ;
   *                               ^
   *                     OR
   * [struct|union|enum]   tagname      ;
   *                               ^
   * skip space and push back
   */
  else if(isspace(c)) {
    c=G__fgetspace(); /* '{' , 'a-zA-Z' or ';' are expected */
    /* if(c==';') return; */
    if(c!=':') {
      fseek(G__ifile.fp,-1,SEEK_CUR);
      if(G__dispsource) G__disp_mask=1;
    }
  }
  else if(c==':') {
#ifndef G__OLDIMPLEMENTATION1733
    /* inheritance or nested class */
    c = G__fgetc();
    if(':'==c) {
      strcat(tagname,"::");
      len=strlen(tagname);
      c=G__fgetname_template(tagname+len,"{:;=&");
      goto doitagain;
    }
    else {
      fseek(G__ifile.fp,-1,SEEK_CUR);
      if(G__dispsource) G__disp_mask=1;
      c=':';
    }
#else
    /* inheritance */
#endif
  }
  else if(c==';') {
    /* tagname declaration */
  }
#ifndef G__OLDIMPLEMENTATION612
  else if(c=='=' && 'n'==type) {
    /* namespace alias=nsn; treat as typedef */
    c=G__fgetstream_template(basename,";");
    tagdefining=G__defined_tagname(basename,0);
    if(-1!=tagdefining) {
      int typenum;
      typenum=G__search_typename(tagname,'u',tagdefining,0);
      G__newtype.parent_tagnum[typenum]=G__get_envtagnum();
    }
    G__var_type='p';
    return;
  }
#endif
#ifndef G__OLDIMPLEMENTATION612
  else if(G__ansiheader && (','==c || ')'==c)) {
    /* dummy argument for func overloading f(class A*) { } */
    G__var_type='p';
    if(')'==c) G__ansiheader=0;
    return;
  }
  else if('&'==c) {
    fseek(G__ifile.fp,-1,SEEK_CUR);
    if(G__dispsource) G__disp_mask=1;
    c=' ';
  }
#endif
  else {
    G__genericerror("Syntax error in class/struct definition");
  }
  
  /*
   * set default tagname if tagname is omitted
   */
  if(tagname[0]=='\0') {
    if('e'==type) {
      strcpy(tagname,"$");
    }
#ifndef G__OLDIMPLEMENTATION612
    else if('n'==type) {
      /* unnamed namespace, treat as global scope, namespace has no effect. 
       * This implementation may be wrong. 
       * Should fix later with using directive in global scope */
      G__var_type='p';
      G__exec_statement();
      return;
    }
#endif
    else {
      sprintf(tagname,"G__NONAME%d",G__struct.alltag);
    }
  }
#ifndef G__STD_NAMESPACE /* ON667 */
  else if('n'==type && strcmp(tagname,"std")==0
#ifndef G__OLDIMPLEMENTATION1285
	  && (G__ignore_stdnamespace
#ifndef G__OLDIMPLEMENTATION1692
	      || -1!=G__def_tagnum
#endif
	      )
#endif
	  ) {
    /* namespace std, treat as global scope, namespace has no effect. */
    G__var_type='p';
    G__exec_statement();
    return;
  }
#endif
  
  /* BUG FIX, 17 Nov 1992
   *  tagnum wasn't saved
   */
  store_tagnum=G__tagnum;
  store_def_tagnum = G__def_tagnum;
  /*
   * Get tagnum, new tagtable is allocated if new
   */
  len=strlen(tagname);
  if(len&&'*'==tagname[len-1]) {
    ispointer=1;
    tagname[len-1]='\0';
  }
#ifndef G__OLDIMPLEMENTATION1206
  switch(c) {
  case '{':
  case ':':
  case ';':
    G__tagnum=G__search_tagname(tagname,toupper(type));
    break;
  default:
    G__tagnum=G__search_tagname(tagname,type);
    break;
  }
#else
  G__tagnum=G__search_tagname(tagname,type);
#endif

  if(';'==c) {
    /* in case of class name declaration 'class A;' */
    G__tagnum=store_tagnum;
    return;
  }
  if(G__tagnum<0) {
    /* This case might not happen */
    G__fignorestream(";");
    G__tagnum=store_tagnum;
    return;
  }
  G__def_tagnum = G__tagnum;
  
  /*
   * judge if new declaration by size
   */
  if(G__struct.size[G__tagnum]==0) {
    newdecl=1;
  }
  else {
    newdecl=0;
  }
  
  /* typenum is -1 for struct,union,enum without typedef */
  G__typenum = -1;
  
  /* Now came to
   * [struct|union|enum]   tagname   { member }  item ;
   *                                 ^
   *                     OR
   * [struct|union|enum]             { member }  item ;
   *                                 ^
   *                     OR
   * [struct|union|enum]   tagname     item ;
   *                                   ^
   * member declaration if exist
   */
  
  /**************************************************************
   * base class declaration
   **************************************************************/
  if(c==':') c=',';
  while(c==',') {
    /* [struct|class] <tagname> : <private|public> base_class { 
     *                           ^                                */

#ifndef G__OLDIMPLEMENTATION605
    /* reset virtualbase flag */
    isvirtualbase = 0;
#endif
    
    /* read base class name */
#ifdef G__TEMPLATECLASS
    c=G__fgetname_template(basename,"{,"); /* case 2) */
#else
    c=G__fgetname(basename,"{,");
#endif

#ifndef G__OLDIMPLEMENTATION1038
    if(strlen(basename)>=G__LONGLINE) {
      G__fprinterr(G__serr,"Limitation: class name too long. Must be < %d"
	      ,G__LONGLINE);
      G__genericerror((char*)NULL);
    }
#endif
    
    /* [struct|class] <tagname> : <private|public> base1 , base2 { 
     *                                            ^  or ^         */

    if(strcmp(basename,"virtual")==0) {
#ifndef G__VIRTUALBASE
      if(G__NOLINK==G__globalcomp&&G__NOLINK==G__store_globalcomp)
	G__genericerror("Limitation: virtual base class not supported in interpretation");
#endif
      c=G__fgetname_template(basename,"{,");
      isvirtualbase = G__ISVIRTUALBASE;
#ifndef G__OLDIMPLEMENTATION1038
      if(strlen(basename)>=G__LONGLINE) {
	G__fprinterr(G__serr,"Limitation: class name too long. Must be < %d"
		,G__LONGLINE);
	G__genericerror((char*)NULL);
      }
#endif
    }
    
    if('c'==type) baseaccess=G__PRIVATE;
    else          baseaccess=G__PUBLIC;
    if(strcmp(basename,"public")==0) {
      baseaccess=G__PUBLIC;
#ifdef G__TEMPLATECLASS
      c=G__fgetname_template(basename,"{,");
#else
      c=G__fgetname(basename,"{,");
#endif
#ifndef G__OLDIMPLEMENTATION1038
      if(strlen(basename)>=G__LONGLINE) {
	G__fprinterr(G__serr,"Limitation: class name too long. Must be < %d"
		,G__LONGLINE);
	G__genericerror((char*)NULL);
      }
#endif
    }
    else if(strcmp(basename,"private")==0) {
      baseaccess=G__PRIVATE;
#ifdef G__TEMPLATECLASS
      c=G__fgetname_template(basename,"{,");
#else
      c=G__fgetname(basename,"{,");
#endif
#ifndef G__OLDIMPLEMENTATION1038
      if(strlen(basename)>=G__LONGLINE) {
	G__fprinterr(G__serr,"Limitation: class name too long. Must be < %d"
		,G__LONGLINE);
	G__genericerror((char*)NULL);
      }
#endif
    }
    else if(strcmp(basename,"protected")==0) {
      baseaccess=G__PROTECTED;
#ifdef G__TEMPLATECLASS
      c=G__fgetname_template(basename,"{,");
#else
      c=G__fgetname(basename,"{,");
#endif
#ifndef G__OLDIMPLEMENTATION1038
      if(strlen(basename)>=G__LONGLINE) {
	G__fprinterr(G__serr,"Limitation: class name too long. Must be < %d"
		,G__LONGLINE);
	G__genericerror((char*)NULL);
      }
#endif
    }

    if(strcmp(basename,"virtual")==0) {
#ifndef G__VIRTUALBASE
      if(G__NOLINK==G__globalcomp&&G__NOLINK==G__store_globalcomp)
	G__genericerror("Limitation: virtual base class not supported in interpretation");
#endif
      c=G__fgetname_template(basename,"{,");
      isvirtualbase = G__ISVIRTUALBASE;
#ifndef G__OLDIMPLEMENTATION1038
      if(strlen(basename)>=G__LONGLINE) {
	G__fprinterr(G__serr,"Limitation: class name too long. Must be < %d"
		,G__LONGLINE);
	G__genericerror((char*)NULL);
      }
#endif
    }

#ifndef G__PHILIPPE8
    if ( strlen(basename)!=0 && isspace(c) ) {
      /* maybe basename is namespace that got cut because
       * G__fgetname_template stop at spaces and the user add:
       * class MyClass : public MyNamespace ::MyTopClass !
       * or 
       * class MyClass : public MyNamespace:: MyTopClass !
      */
      int namespace_tagnum;
      char temp[G__LONGLINE];
  
      namespace_tagnum = G__defined_tagname(basename,2);
      while ( ( ( (namespace_tagnum!=-1)
		  && (G__struct.type[namespace_tagnum]=='n') )
		|| (strcmp("std",basename)==0)
		|| (basename[strlen(basename)-1]==':') )
	      && isspace(c) ) {
	c = G__fgetname_template(temp,"{,");
	strcat(basename,temp);
	namespace_tagnum = G__defined_tagname(basename,2);
      }
    }
#endif

    if(newdecl) {
#ifndef G__OLDIMPLEMENTATION693
      int lstore_tagnum=G__tagnum;
      int lstore_def_tagnum=G__def_tagnum;
      int lstore_tagdefining=G__tagdefining;
      int lstore_def_struct_member=G__def_struct_member;
      G__tagnum = G__struct.parent_tagnum[lstore_tagnum];
      G__def_tagnum = G__tagnum;
      G__tagdefining = G__tagnum;
      if(-1!=G__tagnum) G__def_struct_member=1 ;
      else              G__def_struct_member=0 ;
      /* copy pointer for readability */
      /* member = G__struct.memvar[lstore_tagnum]; */
      baseclass = G__struct.baseclass[lstore_tagnum];
      pbasen= &(baseclass->basen);
#else /* ON693 */
      /* copy pointer for readability */
      /* member = G__struct.memvar[G__tagnum]; */
      baseclass = G__struct.baseclass[G__tagnum];
      pbasen= &(baseclass->basen);
#endif /* ON693 */
      
      /* 
       * set base class information to tag info table 
       */
      baseclass->property[*pbasen]=G__ISDIRECTINHERIT + isvirtualbase;
      baseclass->basetagnum[*pbasen]=G__defined_tagname(basename,0);
#ifndef G__OLDIMPLEMENTATION693
      if(1==G__struct.size[lstore_tagnum]
#ifndef G__OLDIMPLEMENTATION1807
	 && 0==G__struct.memvar[lstore_tagnum]->allvar 
	 && 0==G__struct.baseclass[lstore_tagnum]->basen
#endif
	 )
	baseclass->baseoffset[*pbasen]=0;
      else
	baseclass->baseoffset[*pbasen]=G__struct.size[lstore_tagnum];
#else
      baseclass->baseoffset[*pbasen]=G__struct.size[lstore_tagnum];
#endif
      baseclass->baseaccess[*pbasen]=baseaccess;
      G__tagnum = lstore_tagnum;
      G__def_tagnum = lstore_def_tagnum;
      G__tagdefining = lstore_tagdefining;
      G__def_struct_member=lstore_def_struct_member;
      /* virtual base class for interpretation to be implemented and
       * 2 limitation messages above should be deleted. */
#ifndef G__OLDIMPLEMENTATION693
      if(1==G__struct.size[baseclass->basetagnum[*pbasen]]
#ifndef G__OLDIMPLEMENTATION1807
	 && 0==G__struct.memvar[baseclass->basetagnum[*pbasen]]->allvar 
	 && 0==G__struct.baseclass[baseclass->basetagnum[*pbasen]]->basen
#endif
	 ) {
	if(isvirtualbase)
	  G__struct.size[G__tagnum] += G__DOUBLEALLOC;
	else
	  G__struct.size[G__tagnum] += 0;
      }
      else {
	if(isvirtualbase)
	  G__struct.size[G__tagnum] 
	    += (G__struct.size[baseclass->basetagnum[*pbasen]]+G__DOUBLEALLOC);
	else
	  G__struct.size[G__tagnum] 
	    += G__struct.size[baseclass->basetagnum[*pbasen]];
      }
#else
      if(isvirtualbase)
	G__struct.size[G__tagnum] 
	  += (G__struct.size[baseclass->basetagnum[*pbasen]]+G__DOUBLEALLOC);
      else
	G__struct.size[G__tagnum] 
	  += G__struct.size[baseclass->basetagnum[*pbasen]];
#endif
      
      /* 
       * inherit base class info, variable member, function member 
       */
      G__inheritclass(G__tagnum,baseclass->basetagnum[*pbasen],baseaccess);

      /* ++(*pbasen); */
    }
    
    /* 
     * reading remaining space 
     */
    if(isspace(c)) {
      c=G__fignorestream("{,");
    }
    
    /* rewind 1 char if '{' */
    if(c=='{') {
      fseek(G__ifile.fp,-1,SEEK_CUR);
      if(G__dispsource) G__disp_mask=1;
    }
    
  } /* end of base class declaration */


  /**************************************************************
   * virtual base class isabstract count duplication check
   **************************************************************/
  baseclass = G__struct.baseclass[G__tagnum];
#ifndef G__OLDIMPLEMENTATION943
#ifndef G__PHILIPPE0
  /* When it is not a new declaration, updating purecount is going to
     make us fail because the rest of the code is not going to be run.
     Anyway we already checked once. */
  if (newdecl) {
#else 
  {
#endif
    int purecount= 0;
    int lastdirect = 0;
    int ivb;
    for (ivb = 0; ivb < baseclass->basen; ++ivb) {
      struct G__ifunc_table* itab;

      if (baseclass->property[ivb]&G__ISDIRECTINHERIT)
        lastdirect = ivb;

#ifndef G__OLDIMPLEMENTATION2037
      /* insure the loading of the memfunc */
      G__incsetup_memfunc(baseclass->basetagnum[ivb]); 
#endif

      itab = G__struct.memfunc[baseclass->basetagnum[ivb]];
      while (itab) {
        int ifunc;
        for (ifunc = 0; ifunc < itab->allifunc; ++ifunc) {
          if (itab->ispurevirtual[ifunc]) {
            /* Search to see if this function has an overrider.
               If we get this class through virtual derivation, search
               all classes; otherwise, search only those derived
               from it. */
            int firstb, lastb;
            int b2;
            int found_flag = 0;

            if (baseclass->property[ivb] & G__ISVIRTUALBASE) {
              firstb = 0;
              lastb = baseclass->basen;
            }
            else {
              firstb = lastdirect;
              lastb = ivb;
            }

            for (b2 = firstb; b2 < lastb; ++b2) {
              struct G__ifunc_table* found_tab;
              int found_ndx;
              int basetag;

              if (b2 == ivb)
                continue;

              basetag = baseclass->basetagnum[b2];
              if (G__isanybase (baseclass->basetagnum[ivb], basetag
				, G__STATICRESOLUTION) < 0)
                continue;

              found_tab = G__ifunc_exist (itab, ifunc,
                                          G__struct.memfunc[basetag],
                                          &found_ndx,0xffff);
              if (found_tab) {
                found_flag = 1;
                break;
              }
            }

            if (!found_flag)
              ++purecount;
          }
        }
        itab = itab->next;
      }
    }
    G__struct.isabstract[G__tagnum] = purecount;
  }
#else /* ON943 */
  if(baseclass->basen>0) {
    int basetags[G__MAXBASE];
    int vbcount[G__MAXBASE];
    int nunique=0;
    int ivb;
    int uvb;
    int flag;
    /* Create table of virtual base class reference count */
    for(ivb=0;ivb<baseclass->basen;ivb++) {
      if(baseclass->property[ivb]&G__ISVIRTUALBASE) {
	flag=0;
	for(uvb=0;uvb<nunique;uvb++) {
	  if(basetags[uvb]==baseclass->basetagnum[ivb]) {
	    ++vbcount[uvb];
	    flag=1;
	    break;
	  }
	}
	if(0==flag) {
	  basetags[nunique] = baseclass->basetagnum[ivb];
	  vbcount[nunique] = 1;
	  ++nunique;
	}
      }
    }
    /* Scan virtual base class table and subtract duplication */
    for(uvb=0;uvb<nunique;uvb++) {
      if(vbcount[uvb]>1) {
	G__struct.isabstract[G__tagnum] 
	  -= G__struct.isabstract[basetags[uvb]]*(vbcount[uvb]-1);
      }
    }
  }
#endif /* ON943 */
  
  /* fsetpos(G__ifile.fp,&rewind_fpos); */
  if(c=='{') { /* member declarations */

#ifndef G__OLDIMPLEMENTATION1087
    isclassdef=1;
#endif
    
#ifndef G__OLDIMPLEMENTATION612
    if(newdecl || 'n'==type) {
#else
    if(newdecl) {
#endif
      
      G__struct.line_number[G__tagnum] = G__ifile.line_number;
      G__struct.filenum[G__tagnum] = G__ifile.filenum;
      
      store_access=G__access;
      G__access = G__PUBLIC;
      switch(type) {
      case 's':
	sprintf(category,"struct");
	break;
      case 'c':
	sprintf(category,"class");
	G__access = G__PRIVATE;
	break;
      case 'u':
	sprintf(category,"union");
	break;
      case 'e':
	sprintf(category,"enum");
	break;
#ifndef G__OLDIMPLEMENTATION612
      case 'n':
	sprintf(category,"namespace");
	break;
#endif
      default:
	G__genericerror("Error: Illegal tagtype. struct,union,enum expected");
	break;
      }

      if(type=='e') { /* enum */

#ifdef G__OLDIMPLEMENTATION1386_YET
	G__struct.size[G__def_tagnum] = G__INTALLOC;
#endif
	G__fgetc(); /* skip '{' */
#ifndef G__OLDIMPLEMENTATION1179
	/* Change by Philippe Canal, 1999/8/26 */
        /* enumval.obj.reftype.reftype = -1; */ /* ???Philippe's original??? */
        enumval.obj.reftype.reftype = G__PARANORMAL; /* This may be correct */
        enumval.ref = 0;
#endif
	enumval.obj.i = -1;
	enumval.type = 'i' ;
	enumval.tagnum = G__tagnum ;
	enumval.typenum = -1 ;
#ifndef G__OLDIMPLEMENTATION1259
        enumval.isconst = 0;
#endif
	G__constvar=G__CONSTVAR;
	G__enumdef=1;
	do {
#ifndef G__OLDIMPLEMENTATION1382
	  int store_decl = 0 ;
#endif
	  c=G__fgetstream(memname,"=,}");
	  if(c=='=') {
#ifndef G__OLDIMPLEMENTATION888
	    char store_var_typeX=G__var_type;
	    int store_tagnumX=G__tagnum;
	    int store_def_tagnumX=G__def_tagnum;
	    G__var_type='p';
	    G__tagnum = G__def_tagnum = -1;
#endif
	    c=G__fgetstream(val,",}");
	    store_prerun=G__prerun;
	    G__prerun=0;
	    enumval=G__getexpr(val);
	    G__prerun=store_prerun;
#ifndef G__OLDIMPLEMENTATION888
	    G__var_type=store_var_typeX;
	    G__tagnum=store_tagnumX;
	    G__def_tagnum=store_def_tagnumX;
#endif
	  }
	  else {
	    enumval.obj.i++;
	  }
#ifndef G__OLDIMPLEMENTATION625
	  G__constvar=G__CONSTVAR;
	  G__enumdef=1;
#endif
	  G__var_type='i';
	  if(-1!=store_tagnum) {
	    store_def_struct_member=G__def_struct_member;
	    G__def_struct_member=0;
	    G__static_alloc=1;
#ifndef G__OLDIMPLEMENTATION1382
	    store_decl = G__decl;
	    G__decl = 1;
#endif
#ifndef G__OLDIMPLEMENTATION1885
	    G__access=store_access;
#endif
	  }
	  G__letvariable(memname,enumval,&G__global ,G__p_local);
	  if(-1!=store_tagnum) {
	    G__def_struct_member=store_def_struct_member;
	    G__static_alloc=0;
#ifndef G__OLDIMPLEMENTATION1382
	    G__decl = store_decl;
#endif
	  }
	} while(c!='}') ;
	G__constvar=0;
	G__enumdef=0;
	G__access=store_access;
      }
      
      else { /* class, struct or union */
	/********************************************
	 * Parsing member declaration
	 ********************************************/
	store_local = G__p_local;
	G__p_local=G__struct.memvar[G__tagnum];
	
	store_def_struct_member=G__def_struct_member;
	G__def_struct_member=1;
	G__switch = 0; /* redundant */
	mparen = G__mparen;
	G__mparen=0;
	store_static_alloc=G__static_alloc;
	G__static_alloc=0;
	store_prerun=G__prerun;
	G__prerun=1;
	tagdefining = G__tagdefining;
	G__tagdefining=G__tagnum;

	G__exec_statement(); /* parser body */

	G__tagnum=G__tagdefining;
	G__access=store_access;
	G__prerun=store_prerun;
	G__static_alloc=store_static_alloc;
	
	/********************************************
	 * Padding for PA-RISC, Spark, etc
	 * If struct size can not be divided by G__DOUBLEALLOC
	 * the size is aligned.
	 ********************************************/
#ifndef G__OLDIMPLEMENTATION1777
	if(1==G__struct.memvar[G__tagnum]->allvar
#ifndef G__OLDIMPLEMENTATION1807
	   && 0==G__struct.baseclass[G__tagnum]->basen
#endif
	   ) {
	  /* this is still questionable, inherit0.c */
	  struct G__var_array *v=G__struct.memvar[G__tagnum];
	  if('c'==v->type[0]) { 
	    if(isupper(v->type[0])) {
	      G__struct.size[G__tagnum] = G__LONGALLOC*(v->varlabel[0][1]+1);
	    }
	    else {
	      G__value buf;
	      buf.type = v->type[0];
	      buf.tagnum = v->p_tagtable[0];
	      buf.typenum = v->p_typetable[0];
	      G__struct.size[G__tagnum]
		=G__sizeof(&buf)*(v->varlabel[0][1]+1);
	    }
	  }
	} else
#endif
	if(G__struct.size[G__tagnum]%G__DOUBLEALLOC) {
	  G__struct.size[G__tagnum]
	    += G__DOUBLEALLOC - G__struct.size[G__tagnum]%G__DOUBLEALLOC;
	}
#ifndef G__OLDIMPLEMENTATION591
#ifndef G__OLDIMPLEMENTATION1804
	if(0==G__struct.size[G__tagnum]) {
	  G__struct.size[G__tagnum] = G__CHARALLOC;
	}
#else
	else if(0==G__struct.size[G__tagnum]) {
	  G__struct.size[G__tagnum] = G__CHARALLOC;
	}
#endif
#endif
	
	G__tagdefining = tagdefining;
	
	G__def_struct_member=store_def_struct_member;
	G__mparen=mparen;
	G__p_local = store_local;
      }
    }
    else { /* of newdecl */
      G__fgetc();
      c=G__fignorestream("}");
    }
  }
  
  
  /*
   * Now came to
   * [struct|union|enum]   tagname   { member }  item ;
   *                                           ^
   *                     OR
   * [struct|union|enum]             { member }  item ;
   *                                           ^
   *                     OR
   * [struct|union|enum]   tagname     item ;
   *                                   ^
   * item declaration
   */
  
  G__var_type = 'u';
  
  /* Need to think about this */
  if(type=='e') G__var_type='i';

  if(ispointer) G__var_type=toupper(G__var_type);

  if(G__return>G__RETURN_NORMAL) return;
  
  if('u'==type) { /* union */
    fpos_t pos;
    int linenum;
    fgetpos(G__ifile.fp,&pos);
    linenum=G__ifile.line_number;
    
    c = G__fgetstream(basename,";");
    if(basename[0]) {
      fsetpos(G__ifile.fp,&pos);
      G__ifile.line_number=linenum;
      if(G__dispsource) G__disp_mask=1000;
      G__define_var(G__tagnum,-1);
      G__disp_mask=0;
    }
#ifndef G__OLDIMPLEMENTATION419
    else if('\0'==G__struct.name[G__tagnum][0]) {
#else
    else {
#endif
      /* anonymous union */
      G__add_anonymousunion(G__tagnum,G__def_struct_member,store_def_tagnum);
    }
  }
#ifndef G__OLDIMPLEMENTATION612
  else if('n'==type) { /* namespace */
    /* no instance object for namespace, do nothing */
  }
#endif
  else { /* struct or class instance */
#ifndef G__OLDIMPLEMENTATION2075
    if(G__cintv6) G__bc_struct(G__tagnum);
#endif
#ifndef G__OLDIMPLEMENTATION1087
    if(G__check_semicolumn_after_classdef(isclassdef)) {
#ifndef G__OLDIMPLEMENTATION1195
      G__tagnum=store_tagnum;
      G__def_tagnum = store_def_tagnum;
#endif
      return;
    }
#endif
#ifndef G__OLDIMPLEMENTATION1991
    G__def_tagnum = store_def_tagnum;
#endif
    G__define_var(G__tagnum,-1);
  }
  
  G__tagnum=store_tagnum;
  G__def_tagnum = store_def_tagnum;

#ifdef G__DEBUG
  if(G__asm_dbg) {
    G__fprinterr(G__serr,"G__tagnum=%d G__def_tagnum=%d G__def_struct_member=%d\n"
	    ,G__tagnum,G__def_tagnum,G__def_struct_member);
    G__printlinenum();
  }
#endif

}

#ifndef G__OLDIMPLEMENTATION2030
/******************************************************************
 * G__callfunc0()
 ******************************************************************/
int G__callfunc0(result,ifunc,ifn,libp,p,funcmatch)
G__value *result;
struct G__ifunc_table *ifunc;
int ifn;
struct G__param* libp;
void* p;
int funcmatch;
{
  int stat=0;
  long store_struct_offset;
#ifndef G__OLDIMPLEMENTATION2056
  int store_asm_exec;
#endif

  if(!ifunc->hash[ifn] || !ifunc->pentry[ifn]) {
    /* The function is not defined or masked */
    *result = G__null;
    return(stat);
  }

  store_struct_offset = G__store_struct_offset;
  G__store_struct_offset = (long)p;
#ifndef G__OLDIMPLEMENTATION2056
  store_asm_exec = G__asm_exec;
  G__asm_exec = 0;
#endif

#ifdef G__EXCEPTIONWRAPPER
  if(-1==ifunc->pentry[ifn]->size) {
    /* compiled function. should be stub */
    G__InterfaceMethod pfunc = (G__InterfaceMethod)ifunc->pentry[ifn]->tp2f;
    stat=G__ExceptionWrapper(pfunc,result,(char*)NULL,libp,1);
  }
  else if(G__BYTECODE_SUCCESS==ifunc->pentry[ifn]->bytecodestatus) {
    /* bytecode function */
    struct G__bytecodefunc *pbc = ifunc->pentry[ifn]->bytecode;
    stat=G__ExceptionWrapper(G__exec_bytecode,result,(char*)pbc,libp,1);
  }
  else {
    /* interpreted function */
    /* stat=G__ExceptionWrapper(G__interpret_func,result,ifunc->funcname[ifn]
       ,libp,ifunc->hash[ifn]);  this was wrong! */
    stat=G__interpret_func(result,ifunc->funcname[ifn],libp,ifunc->hash[ifn]
			   ,ifunc,G__EXACT,funcmatch);
  }
#else
  if(-1==ifunc->pentry[ifn]->size) {
    /* compiled function. should be stub */
    G__InterfaceMethod pfunc = (G__InterfaceMethod)ifunc->pentry[ifn]->tp2f;
    stat=(*pfunc)(result,(char*)NULL,libp,1);
  }
  else if(G__BYTECODE_SUCCESS==ifunc->pentry[ifn]->bytecodestatus) {
    /* bytecode function */
    struct G__bytecodefunc *pbc = ifunc->pentry[ifn]->bytecode;
    stat=G__exec_bytecode(result,(char*)pbc,libp,1);
  }
  else {
    /* interpreted function */
    stat=G__interpret_func(result,ifunc->funcname[ifn],libp,ifunc->hash[ifn]
			   ,ifunc,G__EXACT,funcmatch);
  }
#endif

  G__store_struct_offset = store_struct_offset;
#ifndef G__OLDIMPLEMENTATION2056
  G__asm_exec = store_asm_exec;
#endif

  return(stat);
}

/******************************************************************
 * G__calldtor
 ******************************************************************/
int G__calldtor(p,tagnum,isheap)
void* p;
int tagnum;
int isheap;
{
  int stat;
  G__value result;
  struct G__ifunc_table *ifunc;
  struct G__param para;
  int ifn=0;
  long store_gvp;

  if(-1==tagnum) return(0);

  /* destructor must be the first function in the table. -> 2027 */
  ifunc = G__struct.memfunc[tagnum];

  store_gvp = G__getgvp();
  if(isheap) {
    /* p is deleted either with free() or delete */
    G__setgvp(G__PVOID);
  }
  else {
    /* In callfunc0, G__store_sturct_offset is also set to p.
     * Let G__operator_delete() return without doing anything */
    G__setgvp((long)p);
  }

  /* call destructor */
  para.paran=0;
  para.parameter[0][0]=0;
  para.para[0] = G__null;
  stat = G__callfunc0(&result,ifunc,ifn,&para,p,G__TRYDESTRUCTOR);

  G__setgvp(store_gvp);

#ifndef G__OLDIMPLEMENTATION2057
  if(isheap && -1!=ifunc->pentry[ifn]->size) {
    /* interpreted class */
    free((void*)p);
  }
#endif

  return(stat);
}
#endif

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
