/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file sizeof.c
 ************************************************************************
 * Description:
 *  Getting object size 
 ************************************************************************
 * Copyright(c) 1995~2002  Masaharu Goto 
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

/* array index for type_info. This must correspond to the class member 
* layout of class type_info in the <typeinfo.h> */
#define G__TYPEINFO_VIRTUALID 0
#define G__TYPEINFO_TYPE      1
#define G__TYPEINFO_TAGNUM    2
#define G__TYPEINFO_TYPENUM   3
#define G__TYPEINFO_REFTYPE   4
#define G__TYPEINFO_SIZE      5
#define G__TYPEINFO_ISCONST   6

#ifndef G__OLDIMPLEMENTATION849
int G__rootCcomment=0;
#endif

/******************************************************************
* int G__sizeof(G__value *object)
*
* Called by
*   G__bstore()    pointer operation
*   G__bstore()
*   G__bstore()
*   G__bstore()
*   G__bstore()
*
******************************************************************/
int G__sizeof(object)
G__value *object;
{
  if(isupper(object->type) && object->obj.reftype.reftype!=G__PARANORMAL) {
#ifdef G__OLDIMPLEMENTATION707
    switch(object->obj.reftype.reftype) {
    case G__PARANORMAL:
    case G__PARAP2P:
    case G__PARAP2P2P:
      break;
    default:
      G__fprinterr(G__serr,"Internal error: G__sizeof() illegal reftype ID %d\n"
	     ,object->obj.reftype.reftype);
      break;
    }
#endif
    return(G__LONGALLOC);
  }
  switch(toupper(object->type)) {
  case 'B':
  case 'C':
  case 'E': /* file ? */
  case 'Y': /* void */
#ifndef G__OLDIMPLEMENTATION2191
  case '1': /* pointer to function */
#else
  case 'Q': /* pointer to function */
#endif
    return(G__CHARALLOC);
  case 'R':
  case 'S':
    return(G__SHORTALLOC);
  case 'H':
  case 'I':
    return(G__INTALLOC);
  case 'K':
  case 'L':
    return(G__LONGALLOC);
  case 'F':
    return(G__FLOATALLOC);
  case 'D':
    return(G__DOUBLEALLOC);
  case 'U':
    return(G__struct.size[object->tagnum]);
  case 'A': /* pointer to function */
    return(G__P2MFALLOC);
#ifndef G__OLDIMPLEMENTATION1604
  case 'G': /* bool */
#ifdef G__BOOL4BYTE
    return(G__INTALLOC);
#else
    return(G__CHARALLOC);
#endif
#endif
#ifndef G__OLDIMPLEMENTATION2189
  case 'N':
  case 'M':
    return(G__LONGLONGALLOC);
#ifndef G__OLDIMPLEMENTATION2191
  case 'Q':
    return(G__LONGDOUBLEALLOC);
#endif
#endif
  }
  return(1);
}


/******************************************************************
* int G__Loffsetof()
*
******************************************************************/
int G__Loffsetof(tagname,memname)
char *tagname;
char *memname;
{
  int tagnum;
  struct G__var_array *var;
  int i,hash;

  tagnum = G__defined_tagname(tagname,0); /* G__TEMPLATECLASS case 7 */
  if(-1==tagnum) return(-1);

  G__hash(memname,hash,i)
  G__incsetup_memvar(tagnum);
  var = G__struct.memvar[tagnum];

  while(var) {
    for(i=0;i<var->allvar;i++) {
      if(hash==var->hash[i] && strcmp(memname,var->varnamebuf[i])==0) {
	return((int)var->p[i]);
      }
    }
    var=var->next;
  }

  G__fprinterr(G__serr,"Error: member %s not found in %s ",memname,tagname);
  G__genericerror((char*)NULL);
  return(-1);
}

/******************************************************************
* int G__Lsizeof(typename)
*
* Called by
*   G__special_func()
*
******************************************************************/
int G__Lsizeof(typename)
char *typename;
{
  int hash;
  int ig15;
  struct G__var_array *var;
  G__value buf;
  int tagnum,typenum;
  int result;
#ifndef G__OLDIMPLEMENTATION406
  int pointlevel=0;
  char namebody[G__MAXNAME+20];
  char *p;
  int i;
  int pinc;
#endif


  /* return size of pointer if xxx* */
  if('*'==typename[strlen(typename)-1]) {
    return(sizeof(void *));
  }

  /* searching for struct/union tagtable */
  if((strncmp(typename,"struct",6)==0)
#ifndef G__OLDIMPLEMENTATION816
     || strncmp(typename,"signed",6)==0
#endif
     ) {
    typename = typename+6;
  }
  else if((strncmp(typename,"class",5)==0)) {
    typename = typename+5;
  }
  else if((strncmp(typename,"union",5)==0)) {
    typename = typename+5;
  }

  tagnum = G__defined_tagname(typename,1); /* case 8) */
  if(-1 != tagnum) {
#ifndef G__OLDIMPLEMENTATION1389
    if('e'!=G__struct.type[tagnum]) return(G__struct.size[tagnum]);
    else                            return(G__INTALLOC);
#else
    return(G__struct.size[tagnum]);
#endif
  }

  typenum = G__defined_typename(typename);
  if(-1 != typenum) {
    switch(G__newtype.type[typenum]) {
#ifndef G__OLDIMPLEMENTATION2189
    case 'n':
    case 'm':
      result = sizeof(G__int64);
      break;
    case 'q':
      result = sizeof(long double);
      break;
#endif
#ifndef G__OLDIMPLEMENTATION1604
    case 'g':
#ifdef G__BOOL4BYTE
      result = sizeof(int);
      break;
#endif
#endif
    case 'b':
    case 'c':
      result = sizeof(char);
      break;
    case 'h':
    case 'i':
      result = sizeof(int);
      break;
    case 'r':
    case 's':
      result = sizeof(short);
      break;
    case 'k':
    case 'l':
      result = sizeof(long);
      break;
    case 'f':
      result = sizeof(float);
      break;
    case 'd':
      result = sizeof(double);
      break;
    case 'v':
      return(-1);
    default:
      /* if struct or union */
      if(isupper(G__newtype.type[typenum])) {
	result = sizeof(void *);
      }
      else if(G__newtype.tagnum[typenum]>=0) {
	result = G__struct.size[G__newtype.tagnum[typenum]];
      }
      else {
	return(0);
      }
      break;
    }
    if(G__newtype.nindex[typenum]) {
      for(ig15=0;ig15<G__newtype.nindex[typenum];ig15++) {
	result *= G__newtype.index[typenum][ig15] ;
      }
    }
    return(result);
  }

  if((strcmp(typename,"int")==0)||
     (strcmp(typename,"unsignedint")==0))
    return(sizeof(int));
  if((strcmp(typename,"long")==0)||
     (strcmp(typename,"longint")==0)||
     (strcmp(typename,"unsignedlong")==0)||
     (strcmp(typename,"unsignedlongint")==0))
    return(sizeof(long));
  if((strcmp(typename,"short")==0)||
     (strcmp(typename,"shortint")==0)||
     (strcmp(typename,"unsignedshort")==0)||
     (strcmp(typename,"unsignedshortint")==0))
    return(sizeof(short));
  if((strcmp(typename,"char")==0)||
     (strcmp(typename,"unsignedchar")==0))
    return(sizeof(char));
  if((strcmp(typename,"float")==0)||
     (strcmp(typename,"float")==0))
    return(sizeof(float));
  if((strcmp(typename,"double")==0)
#ifdef G__OLDIMPLEMENTATION1533
     ||(strcmp(typename,"longdouble")==0)
#endif
     )
    return(sizeof(double));
#ifndef G__OLDIMPLEMENTATION1533
  if(strcmp(typename,"longdouble")==0) {
    int tagnum,typenum;
    G__loadlonglong(&tagnum,&typenum,G__LONGDOUBLE);
    return(G__struct.size[tagnum]);
  }
#endif
#ifndef G__OLDIMPLEMENTATION1827
  if(strcmp(typename,"longlong")==0
     || strcmp(typename,"longlongint")==0
     ) {
    int tagnum,typenum;
    G__loadlonglong(&tagnum,&typenum,G__LONGLONG);
    return(G__struct.size[tagnum]);
  }
#endif
#ifndef G__OLDIMPLEMENTATION1838
  if(strcmp(typename,"unsignedlonglong")==0
     || strcmp(typename,"unsignedlonglongint")==0
     ) {
    int tagnum,typenum;
    G__loadlonglong(&tagnum,&typenum,G__ULONGLONG);
    return(G__struct.size[tagnum]);
  }
#endif
  if(strcmp(typename,"void")==0)
#ifndef G__OLDIMPLEMENTATION930
    return(sizeof(void*));
#else
    return(-1);
#endif
  if(strcmp(typename,"FILE")==0)
    return(sizeof(FILE));
#ifndef G__OLDIMPLEMENTATION1604
#ifdef G__BOOL4BYTE
  if(strcmp(typename,"bool")==0) return(sizeof(int));
#else
  if(strcmp(typename,"bool")==0) return(sizeof(unsigned char));
#endif
#endif

#ifndef G__OLDIMPLEMENTATION406
  while('*'==typename[pointlevel]) ++pointlevel;
  strcpy(namebody,typename+pointlevel);
  while((char*)NULL!=(p=strrchr(namebody,'['))) {
    *p='\0';
    ++pointlevel;
  }
  G__hash(namebody,hash,ig15)
  var = G__getvarentry(namebody,hash,&ig15,&G__global,G__p_local);
#else
  G__hash(typename,hash,ig15)
  var = G__getvarentry(typename,hash,&ig15,&G__global,G__p_local);
#endif
#ifndef G__OLDIMPLEMENTATION1948
  if(!var) {
    char temp[G__ONELINE];
    if(-1!=G__memberfunc_tagnum) /* questionable */
      sprintf(temp,"%s\\%x\\%x\\%x",namebody,G__func_page,G__func_now
	      ,G__memberfunc_tagnum);
    else
      sprintf(temp,"%s\\%x\\%x" ,namebody,G__func_page,G__func_now);
    
    G__hash(temp,hash,i)
    var = G__getvarentry(temp,hash,&ig15,&G__global,G__p_local);
  }
#endif
  if(var) {
    if(INT_MAX==var->varlabel[ig15][1]) {
#ifndef G__OLDIMPLEMENTATION1634
      if('c'==var->type[ig15]) return(strlen((char*)var->p[ig15]));
      else return(sizeof(void *));
#else
      return(sizeof(void *));
#endif
    }
    buf.type=var->type[ig15];
    buf.tagnum = var->p_tagtable[ig15];
    buf.typenum = var->p_typetable[ig15];
#ifndef G__OLDIMPLEMENTATION908
    if(isupper(buf.type)) buf.obj.reftype.reftype=var->reftype[ig15];
#endif
    if(pointlevel<=var->paran[ig15]) {
      switch(pointlevel) {
      case 0: 
	pinc=var->varlabel[ig15][1]+1; 
	break;
      case 1:
	pinc=var->varlabel[ig15][0]; 
	break;
      default:
	pinc=var->varlabel[ig15][0]; 
	for(i=1;i<pointlevel;i++) pinc/=var->varlabel[ig15][i+1];
	break;
      }
    }
    else {
#ifndef G__OLDIMPLEMENTATION908
      switch(pointlevel) {
      case 0: break;
      case 1:
	if(G__PARANORMAL==buf.obj.reftype.reftype) buf.type=tolower(buf.type);
	else if(G__PARAP2P==buf.obj.reftype.reftype) {
	  buf.obj.reftype.reftype=G__PARANORMAL;
	}
	else --buf.obj.reftype.reftype;
	break;
      case 2:
	if(G__PARANORMAL==buf.obj.reftype.reftype) buf.type=tolower(buf.type);
	else if(G__PARAP2P==buf.obj.reftype.reftype) {
	  buf.type=tolower(buf.type);
	  buf.obj.reftype.reftype=G__PARANORMAL;
	}
	else if(G__PARAP2P2P==buf.obj.reftype.reftype) {
	  buf.obj.reftype.reftype=G__PARANORMAL;
	}
	else buf.obj.reftype.reftype-=2;
	break;
      }
#endif
      return(G__sizeof(&buf));
    }
    if(isupper(var->type[ig15])) return(pinc*sizeof(void *));
    return(pinc*G__sizeof(&buf));
  }

#ifndef G__OLDIMPLEMENTATION649
  buf = G__getexpr(typename);
  if(buf.type) {
#ifndef G__OLDIMPLEMENTATION1637
    if('C'==buf.type && '"'==typename[0]) return(strlen((char*)buf.obj.i)+1);
#endif
    return(G__sizeof(&buf));
  }
#endif

  return(-1);

}

#ifdef G__TYPEINFO
/******************************************************************
* int G__typeid(typename)
*
* Called by
*   G__special_func()
*
******************************************************************/
long *G__typeid(typenamein)
char *typenamein;
{
  G__value buf;
  int c;
  long *type_info;
  int tagnum,typenum,type=0,reftype=G__PARANORMAL,size=0;
  int len;
  int pointlevel=0,isref=0;
  int tag_type_info;
  char typenamebuf[G__MAXNAME*2];
  char *typename;
#ifndef G__OLDIMPLEMENTATION1895
  int isconst = 0;
#endif

  /**********************************************************************
  * Get type_info tagname 
  ***********************************************************************/
  tag_type_info = G__defined_tagname("type_info",1);
  if(-1==tag_type_info) {
    G__genericerror("Error: class type_info not defined. <typeinfo.h> must be included");
    return((long*)NULL);
  }

  /**********************************************************************
  * In case of typeid(X&) , typeid(X*) , strip & or *
  ***********************************************************************/
  strcpy(typenamebuf,typenamein);
  typename=typenamebuf;
  len=strlen(typename);

  while('*'==(c=typename[len-1]) || '&'==c) {
    switch(c) {
    case '*':
      ++pointlevel;
      break;
    case '&':
      isref=1;
      break;
    }
    --len;
    typename[len]='\0';
  }

  /**********************************************************************
  * Search for typedef names
  **********************************************************************/
#ifndef G__OLDIMPLEMENTATION1838
  if(strcmp(typename,"longlong")==0) {
    strcpy(typename,"G__longlong");
  }
  else if(strcmp(typename,"unsignedlonglong")==0) {
    strcpy(typename,"G__ulonglong");
  }
  else if(strcmp(typename,"longdouble")==0) {
    strcpy(typename,"G__longdouble");
  }
#endif
  typenum = G__defined_typename(typename);
  if(-1 != typenum) {
    type    = G__newtype.type[typenum];
    tagnum  = G__newtype.tagnum[typenum];
    reftype = G__newtype.reftype[typenum];
    if(-1!=tagnum) {
      size = G__struct.size[tagnum];
    }
    else {
      switch(tolower(type)) {
#ifndef G__OLDIMPLEMENTATION2189
      case 'n':
      case 'm':
	size = G__LONGLONGALLOC;
	break;
#if 0
      case 'q':
	size = G__LONGDOUBLEALLOC;
	break;
#endif
#endif
#ifndef G__OLDIMPLEMENTATION1604
      case 'g':
#ifdef G__BOOL4BYTE
	size = G__INTALLOC;
	break;
#endif
#endif
      case 'b':
      case 'c':
	size = G__CHARALLOC;
	break;
      case 'r':
      case 's':
	size = G__SHORTALLOC;
	break;
      case 'h':
      case 'i':
	size = G__INTALLOC;
	break;
      case 'k':
      case 'l':
	size = G__LONGALLOC;
	break;
      case 'f':
	size = G__FLOATALLOC;
	break;
      case 'd':
	size = G__DOUBLEALLOC;
	break;
      case 'e':
      case 'y':
	size = -1;
	break;
      case 'a':
	size = G__sizep2memfunc;
	break;
      case 'q':
	break;
      }
    }
  }

  else {

    /*********************************************************************
     * Search for class/struct/union names
     *********************************************************************/
    if((strncmp(typename,"struct",6)==0)) {
      typename = typename+6;
    }
    else if((strncmp(typename,"class",5)==0)) {
      typename = typename+5;
    }
    else if((strncmp(typename,"union",5)==0)) {
      typename = typename+5;
    }
    
    tagnum = G__defined_tagname(typename,1); 
    if(-1 != tagnum) {
      reftype=G__PARANORMAL;
      switch(G__struct.type[tagnum]) {
      case 'u':
      case 's':
      case 'c':
	type = 'u';
	size = G__struct.size[tagnum];
	break;
      case 'e':
	type = 'i';
	size = G__INTALLOC;
	break;
#ifndef G__OLDIMPLEMENTATION612
      case 'n':
	size = G__struct.size[tagnum];
	G__genericerror("Error: can not get sizeof namespace");
	break;
#endif
      }
    }

    else {

      /********************************************************************
       * Search for intrinsic types
       *******************************************************************/
      reftype = G__PARANORMAL;

      if(strcmp(typename,"int")==0) {
        type = 'i';
	size = G__INTALLOC;
      }
      if(strcmp(typename,"unsignedint")==0) {
	type = 'h';
	size = G__INTALLOC;
      }
      if((strcmp(typename,"long")==0)||
	 (strcmp(typename,"longint")==0)) {
	type='l';
	size = G__LONGALLOC;
      }
      if((strcmp(typename,"unsignedlong")==0)||
	 (strcmp(typename,"unsignedlongint")==0)) {
	type = 'k';
	size = G__LONGALLOC;
      }
      if((strcmp(typename,"short")==0)||
	 (strcmp(typename,"shortint")==0)) {
	type = 's';
	size = G__SHORTALLOC;
      }
      if((strcmp(typename,"unsignedshort")==0)||
	 (strcmp(typename,"unsignedshortint")==0)) {
	type = 'r';
	size = G__SHORTALLOC;
      }
      if((strcmp(typename,"char")==0)||
	 (strcmp(typename,"signedchar")==0)) {
	type = 'c';
	size = G__CHARALLOC;
      }
      if(strcmp(typename,"unsignedchar")==0) {
	type = 'b';
	size = G__CHARALLOC;
      }
      if(strcmp(typename,"float")==0) {
	type = 's'; 
	size = G__FLOATALLOC;
      }
      if((strcmp(typename,"double")==0)
#ifdef G__OLDIMPLEMENTATION1838
	 ||(strcmp(typename,"longdouble")==0)
#endif
	 ) {
	type = 'd';
	size = G__DOUBLEALLOC;
      }
      if(strcmp(typename,"void")==0) {
        type = 'y';
#ifndef G__OLDIMPLEMENTATION930
	size = sizeof(void*);
#else
	size = -1;
#endif
      }
      if(strcmp(typename,"FILE")==0) {
        type = 'e';
	size = -1;
      }
    }
  }

  /**********************************************************************
   * If no type name matches, evaluate the expression and get the type 
   * information of the object
   *********************************************************************/
  if(0==type) {
    buf = G__getexpr(typenamein);
    type = buf.type;
    tagnum = buf.tagnum;
    typenum = buf.typenum;
    isref = 0;
#ifndef G__OLDIMPLEMENTATION1895
    isconst = buf.isconst;
#endif

    if(-1!=tagnum && 'u'==tolower(type) && buf.ref && -1!=G__struct.virtual_offset[tagnum]) {
      /* In case of polymorphic object, get the actual tagnum from the hidden
       * virtual identity field.  */
      tagnum = *(long*)(buf.obj.i+G__struct.virtual_offset[tagnum]);
    }
  }

  /*********************************************************************
   * Identify reference and pointer level
   *********************************************************************/
  if(isref) {
    reftype = G__PARAREFERENCE;
    if(pointlevel) type = toupper(type);
  }
  else {
    if(isupper(type)) {
      ++pointlevel;
      type = tolower(type);
    }
    switch(pointlevel) {
    case 0:
      reftype = G__PARANORMAL;
      break;
    case 1:
      type = toupper(type);
      reftype = G__PARANORMAL;
      break;
    case 2:
      type = toupper(type);
      reftype = G__PARAP2P;
      break;
    case 3:
      type = toupper(type);
      reftype = G__PARAP2P2P;
      break;
    }
  }

  if(isupper(type)) size = G__LONGALLOC;

  /**********************************************************************
   * Create temporary object for return value and copy the reslut
   **********************************************************************/
  G__alloc_tempobject(tag_type_info, -1 );
  type_info = (long*)G__p_tempbuf->obj.obj.i;

  type_info[G__TYPEINFO_VIRTUALID] = tag_type_info;
  type_info[G__TYPEINFO_TYPE] = type;
  type_info[G__TYPEINFO_TAGNUM] = tagnum;
  type_info[G__TYPEINFO_TYPENUM] = typenum;
  type_info[G__TYPEINFO_REFTYPE] = reftype;
  type_info[G__TYPEINFO_SIZE] = size;
#ifndef G__OLDIMPLEMENTATION1895
  type_info[G__TYPEINFO_ISCONST] = isconst;
#endif

  return( type_info ) ;

}
#endif


#ifdef G__FONS_TYPEINFO
/******************************************************************
* G__getcomment()
*
******************************************************************/
void G__getcomment(buf,pcomment,tagnum)
char *buf;
struct G__comment_info *pcomment;
int tagnum;
{
  fpos_t pos,store_pos;
  FILE *fp;
  int filenum;
  char *p;
#ifndef G__OLDIMPLEMENTATION469
  int flag=1;
#endif

#ifndef G__OLDIMPLEMENTATION469
  if(-1!=pcomment->filenum) {
#else
  if( pcomment->p.pos ) {
#endif
    if(-1!=tagnum && G__NOLINK==G__struct.iscpplink[tagnum] &&
       pcomment->filenum>=0) {
      pos = pcomment->p.pos;
      filenum = pcomment->filenum;
#ifndef G__OLDIMPLEMENTATION1100
      if(filenum==G__MAXFILE) fp = G__mfp;
      else                    fp = G__srcfile[filenum].fp;
#else
      fp = G__srcfile[filenum].fp;
#endif
      if((FILE*)NULL==fp) {
#ifndef G__PHILIPPE0
	/* Open the right file even in case where we use the preprocessor */
	if ( 
#ifndef G__OLDIMPLEMENTATION1100
	    filenum<G__MAXFILE &&
#endif
	    G__srcfile[filenum].prepname ) {
	  fp = fopen(G__srcfile[filenum].prepname,"r");
	} else {
	  fp = fopen(G__srcfile[filenum].filename,"r");
	}
#else
	fp = fopen(G__srcfile[filenum].filename,"r");
#endif
#ifndef G__OLDIMPLEMENTATION469
	flag=0;
#else
	store_pos = 0;
#endif
      }
      else {
	fgetpos(fp,&store_pos);
      }
      fsetpos(fp,&pos);
      fgets(buf,G__ONELINE-1,fp);
      p = strchr(buf,'\n');
      if(p) *p = '\0';
      p = strchr(buf,'\r');
      if(p) *p = '\0';
#ifndef G__OLDIMPLEMENTATION849
      if(G__rootCcomment) {
	p = G__strrstr(buf,"*/");
	if(p) *p = '\0';
      }
#endif

#ifndef G__OLDIMPLEMENTATION469
      if(flag) {
#else
      if(store_pos) {
#endif
	fsetpos(fp,&store_pos);
      }
      else {
	fclose(fp);
      }
    }
    else if(-2==pcomment->filenum) {
      strcpy(buf,pcomment->p.com);
    }
    else {
      buf[0]='\0';
    }
  }
  else {
    buf[0]='\0';
  }
  return;
}

/******************************************************************
* G__getcommenttypedef()
*
******************************************************************/
void G__getcommenttypedef(buf,pcomment,typenum)
char *buf;
struct G__comment_info *pcomment;
int typenum;
{
  fpos_t pos,store_pos;
  FILE *fp;
  int filenum;
  char *p;
#ifndef G__OLDIMPLEMENTATION469
  int flag=1;
#endif

#ifndef G__OLDIMPLEMENTATION469
  if(-1!=typenum && -1!=pcomment->filenum) {
#else
  if(-1!=typenum && pcomment->p.pos ) {
#endif
    if(G__NOLINK==G__newtype.iscpplink[typenum] && pcomment->filenum>=0) {
      pos = pcomment->p.pos;
      filenum = pcomment->filenum;
#ifndef G__OLDIMPLEMENTATION1100
      if(filenum==G__MAXFILE) fp = G__mfp;
      else                    fp = G__srcfile[filenum].fp;
#else
      fp = G__srcfile[filenum].fp;
#endif
      if((FILE*)NULL==fp) {
#ifndef G__PHILIPPE0
	/* Open the right file even in case where we use the preprocessor */
	if ( 
#ifndef G__OLDIMPLEMENTATION1100
	    filenum<G__MAXFILE &&
#endif
	    G__srcfile[filenum].prepname ) {
	  fp = fopen(G__srcfile[filenum].prepname,"r");
	} else {
	  fp = fopen(G__srcfile[filenum].filename,"r");
	}
#else
	fp = fopen(G__srcfile[filenum].filename,"r");
#endif
#ifndef G__OLDIMPLEMENTATION469
	flag=0;
#else
	store_pos = 0;
#endif
      }
      else {
	fgetpos(fp,&store_pos);
      }
      fsetpos(fp,&pos);
      fgets(buf,G__ONELINE-1,fp);
      p = strchr(buf,'\n');
      if(p) *p = '\0';
      p = strchr(buf,'\r');
      if(p) *p = '\0';
#ifndef G__OLDIMPLEMENTATION1858
      p = strchr(buf,';');
      if(p) *(p+1) = '\0';
#endif

#ifndef G__OLDIMPLEMENTATION469
      if(flag) {
#else
      if(store_pos) {
#endif
	fsetpos(fp,&store_pos);
      }
      else {
	fclose(fp);
      }
    }
    else if(-2==pcomment->filenum) {
      strcpy(buf,pcomment->p.com);
    }
    else {
      buf[0]='\0';
    }
  }
  else {
    buf[0]='\0';
  }
  return;
}


/******************************************************************
* long G__get_classinfo()
*
* Called by
*   G__special_func()
*
******************************************************************/
long G__get_classinfo(item,tagnum)
char *item;
int tagnum;
{
  char *buf;
  int tag_string_buf;
  struct G__inheritance *baseclass;
  int p;
  int i;

  /**********************************************************************
   * get next class/struct
   **********************************************************************/
  if(strcmp("next",item)==0) {
    while(1) {
      ++tagnum;
      if(tagnum<0 || G__struct.alltag<=tagnum) return(-1);
      if(('s'==G__struct.type[tagnum]||'c'==G__struct.type[tagnum])&&
	 -1==G__struct.parent_tagnum[tagnum]) {
	return((long)tagnum);
      }
    }
  }

  /**********************************************************************
   * check validity
   **********************************************************************/
  if(tagnum<0 || G__struct.alltag<=tagnum || 
     ('c'!=G__struct.type[tagnum] && 's'!=G__struct.type[tagnum])) 
    return(0);

  /**********************************************************************
   * return type
   **********************************************************************/
  if(strcmp("type",item)==0) {
    switch(G__struct.type[tagnum]) {
    case 'e':
      return((long)'i');
    default:
      return((long)'u');
    }
  }

  /**********************************************************************
   * size
   **********************************************************************/
  if(strcmp("size",item)==0) {
    return(G__struct.size[tagnum]);
  }

  /**********************************************************************
   * baseclass
   **********************************************************************/
  if(strcmp("baseclass",item)==0) {
    tag_string_buf = G__defined_tagname("G__string_buf",0);
    G__alloc_tempobject(tag_string_buf, -1 );
    buf = (char*)G__p_tempbuf->obj.obj.i;

    baseclass = G__struct.baseclass[tagnum];
    if(!baseclass) return((long)NULL);
    p=0;
    buf[0]='\0';
    for(i=0;i<baseclass->basen;i++) {
      if(baseclass->property[i]&G__ISDIRECTINHERIT) {
	if(p) {
	  sprintf(buf+p,",");
	  ++p;
	}
	sprintf(buf+p,"%s%s" ,G__access2string(baseclass->baseaccess[i])
		,G__struct.name[baseclass->basetagnum[i]]);
	p=strlen(buf);
      }
    }

    return((long)buf);
  }

  /**********************************************************************
   * title
   **********************************************************************/
  if(strcmp("title",item)==0) {
    tag_string_buf = G__defined_tagname("G__string_buf",0);
    G__alloc_tempobject(tag_string_buf, -1 );
    buf = (char*)G__p_tempbuf->obj.obj.i;

    G__getcomment(buf,&G__struct.comment[tagnum],tagnum);
    return((long)buf);
  }

  /**********************************************************************
   * isabstract
   **********************************************************************/
  if(strcmp("isabstract",item)==0) {
    return(G__struct.isabstract[tagnum]);
  }
  return(0);
}

/******************************************************************
* long G__get_variableinfo()
*
* Called by
*   G__special_func()
*
******************************************************************/
long G__get_variableinfo(item,phandle,pindex,tagnum)
char *item;
long *phandle;
long *pindex;
int tagnum;
{
  char *buf;
  int tag_string_buf;
  struct G__var_array *var;
  int index;

  /*******************************************************************
  * new
  *******************************************************************/
  if(strcmp("new",item)==0) {
    *pindex = 0;
    if(-1==tagnum) {
      *phandle = (long)(&G__global);
    }
    else if(G__struct.memvar[tagnum]) {
      G__incsetup_memvar(tagnum);
      *phandle = (long)(G__struct.memvar[tagnum]);
    }
    else {
      *phandle = 0;
    }
    return(0);
  }

  var = (struct G__var_array *)(*phandle);
  index = (*pindex);

  if((struct G__var_array*)NULL==var || var->allvar<=index) {
    *phandle = 0;
    *pindex = 0;
    return(0);
  }


  /*******************************************************************
  * next
  *******************************************************************/
  if(strcmp("next",item)==0) {
    *pindex = index + 1;
    if((*pindex)>=var->allvar) {
      (*phandle) = (long)(var->next);
      *pindex = 0;
    }
    var = (struct G__var_array *)(*phandle);
    index = (*pindex);
    if(var && index<var->allvar) return(1);
    else {
      *phandle = 0;
      return(0);
    }
  }

  /*******************************************************************
  * name
  *******************************************************************/
  if(strcmp("name",item)==0) {
    return((long)var->varnamebuf[index]);
  }

  /*******************************************************************
  * type
  *******************************************************************/
  if(strcmp("type",item)==0) {
    tag_string_buf = G__defined_tagname("G__string_buf",0);
    G__alloc_tempobject(tag_string_buf, -1 );
    buf = (char*)G__p_tempbuf->obj.obj.i;
    strcpy(buf,G__type2string(var->type[index] 
			      ,var->p_tagtable[index]
			      ,var->p_typetable[index] 
			      ,var->reftype[index],0));
    return((long)buf);
  }

  /*******************************************************************
  * offset
  *******************************************************************/
  if(strcmp("offset",item)==0) {
    return(var->p[index]);
  }

  /*******************************************************************
  * title
  *******************************************************************/
  if(strcmp("title",item)==0) {
    if(-1!=tagnum) {
      tag_string_buf = G__defined_tagname("G__string_buf",0);
      G__alloc_tempobject(tag_string_buf, -1 );
      buf = (char*)G__p_tempbuf->obj.obj.i;
      G__getcomment(buf,&var->comment[index],tagnum);
      return((long)buf);
    }
    else {
      G__genericerror("Error: title only supported for class/struct member");
      return((long)NULL);
    }
  }
  return(0);
}

/******************************************************************
* long G__get_functioninfo()
*
* Called by
*   G__special_func()
*
******************************************************************/
long G__get_functioninfo(item,phandle,pindex,tagnum)
char *item;
long *phandle;
long *pindex;
int tagnum;
{
  char *buf;
  int tag_string_buf;
  /* char temp[G__MAXNAME]; */
  struct G__ifunc_table *ifunc;
  int index;
  int i;
  int p;

  /*******************************************************************
  * new
  *******************************************************************/
  if(strcmp("new",item)==0) {
    *pindex = 0;
    if(-1==tagnum) {
      *phandle = (long)(&G__ifunc);
    }
    else if(G__struct.memfunc[tagnum]) {
      G__incsetup_memfunc(tagnum);
      *phandle = (long)(G__struct.memfunc[tagnum]);
    }
    else {
      *phandle = 0;
    }
    return(0);
  }

  ifunc = (struct G__ifunc_table *)(*phandle);
  index = (*pindex);

  if((struct G__ifunc_table*)NULL==ifunc || ifunc->allifunc<=index) {
    *phandle = 0;
    *pindex = 0;
    return(0);
  }


  /*******************************************************************
  * next
  *******************************************************************/
  if(strcmp("next",item)==0) {
    *pindex = index + 1;
    if((*pindex)>=ifunc->allifunc) {
      (*phandle) = (long)(ifunc->next);
      *pindex = 0;
    }
    ifunc = (struct G__ifunc_table *)(*phandle);
    index = (*pindex);
    if( ifunc && index<ifunc->allifunc) return(1);
    else {
      *phandle = 0;
      return(0);
    }
  }

  /*******************************************************************
  * name
  *******************************************************************/
  if(strcmp("name",item)==0) {
    return((long)ifunc->funcname[index]);
  }

  /*******************************************************************
  * type
  *******************************************************************/
  if(strcmp("type",item)==0) {
    tag_string_buf = G__defined_tagname("G__string_buf",0);
    G__alloc_tempobject(tag_string_buf, -1 );
    buf = (char*)G__p_tempbuf->obj.obj.i;
    strcpy(buf,G__type2string(ifunc->type[index] 
			      ,ifunc->p_tagtable[index]
			      ,ifunc->p_typetable[index] 
			      ,ifunc->reftype[index],0));
    return((long)buf);
  }

  /*******************************************************************
  * arglist
  *******************************************************************/
  if(strcmp("arglist",item)==0) {
    tag_string_buf = G__defined_tagname("G__string_buf",0);
    G__alloc_tempobject(tag_string_buf, -1 );
    buf = (char*)G__p_tempbuf->obj.obj.i;

    buf[0]='\0';
    p=0;
    for(i=0;i<ifunc->para_nu[index];i++) {
      if(p) {
	sprintf(buf+p,",");
	++p;
      }
      sprintf(buf+p,"%s",G__type2string(ifunc->para_type[index][i]
					,ifunc->para_p_tagtable[index][i]
					,ifunc->para_p_typetable[index][i]
					,ifunc->para_reftype[index][i],0));
      p=strlen(buf);
      if(ifunc->para_default[index][i]) {
	sprintf(buf+p,"=");
		/* ,G__valuemonitor(*ifunc->para_default[index][i],temp)); */
      }
      p=strlen(buf);
    }
    return((long)buf);
  }

  /*******************************************************************
  * title
  *******************************************************************/
  if(strcmp("title",item)==0) {
    if(-1!=tagnum) {
      tag_string_buf = G__defined_tagname("G__string_buf",0);
      G__alloc_tempobject(tag_string_buf, -1 );
      buf = (char*)G__p_tempbuf->obj.obj.i;
      G__getcomment(buf,&ifunc->comment[index],tagnum);
      return((long)buf);
    }
    else {
      G__genericerror("Error: title only supported for class/struct member");
      return((long)NULL);
    }
  }
  return(0);
}
#endif

#ifndef G__OLDIMPLEMENTATION1473
/**************************************************************************
 * G__va_arg_setalign()
 **************************************************************************/
#ifdef G__VAARG_INC_COPY_N
static int G__va_arg_align_size=G__VAARG_INC_COPY_N;
#else
static int G__va_arg_align_size=0;
#endif
void G__va_arg_setalign(n)
int n;
{
  G__va_arg_align_size = n;
}

/**************************************************************************
 * G__va_arg_copyvalue()
 **************************************************************************/
void G__va_arg_copyvalue(t,p,pval,objsize)
int t;
void* p;
G__value *pval;
int objsize;
{
#ifndef G__OLDIMPLEMENTATION1696
#ifdef G__VAARG_PASS_BY_REFERENCE
  if(objsize>G__VAARG_PASS_BY_REFERENCE) {
    if(pval->ref>0x1000) *(long*)(p) = pval->ref;
    else *(long*)(p) = (long)G__int(*pval);
    return;
  }

#endif
#endif
  switch(t) {
#ifndef G__OLDIMPLEMENTATION2189
  case 'n':
  case 'm':
    *(G__int64*)(p) = (G__int64)G__Longlong(*pval);
    break;
#endif
#ifndef G__OLDIMPLEMENTATION1604
  case 'g':
#ifdef G__BOOL4BYTE
    *(int*)(p) = (int)G__int(*pval);
    break;
#endif
#endif
  case 'c':
  case 'b':
#if defined(__GNUC__)
    *(int*)(p) = (int)G__int(*pval);
#else
    *(char*)(p) = (char)G__int(*pval);
#endif
    break;
  case 'r':
  case 's':
#if defined(__GNUC__)
    *(int*)(p) = (int)G__int(*pval);
#else
    *(short*)(p) = (short)G__int(*pval);
#endif
    break;
  case 'h':
  case 'i':
    *(int*)(p) = (int)G__int(*pval);
    break;
  case 'k':
  case 'l':
    *(long*)(p) = (long)G__int(*pval);
    break;
  case 'f':
    *(float*)(p) = (float)G__double(*pval);
    break;
  case 'd':
    *(double*)(p) = (double)G__double(*pval);
    break;
  case 'u':
    memcpy((void*)(p),(void*)pval->obj.i,objsize);
    break;
  default:
    *(long*)(p) = (long)G__int(*pval);
    break;
  }
}

#if (defined(__PPC__)||defined(__ppc__))&&(defined(_AIX)||defined(__APPLE__))
#define G__alignof_ppc(objsize)  (objsize>4?16:4)
#define G__va_rounded_size_ppc(typesize) ((typesize + 3) & ~3)
#define G__va_align_ppc(AP, objsize)					   \
     ((((unsigned long)(AP)) + ((G__alignof_ppc(objsize) == 16) ? 15 : 3)) \
      & ~((G__alignof_ppc(objsize) == 16) ? 15 : 3))

#elif (defined(__PPC__)||defined(__ppc__))&&(defined(__linux)||defined(__linux__))

#endif

/**************************************************************************
 * G__va_arg_put()
 **************************************************************************/
void G__va_arg_put(pbuf,libp,n)
G__va_arg_buf* pbuf;
struct G__param *libp;
int n;
{
  int objsize;
  int type;
  int i;
#if defined(__hpux) || defined(__hppa__)
  int j2=G__VAARG_SIZE;
#endif
  int j=0;
  int mod;
#ifdef G__VAARG_NOSUPPORT
  G__genericerror("Limitation: Variable argument is not supported for this platform");
#endif
  for(i=n;i<libp->paran;i++) {
    type = libp->para[i].type;
    if(isupper(type)) objsize = G__LONGALLOC;
    else              objsize = G__sizeof(&libp->para[i]);

    /* Platform that decrements address */
#if (defined(__linux)&&defined(__i386))||defined(_WIN32)||defined(G__CYGWIN)
    /* nothing */
#elif defined(__hpux) || defined(__hppa__)
    if(objsize > G__VAARG_PASS_BY_REFERENCE) {
      j2 = j2 - sizeof(long);
      j=j2;
    }
    else {
      j2 = (j2 - objsize) & (objsize > 4 ? 0xfffffff8 : 0xfffffffc );
      j = j2 + ((8 - objsize) % 4);
    }
#elif defined(__sparc) || defined(__sparc__) || defined(__SUNPRO_C)
    /* nothing */

#elif (defined(__PPC__)||defined(__ppc__))&&(defined(_AIX)||defined(__APPLE__))
    /* nothing */
#elif (defined(__PPC__)||defined(__ppc__))&&(defined(__linux)||defined(__linux__))
    /* nothing */
#else
    /* nothing */
#endif
    
    G__va_arg_copyvalue(type,(void*)(&pbuf->x.d[j]),&libp->para[i],objsize);

    /* Platform that increments address */
#if (defined(__linux)&&defined(__i386))||defined(_WIN32)||defined(G__CYGWIN)
    j += objsize;
    mod = j%G__va_arg_align_size;
    if(mod) j = j-mod+G__va_arg_align_size;
#elif defined(__hpux) || defined(__hppa__)
    /* nothing */
#elif defined(__sparc) || defined(__sparc__) || defined(__SUNPRO_C)
    j += objsize;
    mod = j%G__va_arg_align_size;
    if(mod) j = j-mod+G__va_arg_align_size;

#elif (defined(__PPC__)||defined(__ppc__))&&(defined(_AIX)||defined(__APPLE__))
    //j =  G__va_align_ppc(j, objsize) + G__va_rounded_size_ppc(objsize);
#ifdef G__VAARG_PASS_BY_REFERENCE
    if(objsize>G__VAARG_PASS_BY_REFERENCE) objsize=G__VAARG_PASS_BY_REFERENCE;
#endif
    j += objsize;
    mod = j%G__va_arg_align_size;
    if(mod) j = j-mod+G__va_arg_align_size;

#elif (defined(__PPC__)||defined(__ppc__))&&(defined(__linux)||defined(__linux__))
#ifdef G__VAARG_PASS_BY_REFERENCE
    if(objsize>G__VAARG_PASS_BY_REFERENCE) objsize=G__VAARG_PASS_BY_REFERENCE;
#endif
    j += objsize;
    mod = j%G__va_arg_align_size;
    if(mod) j = j-mod+G__va_arg_align_size;

#else
#ifdef G__VAARG_PASS_BY_REFERENCE
    if(objsize>G__VAARG_PASS_BY_REFERENCE) objsize=G__VAARG_PASS_BY_REFERENCE;
#endif
    j += objsize;
    mod = j%G__va_arg_align_size;
    if(mod) j = j-mod+G__va_arg_align_size;
#endif

  }
}

#ifdef G__VAARG_COPYFUNC
/**************************************************************************
 * G__va_arg_copyfunc() , Never used so far
 **************************************************************************/
void G__va_arg_copyfunc(fp,ifunc,ifn)
FILE* fp;
struct G__ifunc_table* ifunc;
int ifn;
{
  FILE *xfp;
  int n;
  int c;
  int nest = 0;
  int double_quote = 0;
  int single_quote = 0;
  int flag = 0;

  if(G__srcfile[ifunc->pentry[ifn]->filenum].fp) 
    xfp = G__srcfile[ifunc->pentry[ifn]->filenum].fp;
  else {
    xfp = fopen(G__srcfile[ifunc->pentry[ifn]->filenum].filename,"r");
    flag = 1;
  }
  if(!xfp) return;
  fsetpos(xfp,&ifunc->pentry[ifn]->pos);

  fprintf(fp,"%s ",G__type2string(ifunc->type[ifn]
				  ,ifunc->p_tagtable[ifn]
				  ,ifunc->p_typetable[ifn]
				  ,ifunc->reftype[ifn]
				  ,ifunc->isconst[ifn]));
  fprintf(fp,"%s(",ifunc->funcname[ifn]);

  /* print out parameter types */
  for(n=0;n<ifunc->para_nu[ifn];n++) {
    
    if(n!=0) {
      fprintf(fp,",");
    }

    if('u'==ifunc->para_type[ifn][n] &&
       0==strcmp(G__struct.name[ifunc->para_p_tagtable[ifn][n]],"va_list")) {
      fprintf(fp,"struct G__param* G__VA_libp,int G__VA_n");
      break;
    }
    /* print out type of return value */
    fprintf(fp,"%s",G__type2string(ifunc->para_type[ifn][n]
				    ,ifunc->para_p_tagtable[ifn][n]
				    ,ifunc->para_p_typetable[ifn][n]
				    ,ifunc->para_reftype[ifn][n]
				    ,ifunc->para_isconst[ifn][n]));
    
    if(ifunc->para_name[ifn][n]) {
      fprintf(fp," %s",ifunc->para_name[ifn][n]);
    }
    if(ifunc->para_def[ifn][n]) {
      fprintf(fp,"=%s",ifunc->para_def[ifn][n]);
    }
  }
  fprintf(fp,")");
  if(ifunc->isconst[ifn]&G__CONSTFUNC) {
    fprintf(fp," const");
  }

  c = 0;
  while(c!='{') c = fgetc(xfp);
  fprintf(fp,"{");

  nest = 1;
  while(c!='}' || nest) {
    c = fgetc(xfp);
    fputc(c,fp);
    switch(c) {
    case '"':
      if(!single_quote) double_quote ^= 1;
      break;
    case '\'':
      if(!double_quote) single_quote ^= 1;
      break;
    case '{':
      if(!single_quote && !double_quote) ++nest;
      break;
    case '}':
      if(!single_quote && !double_quote) --nest;
      break;
    }
  }
  fprintf(fp,"\n");
  if(flag && xfp) fclose(xfp);
}
#endif

#endif

#ifndef G__OLDIMPLEMENTATION2204
/**************************************************************************
 * G__typeconversion
 **************************************************************************/
void G__typeconversion(ifunc,ifn,libp) 
struct G__ifunc_table *ifunc;
int ifn;
struct G__param *libp;
{
  int formal_type,    param_type;
  int formal_reftype, param_reftype;
  int formal_tagnum,  param_tagnum;
  int i;
  for(i=0;i<libp->paran && i<ifunc->para_nu[ifn];i++) {
    formal_type = ifunc->para_type[ifn][i];
    param_type = libp->para[i].type;
    formal_reftype = ifunc->para_reftype[ifn][i];
    param_reftype = libp->para[i].obj.reftype.reftype;
    formal_tagnum = ifunc->para_p_tagtable[ifn][i];
    param_tagnum = libp->para[i].tagnum;
    switch(formal_type) {
    case 'd':
    case 'f':
      switch(param_type) {
      case 'c':
      case 's':
      case 'i':
      case 'l':
      case 'b':
      case 'r':
      case 'h':
      case 'k':
	libp->para[i].obj.d = libp->para[i].obj.i;
	libp->para[i].type = formal_type;
	libp->para[i].ref = (long)(&libp->para[i].obj.d);
	break;
      }
      break;
    case 'c':
    case 's':
    case 'i':
    case 'l':
    case 'b':
    case 'r':
    case 'h':
    case 'k':
      switch(param_type) {
      case 'd':
      case 'f':
	libp->para[i].obj.i = libp->para[i].obj.d;
	libp->para[i].type = formal_type;
	libp->para[i].ref = (long)(&libp->para[i].obj.i);
	break;
      }
      break;
    }
  }
}
#endif

#ifndef G__OLDIMPLEMENTATION1908
/**************************************************************************
 * G__DLL_direct_globalfunc
 **************************************************************************/
int G__DLL_direct_globalfunc(G__value *result7
			     ,G__CONST char *funcname /* ifunc */
			     ,struct G__param *libp
			     ,int hash)   /* ifn */  {
  struct G__ifunc_table *ifunc = (struct G__ifunc_table*)funcname;
  int ifn=hash;

  int (*itp2f)();
  double (*dtp2f)();
  void (*vtp2f)();
  G__va_arg_buf (*utp2f)();

  G__va_arg_buf G__va_arg_return;
  G__va_arg_buf G__va_arg_bufobj;
#ifndef G__OLDIMPLEMENTATION2204
  G__typeconversion(ifunc,ifn,libp);
#endif
  G__va_arg_put(&G__va_arg_bufobj,libp,0);

  switch(ifunc->type[ifn]) {
  case 'd':
  case 'f':
    dtp2f = (double (*)())ifunc->pentry[ifn]->tp2f;
    G__letdouble(result7,ifunc->type[ifn],dtp2f(G__va_arg_bufobj));
    break;
  case 'u':
    utp2f = (G__va_arg_buf (*)())ifunc->pentry[ifn]->tp2f;
    G__va_arg_return = utp2f(G__va_arg_bufobj);
    result7->type = 'u';
    result7->tagnum = ifunc->p_tagtable[ifn];
    result7->typenum = ifunc->p_typetable[ifn];
    result7->obj.i = (long)(&G__va_arg_return); /* incorrect! experimental */
    break;
  case 'y':
    vtp2f = (void (*)())ifunc->pentry[ifn]->tp2f;
    vtp2f(G__va_arg_bufobj);
    G__setnull(result7);
    break;
  default:
    itp2f = (int(*)())ifunc->pentry[ifn]->tp2f;
    G__letint(result7,ifunc->type[ifn],itp2f(G__va_arg_bufobj));
    break;
  }

  result7->isconst = ifunc->isconst[ifn];

  return 1;
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
