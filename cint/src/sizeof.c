/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file sizeof.c
 ************************************************************************
 * Description:
 *  Getting object size 
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto (MXJ02154@niftyserve.or.jp)
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
      fprintf(G__serr,"Internal error: G__sizeof() illegal reftype ID %d\n"
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
  case 'Q': /* pointer to function */
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

  fprintf(G__serr,"Error: member %s not found in %s ",memname,tagname);
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
  if(-1 != tagnum) return(G__struct.size[tagnum]);

  typenum = G__defined_typename(typename);
  if(-1 != typenum) {
    switch(G__newtype.type[typenum]) {
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
  if((strcmp(typename,"double")==0)||
     (strcmp(typename,"longdouble")==0))
    return(sizeof(double));
  if(strcmp(typename,"void")==0)
#ifndef G__OLDIMPLEMENTATION930
    return(sizeof(void*));
#else
    return(-1);
#endif
  if(strcmp(typename,"FILE")==0)
    return(sizeof(FILE));

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
  if(var) {
    if(INT_MAX==var->varlabel[ig15][1]) {
      return(sizeof(void *));
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
  if(buf.type) return(G__sizeof(&buf));
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
      if((strcmp(typename,"double")==0)||
	 (strcmp(typename,"longdouble")==0)) {
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
