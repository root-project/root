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
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"

extern "C" {

/* array index for type_info. This must correspond to the class member
* layout of class type_info in the <typeinfo.h> */
#define G__TYPEINFO_VIRTUALID 0
#define G__TYPEINFO_TYPE      1
#define G__TYPEINFO_TAGNUM    2
#define G__TYPEINFO_TYPENUM   3
#define G__TYPEINFO_REFTYPE   4
#define G__TYPEINFO_SIZE      5
#define G__TYPEINFO_ISCONST   6

int G__rootCcomment=0;

void G__load_longlong(int* ptag, int* ptype, int which);

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
int G__sizeof(G__value *object)
{
  if(isupper(object->type) && object->obj.reftype.reftype!=G__PARANORMAL) {
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
  case 'G': /* bool */
#ifdef G__BOOL4BYTE
    return(G__INTALLOC);
#else
    return(G__CHARALLOC);
#endif
  case 'N':
  case 'M':
    return(G__LONGLONGALLOC);
#ifndef G__OLDIMPLEMENTATION2191
  case 'Q':
    return(G__LONGDOUBLEALLOC);
#endif
  }
  return(1);
}


/******************************************************************
* int G__Loffsetof()
*
******************************************************************/
int G__Loffsetof(const char *tagname,const char *memname)
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
*   and cintex
*
******************************************************************/
int G__Lsizeof(const char *type_name)
{
  int hash;
  int ig15;
  struct G__var_array *var;
  G__value buf;
  int tagnum,typenum;
  int result;
  int pointlevel=0;
  G__FastAllocString namebody(G__MAXNAME+20);
  char *p;
  int i;


  /* return size of pointer if xxx* */
  if('*'==type_name[strlen(type_name)-1]) {
    return(sizeof(void *));
  }

  /* searching for struct/union tagtable */
  if((strncmp(type_name,"struct ",7)==0)
     || strncmp(type_name,"signed ",7)==0
     ) {
    type_name = type_name+7;
    while (isspace(type_name[0])) ++type_name;
  }
  else if((strncmp(type_name,"class ",6)==0)) {
    type_name = type_name+6;
    while (isspace(type_name[0])) ++type_name;
  }
  else if((strncmp(type_name,"union ",6)==0)) {
    type_name = type_name+6;
    while (isspace(type_name[0])) ++type_name;
  }

  tagnum = G__defined_tagname(type_name,1); /* case 8) */
  if(-1 != tagnum) {
    if('e'!=G__struct.type[tagnum]) return(G__struct.size[tagnum]);
    else                            return(G__INTALLOC);
  }

  typenum = G__defined_typename(type_name);
  if(-1 != typenum) {
    switch(G__newtype.type[typenum]) {
    case 'n':
    case 'm':
      result = sizeof(G__int64);
      break;
    case 'q':
      result = sizeof(long double);
      break;
    case 'g':
#ifdef G__BOOL4BYTE
      result = sizeof(int);
      break;
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

  if((strcmp(type_name,"int")==0)||
     (strcmp(type_name,"unsigned int")==0))
    return(sizeof(int));
  if((strcmp(type_name,"long")==0)||
     (strcmp(type_name,"long int")==0)||
     (strcmp(type_name,"unsigned long")==0)||
     (strcmp(type_name,"unsigned long int")==0))
    return(sizeof(long));
  if((strcmp(type_name,"short")==0)||
     (strcmp(type_name,"short int")==0)||
     (strcmp(type_name,"unsigned short")==0)||
     (strcmp(type_name,"unsigned short int")==0))
    return(sizeof(short));
  if((strcmp(type_name,"char")==0)||
     (strcmp(type_name,"unsigned char")==0))
    return(sizeof(char));
  if((strcmp(type_name,"float")==0)||
     (strcmp(type_name,"float")==0))
    return(sizeof(float));
  if((strcmp(type_name,"double")==0)
     )
    return(sizeof(double));
  if(strcmp(type_name,"long double")==0) {
     return(sizeof(long double));
  }
  if(strcmp(type_name,"long long")==0
     || strcmp(type_name,"long long int")==0
     ) {
     return(sizeof(G__int64));
  }
  if(strcmp(type_name,"unsigned long long")==0
     || strcmp(type_name,"unsigned long long int")==0
     ) {
     return(sizeof(G__uint64));
  }
  if(strcmp(type_name,"void")==0)
    return(sizeof(void*));
  if(strcmp(type_name,"FILE")==0)
    return(sizeof(FILE));
#ifdef G__BOOL4BYTE
  if(strcmp(type_name,"bool")==0) return(sizeof(int));
#else
  if(strcmp(type_name,"bool")==0) return(sizeof(unsigned char));
#endif
  while (type_name[pointlevel] == '*') {
    ++pointlevel;
  }
  namebody = (type_name + pointlevel);
  while ((p = strrchr(namebody, '['))) {
    *p = '\0';
    ++pointlevel;
  }
  G__hash(namebody(),hash,ig15)
  var = G__getvarentry(namebody,hash,&ig15,&G__global,G__p_local);
  if (!var) {
    G__FastAllocString temp(G__ONELINE);
    if (G__memberfunc_tagnum != -1) { // questionable
       temp.Format("%s\\%x\\%x\\%x", namebody(), G__func_page, G__func_now, G__memberfunc_tagnum);
    }
    else {
       temp.Format("%s\\%x\\%x", namebody(), G__func_page, G__func_now);
    }
    G__hash(temp(), hash, i)
    var = G__getvarentry(temp, hash, &ig15, &G__global, G__p_local);
  }
  if (var) {
    if (var->varlabel[ig15][1] == INT_MAX /* unspecified size array flag */) {
      if (var->type[ig15] == 'c') {
        return strlen((char*) var->p[ig15]) + 1;
      }
      else {
        return sizeof(void *);
      }
    }
    buf.type = var->type[ig15];
    buf.tagnum = var->p_tagtable[ig15];
    buf.typenum = var->p_typetable[ig15];
    if (isupper(buf.type)) {
      buf.obj.reftype.reftype = var->reftype[ig15];
    }
    size_t num_of_elements = 0;
    if (pointlevel > var->paran[ig15] /* array dimensionality */) {
      switch (pointlevel) {
      case 0:
        break;
      case 1:
        if (G__PARANORMAL == buf.obj.reftype.reftype) {
          buf.type = tolower(buf.type);
        }
        else if (G__PARAP2P == buf.obj.reftype.reftype) {
          buf.obj.reftype.reftype = G__PARANORMAL;
        }
        else {
          --buf.obj.reftype.reftype;
        }
        break;
      case 2:
        if (G__PARANORMAL == buf.obj.reftype.reftype) {
          buf.type = tolower(buf.type);
        }
        else if (G__PARAP2P == buf.obj.reftype.reftype) {
          buf.type = tolower(buf.type);
          buf.obj.reftype.reftype = G__PARANORMAL;
        }
        else if (G__PARAP2P2P == buf.obj.reftype.reftype) {
          buf.obj.reftype.reftype = G__PARANORMAL;
        }
        else {
          buf.obj.reftype.reftype -= 2;
        }
        break;
      }
      return G__sizeof(&buf);
    }
    switch (pointlevel) {
    case 0:
      num_of_elements = var->varlabel[ig15][1] /* num of elements */;
      if (!num_of_elements) {
        num_of_elements = 1;
      }
      break;
    case 1:
      num_of_elements = var->varlabel[ig15][0] /* stride */;
      break;
    default:
      num_of_elements = var->varlabel[ig15][0] /* stride */;
      for (i = 1; i < pointlevel; ++i) {
        num_of_elements /= var->varlabel[ig15][i+1];
      }
      break;
    }
    if (isupper(var->type[ig15])) {
      return num_of_elements * sizeof(void *);
    }
    return num_of_elements * G__sizeof(&buf);
  }
  buf = G__getexpr(type_name);
  if (buf.type) {
    if ((buf.type == 'C') && (type_name[0] == '"')) {
      return strlen((char*) buf.obj.i) + 1;
    }
    return G__sizeof(&buf);
  }
  return -1;
}

#ifdef G__TYPEINFO
/******************************************************************
* int G__typeid(type_name)
*
* Called by
*   G__special_func()
*
******************************************************************/
long *G__typeid(const char *typenamein)
{
  G__value buf;
  int c;
  long *type_info;
  int tagnum,typenum,type=0,reftype=G__PARANORMAL,size=0;
  int pointlevel=0,isref=0;
  int tag_type_info;
  G__FastAllocString typenamebuf_sb(G__MAXNAME*2);
  char* typenamebuf = typenamebuf_sb;
  char *type_name;
  int isconst = 0;

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
  typenamebuf_sb = typenamein;
  type_name=typenamebuf;
  size_t len=strlen(type_name);

  while('*'==(c=type_name[len-1]) || '&'==c) {
    switch(c) {
    case '*':
      ++pointlevel;
      break;
    case '&':
      isref=1;
      break;
    }
    --len;
    type_name[len]='\0';
  }

  /**********************************************************************
  * Search for typedef names
  **********************************************************************/
  typenum = G__defined_typename(type_name);
  if(-1 != typenum) {
    type    = G__newtype.type[typenum];
    tagnum  = G__newtype.tagnum[typenum];
    reftype = G__newtype.reftype[typenum];
    if(-1!=tagnum) {
      size = G__struct.size[tagnum];
    }
    else {
      switch(tolower(type)) {
      case 'n':
      case 'm':
        size = G__LONGLONGALLOC;
        break;
      case 'g':
#ifdef G__BOOL4BYTE
        size = G__INTALLOC;
        break;
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
    if((strncmp(type_name,"struct",6)==0)) {
      type_name = type_name+6;
    }
    else if((strncmp(type_name,"class",5)==0)) {
      type_name = type_name+5;
    }
    else if((strncmp(type_name,"union",5)==0)) {
      type_name = type_name+5;
    }

    tagnum = G__defined_tagname(type_name,1);
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
      case 'n':
        size = G__struct.size[tagnum];
        G__genericerror("Error: can not get sizeof namespace");
        break;
      }
    }

    else {

      /********************************************************************
       * Search for intrinsic types
       *******************************************************************/
      reftype = G__PARANORMAL;

      if(strcmp(type_name,"int")==0) {
        type = 'i';
        size = G__INTALLOC;
      }
      if(strcmp(type_name,"unsigned int")==0) {
        type = 'h';
        size = G__INTALLOC;
      }
      if((strcmp(type_name,"long")==0)||
         (strcmp(type_name,"long int")==0)) {
        type='l';
        size = G__LONGALLOC;
      }
      if((strcmp(type_name,"unsigned long")==0)||
         (strcmp(type_name,"unsigned long int")==0)) {
        type = 'k';
        size = G__LONGALLOC;
      }
      if((strcmp(type_name,"long long")==0)) {
        type='n';
        size = G__LONGLONGALLOC;
      }
      if((strcmp(type_name,"unsigned long long")==0)) {
        type='m';
        size = G__LONGLONGALLOC;
      }
      if((strcmp(type_name,"short")==0)||
         (strcmp(type_name,"short int")==0)) {
        type = 's';
        size = G__SHORTALLOC;
      }
      if((strcmp(type_name,"unsigned short")==0)||
         (strcmp(type_name,"unsigned short int")==0)) {
        type = 'r';
        size = G__SHORTALLOC;
      }
      if((strcmp(type_name,"char")==0)||
         (strcmp(type_name,"signed char")==0)) {
        type = 'c';
        size = G__CHARALLOC;
      }
      if(strcmp(type_name,"unsigned char")==0) {
        type = 'b';
        size = G__CHARALLOC;
      }
      if(strcmp(type_name,"float")==0) {
        type = 's';
        size = G__FLOATALLOC;
      }
      if((strcmp(type_name,"double")==0)
         ) {
        type = 'd';
        size = G__DOUBLEALLOC;
      }
      if((strcmp(type_name,"long double")==0)
         ) {
        type = 'q';
        size = G__LONGDOUBLEALLOC;
      }
      if(strcmp(type_name,"void")==0) {
        type = 'y';
        size = sizeof(void*);
      }
      if(strcmp(type_name,"FILE")==0) {
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
    isconst = buf.isconst;

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
  type_info[G__TYPEINFO_ISCONST] = isconst;

  return( type_info ) ;

}
#endif

/******************************************************************
* G__getcomment()
*
******************************************************************/
void G__getcomment(char *buf,G__comment_info *pcomment,int tagnum)
{
  fpos_t pos,store_pos;
  FILE *fp;
  int filenum;
  char *p;
  int flag=1;

  if(-1!=pcomment->filenum) {
    if(-1!=tagnum && G__NOLINK==G__struct.iscpplink[tagnum] &&
       pcomment->filenum>=0) {
      pos = pcomment->p.pos;
      filenum = pcomment->filenum;
      if(filenum==G__MAXFILE) {
         fp = G__mfp;
         if ((FILE*)NULL==fp) {
            G__genericerror("Error: Unable to open temporary file");
            return;
         }
         else {
            fgetpos(fp,&store_pos);
         }
      }
      else {
         fp = G__srcfile[filenum].fp;
         if((FILE*)NULL==fp) {
            /* Open the right file even in case where we use the preprocessor */
            if (
                filenum<G__MAXFILE &&
                G__srcfile[filenum].prepname ) {
               fp = fopen(G__srcfile[filenum].prepname,"r");
            } else {
               fp = fopen(G__srcfile[filenum].filename,"r");
            }
            flag=0;
         }
         else {
            fgetpos(fp,&store_pos);
         }
      }
      fsetpos(fp,&pos);
      // dummy check; buffer length management needs new function signature.
      if (!fgets(buf,G__ONELINE-1,fp)) {}
      p = strchr(buf,'\n');
      if(p) *p = '\0';
      p = strchr(buf,'\r');
      if(p) *p = '\0';
      if(G__rootCcomment) {
        p = (char*)G__strrstr(buf,"*/");
        if(p) *p = '\0';
      }

      if(flag) {
        fsetpos(fp,&store_pos);
      }
      else {
        fclose(fp);
      }
    }
    else if(-2==pcomment->filenum) {
       G__strlcpy(buf,pcomment->p.com,G__ONELINE);
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
void G__getcommenttypedef(char *buf,G__comment_info *pcomment,int typenum)
{
  fpos_t pos,store_pos;
  FILE *fp;
  int filenum;
  char *p;
  int flag=1;

  if(-1!=typenum && -1!=pcomment->filenum) {
    if(G__NOLINK==G__newtype.iscpplink[typenum] && pcomment->filenum>=0) {
      pos = pcomment->p.pos;
      filenum = pcomment->filenum;
      if(filenum==G__MAXFILE) {
         fp = G__mfp;
         if ((FILE*)NULL==fp) {
            G__genericerror("Error: Unable to open temporary file");
            return;
         }
         else {
            fgetpos(fp,&store_pos);
         }
      }
      else {
         fp = G__srcfile[filenum].fp;
         if((FILE*)NULL==fp) {
            /* Open the right file even in case where we use the preprocessor */
            if (
                filenum<G__MAXFILE &&
                G__srcfile[filenum].prepname ) {
               fp = fopen(G__srcfile[filenum].prepname,"r");
            } else {
               fp = fopen(G__srcfile[filenum].filename,"r");
            }
            flag=0;
         }
         else {
            fgetpos(fp,&store_pos);
         }
      }
      fsetpos(fp,&pos);
      // dummy check; buffer length management needs new function signature.
      if (!fgets(buf,G__ONELINE-1,fp)) {}
      p = strchr(buf,'\n');
      if(p) *p = '\0';
      p = strchr(buf,'\r');
      if(p) *p = '\0';
      p = strchr(buf,';');
      if(p) *(p+1) = '\0';

      if(flag) {
        fsetpos(fp,&store_pos);
      }
      else {
        fclose(fp);
      }
    }
    else if(-2==pcomment->filenum) {
       G__strlcpy(buf,pcomment->p.com,G__ONELINE);
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
long G__get_classinfo(const char *item,int tagnum)
{
  char *buf;
  int tag_string_buf;
  struct G__inheritance *baseclass;
  size_t p;
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
    if(!baseclass) return((long)0);
    p=0;
    buf[0]='\0';
    for(i=0;i<baseclass->basen;i++) {
      if(baseclass->herit[i]->property&G__ISDIRECTINHERIT) {
        if(p) {
           sprintf(buf+p,","); // Legacy, we can't know the buffer length
          ++p;
        }
        sprintf(buf+p,"%s%s" ,G__access2string(baseclass->herit[i]->baseaccess) // Legacy, we can't know the buffer length
                ,G__struct.name[baseclass->herit[i]->basetagnum]);
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
long G__get_variableinfo(const char *item,long *phandle,long *pindex,int tagnum)
{
  char *buf;
  int tag_string_buf;
  struct G__var_array *var;
  long index;

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
    strcpy(buf,G__type2string(var->type[index]         // Legacy use, we can't know the buffer size
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
      return((long)0);
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
long G__get_functioninfo(const char *item,long *phandle,long *pindex,int tagnum)
{
  char *buf;
  int tag_string_buf;
  /* char temp[G__MAXNAME]; */
  struct G__ifunc_table_internal *ifunc;
  long index;
  int i;
  size_t p;

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

  ifunc = (struct G__ifunc_table_internal *)(*phandle);
  index = (*pindex);

  if((struct G__ifunc_table_internal*)NULL==ifunc || ifunc->allifunc<=index) {
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
    ifunc = (struct G__ifunc_table_internal *)(*phandle);
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
    strcpy(buf,G__type2string(ifunc->type[index]           // Legacy use, we can't know the buffer size
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
        sprintf(buf+p,",");  // Legacy use, we can't know the buffer size
        ++p;
      }
      sprintf(buf+p,"%s",G__type2string(ifunc->param[index][i]->type         // Legacy use, we can't know the buffer size
                                        ,ifunc->param[index][i]->p_tagtable
                                        ,ifunc->param[index][i]->p_typetable
                                        ,ifunc->param[index][i]->reftype,0));
      p=strlen(buf);
      if(ifunc->param[index][i]->pdefault) {
        sprintf(buf+p,"=");  // Legacy use, we can't know the buffer size
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
      return((long)0);
    }
  }
  return(0);
}

/**************************************************************************
 * G__va_arg_setalign()
 **************************************************************************/
#ifdef G__VAARG_INC_COPY_N
static int G__va_arg_align_size=G__VAARG_INC_COPY_N;
#else
static int G__va_arg_align_size=0;
#endif
void G__va_arg_setalign(int n)
{
  G__va_arg_align_size = n;
}

/**************************************************************************
 * G__va_arg_copyvalue()
 **************************************************************************/
void G__va_arg_copyvalue(int t,void *p,G__value *pval,int objsize)
{
#ifdef G__VAARG_PASS_BY_REFERENCE
  if(objsize>G__VAARG_PASS_BY_REFERENCE) {
    if(pval->ref>0x1000) *(long*)(p) = pval->ref;
    else *(long*)(p) = (long)G__int(*pval);
    return;
  }

#endif
  switch(t) {
  case 'n':
  case 'm':
    *(G__int64*)(p) = (G__int64)G__Longlong(*pval);
    break;
  case 'g':
#ifdef G__BOOL4BYTE
    *(int*)(p) = (int)G__int(*pval);
    break;
#endif
  case 'c':
  case 'b':
#if defined(__GNUC__) || defined(G__WIN32)
    *(int*)(p) = (int)G__int(*pval);
#else
    *(char*)(p) = (char)G__int(*pval);
#endif
    break;
  case 'r':
  case 's':
#if defined(__GNUC__) || defined(G__WIN32)
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
#define G__OLDIMPLEMENTATION2235
#if defined(__GNUC__) || defined(G__WIN32)
    *(double*)(p) = (double)G__double(*pval);
#else
    *(float*)(p) = (float)G__double(*pval);
#endif
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
#define G__va_align_ppc(AP, objsize)                                           \
     ((((unsigned long)(AP)) + ((G__alignof_ppc(objsize) == 16) ? 15 : 3)) \
      & ~((G__alignof_ppc(objsize) == 16) ? 15 : 3))

#elif (defined(__PPC__)||defined(__ppc__))&&(defined(__linux)||defined(__linux__))

#endif

   // cross-compiling for iOS and iOS simulator (assumes host is Intel Mac OS X)
#if defined(R__IOSSIM) || defined(R__IOS)
#ifdef __x86_64__
#undef __x86_64__
#endif
#ifdef __i386__
#undef __i386
#endif
#ifdef R__IOSSIM
#define __i386 1
#endif
#ifdef R__IOS
#define __arm__ 1
#endif
#endif


/**************************************************************************
 * G__va_arg_put()
 **************************************************************************/
void G__va_arg_put(G__va_arg_buf *pbuf,G__param *libp,int n)
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
#if defined(__GNUC__) || defined(G__WIN32)
    switch(libp->para[i].type) {
    case 'c': case 'b': case 's': case 'r': objsize = sizeof(int); break;
    case 'f': objsize = sizeof(double); break;
    }
#endif

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
#elif ((defined(__sparc) || defined(__i386)) && defined(__SUNPRO_CC))
    /* nothing */
#elif (defined(__PPC__)||defined(__ppc__))&&(defined(_AIX)||defined(__APPLE__))
    /* nothing */
#elif (defined(__PPC__)||defined(__ppc__))&&(defined(__linux)||defined(__linux__))
    /* nothing */
#elif defined(__x86_64__) && defined(__linux)
    /* nothing */
#elif defined(__arm__)
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
#elif ((defined(__sparc) || defined(__i386)) && defined(__SUNPRO_CC))
    j += objsize;
    mod = j%G__va_arg_align_size;
    if(mod) j = j-mod+G__va_arg_align_size;
#elif defined(__arm__)
#ifdef G__VAARG_PASS_BY_REFERENCE
     if(objsize>G__VAARG_PASS_BY_REFERENCE) objsize=G__VAARG_PASS_BY_REFERENCE;
#endif
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
#elif defined(__x86_64__) && defined(__linux)
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
void G__va_arg_copyfunc(FILE *fp,G__ifunc_table_internal *ifunc,int ifn)
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

    if('u'==ifunc->param[ifn][n]->type &&
       0==strcmp(G__struct.name[ifunc->param[ifn][n]->p_tagtable],"va_list")) {
      fprintf(fp,"struct G__param* G__VA_libp,int G__VA_n");
      break;
    }
    /* print out type of return value */
    fprintf(fp,"%s",G__type2string(ifunc->param[ifn][n]->type
                                    ,ifunc->param[ifn][n]->p_tagtable
                                    ,ifunc->param[ifn][n]->p_typetable
                                    ,ifunc->param[ifn][n]->reftype
                                    ,ifunc->param[ifn][n]->isconst));

    if(ifunc->param[ifn][n]->name) {
      fprintf(fp," %s",ifunc->param[ifn][n]->name);
    }
    if(ifunc->param[ifn][n]->def) {
      fprintf(fp,"=%s",ifunc->param[ifn][n]->def);
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


/**************************************************************************
 * G__typeconversion
 **************************************************************************/
void G__typeconversion(G__ifunc_table_internal *ifunc,int ifn
                       ,G__param *libp)
{
  int formal_type,    param_type;
  int formal_reftype, param_reftype;
  int formal_tagnum,  param_tagnum;
  int i;
  for(i=0;i<libp->paran && i<ifunc->para_nu[ifn];i++) {
    formal_type = ifunc->param[ifn][i]->type;
    param_type = libp->para[i].type;
    formal_reftype = ifunc->param[ifn][i]->reftype;
    param_reftype = libp->para[i].obj.reftype.reftype;
    formal_tagnum = ifunc->param[ifn][i]->p_tagtable;
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
         libp->para[i].obj.d = (double)libp->para[i].obj.i;
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
        libp->para[i].obj.i = (long)libp->para[i].obj.d;
        libp->para[i].type = formal_type;
        libp->para[i].ref = (long)(&libp->para[i].obj.i);
        break;
      }
      break;
    }
  }
}

/**************************************************************************
 * G__DLL_direct_globalfunc
 **************************************************************************/
int G__DLL_direct_globalfunc(G__value *result7
                             ,G__CONST char *funcname /* ifunc */
                             ,struct G__param *libp
                             ,int hash)   /* ifn */  {
  struct G__ifunc_table_internal *ifunc = (struct G__ifunc_table_internal*)funcname;
  int ifn=hash;

  int (*itp2f)(G__va_arg_buf);
  double (*dtp2f)(G__va_arg_buf);
  void (*vtp2f)(G__va_arg_buf);
  G__va_arg_buf (*utp2f)(G__va_arg_buf);

  G__va_arg_buf G__va_arg_return;
  G__va_arg_buf G__va_arg_bufobj;
  G__typeconversion(ifunc,ifn,libp);
  G__va_arg_put(&G__va_arg_bufobj,libp,0);

  switch(ifunc->type[ifn]) {
  case 'd':
  case 'f':
    dtp2f = (double (*)(G__va_arg_buf))ifunc->pentry[ifn]->tp2f;
    G__letdouble(result7,ifunc->type[ifn],dtp2f(G__va_arg_bufobj));
    break;
  case 'u':
    utp2f = (G__va_arg_buf (*)(G__va_arg_buf))ifunc->pentry[ifn]->tp2f;
    G__va_arg_return = utp2f(G__va_arg_bufobj);
    result7->type = 'u';
    result7->tagnum = ifunc->p_tagtable[ifn];
    result7->typenum = ifunc->p_typetable[ifn];
    result7->obj.i = (long)(&G__va_arg_return); /* incorrect! experimental */
    break;
  case 'y':
    vtp2f = (void (*)(G__va_arg_buf))ifunc->pentry[ifn]->tp2f;
    vtp2f(G__va_arg_bufobj);
    G__setnull(result7);
    break;
  default:
    itp2f = (int(*)(G__va_arg_buf))ifunc->pentry[ifn]->tp2f;
    G__letint(result7,ifunc->type[ifn],itp2f(G__va_arg_bufobj));
    break;
  }

  result7->isconst = ifunc->isconst[ifn];

  return 1;
}

//______________________________________________________________________________
//
//  Functions in the C interface.
//

//______________________________________________________________________________
void G__loadlonglong(int* ptag, int* ptype, int which)
{
   int lltag = -1, lltype = -1;
   int ulltag = -1, ulltype = -1;
   int ldtag = -1, ldtype = -1;
   int store_decl = G__decl;
   int store_def_struct_member = G__def_struct_member;
   int flag = 0;
   int store_tagdefining = G__tagdefining;
   int store_def_tagnum = G__def_tagnum;
   G__tagdefining = -1;
   G__def_tagnum = -1;
   G__def_struct_member = 0;
   G__decl = 0;
   if (0 == G__defined_macro("G__LONGLONG_H")) {
      G__loadfile("long.dll"); /* used to switch case between .dl and .dll */
      flag = 1;
   }
   G__decl = 1;
   G__def_struct_member = store_def_struct_member;
   if (which == G__LONGLONG || flag) {
      lltag = G__defined_tagname("G__longlong", 2);
      lltype = G__search_typename("long long", 'u', G__tagnum, G__PARANORMAL);
      if (lltag != -1) G__struct.defaulttypenum[lltag] = lltype;
      if (lltype != -1) G__newtype.tagnum[lltype] = lltag;
   }
   if (which == G__ULONGLONG || flag) {
      ulltag = G__defined_tagname("G__ulonglong", 2);
      ulltype = G__search_typename("unsigned long long", 'u', G__tagnum, G__PARANORMAL);
      if (ulltag != -1) G__struct.defaulttypenum[ulltag] = ulltype;
      if (ulltype != -1) G__newtype.tagnum[ulltype] = ulltag;
   }
   if (which == G__LONGDOUBLE || flag) {
      ldtag = G__defined_tagname("G__longdouble", 2);
      ldtype = G__search_typename("long double", 'u', G__tagnum, G__PARANORMAL);
      if (ldtag != -1) G__struct.defaulttypenum[ldtag] = ldtype;
      if (ldtype != -1) G__newtype.tagnum[ldtype] = ldtag;
   }
   switch (which) {
      case G__LONGLONG:
         *ptag = lltag;
         *ptype = lltype;
         break;
      case G__ULONGLONG:
         *ptag = ulltag;
         *ptype = ulltype;
         break;
      case G__LONGDOUBLE:
         *ptag = ldtag;
         *ptype = ldtype;
         break;
   }
   G__def_tagnum = store_def_tagnum;
   G__tagdefining = store_tagdefining;
   G__decl = store_decl;
   return;
}

} /* extern "C" */

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
