/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file aux.c
 ************************************************************************
 * Description:
 *  Auxuary function  
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

/****************************************************************
*  Support libraries. Not essensial to C/C++ interpreter.
*
****************************************************************/


/****************************************************************
* G__split(original,stringbuf,argc,argv)
* split arguments separated by space char.
* CAUTION: input string will be modified. If you want to keep
*         the original string, you should copy it to another string.
****************************************************************/
int G__split(line,string,argc,argv)
char *line;
char *string;
int *argc;
char *argv[];
{
  int lenstring;
  int i=0;
  int flag=0;
  int n_eof=1;
  int single_quote=0,double_quote=0,back_slash=0;
  
  while((string[i]!='\n')&&
	(string[i]!='\r')&&
	(string[i]!='\0')
#ifdef G__OLDIMPLEMENTATION1616
	&& (string[i]!=EOF)
#endif
	) i++;
  string[i]='\0';
  line[i]='\0';
  lenstring=i;
#ifdef G__OLDIMPLEMENTATION1616
  if(string[i]==EOF) n_eof=0;
#endif
  argv[0]=line;

  *argc=0;
  for(i=0;i<lenstring;i++) {
    switch(string[i]) {
    case '\\':
      if(back_slash==0) back_slash=1;
      else              back_slash=0;
      break;
    case '\'':
      if((double_quote==0)&&(back_slash==0)) {
	single_quote ^= 1;
	string[i]='\0';
	flag=0;
      }
      break;
    case '"' :
      if((single_quote==0)&&(back_slash==0)) {
	double_quote ^= 1;
	string[i]='\0';
	flag=0;
      }
      break;
    default  :
      if((isspace(string[i]))&&(back_slash==0)&&
	 (single_quote==0)&&(double_quote==0)) {
	string[i]='\0';
	flag=0;
      }
      else {
	if(flag==0) {
	  (*argc)++;
	  argv[*argc] = &string[i];
	  flag=1;
	}
      }
      back_slash=0;
      break;
    }
  }
  return(n_eof);
}

/****************************************************************
* G__readsimpleline(fp,line)
****************************************************************/
int G__readsimpleline(fp,line)
FILE *fp;
char *line;
{
  char *null_fgets;
  char *p;
  null_fgets=fgets(line,G__LONGLINE*2,fp);
  if(null_fgets!=NULL) {
    p=strchr(line,'\n');
    if(p) *p='\0';
    p=strchr(line,'\r');
    if(p) *p='\0';
  }
  else {
    line[0]='\0';
  }
  if(null_fgets==NULL) return(0);
  else                 return(1);
}

/****************************************************************
* G__readline(fp,line,argbuf,argn,arg)
****************************************************************/
int G__readline(fp,line,argbuf,argn,arg)
FILE *fp;
int *argn;
char *line,*argbuf;
char *arg[];
{
  /* int i; */
  char *null_fgets;
#define G__OLDIMPLEMENTATION1816
#ifndef G__OLDIMPLEMENTATION1816
  struct G__input_file store_ifile = G__ifile;
  G__ifile.fp = fp;
  if(EOF==G__fgetline(line)) null_fgets=(char*)NULL;
  else                       null_fgets=line;
  G__ifile = store_ifile;
#else
  null_fgets=fgets(line,G__LONGLINE*2,fp);
#endif
  if(null_fgets!=NULL) {
    strcpy(argbuf,line);
    G__split(line,argbuf,argn,arg);
  }
  else {
    line[0]='\0';
    argbuf='\0';
    *argn=0;
    arg[0]=line;
  }
  if(null_fgets==NULL) return(0);
  else                 return(1);
}


#ifndef G__SMALLOBJECT

/****************************************************************
* DMA test support functions
*
* G__cmparray(array1,array2,num,mask)
* G__setarray(array,num,mask,mode)
****************************************************************/

/******************************************************************
* int G__cmparray(array1,array2,num,mask)
******************************************************************/
int G__cmparray(array1,array2,num,mask)
short array1[],array2[],mask;
int num;
{
  int i,fail=0,firstfail = -1,fail1=0,fail2=0;
  for(i=0;i<num;i++) {
    if((array1[i]&mask)!=(array2[i]&mask)) {
      if(firstfail == -1) {
	firstfail=i;
	fail1=array1[i];
	fail2=array2[i];
      }
      fail++;
    }
  }
  if(fail!=0) {
    G__fprinterr(G__serr,"G__cmparray() failcount=%d from [%d] , %d != %d\n",
	    fail,firstfail,fail1,fail2);
  }
  return(fail);
}

/******************************************************************
* G__setarray(array,num,mask,mode)
******************************************************************/
void G__setarray(array,num,mask,mode)
short array[],mask;
int num;
char *mode;
{
	int i;

	if(strcmp(mode,"rand")==0) {
		for(i=0;i<num;i++) {
			array[i]=rand()&mask;
		}
	}
	if(strcmp(mode,"inc")==0) {
		for(i=0;i<num;i++) {
			array[i]=i&mask;
		}
	}
	if(strcmp(mode,"dec")==0) {
		for(i=0;i<num;i++) {
			array[i]=(num-i)&mask;
		}
	}
	if(strcmp(mode,"check1")==0)  {
		for(i=0;i<num;i++) {
			array[i]=0xaaaa&mask;
			array[++i]=0x5555&mask;
		}
	}
	if(strcmp(mode,"check2")==0) {
		for(i=0;i<num;i++) {
			array[i]=0x5555&mask;
			array[++i]=0xaaaa&mask;
		}
	}
	if(strcmp(mode,"check3")==0) {
		for(i=0;i<num;i++) {
			array[i]=0xaaaa&mask;
			array[++i]=0xaaaa&mask;
			array[++i]=0x5555&mask;
			array[++i]=0x5555&mask;
		}
	}
	if(strcmp(mode,"check4")==0) {
		for(i=0;i<num;i++) {
			array[i]=0x5555&mask;
			array[++i]=0x5555&mask;
			array[++i]=0xaaaa&mask;
			array[++i]=0xaaaa&mask;
		}
	}
	if(strcmp(mode,"zero")==0) {
		for(i=0;i<num;i++) {
			array[i]=0;
		}
	}
	if(strcmp(mode,"one")==0) {
		for(i=0;i<num;i++) {
			array[i]=0xffff&mask;
		}
	}
}


/************************************************************************
* G__graph.c
************************************************************************/

/************************************************************************
* G__graph(xdata,ydata,ndata,title,mode)
*   xdata[i] : *double pointer of x data array
*   ydata[i] : *double pointer of y data array
*   ndata    : int number of data
*   title    : *char title
*   mode     : int mode 0:wait for close, 
*                       1:leave window and proceed
*                       2:kill xgraph window
************************************************************************/

int G__graph(xdata,ydata,ndata,title,mode)
double *xdata,*ydata;
int ndata,mode;
char *title;
{
  int i;
  FILE *fp;

  if(mode==2) {
    system("killproc xgraph");
    return(1);
  }
  
  switch(mode) {
  case 1:
  case 0:
    fp=fopen("G__graph","w");
    fprintf(fp,"TitleText: %s\n",title);
    break;
  case 2:
    fp=fopen("G__graph","w");
    fprintf(fp,"TitleText: %s\n",title);
    break;
  case 3:
    fp=fopen("G__graph","a");
    fprintf(fp,"\n");
    fprintf(fp,"TitleText: %s\n",title);
    break;
  case 4:
  default:
    fp=fopen("G__graph","a");
    fprintf(fp,"\n");
    fprintf(fp,"TitleText: %s\n",title);
    break;
  }
  fprintf(fp,"\"%s\"\n",title);
  for(i=0;i<ndata;i++) {
    fprintf(fp,"%e %e\n",xdata[i],ydata[i]);
  }
  fclose(fp);
  switch(mode) {
  case 1:
  case 4:
    system("xgraph G__graph&");
    break;
  case 0:
    system("xgraph G__graph");
    break;
  }
  return(0);
}

/****************************************************************
* End of Support libraries. Not essensial to C/C++ interpreter.
****************************************************************/


#ifndef G__NSTOREOBJECT
/****************************************************************
* G__storeobject
*
* Copy object buf2 to buf1 if not a pointer
*
*  This function looks different from interpreted environment 
* and compiled environment.
*  Interpreter : G__storeobject(void *buf1,void *buf2)
*  Compiler    : G__storeobject(G__value *buf1,G__value *buf2)
****************************************************************/
int G__storeobject(buf1,buf2)
G__value *buf1,*buf2;
{
  int i;
  struct G__var_array *var1,*var2;
  G__value lbuf1,lbuf2;
  
  
  if(buf1->type=='U' && buf2->type=='U' && buf1->tagnum==buf2->tagnum ) {
    G__incsetup_memvar(buf1->tagnum);
    G__incsetup_memvar(buf2->tagnum);
    var1 = G__struct.memvar[buf1->tagnum] ;
    var2 = G__struct.memvar[buf2->tagnum] ;
    do {
      for(i=0;i<var1->allvar;i++) {
	switch(var1->type[i]) {
	case 'u':
	  lbuf1.obj.i = buf1->obj.i + var1->p[i];
	  lbuf2.obj.i = buf2->obj.i + var2->p[i];
	  lbuf1.type='U';
	  lbuf2.type='U';
	  lbuf1.tagnum=var1->p_tagtable[i];
	  lbuf2.tagnum=var2->p_tagtable[i];
	  G__storeobject(&lbuf1,&lbuf2);
	  break;
	  
#ifndef G__OLDIMPLEMENTATION1604
	case 'g':
#endif
#ifdef G__BOOL4BYTE
	  memcpy((void *)(buf1->obj.i+var1->p[i])
		 ,(void *)(buf2->obj.i+var2->p[i])
		 ,G__INTALLOC*(var1->varlabel[i][1]+1));
	  break;
#endif
	case 'b':
	case 'c':
	  memcpy((void *)(buf1->obj.i+var1->p[i])
		 ,(void *)(buf2->obj.i+var2->p[i])
		 ,G__CHARALLOC*(var1->varlabel[i][1]+1));
	  break;
	  
	case 'r':
	case 's':
	  memcpy(
	         (void *)(buf1->obj.i+var1->p[i])
		 ,(void *)(buf2->obj.i+var2->p[i])
		 ,G__SHORTALLOC*(var1->varlabel[i][1]+1)
		 );
	  break;
	  
	case 'h':
	case 'i':
	  memcpy(
	         (void *)(buf1->obj.i+var1->p[i])
		 ,(void *)(buf2->obj.i+var2->p[i])
		 ,G__INTALLOC*(var1->varlabel[i][1]+1)
		 );
	  break;
	  
	case 'k':
	case 'l':
	  memcpy(
	         (void *)(buf1->obj.i+var1->p[i])
		 ,(void *)(buf2->obj.i+var2->p[i])
		 ,G__LONGALLOC*(var1->varlabel[i][1]+1)
		 );
	  break;
	  
	case 'f':
	  memcpy(
	         (void *)(buf1->obj.i+var1->p[i])
		 ,(void *)(buf2->obj.i+var2->p[i])
		 ,G__FLOATALLOC*(var1->varlabel[i][1]+1)
		 );
	  break;
	  
	case 'd':
	case 'w':
	  memcpy(
	         (void *)(buf1->obj.i+var1->p[i])
		 ,(void *)(buf2->obj.i+var2->p[i])
		 ,G__DOUBLEALLOC*(var1->varlabel[i][1]+1)
		 );
	  break;
	}
      }
      var1 = var1->next;
      var2 = var2->next;
    } while(var1);
    
    return(0);
  }
  else {
    G__genericerror(
	    "Error:G__storeobject buf1,buf2 different type or non struct"
		    );
    G__fprinterr(G__serr,"buf1->type = %c , buf2->type = %c\n"
	    ,buf1->type,buf2->type);
    G__fprinterr(G__serr,"buf1->tagnum = %d , buf2->tagnum = %d\n"
	    ,buf1->tagnum,buf2->tagnum);
    return(1);
  }
}

/****************************************************************
* G__scanobject
*
* Scan struct object and call 'G__do_scanobject()'
*
*  This function looks different from interpreted environment 
* and compiled environment.
*  Interpreter : G__storeobject(void *buf1,void *buf2)
*  Compiler    : G__storeobject(G__value *buf1,G__value *buf2)
****************************************************************/
int G__scanobject(buf1)
G__value *buf1;
{
  int i;
  struct G__var_array *var1;

  char type;
  char *name;
  char *tagname;
  char *typename;
  long pointer;

  char ifunc[G__ONELINE];

  if(buf1->type=='U') {
    G__incsetup_memvar(buf1->tagnum);
    var1 = G__struct.memvar[buf1->tagnum] ;
    do {
      for(i=0;i<var1->allvar;i++) {
	pointer = buf1->obj.i + var1->p[i];
	name = var1->varnamebuf[i];
	type = var1->type[i] ;
	if(var1->p_tagtable[i]>=0) {
	  tagname = G__struct.name[var1->p_tagtable[i]];
	}
	else {
	  tagname = (char *)NULL;
	}
	if(var1->p_typetable[i]>=0) {
	  typename = G__newtype.name[var1->p_typetable[i]] ;
	}
	else {
	  typename = (char *)NULL;
	}
	sprintf(ifunc,
		"G__do_scanobject((%s *)%ld,%ld,%d,%ld,%ld)"
		,tagname,pointer,(long)name,type,(long)tagname,(long)typename);
	G__getexpr(ifunc);
      }
      var1 = var1->next;
    } while(var1);
    
    return(0);
  }
  else {
    G__genericerror("Error:G__scanobject buf not a struct");
    return(1);
  }
}


/****************************************************************
* G__dumpobject
*
* dump object into a file
*
****************************************************************/
int G__dumpobject(file,buf,size)
char *file;
void *buf;
int size;
{
	FILE *fp;

	fp=fopen(file,"wb");
	fwrite(buf ,(size_t)size ,1,fp);
	fflush(fp);
	fclose(fp);
	return(1);
}

/****************************************************************
* G__loadobject
*
* load object from a file
*
****************************************************************/
int G__loadobject(file,buf,size)
char *file;
void *buf;
int size;
{
	FILE *fp;

	fp=fopen(file,"rb");
	fread(buf ,(size_t)size ,1,fp);
	fclose(fp);
	return(1);
}

#endif


#ifndef G__NSEARCHMEMBER
/****************************************************************
* G__what_type()
*
*
****************************************************************/
long G__what_type(name,type,tagname,typename)
char *name;
char *type;
char *tagname;
char *typename;
{
  G__value buf;
  static char vtype[80];
  char ispointer[3];
  
  buf = G__calc_internal(name);
  
  if(isupper(buf.type)) {
    sprintf(ispointer," *");
  }
  else {
    ispointer[0]='\0';
    /* sprintf(ispointer,""); */
  }
  
  switch(tolower(buf.type)) {
  case 'u':
    sprintf(vtype,"struct %s %s",G__struct.name[buf.tagnum],ispointer);
    break;
  case 'b':
    sprintf(vtype,"unsigned char %s",ispointer);
    break;
  case 'c':
    sprintf(vtype,"char %s",ispointer);
    break;
  case 'r':
    sprintf(vtype,"unsigned short %s",ispointer);
    break;
  case 's':
    sprintf(vtype,"short %s",ispointer);
    break;
  case 'h':
    sprintf(vtype,"unsigned int %s",ispointer);
    break;
  case 'i':
    sprintf(vtype,"int %s",ispointer);
    break;
  case 'k':
    sprintf(vtype,"unsigned long %s",ispointer);
    break;
  case 'l':
    sprintf(vtype,"long %s",ispointer);
    break;
  case 'f':
    sprintf(vtype,"float %s",ispointer);
    break;
  case 'd':
    sprintf(vtype,"double %s",ispointer);
    break;
  case 'e':
    sprintf(vtype,"FILE %s",ispointer);
    break;
  case 'y':
    sprintf(vtype,"void %s",ispointer);
    break;
  case 'w':
    sprintf(vtype,"logic %s",ispointer);
    break;
  case 0:
    sprintf(vtype,"NULL %s",ispointer);
    break;
  case 'p':
    sprintf(vtype,"macro");
    break;
  case 'o':
    sprintf(vtype,"automatic");
    break;
#ifndef G__OLDIMPLEMENTATION1604
  case 'g':
    sprintf(vtype,"bool");
    break;
#endif
  default:
    sprintf(vtype,"unknown %s",ispointer);
    break;
  }
  if(type) strcpy(type,vtype);
#ifndef G__OLDIMPLEMENTATION2108
  if(tagname && buf.tagnum>=0) strcpy(tagname,G__struct.name[buf.tagnum]);
  if(typename && buf.typenum>=0) strcpy(typename,G__newtype.name[buf.typenum]);
#else
  if(tagname) strcpy(tagname,G__struct.name[buf.tagnum]) ;
  if(typename) strcpy(typename,G__newtype.name[buf.typenum]) ;
#endif
  
  sprintf(vtype,"&%s",name);
  buf = G__calc_internal(vtype);

  return(buf.obj.i);
}
#endif

#endif /* G__SMALLOBJECT */

/**************************************************************************
* G__textprocessing()
**************************************************************************/
int G__textprocessing(fp)
FILE *fp;
{
	return(G__readline(fp,G__oline,G__argb,&G__argn,G__arg));
}

#ifdef G__REGEXP
/**************************************************************************
* G__matchtregex()
**************************************************************************/
int G__matchregex(pattern,string)
char *pattern;
char *string;
{
  int i;
  regex_t re;
  /* char buf[256]; */
  i=regcomp(&re,pattern,REG_EXTENDED|REG_NOSUB);
  if(i!=0) return(0); 
  i=regexec(&re,string,(size_t)0,(regmatch_t*)NULL,0);
  regfree(&re);
  if(i!=0) return(0); 
  return(1); /* match */
}
#endif

#ifdef G__REGEXP1
/**************************************************************************
* G__matchtregex()
**************************************************************************/
int G__matchregex(pattern,string)
char *pattern;
char *string;
{
  char *re, *s;
  /* char buf[256]; */
  re=regcmp(pattern, NULL);
  if(re==0) return(0);
  s=regex(re,string);
  free(re);
  if(s==0) return(0);
  return(1); /* match */
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
