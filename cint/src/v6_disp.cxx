/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file disp.c
 ************************************************************************
 * Description:
 *  Display information
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


#include "common.h"

int G__browsing=1; /* used in disp.c and intrpt.c */

 
#ifndef G__OLDIMPLEMENTATION2192
#ifndef G__OLDIMPLEMENTATION1878
#ifndef __CINT__
/**************************************************************************
* G__strtoll, G__strtoull
**************************************************************************/
#include <ctype.h>
#include <errno.h>

#ifndef ULONG_LONG_MAX
/*#define       ULONG_LONG_MAX  ((G__uint64)(~0LL))*/
#define       ULONG_LONG_MAX  (~((G__uint64)0))
#endif

#ifndef LONG_LONG_MAX
#define       LONG_LONG_MAX   ((G__int64)(ULONG_LONG_MAX >> 1))
#endif

#ifndef LONG_LONG_MIN
#define       LONG_LONG_MIN   ((G__int64)(~LONG_LONG_MAX))
#endif


/*
 * Convert a string to a long long integer.
 *
 * Ignores `locale' stuff.  Assumes that the upper and lower case
 * alphabets and digits are each contiguous.
 */
G__int64 G__expr_strtoll(const char *nptr,char **endptr, register int base) {
   register const char *s = nptr;
   register G__uint64 acc;
   register int c;
   register G__uint64 cutoff;
   register int neg = 0, any, cutlim;

   /*
    * Skip white space and pick up leading +/- sign if any.
    * If base is 0, allow 0x for hex and 0 for octal, else
    * assume decimal; if base is already 16, allow 0x.
    */
   do {
      c = *s++;
   }
   while (isspace(c));
   if (c == '-') {
      neg = 1;
      c = *s++;
   } else if (c == '+')
      c = *s++;
   if ((base == 0 || base == 16) && c == '0' && (*s == 'x' || *s == 'X')) {
      c = s[1];
      s += 2;
      base = 16;
   }
   if (base == 0)
      base = c == '0' ? 8 : 10;

   /*
    * Compute the cutoff value between legal numbers and illegal
    * numbers.  That is the largest legal value, divided by the
    * base.  An input number that is greater than this value, if
    * followed by a legal input character, is too big.  One that
    * is equal to this value may be valid or not; the limit
    * between valid and invalid numbers is then based on the last
    * digit.  For instance, if the range for long longs is
    * [-2147483648..2147483647] and the input base is 10,
    * cutoff will be set to 214748364 and cutlim to either
    * 7 (neg==0) or 8 (neg==1), meaning that if we have accumulated
    * a value > 214748364, or equal but the next digit is > 7 (or 8),
    * the number is too big, and we will return a range error.
    *
    * Set any if any `digits' consumed; make it negative to indicate
    * overflow.
    */
   cutoff = neg ? -(G__uint64) LONG_LONG_MIN : LONG_LONG_MAX;
   cutlim = cutoff % (G__uint64) base;
   cutoff /= (G__uint64) base;
   for (acc = 0, any = 0;; c = *s++) {
      if (isdigit(c))
         c -= '0';
      else if (isalpha(c))
         c -= isupper(c) ? 'A' - 10 : 'a' - 10;
      else
         break;
      if (c >= base)
         break;
      if (any<0 || acc>cutoff || (acc==cutoff && c>cutlim) )
         any = -1;
      else {
         any = 1;
         acc *= base;
         acc += c;
      }
   }
   if (any < 0) {
      acc = neg ? LONG_LONG_MIN : LONG_LONG_MAX;
      errno = ERANGE;
   } else if (neg)
      acc = -acc;
   if (endptr != 0)
      *endptr = (char *) (any ? s - 1 : nptr);
   return (acc);
}

/*
 * Convert a string to an unsigned long integer.
 *
 * Ignores `locale' stuff.  Assumes that the upper and lower case
 * alphabets and digits are each contiguous.
 */
G__uint64 G__expr_strtoull(const char *nptr, char **endptr, register int base) {
   register const char *s = nptr;
   register G__uint64 acc;
   register int c;
   register G__uint64 cutoff;
   register int neg = 0, any, cutlim;

   /*
    * See strtoll for comments as to the logic used.
    */
   do {
      c = *s++;
   }
   while (isspace(c));
   if (c == '-') {
      neg = 1;
      c = *s++;
   } else if (c == '+')
      c = *s++;
   if ((base == 0 || base == 16) && c == '0' && (*s == 'x' || *s == 'X')) {
      c = s[1];
      s += 2;
      base = 16;
   }
   if (base == 0)
      base = c == '0' ? 8 : 10;
   cutoff =
       (G__uint64) ULONG_LONG_MAX / (G__uint64) base;
   cutlim =
       (G__uint64) ULONG_LONG_MAX % (G__uint64) base;
   for (acc = 0, any = 0;; c = *s++) {
      if (isdigit(c))
         c -= '0';
      else if (isalpha(c))
         c -= isupper(c) ? 'A' - 10 : 'a' - 10;
      else
         break;
      if (c >= base)
         break;
      if (any < 0 || acc > cutoff || (acc == cutoff && c > cutlim))
         any = -1;
      else {
         any = 1;
         acc *= base;
         acc += c;
      }
   }
   if (any < 0) {
      acc = ULONG_LONG_MAX;
      errno = ERANGE;
   } else if (neg)
      acc = -acc;
   if (endptr != 0)
      *endptr = (char *) (any ? s - 1 : nptr);
   return (acc);
}
#endif /* __CINT__ */
#endif /* 1878 */
#endif /* 2192 */

#ifndef G__OLDIMPLEMENTATION713
/***********************************************************************
* G__redirected_on()
* G__redirected_off()
***********************************************************************/
static int G__redirected = 0;
void G__redirect_on() { G__redirected = 1; }
void G__redirect_off() { G__redirected = 0; }
#endif

#ifndef G__OLDIMPLEMENTATION640
static int G__more_len;
/***********************************************************************
* G__more_col()
***********************************************************************/
void G__more_col(len)
int len;
{
  G__more_len += len;
}

/***********************************************************************
* G__more_pause()
***********************************************************************/
int G__more_pause(fp,len)
FILE* fp;
int len;
{
  static int shownline = 0;
  static int dispsize = 22;
  static int dispcol = 80;
  static int store_dispsize = 0;
  static int onemore=0;

  G__more_len += len;

  /*************************************************
  * initialization
  *************************************************/
  if(!fp) {
    shownline = 0;
#ifdef G__OLDIMPLEMENTATION1300
#ifndef G__ROOT
    if(1>dispsize) {
#endif
#endif
      if(store_dispsize>0) dispsize=store_dispsize;
      else {
	char* lines;
	lines = getenv("LINES");
	if(lines)  dispsize=atoi(lines)-2;
	else       dispsize=22;
	lines = getenv("COLUMNS");
	if(lines)  dispcol=atoi(lines);
	else       dispcol=80;
      }
#ifdef G__OLDIMPLEMENTATION1300
#ifndef G__ROOT
    }
#endif
#endif
    G__more_len=0;
    return(0);
  }

#ifndef G__OLDIMPLEMENTATION713
  if(fp==G__stdout && 0<dispsize && 0==G__redirected ) {
#else
  if(fp==G__stdout && 0<dispsize) {
#endif
    /* ++shownline; */
    shownline += (G__more_len/dispcol + 1);
    /*DEBUG printf("(%d,%d,%d)",G__more_len,dispcol,shownline); */
    /*************************************************
     * judgement for pause
     *************************************************/
    if(shownline>=dispsize || onemore) {
      char buf[G__MAXNAME];
      shownline=0;
      strcpy(buf,G__input("-- Press return for more -- (input [number] of lines, Cont,Step,More) "));
      if(isdigit(buf[0])) { /* change display size */
	dispsize = G__int(G__calc_internal(buf));
	if(dispsize>0) store_dispsize = dispsize;
        onemore=0;
      }
      else if('c'==tolower(buf[0])) { /* continue to the end */
	dispsize = 0;
        onemore=0;
      }
      else if('s'==tolower(buf[0])) { /* one more line */
	onemore = 1;
      }
      else if('q'==tolower(buf[0])) { /* one more line */
	onemore=0;
	G__more_len=0;
	return(1);
      }
      else if(isalpha(buf[0])||isspace(buf[0])) { /* more lines */
	onemore = 0;
      }
    }
  }
  G__more_len=0;
  return(0);
}

/***********************************************************************
* G__more()
***********************************************************************/
int G__more(fp,msg)
FILE* fp;
char* msg;
{
#ifndef G__OLDIMPLEMENTATION1485
#ifndef G__OLDIMPLEMENTATION1759
  if(fp==G__serr) G__fprinterr(G__serr,"%s",msg);
  else fprintf(fp,"%s",msg);
#else
  if(fp==G__serr) G__fprinterr(G__serr,msg);
  else fprintf(fp,msg);
#endif
#else
#ifndef G__OLDIMPLEMENTATION1759
  fprintf(fp,"%s",msg);
#else
  fprintf(fp,msg);
#endif
#endif
  if(strchr(msg,'\n')) {
    return(G__more_pause(fp,strlen(msg)));
  }
  else {
    G__more_col(strlen(msg));
    return(0);
  }
}
#endif /* ON640 */

#ifndef G__OLDIMPLEMENTATION2221
/***********************************************************************
* void G__disp_purevirtualfunc
***********************************************************************/
void G__display_purevirtualfunc(tagnum)
int tagnum;
{
  /* to be implemented */
}
#endif

#ifndef G__OLDIMPLEMENTATION1711
/***********************************************************************
* void G__disp_friend
***********************************************************************/
int G__display_friend(fp,friendtag)
FILE *fp;
struct G__friendtag* friendtag;
{
  char msg[G__LONGLINE];
  sprintf(msg," friend ");
  if(G__more(fp,msg)) return(1);
  while(friendtag) {
    sprintf(msg,"%s,",G__fulltagname(friendtag->tagnum,1));
    if(G__more(fp,msg)) return(1);
    friendtag = friendtag->next;
  }
  return(0);
}
#endif

/***********************************************************************
* void G__listfunc
***********************************************************************/
int G__listfunc(fp,access,fname,ifunc)
FILE *fp;
int access;
char *fname;
struct G__ifunc_table *ifunc;
{
  int i,n;
  char temp[G__ONELINE];
  char msg[G__LONGLINE];

  G__browsing=1;
  
#ifndef G__OLDIMPLEMENTATION511
  if(!ifunc) ifunc = G__p_ifunc;
#else
  if(G__exec_memberfunc) {
    G__incsetup_memfunc(G__tagnum);
    ifunc = G__struct.memfunc[G__tagnum] ;
  }
  else {
    ifunc = G__p_ifunc;
  }
#endif
  
  sprintf(msg,"%-15sline:size busy function type and name  ","filename");
  if(G__more(fp,msg)) return(1);

  if(-1!=ifunc->tagnum) {
    sprintf(msg,"(in %s)\n",G__struct.name[ifunc->tagnum]);
    if(G__more(fp,msg)) return(1);
  }
  else {
    if(G__more(fp,"\n")) return(1);
  }
  
  /***************************************************
   * while interpreted function table list exists
   ***************************************************/
  while(ifunc) {
    for(i=0;i<ifunc->allifunc;i++) {

      if(!G__browsing) return(0);

      if(fname && strcmp(fname,ifunc->funcname[i])!=0) continue;
      
      if(
#ifndef G__OLDIMPLEMENTATION2044
	 ifunc->hash[i] &&
#endif
	 (ifunc->access[i]&access)) {
	
	/* print out file name and line number */
	if(ifunc->pentry[i]->filenum>=0) {
	  sprintf(msg,"%-15s%4d:%-3d%c%2d "
#ifndef G__OLDIMPLEMENTATION1196
		  ,G__stripfilename(G__srcfile[ifunc->pentry[i]->filenum].filename)
#else
		  ,G__srcfile[ifunc->pentry[i]->filenum].filename
#endif
		  ,ifunc->pentry[i]->line_number
#ifdef G__ASM_FUNC
		  ,ifunc->pentry[i]->size
#else
		  ,0
#endif
#ifdef G__ASM_WHOLEFUNC
		  ,(ifunc->pentry[i]->bytecode)? '*' : ' '
#else
		  ,' '
#endif
#ifndef G__OLDIMPLEMENTATION1730
		  ,G__globalcomp?ifunc->globalcomp[i]:ifunc->busy[i]
#else
		  ,ifunc->busy[i]
#endif
		  );
	  if(G__more(fp,msg)) return(1);
#ifdef G__ASM_DBG
	  if(ifunc->pentry[i]->bytecode) {
#ifndef G__OLDIMPLEMENTATION1164
	    G__ASSERT(ifunc->pentry[i]->bytecodestatus==G__BYTECODE_SUCCESS||
		      ifunc->pentry[i]->bytecodestatus==G__BYTECODE_ANALYSIS);
#else
	    G__ASSERT(ifunc->pentry[i]->bytecodestatus==G__BYTECODE_SUCCESS);
#endif
	  }
#ifndef G__OLDIMPLEMENTATIN2021
	  else if(ifunc->pentry[i]->size<0) {
	  }
#endif
	  else {
	    G__ASSERT(ifunc->pentry[i]->bytecodestatus==G__BYTECODE_FAILURE||
		      ifunc->pentry[i]->bytecodestatus==G__BYTECODE_NOTYET);
	  }
	  if(ifunc->pentry[i]->bytecodestatus==G__BYTECODE_SUCCESS
#ifndef G__OLDIMPLEMENTATION1164
	     ||ifunc->pentry[i]->bytecodestatus==G__BYTECODE_ANALYSIS
#endif
	     ) {
	    G__ASSERT(ifunc->pentry[i]->bytecode);
	  }
	  else {
	    G__ASSERT(!ifunc->pentry[i]->bytecode);
	  }
#endif
	}
	else {
	  sprintf(msg,"%-15s%4d:%-3d%3d " ,"(compiled)" ,0,0 ,ifunc->busy[i]);
	  if(G__more(fp,msg)) return(1);
	}
	
	if(ifunc->hash[i])
	  sprintf(msg,"%s ",G__access2string(ifunc->access[i]));
	else
	  sprintf(msg,"------- ");
	if(G__more(fp,msg)) return(1);
#ifndef G__OLDIMPLEMENTATION1250
	if(ifunc->isexplicit[i]) {
	  sprintf(msg,"explicit ");
	  if(G__more(fp,msg)) return(1);
	}
#endif
#ifndef G__NEWINHERIT
	if(ifunc->isinherit[i]) { 
	  sprintf(msg,"inherited ");
	  if(G__more(fp,msg)) return(1);
	}
#endif
	if(ifunc->isvirtual[i]) {
	  sprintf(msg,"virtual ");
	  if(G__more(fp,msg)) return(1);
	}

	if(ifunc->staticalloc[i]) {
	  sprintf(msg,"static ");
	  if(G__more(fp,msg)) return(1);
	}

#ifdef G__OLDIMPLEMENTATION401
#ifndef G__OLDIMPLEMENATTION379
	if(ifunc->isconst[i]&G__CONSTVAR) {
	  sprintf(msg,"const ");
	  if(G__more(fp,msg)) return(1);
	}
#else
	if(ifunc->isconst[i]) {
	  sprintf(msg,"const ");
	  if(G__more(fp,msg)) return(1);
	}
#endif
#endif
	
	/* print out type of return value */
#ifndef G__OLDIMPLEMENTATION401
	sprintf(msg,"%s ",G__type2string(ifunc->type[i]
					,ifunc->p_tagtable[i]
					,ifunc->p_typetable[i]
					,ifunc->reftype[i]
					,ifunc->isconst[i]));
#else
	sprintf(msg,"%s ",G__type2string(ifunc->type[i]
					,ifunc->p_tagtable[i]
					,ifunc->p_typetable[i]
					,ifunc->reftype[i]));
#endif
	if(G__more(fp,msg)) return(1);
	
	/*****************************************************
	 * to get type of function parameter
	 *****************************************************/
	/**********************************************************
	 * print out type and name of function and parameters
	 **********************************************************/
	/* print out function name */
#ifndef G__OLDIMPLEMENTATION1803
	if(strlen(ifunc->funcname[i])>=sizeof(msg)-6) {
	  strncpy(msg,ifunc->funcname[i],sizeof(msg)-3);
	  msg[sizeof(msg)-6]=0;
	  strcat(msg,"...(");
	}
	else {
	  sprintf(msg,"%s(",ifunc->funcname[i]);
	}
#else
	sprintf(msg,"%s(",ifunc->funcname[i]);
#endif
	if(G__more(fp,msg)) return(1);

	if(ifunc->ansi[i] && 0==ifunc->para_nu[i]) {
	  sprintf(msg,"void");
	  if(G__more(fp,msg)) return(1);
	}
	
	/* print out parameter types */
	for(n=0;n<ifunc->para_nu[i];n++) {
	  
	  if(n!=0) {
	    sprintf(msg,",");
	    if(G__more(fp,msg)) return(1);
	  }
	  /* print out type of return value */
#ifndef G__OLDIMPLEMENATTION401
	  sprintf(msg,"%s",G__type2string(ifunc->para_type[i][n]
					 ,ifunc->para_p_tagtable[i][n]
					 ,ifunc->para_p_typetable[i][n]
					 ,ifunc->para_reftype[i][n]
					 ,ifunc->para_isconst[i][n]));
#else
	  sprintf(msg,"%s",G__type2string(ifunc->para_type[i][n]
					 ,ifunc->para_p_tagtable[i][n]
					 ,ifunc->para_p_typetable[i][n]
					 ,ifunc->para_reftype[i][n]));
#endif
	  if(G__more(fp,msg)) return(1);

	  if(ifunc->para_name[i][n]) {
	    sprintf(msg," %s",ifunc->para_name[i][n]);
	    if(G__more(fp,msg)) return(1);
	  }
	  if(ifunc->para_def[i][n]) {
	    sprintf(msg,"=%s",ifunc->para_def[i][n]);
	    if(G__more(fp,msg)) return(1);
	  }
	}
#ifndef G__OLDIMPLEMENTATION1471
	if(2==ifunc->ansi[i]) {
	  sprintf(msg," ...");
	  if(G__more(fp,msg)) return(1);
	}
#endif
	sprintf(msg,")");
	if(G__more(fp,msg)) return(1);
	if(ifunc->isconst[i]&G__CONSTFUNC) {
	  sprintf(msg," const");
	  if(G__more(fp,msg)) return(1);
	}
	if(ifunc->ispurevirtual[i]) {
	  sprintf(msg,"=0");
	  if(G__more(fp,msg)) return(1);
	}
	sprintf(msg,";");
	if(G__more(fp,msg)) return(1);
	temp[0] = '\0';
	G__getcomment(temp,&ifunc->comment[i],ifunc->tagnum);
	if(temp[0]) {
	  sprintf(msg," //%s",temp);
	  if(G__more(fp,msg)) return(1);
	}
#ifndef G__OLDIMPLEMENTATION1711
	if(ifunc->friendtag[i]) 
	  if(G__display_friend(fp,ifunc->friendtag[i])) return(1);
#endif
	if(G__more(fp,"\n")) return(1);
      }
      
    }
    /***************************************************
     * next page of interpterive function table
     ***************************************************/
    ifunc=ifunc->next;
  } /* end of while(ifunc) */

  return(0);
}




/**************************************************************************
* G__showstack()
*
**************************************************************************/
int G__showstack(fout)
FILE *fout;
{
  int temp,temp1;
  struct G__var_array *local;
  char syscom[G__MAXNAME];
  char msg[G__LONGLINE];

  local=G__p_local;
  temp=0;
  while(local) {
#ifdef G__VAARG
    sprintf(msg,"%d ",temp);
    if(G__more(fout,msg)) return(1);
    if(local->exec_memberfunc && -1!=local->tagnum) {
      sprintf(msg,"%s::",G__struct.name[local->tagnum]);
      if(G__more(fout,msg)) return(1);
    }
    sprintf(msg,"%s(",local->ifunc->funcname[local->ifn]);
    if(G__more(fout,msg)) return(1);
    for(temp1=0;temp1<local->libp->paran;temp1++) {
      if(temp1) {
	sprintf(msg,",");
	if(G__more(fout,msg)) return(1);
      }
      G__valuemonitor(local->libp->para[temp1],syscom);
      if(G__more(fout,syscom)) return(1);
    }
    if(-1!=local->prev_filenum) {
      sprintf(msg,") [%s: %d]\n" 
#ifndef G__OLDIMPLEMENTATION1196
	      ,G__stripfilename(G__srcfile[local->prev_filenum].filename)
#else
	      ,G__srcfile[local->prev_filenum].filename
#endif
	      ,local->prev_line_number);
      if(G__more(fout,msg)) return(1);
    }
    else {
      if(G__more(fout,") [entry]\n")) return(1);
    }
#else
    sprintf(msg,"%d %s() [%s: %d]\n" ,temp ,local->ifunc->funcname[local->ifn]
	    ,G__filenameary[local->prev_filenum] ,local->prev_line_number);
    if(G__more(fout,msg)) return(1) ;
#endif
    ++temp;
    local=local->prev_local;
  }
  return(0);
}

#ifndef G__OLDIMPLEMENTATION873
/**************************************************************************
* G__getdictpos()
**************************************************************************/
struct G__dictposition* G__get_dictpos(fname)
char *fname;
{
  struct G__dictposition *dict = (struct G__dictposition*)NULL;
  int i;
  /* search for source file entry */
  for(i=0;i<G__nfile;i++) {
#ifndef G__OLDIMPLEMENTATION1196
    if(G__matchfilename(i,fname)) {
#else
    if(strcmp(G__srcfile[i].filename,fname)==0) {
#endif
      dict = G__srcfile[i].dictpos;
      break;
    }
  }
  return(dict);
}

/**************************************************************************
* G__display_newtypes()
*
**************************************************************************/
int G__display_newtypes(fout,fname)
FILE* fout;
char *fname;
{
  struct G__dictposition *dict = (struct G__dictposition*)NULL;
  int i;

  /* search for source file entry */
  for(i=0;i<G__nfile;i++) {
#ifndef G__OLDIMPLEMENTATION1196
    if(G__matchfilename(i,fname)) {
#else
    if(strcmp(G__srcfile[i].filename,fname)==0) {
#endif
      dict = G__srcfile[i].dictpos;
      break;
    }
  }

  if(dict) {
    /* listup new class/struct/enum/union */
    if(G__display_class(fout,"",0,dict->tagnum)) return(1);
    /* listup new typedef */
    if(G__display_typedef(fout,"",dict->typenum)) return(1);
    return(0);
  }

  G__fprinterr(G__serr,"File %s is not loaded\n",fname);
  return(1);
}
#endif

/**************************************************************************
* G__display_string()
*
**************************************************************************/
int G__display_string(fout)
FILE *fout;
{
  int len;
  unsigned long totalsize=0;
  struct G__ConstStringList *pconststring;
  char msg[G__ONELINE];

  pconststring = G__plastconststring;
  while(pconststring->prev) {
    len=strlen(pconststring->string);
    totalsize+=len+1;
#ifndef G__OLDIMPLEMENTATION1803
    if(totalsize>=sizeof(msg)-5) {
      sprintf(msg,"%3d ",len);
      strncpy(msg+4,pconststring->string,sizeof(msg)-5);
      msg[sizeof(msg)-1]=0;
    }
    else {
      sprintf(msg,"%3d %s\n",len,pconststring->string);
    }
#else
    sprintf(msg,"%3d %s\n",len,pconststring->string);
#endif
    if(G__more(fout,msg)) return(1);
    pconststring=pconststring->prev;
  }
  sprintf(msg,"Total string constant size = %ld\n",totalsize);
  if(G__more(fout,msg)) return(1);
  return(0);
}

/****************************************************************
* G__display_classinheritance()
*
****************************************************************/
static int G__display_classinheritance(fout,tagnum,space)
FILE *fout;
int tagnum;
char *space;
{
  int i;
  struct G__inheritance *baseclass;
  char addspace[50];
  char temp[G__ONELINE];
  char msg[G__LONGLINE];

  baseclass = G__struct.baseclass[tagnum];

  if(NULL==baseclass) return(0);

  sprintf(addspace,"%s  ",space);

  for(i=0;i<baseclass->basen;i++) {
    if(baseclass->property[i]&G__ISDIRECTINHERIT) {
      sprintf(msg,"%s0x%-8lx ",space ,baseclass->baseoffset[i]);
      if(G__more(fout,msg)) return(1);
      if(baseclass->property[i]&G__ISVIRTUALBASE) {
	sprintf(msg,"virtual ");
	if(G__more(fout,msg)) return(1);
      }
#ifndef G__OLDIMPLEMENTATION2151
      if(baseclass->property[i]&G__ISINDIRECTVIRTUALBASE) {
	sprintf(msg,"(virtual) ");
	if(G__more(fout,msg)) return(1);
      }
#endif
      sprintf(msg,"%s %s"
	      ,G__access2string(baseclass->baseaccess[i])
	      ,G__fulltagname(baseclass->basetagnum[i],0));
      if(G__more(fout,msg)) return(1);
      temp[0]='\0';
      G__getcomment(temp,&G__struct.comment[baseclass->basetagnum[i]]
		    ,baseclass->basetagnum[i]);
      if(temp[0]) {
	sprintf(msg," //%s",temp);
	if(G__more(fout,msg)) return(1);
      }
      if(G__more(fout,"\n")) return(1);
      if(G__display_classinheritance(fout,baseclass->basetagnum[i],addspace))
	return(1);
    }
  }
  return(0);
}

/****************************************************************
* G__display_membervariable()
*
****************************************************************/
static int G__display_membervariable(fout,tagnum,base)
FILE *fout;
int tagnum;
int base;
{
  struct G__var_array *var;
  struct G__inheritance *baseclass;
  int i;
  baseclass = G__struct.baseclass[tagnum];

  if(base) {
    for(i=0;i<baseclass->basen;i++) {
      if(!G__browsing) return(0);
      if(baseclass->property[i]&G__ISDIRECTINHERIT) {
	if(G__display_membervariable(fout,baseclass->basetagnum[i],base))
	  return(1);
      }
    }
  }

  G__incsetup_memvar(tagnum);
  var = G__struct.memvar[tagnum];
  /* member variable */
  if(var) {
    fprintf(fout,"Defined in %s\n",G__struct.name[tagnum]);
    if(G__more_pause(fout,1)) return(1);
    if(G__varmonitor(fout,var,"","",(long)(-1))) return(1);
  }
  return(0);
}

/****************************************************************
* G__display_memberfunction()
*
****************************************************************/
static int G__display_memberfunction(fout,tagnum,access,base)
FILE *fout;
int tagnum;
int access;
int base;
{
  struct G__ifunc_table *store_ifunc;
  int store_exec_memberfunc;
  struct G__inheritance *baseclass;
  int i;
  int tmp;
  baseclass = G__struct.baseclass[tagnum];

  if(base) {
    for(i=0;i<baseclass->basen;i++) {
      if(!G__browsing) return(0);
      if(baseclass->property[i]&G__ISDIRECTINHERIT) {
	if(G__display_memberfunction(fout,baseclass->basetagnum[i]
				     ,access,base)) return(1);
      }
    }
  }

  /* member function */
  if(G__struct.memfunc[tagnum]) {
    G__incsetup_memfunc(tagnum);
#ifdef G__OLDIMPLEMENTATION1079
    fprintf(fout,"Defined in %s\n",G__struct.name[tagnum]);
    if(G__more_pause(fout,1)) return(1);
#endif
    store_ifunc = G__p_ifunc;
    store_exec_memberfunc=G__exec_memberfunc;
    G__p_ifunc = G__struct.memfunc[tagnum];
    G__exec_memberfunc=0;
    tmp=G__listfunc(fout,access,(char*)NULL,(struct G__ifunc_table*)NULL);
    G__p_ifunc=store_ifunc;
    G__exec_memberfunc=store_exec_memberfunc;
    if(tmp) return(1);
  }
  return(0);
}
  
#ifndef G__OLDIMPLEMENTATION2014
extern int G__class_autoloading G__P((int tagnum));
#endif

/****************************************************************
* G__display_class()
*
****************************************************************/
int G__display_class(fout,name,base,start)
FILE *fout;
char *name;
int base;
int start;
{
  int tagnum;
  int i,j;
  struct G__inheritance *baseclass;
  char temp[G__ONELINE];
  char msg[G__LONGLINE];
  char *p;
#ifndef G__OLDIMPLEMENTATION1085
  int store_globalcomp;
  int store_iscpp;
#endif

  G__browsing=1;

  i=0;
  while(isspace(name[i])) i++;

  /*******************************************************************
  * List of classes
  *******************************************************************/
  if('\0'==name[i]) {
#ifndef G__OLDIMPLEMENTATION1514
    if(base) {
      /* In case of 'Class' command */
      for(i=0;i<G__struct.alltag;i++) {
	sprintf(temp,"%d",i);
	G__display_class(fout,temp,0,0);
      }
      return(0);
    }
#endif
    /* no class name specified, list up all tagnames */
    if(G__more(fout,"List of classes\n")) return(1);
    sprintf(msg,"%-15s%5s\n","file","line");
    if(G__more(fout,msg)) return(1);
    for(i=start;i<G__struct.alltag;i++) {
      if(!G__browsing) return(0);
      switch(G__struct.iscpplink[i]) {
#ifndef G__OLDIMPLEMENTATION2012
      case G__CLINK:
	if (G__struct.filenum[i] == -1) sprintf(msg,"%-20s " ,"(C compiled)");
	else
	  sprintf(msg,"%-15s%5d " 
		  ,G__stripfilename(G__srcfile[G__struct.filenum[i]].filename)
		  ,G__struct.line_number[i]);
	if(G__more(fout,msg)) return(1);
	break;
      case G__CPPLINK:
	if (G__struct.filenum[i] == -1) sprintf(msg,"%-20s " ,"(C++ compiled)");
	else
	  sprintf(msg,"%-15s%5d " 
		  ,G__stripfilename(G__srcfile[G__struct.filenum[i]].filename)
		  ,G__struct.line_number[i]);
	if(G__more(fout,msg)) return(1);
	break;
#else
      case G__CLINK:
	sprintf(msg,"%-20s " ,"(C compiled)");
	if(G__more(fout,msg)) return(1);
	break;
      case G__CPPLINK:
	sprintf(msg,"%-20s " ,"(C++ compiled)");
	if(G__more(fout,msg)) return(1);
	break;
#endif
      case 1:
	sprintf(msg,"%-20s " ,"(C compiled old 1)");
	if(G__more(fout,msg)) return(1);
	break;
      case 2:
	sprintf(msg,"%-20s " ,"(C compiled old 2)");
	if(G__more(fout,msg)) return(1);
	break;
      case 3:
	sprintf(msg,"%-20s " ,"(C compiled old 3)");
	if(G__more(fout,msg)) return(1);
	break;
      default:
	if (G__struct.filenum[i] == -1)
	  sprintf(msg,"%-20s " ," ");
	else
	  sprintf(msg,"%-15s%5d " 
#ifndef G__OLDIMPLEMENTATION1196
		  ,G__stripfilename(G__srcfile[G__struct.filenum[i]].filename)
#else
		  ,G__srcfile[G__struct.filenum[i]].filename
#endif
		  ,G__struct.line_number[i]);
	if(G__more(fout,msg)) return(1);
	break;
      }
      if(G__struct.isbreak[i]) fputc('*',fout);
      else                     fputc(' ',fout);
      if(G__struct.istrace[i]) fputc('-',fout);
      else                     fputc(' ',fout);
      G__more_col(2);
#ifndef G__OLDIMPLEMENTATION1085
      store_iscpp=G__iscpp; /* This is a dirty trick to display 'class' */
      G__iscpp=0;           /* 'struct','union' or 'namespace' in msg   */
      store_globalcomp=G__globalcomp;
      G__globalcomp=G__NOLINK;
#endif
      sprintf(msg," %s ",G__type2string('u',i,-1,0,0));
#ifndef G__OLDIMPLEMENTATION1085
      G__iscpp=store_iscpp; /* dirty trick reset */
      G__globalcomp=store_globalcomp;
#endif
      if(G__more(fout,msg)) return(1);
      baseclass = G__struct.baseclass[i];
      if(baseclass) {
	for(j=0;j<baseclass->basen;j++) {
	  if(baseclass->property[j]&G__ISDIRECTINHERIT) {
	    if(baseclass->property[j]&G__ISVIRTUALBASE) {
	      sprintf(msg,"virtual ");
	      if(G__more(fout,msg)) return(1);
	    }
	    sprintf(msg,"%s%s " 
		    ,G__access2string(baseclass->baseaccess[j])
		    ,G__fulltagname(baseclass->basetagnum[j],0));
	    if(G__more(fout,msg)) return(1);
	  }
	}
      }
      if('$'==G__struct.name[i][0]) {
	sprintf(msg," (typedef %s)",G__struct.name[i]+1);
	if(G__more(fout,msg)) return(1);
      }
      temp[0]='\0';
      G__getcomment(temp,&G__struct.comment[i],i);
      if(temp[0]) {
	sprintf(msg," //%s",temp);
	if(G__more(fout,msg)) return(1);
      }
      if(G__more(fout,"\n")) return(1);
    }
    return(0);
  }

  /*******************************************************************
  * Detail of a specific class
  *******************************************************************/

  p = name+i+strlen(name+i)-1;
  while(isspace(*p)) {
    *p = '\0';
    --p;
  }

#ifndef G__OLDIMPLEMENTATION556
  if((char*)NULL!=strstr(name+i,">>")) {
    /* dealing with A<A<int>> -> A<A<int> > */
    char *pt1;
    char tmpbuf[G__ONELINE];
    pt1 = strstr(name+i,">>");
    ++pt1;
    strcpy(tmpbuf,pt1);
    *pt1=' ';
    ++pt1;
    strcpy(pt1,tmpbuf);
  }
#endif

  if(isdigit(*(name+i))) tagnum = atoi(name+i);
  else                   tagnum = G__defined_tagname(name+i,0);

  /* no such class,struct */
  if(-1==tagnum||G__struct.alltag<=tagnum) return(0); 

#ifndef G__OLDIMPLEMENTATION2014
      G__class_autoloading(tagnum);
#endif

  G__more(fout,"===========================================================================\n");
  sprintf(msg,"%s ",G__tagtype2string(G__struct.type[tagnum]));
  if(G__more(fout,msg)) return(1);
  sprintf(msg,"%s",G__fulltagname(tagnum,0));
  if(G__more(fout,msg)) return(1);
  temp[0]='\0';
  G__getcomment(temp,&G__struct.comment[tagnum],tagnum);
  if(temp[0]) {
    sprintf(msg," //%s",temp);
    if(G__more(fout,msg)) return(1);
  }
  if(G__more(fout,"\n")) return(1);
  if (G__struct.filenum[tagnum] == -1)
    sprintf(msg," size=0x%x\n" ,G__struct.size[tagnum]);
  else {
    sprintf(msg," size=0x%x FILE:%s LINE:%d\n" ,G__struct.size[tagnum]
#ifndef G__OLDIMPLEMENTATION1196
	    ,G__stripfilename(G__srcfile[G__struct.filenum[tagnum]].filename)
#else
	    ,G__srcfile[G__struct.filenum[tagnum]].filename
#endif
	    ,G__struct.line_number[tagnum]);
  }
  if(G__more(fout,msg)) return(1);
  sprintf(msg
	  ," (tagnum=%d,voffset=%d,isabstract=%d,parent=%d,gcomp=%d:%d,d21=~cd=%x)" 
	  ,tagnum ,G__struct.virtual_offset[tagnum]
	  ,G__struct.isabstract[tagnum] ,G__struct.parent_tagnum[tagnum]
	  ,G__struct.globalcomp[tagnum],G__struct.iscpplink[tagnum]
	  ,G__struct.funcs[tagnum]);
  if(G__more(fout,msg)) return(1);
  if('$'==G__struct.name[tagnum][0]) {
    sprintf(msg," (typedef %s)",G__struct.name[tagnum]+1);
    if(G__more(fout,msg)) return(1);
  }
  if(G__more(fout,"\n")) return(1);

  baseclass = G__struct.baseclass[tagnum];

#ifndef G__OLDIMPLEMENTATION2162
  if(G__cintv6) {
    if(G__more(fout,"Virtual table--------------------------------------------------------------\n")) return(1);
    G__bc_disp_vtbl(fout,tagnum);
  }
#endif

  /* inheritance */
  if(baseclass) {
    if(G__more(fout,"List of base class--------------------------------------------------------\n")) return(1);
    if(G__display_classinheritance(fout,tagnum,"")) return(1);
  }

  if(G__more(fout,"List of member variable---------------------------------------------------\n")) return(1);
  if(G__display_membervariable(fout,tagnum,base)) return(1);
  if(!G__browsing) return(0);

  /* member function */
  if(G__more(fout,"List of member function---------------------------------------------------\n")) return(0);
  if(G__display_memberfunction(fout,tagnum,G__PUBLIC_PROTECTED_PRIVATE,base))
    return(1);
  return(0);
}

/****************************************************************
* G__display_typedef()
*
****************************************************************/
int G__display_typedef(fout,name,startin)
FILE *fout;
char *name;
int startin;
{
  int i,j;
  int start,stop;
  char temp[G__ONELINE];
  char msg[G__LONGLINE];

  i=0;
  while(name[i]&&isspace(name[i])) i++;
  if(name[i]) {
    start = G__defined_typename(name+i);
    if(-1==start) {
      G__fprinterr(G__serr,"!!!Type %s is not defined\n",name+i);
      return(0);
    }
    stop = start+1;
  }
  else {
    start = startin;
    stop = G__newtype.alltype;
  }

  G__browsing=1;

  G__more(fout,"List of typedefs\n");
  
  for(i=start;i<stop;i++) {
    if(!G__browsing) return(0);
#ifdef G__TYPEDEFFPOS
    if(G__newtype.filenum[i]>=0) 
      sprintf(msg,"%-15s%4d "
#ifndef G__OLDIMPLEMENTATION1196
	      ,G__stripfilename(G__srcfile[G__newtype.filenum[i]].filename)
#else
	      ,G__srcfile[G__newtype.filenum[i]].filename
#endif
	      ,G__newtype.linenum[i]);
    else
      sprintf(msg,"%-15s     " ,"(compiled)");
    if(G__more(fout,msg)) return(1);
#endif
    if(
#ifndef G__OLDIMPLEMENTATION2191
       '1'==G__newtype.type[i]
#else
       'Q'==G__newtype.type[i]
#endif
       ) {
      /* pointer to statuc function */
      sprintf(msg,"typedef void* %s",G__newtype.name[i]); 
      if(G__more(fout,msg)) return(1);
    }
    else if('a'==G__newtype.type[i]) {
      /* pointer to member */
      sprintf(msg,"typedef G__p2memfunc %s",G__newtype.name[i]); 
      if(G__more(fout,msg)) return(1);
    }
    else {
#ifndef G__OLDIMPLEMENTATION1394
      /* G__typedef may need to be changed to add isconst member */
      sprintf(msg,"typedef %s" ,G__type2string(tolower(G__newtype.type[i])
						,G__newtype.tagnum[i],-1
						,G__newtype.reftype[i]
						,G__newtype.isconst[i])); 
#else
      /* G__typedef may need to be changed to add isconst member */
      sprintf(msg,"typedef %s" ,G__type2string(tolower(G__newtype.type[i])
						,G__newtype.tagnum[i],-1
						,G__newtype.reftype[i]
						,0));  /* isconst */
#endif
      if(G__more(fout,msg)) return(1);
      if(G__more(fout," ")) return(1);
      if(isupper(G__newtype.type[i])&&G__newtype.nindex[i]) {
	if(0<=G__newtype.parent_tagnum[i]) 
	  sprintf(msg,"(*%s::%s)"
		  ,G__fulltagname(G__newtype.parent_tagnum[i],1)
		  ,G__newtype.name[i]);
	else
	  sprintf(msg,"(*%s)",G__newtype.name[i]);
	if(G__more(fout,msg)) return(1);
      }
      else {
	if(isupper(G__newtype.type[i])) {
#ifndef G__OLDIMPLEMENTATION1396
	  if(G__newtype.isconst[i]&G__PCONSTVAR) sprintf(msg,"*const ");
	  else sprintf(msg,"*");
#else
	  sprintf(msg,"*");
#endif
	  if(G__more(fout,msg)) return(1);
	}
	if(0<=G__newtype.parent_tagnum[i]) {
	  sprintf(msg,"%s::",G__fulltagname(G__newtype.parent_tagnum[i],1));
	  if(G__more(fout,msg)) return(1);
	}
	sprintf(msg,"%s",G__newtype.name[i]);
	if(G__more(fout,msg)) return(1);
      }
      for(j=0;j<G__newtype.nindex[i];j++) {
	sprintf(msg,"[%d]",G__newtype.index[i][j]);
	if(G__more(fout,msg)) return(1);
      }
    }
    temp[0]='\0';
    G__getcommenttypedef(temp,&G__newtype.comment[i],i);
    if(temp[0]) {
      sprintf(msg," //%s",temp);
      if(G__more(fout,msg)) return(1);
    }
    if(G__more(fout,"\n")) return(1);
  }
  return(0);
}

/****************************************************************
* G__display_eachtemplate()
*
****************************************************************/
int G__display_eachtemplate(fout,deftmplt,detail)
FILE *fout;
struct G__Definedtemplateclass *deftmplt;
int detail;
{
  struct G__Templatearg *def_para;
  struct G__Definedtemplatememfunc *memfunctmplt;
  fpos_t store_pos;
  /* char buf[G__LONGLINE]; */
  char msg[G__LONGLINE];
  int c;

#ifndef G__OLDIMPLEMENTATION608
  if(!deftmplt->def_fp) return(0);
#endif

  sprintf(msg,"%-20s%5d "
#ifndef G__OLDIMPLEMENTATION1196
	  ,G__stripfilename(G__srcfile[deftmplt->filenum].filename)
#else
	  ,G__srcfile[deftmplt->filenum].filename
#endif
	  ,deftmplt->line);
  if(G__more(fout,msg)) return(1);
  sprintf(msg,"template<");
  if(G__more(fout,msg)) return(1);
  def_para=deftmplt->def_para;
  while(def_para) {
    switch(def_para->type) {
    case G__TMPLT_CLASSARG:
      sprintf(msg,"class ");
      if(G__more(fout,msg)) return(1);
      break;
    case G__TMPLT_TMPLTARG:
      sprintf(msg,"template<class U> class ");
      if(G__more(fout,msg)) return(1);
      break;
    case G__TMPLT_SIZEARG:
      sprintf(msg,"size_t ");
      if(G__more(fout,msg)) return(1);
      break;
    default:
#ifndef G__OLDIMPLEMENTATION401
      sprintf(msg,"%s ",G__type2string(def_para->type,-1,-1,0,0));
#else
      sprintf(msg,"%s ",G__type2string(def_para->type,-1,-1,0));
#endif
      if(G__more(fout,msg)) return(1);
      break;
    }
    sprintf(msg,"%s",def_para->string);
    if(G__more(fout,msg)) return(1);
    def_para=def_para->next;
    if(def_para) fprintf(fout,",");
    else         fprintf(fout,">");
    G__more_col(1);
  }
#ifndef G__OLDIMPLEMENTATION682
  sprintf(msg," class ");
  if(G__more(fout,msg)) return(1);
  if(-1!=deftmplt->parent_tagnum) {
    sprintf(msg,"%s::",G__fulltagname(deftmplt->parent_tagnum,1));
    if(G__more(fout,msg)) return(1);
  }
  sprintf(msg,"%s\n",deftmplt->name);
  if(G__more(fout,msg)) return(1);
#else
  sprintf(msg," class %s\n",deftmplt->name);
  if(G__more(fout,msg)) return(1);
#endif

  if(detail) {
    memfunctmplt = &deftmplt->memfunctmplt;
    while(memfunctmplt->next) {
      sprintf(msg,"%-20s%5d "
#ifndef G__OLDIMPLEMENTATION1196
	      ,G__stripfilename(G__srcfile[memfunctmplt->filenum].filename)
#else
	      ,G__srcfile[memfunctmplt->filenum].filename
#endif
	      ,memfunctmplt->line);
      if(G__more(fout,msg)) return(1);
      fgetpos(memfunctmplt->def_fp,&store_pos);
      fsetpos(memfunctmplt->def_fp,&memfunctmplt->def_pos);
      do {
	c=fgetc(memfunctmplt->def_fp);
	if('\n'==c||'\r'==c) fputc(' ',fout);
	else        fputc(c,fout);
	G__more_col(1);
      } while(';'!=c && '{'!=c) ;
      fputc('\n',fout);
      if(G__more_pause(fout,1)) return(1);
      fsetpos(memfunctmplt->def_fp,&store_pos);
      memfunctmplt=memfunctmplt->next;
    }
  }
#ifndef G__OLDIMPLEMENTATION691
  if(detail) {
    struct G__IntList *ilist = deftmplt->instantiatedtagnum;
    while(ilist) {
      sprintf(msg,"      %s\n",G__fulltagname(ilist->i,1));
      if(G__more(fout,msg)) return(1);
      ilist=ilist->next;
    }
  }
#endif
  return(0);
}

/****************************************************************
* G__display_eachtemplatefunc()
*
****************************************************************/
int G__display_eachtemplatefunc(fout,deftmpfunc)
FILE *fout;
struct G__Definetemplatefunc *deftmpfunc;
{
  char msg[G__LONGLINE];
  struct G__Templatearg *def_para;
  struct G__Templatefuncarg *pfuncpara;
  int i;
  sprintf(msg,"%-20s%5d "
#ifndef G__OLDIMPLEMENTATION1196
	  ,G__stripfilename(G__srcfile[deftmpfunc->filenum].filename)
#else
	  ,G__srcfile[deftmpfunc->filenum].filename
#endif
	  ,deftmpfunc->line);
  if(G__more(fout,msg)) return(1);
  sprintf(msg,"template<");
  if(G__more(fout,msg)) return(1);
  def_para=deftmpfunc->def_para;
  while(def_para) {
    switch(def_para->type) {
    case G__TMPLT_CLASSARG:
#ifndef G__OLDIMPLEMENTATION1026
    case G__TMPLT_POINTERARG1:
    case G__TMPLT_POINTERARG2:
    case G__TMPLT_POINTERARG3:
#endif
      sprintf(msg,"class ");
      if(G__more(fout,msg)) return(1);
      break;
    case G__TMPLT_TMPLTARG:
      sprintf(msg,"template<class U> class ");
      if(G__more(fout,msg)) return(1);
      break;
    case G__TMPLT_SIZEARG:
      sprintf(msg,"size_t ");
      if(G__more(fout,msg)) return(1);
      break;
    default:
#ifndef G__OLDIMPLEMENTATION401
      sprintf(msg,"%s ",G__type2string(def_para->type,-1,-1,0,0));
#else
      sprintf(msg,"%s ",G__type2string(def_para->type,-1,-1,0));
#endif
      if(G__more(fout,msg)) return(1);
      break;
    }
    sprintf(msg,"%s",def_para->string);
    if(G__more(fout,msg)) return(1);
#ifndef G__OLDIMPLEMENTATION1026
    switch(def_para->type) {
    case G__TMPLT_POINTERARG3: fprintf(fout,"*"); G__more_col(1);
    case G__TMPLT_POINTERARG2: fprintf(fout,"*"); G__more_col(1);
    case G__TMPLT_POINTERARG1: fprintf(fout,"*"); G__more_col(1);
    }
#endif
    def_para=def_para->next;
    if(def_para) fprintf(fout,",");
    else         fprintf(fout,">");
    G__more_col(1);
  }
  sprintf(msg," func ");
  if(G__more(fout,msg)) return(1);
  if(-1!=deftmpfunc->parent_tagnum) {
    sprintf(msg,"%s::",G__fulltagname(deftmpfunc->parent_tagnum,1));
    if(G__more(fout,msg)) return(1);
  }
  sprintf(msg,"%s(",deftmpfunc->name);
  if(G__more(fout,msg)) return(1);
  def_para=deftmpfunc->def_para;
  pfuncpara = &deftmpfunc->func_para;
  for(i=0;i<pfuncpara->paran;i++) {
    if(i) {
      sprintf(msg,",");
      if(G__more(fout,msg)) return(1);
    }
    if(pfuncpara->argtmplt[i]>0) {
      sprintf(msg,"%s",G__gettemplatearg(pfuncpara->argtmplt[i],def_para));
      if(G__more(fout,msg)) return(1);
#ifndef G__OLDIMPLEMENTATION1027
      if(isupper(pfuncpara->type[i])) {
	fprintf(fout,"*");
	G__more_col(1);
      }
#endif
    }
#ifndef G__OLDIMPLEMENTATION1025
    else if(pfuncpara->argtmplt[i]<-1) {
#else
    else if(pfuncpara->argtmplt[i]<0) {
#endif
      if(pfuncpara->typenum[i]) 
	sprintf(msg,"%s<",G__gettemplatearg(pfuncpara->typenum[i],def_para));
      else
	sprintf(msg,"X<");
      if(G__more(fout,msg)) return(1);
      if(pfuncpara->tagnum[i]) 
	sprintf(msg,"%s>",G__gettemplatearg(pfuncpara->tagnum[i],def_para));
      else
	sprintf(msg,"Y>");
      if(G__more(fout,msg)) return(1);
    }
    else {
#ifndef G__OLDIMPLEMENTATION401
      sprintf(msg,"%s",G__type2string(pfuncpara->type[i]
				       ,pfuncpara->tagnum[i]
				       ,pfuncpara->typenum[i]
				       ,pfuncpara->reftype[i]
				       ,0));
#else
      sprintf(msg,"%s",G__type2string(pfuncpara->type[i]
				       ,pfuncpara->tagnum[i]
				       ,pfuncpara->typenum[i]
				       ,pfuncpara->reftype[i]));
#endif
      if(G__more(fout,msg)) return(1);
      if(pfuncpara->paradefault[i]) {
	fprintf(fout,"=");
	G__more_col(1);
      }
    }
  }
  if(G__more(fout,");\n")) return(1);
  return(0);
}

/****************************************************************
* G__display_template()
*
****************************************************************/
int G__display_template(fout,name)
FILE *fout;
char *name;
{
  int i /* ,j */;
  struct G__Definedtemplateclass *deftmplt;
  struct G__Definetemplatefunc *deftmpfunc;
  i=0;
  G__browsing=1;
  while(name[i]&&isspace(name[i])) i++;
  if(name[i]) {
    deftmpfunc = &G__definedtemplatefunc;
    while(deftmpfunc->next) {
      if(strcmp(name+i,deftmpfunc->name)==0)
	if(G__display_eachtemplatefunc(fout,deftmpfunc)) return(1);
      deftmpfunc = deftmpfunc->next;
    }
    deftmplt = G__defined_templateclass(name+i);
    if(deftmplt) {
      if(G__display_eachtemplate(fout,deftmplt,1)) return(1);
    }
  }
  else {
    deftmplt = &G__definedtemplateclass;
    while(deftmplt->next) {
      if(!G__browsing) return(0);
      if(strlen(name)) {
	if(G__display_eachtemplate(fout,deftmplt,1)) return(1);
      }
      else {
	if(G__display_eachtemplate(fout,deftmplt,0)) return(1);
      }
      deftmplt=deftmplt->next;
    }
    deftmpfunc = &G__definedtemplatefunc;
    while(deftmpfunc->next) {
      if(G__display_eachtemplatefunc(fout,deftmpfunc)) return(1);
      deftmpfunc = deftmpfunc->next;
    }
  }
  return(0);
}

/****************************************************************
* G__display_includepath()
*
****************************************************************/
int G__display_includepath(fout)
FILE *fout;
{
  fprintf(fout,"include path: %s\n",G__allincludepath);
  return(0);
}

/****************************************************************
* G__display_macro()
*
****************************************************************/
int G__display_macro(fout,name)
FILE *fout;
char *name;
{
  struct G__Deffuncmacro *deffuncmacro;
  struct G__Charlist *charlist;
  int i=0;

#ifndef G__OLDIMPLEMENTATION2034
  struct G__var_array *var = &G__global;
  int ig15;
  char msg[G__LONGLINE];
  while(name[i]&&isspace(name[i])) i++;

  while(var) {
    for(ig15=0;ig15<var->allvar;ig15++) {
      if(name && name[i] && strcmp(name+i,var->varnamebuf[ig15])!=0) continue;
      if('p'==var->type[ig15]) {
	sprintf(msg,"#define %s %d\n",var->varnamebuf[ig15]
		,*(int*)var->p[ig15]);
	G__more(fout,msg);
      }
      else if('T'==var->type[ig15]) {
	sprintf(msg,"#define %s \"%s\"\n",var->varnamebuf[ig15]
		,*(char**)var->p[ig15]);
	G__more(fout,msg);
      }
      if(name && name[i]) return(0);
    }
    var=var->next;
  }

  if(G__display_replacesymbol(fout,name+i)) return(0);

#else
  while(name[i]&&isspace(name[i])) i++;
#endif

  if(name[i]) {
    deffuncmacro = &G__deffuncmacro;
    while(deffuncmacro->next) {
      if(deffuncmacro->name && strcmp(deffuncmacro->name,name+i)==0) {
	fprintf(fout,"#define %s(",deffuncmacro->name);
	charlist = &deffuncmacro->def_para;
	while(charlist) {
	  if(charlist->string) fprintf(fout,"%s",charlist->string);
	  charlist=charlist->next;
	  if(charlist && charlist->next) fprintf(fout,",");
	}
	G__more(fout,")\n");
	return(0);
      }
      deffuncmacro=deffuncmacro->next;
    }
    return(0);
  }

  deffuncmacro = &G__deffuncmacro;
  while(deffuncmacro->next) {
    if(deffuncmacro->name) {
      fprintf(fout,"#define %s(",deffuncmacro->name);
      charlist = &deffuncmacro->def_para;
      while(charlist) {
#ifndef G__OLDIMPLEMENTATION1936
	if(charlist->string) fprintf(fout,"%s%s",charlist->string,"");
#else
	if(charlist->string) fprintf(fout,"%s",charlist->string);
#endif
	charlist=charlist->next;
	if(charlist && charlist->next) fprintf(fout,",");
      }
      G__more(fout,")\n");
    }
    deffuncmacro=deffuncmacro->next;
  }

  fprintf(fout,"command line: %s\n",G__macros);
  if(G__more_pause(fout,1)) return(1);
  return(0);
}

#if defined(_MSC_VER) && (_MSC_VER>1200)
#pragma optimize("g",off)
#endif

/****************************************************************
* G__display_files()
*
****************************************************************/
int G__display_files(fout)
FILE *fout;
{
  char msg[G__ONELINE];
  int i;
  for(i=0;i<G__nfile;i++) {
#ifndef G__OLDIMPLEMENTATION1273
    if(G__srcfile[i].hasonlyfunc)
      sprintf(msg,"%3d fp=0x%8lx lines=%-4d*file=\"%s\" "
	      ,i,(long)G__srcfile[i].fp,G__srcfile[i].maxline 
	      ,G__srcfile[i].filename);
    else
      sprintf(msg,"%3d fp=0x%8lx lines=%-4d file=\"%s\" "
	      ,i,(long)G__srcfile[i].fp,G__srcfile[i].maxline 
	      ,G__srcfile[i].filename);
#else
    sprintf(msg,"%3d fp=0x%8lx lines=%-4d file=\"%s\" "
	    ,i,(long)G__srcfile[i].fp,G__srcfile[i].maxline 
	    ,G__srcfile[i].filename);
#endif
    if(G__more(fout,msg)) return(1);
    if(G__srcfile[i].prepname) {
      sprintf(msg,"cppfile=\"%s\"",G__srcfile[i].prepname);
      if(G__more(fout,msg)) return(1);
    }
    if(G__more(fout,"\n")) return(1);
  }
  sprintf(msg,"G__MAXFILE = %d\n",G__MAXFILE);
  if(G__more(fout,"\n")) return(1);
  return(0);
}

/********************************************************************
* G__pr
*
*  print source file
*
********************************************************************/
int G__pr(fout,view)
FILE *fout;
struct G__input_file view;
{
  int center,thisline,filenum;
  char G__oneline[G__LONGLINE*2];
  int top,bottom,screen,line=0;
  fpos_t store_fpos;
  /* char original[G__MAXFILENAME]; */
  FILE *G__fp;
  int tempopen;
  char *lines;
  
  if(G__srcfile[view.filenum].prepname||(FILE*)NULL==view.fp) {
    /*************************************************************
     * using C preprocessor , re-open original .c file
     *************************************************************/
    if((char*)NULL==G__srcfile[view.filenum].filename) {
      G__genericerror("Error: File maybe unloaded");
      return(0);
    }
    G__fp = fopen(G__srcfile[view.filenum].filename,"r");
    tempopen=1;
  }
  else {
    /*************************************************************
     * store current file position and rewind file to the beginning
     *************************************************************/
    G__fp=view.fp;
    fgetpos(G__fp,&store_fpos);
    fseek(G__fp,0,SEEK_SET);
    tempopen=0;
  }
  
  /*************************************************************
   * If no file, print error message and return
   *************************************************************/
  if(G__fp==NULL) {
    fprintf(stdout,"Filename not specified. Can not display source!\n");
    return(0);
  }

  /*************************************************************
   * set center and thisline
   *************************************************************/
  filenum = view.filenum;
  center = view.line_number;
  thisline=center;
  
  /*************************************************************
   * Get screensize
   *************************************************************/
#if defined(__hpux) || defined(__GNUC__)
  lines = getenv("LINES");
  if(lines) screen = atoi(lines);
  else      screen = 24;
#else
  screen=24;
#endif
  if(screen<=0) screen=24;

#ifndef G__OLDIMPLEMENTATION2133
  if(G__istrace&0x80) screen = 2;
#endif

  if(0==view.line_number) {
    top=0;
    bottom=1000000;
  }
  else {
    top=center-screen/2;
    if( top < 0) top=0;
    bottom=top+screen;
  }
    
  /********************************************************
   * Read lines until end of file
   ********************************************************/
  while(G__readsimpleline(G__fp,G__oneline)!=0) {
    /************************************************
     *  If input line is "abcdefg hijklmn opqrstu"
     *
     *           arg[0]
     *             |
     *     +-------+-------+
     *     |       |       |
     *  abcdefg hijklmn opqrstu
     *     |       |       |
     *   arg[1]  arg[2]  arg[3]    argn=3
     *
     ************************************************/
    line++;
    if(line>=bottom) break;
    if(top<line) {
      fprintf(fout,"%d",line);
      if(G__srcfile[filenum].breakpoint && G__srcfile[filenum].maxline>line) {
	if(G__BREAK&G__srcfile[filenum].breakpoint[line])  
	  fprintf(fout,"*");
	else if(G__TRACED&G__srcfile[filenum].breakpoint[line]) 
	  fprintf(fout,"-");
	else
	  fprintf(fout," ");
      }
      else
	fprintf(fout," ");
	
      if(line==thisline) fprintf(fout,">");
      else               fprintf(fout," ");
      fprintf(fout,"\t%s\n",G__oneline);
    }
  }
  
  /*************************************************************
   * After reading file
   *************************************************************/
  if(tempopen) {
    /************************************************
     * close .c file
     ************************************************/
    fclose(G__fp);
  }
  else {
    /************************************************
     * restore file position
     ************************************************/
    fsetpos(G__fp,&store_fpos);
  }
  
  return(1);
}
/********************************************************************
* end of G__pr
********************************************************************/


/***********************************************************************
* G__dump_tracecoverage()
*
***********************************************************************/
int G__dump_tracecoverage(fout)
FILE *fout;
{
  int iarg;
  struct G__input_file view;
  for(iarg=0;iarg<G__nfile;iarg++) {
    if(G__srcfile[iarg].fp) {
      view.line_number=0;
      view.filenum=iarg;
      view.fp=G__srcfile[iarg].fp;
      strcpy(view.name,G__srcfile[iarg].filename);
      fprintf(fout
	      ,"%s trace coverage==========================================\n"
	      ,view.name);
      G__pr(fout,view);
    }
  }
  return(0);  
}

#ifndef G__OLDIMPLEMENTATION444
/******************************************************************
* void G__objectmonitor()
*
******************************************************************/
int G__objectmonitor(fout,pobject,tagnum,addspace)
FILE *fout;
long pobject;
int tagnum;
char *addspace;
{
  struct G__inheritance *baseclass;
  char space[G__ONELINE];
  char msg[G__LONGLINE];
  int i;

  sprintf(space,"%s  ",addspace);

  baseclass = G__struct.baseclass[tagnum];
  for(i=0;i<baseclass->basen;i++) {
    if(baseclass->property[i]&G__ISDIRECTINHERIT) {
      if(baseclass->property[i]&G__ISVIRTUALBASE) {
	if(0>G__getvirtualbaseoffset(pobject,tagnum,baseclass,i)) {
	  sprintf(msg,"%s-0x%-7lx virtual ",space
		  ,-1*G__getvirtualbaseoffset(pobject,tagnum,baseclass,i));
	}
	else {
	  sprintf(msg,"%s0x%-8lx virtual ",space
		  ,G__getvirtualbaseoffset(pobject,tagnum,baseclass,i));
	}
	if(G__more(fout,msg)) return(1);
	msg[0] = 0;
	switch(baseclass->baseaccess[i]) {
	case G__PRIVATE:   sprintf(msg,"private: "); break;
	case G__PROTECTED: sprintf(msg,"protected: "); break;
	case G__PUBLIC:    sprintf(msg,"public: "); break;
	}
	if(G__more(fout,msg)) return(1);
	sprintf(msg,"%s\n",G__fulltagname(baseclass->basetagnum[i],1));
	if(G__more(fout,msg)) return(1);
#ifdef G__NEVER_BUT_KEEP
	if(G__objectmonitor(fout
			 ,pobject+(*(long*)(pobject+baseclass->baseoffset[i]))
			 ,baseclass->basetagnum[i],space))
	  return(1);
#endif
      }
      else {
	sprintf(msg,"%s0x%-8lx ",space ,baseclass->baseoffset[i]);
	if(G__more(fout,msg)) return(1);
	msg[0] = 0;
	switch(baseclass->baseaccess[i]) {
	case G__PRIVATE:   sprintf(msg,"private: "); break;
	case G__PROTECTED: sprintf(msg,"protected: "); break;
	case G__PUBLIC:    sprintf(msg,"public: "); break;
	}
	if(G__more(fout,msg)) return(1);
	sprintf(msg,"%s\n",G__fulltagname(baseclass->basetagnum[i],1));
	if(G__more(fout,msg)) return(1);
	if(G__objectmonitor(fout
			    ,pobject+baseclass->baseoffset[i]
			    ,baseclass->basetagnum[i],space))
	  return(1);
      }
    }
  }
  G__incsetup_memvar(tagnum);
  if(G__varmonitor(fout,G__struct.memvar[tagnum],"",space,pobject)) return(1);
  return(0);
}
#endif

/******************************************************************
* void G__varmonitor()
*
******************************************************************/
int G__varmonitor(fout,var,index,addspace,offset)
FILE *fout;
struct G__var_array *var;
char *index;
char *addspace;
long offset;
{
  int imon1;
  long addr;
  char space[50];
  char temp[G__ONELINE];
  char msg[G__ONELINE];
  int startindex,stopindex;
  int precompiled_private;

  
  if((struct G__var_array *)NULL == var) {
    fprintf(fout,"No variable table\n");
    return(0);
  }
  
  if(index[0]=='\0') {
    startindex=0;
    stopindex=var->allvar;
  }
  else {
    if(isdigit(index[0])) {
      G__fprinterr(G__serr,"variable name must be specified\n");
      return(0);
    }
    else {
    search_again:
      startindex=0;
      while((strcmp(index,var->varnamebuf[startindex])!=0)) {
	++startindex;
	if(startindex>=var->allvar) break;
      }
      if(startindex==var->allvar&&var->next) {
	var=var->next;
	goto search_again;
      }
      if(startindex>=var->allvar) {
	fprintf(fout,"Variable %s not found\n" ,index);
	return(0);
      }
    }
    stopindex=startindex+1;
  }

  sprintf(space,"%s  ",addspace);
  
  G__browsing=1;

  for(imon1=startindex;imon1<stopindex;imon1++) {

    if(!G__browsing) return(0);

#ifndef G__OLDIMPLEMENTATION546
    if(0==var->hash[imon1]) continue;
#endif

    if(G__LOCALSTATIC==var->statictype[imon1] && offset) addr=var->p[imon1];
    else addr=offset+var->p[imon1];

#ifdef G__VARIABLEFPOS
    if(var->filenum[imon1]>=0) 
      sprintf(msg,"%-15s%4d "
#ifndef G__OLDIMPLEMENTATION1196
	      , G__stripfilename(G__srcfile[var->filenum[imon1]].filename)
#else
	      , G__srcfile[var->filenum[imon1]].filename
#endif
	      ,var->linenum[imon1]);
    else
      sprintf(msg,"%-15s     " ,"(compiled)");
    if(G__more(fout,msg)) return(1);
#endif
    sprintf(msg,"%s",addspace);
    if(G__more(fout,msg)) return(1);
    sprintf(msg,"0x%-8lx ",addr);
    if(G__more(fout,msg)) return(1);

#ifndef G__NEWINHERIT
    if(var->isinherit[imon1]) {
      sprintf(msg,"inherited ");
      if(G__more(fout,msg)) return(1);
    }
#endif

    precompiled_private=0;
    
    switch(var->access[imon1]) {
    case G__PUBLIC:
      /* fprintf(fout,"public: "); */
      break;
    case G__PROTECTED:
      sprintf(msg,"protected: ");
      if(G__more(fout,msg)) return(1);
      if(-1!=var->tagnum && G__CPPLINK==G__struct.iscpplink[var->tagnum]) {
	precompiled_private=1;
      }
      break;
    case G__PRIVATE:
      sprintf(msg,"private: ");
      if(G__more(fout,msg)) return(1);
      if(-1!=var->tagnum && G__CPPLINK==G__struct.iscpplink[var->tagnum]) {
	precompiled_private=1;
      }
      break;
    }
    switch(var->statictype[imon1]) {
    case G__COMPILEDGLOBAL : /* compiled global variable */
    case G__AUTO : /* auto */
      break;
    case G__LOCALSTATIC : /* static for function */
      sprintf(msg,"static ");
      if(G__more(fout,msg)) return(1);
      break;
    case G__LOCALSTATICBODY : /* body for function static */
      sprintf(msg,"body of static ");
      if(G__more(fout,msg)) return(1);
      break;
    default : /* static for file 0,1,2,... */
      if(var->statictype[imon1]>=0) { /* bug fix */
	sprintf(msg,"file=%s static "
		,G__srcfile[var->statictype[imon1]].filename);
	if(G__more(fout,msg)) return(1);
      }
      else {
        sprintf(msg,"static ");
	if(G__more(fout,msg)) return(1);
      }
      break;
    }
    
#ifndef G__OLDIMPLEMENTATION401
    sprintf(msg,"%s"
	    ,G__type2string((int)var->type[imon1],var->p_tagtable[imon1]
			    ,var->p_typetable[imon1],var->reftype[imon1]
			    ,var->constvar[imon1]));
#else
    sprintf(msg,"%s"
	    ,G__type2string((int)var->type[imon1],var->p_tagtable[imon1]
			    ,var->p_typetable[imon1],var->reftype[imon1]));
#endif
    if(G__more(fout,msg)) return(1);
    sprintf(msg," ");
    if(G__more(fout,msg)) return(1);
    sprintf(msg,"%s",var->varnamebuf[imon1]);
    if(G__more(fout,msg)) return(1);
    if(var->varlabel[imon1][1] 
#ifndef G__OLDIMPLEMENTATION2011
       || var->paran[imon1]
#endif
       ) {
      int ixxx;
      for(ixxx=0;ixxx<var->paran[imon1];ixxx++) {
        if(ixxx) {
          sprintf(msg,"[%d]",var->varlabel[imon1][ixxx+1]);
	  if(G__more(fout,msg)) return(1);
        }
#ifndef G__OLDIMPLEMENTATION2217
	else if(var->varlabel[imon1][1]==INT_MAX) {
          strcpy(msg,"[]");
	  if(G__more(fout,msg)) return(1);
	}
#endif
        else {
          sprintf(msg,"[%d]"
                  ,(var->varlabel[imon1][1]+1)/var->varlabel[imon1][0]);
	  if(G__more(fout,msg)) return(1);
        }
      }
    }

#ifndef G__OLDIMPLEMENTATION401
    if(var->bitfield[imon1]) {
      sprintf(msg," : %d (%d)",var->bitfield[imon1]
	      ,var->varlabel[imon1][G__MAXVARDIM-1]);
      if(G__more(fout,msg)) return(1);
    }
#endif

    if(-1!=offset && 0==precompiled_private && addr) {
      if(0==var->varlabel[imon1][1]
#ifndef G__OLDIMPLEMENTATION2011
	 && 0==var->paran[imon1]
#endif
	 ) {
	switch(var->type[imon1]) {
#ifndef G__OLDIMPLEMENTATION904
	case 'T': 
	  sprintf(msg,"=\"%s\"",*(char**)addr); 
	  if(G__more(fout,msg)) return(1);
	  break;
#endif
#ifndef G__OLDIMPLEMENTATION2191
	case 'j': break;
#else
	case 'm': break;
#endif
	case 'p':
	case 'o': 
	  sprintf(msg,"=%d",*(int*)addr); 
	  if(G__more(fout,msg)) return(1);
	  break;
	case 'P':
	case 'O': 
	  sprintf(msg,"=%g",*(double*)addr); 
	  if(G__more(fout,msg)) return(1);
	  break;
	case 'u':
	  sprintf(msg," , size=%d",G__struct.size[var->p_tagtable[imon1]]);
	  if(G__more(fout,msg)) return(1);
	  temp[0]='\0';
	  G__getcomment(temp,&var->comment[imon1],var->tagnum);
	  if(temp[0]) {
	    sprintf(msg," //%s",temp);
	    if(G__more(fout,msg)) return(1);
	  }
	  if(G__more(fout,"\n")) return(1);
	  G__incsetup_memvar(var->p_tagtable[imon1]);
	  if(G__varmonitor(fout,G__struct.memvar[var->p_tagtable[imon1]]
			   ,"",space,addr)) return(1);
	  break;
	case 'b': 
	  sprintf(msg,"=%d",*(unsigned char*)addr); 
	  if(G__more(fout,msg)) return(1);
	  break;
	case 'c': 
	  sprintf(msg,"=%d ('%c')",*(char*)addr,*(char*)addr); 
	  if(G__more(fout,msg)) return(1);
	  break;
	case 's': 
	  sprintf(msg,"=%d",*(short*)addr); 
	  if(G__more(fout,msg)) return(1);
	  break;
	case 'r': 
	  sprintf(msg,"=%d",*(unsigned short*)addr); 
	  if(G__more(fout,msg)) return(1);
	  break;
	case 'i': 
	  sprintf(msg,"=%d",*(int*)addr); 
	  if(G__more(fout,msg)) return(1);
	  break;
	case 'h': 
	  sprintf(msg,"=%d",*(unsigned int*)addr); 
	  if(G__more(fout,msg)) return(1);
	  break;
#ifndef G__FONS31
	case 'l': 
	  sprintf(msg,"=%ld",*(long*)addr); 
	  if(G__more(fout,msg)) return(1);
	  break;
	case 'k': 
	  sprintf(msg,"=0x%lx",*(unsigned long*)addr); 
	  if(G__more(fout,msg)) return(1);
	  break;
#else
	case 'l': 
	  sprintf(msg,"=%d",*(long*)addr); 
	  if(G__more(fout,msg)) return(1);
	  break;
	case 'k':
	  sprintf(msg,"=0x%x",*(unsigned long*)addr); 
	  if(G__more(fout,msg)) return(1);
	  break;
#endif
	case 'f': 
	  sprintf(msg,"=%g",*(float*)addr); 
	  if(G__more(fout,msg)) return(1);
	  break;
	case 'd': 
	  sprintf(msg,"=%g",*(double*)addr); 
	  if(G__more(fout,msg)) return(1);
	  break;
#ifndef G__OLDIMPLEMENTATION1604
	case 'g': 
#ifdef G__BOOL4BYTE
	  sprintf(msg,"=%d",(*(int*)addr)?1:0); 
#else
	  sprintf(msg,"=%d",(*(unsigned char*)addr)?1:0); 
#endif
	  if(G__more(fout,msg)) return(1);
	  break;
#endif
#ifndef G__OLDIMPLEMENTATION2189
	case 'n': /* long long */
	  sprintf(msg,"=%lld",(*(G__int64*)addr)); 
	  if(G__more(fout,msg)) return(1);
	  break;
	case 'm': /* unsigned long long */
	  sprintf(msg,"=%llu",(*(G__uint64*)addr)); 
	  if(G__more(fout,msg)) return(1);
	  break;
	case 'q': /* long double */
	  sprintf(msg,"=%Lg",(*(long double*)addr)); 
	  if(G__more(fout,msg)) return(1);
	  break;
#endif
#ifndef G__FONS31
	default: 
	  sprintf(msg,"=0x%lx",*(long*)addr); 
	  if(G__more(fout,msg)) return(1);
	  break;
#else
	default: 
	  sprintf(msg,"=0x%x",*(long*)addr); 
	  if(G__more(fout,msg)) return(1);
	  break;
#endif
	}
	if('u'!=var->type[imon1]) if(G__more(fout,"\n")) return(1);
      }
      else {
	switch(var->type[imon1]) {
	case 'c':
#ifndef G__FONS31
	  if(isprint(*(char*)addr))
	    sprintf(msg,"=0x%lx=\"%s\"",addr,(char*)addr); 
	  else
	    sprintf(msg,"=0x%lx",addr); 
#else
	  if(isprint(*(char*)addr))
	    sprintf(msg,"=0x%x=\"%s\"",addr,(char*)addr); 
	  else
	    sprintf(msg,"=0x%x",addr); 
#endif
	  if(G__more(fout,msg)) return(1);
	  break;
#ifndef G__FONS31
	default: 
	  sprintf(msg,"=0x%lx",addr); 
	  if(G__more(fout,msg)) return(1);
	  break;
#else
	default: 
	  sprintf(msg,"=0x%x",addr); 
	  if(G__more(fout,msg)) return(1);
	  break;
#endif
	}
	temp[0]='\0';
	G__getcomment(temp,&var->comment[imon1],var->tagnum);
	if(temp[0]) {
	  sprintf(msg," //%s",temp);
	  if(G__more(fout,msg)) return(1);
	}
	if(G__more(fout,"\n")) return(1);
      }
    }
    else {
      if('u'==var->type[imon1]) {
	sprintf(msg," , size=%d",G__struct.size[var->p_tagtable[imon1]]);
	if(G__more(fout,msg)) return(1);
	temp[0]='\0';
	G__getcomment(temp,&var->comment[imon1],var->tagnum);
	if(temp[0]) {
	  sprintf(msg," //%s",temp);
	  if(G__more(fout,msg)) return(1);
	}
	if(G__more(fout,"\n")) return(1);
	G__incsetup_memvar(var->p_tagtable[imon1]);
	if(G__varmonitor(fout,G__struct.memvar[var->p_tagtable[imon1]]
			 ,"",space,offset)) return(1);
      }
      else {
	temp[0]='\0';
	G__getcomment(temp,&var->comment[imon1],var->tagnum);
	if(temp[0]) {
	  sprintf(msg," //%s",temp);
	  if(G__more(fout,msg)) return(1);
	}
	if(G__more(fout,"\n")) return(1);
      }
    }
  }
  
  if((var->next)&&(index[0]=='\0')) {
    if(G__varmonitor(fout,var->next,index,addspace,offset)) return(1);
  }
  return(0);
}


#ifdef G__WIN32
/**************************************************************************
* status flags
**************************************************************************/
#ifdef G__SPECIALSTDIO
static int G__autoconsole=1;
static int G__isconsole=0;
#else
static int G__autoconsole=0;
static int G__isconsole=1;
#endif
static int G__lockstdio=0;
#endif


#ifndef G__OLDIMPLEMENTATION1485
#include <stdarg.h>

typedef void (*G__ErrMsgCallback_t) G__P((char* msg));
static G__ErrMsgCallback_t G__ErrMsgCallback;

/**************************************************************************
* G__set_errmsgcallback()
**************************************************************************/
void G__set_errmsgcallback(p)
void *p;
{
  G__ErrMsgCallback = (G__ErrMsgCallback_t)p;
}

#ifndef G__OLDIMPLEMENTATION2000
/**************************************************************************
* G__mask_errmsg()
**************************************************************************/
void G__mask_errmsg(msg)
char *msg;
{
}

/**************************************************************************
* G__get_errmsgcallback()
**************************************************************************/
void* G__get_errmsgcallback()
{
  return((void*)G__ErrMsgCallback);
}
#endif

#ifndef G__TESTMAIN
#undef G__fprinterr
/**************************************************************************
* G__fprinterr()
*
* CAUTION:
*  In case you have problem compiling following function, define G__FIX1
* in G__ci.h
**************************************************************************/
#if defined(G__ANSI) || defined(G__WIN32) || defined(G__FIX1) || defined(__sun)
int G__fprinterr(FILE* fp,char* fmt,...)
#elif defined(__GNUC__)
int G__fprinterr(fp,fmt)
FILE* fp;
char* fmt;
...
#else
int G__fprinterr(fp,fmt,arg)
FILE* fp;
char* fmt;
va_list arg;
#endif
{
  int result;
  va_list argptr;
  va_start(argptr,fmt);
  if(G__ErrMsgCallback && G__serr==G__stderr) {
    char buf[G__LONGLINE];
    result = vsprintf(buf,fmt,argptr);
    (*G__ErrMsgCallback)(buf);
  }
  else {
#ifdef G__WIN32
    if(stdout==fp||stderr==fp) {
      if(G__autoconsole&&0==G__isconsole) G__AllocConsole();
    }
#endif
    if(fp) result = vfprintf(fp,fmt,argptr);
    else if(G__serr) result = vfprintf(G__serr,fmt,argptr);
    else result = vfprintf(stderr,fmt,argptr);
  }
  va_end(argptr);
  return(result);
}
#endif

/**************************************************************************
* G__fputerr()
**************************************************************************/
int G__fputerr(c)
int c;
{
  int result;
  if(G__ErrMsgCallback && G__serr==G__stderr) {
    char buf[2]={0,0};
    buf[0] = c;
    (*G__ErrMsgCallback)(buf);
    result = c;
  }
  else {
#ifdef G__WIN32
    if(stdout==G__serr||stderr==G__serr) {
      if(G__autoconsole&&0==G__isconsole) G__AllocConsole();
    }
#endif
    result = fputc(c,G__serr);
  }
  return(result);
}
#endif


#ifndef G__OLDIMPLEMENTATION562
#ifdef G__WIN32
/**************************************************************************
***************************************************************************
* Create new console window and re-open stdio ports
***************************************************************************
**************************************************************************/
#include <windows.h>

/**************************************************************************
* Undefine special macros
**************************************************************************/
#undef printf  
#undef fprintf 
#undef fputc   
#undef putc    
#undef putchar 
#undef fputs   
#undef puts    
#undef fgets   
#undef gets    
#ifndef G__OLDIMPLEMENTATION614
#undef signal
#endif


#ifndef G__OLDIMPLEMENTATION614
static int G__masksignal=0;

/**************************************************************************
* G__signal()
**************************************************************************/
#ifndef G__SYMANTEC
G__signaltype G__signal(sgnl,f)
#else
void* G__signal(sgnl,f)
#endif
int sgnl;
void (*f) G__P((int));
{
#ifndef G__SYMANTEC
  if(!G__masksignal) return((G__signaltype)signal(sgnl,f));
  else               return((G__signaltype)1);
#else
  if(!G__masksignal) return((void*)signal(sgnl,f));
  else               return((void*)1);
#endif
}

/**************************************************************************
* G__setmasksignal()
**************************************************************************/
int G__setmasksignal(masksignal)
int masksignal;
{
  G__masksignal=masksignal;
  return(0);
}
#endif /* ON614 */

/**************************************************************************
* G__setautoconsole()
**************************************************************************/
void G__setautoconsole(autoconsole)
int autoconsole;
{
  G__autoconsole=autoconsole;
  G__isconsole=0;
}

/**************************************************************************
* G__AllocConsole()
**************************************************************************/
int G__AllocConsole()
{
  BOOL result=TRUE;
  if(0==G__isconsole) {
    result=FreeConsole();
    result = AllocConsole();
    SetConsoleTitle("CINT : C++ interpreter");
    G__isconsole=1;
    if(TRUE==result) {
      G__stdout=G__sout=freopen("CONOUT$","w",stdout);
      G__stderr=G__serr=freopen("CONOUT$","w",stderr);
      G__stdin=G__sin=freopen("CONIN$","r",stdin);
#ifndef G__OLDIMPLEMENTATION713
      G__update_stdio();
#endif
    }
  }
  return result;
}

/**************************************************************************
* G__FreeConsole()
**************************************************************************/
int G__FreeConsole()
{
  BOOL result=TRUE;
  if(G__isconsole && !G__lockstdio) {
    G__isconsole=0;
    result=FreeConsole();
  }
  else {
    result=FALSE;
  }
  return result;
}

/**************************************************************************
* G__printf()
**************************************************************************/
int G__printf(char *fmt,...)
{
  int result;
  va_list argptr;
  va_start(argptr,fmt);
  G__lockstdio=1;
  if(G__autoconsole&&0==G__isconsole) G__AllocConsole();
  result = vprintf(fmt,argptr);
  G__lockstdio=0;
  va_end(argptr);
  return(result);
}


/**************************************************************************
* G__fprintf()
**************************************************************************/
int G__fprintf(FILE *fp,char *fmt,...)
{
  int result;
  va_list argptr;
  va_start(argptr,fmt);
  G__lockstdio=1;
  if(stdout==fp||stderr==fp) {
    if(G__autoconsole&&0==G__isconsole) G__AllocConsole();
  }
  result = vfprintf(fp,fmt,argptr);
  G__lockstdio=0;
  va_end(argptr);
  return(result);
}

/**************************************************************************
* G__fputc()
**************************************************************************/
int G__fputc(int character,FILE *fp)
{
  int result;
  G__lockstdio=1;
  if(stdout==fp||stderr==fp) {
    if(G__autoconsole&&0==G__isconsole) G__AllocConsole();
  }
  result=fputc(character,fp);
  G__lockstdio=0;
  return(result);
}

/**************************************************************************
* G__putchar()
**************************************************************************/
int G__putchar(int character)
{
   int result;
   G__lockstdio=1;
   if(G__autoconsole&&0==G__isconsole) G__AllocConsole();
   result=putchar(character);
   G__lockstdio=0;
   return(result);
}

/**************************************************************************
* G__fputs()
**************************************************************************/
int G__fputs(char *string,FILE *fp)
{
  int result;
  G__lockstdio=1;
  if(stdout==fp||stderr==fp) {
    if(G__autoconsole&&0==G__isconsole) G__AllocConsole();
  }
  result=fputs(string,fp);
  G__lockstdio=0;
  return(result);
}

/**************************************************************************
* G__puts()
**************************************************************************/
int G__puts(char *string)
{
  int result;
  G__lockstdio=1;
  if(G__autoconsole&&0==G__isconsole) G__AllocConsole();
  result=puts(string);
  G__lockstdio=0;
  return(result);
}

/**************************************************************************
* G__fgets()
**************************************************************************/
char *G__fgets(char *string,int n,FILE *fp)
{
  char *result;
  G__lockstdio=1;
  if(fp==stdin) {
    if(G__autoconsole&&0==G__isconsole) G__AllocConsole();
  }
  result=fgets(string,n,fp);
  G__lockstdio=0;
  return(result);
}
/**************************************************************************
* G__gets()
**************************************************************************/
char *G__gets(char *buffer)
{
  char *result;
  G__lockstdio=1;
  if(G__autoconsole&&0==G__isconsole) G__AllocConsole();
  result=gets(buffer);
  G__lockstdio=0;
  return(result);
}

/**************************************************************************
* G__system()
**************************************************************************/
int G__system(char *com)
{
#if 1

#undef system
  /* Simply call system() system call */
  return(system(com));

#else
  /* This code does not work because of following reasons
   *  1. CreateProcess() WIN32 API does not work with GNU library
   *  2. So far, I can not redirect child process I/O 
   *  3. Absolute path name is needed to start child process
   */
  int result;
  BOOL fSuccess;
  int i=0,j;
  HANDLE hProcess;
  DWORD dwExitCode;
  BOOL fExist = FALSE;

  char comName[G__ONELINE];
  char comLine[G__ONELINE];
  STARTUPINFO StartupInformation;
  PROCESS_INFORMATION ProcessInformation;

  while(isspace(com[i])) ++i;
  j=0;
  while(com[i] && !isspace(com[i])) comName[j++] = com[i++]; 
  comName[j] = 0;
  if(strstr(comName,".exe")==0 && strstr(comName,".EXE")==0) { 
    strcat(comName,".exe");
  }

  j=0;
  while(com[i]) comLine[j++] = com[i++]; 
  comLine[j] = 0;
  
  StartupInformation.cb = sizeof(STARTUPINFO);
  StartupInformation.lpReserved = NULL;
  StartupInformation.lpDesktop = NULL;
  StartupInformation.lpTitle = NULL;
  StartupInformation.dwX = 0;
  StartupInformation.dwY = 0;
  StartupInformation.dwXSize = 100; 
  StartupInformation.dwYSize = 100;
  StartupInformation.dwXCountChars = 80;
  StartupInformation.dwYCountChars = 24;
  StartupInformation.dwFillAttribute = 0;
  StartupInformation.dwFlags = STARTF_USESTDHANDLES;
  StartupInformation.wShowWindow = 0;
  StartupInformation.cbReserved2 = 0;
  StartupInformation.lpReserved2 = NULL;
  StartupInformation.hStdInput  = G__sin;
  StartupInformation.hStdOutput = G__sout;
  StartupInformation.hStdError  = G__serr;

  fSuccess = CreateProcess(comName
                           ,comLine
                           ,NULL     /* lpProcessAttributes */
                           ,NULL     /* lpThreadAttributes */
                           ,TRUE     /* bInheritHandles */
                           ,CREATE_DEFAULT_ERROR_MODE /* dwCreationFlags */
                           ,NULL     /* lpEnvironment */
                           ,NULL     /* lpCurrentDirectory */
                           ,&StartupInformation
                           ,&ProcessInformation);

  if(!fSuccess) return(-1);

  hProcess = ProcessInformation.hProcess;

  CloseHandle(ProcessInformation.hThread);

  if(WaitForSingleObject(hProcess,INFINITE) != WAIT_FAILED) {
    /* the process terminated */
    fExist = GetExitCodeProcess(hProcess,&dwExitCode);
  }
  else {
    fExist = FALSE;
  }

  CloseHandle(hProcess);

  return(fExist?-1:0);

#endif
}

#else /* G__WIN32 */

/**************************************************************************
* G__setautoconsole()
**************************************************************************/
void G__setautoconsole(autoconsole)
int autoconsole;
{
  autoconsole=autoconsole; /* no effect */
}

/**************************************************************************
* G__AllocConsole()
**************************************************************************/
int G__AllocConsole()
{
  return(0);
}

/**************************************************************************
* G__FreeConsole()
**************************************************************************/
int G__FreeConsole()
{
  return(0);
}

#endif /* G__WIN32 */
#endif /* ON562 */


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
