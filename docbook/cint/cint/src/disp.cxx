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
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/


#include "common.h"
#include "Api.h"

extern "C" {

int G__browsing=1; /* used in disp.c and intrpt.c */

 
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
   register G__int64 result;
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
   if (neg) {
      // -(-2147483648) is not a valid long long, but -(-2147483648 + 42) is!
      cutoff = -(LONG_LONG_MIN + 42);
      cutoff += 42; // fixup offset for unary -
   } else {
      cutoff = LONG_LONG_MAX;
   }
   cutlim = (int)( cutoff % (G__uint64) base );
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
      result = neg ? LONG_LONG_MIN : LONG_LONG_MAX;
      errno = ERANGE;
   } else {
      result = acc;
      if (neg) {
        result = acc;
      }
   }
   if (endptr != 0)
      *endptr = (char *) (any ? s - 1 : nptr);
   return (result);
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
   cutlim = (int) 
       ((G__uint64) ULONG_LONG_MAX % (G__uint64) base);
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
   } else if (neg) {
      // IGNORE - we're unsigned!
      // acc = -acc;
   }
   if (endptr != 0)
      *endptr = (char *) (any ? s - 1 : nptr);
   return (acc);
}
#endif /* __CINT__ */

/***********************************************************************
* G__redirected_on()
* G__redirected_off()
***********************************************************************/
static int G__redirected = 0;
void G__redirect_on() { G__redirected = 1; }
void G__redirect_off() { G__redirected = 0; }

static int G__more_len;
/***********************************************************************
* G__more_col()
***********************************************************************/
void G__more_col(int len)
{
  G__more_len += len;
}

/***********************************************************************
* G__more_pause()
***********************************************************************/
int G__more_pause(FILE *fp,int len)
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
    G__more_len=0;
    return(0);
  }

  if(fp==G__stdout && 0<dispsize && 0==G__redirected ) {
    /* ++shownline; */
    shownline += (G__more_len/dispcol + 1);
    /*DEBUG printf("(%d,%d,%d)",G__more_len,dispcol,shownline); */
    /*************************************************
     * judgement for pause
     *************************************************/
    if(shownline>=dispsize || onemore) {
      shownline=0;
      G__FastAllocString buf(G__input("-- Press return for more -- (input [number] of lines, Cont,Step,More) "));
      if(isdigit(buf[0])) { /* change display size */
         dispsize = (int)G__int(G__calc_internal(buf));
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
int G__more(FILE *fp,const char *msg)
{
#ifndef G__OLDIMPLEMENTATION1485
  if(fp==G__serr) G__fprinterr(G__serr,"%s",msg);
  else fprintf(fp,"%s",msg);
#else
  fprintf(fp,"%s",msg);
#endif
  if(strchr(msg,'\n')) {
     return(G__more_pause(fp,(int)strlen(msg)));
  }
  else {
     G__more_col((int)strlen(msg));
    return(0);
  }
}

/***********************************************************************
* void G__disp_purevirtualfunc
***********************************************************************/
void G__display_purevirtualfunc(int /* tagnum */)
{
  /* to be implemented */
}

/***********************************************************************
* void G__disp_friend
***********************************************************************/
int G__display_friend(FILE *fp,G__friendtag*friendtag)
{
  G__FastAllocString msg(" friend ");
  if(G__more(fp,msg)) return(1);
  while(friendtag) {
    msg = G__fulltagname(friendtag->tagnum,1);
    msg += ",";
    if(G__more(fp,msg)) return(1);
    friendtag = friendtag->next;
  }
  return(0);
}

/***********************************************************************
* void G__listfunc
***********************************************************************/
int G__listfunc(FILE *fp,int access,const char *fname,G__ifunc_table *ifunc)
{
   return G__listfunc_pretty(fp,access,fname,ifunc,0);
}

/***********************************************************************
* void G__listfunc_pretty
***********************************************************************/
int G__listfunc_pretty(FILE *fp,int access,const char *fname,G__ifunc_table *iref, char friendlyStyle)
{
  int i,n;
  G__FastAllocString temp(G__ONELINE);
  G__FastAllocString msg(G__LONGLINE);

  G__browsing=1;
  
  G__ifunc_table_internal* ifunc = iref ? G__get_ifunc_internal(iref) : 0;
  if(!ifunc) ifunc = G__p_ifunc;
  
  bool showHeader = !friendlyStyle;
  showHeader |= (ifunc->allifunc>0 && ifunc->pentry[0]->filenum>=0); // if we need to display filenames

  if (showHeader) {
     if (!friendlyStyle || -1==ifunc->tagnum) {
        msg.Format("%-15sline:size busy function type and name  ","filename");
        if(G__more(fp,msg)) return(1);
     }
     if(-1!=ifunc->tagnum) {
        msg.Format("(in %s)\n",G__struct.name[ifunc->tagnum]);
       if(G__more(fp,msg)) return(1);
     }
     else {
       if(G__more(fp,"\n")) return(1);
     }
  }

  const char* parentname = (-1==ifunc->tagnum)? "" : G__struct.name[ifunc->tagnum];

  /***************************************************
   * while interpreted function table list exists
   ***************************************************/
  while(ifunc) {
    for(i=0;i<ifunc->allifunc;i++) {

      if(!G__browsing) return(0);

      if(fname && strcmp(fname,ifunc->funcname[i])!=0) continue;

      if(
          ifunc->hash[i] &&
         (ifunc->access[i]&access)) {
        
        /* print out file name and line number */
        if(ifunc->pentry[i]->filenum>=0) {
           msg.Format("%-15s%4d:%-3d%c%2d "
                  ,G__stripfilename(G__srcfile[ifunc->pentry[i]->filenum].filename)
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
                  ,G__globalcomp?ifunc->globalcomp[i]:ifunc->busy[i]
                  );
          if(G__more(fp,msg)) return(1);
#ifdef G__ASM_DBG
          if(ifunc->pentry[i]->bytecode) {
            G__ASSERT(ifunc->pentry[i]->bytecodestatus==G__BYTECODE_SUCCESS||
                      ifunc->pentry[i]->bytecodestatus==G__BYTECODE_ANALYSIS);
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
             ||ifunc->pentry[i]->bytecodestatus==G__BYTECODE_ANALYSIS
             ) {
            G__ASSERT(ifunc->pentry[i]->bytecode);
          }
          else {
            G__ASSERT(!ifunc->pentry[i]->bytecode);
          }
#endif
        }
        else {
          if (!friendlyStyle) {
             msg.Format("%-15s%4d:%-3d%3d " ,"(compiled)" ,0,0 ,ifunc->busy[i]);
            if(G__more(fp,msg)) return(1);
          }
        }
        
        if(ifunc->hash[i]) {
           msg.Format("%s ",G__access2string(ifunc->access[i]));
           if (G__more(fp, msg)) return(1);
        } else {
           if(G__more(fp,"------- ")) return(1);
        }
        if(ifunc->isexplicit[i]) {
          if(G__more(fp,"explicit ")) return(1);
        }
#ifndef G__NEWINHERIT
        if(ifunc->isinherit[i]) { 
          if(G__more(fp,"inherited ")) return(1);
        }
#endif
        if(ifunc->isvirtual[i]) {
          if(G__more(fp,"virtual ")) return(1);
        }

        if(ifunc->staticalloc[i]) {
          if(G__more(fp,"static ")) return(1);
        }

        
        /* print out type of return value */
        msg.Format("%s ",G__type2string(ifunc->type[i]
                                        ,ifunc->p_tagtable[i]
                                        ,ifunc->p_typetable[i]
                                        ,ifunc->reftype[i]
                                        ,ifunc->isconst[i]));
        if(G__more(fp,msg)) return(1);
        
        /*****************************************************
         * to get type of function parameter
         *****************************************************/
        /**********************************************************
         * print out type and name of function and parameters
         **********************************************************/
        /* print out function name */
        if(strlen(ifunc->funcname[i])>=msg.Capacity()-6) {
          strncpy(msg,ifunc->funcname[i],msg.Capacity()-3);
          msg[msg.Capacity()-6]=0;
          msg += "...(";
        }
        else {
          if (friendlyStyle) {
             msg = parentname;
             msg += "::";
             if(G__more(fp,msg)) return(1);
          }
          msg = ifunc->funcname[i];
          msg += "(";
        }
        if(G__more(fp,msg)) return(1);

        if(ifunc->ansi[i] && 0==ifunc->para_nu[i]) {
          if(G__more(fp,"void")) return(1);
        }
        
        /* print out parameter types */
        for(n=0;n<ifunc->para_nu[i];n++) {
          
          if(n!=0) {
            if(G__more(fp,",")) return(1);
          }
          /* print out type of return value */
#ifndef G__OLDIMPLEMENATTION401
          msg = G__type2string(ifunc->param[i][n]->type
                                         ,ifunc->param[i][n]->p_tagtable
                                         ,ifunc->param[i][n]->p_typetable
                                         ,ifunc->param[i][n]->reftype
                                         ,ifunc->param[i][n]->isconst);
#else
          msg = G__type2string(ifunc->param[i][n]->type
                                         ,ifunc->param[i][n]->p_tagtable
                                         ,ifunc->param[i][n]->p_typetable
                                         ,ifunc->param[i][n]->reftype));
#endif
          if(G__more(fp,msg)) return(1);

          if(ifunc->param[i][n]->name) {
             msg.Format(" %s",ifunc->param[i][n]->name);
            if(G__more(fp,msg)) return(1);
          }
          if(ifunc->param[i][n]->def) {
             msg.Format("=%s",ifunc->param[i][n]->def);
            if(G__more(fp,msg)) return(1);
          }
        }
        if(2==ifunc->ansi[i]) {
          ;
          if(G__more(fp," ...")) return(1);
        }
        if(G__more(fp,")")) return(1);
        if(ifunc->isconst[i]&G__CONSTFUNC) {
          if(G__more(fp," const")) return(1);
        }
        if(ifunc->ispurevirtual[i]) {
          if(G__more(fp,"=0")) return(1);
        }
        if(G__more(fp,";")) return(1);
        temp[0] = '\0';
        G__getcomment(temp,&ifunc->comment[i],ifunc->tagnum);
        if(temp[0]) {
           msg.Format(" //%s",temp());
          if(G__more(fp,msg)) return(1);
        }
        if(ifunc->friendtag[i]) 
          if(G__display_friend(fp,ifunc->friendtag[i])) return(1);
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
int G__showstack(FILE *fout)
{
  int temp,temp1;
  struct G__var_array *local;
  G__FastAllocString syscom(G__MAXNAME);
  G__FastAllocString msg(G__LONGLINE);

  local=G__p_local;
  temp=0;
  while(local) {
#ifdef G__VAARG
     msg.Format("%d ",temp);
    if(G__more(fout,msg)) return(1);
    if(local->exec_memberfunc && -1!=local->tagnum) {
       msg.Format("%s::",G__struct.name[local->tagnum]);
      if(G__more(fout,msg)) return(1);
    }
    msg.Format("%s(",G__get_ifunc_internal(local->ifunc)->funcname[local->ifn]);
    if(G__more(fout,msg)) return(1);
    for(temp1=0;temp1<local->libp->paran;temp1++) {
      if(temp1) {
         msg.Format(",");
        if(G__more(fout,msg)) return(1);
      }
      G__valuemonitor(local->libp->para[temp1],syscom);
      if(G__more(fout,syscom)) return(1);
    }
    if(-1!=local->prev_filenum) {
       msg.Format(") [%s: %d]\n" 
              ,G__stripfilename(G__srcfile[local->prev_filenum].filename)
              ,local->prev_line_number);
      if(G__more(fout,msg)) return(1);
    }
    else {
      if(G__more(fout,") [entry]\n")) return(1);
    }
#else
    msg.Format("%d %s() [%s: %d]\n" ,temp ,local->ifunc->funcname[local->ifn]
            ,G__filenameary[local->prev_filenum] ,local->prev_line_number);
    if(G__more(fout,msg)) return(1) ;
#endif
    ++temp;
    local=local->prev_local;
  }
  return(0);
}

/**************************************************************************
* G__getdictpos()
**************************************************************************/
struct G__dictposition* G__get_dictpos(char *fname)
{
  struct G__dictposition *dict = (struct G__dictposition*)NULL;
  int i;
  /* search for source file entry */
  for(i=0;i<G__nfile;i++) {
    if(G__matchfilename(i,fname)) {
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
int G__display_newtypes(FILE *fout,const char *fname)
{
  struct G__dictposition *dict = (struct G__dictposition*)NULL;
  int i;

  /* search for source file entry */
  for(i=0;i<G__nfile;i++) {
    if(G__matchfilename(i,fname)) {
      dict = G__srcfile[i].dictpos;
      break;
    }
  }

  if(dict) {
    /* listup new class/struct/enum/union */
    static char emptystring[1] = {0};
    if(G__display_class(fout,emptystring,0,dict->tagnum)) return(1);
    /* listup new typedef */
    if(G__display_typedef(fout,"",dict->typenum)) return(1);
    return(0);
  }

  G__fprinterr(G__serr,"File %s is not loaded\n",fname);
  return(1);
}

/**************************************************************************
* G__display_string()
*
**************************************************************************/
int G__display_string(FILE *fout)
{
  size_t len;
  unsigned long totalsize=0;
  struct G__ConstStringList *pconststring;
  G__FastAllocString msg(G__ONELINE);

  pconststring = G__plastconststring;
  while(pconststring->prev) {
    len=strlen(pconststring->string);
    totalsize+=len+1;
    if(totalsize>=msg.Capacity()-5) {
       msg.Format("%3d ",len);
       strncpy(msg+4,pconststring->string,msg.Capacity()-5);
       msg[msg.Capacity()-1]=0;
    }
    else {
       msg.Format("%3d %s\n",len,pconststring->string);
    }
    if(G__more(fout,msg)) return(1);
    pconststring=pconststring->prev;
  }
  msg.Format("Total string constant size = %ld\n",totalsize);
  if(G__more(fout,msg)) return(1);
  return(0);
}

/****************************************************************
* G__display_classinheritance()
*
****************************************************************/
static int G__display_classinheritance(FILE *fout,int tagnum,const char *space)
{
  int i;
  struct G__inheritance *baseclass;
  G__FastAllocString addspace(50);
  G__FastAllocString temp(G__ONELINE);
  G__FastAllocString msg(G__LONGLINE);

  baseclass = G__struct.baseclass[tagnum];

  if(NULL==baseclass) return(0);

  addspace.Format("%s  ",space);

  for(i=0;i<baseclass->basen;i++) {
    if(baseclass->herit[i]->property&G__ISDIRECTINHERIT) {
       msg.Format("%s0x%-8lx ",space ,baseclass->herit[i]->baseoffset);
      if(G__more(fout,msg)) return(1);
      if(baseclass->herit[i]->property&G__ISVIRTUALBASE) {
        if(G__more(fout,"virtual ")) return(1);
      }
      if(baseclass->herit[i]->property&G__ISINDIRECTVIRTUALBASE) {
        if(G__more(fout,"(virtual) ")) return(1);
      }
      msg.Format("%s %s"
                 ,G__access2string(baseclass->herit[i]->baseaccess)
                 ,G__fulltagname(baseclass->herit[i]->basetagnum,0));
      if(G__more(fout,msg)) return(1);
      temp[0]='\0';
      G__getcomment(temp,&G__struct.comment[baseclass->herit[i]->basetagnum]
                    ,baseclass->herit[i]->basetagnum);
      if(temp[0]) {
         msg.Format(" //%s",temp());
        if(G__more(fout,msg)) return(1);
      }
      if(G__more(fout,"\n")) return(1);
      if(G__display_classinheritance(fout,baseclass->herit[i]->basetagnum,addspace))
        return(1);
    }
  }
  return(0);
}

/****************************************************************
* G__display_membervariable()
*
****************************************************************/
static int G__display_membervariable(FILE *fout,int tagnum,int base)
{
  struct G__var_array *var;
  struct G__inheritance *baseclass;
  int i;
  baseclass = G__struct.baseclass[tagnum];

  if(base) {
    for(i=0;i<baseclass->basen;i++) {
      if(!G__browsing) return(0);
      if(baseclass->herit[i]->property&G__ISDIRECTINHERIT) {
        if(G__display_membervariable(fout,baseclass->herit[i]->basetagnum,base))
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
static int G__display_memberfunction(FILE *fout,int tagnum,int access,int base)
{
  struct G__ifunc_table_internal *store_ifunc;
  int store_exec_memberfunc;
  struct G__inheritance *baseclass;
  int i;
  int tmp;
  baseclass = G__struct.baseclass[tagnum];

  if(base) {
    for(i=0;i<baseclass->basen;i++) {
      if(!G__browsing) return(0);
      if(baseclass->herit[i]->property&G__ISDIRECTINHERIT) {
        if(G__display_memberfunction(fout,baseclass->herit[i]->basetagnum
                                     ,access,base)) return(1);
      }
    }
  }

  /* member function */
  if(G__struct.memfunc[tagnum]) {
    G__incsetup_memfunc(tagnum);
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
  

/****************************************************************
* G__display_class()
*
****************************************************************/
int G__display_class(FILE *fout, char *name,int base,int start)
{
  int tagnum;
  int i,j;
  struct G__inheritance *baseclass;
  G__FastAllocString temp(G__ONELINE);
  G__FastAllocString msg(G__LONGLINE);
  char *p;
  int store_globalcomp;
  short int store_iscpp;

  G__browsing=1;

  i=0;
  while(isspace(name[i])) i++;

  /*******************************************************************
  * List of classes
  *******************************************************************/
  if('\0'==name[i]) {
    if(base) {
      /* In case of 'Class' command */
      for(i=0;i<G__struct.alltag;i++) {
         temp.Format("%d",i);
        G__display_class(fout,temp,0,0);
      }
      return(0);
    }
    /* no class name specified, list up all tagnames */
    if(G__more(fout,"List of classes\n")) return(1);
    msg.Format("%-15s%5s\n","file","line");
    if(G__more(fout,msg)) return(1);
    for(i=start;i<G__struct.alltag;i++) {
      if(!G__browsing) return(0);
      switch(G__struct.iscpplink[i]) {
      case G__CLINK:
         if (G__struct.filenum[i] == -1) msg.Format("%-20s " ,"(C compiled)");
        else
           msg.Format("%-15s%5d " 
                  ,G__stripfilename(G__srcfile[G__struct.filenum[i]].filename)
                  ,G__struct.line_number[i]);
        if(G__more(fout,msg)) return(1);
        break;
      case G__CPPLINK:
         if (G__struct.filenum[i] == -1) msg.Format("%-20s " ,"(C++ compiled)");
        else
           msg.Format("%-15s%5d " 
                  ,G__stripfilename(G__srcfile[G__struct.filenum[i]].filename)
                  ,G__struct.line_number[i]);
        if(G__more(fout,msg)) return(1);
        break;
      case 1:
         msg.Format("%-20s " ,"(C compiled old 1)");
        if(G__more(fout,msg)) return(1);
        break;
      case 2:
         msg.Format("%-20s " ,"(C compiled old 2)");
        if(G__more(fout,msg)) return(1);
        break;
      case 3:
         msg.Format("%-20s " ,"(C compiled old 3)");
        if(G__more(fout,msg)) return(1);
        break;
      default:
        if (G__struct.filenum[i] == -1)
           msg.Format("%-20s " ," ");
        else
           msg.Format("%-15s%5d " 
                  ,G__stripfilename(G__srcfile[G__struct.filenum[i]].filename)
                  ,G__struct.line_number[i]);
        if(G__more(fout,msg)) return(1);
        break;
      }
      if(G__struct.isbreak[i]) fputc('*',fout);
      else                     fputc(' ',fout);
      if(G__struct.istrace[i]) fputc('-',fout);
      else                     fputc(' ',fout);
      G__more_col(2);
      store_iscpp=G__iscpp; /* This is a dirty trick to display 'class' */
      G__iscpp=0;           /* 'struct','union' or 'namespace' in msg   */
      store_globalcomp=G__globalcomp;
      G__globalcomp=G__NOLINK;
      msg.Format(" %s ",G__type2string('u',i,-1,0,0));
      G__iscpp=store_iscpp; /* dirty trick reset */
      G__globalcomp=store_globalcomp;
      if(G__more(fout,msg)) return(1);
      baseclass = G__struct.baseclass[i];
      if(baseclass) {
        for(j=0;j<baseclass->basen;j++) {
          if(baseclass->herit[j]->property&G__ISDIRECTINHERIT) {
            if(baseclass->herit[j]->property&G__ISVIRTUALBASE) {
              if(G__more(fout,"virtual ")) return(1);
            }
            msg.Format("%s%s " 
                    ,G__access2string(baseclass->herit[j]->baseaccess)
                    ,G__fulltagname(baseclass->herit[j]->basetagnum,0));
            if(G__more(fout,msg)) return(1);
          }
        }
      }
      if('$'==G__struct.name[i][0]) {
         msg.Format(" (typedef %s)",G__struct.name[i]+1);
        if(G__more(fout,msg)) return(1);
      }
      temp[0]='\0';
      G__getcomment(temp,&G__struct.comment[i],i);
      if(temp[0]) {
         msg.Format(" //%s",temp());
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

  if((char*)NULL!=strstr(name+i,">>")) {
    /* dealing with A<A<int>> -> A<A<int> > */
    char *pt1;
    pt1 = strstr(name+i,">>");
    ++pt1;
    G__FastAllocString tmpbuf(pt1);
    *pt1=' ';
    ++pt1;
    strcpy(pt1,tmpbuf);  // Legacy, we hope the caller created a bit of wiggle room
  }

  if(isdigit(*(name+i))) tagnum = atoi(name+i);
  else                   tagnum = G__defined_tagname(name+i,0);

  /* no such class,struct */
  if(-1==tagnum||G__struct.alltag<=tagnum) return(0); 

      G__class_autoloading(&tagnum);

  G__more(fout,"===========================================================================\n");
  msg.Format("%s ",G__tagtype2string(G__struct.type[tagnum]));
  if(G__more(fout,msg)) return(1);
  msg.Format("%s",G__fulltagname(tagnum,0));
  if(G__more(fout,msg)) return(1);
  temp[0]='\0';
  G__getcomment(temp,&G__struct.comment[tagnum],tagnum);
  if(temp[0]) {
     msg.Format(" //%s",temp());
    if(G__more(fout,msg)) return(1);
  }
  if(G__more(fout,"\n")) return(1);
  if (G__struct.filenum[tagnum] == -1)
    msg.Format(" size=0x%x\n" ,G__struct.size[tagnum]);
  else {
    msg.Format(" size=0x%x FILE:%s LINE:%d\n" ,G__struct.size[tagnum]
            ,G__stripfilename(G__srcfile[G__struct.filenum[tagnum]].filename)
            ,G__struct.line_number[tagnum]);
  }
  if(G__more(fout,msg)) return(1);
    msg.Format(" (tagnum=%d,voffset=%d,isabstract=%d,parent=%d,gcomp=%d:%d,funcs(dn21=~xcpd)=%x)",
               tagnum,
               G__struct.virtual_offset[tagnum],
               G__struct.isabstract[tagnum],
               G__struct.parent_tagnum[tagnum],
               G__struct.globalcomp[tagnum],
               G__struct.iscpplink[tagnum],
               G__struct.funcs[tagnum]
               );
  if(G__more(fout,msg)) return(1);
  if('$'==G__struct.name[tagnum][0]) {
    msg.Format(" (typedef %s)",G__struct.name[tagnum]+1);
    if(G__more(fout,msg)) return(1);
  }
  if(G__more(fout,"\n")) return(1);

  baseclass = G__struct.baseclass[tagnum];

  if(G__cintv6) {
    if(G__more(fout,"Virtual table--------------------------------------------------------------\n")) return(1);
    G__bc_disp_vtbl(fout,tagnum);
  }

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
int G__display_typedef(FILE *fout,const char *name,int startin)
{
  int i,j;
  int start,stop;
  G__FastAllocString temp(G__ONELINE);
  G__FastAllocString msg(G__LONGLINE);

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
      msg.Format("%-15s%4d "
              ,G__stripfilename(G__srcfile[G__newtype.filenum[i]].filename)
              ,G__newtype.linenum[i]);
    else
      msg.Format("%-15s     " ,"(compiled)");
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
      msg.Format("typedef void* %s",G__newtype.name[i]); 
      if(G__more(fout,msg)) return(1);
    }
    else if('a'==G__newtype.type[i]) {
      /* pointer to member */
      msg.Format("typedef G__p2memfunc %s",G__newtype.name[i]); 
      if(G__more(fout,msg)) return(1);
    }
    else {
      /* G__typedef may need to be changed to add isconst member */
      msg.Format("typedef %s" ,G__type2string(tolower(G__newtype.type[i])
                                                ,G__newtype.tagnum[i],-1
                                                ,G__newtype.reftype[i]
                                                ,G__newtype.isconst[i])); 
      if(G__more(fout,msg)) return(1);
      if(G__more(fout," ")) return(1);
      if(isupper(G__newtype.type[i])&&G__newtype.nindex[i]) {
        if(0<=G__newtype.parent_tagnum[i]) 
          msg.Format("(*%s::%s)"
                  ,G__fulltagname(G__newtype.parent_tagnum[i],1)
                  ,G__newtype.name[i]);
        else
          msg.Format("(*%s)",G__newtype.name[i]);
        if(G__more(fout,msg)) return(1);
      }
      else {
        if(isupper(G__newtype.type[i])) {
          if(G__newtype.isconst[i]&G__PCONSTVAR) msg.Format("*const ");
          else msg.Format("*");
          if(G__more(fout,msg)) return(1);
        }
        if(0<=G__newtype.parent_tagnum[i]) {
          msg.Format("%s::",G__fulltagname(G__newtype.parent_tagnum[i],1));
          if(G__more(fout,msg)) return(1);
        }
        msg.Format("%s",G__newtype.name[i]);
        if(G__more(fout,msg)) return(1);
      }
      for(j=0;j<G__newtype.nindex[i];j++) {
        msg.Format("[%d]",G__newtype.index[i][j]);
        if(G__more(fout,msg)) return(1);
      }
    }
    temp[0]='\0';
    G__getcommenttypedef(temp,&G__newtype.comment[i],i);
    if(temp[0]) {
       msg.Format(" //%s",temp());
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
int G__display_eachtemplate(FILE *fout,G__Definedtemplateclass *deftmplt,int detail)
{
  struct G__Templatearg *def_para;
  struct G__Definedtemplatememfunc *memfunctmplt;
  fpos_t store_pos;
  /* char buf[G__LONGLINE]; */
  G__FastAllocString msg(G__LONGLINE);
  int c;

  if(!deftmplt->def_fp) return(0);

  msg.Format("%-20s%5d "
          ,G__stripfilename(G__srcfile[deftmplt->filenum].filename)
          ,deftmplt->line);
  if(G__more(fout,msg)) return(1);
  msg.Format("template<");
  if(G__more(fout,msg)) return(1);
  def_para=deftmplt->def_para;
  while(def_para) {
    switch(def_para->type) {
    case G__TMPLT_CLASSARG:
      msg.Format("class ");
      if(G__more(fout,msg)) return(1);
      break;
    case G__TMPLT_TMPLTARG:
      msg.Format("template<class U> class ");
      if(G__more(fout,msg)) return(1);
      break;
    case G__TMPLT_SIZEARG:
      msg.Format("size_t ");
      if(G__more(fout,msg)) return(1);
      break;
    default:
      msg.Format("%s ",G__type2string(def_para->type,-1,-1,0,0));
      if(G__more(fout,msg)) return(1);
      break;
    }
    msg.Format("%s",def_para->string);
    if(G__more(fout,msg)) return(1);
    def_para=def_para->next;
    if(def_para) fprintf(fout,",");
    else         fprintf(fout,">");
    G__more_col(1);
  }
  msg.Format(" class ");
  if(G__more(fout,msg)) return(1);
  if(-1!=deftmplt->parent_tagnum) {
    msg.Format("%s::",G__fulltagname(deftmplt->parent_tagnum,1));
    if(G__more(fout,msg)) return(1);
  }
  msg.Format("%s\n",deftmplt->name);
  if(G__more(fout,msg)) return(1);

  if(detail) {
    memfunctmplt = &deftmplt->memfunctmplt;
    while(memfunctmplt->next) {
      msg.Format("%-20s%5d "
              ,G__stripfilename(G__srcfile[memfunctmplt->filenum].filename)
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
  if(detail) {
    struct G__IntList *ilist = deftmplt->instantiatedtagnum;
    while(ilist) {
       msg.Format("      %s\n",G__fulltagname((int)ilist->i,1));
      if(G__more(fout,msg)) return(1);
      ilist=ilist->next;
    }
  }
  return(0);
}

/****************************************************************
* G__display_eachtemplatefunc()
*
****************************************************************/
int G__display_eachtemplatefunc(FILE *fout, G__Definetemplatefunc *deftmpfunc)
{
  G__FastAllocString msg(G__LONGLINE);
  struct G__Templatearg *def_para;
  struct G__Templatefuncarg *pfuncpara;
  int i;
  msg.Format("%-20s%5d "
          ,G__stripfilename(G__srcfile[deftmpfunc->filenum].filename)
          ,deftmpfunc->line);
  if(G__more(fout,msg)) return(1);
  msg.Format("template<");
  if(G__more(fout,msg)) return(1);
  def_para=deftmpfunc->def_para;
  while(def_para) {
    switch(def_para->type) {
    case G__TMPLT_CLASSARG:
    case G__TMPLT_POINTERARG1:
    case G__TMPLT_POINTERARG2:
    case G__TMPLT_POINTERARG3:
      msg.Format("class ");
      if(G__more(fout,msg)) return(1);
      break;
    case G__TMPLT_TMPLTARG:
      msg.Format("template<class U> class ");
      if(G__more(fout,msg)) return(1);
      break;
    case G__TMPLT_SIZEARG:
      msg.Format("size_t ");
      if(G__more(fout,msg)) return(1);
      break;
    default:
      msg.Format("%s ",G__type2string(def_para->type,-1,-1,0,0));
      if(G__more(fout,msg)) return(1);
      break;
    }
    msg.Format("%s",def_para->string);
    if(G__more(fout,msg)) return(1);
    switch(def_para->type) {
    case G__TMPLT_POINTERARG3: fprintf(fout,"*"); G__more_col(1);
    // Fallthrough
    case G__TMPLT_POINTERARG2: fprintf(fout,"*"); G__more_col(1);
    // Fallthrough
    case G__TMPLT_POINTERARG1: fprintf(fout,"*"); G__more_col(1);
    }
    def_para=def_para->next;
    if(def_para) fprintf(fout,",");
    else         fprintf(fout,">");
    G__more_col(1);
  }
  msg.Format(" func ");
  if(G__more(fout,msg)) return(1);
  if(-1!=deftmpfunc->parent_tagnum) {
    msg.Format("%s::",G__fulltagname(deftmpfunc->parent_tagnum,1));
    if(G__more(fout,msg)) return(1);
  }
  msg.Format("%s(",deftmpfunc->name);
  if(G__more(fout,msg)) return(1);
  def_para=deftmpfunc->def_para;
  pfuncpara = &deftmpfunc->func_para;
  for(i=0;i<pfuncpara->paran;i++) {
    if(i) {
      msg.Format(",");
      if(G__more(fout,msg)) return(1);
    }
    if(pfuncpara->argtmplt[i]>0) {
      msg.Format("%s",G__gettemplatearg(pfuncpara->argtmplt[i],def_para));
      if(G__more(fout,msg)) return(1);
      if(isupper(pfuncpara->type[i])) {
        fprintf(fout,"*");
        G__more_col(1);
      }
    }
    else if(pfuncpara->argtmplt[i]<-1) {
      if(pfuncpara->typenum[i]) 
        msg.Format("%s<",G__gettemplatearg(pfuncpara->typenum[i],def_para));
      else
        msg.Format("X<");
      if(G__more(fout,msg)) return(1);
      if(pfuncpara->tagnum[i]) 
        msg.Format("%s>",G__gettemplatearg(pfuncpara->tagnum[i],def_para));
      else
        msg.Format("Y>");
      if(G__more(fout,msg)) return(1);
    }
    else {
      msg.Format("%s",G__type2string(pfuncpara->type[i]
                                       ,pfuncpara->tagnum[i]
                                       ,pfuncpara->typenum[i]
                                       ,pfuncpara->reftype[i]
                                       ,0));
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
int G__display_template(FILE *fout,const char *name)
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
int G__display_includepath(FILE *fout)
{
  fprintf(fout,"include path: %s\n",G__allincludepath);
  return(0);
}

/****************************************************************
* G__display_macro()
*
****************************************************************/
int G__display_macro(FILE *fout,const char *name)
{
  struct G__Deffuncmacro *deffuncmacro;
  struct G__Charlist *charlist;
  int i=0;

  struct G__var_array *var = &G__global;
  int ig15;
  G__FastAllocString msg(G__LONGLINE);
  while(name&&name[i]&&isspace(name[i])) i++;

  while(var) {
    for(ig15=0;ig15<var->allvar;ig15++) {
      if(name && name[i] && strcmp(name+i,var->varnamebuf[ig15])!=0) continue;
      if('p'==var->type[ig15]) {
        msg.Format("#define %s %d\n",var->varnamebuf[ig15]
                ,*(int*)var->p[ig15]);
        G__more(fout,msg);
      }
      else if('T'==var->type[ig15]) {
        msg.Format("#define %s \"%s\"\n",var->varnamebuf[ig15]
                ,*(char**)var->p[ig15]);
        G__more(fout,msg);
      }
      if(name && name[i]) return(0);
    }
    var=var->next;
  }

  if(G__display_replacesymbol(fout,name+i)) return(0);


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
        if(charlist->string) fprintf(fout,"%s%s",charlist->string,"");
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
int G__display_files(FILE *fout)
{
   G__FastAllocString msg(G__ONELINE);
   int i;
   for(i=0;i<G__nfile;i++) {
      if (G__srcfile[i].ispermanentsl==2) {
         msg.Format("%3d fp=%14s lines=%-4d*file=\"%s\" "
                 ,i,"via hard link",G__srcfile[i].maxline 
                 ,G__srcfile[i].filename);
      } else if(G__srcfile[i].hasonlyfunc) {
         msg.Format("%3d fp=0x%012lx lines=%-4d*file=\"%s\" "
                 ,i,(long)G__srcfile[i].fp,G__srcfile[i].maxline 
                 ,G__srcfile[i].filename);
      } else {
         msg.Format("%3d fp=0x%012lx lines=%-4d file=\"%s\" "
                 ,i,(long)G__srcfile[i].fp,G__srcfile[i].maxline 
                 ,G__srcfile[i].filename);
      }
      if(G__more(fout,msg)) return(1);
      if(G__srcfile[i].prepname) {
         msg.Format("cppfile=\"%s\"",G__srcfile[i].prepname);
         if(G__more(fout,msg)) return(1);
      }
      if(G__more(fout,"\n")) return(1);
   }
   msg.Format("G__MAXFILE = %d\n",G__MAXFILE);
   if(G__more(fout,"\n")) return(1);
   return(0);
}
   
/********************************************************************
* G__pr
*
*  print source file
*
********************************************************************/
int G__pr(FILE *fout,const G__input_file &view)
{
  int center,thisline,filenum;
  G__FastAllocString G__oneline(G__LONGLINE*2);
  int top,bottom,screen,line=0;
  fpos_t store_fpos;
  /* char original[G__MAXFILENAME]; */
  FILE *G__fp;
  int tempopen;
#if defined(__hpux) || defined(__GNUC__)
  char *lines;
#endif

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

  if(G__istrace&0x80) screen = 2;

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
      fprintf(fout,"\t%s\n",G__oneline());
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
int G__dump_tracecoverage(FILE *fout)
{
  short int iarg;
  struct G__input_file view;
  for(iarg=0;iarg<G__nfile;iarg++) {
    if(G__srcfile[iarg].fp) {
      view.line_number=0;
      view.filenum=iarg;
      view.fp=G__srcfile[iarg].fp;
      G__strlcpy(view.name,G__srcfile[iarg].filename,G__MAXFILENAME);
      fprintf(fout
              ,"%s trace coverage==========================================\n"
              ,view.name);
      G__pr(fout,view);
    }
  }
  return(0);  
}
/******************************************************************
* void G__objectmonitor()
*
******************************************************************/
int G__objectmonitor(FILE *fout,long pobject,int tagnum,const char *addspace)
{
  struct G__inheritance *baseclass;
  G__FastAllocString space(G__ONELINE);
  G__FastAllocString msg(G__LONGLINE);
  int i;

  space.Format("%s  ",addspace);

  baseclass = G__struct.baseclass[tagnum];
  for(i=0;i<baseclass->basen;i++) {
    if(baseclass->herit[i]->property&G__ISDIRECTINHERIT) {
      if(baseclass->herit[i]->property&G__ISVIRTUALBASE) {
        if(0>G__getvirtualbaseoffset(pobject,tagnum,baseclass,i)) {
           msg.Format("%s-0x%-7lx virtual ",space()
                  ,-1*G__getvirtualbaseoffset(pobject,tagnum,baseclass,i));
        }
        else {
           msg.Format("%s0x%-8lx virtual ",space()
                  ,G__getvirtualbaseoffset(pobject,tagnum,baseclass,i));
        }
        if(G__more(fout,msg)) return(1);
        msg[0] = 0;
        switch(baseclass->herit[i]->baseaccess) {
        case G__PRIVATE:   msg.Format("private: "); break;
        case G__PROTECTED: msg.Format("protected: "); break;
        case G__PUBLIC:    msg.Format("public: "); break;
        }
        if(G__more(fout,msg)) return(1);
        msg.Format("%s\n",G__fulltagname(baseclass->herit[i]->basetagnum,1));
        if(G__more(fout,msg)) return(1);
#ifdef G__NEVER_BUT_KEEP
        if(G__objectmonitor(fout
                         ,pobject+(*(long*)(pobject+baseclass->herit[i]->baseoffset))
                         ,baseclass->herit[i]->basetagnum,space))
          return(1);
#endif
      }
      else {
         msg.Format("%s0x%-8lx ",space(),baseclass->herit[i]->baseoffset);
        if(G__more(fout,msg)) return(1);
        msg[0] = 0;
        switch(baseclass->herit[i]->baseaccess) {
        case G__PRIVATE:   msg.Format("private: "); break;
        case G__PROTECTED: msg.Format("protected: "); break;
        case G__PUBLIC:    msg.Format("public: "); break;
        }
        if(G__more(fout,msg)) return(1);
        msg.Format("%s\n",G__fulltagname(baseclass->herit[i]->basetagnum,1));
        if(G__more(fout,msg)) return(1);
        if(G__objectmonitor(fout
                            ,pobject+baseclass->herit[i]->baseoffset
                            ,baseclass->herit[i]->basetagnum,space))
          return(1);
      }
    }
  }
  G__incsetup_memvar(tagnum);
  if(G__varmonitor(fout,G__struct.memvar[tagnum],"",space,pobject)) return(1);
  return(0);
}

/******************************************************************
* void G__varmonitor()
*
******************************************************************/
int G__varmonitor(FILE *fout,G__var_array *var,const char *index,const char *addspace,long offset)
{
  int imon1;
  long addr;
  G__FastAllocString space(50);
  G__FastAllocString temp(G__ONELINE);
  G__FastAllocString msg(G__ONELINE);
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

  space.Format("%s  ",addspace);
  
  G__browsing=1;

  for(imon1=startindex;imon1<stopindex;imon1++) {

    if(!G__browsing) return(0);

    if(0==var->hash[imon1]) continue;

    if(G__LOCALSTATIC==var->statictype[imon1] && offset) addr=var->p[imon1];
    else addr=offset+var->p[imon1];

#ifdef G__VARIABLEFPOS
    if(var->filenum[imon1]>=0) 
      msg.Format("%-15s%4d "
              , G__stripfilename(G__srcfile[var->filenum[imon1]].filename)
              ,var->linenum[imon1]);
    else
      msg.Format("%-15s     " ,"(compiled)");
    if(G__more(fout,msg)) return(1);
#endif
    msg.Format("%s",addspace);
    if(G__more(fout,msg)) return(1);
    msg.Format("0x%-8lx ",addr);
    if(G__more(fout,msg)) return(1);

#ifndef G__NEWINHERIT
    if(var->isinherit[imon1]) {
      msg.Format("inherited ");
      if(G__more(fout,msg)) return(1);
    }
#endif

    precompiled_private=0;
    
    switch(var->access[imon1]) {
    case G__PUBLIC:
      /* fprintf(fout,"public: "); */
      break;
    case G__PROTECTED:
      msg.Format("protected: ");
      if(G__more(fout,msg)) return(1);
      if(-1!=var->tagnum && G__CPPLINK==G__struct.iscpplink[var->tagnum]) {
        precompiled_private=1;
      }
      break;
    case G__PRIVATE:
      msg.Format("private: ");
      if(G__more(fout,msg)) return(1);
      if(-1!=var->tagnum && G__CPPLINK==G__struct.iscpplink[var->tagnum]) {
        precompiled_private=1;
      }
      break;
    }
    switch(var->statictype[imon1]) {
       case G__USING_VARIABLE : /* variable brought in by a using statement */
          msg.Format("[using] ");
          if(G__more(fout,msg)) return(1);
          break;
       case G__USING_STATIC_VARIABLE : /* variable brought in by a using statement */
          msg.Format("[using] static ");
          if(G__more(fout,msg)) return(1);
          break;
       case G__COMPILEDGLOBAL : /* compiled global variable */
       case G__AUTO : /* auto */
          break;
       case G__LOCALSTATIC : /* static for function */
          msg.Format("static ");
          if(G__more(fout,msg)) return(1);
          break;
       case G__LOCALSTATICBODY : /* body for function static */
          msg.Format("body of static ");
          if(G__more(fout,msg)) return(1);
          break;
       default : /* static for file 0,1,2,... */
          if(var->statictype[imon1]>=0) { /* bug fix */
             msg.Format("file=%s static "
                        ,G__srcfile[var->statictype[imon1]].filename);
             if(G__more(fout,msg)) return(1);
          }
          else {
             msg.Format("static ");
             if(G__more(fout,msg)) return(1);
          }
          break;
    }
    
    msg.Format("%s"
            ,G__type2string((int)var->type[imon1],var->p_tagtable[imon1]
                            ,var->p_typetable[imon1],var->reftype[imon1]
                            ,var->constvar[imon1]));
    if(G__more(fout,msg)) return(1);
    msg.Format(" ");
    if(G__more(fout,msg)) return(1);
    msg.Format("%s",var->varnamebuf[imon1]);
    if(G__more(fout,msg)) return(1);
    if (var->varlabel[imon1][1] /* num of elements */ || var->paran[imon1]) {
      for (int ixxx = 0; ixxx < var->paran[imon1]; ++ixxx) {
        if (ixxx) {
          // -- Not a special case dimension, just print it.
          msg.Format( "[%d]", var->varlabel[imon1][ixxx+1]);
          if (G__more(fout, msg)) {
            return 1;
          }
        }
        else if (var->varlabel[imon1][1] /* num of elements */ == INT_MAX /* unspecified length flag */) {
          // -- Special case dimension, unspecified length.
          msg = "[]";
          if (G__more(fout, msg)) {
            return 1;
          }
        }
        else {
          // -- Special case dimension, first dimension must be calculated.
          msg.Format( "[%d]", var->varlabel[imon1][1] /* num of elements */ / var->varlabel[imon1][0] /* stride */);
          if (G__more(fout, msg)) {
            return 1;
          }
        }
      }
    }

    if (var->bitfield[imon1]) {
      msg.Format( " : %d (%d)", var->bitfield[imon1], var->varlabel[imon1][G__MAXVARDIM-1]);
      if (G__more(fout, msg)) {
        return 1;
      }
    }

    if ((offset != -1) && !precompiled_private && addr) {
      if (!var->varlabel[imon1][1] && !var->paran[imon1]) {
        switch (var->type[imon1]) {
        case 'T': 
          msg.Format("=\"%s\"",*(char**)addr); 
          if(G__more(fout,msg)) return(1);
          break;
#ifndef G__OLDIMPLEMENTATION2191
        case 'j': break;
#else
        case 'm': break;
#endif
        case 'p':
        case 'o': 
          msg.Format("=%d",*(int*)addr); 
          if(G__more(fout,msg)) return(1);
          break;
        case 'P':
        case 'O': 
          msg.Format("=%g",*(double*)addr); 
          if(G__more(fout,msg)) return(1);
          break;
        case 'u':
          msg.Format(" , size=%d",G__struct.size[var->p_tagtable[imon1]]);
          if(G__more(fout,msg)) return(1);
          temp[0]='\0';
          G__getcomment(temp,&var->comment[imon1],var->tagnum);
          if(temp[0]) {
             msg.Format(" //%s",temp());
            if(G__more(fout,msg)) return(1);
          }
          if(G__more(fout,"\n")) return(1);
          G__incsetup_memvar(var->p_tagtable[imon1]);
          if(G__varmonitor(fout,G__struct.memvar[var->p_tagtable[imon1]]
                           ,"",space,addr)) return(1);
          break;
        case 'b': 
          msg.Format("=%d",*(unsigned char*)addr); 
          if(G__more(fout,msg)) return(1);
          break;
        case 'c': 
          msg.Format("=%d ('%c')",*(char*)addr,*(char*)addr); 
          if(G__more(fout,msg)) return(1);
          break;
        case 's': 
          msg.Format("=%d",*(short*)addr); 
          if(G__more(fout,msg)) return(1);
          break;
        case 'r': 
          msg.Format("=%d",*(unsigned short*)addr); 
          if(G__more(fout,msg)) return(1);
          break;
        case 'i': 
          msg.Format("=%d",*(int*)addr); 
          if(G__more(fout,msg)) return(1);
          break;
        case 'h': 
          msg.Format("=%d",*(unsigned int*)addr); 
          if(G__more(fout,msg)) return(1);
          break;
        case 'l': 
          msg.Format("=%ld",*(long*)addr); 
          if(G__more(fout,msg)) return(1);
          break;
        case 'k': 
          msg.Format("=0x%lx",*(unsigned long*)addr); 
          if(G__more(fout,msg)) return(1);
          break;
        case 'f': 
          msg.Format("=%g",*(float*)addr); 
          if(G__more(fout,msg)) return(1);
          break;
        case 'd': 
          msg.Format("=%g",*(double*)addr); 
          if(G__more(fout,msg)) return(1);
          break;
        case 'g': 
#ifdef G__BOOL4BYTE
          msg.Format("=%d",(*(int*)addr)?1:0); 
#else
          msg.Format("=%d",(*(unsigned char*)addr)?1:0); 
#endif
          if(G__more(fout,msg)) return(1);
          break;
        case 'n': /* long long */
          msg.Format("=%lld",(*(G__int64*)addr)); 
          if(G__more(fout,msg)) return(1);
          break;
        case 'm': /* unsigned long long */
          msg.Format("=%llu",(*(G__uint64*)addr)); 
          if(G__more(fout,msg)) return(1);
          break;
        case 'q': /* long double */
          msg.Format("=%Lg",(*(long double*)addr)); 
          if(G__more(fout,msg)) return(1);
          break;
        default: 
          msg.Format("=0x%lx",*(long*)addr); 
          if(G__more(fout,msg)) return(1);
          break;
        }
        if('u'!=var->type[imon1]) if(G__more(fout,"\n")) return(1);
      }
      else {
        switch(var->type[imon1]) {
        case 'c':
          if(isprint(*(char*)addr))
            msg.Format("=0x%lx=\"%s\"",addr,(char*)addr); 
          else
            msg.Format("=0x%lx",addr); 
          if(G__more(fout,msg)) return(1);
          break;
        default: 
          msg.Format("=0x%lx",addr); 
          if(G__more(fout,msg)) return(1);
          break;
        }
        temp[0]='\0';
        G__getcomment(temp,&var->comment[imon1],var->tagnum);
        if(temp[0]) {
           msg.Format(" //%s",temp());
          if(G__more(fout,msg)) return(1);
        }
        if(G__more(fout,"\n")) return(1);
      }
    }
    else {
      if('u'==var->type[imon1]) {
        msg.Format(" , size=%d",G__struct.size[var->p_tagtable[imon1]]);
        if(G__more(fout,msg)) return(1);
        temp[0]='\0';
        G__getcomment(temp,&var->comment[imon1],var->tagnum);
        if(temp[0]) {
           msg.Format(" //%s",temp());
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
           msg.Format(" //%s",temp());
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
void G__set_errmsgcallback(void *p)
{
  G__ErrMsgCallback = (G__ErrMsgCallback_t)p;
}

/**************************************************************************
* G__mask_errmsg()
**************************************************************************/
void G__mask_errmsg(char * /* msg */)
{
}

/**************************************************************************
* G__get_errmsgcallback()
**************************************************************************/
void* G__get_errmsgcallback()
{
  return((void*)G__ErrMsgCallback);
}

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
int G__fprinterr(FILE* fp,const char* fmt,...)
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
  int result = 0;
  va_list argptr;
  va_start(argptr,fmt);
  if(G__ErrMsgCallback && G__serr==G__stderr) {
    char *buf;
#ifdef G__WIN32
    FILE *fpnull = fopen("NUL","w");
#else
    FILE *fpnull = fopen("/dev/null","w");
#endif
    if (fpnull==0) {
       vfprintf(stderr,"Could not open /dev/null!\n",argptr);
    } else {
       int len;
       len = vfprintf(fpnull,fmt,argptr);
       buf = (char*)malloc(len+5);
       /* Reset the counter */
       va_start(argptr,fmt);       
       result = vsprintf(buf,fmt,argptr); // Okay, we allocated the right size.
       (*G__ErrMsgCallback)(buf);
       free((void*)buf);
       fclose(fpnull);
    }
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
int G__fputerr(int c)
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
#undef signal


static int G__masksignal=0;

/**************************************************************************
* G__signal()
**************************************************************************/
#ifndef G__SYMANTEC
G__signaltype G__signal(int sgnl,void (*f) G__P((int)))
#else
void* G__signal(sgnl,f)
int sgnl;
void (*f) G__P((int));
#endif
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
int G__setmasksignal(int masksignal)
{
  G__masksignal=masksignal;
  return(0);
}

/**************************************************************************
* G__setautoconsole()
**************************************************************************/
void G__setautoconsole(int autoconsole)
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
      G__update_stdio();
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
int G__printf(const char *fmt,...)
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
int G__fprintf(FILE *fp,const char *fmt,...)
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

#undef system
  /* Simply call system() system call */
  return(system(com));

}

static void G__PrintLastErrorMsg(const char* text) {
  char* lpMsgBuf;
  DWORD dw = GetLastError(); 

  ::FormatMessage(FORMAT_MESSAGE_ALLOCATE_BUFFER | 
                  FORMAT_MESSAGE_FROM_SYSTEM |
                  FORMAT_MESSAGE_IGNORE_INSERTS,
                  NULL,
                  dw,
                  MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                  (char*)&lpMsgBuf,
                  0, NULL );

  G__fprinterr(G__serr,"%s\n%s (error code %d)\n", text, lpMsgBuf, dw);
  LocalFree(lpMsgBuf);
}

/**************************************************************************
* G__tmpfile()
**************************************************************************/
const char* G__tmpfilenam() {
   G__FastAllocString dirname(MAX_PATH);
   static char filename[MAX_PATH];
   
   if (!::GetTempPath(MAX_PATH, dirname)) {
      G__PrintLastErrorMsg("G__tmpfilenam: failed to determine temp directory!\n");
      return 0;
   }
   int trynumber = 0;
   while (trynumber < 50 && !::GetTempFileName(dirname, "cint_", 0, filename)) {
      if (++trynumber < 50)
         Sleep(200);
   }
   if (trynumber >= 50) {
      G__PrintLastErrorMsg("G__tmpfilenam: failed to create temporary file!\n");
      return 0;
   }
   return filename;
}

FILE* G__tmpfile() {
   return fopen(G__tmpfilenam(), "w+bTD"); // write and read (but write first), binary, temp, and delete when closed
}


#else /* G__WIN32 */

/**************************************************************************
* G__setautoconsole()
**************************************************************************/
void G__setautoconsole(int autoconsole)
{
  (void)autoconsole; /* no effect */
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
