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
#include "Dict.h"

using namespace Cint::Internal;

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

int Cint::Internal::G__browsing=1; /* used in disp.c and intrpt.c */
 
/*
 * Convert a string to a long long integer.
 *
 * Ignores `locale' stuff.  Assumes that the upper and lower case
 * alphabets and digits are each contiguous.
 */
extern "C" G__int64 G__expr_strtoll(const char *nptr,char **endptr, register int base) {
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
   cutoff = neg ? - (G__uint64) (LONG_LONG_MIN) : LONG_LONG_MAX;
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
extern "C" G__uint64 G__expr_strtoull(const char *nptr, char **endptr, register int base) {
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
   } else if (neg)
      acc = -acc;
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
void Cint::Internal::G__redirect_on() { G__redirected = 1; }
void Cint::Internal::G__redirect_off() { G__redirected = 0; }

static int G__more_len;
/***********************************************************************
* G__more_col()
***********************************************************************/
void Cint::Internal::G__more_col(int len)
{
  G__more_len += len;
}

/***********************************************************************
* G__more_pause()
***********************************************************************/
int Cint::Internal::G__more_pause(FILE *fp,int len)
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
      G__StrBuf buf_sb(G__MAXNAME);
      char *buf = buf_sb;
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
int Cint::Internal::G__more(FILE *fp,char *msg)
{
#ifndef G__OLDIMPLEMENTATION1485
  if(fp==G__serr) G__fprinterr(G__serr,"%s",msg);
  else fprintf(fp,"%s",msg);
#else
  fprintf(fp,"%s",msg);
#endif
  if(strchr(msg,'\n')) {
    return(G__more_pause(fp,strlen(msg)));
  }
  else {
    G__more_col(strlen(msg));
    return(0);
  }
}

/***********************************************************************
* void G__disp_purevirtualfunc
***********************************************************************/
void Cint::Internal::G__display_purevirtualfunc(int /* tagnum */)
{
  /* to be implemented */
}

/***********************************************************************
* void G__disp_friend
***********************************************************************/
static int G__display_friend(FILE *fp,const ::Reflex::Member &func)
{
  G__friendtag*friendtag = G__get_funcproperties(func)->entry.friendtag;
  G__StrBuf msg_sb(G__LONGLINE);
  char *msg = msg_sb;
  sprintf(msg," friend ");
  if(G__more(fp,msg)) return(1);
  while(friendtag) {
    sprintf(msg,"%s,",G__fulltagname(friendtag->tagnum,1));
    if(G__more(fp,msg)) return(1);
    friendtag = friendtag->next;
  }
  return(0);
}

/***********************************************************************
* void G__listfunc
***********************************************************************/
int Cint::Internal::G__listfunc(FILE *fp,int access,char *fname,const ::Reflex::Scope &ifunc)
{
   return G__listfunc_pretty(fp,access,fname,ifunc,0);
}

/***********************************************************************
* void G__listfunc_pretty
***********************************************************************/
int Cint::Internal::G__listfunc_pretty(FILE *fp,int access,const char *fname,const ::Reflex::Scope &i_func, char friendlyStyle)
{
  char msg[G__LONGLINE);];

  G__browsing=1;
  
  ::Reflex::Scope ifunc = i_func;
  if(!ifunc) ifunc = G__p_ifunc;
  
  bool showHeader = !friendlyStyle;
  showHeader |= (ifunc.FunctionMemberSize()>0 && G__get_funcproperties(*ifunc.FunctionMember_Begin())->filenum>=0); // if we need to display filenames

  if (showHeader) {
     if (!friendlyStyle || ifunc.IsTopScope()) {
        sprintf(msg,"%-15sline:size busy function type and name  ","filename");
        if(G__more(fp,msg)) return(1);
     }
     if(!ifunc.IsTopScope()) {
       sprintf(msg,"(in %s)\n",ifunc.Name().c_str());
       if(G__more(fp,msg)) return(1);
     }
     else {
       if(G__more(fp,"\n")) return(1);
     }
  }

  std::string parentname = ifunc.Name();
  
  /***************************************************
   * while interpreted function table list exists
   ***************************************************/
  {
     for(::Reflex::Member_Iterator i = ifunc.FunctionMember_Begin();
        i != ifunc.FunctionMember_End();
        ++i) {

      if(!G__browsing) return(0);

      if(fname && i->Name() != fname) continue;

      if(
          /* ifunc->hash[i] && */
         G__test_access(*i,access)) {
        
        /* print out file name and line number */
        if(G__get_funcproperties(*i)->filenum>=0) {
           int filenum = G__get_funcproperties(*i)->filenum;
           int linenum = G__get_funcproperties(*i)->linenum;
           if (G__get_funcproperties(*i)->entry.filenum>=0) {
              filenum = G__get_funcproperties(*i)->entry.filenum;
              linenum = G__get_funcproperties(*i)->entry.line_number;
           }
          sprintf(msg,"%-15s%4d:%-3d%c%2d "
                  ,G__stripfilename(G__srcfile[filenum].filename)
                  ,linenum
#ifdef G__ASM_FUNC
                  ,G__get_funcproperties(*i)->entry.size
#else
                  ,0
#endif
#ifdef G__ASM_WHOLEFUNC
                  ,(G__get_funcproperties(*i)->entry.bytecode)? '*' : ' '
#else
                  ,' '
#endif
                  ,G__globalcomp?G__get_funcproperties(*i)->globalcomp:G__get_funcproperties(*i)->entry.busy
                  );
          if(G__more(fp,msg)) return(1);
#ifdef G__ASM_DBG
          if(G__get_funcproperties(*i)->entry.bytecode) {
            G__ASSERT(G__get_funcproperties(*i)->entry.bytecodestatus==G__BYTECODE_SUCCESS||
                      G__get_funcproperties(*i)->entry.bytecodestatus==G__BYTECODE_ANALYSIS);
          }
#ifndef G__OLDIMPLEMENTATIN2021
          else if(G__get_funcproperties(*i)->entry.size<0) {
          }
#endif
          else {
            G__ASSERT(G__get_funcproperties(*i)->entry.bytecodestatus==G__BYTECODE_FAILURE||
                      G__get_funcproperties(*i)->entry.bytecodestatus==G__BYTECODE_NOTYET);
          }
          if(G__get_funcproperties(*i)->entry.bytecodestatus==G__BYTECODE_SUCCESS
             ||G__get_funcproperties(*i)->entry.bytecodestatus==G__BYTECODE_ANALYSIS
             ) {
            G__ASSERT(G__get_funcproperties(*i)->entry.bytecode);
          }
          else {
            G__ASSERT(!G__get_funcproperties(*i)->entry.bytecode);
          }
#endif
        }
        else {
          if (!friendlyStyle) {
            sprintf(msg,"%-15s%4d:%-3d%3d " ,"(compiled)" ,0,0 ,G__get_funcproperties(*i)->entry.busy);
            if(G__more(fp,msg)) return(1);
          }
        }
        
        if(1 /* ifunc->hash[i] */ ) {
          // sprintf(msg,"%s ",G__access2string(ifunc->access[i]));
           if (i->IsPublic()) strcpy(msg,"public: ");
           else if (i->IsProtected()) strcpy(msg,"protected: ");
           else strcpy(msg,"private: ");
        } else
          sprintf(msg,"------- ");
        if(G__more(fp,msg)) return(1);
        if(i->IsExplicit()) {
          sprintf(msg,"explicit ");
          if(G__more(fp,msg)) return(1);
        }
#ifndef G__NEWINHERIT
        if(ifunc->isinherit[i]) { 
          sprintf(msg,"inherited ");
          if(G__more(fp,msg)) return(1);
        }
#endif
        if(i->IsVirtual()) {
          sprintf(msg,"virtual ");
          if(G__more(fp,msg)) return(1);
        }

        if(i->IsStatic()) {
          sprintf(msg,"static ");
          if(G__more(fp,msg)) return(1);
        }

        
        /* print out type of return value */
        sprintf(msg,"%s ",i->TypeOf().ReturnType().Name(::Reflex::SCOPED | ::Reflex::QUALIFIED).c_str());
        if(G__more(fp,msg)) return(1);
        
        /*****************************************************
         * to get type of function parameter
         *****************************************************/
        /**********************************************************
         * print out type and name of function and parameters
         **********************************************************/
        /* print out function name */
        if(i->Name().length()>=sizeof(msg)-6) {
          strncpy(msg,i->Name().c_str(),sizeof(msg)-3);
          msg[sizeof(msg)-6]=0;
          strcat(msg,"...(");
        }
        else {
          if (friendlyStyle) {
             sprintf(msg,"%s::",parentname.c_str());
             if(G__more(fp,msg)) return(1);
          }
          sprintf(msg,"%s(",i->Name().c_str());
        }
        if(G__more(fp,msg)) return(1);

        if(G__get_funcproperties(*i)->entry.ansi && 0==i->FunctionParameterSize()) {
          sprintf(msg,"void");
          if(G__more(fp,msg)) return(1);
        }
        
        /* print out parameter types */
        for(unsigned int n=0;n<i->FunctionParameterSize();n++) {
          
          if(n!=0) {
            sprintf(msg,",");
            if(G__more(fp,msg)) return(1);
          }
          /* print out type of return value */
#ifndef G__OLDIMPLEMENATTION401
          sprintf(msg,"%s",i->TypeOf().FunctionParameterAt(n).Name(::Reflex::SCOPED | ::Reflex::QUALIFIED).c_str());
             //G__type2string(ifunc->para_type[i][n]
             //                            ,ifunc->para_p_tagtable[i][n]
             //                            ,G__get_typenum(ifunc->para_p_typetable[i][n])
             //                            ,ifunc->para_reftype[i][n]
             //                            ,ifunc->para_isconst[i][n]));
#else
          sprintf(msg,"%s",G__type2string(ifunc->para_type[i][n]
                                         ,ifunc->para_p_tagtable[i][n]
                                         ,ifunc->para_p_typetable[i][n]
                                         ,ifunc->para_reftype[i][n]));
#endif
          if(G__more(fp,msg)) return(1);

          if(i->FunctionParameterNameAt(n).c_str()[0]) {
            sprintf(msg," %s",i->FunctionParameterNameAt(n).c_str());
            if(G__more(fp,msg)) return(1);
          }
          if(i->FunctionParameterDefaultAt(n).c_str()[0]) {
            sprintf(msg,"=%s",i->FunctionParameterDefaultAt(n).c_str());
            if(G__more(fp,msg)) return(1);
          }
        }
        if(2==G__get_funcproperties(*i)->entry.ansi) {
          sprintf(msg," ...");
          if(G__more(fp,msg)) return(1);
        }
        sprintf(msg,")");
        if(G__more(fp,msg)) return(1);
        if(i->IsConst()) {
          sprintf(msg," const");
          if(G__more(fp,msg)) return(1);
        }
        if(i->IsAbstract()) {
          sprintf(msg,"=0");
          if(G__more(fp,msg)) return(1);
        }
        sprintf(msg,";");
        if(G__more(fp,msg)) return(1);
        G__StrBuf temp_sb(G__ONELINE);
        char *temp = temp_sb;
        temp[0] = '\0';
        G__getcomment(temp,&(G__get_funcproperties(*i)->comment),G__get_tagnum(i->DeclaringScope()));
        if(temp[0]) {
          sprintf(msg," //%s",temp);
          if(G__more(fp,msg)) return(1);
        }
        if(G__get_funcproperties(*i)->entry.friendtag) 
          if(G__display_friend(fp,*i)) return(1);
        if(G__more(fp,"\n")) return(1);
      }
      
    }
  } /* end of while(ifunc) */

  return(0);
}




/**************************************************************************
* G__showstack()
*
**************************************************************************/
int Cint::Internal::G__showstack(FILE *fout)
{
   int temp,temp1;
   ::Reflex::Scope local;
   G__StrBuf syscom_sb(G__MAXNAME);
   char *syscom = syscom_sb;
   G__StrBuf msg_sb(G__LONGLINE);
   char *msg = msg_sb;

   local=G__p_local;
   temp=0;
   while(local) {
#ifdef G__VAARG
      sprintf(msg,"%d ",temp);
      if(G__more(fout,msg)) return(1);
      if(G__get_properties(local)->stackinfo.exec_memberfunc && !local.IsTopScope()) {
         sprintf(msg,"%s::",local.Name(::Reflex::SCOPED).c_str());
         if(G__more(fout,msg)) return(1);
      }
      sprintf(msg,"%s(",G__get_properties(local)->stackinfo.ifunc.Name().c_str());
      if(G__more(fout,msg)) return(1);
      for(temp1=0;temp1<G__get_properties(local)->stackinfo.libp->paran;temp1++) {
         if(temp1) {
            sprintf(msg,",");
            if(G__more(fout,msg)) return(1);
         }
         G__valuemonitor(G__get_properties(local)->stackinfo.libp->para[temp1],syscom);
         if(G__more(fout,syscom)) return(1);
      }
      if(-1!=G__get_properties(local)->stackinfo.prev_filenum) {
         sprintf(msg,") [%s: %d]\n" 
            ,G__stripfilename(G__srcfile[G__get_properties(local)->stackinfo.prev_filenum].filename)
            ,G__get_properties(local)->stackinfo.prev_line_number);
         if(G__more(fout,msg)) return(1);
      }
      else {
         if(G__more(fout,") [entry]\n")) return(1);
      }
#else
      sprintf(msg,"%d %s() [%s: %d]\n" ,temp ,G__get_properties(local)->stackinfo.ifunc->funcname[local->ifn]
      ,G__filenameary[G__get_properties(local)->stackinfo.prev_filenum] ,G__get_properties(local)->stackinfo.prev_line_number);
      if(G__more(fout,msg)) return(1) ;
#endif
      ++temp;
      local=G__get_properties(local)->stackinfo.calling_scope;
   }
   return(0);
}

/**************************************************************************
* G__getdictpos()
**************************************************************************/
struct G__dictposition* Cint::Internal::G__get_dictpos(char *fname)
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
int Cint::Internal::G__display_newtypes(FILE *fout,char *fname)
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
    if(G__display_class(fout,"",0,dict->tagnum)) return(1);
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
int Cint::Internal::G__display_string(FILE *fout)
{
  int len;
  unsigned long totalsize=0;
  struct G__ConstStringList *pconststring;
  G__StrBuf msg_sb(G__ONELINE);
  char *msg = msg_sb;

  pconststring = G__plastconststring;
  while(pconststring->prev) {
    len=strlen(pconststring->string);
    totalsize+=len+1;
    if(totalsize>=sizeof(msg)-5) {
      sprintf(msg,"%3d ",len);
      strncpy(msg+4,pconststring->string,sizeof(msg)-5);
      msg[sizeof(msg)-1]=0;
    }
    else {
      sprintf(msg,"%3d %s\n",len,pconststring->string);
    }
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
static int G__display_classinheritance(FILE *fout,int tagnum,char *space)
{
  int i;
  struct G__inheritance *baseclass;
  char addspace[50];
  G__StrBuf temp_sb(G__ONELINE);
  char *temp = temp_sb;
  G__StrBuf msg_sb(G__LONGLINE);
  char *msg = msg_sb;

  baseclass = G__struct.baseclass[tagnum];

  if(NULL==baseclass) return(0);

  sprintf(addspace,"%s  ",space);

  for(i=0;i<baseclass->basen;i++) {
    if(baseclass->property[i]&G__ISDIRECTINHERIT) {
      sprintf(msg,"%s0x%-8p ",space ,baseclass->baseoffset[i]);
      if(G__more(fout,msg)) return(1);
      if(baseclass->property[i]&G__ISVIRTUALBASE) {
        sprintf(msg,"virtual ");
        if(G__more(fout,msg)) return(1);
      }
      if(baseclass->property[i]&G__ISINDIRECTVIRTUALBASE) {
        sprintf(msg,"(virtual) ");
        if(G__more(fout,msg)) return(1);
      }
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
static int G__display_membervariable(FILE *fout,int tagnum,int base)
{
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
  ::Reflex::Type var = G__Dict::G__Dict().GetType(tagnum);
  /* member variable */
  if(var) {
    fprintf(fout,"Defined in %s\n",var.Name().c_str());
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
   ::Reflex::Scope store_ifunc;
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
  ::Reflex::Scope scope = G__Dict::G__Dict().GetScope(tagnum);
  G__incsetup_memfunc(tagnum);
  if(scope.FunctionMemberSize()) {

     store_ifunc = G__p_ifunc;
     store_exec_memberfunc=G__exec_memberfunc;

     G__p_ifunc = scope;
     G__exec_memberfunc=0;

     tmp=G__listfunc(fout,access);

     G__p_ifunc=store_ifunc;
     G__exec_memberfunc=store_exec_memberfunc;
     if(tmp) return(1);
  }
  return(0);
}
  
/****************************************************************
* G__display_typedef()
*
****************************************************************/
int Cint::Internal::G__display_typedef(FILE *fout,char *name,int startin)
{
   int k;
   ::Reflex::Type_Iterator start,stop;
   G__StrBuf temp_sb(G__ONELINE);
   char *temp = temp_sb;
   G__StrBuf msg_sb(G__LONGLINE);
   char *msg = msg_sb;

   k=0;
   while(name[k]&&isspace(name[k])) k++;
   if(name[k]) {
      ::Reflex::Type starttype = G__find_typedef(name+k);
      if(!starttype) {
         G__fprinterr(G__serr,"!!!Type %s is not defined\n",name+k);
         return(0);
      }
      stop = std::find(::Reflex::Type::Type_Begin(),
         ::Reflex::Type::Type_End(), starttype);
      start = stop++;
   }
   else {
      if (startin>0) {
         ::Reflex::Type starttype = G__Dict::GetDict().GetTypedef(startin);
         if(!starttype) {
            G__fprinterr(G__serr,"!!!Type %d is not defined\n",startin);
            return(0);
         }
         start = std::find(::Reflex::Type::Type_Begin(),
            ::Reflex::Type::Type_End(), starttype);
      } else {
         start = ::Reflex::Type::Type_Begin();
      }
      stop = ::Reflex::Type::Type_End();
   }

   G__browsing=1;

   G__more(fout,"List of typedefs\n");

   for(::Reflex::Type_Iterator iTypedef = start; iTypedef != stop; ++iTypedef) {
      if(!G__browsing) return(0);
      if (!iTypedef->IsTypedef()) continue;
#ifdef G__TYPEDEFFPOS
      G__RflxProperties* prop = G__get_properties(*iTypedef);
      if(prop && prop->filenum>=0) 
         sprintf(msg,"%-15s%4d "
         ,G__stripfilename(G__srcfile[prop->filenum].filename)
         ,prop->linenum);
      else
         sprintf(msg,"%-15s     " ,"(compiled)");
      if(G__more(fout,msg)) return(1);
      int typedef_char_type = G__get_type(*iTypedef);
#endif
      if(
#ifndef G__OLDIMPLEMENTATION2191
         '1'==typedef_char_type
#else
         'Q'==typedef_char_type
#endif
         ) {
            /* pointer to static function */
            sprintf(msg,"typedef void* %s", iTypedef->Name(::Reflex::SCOPED).c_str()); 
            if(G__more(fout,msg)) return(1);
      }
      else if('a'==typedef_char_type) {
         /* pointer to member */
         sprintf(msg,"typedef G__p2memfunc %s", iTypedef->Name(::Reflex::SCOPED).c_str()); 
         if(G__more(fout,msg)) return(1);
      }
      else {
         /* G__typedef may need to be changed to add isconst member */
         sprintf(msg,"typedef %s" ,G__type2string(tolower(typedef_char_type)
            ,G__get_tagnum(*iTypedef),-1
            ,G__get_reftype(*iTypedef)
            ,G__get_isconst(*iTypedef))); 
         if(G__more(fout,msg)) return(1);
         if(G__more(fout," ")) return(1);
         if(isupper(typedef_char_type) && iTypedef->ToType().IsArray()) {
            sprintf(msg,"(*%s)",iTypedef->Name(::Reflex::SCOPED).c_str());
            if(G__more(fout,msg)) return(1);
         }
         else {
            if(isupper(typedef_char_type)) {
               if(iTypedef->ToType().IsConst() && iTypedef->ToType().IsPointer()) sprintf(msg,"*const ");
               else sprintf(msg,"*");
               if(G__more(fout,msg)) return(1);
            }
            sprintf(msg,"%s",iTypedef->Name(::Reflex::SCOPED).c_str());
            if(G__more(fout,msg)) return(1);
         }

         for(::Reflex::Type arrayType = *iTypedef; 
            arrayType.IsArray(); 
            arrayType = arrayType.ToType()) {
               sprintf(msg,"[%lu]", arrayType.ArrayLength());

               if(G__more(fout,msg)) return(1);
         }
      }
      temp[0]='\0';
      if (G__get_properties(*iTypedef))
         G__getcommenttypedef(temp,&G__get_properties(*iTypedef)->comment,*iTypedef);
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
static int G__display_eachtemplate(FILE *fout,G__Definedtemplateclass *deftmplt,int detail)
{
  struct G__Templatearg *def_para;
  struct G__Definedtemplatememfunc *memfunctmplt;
  fpos_t store_pos;
  /* char buf[G__LONGLINE]; */
  G__StrBuf msg_sb(G__LONGLINE);
  char *msg = msg_sb;
  int c;

  if(!deftmplt->def_fp) return(0);

  sprintf(msg,"%-20s%5d "
          ,G__stripfilename(G__srcfile[deftmplt->filenum].filename)
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
      sprintf(msg,"%s ",G__type2string(def_para->type,-1,-1,0,0));
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
  sprintf(msg," class ");
  if(G__more(fout,msg)) return(1);
  if(-1!=deftmplt->parent_tagnum) {
    sprintf(msg,"%s::",G__fulltagname(deftmplt->parent_tagnum,1));
    if(G__more(fout,msg)) return(1);
  }
  sprintf(msg,"%s\n",deftmplt->name);
  if(G__more(fout,msg)) return(1);

  if(detail) {
    memfunctmplt = &deftmplt->memfunctmplt;
    while(memfunctmplt->next) {
      sprintf(msg,"%-20s%5d "
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
      sprintf(msg,"      %s\n",G__fulltagname(ilist->i,1));
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
static int G__display_eachtemplatefunc(FILE *fout, G__Definetemplatefunc *deftmpfunc)
{
  G__StrBuf msg_sb(G__LONGLINE);
  char *msg = msg_sb;
  struct G__Templatearg *def_para;
  struct G__Templatefuncarg *pfuncpara;
  int i;
  sprintf(msg,"%-20s%5d "
          ,G__stripfilename(G__srcfile[deftmpfunc->filenum].filename)
          ,deftmpfunc->line);
  if(G__more(fout,msg)) return(1);
  sprintf(msg,"template<");
  if(G__more(fout,msg)) return(1);
  def_para=deftmpfunc->def_para;
  while(def_para) {
    switch(def_para->type) {
    case G__TMPLT_CLASSARG:
    case G__TMPLT_POINTERARG1:
    case G__TMPLT_POINTERARG2:
    case G__TMPLT_POINTERARG3:
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
      sprintf(msg,"%s ",G__type2string(def_para->type,-1,-1,0,0));
      if(G__more(fout,msg)) return(1);
      break;
    }
    sprintf(msg,"%s",def_para->string);
    if(G__more(fout,msg)) return(1);
    switch(def_para->type) {
    case G__TMPLT_POINTERARG3: fprintf(fout,"*"); G__more_col(1);
    case G__TMPLT_POINTERARG2: fprintf(fout,"*"); G__more_col(1);
    case G__TMPLT_POINTERARG1: fprintf(fout,"*"); G__more_col(1);
    }
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
      if(isupper(pfuncpara->type[i])) {
        fprintf(fout,"*");
        G__more_col(1);
      }
    }
    else if(pfuncpara->argtmplt[i]<-1) {
      if(pfuncpara->typenum[i]) 
        sprintf(msg,"%s<",G__gettemplatearg(G__get_typenum(pfuncpara->typenum[i]),def_para));
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
      sprintf(msg,"%s",G__type2string(pfuncpara->type[i]
                                       ,pfuncpara->tagnum[i]
                                       ,G__get_typenum(pfuncpara->typenum[i])
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
int Cint::Internal::G__display_template(FILE *fout,char *name)
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
extern "C" int G__display_includepath(FILE *fout)
{
  fprintf(fout,"include path: %s\n",G__allincludepath);
  return(0);
}

/****************************************************************
* G__display_macro()
*
****************************************************************/
int Cint::Internal::G__display_macro(FILE *fout,char *name)
{
  struct G__Deffuncmacro *deffuncmacro;
  struct G__Charlist *charlist;
  int i=0;

  ::Reflex::Scope var = ::Reflex::Scope::GlobalScope(); 
  G__StrBuf msg_sb(G__LONGLINE);
  char *msg = msg_sb;
  while(name[i]&&isspace(name[i])) i++;

  for(::Reflex::Member_Iterator ig15 = var.DataMember_Begin();
     ig15 != var.DataMember_End(); ++ig15) {
        if(name && name[i] && var.Name()!=(name+i)) continue;
        if(G__get_type(ig15->TypeOf())) {
           sprintf(msg,"#define %s %d\n",ig15->Name().c_str()
           ,*(int*)G__get_offset(*ig15));
           G__more(fout,msg);
        }
        else if(G__get_type(ig15->TypeOf())=='T') {
           sprintf(msg,"#define %s \"%s\"\n",ig15->Name().c_str()
           ,*(char**)G__get_offset(*ig15));
           G__more(fout,msg);
        }
        if(name && name[i]) return(0);
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
int Cint::Internal::G__display_files(FILE *fout)
{
  G__StrBuf msg_sb(G__ONELINE);
  char *msg = msg_sb;
  int i;
  for(i=0;i<G__nfile;i++) {
    if(G__srcfile[i].hasonlyfunc)
      sprintf(msg,"%3d fp=0x%8lx lines=%-4d*file=\"%s\" "
              ,i,(long)G__srcfile[i].fp,G__srcfile[i].maxline 
              ,G__srcfile[i].filename);
    else
      sprintf(msg,"%3d fp=0x%8lx lines=%-4d file=\"%s\" "
              ,i,(long)G__srcfile[i].fp,G__srcfile[i].maxline 
              ,G__srcfile[i].filename);
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
int Cint::Internal::G__pr(FILE *fout,G__input_file view)
{
  int center,thisline,filenum;
  G__StrBuf G__oneline_sb(G__LONGLINE*2);
  char *G__oneline = G__oneline_sb;
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
int Cint::Internal::G__dump_tracecoverage(FILE *fout)
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

/******************************************************************
* void G__objectmonitor()
*
******************************************************************/
int Cint::Internal::G__objectmonitor(FILE *fout,char *pobject,const ::Reflex::Type &tagnum,char *addspace)
{
  struct G__inheritance *baseclass;
  G__StrBuf space_sb(G__ONELINE);
  char *space = space_sb;
  G__StrBuf msg_sb(G__LONGLINE);
  char *msg = msg_sb;
  int i;

  sprintf(space,"%s  ",addspace);

  baseclass = G__struct.baseclass[G__get_tagnum(tagnum)];
  for(i=0;i<baseclass->basen;i++) {
    if(baseclass->property[i]&G__ISDIRECTINHERIT) {
      if(baseclass->property[i]&G__ISVIRTUALBASE) {
        if(0>G__getvirtualbaseoffset(pobject,G__get_tagnum(tagnum),baseclass,i)) {
          sprintf(msg,"%s-0x%-7lx virtual ",space
                  ,-1*G__getvirtualbaseoffset(pobject,G__get_tagnum(tagnum),baseclass,i));
        }
        else {
          sprintf(msg,"%s0x%-8lx virtual ",space
                  ,G__getvirtualbaseoffset(pobject,G__get_tagnum(tagnum),baseclass,i));
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
        sprintf(msg,"%s0x%-8p ",space ,baseclass->baseoffset[i]);
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
                            ,pobject+(size_t)baseclass->baseoffset[i]
                            ,G__Dict::G__Dict().GetType(baseclass->basetagnum[i]),space))
          return(1);
      }
    }
  }
  G__incsetup_memvar(G__get_tagnum(tagnum));
  if(G__varmonitor(fout,(::Reflex::Scope)tagnum,"",space,(long)pobject)) return(1);
  return(0);
}

/******************************************************************
* void G__varmonitor()
*
******************************************************************/
int Cint::Internal::G__varmonitor(FILE *fout,const ::Reflex::Scope &var,char *index,char *addspace,long offset)
{
   char *addr;
   char space[50];
   G__StrBuf temp_sb(G__ONELINE);
   char *temp = temp_sb;
   G__StrBuf msg_sb(G__ONELINE);
   char *msg = msg_sb;
   unsigned int startindex,stopindex;
   int precompiled_private;


   if(!var) {
      fprintf(fout,"No variable table\n");
      return(0);
   }

   if(index[0]=='\0') {
      startindex=0;
      stopindex=var.DataMemberSize();
   }
   else {
      if(isdigit(index[0])) {
         G__fprinterr(G__serr,"variable name must be specified\n");
         return(0);
      }
      else {
         startindex=0;
         while(index != var.DataMemberAt(startindex).Name()) {
            ++startindex;
            if(startindex>=var.DataMemberSize()) break;
         }
         if(startindex>=var.DataMemberSize()) {
            fprintf(fout,"Variable %s not found\n" ,index);
            return(0);
         }
      }
      stopindex=startindex+1;
   }

   sprintf(space,"%s  ",addspace);

   G__browsing=1;

   for(unsigned int imon1=startindex;imon1<stopindex;imon1++) {

      if(!G__browsing) return(0);

      ::Reflex::Member mon = var.DataMemberAt(imon1);

      //if(0==var->hash[imon1]) continue;
      if(!mon) continue;

      if(G__test_static(mon,G__LOCALSTATIC) && offset) addr=G__get_offset(mon);
      else addr=offset+G__get_offset(mon);

#ifdef G__VARIABLEFPOS
      if(G__get_properties(mon)->filenum>=0) 
         sprintf(msg,"%-15s%4d "
         , G__stripfilename(G__srcfile[G__get_properties(mon)->filenum].filename)
         ,G__get_properties(mon)->linenum);
      else
         sprintf(msg,"%-15s     " ,"(compiled)");
      if(G__more(fout,msg)) return(1);
#endif
      sprintf(msg,"%s",addspace);
      if(G__more(fout,msg)) return(1);
      sprintf(msg,"0x%-8p ",addr);
      if(G__more(fout,msg)) return(1);

#ifndef G__NEWINHERIT
      if(var->isinherit[imon1]) {
         sprintf(msg,"inherited ");
         if(G__more(fout,msg)) return(1);
      }
#endif

      precompiled_private=0;

      //switch(var->access[imon1]) {
      if (G__test_access(mon,G__PUBLIC)) {
         //case G__PUBLIC:
         /* fprintf(fout,"public: "); */
         //break;
      } else if (G__test_access(mon,G__PROTECTED)) {
         //case G__PROTECTED:
         sprintf(msg,"protected: ");
         if(G__more(fout,msg)) return(1);
         if(G__get_properties(var)->iscpplink==G__CPPLINK) {
            precompiled_private=1;
         }
         //break;
      } else if (G__test_access(mon,G__PRIVATE)) {
         //case G__PRIVATE:
         sprintf(msg,"private: ");
         if(G__more(fout,msg)) return(1);
         if(G__get_properties(var)->iscpplink==G__CPPLINK) {
            precompiled_private=1;
         }
         //break;
      }

      //switch(var->statictype[imon1]) {
      if (G__test_static(mon,G__COMPILEDGLOBAL)) {
         //case G__COMPILEDGLOBAL : /* compiled global variable */
      } else if (G__test_static(mon,G__AUTO)) {
         //case G__AUTO : /* auto */
         //break;
      } else if (G__test_static(mon,G__LOCALSTATIC)) {
         //case G__LOCALSTATIC : /* static for function */
         sprintf(msg,"static ");
         if(G__more(fout,msg)) return(1);
         //break;
      } else if (G__test_static(mon,G__LOCALSTATICBODY)) {
         //case G__LOCALSTATICBODY : /* body for function static */
         sprintf(msg,"body of static ");
         if(G__more(fout,msg)) return(1);
         //break;
      } else {
         //default : /* static for file 0,1,2,... */
         int statictype = G__get_properties(mon)->filenum;
         if (mon.IsStatic() && statictype>=0) { /* bug fix */
            sprintf(msg,"file=%s static "
               ,G__srcfile[statictype].filename);
            if(G__more(fout,msg)) return(1);
         }
         else {
            sprintf(msg,"static ");
            if(G__more(fout,msg)) return(1);
         }
         //break;
      }

      //strcpy(msg,mon.TypeOf().Name(Reflex::QUALIFIED).c_str());
      {
         ::Reflex::Type ty = mon.TypeOf();
         sprintf(msg, "%s", G__type2string(G__get_type(ty), G__get_tagnum(ty), G__get_typenum(ty), G__get_reftype(ty), G__get_isconst(ty)));
      }
      //sprintf(msg,"%s"
      //        ,G__type2string((int)var->type[imon1],var->p_tagtable[imon1]
      //                        ,G__get_typenum(var->p_typetable[imon1]),var->reftype[imon1]
      //                        ,var->constvar[imon1]));
      if(G__more(fout,msg)) return(1);
      sprintf(msg," ");
      if(G__more(fout,msg)) return(1);
      sprintf(msg,"%s",mon.Name().c_str());
      if(G__more(fout,msg)) return(1);
      if(G__get_varlabel(mon, 1) /* num of elements */ || G__get_paran(mon)) {
         int ixxx;
         for(ixxx=0;ixxx<G__get_paran(mon);ixxx++) {
            if(ixxx) {
               sprintf(msg,"[%d]",G__get_varlabel(var,ixxx+1));
               if(G__more(fout,msg)) return(1);
            }
            else if (G__get_varlabel(mon, 1) /* num of elements */ == INT_MAX /* unspecified length flag */) {
               strcpy(msg,"[]");
               if(G__more(fout,msg)) return(1);
            }
            else {
               sprintf(msg, "[%d]", G__get_varlabel(mon, 1) /* number of elements */ / G__get_varlabel(mon, 0) /* stride */);
               if(G__more(fout,msg)) return(1);
            }
         }
      }

    if(G__get_bitfield_width(mon)) {
      sprintf(msg," : %ld (%ld)",G__get_bitfield_width(mon)
              ,G__get_bitfield_start(mon));
      if(G__more(fout,msg)) return(1);
    }

    if(-1!=offset && 0==precompiled_private && addr) {
      if(!G__get_varlabel(mon, 1) && !G__get_paran(mon)) {
        switch(G__get_type(mon.TypeOf())) {
        case 'T': 
          sprintf(msg,"=\"%s\"",*(char**)addr); 
          if(G__more(fout,msg)) return(1);
          break;
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
          sprintf(msg," , size=%ld",mon.TypeOf().RawType().SizeOf());
          if(G__more(fout,msg)) return(1);
          temp[0]='\0';
          G__getcomment(temp,&G__get_properties(mon)->comment,G__get_tagnum(var));
          if(temp[0]) {
            sprintf(msg," //%s",temp);
            if(G__more(fout,msg)) return(1);
          }
          if(G__more(fout,"\n")) return(1);
          G__incsetup_memvar(G__get_tagnum(mon.TypeOf().RawType()));
          if(G__varmonitor(fout,mon.TypeOf().RawType()
                           ,"",space,(long)addr)) return(1);
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
        case 'l': 
          sprintf(msg,"=%ld",*(long*)addr); 
          if(G__more(fout,msg)) return(1);
          break;
        case 'k': 
          sprintf(msg,"=0x%lx",*(unsigned long*)addr); 
          if(G__more(fout,msg)) return(1);
          break;
        case 'f': 
          sprintf(msg,"=%g",*(float*)addr); 
          if(G__more(fout,msg)) return(1);
          break;
        case 'd': 
          sprintf(msg,"=%g",*(double*)addr); 
          if(G__more(fout,msg)) return(1);
          break;
        case 'g': 
#ifdef G__BOOL4BYTE
          sprintf(msg,"=%d",(*(int*)addr)?1:0); 
#else
          sprintf(msg,"=%d",(*(unsigned char*)addr)?1:0); 
#endif
          if(G__more(fout,msg)) return(1);
          break;
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
        default: 
          sprintf(msg,"=0x%lx",*(long*)addr); 
          if(G__more(fout,msg)) return(1);
          break;
        }
        if('u'!=G__get_type(mon.TypeOf())) if(G__more(fout,"\n")) return(1);
      }
      else {
        switch(G__get_type(mon.TypeOf())) {
        case 'c':
          if(isprint(*(char*)addr))
            sprintf(msg,"=0x%p=\"%s\"",addr,(char*)addr); 
          else
            sprintf(msg,"=0x%p",addr); 
          if(G__more(fout,msg)) return(1);
          break;
        default: 
          sprintf(msg,"=0x%p",addr); 
          if(G__more(fout,msg)) return(1);
          break;
        }
        temp[0]='\0';
        G__getcomment(temp,&G__get_properties(mon)->comment,G__get_tagnum(var));
        if(temp[0]) {
          sprintf(msg," //%s",temp);
          if(G__more(fout,msg)) return(1);
        }
        if(G__more(fout,"\n")) return(1);
      }
    }
    else {
      if('u'==G__get_type(mon.TypeOf())) {
        sprintf(msg," , size=%ld",mon.TypeOf().RawType().SizeOf());
        if(G__more(fout,msg)) return(1);
        temp[0]='\0';
        G__getcomment(temp,&G__get_properties(mon)->comment,G__get_tagnum(var));
        if(temp[0]) {
          sprintf(msg," //%s",temp);
          if(G__more(fout,msg)) return(1);
        }
        if(G__more(fout,"\n")) return(1);
        G__incsetup_memvar(G__get_tagnum(mon.TypeOf().RawType()));
        if(G__varmonitor(fout,mon.TypeOf().RawType()
                         ,"",space,offset)) return(1);
      }
      else {
        temp[0]='\0';
        G__getcomment(temp,&G__get_properties(mon)->comment,G__get_tagnum(var));
        if(temp[0]) {
          sprintf(msg," //%s",temp);
          if(G__more(fout,msg)) return(1);
        }
        if(G__more(fout,"\n")) return(1);
      }
    }
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

typedef void (*G__ErrMsgCallback_t)(char* msg);
static G__ErrMsgCallback_t G__ErrMsgCallback;

/**************************************************************************
* G__set_errmsgcallback()
**************************************************************************/
extern "C" void G__set_errmsgcallback(void *p)
{
  G__ErrMsgCallback = (G__ErrMsgCallback_t)p;
}

/**************************************************************************
* G__mask_errmsg()
**************************************************************************/
extern "C" void G__mask_errmsg(char * /* msg */)
{
}

/**************************************************************************
* G__get_errmsgcallback()
**************************************************************************/
extern "C" void* G__get_errmsgcallback()
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
extern "C" int G__fprinterr(FILE* fp,char* fmt,...)
#elif defined(__GNUC__)
extern "C" int G__fprinterr(fp,fmt)
FILE* fp;
char* fmt;
...
#else
extern "C" int G__fprinterr(fp,fmt,arg)
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
       result = vsprintf(buf,fmt,argptr);
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
extern "C" int G__fputerr(int c)
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
G__signaltype Cint::Internal::G__signal(int sgnl,void (*f)(int))
#else
void* G__signal(sgnl,f)
int sgnl;
void (*f)(int);
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
extern "C" int G__setmasksignal(int masksignal)
{
  G__masksignal=masksignal;
  return(0);
}

/**************************************************************************
* G__setautoconsole()
**************************************************************************/
extern "C" void G__setautoconsole(int autoconsole)
{
  G__autoconsole=autoconsole;
  G__isconsole=0;
}

/**************************************************************************
* G__AllocConsole()
**************************************************************************/
extern "C" int G__AllocConsole()
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
extern "C" int G__FreeConsole()
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
extern "C" int G__printf(char *fmt,...)
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
extern "C" int G__fprintf(FILE *fp,char *fmt,...)
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
int Cint::Internal::G__fputc(int character,FILE *fp)
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
int Cint::Internal::G__putchar(int character)
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
int Cint::Internal::G__fputs(char *string,FILE *fp)
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
int Cint::Internal::G__puts(char *string)
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
char *Cint::Internal::G__fgets(char *string,int n,FILE *fp)
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
char *Cint::Internal::G__gets(char *buffer)
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
int Cint::Internal::G__system(char *com)
{

#undef system
  /* Simply call system() system call */
  return(system(com));

}

/**************************************************************************
* G__tmpfile()
**************************************************************************/
const char* Cint::Internal::G__tmpfilenam() {
   char dirname[MAX_PATH];
   static char filename[MAX_PATH];
   if (!::GetTempPath(MAX_PATH, dirname)) return 0;
   if (!::GetTempFileName(dirname, "cint_", 0, filename)) return 0;
   return filename; // write and read (but write first), binary, temp, and delete when closed
}
FILE *Cint::Internal::G__tmpfile() {
   return fopen(G__tmpfilenam(), "w+bTD"); // write and read (but write first), binary, temp, and delete when closed
}

#else /* G__WIN32 */

/**************************************************************************
* G__setautoconsole()
**************************************************************************/
extern "C" void G__setautoconsole(int autoconsole)
{
  autoconsole=autoconsole; /* no effect */
}

/**************************************************************************
* G__AllocConsole()
**************************************************************************/
extern "C" int G__AllocConsole()
{
  return(0);
}

/**************************************************************************
* G__FreeConsole()
**************************************************************************/
extern "C" int G__FreeConsole()
{
  return(0);
}

#endif /* G__WIN32 */

/****************************************************************
* G__display_class()
*
****************************************************************/
extern "C" int G__display_class(FILE *fout,char *name,int base,int start)
{
  using namespace ::Cint::Internal;
  int tagnum;
  int i,j;
  struct G__inheritance *baseclass;
  G__StrBuf temp_sb(G__ONELINE);
  char *temp = temp_sb;
  G__StrBuf msg_sb(G__LONGLINE);
  char *msg = msg_sb;
  char *p;
  int store_globalcomp;
  int store_iscpp;

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
        sprintf(temp,"%d",i);
        G__display_class(fout,temp,0,0);
      }
      return(0);
    }
    /* no class name specified, list up all tagnames */
    if(G__more(fout,"List of classes\n")) return(1);
    sprintf(msg,"%-15s%5s\n","file","line");
    if(G__more(fout,msg)) return(1);
    for(i=start;i<G__struct.alltag;i++) {
      if(!G__browsing) return(0);
      switch(G__struct.iscpplink[i]) {
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
      sprintf(msg," %s ",G__type2string('u',i,-1,0,0));
      G__iscpp=store_iscpp; /* dirty trick reset */
      G__globalcomp=store_globalcomp;
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

  if((char*)NULL!=strstr(name+i,">>")) {
    /* dealing with A<A<int>> -> A<A<int> > */
    char *pt1;
    G__StrBuf tmpbuf_sb(G__ONELINE);
    char *tmpbuf = tmpbuf_sb;
    pt1 = strstr(name+i,">>");
    ++pt1;
    strcpy(tmpbuf,pt1);
    *pt1=' ';
    ++pt1;
    strcpy(pt1,tmpbuf);
  }

  if(isdigit(*(name+i))) tagnum = atoi(name+i);
  else                   tagnum = G__defined_tagname(name+i,0);

  /* no such class,struct */
  if(-1==tagnum||G__struct.alltag<=tagnum) return(0); 

      G__class_autoloading(tagnum);

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
            ,G__stripfilename(G__srcfile[G__struct.filenum[tagnum]].filename)
            ,G__struct.line_number[tagnum]);
  }
  if(G__more(fout,msg)) return(1);
  sprintf(msg
          ," (tagnum=%d,voffset=%p,isabstract=%d,parent=%d,gcomp=%d:%d,d21=~cd=%x)" 
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
