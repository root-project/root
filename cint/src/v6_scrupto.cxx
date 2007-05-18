/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file scrupto.c
 ************************************************************************
 * Description:
 *  Partial cleanup function
 ************************************************************************
 * Copyright(c) 1995~2004  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"

extern "C" {

/***********************************************************************
* How to use environment rewinding feature
*
*    G__dictposition pos;
*    G__store_dictposition(&pos);
*           .
*    // Do anything, 
*           .
*    if(!G__isfuncbusy(pos.nfile)) G__scratch_upto(&pos);
*
*
***********************************************************************/


/***********************************************************************
* void G__store_dictposition()
*
*
***********************************************************************/
void G__store_dictposition(G__dictposition *dictpos)
{
  G__LockCriticalSection();
  /* global variable position */
  dictpos->var = &G__global;
  while(dictpos->var->next) dictpos->var=dictpos->var->next;
  dictpos->ig15 = dictpos->var->allvar;

  dictpos->tagnum = G__struct.alltag;
  dictpos->conststringpos = G__plastconststring;
  dictpos->typenum = G__newtype.alltype;

  /* global function position */
  G__ifunc_table_internal* lastifunc = &G__ifunc;
  while(lastifunc->next) lastifunc=lastifunc->next;
  dictpos->ifunc = G__get_ifunc_ref(lastifunc);
  dictpos->ifn = lastifunc->allifunc;
  
  /* include path */
  dictpos->ipath = &G__ipathentry;
  while(dictpos->ipath->next) dictpos->ipath=dictpos->ipath->next;

  /* preprocessfilekey */
  dictpos->preprocessfilekey = &G__preprocessfilekey;
  while(dictpos->preprocessfilekey->next) 
    dictpos->preprocessfilekey=dictpos->preprocessfilekey->next;

#ifdef G__SHAREDLIB
  dictpos->allsl = G__allsl;
#endif

  dictpos->nfile = G__nfile;

  /* function macro */
  dictpos->deffuncmacro = &G__deffuncmacro;
  while(dictpos->deffuncmacro->next)
    dictpos->deffuncmacro=dictpos->deffuncmacro->next;

  /* template class */
  dictpos->definedtemplateclass = &G__definedtemplateclass;
  while(dictpos->definedtemplateclass->next)
    dictpos->definedtemplateclass=dictpos->definedtemplateclass->next;

  /* function template */
  dictpos->definedtemplatefunc = &G__definedtemplatefunc;
  while(dictpos->definedtemplatefunc->next)
    dictpos->definedtemplatefunc=dictpos->definedtemplatefunc->next;

  if(0!=dictpos->ptype && (char*)G__PVOID!=dictpos->ptype) {
    free((void*)dictpos->ptype);
    dictpos->ptype = (char*)NULL;
  }
  if(0==dictpos->ptype) {
    int i;
    dictpos->ptype = (char*)malloc(G__struct.alltag+1);
    for(i=0;i<G__struct.alltag;i++) dictpos->ptype[i] = G__struct.type[i];
  }

  G__UnlockCriticalSection();
}

/***********************************************************************
* void G__scratch_upto()
*
*
***********************************************************************/
void G__scratch_upto(G__dictposition *dictpos)
{
  /* int i; */

  /* struct G__var_array *local; */

  if(!dictpos) return;

  G__LockCriticalSection();

#ifdef G__MEMTEST
  fprintf(G__memhist,"G__scratch_upto() start\n");
#endif


  /*******************************************
   * clear interpriveve global variables
   *******************************************/
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing global variables\n");
#endif
  G__destroy_upto(dictpos->var,1,dictpos->var,dictpos->ig15);

#ifdef G__SECURITY
  /*************************************************************
   * Garbage collection
   *************************************************************/
  if(G__security&G__SECURE_GARBAGECOLLECTION) {
    G__garbagecollection();
  }
#endif

  /*************************************************************
   * Free struct tag name and member table
   *************************************************************/
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing struct tag table\n");
#endif
  G__free_struct_upto(dictpos->tagnum);


  /*************************************************************
   * Free typedef table
   *************************************************************/
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing typedef table\n");
#endif
  G__free_typedef_upto(dictpos->typenum);

  /*************************************************************
   * Initialize interpreted function table list
   *************************************************************/
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing interpret function table\n");
#endif
  G__free_ifunc_table_upto(&G__ifunc,G__get_ifunc_internal(dictpos->ifunc),dictpos->ifn);


  /********************************************
   * free include path list
   *********************************************/
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing include path list\n");
#endif
  G__free_ipath(dictpos->ipath);

#ifdef G__SHAREDLIB
  /********************************************
   * free dynamic link library
   *********************************************/
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing shared library\n");
#endif
  G__free_shl_upto(dictpos->allsl);
#endif

  /********************************************
   * free preprocessfilekey list
   *********************************************/
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing preprocess file key list\n");
#endif
  G__free_preprocessfilekey(dictpos->preprocessfilekey);

#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing function macro list\n");
#endif
  G__freedeffuncmacro(dictpos->deffuncmacro);

#ifdef G__TEMPLATECLASS
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing template class list\n");
#endif
  G__freedeftemplateclass(dictpos->definedtemplateclass);
#endif

#ifdef G__TEMPLATEFUNC
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing template function list\n");
#endif
  G__freetemplatefunc(dictpos->definedtemplatefunc);
#endif

#ifndef G__OLDIMPLEMENTATION2190
 {
   int nfile = G__nfile;
   while(nfile > dictpos->nfile) {
     struct G__dictposition *dictposx = G__srcfile[nfile].dictpos;
     if(dictposx && dictposx->ptype && (char*)G__PVOID!=dictposx->ptype ) {
       free((void*)dictposx->ptype);
       dictposx->ptype = (char*)NULL;
     }
     --nfile;
   }
 }
#endif
  if(dictpos->ptype && (char*)G__PVOID!=dictpos->ptype ) {
    int i;
    for(i=0; i<G__struct.alltag; i++) G__struct.type[i] = dictpos->ptype[i];
    free((void*)dictpos->ptype);
    dictpos->ptype = (char*)NULL;
  }

  /* close source files */
#ifdef G__MEMTEST
  fprintf(G__memhist,"Closing input files\n");
#endif
  G__close_inputfiles_upto(dictpos);

  G__tagdefining = -1;

  G__UnlockCriticalSection();
}


/***********************************************************************
* int G__free_ifunc_table_upto_ifunc()
*
***********************************************************************/
static
int G__free_ifunc_table_upto_ifunc(G__ifunc_table_internal *ifunc,G__ifunc_table_internal *dictpos,int ifn)
{
  int i,j;

  /* Freeing default parameter storage */
  if(ifunc==dictpos && ifn==ifunc->allifunc) {return(1);}
  for(i=ifunc->allifunc-1;i>=0;i--) {
#ifdef G__MEMTEST
    fprintf(G__memhist,"func %s\n",ifunc->funcname[i]);
#endif
    if(ifunc->funcname[i]) {
      free((void*)ifunc->funcname[i]);
      ifunc->funcname[i] = (char*)NULL;
    }
#ifdef G__ASM_WHOLEFUNC
    if(
       ifunc->pentry[i] && 
       ifunc->pentry[i]->bytecode) {
      G__free_bytecode(ifunc->pentry[i]->bytecode);
      ifunc->pentry[i]->bytecode = (struct G__bytecodefunc*)NULL;
    }
#endif
#ifdef G__FRIEND
    G__free_friendtag(ifunc->friendtag[i]);
#endif
    for(j=ifunc->para_nu[i]-1;j>=0;j--) {
      if(ifunc->param[i][j]->name) {
        free((void*)ifunc->param[i][j]->name);
        ifunc->param[i][j]->name=(char*)NULL;
      }
      if(ifunc->param[i][j]->def) {
        free((void*)ifunc->param[i][j]->def);
        ifunc->param[i][j]->def=(char*)NULL;
      }
      if(ifunc->param[i][j]->pdefault &&
         (&G__default_parameter)!=ifunc->param[i][j]->pdefault &&
         (G__value*)(-1)!=ifunc->param[i][j]->pdefault) {
        free((void*)ifunc->param[i][j]->pdefault);
        ifunc->param[i][j]->pdefault=(G__value*)NULL;
      }
    }

    if(ifunc==dictpos && ifn==i) {
      ifunc->allifunc=ifn;
      return(1);
    }
  }
  ifunc->page=0;
  return(0);
  /* Do not free 'ifunc' because it can be a global/static object */
}

/***********************************************************************
* int G__free_ifunc_table_upto()
*
***********************************************************************/
int G__free_ifunc_table_upto(G__ifunc_table_internal *ifunc,G__ifunc_table_internal *dictpos,int ifn)
{
  while (ifunc && ifunc != dictpos)
     ifunc = ifunc->next;
  if (ifunc != dictpos) {
     G__fprinterr(G__serr,"G__free_ifunc_table_upto: dictpos not found in ifunc list!\n");
     return 1;
  }

  G__ifunc_table_internal* next = ifunc->next;
  int ret = G__free_ifunc_table_upto_ifunc(ifunc, dictpos, ifn);
  ifunc->next=(struct G__ifunc_table_internal *)NULL;

  while (next) {
     ifunc = next;
     next = ifunc->next;
     ret += G__free_ifunc_table_upto_ifunc(ifunc, dictpos, ifn);
     free((void*)ifunc);
  }

  return ret;
}

/***********************************************************************
* int G__free_string_upto()
*
* Can replace G__free_string();
*
***********************************************************************/
int G__free_string_upto(G__ConstStringList *conststringpos)
{
  struct G__ConstStringList *pconststring;
  pconststring = G__plastconststring;
  while(pconststring && pconststring != conststringpos) {
    G__plastconststring = pconststring;
    pconststring = pconststring->prev;
    free((void*)G__plastconststring->string);
    free((void*)G__plastconststring);
  }
  G__plastconststring = conststringpos;
  return(0);
}

/***********************************************************************
* int G__free_typedef_upto()
*
* Can replace G__free_typedef();
*
***********************************************************************/
int G__free_typedef_upto(int typenum)
{
  while((--G__newtype.alltype)>=typenum) {
    free((void*)G__newtype.name[G__newtype.alltype]);
    G__newtype.name[G__newtype.alltype]=(char *)NULL;
    if(G__newtype.nindex[G__newtype.alltype]) {
      free((void*)G__newtype.index[G__newtype.alltype]);
      G__newtype.nindex[G__newtype.alltype]=0;
    }
  }
  G__newtype.alltype=typenum;
  return(0);
}

/***********************************************************************
* int G__free_struct_upto()
*
* Can replace G__free_struct();
*
***********************************************************************/
int G__free_struct_upto(int tagnum)
{
  struct G__var_array *var;
  int i,j,done;
  char com[G__ONELINE];
  long store_struct_offset;
  int store_tagnum;
  int ialltag;

  /*****************************************************************
  * clearing static members
  *****************************************************************/
  for(ialltag=G__struct.alltag-1;ialltag>=tagnum;ialltag--) {
    if(G__struct.libname[ialltag]) {
      free((void*)G__struct.libname[ialltag]);
      G__struct.libname[ialltag] = (char*)NULL;
    }
    if(G__NOLINK==G__struct.iscpplink[ialltag]) {
      /* freeing static member variable if not precompiled */
      var=G__struct.memvar[ialltag];
      while(var) {
        for(i=0;i<var->allvar;i++) {
          if (
            (var->statictype[i] == G__LOCALSTATIC) && 
            (var->globalcomp[i] != G__COMPILEDGLOBAL) &&
            (var->reftype[i] == G__PARANORMAL)
          ) {
            if (var->type[i] == 'u') {
              // -- Static class object member try destructor.
              sprintf(com, "~%s()", G__struct.name[var->p_tagtable[i]]);
              store_struct_offset = G__store_struct_offset;
              store_tagnum = G__tagnum;
              G__store_struct_offset = var->p[i];
              G__tagnum = var->p_tagtable[i];
              if (G__dispsource) {
                G__fprinterr(G__serr, "!!!Destroy static member object 0x%lx %s::~%s()\n", var->p[i], G__struct.name[ialltag], G__struct.name[i]);
              }
              done = 0;
              j = var->varlabel[i][1] /* num of elements */;
              if (!j) {
                j = 1;
              }
              --j;
              for (; j >= 0; --j) {
                G__getfunction(com, &done, G__TRYDESTRUCTOR);
                if (!done) {
                  break;
                }
                G__store_struct_offset += G__struct.size[i];
              }
              G__store_struct_offset = store_struct_offset;
              G__tagnum=store_tagnum;
            }
            if(G__CPPLINK!=G__struct.iscpplink[var->p_tagtable[i]])
              free((void*)var->p[i]);
          }
          if(var->varnamebuf[i]) {
            free((void*)var->varnamebuf[i]);
            var->varnamebuf[i] = (char*)NULL;
          }
        }
        var=var->next;
      }
    }
    else {
      var=G__struct.memvar[ialltag];
      while(var) {
        for(i=0;i<var->allvar;i++) {
          /* need to free compiled enum value */
          if(G__LOCALSTATIC==var->statictype[i] && 
             G__COMPILEDGLOBAL!=var->globalcomp[i] &&
             -1!=var->p_tagtable[i]&&'e'==G__struct.type[var->p_tagtable[i]]) {
            free((void*)var->p[i]);
          }
          if(var->varnamebuf[i]) {
            free((void*)var->varnamebuf[i]);
            var->varnamebuf[i] = (char*)NULL;
          }
        }
        var=var->next;
      }
    }
  }

  /*****************************************************************
  * clearing class definition
  *****************************************************************/
  while((--G__struct.alltag)>=tagnum) {
#ifdef G__MEMTEST
    fprintf(G__memhist,"struct %s\n",G__struct.name[G__struct.alltag]);
#endif

    G__reset_ifunc_refs_for_tagnum(G__struct.alltag);

    G__bc_delete_vtbl(G__struct.alltag);    
    if(G__struct.rootspecial[G__struct.alltag]) {
      free((void*)G__struct.rootspecial[G__struct.alltag]);
    }
#ifdef G__FRIEND
    G__free_friendtag(G__struct.friendtag[G__struct.alltag]);
#endif
    /* freeing class inheritance table */
    free((void*)G__struct.baseclass[G__struct.alltag]);
    G__struct.baseclass[G__struct.alltag] = (struct G__inheritance *)NULL;

    /* freeing member function table */
    G__free_ifunc_table(G__struct.memfunc[G__struct.alltag]);
    free((void*)G__struct.memfunc[G__struct.alltag]);
    G__struct.memfunc[G__struct.alltag]=(struct G__ifunc_table_internal *)NULL;


    /* freeing member variable table */
    G__free_member_table(G__struct.memvar[G__struct.alltag]);
    free((void*)G__struct.memvar[G__struct.alltag]);
    G__struct.memvar[G__struct.alltag]=(struct G__var_array *)NULL;

    /* freeing tagname */
    free((void*)G__struct.name[G__struct.alltag]);
    G__struct.name[G__struct.alltag]=(char *)NULL;
  }
  G__struct.alltag=tagnum;
  return(0);
}

/***********************************************************************
* void G__scratch_globals_upto(dictpos)
*
***********************************************************************/
void G__scratch_globals_upto(G__dictposition *dictpos)
{
  /*******************************************
   * clear interpreted global variables
   *******************************************/
  G__LockCriticalSection();
#ifdef G__MEMTEST
  fprintf(G__memhist,"Freeing global variables\n");
#endif
  G__destroy_upto(dictpos->var,1,dictpos->var,dictpos->ig15);
  G__UnlockCriticalSection();
}

/***********************************************************************
* void G__destroy_upto_vararray()
*
***********************************************************************/
static int G__destroy_upto_vararray(G__var_array *var,int global
                    ,G__var_array* /*dictpos*/,int ig15)
{
  int itemp=0,itemp1=0;
  
  int store_tagnum;
  long store_struct_offset; /* used to be int */
  int store_return;
  int store_prerun;

  int remain=ig15;
  if (remain < 0) 
     // called via G__destroy
     remain = 0;

  int cpplink;
  int i,size;
  long address;
  
  for (itemp = var->allvar - 1; itemp >= remain; --itemp) {
    /*****************************************
     * If the variable is a function scope
     * static variable, it will be kept.
     * else (auto and static body) free
     * allocated memory area.
     *****************************************/
    if (
#ifdef G__ASM_WHOLEFUNC
      (
        (ig15 < 0) &&
        (
          ((var->statictype[itemp] != G__LOCALSTATIC) || (global == G__GLOBAL_VAR)) && 
          (var->statictype[itemp] != G__COMPILEDGLOBAL) &&
          (global != G__BYTECODELOCAL_VAR)
        )
      ) 
      || (ig15 >= 0) &&
#endif
      ((var->statictype[itemp] != G__LOCALSTATIC) || global) &&
      (var->statictype[itemp] != G__COMPILEDGLOBAL)
    ) {
      cpplink = 0;
      /****************************************************
       * If C++ class, destructor has to be called
       ****************************************************/
      if (var->type[itemp] == 'u' && !G__ansiheader && !G__prerun) {
        char vv[G__BUFLEN];
        char *temp=vv;
        store_struct_offset = G__store_struct_offset;
        G__store_struct_offset = var->p[itemp]; /* duplication */
        store_tagnum = G__tagnum;
        G__tagnum = var->p_tagtable[itemp];
        store_return=G__return;
        G__return=G__RETURN_NON;
        if (strlen(G__struct.name[G__tagnum])>G__BUFLEN-5) {
          temp = (char*)malloc(strlen(G__struct.name[G__tagnum])+5);
        }
        sprintf(temp,"~%s()",G__struct.name[G__tagnum]);
        if (G__dispsource) {
          G__fprinterr(G__serr, "\n!!!Calling destructor 0x%lx.%s for %s ary%d:link%d", G__store_struct_offset, temp, var->varnamebuf[itemp], var->varlabel[itemp][1] /* num of elements */, G__struct.iscpplink[G__tagnum]);
        }
        store_prerun = G__prerun;
        G__prerun = 0;
        /********************************************************
         * destruction of array 
         ********************************************************/
        if(G__CPPLINK==G__struct.iscpplink[G__tagnum]) {
          if(G__AUTOARYDISCRETEOBJ==var->statictype[itemp]) {
            long store_globalvarpointer = G__globalvarpointer;
            size=G__struct.size[G__tagnum];
            i = var->varlabel[itemp][1] /* num of elements */;
            if (!i) {
              i = 1;
            }
            --i;
            for (; i >= 0; --i) {
              G__store_struct_offset = var->p[itemp] + (i * size);
              G__globalvarpointer = G__store_struct_offset;
              int known = 0;
              G__getfunction(temp, &known, G__TRYDESTRUCTOR); 
              if (!known) {
                break;
              }
            }
            G__globalvarpointer = store_globalvarpointer;
            free((void*)var->p[itemp]);
          }
          else {
            G__store_struct_offset = var->p[itemp];
            i = var->varlabel[itemp][1] /* num of elements */;
            if (i || var->paran[itemp])  {
              G__cpp_aryconstruct = i;
            }
            int known = 0;
            G__getfunction(temp, &known, G__TRYDESTRUCTOR); 
            G__cpp_aryconstruct = 0;
          }
          cpplink = 1;
        }
        else {
          size=G__struct.size[G__tagnum];
          i = var->varlabel[itemp][1] /* num of elements */;
          if (!i) {
            i = 1;
          }
          --i;
          for (; i >= 0; --i) {
            G__store_struct_offset = var->p[itemp] + (i * size);
            if (G__dispsource) {
              G__fprinterr(G__serr, "\n0x%lx.%s", G__store_struct_offset, temp);
            }
            int known = 0;
            G__getfunction(temp, &known, G__TRYDESTRUCTOR); 
            if (!known) {
              break;
            }
          }
        }
        if(vv!=temp) free((void*)temp);
        G__prerun = store_prerun;
        G__store_struct_offset = store_struct_offset;
        G__tagnum = store_tagnum;
        G__return=store_return;
      } /*  end of type=='u' */
      else if (
        (G__security & G__SECURE_GARBAGECOLLECTION) && 
        !G__no_exec_compile &&
        isupper(var->type[itemp]) &&
        var->p[itemp]
      ) {
        i = var->varlabel[itemp][1];
        if (!i) {
          i = 1;
        }
        --i;
        for (; i >= 0; --i) {
          address = var->p[itemp] + (i * G__LONGALLOC);
          if (*((long*) address)) {
            G__del_refcount((void*) (*((long*) address)), (void**) address);
          }
        }
      }

#ifdef G__MEMTEST
      fprintf(G__memhist,"Free(%s)\n",var->varnamebuf[itemp]);
#endif
      if(G__NOLINK==cpplink && var->p[itemp]) free((void*)var->p[itemp]);
      
    } /* end of statictype==LOCALSTATIC or COMPILEDGLOBAL */
    
#ifdef G__DEBUG
    else if(G__memhist) {
      fprintf(G__memhist
              ,"0x%lx (%s) not freed localstatic or compiledglobal FILE:%s LINE:%d\n"
              ,var->p[itemp],var->varnamebuf[itemp]
              ,G__ifile.name,G__ifile.line_number);
    } 
#endif
    
    // Initialize varpointer and varlabel.
    for (itemp1 = 0; itemp1 < G__MAXVARDIM; ++itemp1) {
      var->varlabel[itemp][itemp1] = 0;
    }
    if(var->varnamebuf[itemp]) {
      free((void*)var->varnamebuf[itemp]);
      var->varnamebuf[itemp] = (char*)NULL;
    }

  }
  
  var->allvar = remain;
  return(0);
  
}


/***********************************************************************
* void G__destroy_upto()
*
***********************************************************************/
/* destroy local variable and free memory*/
int G__destroy_upto(G__var_array *var,int global
                    ,G__var_array *dictpos,int ig15)
{

   if (!var) return 0;
   G__var_array *tail = var;
   G__var_array *prev = 0;

#ifndef G__OLDIMPLEMENTATION2038
   if (ig15 == -1) {
     // called via G__destroy
     /* This part is not needed in G__destroy_upto.  enclosing_scope and
      * inner_scope members are assigned only as local variable table for
      * bytecode function  in which case  G__destroy() is always used to
      * deallocate the table */
     var->enclosing_scope = (struct G__var_array*)NULL;
     if(var->inner_scope) {
       int i=0;
       while(var->inner_scope[i]) {
         G__destroy(var->inner_scope[i],global);
         free((void*)var->inner_scope[i]);
         ++i;
       }
     }
   }
#endif
  
  /*******************************************
   * If there are any sub var array list,
   * destroy it too.
   *******************************************/
   while (tail->next) {
      if(tail->allvar!=G__MEMDEPTH) {
        fprintf(stderr,"!!!Fatal Error: Interpreter memory overwritten by illegal access.!!!\n");
        fprintf(stderr,"!!!Terminate session!!!\n");
      }
      // make tail->next point to prev instead
      G__var_array *next = tail->next;
      tail->next = prev;
      prev = tail;
      tail = next;
   }
   tail->next = prev;

   int ret = 0;
   do {
      int remain = 0;
      if (!tail->next) remain = ig15;
      if (ig15 < 0) remain = ig15; // always pass "called by G__destroy" flag
      ret += G__destroy_upto_vararray(tail, global, dictpos, remain);
      G__var_array *next = tail->next;
      if (next) free(tail);
      else tail->next = 0;
      tail = next;
   } while (tail);

   return ret;
}


/***********************************************************************
* G__close_inputfiles_upto()
*
* Can not replace G__close_inputfiles()
*
***********************************************************************/
void G__close_inputfiles_upto(G__dictposition* pos)
{
#ifdef G__SHAREDLIB
  struct G__filetable permanentsl[G__MAX_SL];
  int nperm=0;
#endif

#ifdef G__DUMPFILE
  if(G__dumpfile) G__dump_tracecoverage(G__dumpfile);
#endif
  int nfile = pos->nfile;
  while(G__nfile>nfile) {
    --G__nfile;
    if(G__srcfile[G__nfile].dictpos) {
      free((void*)G__srcfile[G__nfile].dictpos);
      G__srcfile[G__nfile].dictpos=(struct G__dictposition*)NULL;
    }
    if(G__srcfile[G__nfile].hasonlyfunc) {
      free((void*)G__srcfile[G__nfile].hasonlyfunc);
      G__srcfile[G__nfile].hasonlyfunc=(struct G__dictposition*)NULL;
    }
#ifdef G__SHAREDLIB
    if(G__srcfile[G__nfile].ispermanentsl) {
      permanentsl[nperm++] = G__srcfile[G__nfile];
      G__srcfile[G__nfile].initsl=0;

      // reset autoload struct entries
      for (int itag = 0; itag < pos->tagnum; ++itag)
         if (G__struct.filenum[itag] == G__nfile) {

            // keep name, libname; reset everything else
            char* name = G__struct.name[itag];
            int hash = G__struct.hash[itag];
            char* libname = G__struct.libname[itag];
            G__struct.name[itag] = 0; // tree_struct must not delete it
            G__struct.libname[itag] = 0; // tree_struct must not delete it
            int alltag = G__struct.alltag;
            G__struct.alltag = itag + 1; // to only free itag
            G__free_struct_upto(itag);
            G__struct.alltag = alltag;
            G__struct.name[itag] = name;
            G__struct.libname[itag] = libname;
            G__struct.type[itag] = 'a';
            G__struct.hash[itag] = hash;
            G__struct.size[itag] = 0;

            G__struct.memvar[itag] = (struct G__var_array *)malloc(sizeof(struct G__var_array));
            memset(G__struct.memvar[itag],0,sizeof(struct G__var_array));
            G__struct.memvar[itag]->tagnum = itag;
            G__struct.memfunc[itag] = (struct G__ifunc_table_internal *)malloc(sizeof(struct G__ifunc_table_internal));
            memset(G__struct.memfunc[itag],0,sizeof(struct G__ifunc_table_internal));
            G__struct.memfunc[itag]->tagnum = itag;
            G__struct.memfunc[itag]->funcname[0]=(char*)malloc(2);
            G__struct.memfunc[itag]->funcname[0][0]=0;
            G__struct.memfunc[itag]->pentry[0] = &G__struct.memfunc[itag]->entry[0];
            G__struct.memfunc[itag]->pentry[0]->bytecodestatus = G__BYTECODE_NOTYET;
            G__struct.memfunc[itag]->access[0] = G__PUBLIC;
            G__struct.memfunc[itag]->ansi[0] = 1;
            G__struct.memfunc[itag]->p_tagtable[0] = -1;
            G__struct.memfunc[itag]->p_typetable[0] = -1;
            G__struct.memfunc[itag]->comment[0].filenum = -1;
            {
               struct G__ifunc_table_internal *store_ifunc;
               store_ifunc = G__p_ifunc;
               G__p_ifunc = G__struct.memfunc[itag];
               G__memfunc_next();
               G__p_ifunc = store_ifunc;
            }

            G__struct.baseclass[itag] = (struct G__inheritance *)malloc(sizeof(struct G__inheritance));
            memset(G__struct.baseclass[itag],0,sizeof(struct G__inheritance));

            G__struct.virtual_offset[itag] = -1;
            G__struct.globalcomp[itag] = 0;
            G__struct.iscpplink[itag] = G__default_link?G__globalcomp:G__NOLINK;
            G__struct.isabstract[itag] = 0;
            G__struct.protectedaccess[itag] = 0; 
            G__struct.line_number[itag] = -1;
            G__struct.filenum[itag] = -1;
            G__struct.parent_tagnum[itag] = -1;
            G__struct.funcs[itag] = 0;
            G__struct.istypedefed[itag] = 0;
            G__struct.istrace[itag] = 0;
            G__struct.isbreak[itag] = 0;
            G__struct.comment[itag].p.com = NULL;
            G__struct.comment[itag].filenum = -1;
            G__struct.friendtag[itag] = 0;
            G__struct.incsetup_memvar[itag] = 0;
            G__struct.incsetup_memfunc[itag] = 0;
            G__struct.rootflag[itag] = 0;
            G__struct.rootspecial[itag] = 0;
            G__struct.isctor[itag] = 0;
            G__struct.defaulttypenum[itag] = 0;
            G__struct.vtable[itag] = 0;
        }
      continue;
    }
#endif
    if(G__srcfile[G__nfile].initsl) {
      delete(G__srcfile[G__nfile].initsl);
      G__srcfile[G__nfile].initsl=0;
    }

    if(G__srcfile[G__nfile].fp) { 
      fclose(G__srcfile[G__nfile].fp);
      if(G__srcfile[G__nfile].prepname) {
        int j;
        for(j=G__nfile-1;j>=0;j--) {
          if(G__srcfile[j].fp==G__srcfile[G__nfile].fp) 
            G__srcfile[j].fp=(FILE*)NULL;
        }
      }
      G__srcfile[G__nfile].fp=(FILE*)NULL;
    }
    if(G__srcfile[G__nfile].breakpoint) {
      free((void*)G__srcfile[G__nfile].breakpoint);
      G__srcfile[G__nfile].breakpoint=(char*)NULL;
      G__srcfile[G__nfile].maxline=0;
    }
    if(G__srcfile[G__nfile].prepname) {
      if('('!=G__srcfile[G__nfile].prepname[0])
        remove(G__srcfile[G__nfile].prepname);
      free((void*)G__srcfile[G__nfile].prepname);
      G__srcfile[G__nfile].prepname=(char*)NULL;
    }
    if(G__srcfile[G__nfile].filename) {
#ifndef G__OLDIMPLEMENTATION1546
      unsigned int len = strlen(G__srcfile[G__nfile].filename);
      if(len>strlen(G__NAMEDMACROEXT2) && 
         strcmp(G__srcfile[G__nfile].filename+len-strlen(G__NAMEDMACROEXT2),
                G__NAMEDMACROEXT2)==0) {
        remove(G__srcfile[G__nfile].filename);
      }
#endif
      free((void*)G__srcfile[G__nfile].filename);
      G__srcfile[G__nfile].filename=(char*)NULL;
    }
    G__srcfile[G__nfile].hash=0;
  }
  G__nfile=nfile;

#ifdef G__SHAREDLIB
  while(nperm) {
    --nperm;
    G__srcfile[G__nfile++] = permanentsl[nperm];

    if (permanentsl[nperm].initsl) {
       G__input_file store_ifile = G__ifile;
       G__ifile.filenum = G__nfile - 1;
       G__ifile.line_number = -1;
       G__ifile.str = 0;
       G__ifile.pos = 0;
       G__ifile.vindex = 0;
       G__ifile.fp = G__srcfile[G__nfile - 1].fp;
       strcpy(G__ifile.name,G__srcfile[G__nfile - 1].filename);

       for (std::list<G__DLLINIT>::const_iterator iInitsl = permanentsl[nperm].initsl->begin();
            iInitsl != permanentsl[nperm].initsl->end(); ++iInitsl)
          (*(*iInitsl))();

       G__ifile = store_ifile;
    }
  }
#endif

  if(G__tempc[0]) {
    remove(G__tempc);
    G__tempc[0]='\0';
  }

  /*****************************************************************
  * Closing modified STDIOs.  May need to modify here.
  *  init.c, end.c, scrupto.c
  *****************************************************************/
  if(G__serr!=G__stderr && G__serr) {
    fclose(G__serr);
    G__serr=G__stderr;
  }
  if(G__sout!=G__stdout && G__sout) {
    fclose(G__sout);
    G__sout=G__stdout;
  }
  if(G__sin!=G__stdin && G__sin) {
    fclose(G__sin);
    G__sin=G__stdin;
  }
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
