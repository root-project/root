/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file var.c
 ************************************************************************
 * Description:
 *  Variable initialization, assignment and referencing
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

char G__declctor[G__LONGLINE];
#ifndef G__OLDIMPLEMENTATION1103
extern int G__const_noerror;
#endif

#ifndef G__OLDIMPLEMENTATION1119
static int G__getarraydim=0;
extern int G__dynconst;
#endif

#ifndef G__OLDIMPLEMENTATION1151
extern int G__initval_eval;
#endif

/**************************************************************************
* G__ASSIGN_VAR()
*
*  Variable allocation
*
*  MUST CORRESPOND TO G__letstruct
**************************************************************************/

#define G__ASSIGN_VAR(SIZE,CASTTYPE,CONVFUNC,X)                           \
switch(G__var_type) {                                                     \
case 'v': /* int var[10]; *var=expr; assign to value */                   \
        if(var->paran[ig15]==paran+1&&p_inc==0  /* 1070 */                \
           && islower(result.type)) { /* 1650 */                          \
          ++paran;                                                        \
	}                                                                 \
	else {                                                            \
	  G__assign_error(item,&result);                                  \
	  break;                                                          \
	}                                         /*end 1070*/            \
case 'p': /* var = expr; assign to value */                               \
	if(var->paran[ig15]<=paran) { /*assign to type element*/          \
    /*1068*/    result.ref=G__struct_offset+var->p[ig15]+p_inc*SIZE;      \
		*(CASTTYPE *)(G__struct_offset+var->p[ig15]+p_inc*SIZE)   \
			= (CASTTYPE)CONVFUNC(result);                     \
                result.obj.reftype.reftype = G__PARANORMAL; /*1669???*/   \
                X = *(CASTTYPE*)result.ref; /*1669*/                      \
                result.type = var->type[ig15]; /*1669*/                   \
		break;                                                    \
	}                                                                 \
	else if(G__funcheader && paran==0 && isupper(result.type)) {      \
		/* K&R style 'type a[]' initialization */                 \
	    /* if(var->p[ig15]!=G__PINVALID) free((void*)var->p[ig15]);*/ \
		if(var->p[ig15]!=G__PINVALID &&   /* ON457 */             \
		   G__COMPILEDGLOBAL!=var->statictype[ig15])              \
                  free((void*)var->p[ig15]);                              \
		var->p[ig15] = result.obj.i;                              \
		var->statictype[ig15]=G__COMPILEDGLOBAL;                  \
		break;                                                    \
	}                                                                 \
default :                                                                 \
	G__assign_error(item,&result);                                    \
	break;                                                            \
}                                                                         \
G__var_type = 'p';                                                        \
if(vv!=varname)free((void*)varname); /* 1802 */ \
return(result);                                                           

/**************************************************************************
* G__ASSIGN_PVAR()
*
*  Pointer Variable allocation
*
*  MUST CORRESPOND TO G__letstructp
**************************************************************************/


#define G__ASSIGN_PVAR(CASTTYPE,CONVFUNC,X)                                  \
switch(G__var_type) {                                                        \
case 'v': /* *var = expr ;  assign to contents of pointer */                 \
  switch(var->reftype[ig15]) {                                               \
  case G__PARANORMAL:                                                        \
    if(INT_MAX!=var->varlabel[ig15][1]) {   /* below line 1068 */            \
    result.ref=(*(long*)(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC)); \
      *(CASTTYPE *)result.ref = (CASTTYPE)CONVFUNC(result);                  \
      result.type=tolower(var->type[ig15]); /*1669*/                         \
      X = *(CASTTYPE*)result.ref; /*1669*/                                   \
    }                                                                        \
    else {                                                                   \
      result.ref=G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC;/*1068*/   \
      *(long *)(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC)            \
			= G__int(result);                                    \
      result.type=var->type[ig15]; /*1669*/ \
    }                                                                        \
    break;                                                                   \
  case G__PARAP2P:                                                           \
    if(var->paran[ig15]<paran) {                                             \
      address = G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC;            \
      result.ref=(*(((long*)(*(long *)address))+pp_inc));  /*1068*/          \
      *(CASTTYPE*)result.ref = (CASTTYPE)CONVFUNC(result);                   \
      result.type=tolower(var->type[ig15]); /*1669*/                         \
      X = *(CASTTYPE*)result.ref; /*1669*/                                   \
    }                                                                        \
    else { /* below line 1068*/                                              \
   result.ref=(*(long *)(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC)); \
      *(long *)(*(long *)(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC)) \
	= G__int(result);                                                    \
    }                                                                        \
    break;                                                                   \
  }                                                                          \
  break;                                                                     \
case 'p': /* var = expr; assign to pointer variable  */                      \
  if(var->paran[ig15]<=paran) {                                              \
    if(var->paran[ig15]<paran) {                                             \
      address = G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC;            \
      if(G__PARANORMAL==var->reftype[ig15]) {                                \
        /* result.ref=(((long)(*(long *)address))+pp_inc);1757*/ /*1068*/    \
        result.ref= (long) (((CASTTYPE *)(*(long *)address))+pp_inc);/*1757*/\
        *(CASTTYPE*)result.ref = (CASTTYPE)CONVFUNC(result);                 \
        result.type=tolower(var->type[ig15]); /*1669*/                       \
        X = *(CASTTYPE*)result.ref; /*1669*/                                 \
      }                                                                      \
      else if(var->paran[ig15]==paran-1) {                                   \
        result.ref=(long)(((long*)(*(long *)address))+pp_inc);/*1068*/       \
        *(long*)result.ref = G__int(result);                                 \
      }                                                                      \
      else {                                                                 \
        /* ron eastman change begins */                                      \
/**((CASTTYPE*)(*(((long*)(*(long *)address))+para[0].obj.i))+para[1].obj.i)*/\
		/* = CONVFUNC(result); */                                    \
	int ip;                                                              \
 	long *phyaddress=(long*)(*(long *)address);                          \
 	for (ip = 0; ip < paran-1; ip++) {				     \
          phyaddress=(long*)phyaddress[para[ip].obj.i];                      \
 	}                                                                    \
        /*1068 Dont know how to implement*/                                  \
        switch(var->reftype[ig15]-paran+var->paran[ig15]) { /*1540*/\
	case G__PARANORMAL: /*1540*/                                         \
          ((CASTTYPE*)(phyaddress))[para[paran-1].obj.i] = CONVFUNC(result); \
          break; /*1540*/                                                    \
        default: /*1540*/                                                    \
          ((long*)(phyaddress))[para[paran-1].obj.i]=G__int(result);/*1540*/ \
          break; /*1540*/                                                    \
	} /*1540*/                                                           \
        /* ron eastman change ends */                                        \
      }                                                                      \
    }                                                                        \
    else {                                                                   \
      result.ref=(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC);/*1068*/ \
      *(long *)(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC)            \
			= G__int(result);                                    \
    }                                                                        \
  }                                                                          \
  else { /* K&R pointer to pointer 'type **a,type *a[]' initialization */    \
    if(var->p[ig15]!=G__PINVALID &&    /* ON457 */                           \
       G__COMPILEDGLOBAL!=var->statictype[ig15])                             \
      free((void*)var->p[ig15]);                                             \
    var->p[ig15] = result.obj.i;                                             \
    var->statictype[ig15]=G__COMPILEDGLOBAL;                                 \
  }                                                                          \
  break;                                                                     \
default :                                                                    \
  G__assign_error(item,&result);                                             \
  break;                                                                     \
}



/**************************************************************************
* G__ALLOC_VAR_REF()
*
*  Variable allocation
**************************************************************************/
#ifdef G__ASM_WHOLEFUNC

#define G__ALLOC_VAR_REF(SIZE,CASTTYPE,CONVFUNC)                 \
if(islower(G__var_type)) { /* type var; normal variable */       \
	var->p[ig15] = G__malloc(p_inc,SIZE,item);               \
	if(((G__def_struct_member==0&&G__ASM_FUNC_NOP==G__asm_wholefunction)\
 /*1454*/   ||G__static_alloc||var->statictype[ig15]==G__LOCALSTATIC)&&     \
	   ((!G__static_alloc)||(G__prerun))&&                   \
	   (G__globalvarpointer==G__PVOID||                      \
	    result.type!='\0'))                                  \
			*((CASTTYPE *)var->p[ig15])              \
				= (CASTTYPE)CONVFUNC(result);    \
}                                                                \
else { /* type *var; pointer */                                  \
    if(p_inc>1 && result.type!='\0') { /* char *argv[]; */       \
	var->p[ig15] = G__int(result);                           \
    }                                                            \
    else { /* pointer */                                         \
	var->p[ig15] = G__malloc(p_inc,G__LONGALLOC,item);       \
	if(((G__def_struct_member==0&&G__ASM_FUNC_NOP==G__asm_wholefunction)\
 /*1454*/  ||G__static_alloc||var->statictype[ig15]==G__LOCALSTATIC)&&      \
	   ((!G__static_alloc)||(G__prerun))&&                   \
	   (G__globalvarpointer==G__PVOID||                      \
	    result.type!='\0'))                                  \
		    *((long *)var->p[ig15])=G__int(result);      \
    }                                                            \
}

#else

#define G__ALLOC_VAR_REF(SIZE,CASTTYPE,CONVFUNC)                 \
if(islower(G__var_type)) { /* type var; normal variable */       \
	var->p[ig15] = G__malloc(p_inc,SIZE,item);               \
	if((G__def_struct_member==0||G__static_alloc)&&          \
	   ((!G__static_alloc)||(G__prerun))&&                   \
	   (G__globalvarpointer==G__PVOID||                      \
	    result.type!='\0'))                                  \
			*((CASTTYPE *)var->p[ig15])              \
				= (CASTTYPE)CONVFUNC(result);    \
}                                                                \
else { /* type *var; pointer */                                  \
    if(p_inc>1 && result.type!='\0') { /* char *argv[]; */       \
	var->p[ig15] = G__int(result);                           \
    }                                                            \
    else { /* pointer */                                         \
	var->p[ig15] = G__malloc(p_inc,G__LONGALLOC,item);       \
	if((G__def_struct_member==0||G__static_alloc)&&          \
	   ((!G__static_alloc)||(G__prerun))&&                   \
	   (G__globalvarpointer==G__PVOID||                      \
	    result.type!='\0'))                                  \
		    *((long *)var->p[ig15])=G__int(result);      \
    }                                                            \
}

#endif



/**************************************************************************
* G__GET_VAR()
*
*  get variable 
**************************************************************************/
#ifndef G__OLDIMPLEMENTATION1619

#define G__GET_VAR(SIZE,CASTTYPE,CONVFUNC,TYPE,PTYPE)                        \
switch(G__var_type) {                                                        \
case 'p': /* return value */                                                 \
  if(var->paran[ig15]<=paran) {                                              \
 /* if(var->varlabel[ig15][paran+1]==0) { */                                 \
    /* value , an integer */                                                 \
    result.ref = (G__struct_offset+var->p[ig15]+p_inc*SIZE);                 \
    CONVFUNC(&result,TYPE,(CASTTYPE)(*(CASTTYPE *)(result.ref)));            \
  }                                                                          \
  else { /* array , pointer */                                               \
    G__letint(&result,PTYPE,(G__struct_offset+var->p[ig15]+p_inc*SIZE));     \
    if(var->paran[ig15]-paran>1)                            /*993*/          \
      result.obj.reftype.reftype=var->paran[ig15]-paran;    /*993*/          \
  }                                                                          \
  break;                                                                     \
case 'P': /* return pointer */                                               \
  G__letint(&result,PTYPE,(G__struct_offset+var->p[ig15]+p_inc*SIZE));       \
  break;                                                                     \
/* case 'v': */                                                              \
default :                                                                    \
  if(var->paran[ig15]<=paran)  /* 1619 */ \
    G__reference_error(item);                                                \
  else { /* 1619 */ \
    if(var->paran[ig15]-paran==1) { /*1619*/ \
      result.ref = (G__struct_offset+var->p[ig15]+p_inc*SIZE); /*1619*/ \
      CONVFUNC(&result,TYPE,(CASTTYPE)(*(CASTTYPE *)(result.ref))); /*1619*/ \
    } /* 1619 */ \
    else { /* 1619 */ \
      G__letint(&result,PTYPE,(G__struct_offset+var->p[ig15]+p_inc*SIZE)); /*1619*/ \
      if(var->paran[ig15]-paran>2) /*1619*/ \
        result.obj.reftype.reftype=var->paran[ig15]-paran-1; /*1619*/ \
    } /* 1619 */ \
  } /* 1619 */ \
  break;                                                                     \
}                                                                            \
G__var_type='p';                                                             \
return(result);               

#else

#define G__GET_VAR(SIZE,CASTTYPE,CONVFUNC,TYPE,PTYPE)                        \
switch(G__var_type) {                                                        \
case 'p': /* return value */                                                 \
  if(var->paran[ig15]<=paran) {                                              \
 /* if(var->varlabel[ig15][paran+1]==0) { */                                 \
    /* value , an integer */                                                 \
    result.ref = (G__struct_offset+var->p[ig15]+p_inc*SIZE);                 \
    CONVFUNC(&result,TYPE,(CASTTYPE)(*(CASTTYPE *)(result.ref)));            \
  }                                                                          \
  else { /* array , pointer */                                               \
    G__letint(&result,PTYPE,(G__struct_offset+var->p[ig15]+p_inc*SIZE));     \
    if(var->paran[ig15]-paran>1)                            /*993*/          \
      result.obj.reftype.reftype=var->paran[ig15]-paran;    /*993*/          \
  }                                                                          \
  break;                                                                     \
case 'P': /* return pointer */                                               \
  G__letint(&result,PTYPE,(G__struct_offset+var->p[ig15]+p_inc*SIZE));       \
  break;                                                                     \
/* case 'v': */                                                              \
default :                                                                    \
  G__reference_error(item);                                                  \
  break;                                                                     \
}                                                                            \
G__var_type='p';                                                             \
return(result);               

#endif

/**************************************************************************
* G__GET_STRUCTVAR()
*
*  get variable 
**************************************************************************/
#define G__GET_STRUCTVAR                                                     \
switch(G__var_type) {                                                        \
      case 'p': /* return value */                                           \
	if(var->paran[ig15]<=paran) {                                        \
		/* value, but return pointer */                              \
		result.ref=(long)(G__struct_offset+(var->p[ig15])            \
			  +p_inc*G__struct.size[var->p_tagtable[ig15]]);     \
		G__letint(&result,'u',(result.ref));                         \
	}                                                                    \
	else { /* array , pointer */                                         \
		G__letint(&result,'U'                                        \
			  ,(long)(G__struct_offset+(var->p[ig15])            \
			  +p_inc*G__struct.size[var->p_tagtable[ig15]]));    \
                if(var->paran[ig15]-paran>1)                         /*993*/ \
                  result.obj.reftype.reftype=var->paran[ig15]-paran; /*993*/ \
	}                                                                    \
	break;                                                               \
      case 'P': /* return pointer */                                         \
	G__letint(&result,'U'                                                \
		  ,(long)(G__struct_offset+(var->p[ig15])                    \
		  +p_inc*G__struct.size[var->p_tagtable[ig15]]));            \
	break;                                                               \
	/* case 'v': */                                                      \
      default :                                                              \
        if('v'==G__var_type) {                                               \
          char refopr[G__MAXNAME];                                           \
          long store_struct_offsetX = G__store_struct_offset;                \
	  int store_tagnumX = G__tagnum;                                     \
	  int done=0;                                                        \
          int store_asm_exec=G__asm_exec;  /* ON979 */                       \
          int store_asm_noverflow=G__asm_noverflow;  /* ON979 */             \
          G__asm_exec = 0;                 /* ON979 */                       \
          G__asm_noverflow = 0;            /* ON979 */                       \
          G__store_struct_offset = (long)(G__struct_offset+(var->p[ig15])    \
			  +p_inc*G__struct.size[var->p_tagtable[ig15]]);     \
          G__tagnum = var->p_tagtable[ig15];                                 \
          strcpy(refopr,"operator*()");                                      \
          result=G__getfunction(refopr,&done,G__TRYMEMFUNC);                 \
          G__asm_exec=store_asm_exec;              /* ON979 */               \
          G__asm_noverflow=store_asm_noverflow;    /* ON979 */               \
          G__tagnum = store_tagnumX;                                         \
          G__store_struct_offset = store_struct_offsetX;                     \
          if(0==done) {                                                      \
            G__reference_error(item);                                        \
          }                                                                  \
        }                                                                    \
        else                                                                 \
	  G__reference_error(item);                                          \
	break;                                                               \
}

/**************************************************************************
* G__GET_PVAR()
*
*  get variable 
**************************************************************************/


#define G__GET_PVAR(CASTTYPE,CONVFUNC,CONVTYPE,TYPE,PTYPE)                    \
switch(G__var_type) {                                                         \
case 'v': /* *var; get value */                                               \
  switch(var->reftype[ig15]) {                                                \
  case G__PARANORMAL:                                                         \
   result.ref=(*(long *)(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC));  \
   if(result.ref)                                                             \
     CONVFUNC(&result,TYPE,(CONVTYPE)(*(CASTTYPE *)(result.ref)));            \
   break;                                                                     \
  case G__PARAP2P:                                                            \
    if(var->paran[ig15]<paran) {                                              \
      address = G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC;             \
      /*result.ref=(*(((long*)(*(long *)address))+pp_inc)); 1757*/            \
      result.ref=(*(((CASTTYPE*)(*(long *)address))+pp_inc));/*1757*/         \
      if(result.ref)                                                          \
        CONVFUNC(&result,TYPE,(CONVTYPE)(*(CASTTYPE *)(result.ref)));         \
    }                                                                         \
    else {                                                                    \
      result.ref=                                                             \
               (*(long *)(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC)); \
      G__letint(&result,PTYPE,                                                \
      *(long *)(*(long *)(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC)));\
    }                                                                         \
   break;                                                                     \
  }                                                                           \
  break;                                                                      \
case 'P': /* &var; get pointer to pointer */                                  \
   if(var->paran[ig15]==paran) { /* must be PPTYPE */                         \
	  G__letint(&result,PTYPE                                             \
	     ,(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC));            \
   }                                                                          \
   else if(var->paran[ig15]<paran) {                                          \
	address = G__struct_offset + var->p[ig15]+p_inc*G__LONGALLOC;         \
        if(G__PARANORMAL==var->reftype[ig15])                                 \
	  G__letint(&result,PTYPE                                             \
	      ,(long)((CASTTYPE*)(*(long *)(address))+pp_inc));               \
        else {                                                                \
	  G__letint(&result,PTYPE                                             \
	      ,(long)((long*)(*(long *)(address))+pp_inc));                   \
          result.obj.reftype.reftype=G__PARAP2P;                              \
	}                                                                     \
   }                                                                          \
   else {                                                                     \
        G__letint(&result,PTYPE                                               \
	  ,(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC));               \
   }                                                                          \
   break;                                                                     \
default : /* 'p' */                                                           \
   if(var->paran[ig15]==paran) {                                              \
        /* type *p[];  (p[x]); */                                             \
        result.ref=(long)(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC);  \
	G__letint(&result,PTYPE                                               \
	 ,*(long *)(result.ref));                                             \
   }                                                                          \
   else if(var->paran[ig15]<paran) {                                          \
        /* type *p[];  p[x][y]; */                                            \
	address = G__struct_offset + var->p[ig15]+p_inc*G__LONGALLOC;         \
        if(G__PARANORMAL==var->reftype[ig15]) {                               \
	  result.ref=(long)((CASTTYPE *)(*(long *)(address))+pp_inc);         \
	  CONVFUNC(&result,TYPE                                               \
	      ,(CONVTYPE)(*((CASTTYPE *)(result.ref))));                      \
	}                                                                     \
        else if(var->paran[ig15]==paran-1) {                                  \
          result.ref=(long)((long*)(*(long *)(address))+pp_inc);              \
	  G__letint(&result,PTYPE                                             \
	      ,(long)((CASTTYPE*)(*((long*)(result.ref)))));                  \
          /* ron eastman change begins */                                     \
          if(G__PARAP2P<var->reftype[ig15])                                   \
            result.obj.reftype.reftype=var->reftype[ig15]-1;                  \
          /* ron eastman change ends */                                       \
	}                                                                     \
        else  {                                                               \
          result.ref=(long)((long*)(*(long *)(address))+para[0].obj.i);       \
        /* ron eastman change begins */                                       \
        {                                                                     \
 	  int ip;							      \
 	  for (ip = 1; ip < paran-1; ip++) {				      \
            result.ref=(long)((long*)(*(long *)(result.ref))+para[ip].obj.i); \
 	  }                                                                   \
 	}						                      \
/*result.ref=(long)((CASTTYPE*)(*((long*)(result.ref)))+para[paran-1].obj.i);1540*/\
        result.obj.reftype.reftype=var->reftype[ig15]-paran+var->paran[ig15]; \
        switch(result.obj.reftype.reftype) {                                  \
	case G__PARANORMAL:                                                   \
   result.ref=(long)((CASTTYPE*)(*((long*)(result.ref)))+para[paran-1].obj.i);/*1540*/\
	  CONVFUNC(&result,TYPE,*((CASTTYPE*)(result.ref)));                  \
          break;                                                              \
        case 1:                                                               \
       result.ref=(long)((long*)(*((long*)(result.ref)))+para[paran-1].obj.i);/*1540*/\
	  G__letint(&result,PTYPE,*((long*)(result.ref)));                    \
          result.obj.reftype.reftype=G__PARANORMAL;                           \
          break;                                                              \
        default:                                                              \
       result.ref=(long)((long*)(*((long*)(result.ref)))+para[paran-1].obj.i);/*1540*/\
	  G__letint(&result,PTYPE,*((long*)(result.ref)));                    \
        result.obj.reftype.reftype=var->reftype[ig15]-paran+var->paran[ig15]; \
          break;                                                              \
	}                                                                     \
   /* result.ref=(long)((CASTTYPE*)(*((long*)(result.ref)))+para[1].obj.i);*/ \
	  /* CONVFUNC(&result,TYPE,*((CASTTYPE*)(result.ref))); */            \
        /* ron eastman change ends */                                         \
	}                                                                     \
   }                                                                          \
   else {                                                                     \
        /* type *p[];  p; */                                                  \
        result.ref = (long)(&var->p[ig15]);                                   \
        G__letint(&result,PTYPE                                               \
	  ,(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC));               \
   }                                                                          \
   break;                                                                     \
}                                                                             



/**************************************************************************
* G__GET_STRUCTPVAR()
*
*  get variable 
**************************************************************************/


#define G__GET_STRUCTPVAR1(SIZE,CONVFUNC,TYPE,PTYPE)                          \
switch(G__var_type) {                                                         \
case 'v': /* *var; get value */                                               \
  switch(var->reftype[ig15]) {                                                \
  case G__PARANORMAL:                                                         \
   result.ref=(*(long *)(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC));  \
   if(result.ref)                                                             \
     CONVFUNC(&result,TYPE,result.ref);                                       \
   break;                                                                     \
  case G__PARAP2P:                                                            \
    if(var->paran[ig15]<paran) {                                              \
      address = G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC;             \
      result.ref=(*(((long*)(*(long *)address))+pp_inc));                     \
      if(result.ref)                                                          \
        CONVFUNC(&result,TYPE,result.ref);                                    \
    }                                                                         \
    else {                                                                    \
      result.ref=                                                             \
               (*(long *)(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC)); \
      G__letint(&result,PTYPE,                                                \
      *(long *)(*(long *)(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC)));\
    }                                                                         \
   break;                                                                     \
  }                                                                           \
  break;                                                                      \
case 'P':                                                                     \
   if(var->paran[ig15]==paran) {                                              \
	G__letint(&result,PTYPE                                               \
	   ,(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC));              \
   }                                                                          \
   else if(var->paran[ig15]<paran) {                                          \
	address = G__struct_offset + var->p[ig15]+p_inc*G__LONGALLOC;         \
        if(G__PARANORMAL==var->reftype[ig15])                                 \
	  G__letint(&result,PTYPE                                             \
		      ,(*(long *)(address))+pp_inc*SIZE);                     \
        else {                                                                \
	  G__letint(&result,PTYPE                                             \
	      ,(long)((long*)(*(long *)(address))+pp_inc));                   \
          result.obj.reftype.reftype=G__PARAP2P;                              \
	}                                                                     \
   }                                                                          \
   else {                                                                     \
        G__letint(&result,PTYPE                                               \
	  ,(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC));               \
   }                                                                          \
   break;

#define G__GET_STRUCTPVAR2(SIZE,CONVFUNC,TYPE,PTYPE)                          \
default : /* 'p' */                                                           \
   if(var->paran[ig15]==paran) {                                              \
        /* type *p[];  (p[x]); */                                             \
        result.ref=(long)(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC);  \
	G__letint(&result,PTYPE                                               \
	 ,*(long *)(result.ref));                                             \
   }                                                                          \
   else if(var->paran[ig15]<paran) {                                          \
        /* type *p[];  p[x][y]; */                                            \
	address = G__struct_offset + var->p[ig15]+p_inc*G__LONGALLOC;         \
        if(G__PARANORMAL==var->reftype[ig15]) {                               \
	  result.ref=((*(long *)(address))+pp_inc*SIZE);                      \
	  CONVFUNC(&result,TYPE,result.ref);                                  \
	}                                                                     \
        else if(var->paran[ig15]==paran-1) {                                  \
          result.ref=(long)((long*)(*(long *)(address))+pp_inc);              \
	  G__letint(&result,PTYPE,((*((long*)(result.ref)))));                \
          if(G__PARAP2P<var->reftype[ig15]) {                                 \
            result.obj.reftype.reftype = var->reftype[ig15]-1;                \
          }                                                                   \
	}                                                                     \
        else if(var->paran[ig15]==paran-2) {                                  \
          result.ref=(long)((long*)(*(long *)(address))+para[0].obj.i);       \
          if(G__PARAP2P==var->reftype[ig15]) {                                \
            result.ref=(long)((*((long*)(result.ref)))+para[1].obj.i*SIZE);   \
	    CONVFUNC(&result,TYPE,result.ref);                                \
          }                                                                   \
          else if(G__PARAP2P<var->reftype[ig15]) {                            \
            result.ref=(long)((long*)(*(long *)(result.ref))+para[1].obj.i);  \
	    G__letint(&result,PTYPE,((*((long*)(result.ref)))));              \
            if(G__PARAP2P2P<var->reftype[ig15]) {                             \
              result.obj.reftype.reftype = var->reftype[ig15]-2;              \
            }                                                                 \
          }                                                                   \
	  paran -= 1;                                                         \
	}                                                                     \
        else if(var->paran[ig15]==paran-3) {                                  \
          result.ref=(long)((long*)(*(long *)(address))+para[0].obj.i);       \
          result.ref=(long)((long*)(*(long *)(result.ref))+para[1].obj.i);    \
          if(G__PARAP2P2P==var->reftype[ig15]) {                              \
            result.ref=(long)((*((long*)(result.ref)))+para[2].obj.i*SIZE);   \
	    CONVFUNC(&result,TYPE,result.ref);                                \
          }                                                                   \
          else if(G__PARAP2P2P<var->reftype[ig15]) {                          \
            result.ref=(long)((long*)(*(long *)(result.ref))+para[2].obj.i);  \
	    G__letint(&result,PTYPE,((*((long*)(result.ref)))));              \
            if(G__PARAP2P2P<var->reftype[ig15]) {                             \
              result.obj.reftype.reftype = var->reftype[ig15]-3;              \
            }                                                                 \
          }                                                                   \
	  paran -= 2;                                                         \
	}                                                                     \
        else  { /* This part is not correct. leave this for keeping old beh*/ \
          result.ref=(long)((long*)(*(long *)(address))+para[0].obj.i);       \
          result.ref=(long)((*((long*)(result.ref)))+para[1].obj.i*SIZE);     \
	  CONVFUNC(&result,TYPE,result.ref);                                  \
	  paran -= 2;                                                         \
	}                                                                     \
   }                                                                          \
   else {                                                                     \
        /* type *p[];  p; */                                                  \
        result.ref = (long)(&var->p[ig15]);                                   \
        G__letint(&result,PTYPE                                               \
	  ,(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC));               \
   }                                                                          \
   break;                                                                     \
}                                                                             


#ifndef G__OLDIMPLEMENTATION1135
/******************************************************************
* G__filescopeaccess()
******************************************************************/
int G__filescopeaccess(filenum,statictype)
int filenum;
int statictype;
{
#ifndef G__OLDIMPLEMENTATION1323
  int store_filenum = filenum;
  int store_statictype = statictype;
#endif
  if(filenum==statictype) return(1);
  while(statictype>=0) {
    statictype = G__srcfile[statictype].included_from;
    if(filenum==statictype) return(1);
  }
#ifndef G__OLDIMPLEMENTATION1323
  statictype = store_statictype;
  while(statictype>=0) {
    filenum = store_filenum;
    if(filenum==statictype) return(1);
    statictype = G__srcfile[statictype].included_from;
    while(filenum>=0) {
      if(filenum==statictype) return(1);
      filenum = G__srcfile[filenum].included_from;
    }
  }
#endif
  return(0);
}
#endif

#ifndef G__OLDIMPLEMENTATION674
/******************************************************************
* G__class_conversion_operator()
*
*  conversion operator for assignment to class object
*  bytecode compilation turned off if conversion operator is found
******************************************************************/
int G__class_conversion_operator(tagnum,presult,ttt)
int tagnum;
G__value *presult;
char* ttt;
{
  G__value conv_result;
  int conv_done=0;
  int conv_tagnum = G__tagnum;
  int conv_typenum = G__typenum;
  int conv_constvar = G__constvar;
  int conv_reftype = G__reftype;
  int conv_var_type = G__var_type;
  long conv_store_struct_offset = G__store_struct_offset;

  switch(G__struct.type[presult->tagnum]) {
  case 'c':
  case 's':
    /* stack environment */
    G__tagnum = presult->tagnum;
    G__typenum = -1;
    G__constvar = 0;
    G__reftype = 0;
    G__var_type = 'p';
    G__store_struct_offset = presult->obj.i;

    /* call conversion operator */

    /* synthesize function name */
    strcpy(ttt,"operator ");
    strcpy(ttt+9,G__struct.name[tagnum]);
    strcpy(ttt+strlen(ttt),"()");
#ifdef G__OLDIMPLEMENTATION854
    if(G__dispsource) {
      G__fprinterr(G__serr,"\n!!!Calling conversion operator 0x%lx.%s"
	      ,G__store_struct_offset,ttt);
    }
#endif
    conv_result=G__getfunction(ttt,&conv_done ,G__TRYMEMFUNC);
#if G__TO_BE_IMPLEMENTED
    if(0==conv_done && -1!=var->p_typetable[ig15]) {
      /* another try with typedef alias */
      strcpy(ttt+9,G__type2string(var->type[ig15],-1,-1
                                  ,var->reftype[ig15],var->constvar[ig15]));
      strcpy(ttt+strlen(ttt),"()");
      conv_result=G__getfunction(ttt,&conv_done ,G__TRYMEMFUNC);
    }
#endif
    if(conv_done) {
#ifndef G__OLDIMPLEMENTATION854
      if(G__dispsource) {
	G__fprinterr(G__serr,"!!!Conversion operator called 0x%lx.%s\n"
		,G__store_struct_offset,ttt);
      }
#endif
#ifdef G__ASM
      G__abortbytecode();
#endif
      *presult = conv_result;
    }

    /* restore environment */
    G__tagnum = conv_tagnum;
    G__typenum = conv_typenum;
    G__constvar = conv_constvar;
    G__reftype = conv_reftype;
    G__var_type = conv_var_type;
    G__store_struct_offset = conv_store_struct_offset;
    break;
  }

  return conv_done;
}
#endif

#ifndef G__OLDIMPLEMENTATION672
/******************************************************************
* G__fundamental_conversion_operator()
*
*  conversion operator for assignment to fundamental type object
*  bytecode compilation is alive after conversion operator is used
******************************************************************/
int G__fundamental_conversion_operator(type,tagnum,typenum,reftype
                                             ,constvar,presult,ttt)
int type,tagnum,typenum,reftype,constvar;
G__value *presult;
char* ttt;
{
  G__value conv_result;
  int conv_done=0;
  int conv_tagnum = G__tagnum;
  int conv_typenum = G__typenum;
  int conv_constvar = G__constvar;
  int conv_reftype = G__reftype;
  int conv_var_type = G__var_type;
  long conv_store_struct_offset = G__store_struct_offset;
  switch(G__struct.type[presult->tagnum]) {
  case 'c':
  case 's':
    /* stack environment */
    G__tagnum = presult->tagnum;
    G__typenum = -1;
    G__constvar = 0;
    G__reftype = 0;
    G__var_type = 'p';
    G__store_struct_offset = presult->obj.i;

    /* call conversion operator */
#ifdef G__ASM
    G__asm_inst[G__asm_cp] = G__PUSHSTROS;
    G__asm_inst[G__asm_cp+1] = G__SETSTROS;
    G__inc_cp_asm(2,0);
#ifdef G__ASM_DBG
    if(G__asm_dbg) {
      G__fprinterr(G__serr,"%3x: PUSHSTROS\n",G__asm_cp-2);
      G__fprinterr(G__serr,"%3x: SETSTROS\n",G__asm_cp-1);
    }
#endif
#endif

    /* synthesize function name */
    strcpy(ttt,"operator ");
    strcpy(ttt+9,G__type2string(type,tagnum,typenum,reftype,constvar));
    strcpy(ttt+strlen(ttt),"()");
    conv_result=G__getfunction(ttt,&conv_done ,G__TRYMEMFUNC);
    if(0==conv_done && -1!=typenum) {
      /* another try after removing typedef alias */
      strcpy(ttt+9,G__type2string(type,-1,-1 ,reftype,constvar));
      strcpy(ttt+strlen(ttt),"()");
      conv_result=G__getfunction(ttt,&conv_done ,G__TRYMEMFUNC);
    }
#ifndef G__OLDIMPLEMENTATION962
    if(0==conv_done) {
      /* another try constness reverting */
      constvar ^= 1;
      strcpy(ttt+9,G__type2string(type,tagnum,typenum,reftype,constvar));
      strcpy(ttt+strlen(ttt),"()");
      conv_result=G__getfunction(ttt,&conv_done ,G__TRYMEMFUNC);
      if(0==conv_done && -1!=typenum) {
	/* another try after removing typedef alias */
	strcpy(ttt+9,G__type2string(type,-1,-1 ,reftype,constvar));
	strcpy(ttt+strlen(ttt),"()");
	conv_result=G__getfunction(ttt,&conv_done ,G__TRYMEMFUNC);
      }
    }
#endif
#ifndef G__OLDIMPLEMENTATION1426
    if(0==conv_done) {
      int itype;
      for(itype=0;itype<G__newtype.alltype;itype++) {
	if(type==G__newtype.type[itype]&&tagnum==G__newtype.tagnum[itype]) {
	  constvar ^= 1;
	  strcpy(ttt+9,G__type2string(type,tagnum,itype,reftype,constvar));
	  strcpy(ttt+strlen(ttt),"()");
	  conv_result=G__getfunction(ttt,&conv_done ,G__TRYMEMFUNC);
	  if(0==conv_done) {
	    constvar ^= 1;
	    strcpy(ttt+9,G__type2string(type,tagnum,typenum,reftype,constvar));
	    strcpy(ttt+strlen(ttt),"()");
	    conv_result=G__getfunction(ttt,&conv_done ,G__TRYMEMFUNC);
	  }
	  if(conv_done) break;
	}
      }
    }
#endif
    if(conv_done) {
      if(G__dispsource) {
	G__fprinterr(G__serr,"!!!Conversion operator called 0x%lx.%s\n"
		,G__store_struct_offset,ttt);
      }
      *presult = conv_result;
#ifdef G__ASM
      G__asm_inst[G__asm_cp] = G__POPSTROS;
      G__inc_cp_asm(1,0);
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp-1);
#endif
#endif
    }
    else {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"PUSHSTROS, SETSTROS cancelled\n");
#endif
      G__inc_cp_asm(-2,0);
    }

    /* restore environment */
    G__tagnum = conv_tagnum;
    G__typenum = conv_typenum;
    G__constvar = conv_constvar;
    G__reftype = conv_reftype;
    G__var_type = conv_var_type;
    G__store_struct_offset = conv_store_struct_offset;
    break;
  }
#ifndef G__OLDIMPLEMENTATION1188
  return(conv_done);
#else
  return(0);
#endif
}
#endif /* ON672 */


/******************************************************************
* G__redecl()
******************************************************************/
void G__redecl(var,ig15)
struct G__var_array *var;
int ig15;
{
  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: REDECL\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp] = G__REDECL;
    G__asm_inst[G__asm_cp+1]=ig15;
    G__asm_inst[G__asm_cp+2]=(long)var;
    G__inc_cp_asm(3,0);
  }
}


/**************************************************************************
* G__asm_gen_stvar()
*
**************************************************************************/
int G__asm_gen_stvar(G__struct_offset,ig15,paran,var,item
		     ,store_struct_offset,var_type
#ifndef G__OLDIMPLEMENTATION2089
		     ,presult
#endif
                     )
long G__struct_offset;
int ig15,paran;
struct G__var_array *var;
char *item;
long store_struct_offset;
int var_type;
#ifndef G__OLDIMPLEMENTATION2089
G__value *presult;
#endif
{
#ifndef G__OLDIMPLEMENTATION1911
  if(0 && item) return 0;
#endif
#ifndef G__OLDIMPLEMENTATION2122
  if(G__cintv6) {
    G__value ltype = G__null;
    ltype.isconst = 0;
    ltype.type = var->type[ig15];
    ltype.tagnum = var->p_tagtable[ig15];
    ltype.typenum = var->p_typetable[ig15];
    ltype.obj.reftype.reftype = var->reftype[ig15];
    if(G__Isvalidassignment_val(&ltype,var->paran[ig15],paran,var_type
				,presult)) {
      G__bc_conversion(presult,var,ig15,var_type,paran); 
    }
    else {
      G__fprinterr(G__serr,"Error: assignment type mismatch %s "
                    ,var->varnamebuf[ig15]);
      G__genericerror((char*)NULL);
    }
  }
#endif
#ifndef G__OLDIMPLEMENTATION2089 /* TODO, duplication with G__bc_conversion */
  if(G__cintv6 &&
     ('U'==var->type[ig15] || ('u'==var->type[ig15]&&
                               G__PARAREFERENCE==var->reftype[ig15]))
     && var->type[ig15]==presult->type
     && -1!=var->p_tagtable[ig15] && -1!=presult->tagnum
     && var->p_tagtable[ig15]!=presult->tagnum 
     && -1!=G__ispublicbase(var->p_tagtable[ig15],presult->tagnum,(long)0)) {
#ifndef G__OLDIMPLEMENTATION2154
    if(paran) G__bc_REWINDSTACK(paran);
#endif
#ifdef G__ASM_DBG
    if(G__asm_dbg&&G__asm_noverflow) {
      G__fprinterr(G__serr,"%3x: CAST to %c\n",G__asm_cp,var->type[ig15]);
    }
#endif
    G__asm_inst[G__asm_cp]=G__CAST;
    G__asm_inst[G__asm_cp+1]=var->type[ig15];
    G__asm_inst[G__asm_cp+2]=var->p_typetable[ig15];
    G__asm_inst[G__asm_cp+3]=var->p_tagtable[ig15];
    G__asm_inst[G__asm_cp+4]=(var->reftype[ig15]==G__PARAREFERENCE)?1:0;
    G__inc_cp_asm(5,0);
#ifndef G__OLDIMPLEMENTATION2154
    if(paran) G__bc_REWINDSTACK(-paran);
#endif
  }
#endif
  /************************************
   * ST_GVAR or ST_VAR instruction
   ************************************/
  if(G__struct_offset) {
#ifdef G__NEWINHERIT
    if(G__struct_offset!=store_struct_offset) {
#ifdef G__ASM_DBG
      if(G__asm_dbg)
	G__fprinterr(G__serr,"%3x: ADDSTROS %d\n"
		,G__asm_cp,G__struct_offset-store_struct_offset);
#endif
      G__asm_inst[G__asm_cp]=G__ADDSTROS;
      G__asm_inst[G__asm_cp+1]=G__struct_offset-store_struct_offset;
      G__inc_cp_asm(2,0);
    }
#endif
#ifdef G__ASM_DBG
    if(G__asm_dbg)
      G__fprinterr(G__serr,"%3x: ST_MSTR  %s index=%d paran=%d\n"
	      ,G__asm_cp,item,ig15,paran);
#endif
    G__asm_inst[G__asm_cp]=G__ST_MSTR;
    G__asm_inst[G__asm_cp+1]=ig15;
    G__asm_inst[G__asm_cp+2]=paran;
    G__asm_inst[G__asm_cp+3]=var_type;
    G__asm_inst[G__asm_cp+4]=(long)var;
    G__inc_cp_asm(5,0);
#ifdef G__NEWINHERIT
    if(G__struct_offset!=store_struct_offset) {
#ifdef G__ASM_DBG
      if(G__asm_dbg)
	G__fprinterr(G__serr,"%3x: ADDSTROS %d\n"
		,G__asm_cp,-G__struct_offset+store_struct_offset);
#endif
      G__asm_inst[G__asm_cp]=G__ADDSTROS;
      G__asm_inst[G__asm_cp+1]= -G__struct_offset+store_struct_offset;
      G__inc_cp_asm(2,0);
    }
#endif
  }
#ifndef G__OLDIMPLEMENTATION1517
  else if(G__decl && 
#ifndef G__OLDIMPLEMENTATION1542
	  G__PARAREFERENCE==G__reftype 
#else
	  G__reftype 
#endif
	  && !G__asm_wholefunction) {
    G__redecl(var,ig15);
#ifndef G__OLDIMPLEMENTATION1720
    if(G__no_exec_compile) G__abortbytecode();
#endif
  }
#endif
  else {
#ifdef G__ASM_DBG
    if(G__asm_dbg)
      G__fprinterr(G__serr,
	      "%3x: ST_VAR  %s index=%d paran=%d\n"
	      ,G__asm_cp,item,ig15,paran);
#endif
#ifdef G__ASM_WHOLEFUNC
    if(G__asm_wholefunction && G__ASM_VARLOCAL==store_struct_offset 
#ifndef G__OLDIMPLEMENTATION776
       && G__LOCALSTATIC!=var->statictype[ig15]
#endif
	) {
#ifdef G__ASM_DBG
      if(G__asm_dbg)
        G__fprinterr(G__serr,
	        "%3x: ST_LVAR  %s index=%d paran=%d\n"
	        ,G__asm_cp,item,ig15,paran);
#endif
      G__asm_inst[G__asm_cp]=G__ST_LVAR;
    }
    else {
#ifdef G__ASM_DBG
      if(G__asm_dbg)
        G__fprinterr(G__serr,
	        "%3x: ST_VAR  %s index=%d paran=%d\n"
	        ,G__asm_cp,item,ig15,paran);
#endif
      G__asm_inst[G__asm_cp]=G__ST_VAR;
    }
#else
    G__asm_inst[G__asm_cp]=G__ST_VAR;
#endif
    G__asm_inst[G__asm_cp+1]=ig15;
    G__asm_inst[G__asm_cp+2]=paran;
    G__asm_inst[G__asm_cp+3]=var_type;
    G__asm_inst[G__asm_cp+4]=(long)var;
    G__inc_cp_asm(5,0);
  }

#ifndef G__NEWINHERIT /* NEVER */
  if(G__no_exec_compile) {
    switch(var->type[ig15]) {
    case 'u':
      if(ig25>=paran) {
	G__letstruct(&result ,p_inc ,var ,ig15 ,item ,paran ,G__struct_offset);
      }
      break;
    case 'U':
      if(ig25>=paran) {
	G__letstructp(result ,G__struct_offset ,ig15 ,p_inc ,var ,paran ,item
			/* ,para */ ,pp_inc);
      }
      break;
    }
  }
#endif
  return(0);
}


/**************************************************************************
* G__classassign()
**************************************************************************/
G__value G__classassign(pdest,tagnum,result)
long pdest;
int tagnum;
G__value result;
{
#ifndef G__OLDIMPLEMENTATION1823
  char buf[G__BUFLEN*2];
  char buf2[G__BUFLEN*2];
  char *ttt=buf;
  char *result7=buf2;
  int lenttt;
#else
  char ttt[G__ONELINE],result7[G__ONELINE];
#endif
  long store_struct_offset;
  int store_tagnum;
  int ig2;
  G__value para;
  long store_asm_inst=0;
  int letvvalflag=0;
  long addstros_value=0;

  if(G__asm_exec) {
    memcpy((void*)(pdest),(void*)(G__int(result))
	   ,(size_t)G__struct.size[tagnum]);
    return(result);
  }

  if(result.type=='u') {
#ifndef G__OLDIMPLEMENTATION1823
    char *xp = G__fulltagname(result.tagnum,1);
    lenttt = strlen(xp);
    if(lenttt>G__BUFLEN*2-10) {
      ttt=(char*)malloc(lenttt+20);
      result7=(char*)malloc(lenttt+30);
    }
#ifndef G__OLDIMPLEMENTATION1825
    G__setiparseobject(&result,ttt);
#else
    if(result.obj.i<0) 
      sprintf(ttt,"(%s)(%ld)" ,xp ,result.obj.i);
    else
      sprintf(ttt,"(%s)%ld" ,xp,result.obj.i);
#endif
#else
    if(result.obj.i<0) 
      sprintf(ttt,"(%s)(%ld)" ,G__struct.name[result.tagnum] ,result.obj.i);
    else
      sprintf(ttt,"(%s)%ld" ,G__struct.name[result.tagnum] ,result.obj.i);
#endif
  }
  else {
    G__valuemonitor(result,ttt);
  }

  /**************************************
   * operator=() overloading
   **************************************/

#ifdef G__ASM
  if(G__asm_noverflow) {
    if(G__LETVVAL==G__asm_inst[G__asm_cp-1]) {
      G__inc_cp_asm(-1,0);
      letvvalflag=1;
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	G__fprinterr(G__serr,"LETVVAL cancelled");
	G__printlinenum();
      }
#endif
    }
    else {
      if(G__ADDSTROS==G__asm_inst[G__asm_cp-2]) {
	addstros_value=G__asm_inst[G__asm_cp-1];
	G__inc_cp_asm(-2,0);
      }
      else addstros_value=0;
      letvvalflag=0;
      store_asm_inst=G__asm_inst[G__asm_cp-5];
      if(G__ST_VAR==store_asm_inst) G__asm_inst[G__asm_cp-5]=G__LD_VAR;
#ifndef G__OLDIMPLEMENTATION2060
      else if(G__ST_LVAR==store_asm_inst) G__asm_inst[G__asm_cp-5]=G__LD_LVAR;
#endif
      else                          G__asm_inst[G__asm_cp-5]=G__LD_MSTR;
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	G__fprinterr(G__serr,"ST_VAR or ST_MSTR replaced with LD_VAR or LD_MSTR(1)\n");
	G__printlinenum();
      }
#endif
    }
#ifdef G__ASM_DBG
    if(G__asm_dbg) {
      G__fprinterr(G__serr,"%3x: PUSHSTROS\n",G__asm_cp);
      G__fprinterr(G__serr,"%3x: SETSTROS\n",G__asm_cp+1);
    }
#endif
    G__asm_inst[G__asm_cp] = G__PUSHSTROS;
    G__asm_inst[G__asm_cp+1] = G__SETSTROS;
    G__inc_cp_asm(2,0);
  }
  G__oprovld = 1;
#endif

  /* searching for member function */
  sprintf(result7,"operator=(%s)" ,ttt);

  store_tagnum = G__tagnum;
  G__tagnum = tagnum;
  store_struct_offset = G__store_struct_offset;
  G__store_struct_offset=pdest;
  
  ig2=0;
  para=G__getfunction(result7,&ig2 ,G__TRYMEMFUNC);

#ifndef G__OLDIMPLEMENTATION656
  if(0==ig2 && tagnum!=result.tagnum) {
    /**************************************
     * copy constructor
     **************************************/
    long store_globalvarpointer;
#ifndef G__OLDIMPLEMENTATION1823
    char *xp2= G__fulltagname(tagnum,1);
    int len2;
    lenttt = strlen(ttt);
    len2=strlen(xp2)+lenttt+10;
    if(buf2==result7) {
      if(len2>G__BUFLEN*2) result7=(char*)malloc(len2);
    }
    else {
      if(len2>lenttt+30) {
	free((void*)result7);
	result7=(char*)malloc(len2);
      }
    }
    sprintf(result7,"%s(%s)",xp2,ttt);
#else
    sprintf(result7,"%s(%s)",G__struct.name[tagnum],ttt);
#endif
    if(G__CPPLINK==G__struct.iscpplink[tagnum]) {
      G__abortbytecode();
      store_globalvarpointer = G__globalvarpointer;
      G__globalvarpointer = G__store_struct_offset;
      G__getfunction(result7,&ig2 ,G__TRYCONSTRUCTOR);
      G__globalvarpointer = store_globalvarpointer;
    }
    else {
      G__getfunction(result7,&ig2 ,G__TRYCONSTRUCTOR);
    }
  }
#endif
	
  G__store_struct_offset = store_struct_offset;
  G__tagnum = store_tagnum;

  /* searching for global function */
  if(ig2==0) {
#ifdef G__ASM
    if(G__asm_noverflow) {
      G__inc_cp_asm(-2,0); 
#ifdef G__ASM_DBG
      if(G__asm_dbg) {
	G__fprinterr(G__serr,"PUSHSTROS,SETSTROS cancelled");
	G__printlinenum();
      }
#endif
    }
#endif
    if(pdest<0) 
      sprintf(result7,"operator=((%s)(%ld),%s)"
	      ,G__struct.name[tagnum],pdest,ttt);
    else
      sprintf(result7,"operator=((%s)%ld,%s)"
	      ,G__struct.name[tagnum],pdest,ttt);
    para=G__getfunction(result7,&ig2 ,G__TRYNORMAL);
#ifdef G__ASM
    if(G__asm_noverflow && addstros_value) {
      G__asm_inst[G__asm_cp]=G__ADDSTROS;
      G__asm_inst[G__asm_cp+1]=addstros_value;
#ifdef G__ASM_DBG
      if(G__asm_dbg) 
	G__fprinterr(G__serr,"ADDSTROS %d recovered\n",addstros_value);
#endif
      G__inc_cp_asm(2,0);
    }
#endif
  }
#ifdef G__ASM
  else {
    if(G__asm_noverflow) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp);
#endif
      G__asm_inst[G__asm_cp] = G__POPSTROS;
      G__inc_cp_asm(1,0); 
    }
  }
  G__oprovld = 0;
#endif
  
  if(ig2) { /* in case overloaded = or constructor is found */
#ifndef G__OLDIMPLEMENTATION1823
    if(buf!=ttt) free((void*)ttt);
    if(buf2!=result7) free((void*)result7);
#endif
    return(para);
  }

  /* in case no overloaded = or constructor, memberwise copy */
#ifdef G__ASM
  if(G__asm_noverflow) {
    if(letvvalflag) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"LETVVAL recovered\n");
#endif
      G__asm_inst[G__asm_cp]=G__LETVVAL;
      G__inc_cp_asm(1,0);
    }
    else {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"ST_VAR or ST_MSTR recovered no_exec_compile=%d\n",G__no_exec_compile);
#endif
      G__asm_inst[G__asm_cp-5]=store_asm_inst;
      if(addstros_value) {
	G__asm_inst[G__asm_cp]=G__ADDSTROS;
	G__asm_inst[G__asm_cp+1]=addstros_value;
#ifdef G__ASM_DBG
	if(G__asm_dbg) 
	  G__fprinterr(G__serr,"ADDSTROS %d recovered\n",addstros_value);
#endif
	G__inc_cp_asm(2,0);
      }
    }
  }

#ifndef G__OLDIMPLEMENTATION674
  /* try conversion operator for class object */
  if('u'==result.type && -1!=result.tagnum) {
    if(G__class_conversion_operator(tagnum,&result,ttt)) {
#ifndef G__OLDIMPLEMENTATION1823
      if(buf!=ttt) free((void*)ttt);
      if(buf2!=result7) free((void*)result7);
#endif
      return(G__classassign(pdest,tagnum,result));
    }
  }
#endif /* ON674 */

  /* return from this function if this is pure bytecode compilation */
  if(G__no_exec_compile) {
#ifndef G__OLDIMPLEMENTATION1823
    if(buf!=ttt) free((void*)ttt);
    if(buf2!=result7) free((void*)result7);
#endif
    return(result);
  }

#endif /* of G__ASM */
  if(result.tagnum==tagnum) {
    memcpy((void *)(pdest)
	   ,(void *)(G__int(result))
	   ,(size_t)G__struct.size[tagnum]);
  }
#ifndef G__OLDIMPLEMENTATION668
  else if(-1!=(addstros_value=G__ispublicbase(tagnum,result.tagnum,0))) {
    memcpy((void *)(pdest)
	   ,(void *)(G__int(result)+addstros_value)
	   ,(size_t)G__struct.size[tagnum]);
    if(-1!=G__struct.virtual_offset[tagnum]) 
      *(long*)(pdest+G__struct.virtual_offset[tagnum]) = tagnum;
  }
#endif
  else {
    G__fprinterr(G__serr,
	    "Error: Assignment type incompatible FILE:%s LINE:%d\n"
	    ,G__ifile.name,G__ifile.line_number);
  }
#ifndef G__OLDIMPLEMENTATION1823
  if(buf!=ttt) free((void*)ttt);
  if(buf2!=result7) free((void*)result7);
#endif
  return(result);
}


/******************************************************************
* G__searchvariable()
*
******************************************************************/
struct G__var_array *G__searchvariable(varname,varhash
				       ,varlocal,varglobal
				       ,pG__struct_offset
				       ,pstore_struct_offset
				       ,pig15
				       ,isdecl)
char *varname;
int varhash;
struct G__var_array *varlocal;
struct G__var_array *varglobal;
long *pG__struct_offset;
long *pstore_struct_offset;
int *pig15;
int isdecl;
{
  struct G__var_array *var=NULL;
  int ig15;
  int ilg;
  int in_memfunc=0;
  long scope_struct_offset;
  int scope_tagnum;
  int basen;
  int isbase;
  int accesslimit;
  int memfunc_or_friend=0;
  struct G__inheritance *baseclass=NULL;
#ifdef G__ROOT
#ifndef G__OLDIMPLEMENTATION481
  int specialflag=0;
#endif
#endif
  /* #define G__OLDIMPLEMENTATION1059 */
#ifndef G__OLDIMPLEMENTATION1059
  int save_scope_tagnum;
#endif
#ifndef G__OLDIMPLEMENTATION2038
  struct G__var_array *enclosing_scope=NULL;
#endif

#ifdef G__ROOT
#ifndef G__OLDIMPLEMENTATION481
  if('$'==varname[0] && 
     G__GetSpecialObject && G__GetSpecialObject!=G__getreserved) {
    char temp[G__MAXNAME];
    strcpy(temp,varname+1);
    strcpy(varname,temp);
    specialflag=1;
  }
#endif
#endif

  
  ilg=G__LOCAL; /* start from local variable */
  /* done=0; */
  
  /*
   *
   */
  scope_struct_offset=G__store_struct_offset;
  G__ASSERT(0==G__decl || 1==G__decl);
  if(G__def_struct_member) scope_tagnum=G__get_envtagnum();
  else if(G__decl&&G__exec_memberfunc && -1!=G__memberfunc_tagnum) 
    scope_tagnum=G__memberfunc_tagnum;
  else                     scope_tagnum=G__tagnum;
  switch(G__scopeoperator(varname,&varhash
			  ,&scope_struct_offset,&scope_tagnum)){
  case G__GLOBALSCOPE:
    ilg=G__GLOBAL;
    break;
  case G__CLASSSCOPE:
    ilg=G__MEMBER;
    break;
  }

#ifndef G__OLDIMPLEMENTATION1059
  save_scope_tagnum = scope_tagnum;
#endif
  
  while(ilg<=G__GLOBAL+1) {
    
#ifndef G__OLDIMPLEMENTATION1059
    scope_tagnum = save_scope_tagnum;
#endif
    /***********************************************
     * switch local and global for letvariable
     ************************************************/
    switch(ilg) {
    case G__LOCAL:
#ifdef G__NEWINHERIT
      in_memfunc=0;
#else
      in_memfunc=G__def_struct_member;
#endif
      /******************************************
       * Beginning , local or global entry
       ******************************************/
      if(varlocal) {
#ifdef G__ASM_WHOLEFUNC
	*pstore_struct_offset = G__ASM_VARLOCAL;
#endif
	var=varlocal;
#ifndef G__OLDIMPLEMENTATION2038
	if(var->enclosing_scope) enclosing_scope = var->enclosing_scope;
#endif
	if(varglobal&&0==isdecl) {
#ifndef G__OLDIMPLEMENTATION1366
	  if(G__exec_memberfunc||(-1!=G__tagdefining&&-1!=scope_tagnum)) {
#else
	  if(G__exec_memberfunc) {
#endif
	    ilg=G__MEMBER;
	  }
	  else {
	    ilg=G__GLOBAL;
	  }
	}
	else
	  ilg=G__NOTHING;
      }
	
      else {
	var=varglobal;
	ilg=G__NOTHING;
      }
      break;

    case G__MEMBER:
#ifndef G__OLDIMPLEMENTATION2223
      if(-1!=scope_tagnum) {
	in_memfunc=1;
	*pG__struct_offset = scope_struct_offset;
	G__incsetup_memvar(scope_tagnum);
	var = G__struct.memvar[scope_tagnum] ;
      }
      else {
	in_memfunc=0;
	*pG__struct_offset = scope_struct_offset;
	var = (struct G__var_array*)NULL;
      }
#else
      in_memfunc=1;
      *pG__struct_offset = scope_struct_offset;
      G__incsetup_memvar(scope_tagnum);
      var = G__struct.memvar[scope_tagnum] ;
#endif
      ilg = G__GLOBAL;
      break;
      
    case G__GLOBAL:
      /******************************************
       * global entry
       ******************************************/
      in_memfunc=0;
      *pG__struct_offset = 0;
#ifdef G__ASM_WHOLEFUNC
      *pstore_struct_offset = G__ASM_VARGLOBAL;
#endif
      var=varglobal;
      ilg=G__NOTHING;
      break;
    }
    
    /*************************************************
     * Searching for hash and variable name 
     *************************************************/
    /* If searching for class member, check access rule */
    if(in_memfunc ||(struct G__var_array*)NULL==varglobal) {
      *pstore_struct_offset = *pG__struct_offset;
      isbase=1;
      basen=0;
      baseclass = G__struct.baseclass[scope_tagnum];
      if(G__exec_memberfunc || isdecl || G__isfriend(G__tagnum)) {
	accesslimit = G__PUBLIC_PROTECTED_PRIVATE ;
	memfunc_or_friend = 1;
      }
      else {
	accesslimit = G__PUBLIC;
	memfunc_or_friend = 0;
      }
    }
    else {
      accesslimit = G__PUBLIC;
      isbase=0;
      basen=0;
#ifndef G__OLDIMPLEMENTATION1060
      if (var && var == varglobal) {
        isbase = 1;
        baseclass = &G__globalusingnamespace;
      }
#endif
    }
    /* search for variable name and access rule match */
    do {
    next_base:
      while(var) {
	ig15=0;
	while(ig15<var->allvar) {
	  if(varhash==var->hash[ig15] && 
	     strcmp(varname,var->varnamebuf[ig15])==0 &&
	     (var->statictype[ig15]<0||
#ifndef G__OLDIMPLEMENTATION1135
	      G__filescopeaccess(G__ifile.filenum,var->statictype[ig15])
#else
	      G__ifile.filenum==var->statictype[ig15]
#endif
	     )&&
	     (var->access[ig15]&accesslimit)) {
	    *pig15 = ig15;
	    return(var);
	  }
	  ++ig15;
	}
	var=var->next;
      }
#ifndef G__OLDIMPLEMENTATION2038
      /* enclosing local scope */
      if(enclosing_scope) {
	var=enclosing_scope;
	enclosing_scope = var->enclosing_scope;
	goto next_base;
      }
#endif
      /* next base class if searching for class member */
#ifndef G__OLDIMPLEMENTATION1889
      if(isbase &&
#ifndef G__OLDIMPLEMENTATION1951
	 0<=scope_tagnum &&
#endif
	 'e'==G__struct.type[scope_tagnum] 
	 && G__dispmsg>=G__DISPROOTSTRICT) isbase=0;
#endif
      if(isbase) {
	while(baseclass && basen<baseclass->basen) {
	  if(memfunc_or_friend) {
	    if((baseclass->baseaccess[basen]&G__PUBLIC_PROTECTED) ||
	       baseclass->property[basen]&G__ISDIRECTINHERIT) {
	      accesslimit = G__PUBLIC_PROTECTED;
	      G__incsetup_memvar(baseclass->basetagnum[basen]);
	      var = G__struct.memvar[baseclass->basetagnum[basen]];
#ifdef G__VIRTUALBASE
	      if(baseclass->property[basen]&G__ISVIRTUALBASE) {
		*pG__struct_offset = *pstore_struct_offset 
		  + G__getvirtualbaseoffset(*pstore_struct_offset,scope_tagnum
					    ,baseclass,basen);
	      }
	      else {
		*pG__struct_offset
		  = *pstore_struct_offset + baseclass->baseoffset[basen];
	      }
#else
	      *pG__struct_offset
		= *pstore_struct_offset + baseclass->baseoffset[basen];
#endif
	      ++basen;
	      goto next_base;
	    }
	  }
	  else {
	    if(baseclass->baseaccess[basen]&G__PUBLIC) {
	      accesslimit = G__PUBLIC;
	      G__incsetup_memvar(baseclass->basetagnum[basen]);
	      var = G__struct.memvar[baseclass->basetagnum[basen]];
#ifdef G__VIRTUALBASE
	      if(baseclass->property[basen]&G__ISVIRTUALBASE) {
		*pG__struct_offset = *pstore_struct_offset 
		  + G__getvirtualbaseoffset(*pstore_struct_offset,scope_tagnum
					    ,baseclass,basen);
	      }
	      else {
		*pG__struct_offset
		  = *pstore_struct_offset + baseclass->baseoffset[basen];
	      }
#else
	      *pG__struct_offset
		= *pstore_struct_offset + baseclass->baseoffset[basen];
#endif
	      ++basen;
	      goto next_base;
	    }
	  }
	  ++basen;
	}
#ifndef G__OLDIMPLEMENTATION1059
        /* Also search enclosing scopes. */
        if (scope_tagnum >= 0 && baseclass != &G__globalusingnamespace &&
	    -1!=G__struct.parent_tagnum[scope_tagnum]) {
          scope_tagnum = G__struct.parent_tagnum[scope_tagnum];
          basen =0;
          baseclass = G__struct.baseclass[scope_tagnum];
          var = G__struct.memvar[scope_tagnum];
          goto next_base;
        }
#endif
	isbase=0;
      }
    } while(isbase);

    /* not found */
    *pG__struct_offset = *pstore_struct_offset;
  }

#ifdef G__ROOT
#ifndef G__OLDIMPLEMENTATION481
  if(specialflag) {
    int store_var_type;
    G__value para[1];
#ifndef G__FONS81
    struct G__var_array *store_local;
    store_local = G__p_local;
    G__p_local = NULL;
#endif
    store_var_type = G__var_type;
    G__var_type = 'Z';
    G__allocvariable(G__null,para,varglobal,(struct G__var_array*)NULL,0
		     ,varhash ,varname ,varname , 0);
    G__var_type = store_var_type;
#ifndef G__FONS81
    G__p_local = store_local;
#endif
    var=G__searchvariable(varname,varhash,varlocal,varglobal,pG__struct_offset
			  ,pstore_struct_offset,pig15,isdecl);
#ifndef G__OLDIMPLEMENTATION1891
    if(var) G__gettingspecial=0;
#endif
  }
#endif
#endif

  return(var);
}

/******************************************************************
* G__handle_var_type
*
******************************************************************/
static void G__handle_var_type(item,ttt)
char *item;
char *ttt;
{
#ifndef G__OLDIMPLEMENTATION830
  int pointlevel=0;
  int i=0;
  if(isupper(G__var_type)) ++pointlevel;
  while('*'==item[i++]) ++pointlevel;
  if(G__funcheader 
#ifndef G__OLDIMPLEMENTATION877
     && G__ASM_FUNC_NOP==G__asm_wholefunction
#endif
     ) {
    switch(pointlevel) {
    case 0:
      strcpy(ttt,item+i-1);
      break;
    case 1:
      G__reftype=G__PARANORMAL;
      strcpy(ttt,item+i-1);
      break;
    case 2:
      G__reftype=G__PARANORMAL;
      sprintf(ttt,"%s[]",item+i-1);
      break;
    default:
      G__reftype=G__PARAP2P + pointlevel-3;
      sprintf(ttt,"%s[]",item+i-1);
      break;
    }
  }
  else {	
    switch(pointlevel) {
    case 0:
      break;
    case 1:
      G__reftype=G__PARANORMAL;
      break;
    default:
      G__reftype=G__PARAP2P + pointlevel-2;
      break;
    }
    strcpy(ttt,item+i-1);
  }

#else /* ON830 */
  if(item[1]=='*') {
    if(isupper(G__var_type)) {
      if(G__funcheader) {
	sprintf(ttt,"%s[]",item+2);
	G__reftype=G__PARAP2P;
      }
      else {
	sprintf(ttt,"%s",item+2);
	G__reftype=G__PARAP2P2P;
      }
    }
    else {
      if(G__funcheader) {
	sprintf(ttt,"%s[]",item+2);
      }
      else {
	sprintf(ttt,"%s",item+2);
	G__reftype=G__PARAP2P;
      }
    }
  }
  else if(isupper(G__var_type)) {
    if(G__funcheader) {
      sprintf(ttt,"%s[]",item+1);
    }
    else {
      strcpy(ttt,item+1);
      G__reftype=G__PARAP2P;
    }
  }
  else {
    strcpy(ttt,item+1);
  }
#endif /* ON830 */
  strcpy(item,ttt);
  if(G__var_type=='p') {
    G__var_type='v';
  }
  else {
    G__var_type = toupper(G__var_type);
  }
}


/******************************************************************
* G__class_2nd_decl()
*
******************************************************************/
static void G__class_2nd_decl(var,ig15)
struct G__var_array *var;
int ig15;
{
  int store_cpp_aryconstruct;
  long store_globalvarpointer;
  long store_struct_offset;
  int store_tagnum;
  int store_decl;
  int store_var_type;
  char temp[G__ONELINE];
  int tagnum;
  int known;
  int i;

#define G__OLDIMPLEMENTATION1573
#ifndef G__OLDIMPLEMENTATION1573
  if(G__no_exec_compile && G__asm_noverflow && 0==G__asm_wholefunction) {
    G__abortbytecode();
    return;
  }
#endif

  tagnum = var->p_tagtable[ig15];

  store_var_type=G__var_type;
  G__var_type='p';
  store_tagnum=G__tagnum;
  G__tagnum=tagnum;
  store_struct_offset = G__store_struct_offset;
  G__store_struct_offset=var->p[ig15];
  store_globalvarpointer=G__globalvarpointer;
  G__globalvarpointer=G__PVOID;
  store_cpp_aryconstruct=G__cpp_aryconstruct;
  if(var->varlabel[ig15][1]
#ifndef G__OLDIMPLEMENTATION2011
     || var->paran[ig15]
#endif
     ) G__cpp_aryconstruct=var->varlabel[ig15][1]+1;
  else                       G__cpp_aryconstruct=0;
  store_decl=G__decl;
  G__decl=0;

  sprintf(temp,"~%s()",G__struct.name[tagnum]);
  if(G__dispsource){
    G__fprinterr(G__serr,
	    "\n!!!Calling destructor 0x%lx.%s for declaration of %s"
	    ,G__store_struct_offset
	    ,temp,var->varnamebuf[ig15]);
  }

  if(G__CPPLINK==G__struct.iscpplink[tagnum]) {
    /* delete current object */
#ifndef G__OLDIMPLEMENTATION1568
    if(var->p[ig15]) G__getfunction(temp,&known,G__TRYDESTRUCTOR);
#else  
    G__getfunction(temp,&known,G__TRYDESTRUCTOR);
#endif
    /* set newly constructed object */
    var->p[ig15]=store_globalvarpointer;
    if(G__dispsource){
      G__fprinterr(G__serr," 0x%lx is set",store_globalvarpointer);
    }
  }
  else {
    if(G__cpp_aryconstruct) {
      for(i=G__cpp_aryconstruct-1;i>=0;i--) {
	/* call destructor without freeing memory */
	G__store_struct_offset=var->p[ig15]+G__struct.size[tagnum]*i;
#ifndef G__OLDIMPLEMENTATION1568
	if(var->p[ig15]) G__getfunction(temp,&known,G__TRYDESTRUCTOR);
#else
	G__getfunction(temp,&known,G__TRYDESTRUCTOR);
#endif
	if(G__return>G__RETURN_NORMAL||0==known) break;
      }
    }
    else {
      G__store_struct_offset=var->p[ig15];
#ifndef G__OLDIMPLEMENTATION1568
      if(var->p[ig15]) G__getfunction(temp,&known,G__TRYDESTRUCTOR);
#else
      G__getfunction(temp,&known,G__TRYDESTRUCTOR);
#endif
    }
  }

  G__decl=store_decl;
  G__ASSERT(0==G__decl || 1==G__decl);
  G__cpp_aryconstruct=store_cpp_aryconstruct;
  G__globalvarpointer=store_globalvarpointer;
  G__store_struct_offset=store_struct_offset;
  G__tagnum=store_tagnum;
  G__var_type=store_var_type;
}

/******************************************************************
* G__class_2nd_decl_i()
*
******************************************************************/
static void G__class_2nd_decl_i(var,ig15)
struct G__var_array *var;
int ig15;
{
  int store_no_exec_compile;
  long store_globalvarpointer;
  long store_struct_offset;
  int store_tagnum;
  int size;
  int pinc;
  int known;
  int i;
  char temp[G__ONELINE];

#ifndef G__OLDIMPLEMENTATION1573
  if(G__no_exec_compile && G__asm_noverflow && 0==G__asm_wholefunction) {
    G__abortbytecode();
    return;
  }
#endif

  store_no_exec_compile=G__no_exec_compile;
  G__no_exec_compile=1;
  store_tagnum=G__tagnum;
  G__tagnum=var->p_tagtable[ig15];
  store_struct_offset=G__store_struct_offset;
  store_globalvarpointer=G__globalvarpointer;
  G__globalvarpointer=G__PVOID;

#ifdef G__ASM_DBG
  if(G__asm_dbg) G__fprinterr(G__serr,"%3x: LD_VAR  %s index=%d paran=%d\n"
			 ,G__asm_cp,var->varnamebuf[ig15],ig15,0);
#endif
  G__asm_inst[G__asm_cp]=G__LD_VAR;
  G__asm_inst[G__asm_cp+1]=ig15;
  G__asm_inst[G__asm_cp+2]=0;
  G__asm_inst[G__asm_cp+3]='p';
  G__asm_inst[G__asm_cp+4]=(long)var;
  G__inc_cp_asm(5,0);
  
  G__asm_inst[G__asm_cp] = G__PUSHSTROS;
  G__asm_inst[G__asm_cp+1] = G__SETSTROS;
  G__inc_cp_asm(2,0);
#ifdef G__ASM_DBG
  if(G__asm_dbg) {
    G__fprinterr(G__serr,"%3x: PUSHSTROS\n",G__asm_cp-2);
    G__fprinterr(G__serr,"%3x: SETSTROS\n",G__asm_cp-1);
  }
#endif

  sprintf(temp,"~%s()",G__struct.name[G__tagnum]);
 
  if(var->varlabel[ig15][1]
#ifndef G__OLDIMPLEMENTATION2011
     || var->paran[ig15]
#endif
     ) { /* array */
    size = G__struct.size[G__tagnum];
    pinc=var->varlabel[ig15][1]+1;
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: ADDSTROS %d\n",G__asm_cp,-size*pinc);
#endif
    G__asm_inst[G__asm_cp] = G__ADDSTROS;
    G__asm_inst[G__asm_cp+1] = (long)(size*pinc);
    G__inc_cp_asm(2,0);
    for(i=pinc-1;i>=0;--i) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: ADDSTROS %d\n",G__asm_cp,-size);
#endif
      G__asm_inst[G__asm_cp] = G__ADDSTROS;
      G__asm_inst[G__asm_cp+1] = (long)(-size);
      G__inc_cp_asm(2,0);
      G__getfunction(temp,&known,G__TRYDESTRUCTOR);
    }
  }
  else {
    G__getfunction(temp,&known,G__TRYDESTRUCTOR);
  }

  G__store_struct_offset=store_struct_offset;
  G__tagnum=store_tagnum;
  G__no_exec_compile=store_no_exec_compile;
  G__globalvarpointer=store_globalvarpointer;
}

/******************************************************************
* G__class_2nd_decl_c()
*
******************************************************************/
static void G__class_2nd_decl_c(var,ig15)
struct G__var_array *var;
int ig15;
{
  int store_no_exec_compile;
  long store_globalvarpointer;
  long store_struct_offset;
  int store_tagnum;
  int known;
  char temp[G__ONELINE];

#ifndef G__OLDIMPLEMENTATION1573
  if(G__no_exec_compile && G__asm_noverflow && 0==G__asm_wholefunction) {
    G__abortbytecode();
    return;
  }
#endif

  store_globalvarpointer=G__globalvarpointer;
  G__globalvarpointer=G__PVOID;
  store_no_exec_compile=G__no_exec_compile;
  G__no_exec_compile=1;
  store_tagnum=G__tagnum;
  G__tagnum=var->p_tagtable[ig15];
  store_struct_offset=G__store_struct_offset;

#ifdef G__ASM_DBG
  if(G__asm_dbg) G__fprinterr(G__serr,"%3x: LD_VAR  %s index=%d paran=%d\n"
			 ,G__asm_cp,var->varnamebuf[ig15],ig15,0);
#endif
  G__asm_inst[G__asm_cp]=G__LD_VAR;
  G__asm_inst[G__asm_cp+1]=ig15;
  G__asm_inst[G__asm_cp+2]=0;
  G__asm_inst[G__asm_cp+3]='p';
  G__asm_inst[G__asm_cp+4]=(long)var;
  G__inc_cp_asm(5,0);
  
  G__asm_inst[G__asm_cp] = G__PUSHSTROS;
  G__asm_inst[G__asm_cp+1] = G__SETSTROS;
  G__inc_cp_asm(2,0);
#ifdef G__ASM_DBG
  if(G__asm_dbg) {
    G__fprinterr(G__serr,"%3x: PUSHSTROS\n",G__asm_cp-2);
    G__fprinterr(G__serr,"%3x: SETSTROS\n",G__asm_cp-1);
  }
#endif

  sprintf(temp,"~%s()",G__struct.name[G__tagnum]);
 
  G__getfunction(temp,&known,G__TRYDESTRUCTOR);

#ifdef G__OLDIMPLEMENTATION907
  if(known) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POP\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp] = G__POP;
    G__inc_cp_asm(1,0);
  }
#endif

  G__redecl(var,ig15);
#ifndef G__OLDIMPLEMENTATION1720
  if(store_no_exec_compile) G__abortbytecode();
#endif

  G__store_struct_offset=store_struct_offset;
  G__tagnum=store_tagnum;
  G__no_exec_compile=store_no_exec_compile;
  G__globalvarpointer=store_globalvarpointer;
}

/******************************************************************
* G__value G__letvariable(item,expression,varglobal,varlocal)
*
******************************************************************/
G__value G__letvariable(item,expression,varglobal,varlocal)
char *item;
G__value expression;
struct G__var_array *varglobal,*varlocal;
{
  struct G__var_array *var;
#ifndef G__OLDIMPLEMENTATION1802
  char vv[G__BUFLEN];
  char *varname=vv;
#else
  char varname[G__MAXNAME*2];
#endif
  char parameter[G__MAXVARDIM][G__ONELINE];
  G__value para[G__MAXVARDIM],result = G__null;
  char result7[G__ONELINE];
  int ig15,paran,ig35,ig25,ary;
  int lenitem,nest=0;
  int single_quote=0,double_quote=0,paren=0,flag=0;
  int done=0;
  int p_inc;
  int pp_inc;
  int ig2;
  int store_var_type;
  long G__struct_offset; /* used to be int */
  char ttt[G__ONELINE];
  char *tagname=NULL;
  char *membername=NULL;
  int varhash=0;
  /* int largestep=0; */
  /* int store_prerun,store_debug,store_step; */
  long address;
  long store_struct_offset;
  int store_tagnum;
  int store_exec_memberfunc;
  int store_def_struct_member;
  int store_vartype;
#ifndef G__OLDIMPLEMENTATION1119
  int store_asm_wholefunction;
  int store_no_exec_compile;
  int store_no_exec;
  int store_getarraydim;
  int store_asm_noverflow;
#endif


#ifdef G__ASM
  if(G__asm_exec) {
    ig15=G__asm_index;
    paran = G__asm_param->paran;
    
    for(ig35=0;ig35<paran;ig35++) {
      para[ig35] = G__asm_param->para[ig35];
    }
    
    para[paran]=G__null;
    var=varglobal;
    if(varlocal==NULL) 
      G__struct_offset =0;
    else
      G__struct_offset=G__store_struct_offset;
    result=expression;
    goto exec_asm_letvar;
  }
#endif

  /* This initialization is not necessary. Just to avoid purify error */
  parameter[0][0] = '\0';		/* initialize it */


#ifndef G__OLDIMPLEMENTATION1802
  lenitem=strlen(item);
  if(lenitem>G__BUFLEN-10) varname = (char*)malloc(lenitem+20);
  if(!varname) {
    G__genericerror("Internal error: malloc, G__letvariable(), varname");
    return(G__null);
  }
#endif


  switch(item[0]) {
  case '*': /* value of pointer */
    if(item[1]=='(' || 
       '+'==item[1] || '-'==item[1] ||
       '+'==item[lenitem-1] || '-'==item[lenitem-1]
#ifndef G__OLDIMPLEMENTATION1648
       || ('*'==item[1] && 0==G__decl)
#endif
       ) {
#ifndef G__OLDIMPLEMENTATION2184
      if(G__cintv6 && G__asm_noverflow) {
	result=G__getexpr(item);
	G__bc_objassignment(&result,&expression);
	return(expression); /* ??? or result */
      }
      else {
	result=G__getexpr(item+1);
	G__ASSERT(isupper(result.type)||'u'==result.type);
	para[0]=G__letPvalue(&result,expression);
	if(vv!=varname) free((void*)varname);
	return(para[0]);
      }
#else
      result=G__getexpr(item+1);
      G__ASSERT(isupper(result.type)||'u'==result.type);
      para[0]=G__letPvalue(&result,expression);
#ifndef G__OLDIMPLEMENTATION1802
      if(vv!=varname) free((void*)varname);
#endif
      return(para[0]);
#endif
    }
    G__handle_var_type(item,ttt);
    break;
  case '(': /* parenthesis */
    /* (xxx)=xxx; or (xxx)xxx=xxx; */
    result=G__getfunction(item,&ig15,G__TRYNORMAL);
#ifndef G__OLDIMPLEMENTATION1395
    if(G__CONSTVAR&result.isconst) {
      G__changeconsterror(item,"ignored const");
#ifndef G__OLDIMPLEMENTATION1802
      if(vv!=varname) free((void*)varname);
#endif
      return(result);
    }
#endif
    para[0]=G__letVvalue(&result,expression);
#ifndef G__OLDIMPLEMENTATION1802
    if(vv!=varname) free((void*)varname);
#endif
    return(para[0]);
  case '&': /* pointer */
    /* should not happen */
    G__var_type = 'P'; /* set variable type flag */
    strcpy(ttt,item+1);
    strcpy(item,ttt);
    break;
#ifndef G__OLDIMPLEMENTATION958
  case '0':
  case '1':
  case '2':
  case '3':
  case '4':
  case '5':
  case '6':
  case '7':
  case '8':
  case '9':
  case '.':
  case '-':
  case '+':
    G__fprinterr(G__serr,"Error: assignment to %s",item);
    G__genericerror((char*)NULL);
    break;
#endif
  }
  
  store_var_type=G__var_type;
  G__var_type='p';
  
  /* struct , union member */
  lenitem=strlen(item);
  /* strcpy(ttt,item); */
  ig2=0;
  while((ig2<lenitem)&&(flag==0)) {
    switch(item[ig2]) {
    case '.':
      if((paren==0)&&(double_quote==0)&&(single_quote==0)) {
	/***************************************
	 * To get full struct member name path
	 * when not found
	 ***************************************/
	strcpy(result7,item);
	result7[ig2++]='\0';
	tagname=result7;
	membername=result7+ig2;
	flag=1;
	
      }
      break;
    case '-':
      if((paren==0)&&(double_quote==0)&&(single_quote==0)&&
	 (item[ig2+1]=='>')) {
	/***************************************
	 * To get full struct member name path
	 * when not found
	 ***************************************/
	strcpy(result7,item);
	result7[ig2++]='\0';
	result7[ig2++]='\0';
	tagname=result7;
	membername=result7+ig2;
#ifndef G__OLDIMPLEMENTATION1013
	flag=2;
#else
	flag=1;
#endif
      }
      break;
    case '\\':
      ig2++;
      break;
    case '\'':
      if(double_quote==0) {
	single_quote ^= 1;
      }
      break;
    case '\"':
      if(single_quote==0) {
	double_quote ^= 1;
      }
      break;
    case '{':
    case '[':
    case '(':
      if((single_quote==0)&&(double_quote==0)) {
	paren++;
      }
      break;
    case '}':
    case ']':
    case ')':
      if((single_quote==0)&&(double_quote==0)) {
	paren--;
      }
      break;
    }
    ig2++;
  }
  single_quote=0;double_quote=0;paren=0;

#ifndef G__OLDIMPLEMENTATION1013
  if(flag) {
    result = G__letstructmem(store_var_type
			     ,varname
			     ,membername
			     ,tagname
			     ,varglobal
			     ,expression
			     ,flag
			     );
#ifndef G__OLDIMPLEMENTATION1802
    if(vv!=varname) free((void*)varname);
#endif
    return(result);
  }
#else
  if(flag==1) {
    result = G__letstructmem(store_var_type
			     ,varname
			     ,membername
			     ,tagname
			     ,varglobal
			     ,expression
			     );
#ifndef G__OLDIMPLEMENTATION1802
    if(vv!=varname) free((void*)varname);
#endif
    return(result);
  }
#endif
  /************************************************************
   * end of struct/union member handling.
   * If variable is struct,union member like 'tag.mem', it should 
   * return value upto this point.
   ************************************************************/
  
  /************************************************************
   * varglobal==NULL means, G__letvariable() is called from
   * G__getvariable() or G__letvariable() and G__store_struct_offset 
   * is set by parent G__getvariable().
   *  This is done in different manner in case of loop
   * compilation execution.
   ************************************************************/
  if(varglobal==NULL) {
    G__struct_offset = G__store_struct_offset;
  }
  else {
    G__struct_offset =0;
  }
  
  result=expression;
  
  /* Separate variable name */
  ig15=0;
  varhash=0;
  while((item[ig15]!='(')&&(item[ig15]!='[')&&(ig15<lenitem)) {
    varname[ig15]=item[ig15];
    varhash+=item[ig15++];
  }
  if('('==item[ig15] || 0==ig15) {
    /*  'a.sub(50,20) = b;' */
    if(varglobal==NULL)
      para[0]=G__getfunction(item,&ig15,G__CALLMEMFUNC);
    else
      para[0]=G__getfunction(item,&ig15,G__TRYNORMAL);
#ifndef G__OLDIMPLEMENTATION1802
    if(vv!=varname) free((void*)varname);
#endif
    if(ig15) {
      para[1]=G__letVvalue(&para[0],expression);
      return(para[1]);
    }
    else return(G__null);
  }
  
  varname[ig15++]='\0';
  
  /* Get Parenthesis */
  paran=0;
  if(ig15<lenitem) {
    /* while((item[ig15]!='!')&&(ig15<lenitem)) { */
    while(ig15<lenitem) {
      ig35 = 0;
      nest=0;
      single_quote=0;
      double_quote=0;
      while(((item[ig15]!=']')||(nest>0)
	     ||(single_quote>0)||(double_quote>0))
	    &&(ig15<lenitem)) {
	switch(item[ig15]) {
	case '"' : /* double quote */
	  if(single_quote==0) {
	    double_quote ^= 1;
	  }
	  break;
	case '\'' : /* single quote */
	  if(double_quote==0) {
	    single_quote ^= 1;
	  }
	  break;
	case '(':
	case '[':
	case '{':
	  if((double_quote==0)&&
	     (single_quote==0)) { 
	    nest++;
	  }
	  break;
	case ')':
	case ']':
	case '}':
	  if((double_quote==0)&&
	     (single_quote==0)) { 
	    nest--;
	  }
	  break;
	}
	parameter[paran][ig35++]=item[ig15++];
      }
      ig15++;
      if((item[ig15]=='[')&&(ig15<lenitem)) ig15++;
      parameter[paran++][ig35]='\0';
      parameter[paran][0]='\0';
    }
  }
  
  /************************************************************
   * evaluate parameter if any
   ************************************************************/
  /* restore base environment */
  store_struct_offset = G__store_struct_offset;
  store_tagnum = G__tagnum;
  store_exec_memberfunc=G__exec_memberfunc;
  store_def_struct_member=G__def_struct_member;
  store_vartype = G__var_type;
  if(-1!=G__def_tagnum && G__def_struct_member) {
    G__tagnum = G__def_tagnum;
    G__store_struct_offset = 0;
    G__exec_memberfunc=1;
    G__def_struct_member=0;
  }
  else {
#ifdef G__ASM
    if(G__asm_noverflow&&paran&&
       (G__store_struct_offset!=G__memberfunc_struct_offset
#ifndef G__OLDIMPLEMENTATION2155
	|| G__do_setmemfuncenv
#endif
	)) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: SETMEMFUNCENV\n",G__asm_cp);
#endif
      G__asm_inst[G__asm_cp]=G__SETMEMFUNCENV;
      G__inc_cp_asm(1,0);
    }
#endif
    G__tagnum = G__memberfunc_tagnum;
    G__store_struct_offset = G__memberfunc_struct_offset;
    G__var_type = 'p';
  }

#ifndef G__OLDIMPLEMENTATION1119
  store_asm_wholefunction = G__asm_wholefunction;
  store_no_exec_compile = G__no_exec_compile;  
  store_no_exec = G__no_exec;
  store_getarraydim = G__getarraydim;
  store_asm_noverflow = G__asm_noverflow;
  if(G__decl) {
    G__getarraydim=1;
    if(store_asm_wholefunction) {
      G__asm_wholefunction=0;
      G__no_exec_compile=0;
      G__no_exec=0;
      G__asm_noverflow=0;
    }
  }
#ifndef G__OLDIMPLEMENTATION1443
  if(G__cppconstruct) {
    G__asm_noverflow=0;
  }
#endif
#endif


  /* evaluate parameter */
  for(ig15=0;ig15<paran;ig15++) {
    para[ig15]=G__getexpr(parameter[ig15]);
  }


#ifndef G__OLDIMPLEMENTATION1119
  G__asm_wholefunction = store_asm_wholefunction;
  G__no_exec_compile = store_no_exec_compile;  
  G__no_exec = store_no_exec;
  G__getarraydim = store_getarraydim;
  G__asm_noverflow = store_asm_noverflow;
#endif

  /* recover function call environment */
#ifdef G__ASM
  if(G__asm_noverflow&&paran&&
     (G__store_struct_offset!=store_struct_offset
#ifndef G__OLDIMPLEMENTATION2155
      || G__do_setmemfuncenv
#endif
      )) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: RECMEMFUNCENV\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp]=G__RECMEMFUNCENV;
    G__inc_cp_asm(1,0);
  }
#endif
  G__store_struct_offset=store_struct_offset;
  G__tagnum=store_tagnum;
  G__exec_memberfunc=store_exec_memberfunc;
  G__def_struct_member=store_def_struct_member;
  G__var_type = store_vartype;
  


  G__var_type=store_var_type;
  
  /***********************************************************
   * Search old local and global variables.
   *
   * Start from local, local->next , local->next->next ,,,
   *            global,global->next, global->next->next ,,,
   ***********************************************************/
  
#ifndef G__OLDIMPLEMENTATION1119
  /* avoid searching global variable when allocating in function (static) 
   * const in prerun */
  if(G__func_now<0||!G__decl||!G__static_alloc||!G__constvar||!G__prerun)
    var=G__searchvariable(varname,varhash,varlocal,varglobal,&G__struct_offset
			  ,&store_struct_offset,&ig15
			  ,G__decl||G__def_struct_member);
  else 
    var = (struct G__var_array*)NULL;
#else
  G__ASSERT(0==G__decl || 1==G__decl);
  var = G__searchvariable(varname,varhash,varlocal,varglobal,&G__struct_offset
			  ,&store_struct_offset,&ig15
			  ,G__decl||G__def_struct_member);
#endif

    
    /* assign value */
  if(var) {

#ifndef G__OLDIMPLEMENTATION2182
    if( (G__cintv6 /* &G__BC_DEBUG */)
       && G__asm_noverflow && 0==G__asm_exec) {
      G__bc_assignment(var,ig15,paran,G__var_type,&result
		       ,G__struct_offset,store_struct_offset);
      return(result);
    }
#endif

#ifndef G__OLDIMPLEMENTATION536
    /*******************************************************
     * Experimental code to block duplicate declaration
     *******************************************************/ 
    G__ASSERT(0==G__decl || 1==G__decl);
    if((G__decl||G__cppconstruct) && 'p'!=G__var_type && 
       G__AUTO==var->statictype[ig15] &&
       (var->type[ig15]!=G__var_type||var->p_tagtable[ig15]!=G__tagnum)) {
      G__fprinterr(G__serr,"Error: %s already declared as different type",item);
      if(isupper(var->type[ig15])&&isupper(G__var_type)&&
         0==var->varlabel[ig15][1]  /* 2011 ??? */
	 &&0==(*(long*)var->p[ig15])) { 
        G__fprinterr(G__serr,". Switch to new type\n");
        var->type[ig15]=G__var_type;
        var->p_tagtable[ig15]=G__tagnum;
        var->p_typetable[ig15]=G__typenum;
      }
      else {
#ifndef G__OLDIMPLEMENTATION538
	if(G__PVOID!=G__globalvarpointer&&'u'==G__var_type&&-1!=G__tagnum&&
	   G__CPPLINK==G__struct.iscpplink[G__tagnum]) {
	  char protect_temp[G__ONELINE];
	  long protect_struct_offset=G__store_struct_offset;
          int done=0;
	  G__store_struct_offset=G__globalvarpointer;
          G__globalvarpointer=G__PVOID;
	  sprintf(protect_temp,"~%s()",G__struct.name[G__tagnum]);
          G__fprinterr(G__serr,". %s called\n",protect_temp);
	  G__getfunction(protect_temp,&done,G__TRYDESTRUCTOR);
	  G__store_struct_offset=protect_struct_offset;
	}
#endif
        G__genericerror(NULL);
#ifndef G__OLDIMPLEMENTATION1802
	if(vv!=varname) free((void*)varname);
#endif
        return(G__null);
      }
    }
#endif /* ON536 */

#ifndef G__OLDIMPLEMENTATION672
    if('u'!=tolower(var->type[ig15])&&'u'==result.type&&-1!=result.tagnum) {
#ifndef G__OLDIMPLEMENTATION1880
      if(G__asm_noverflow && paran) {
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: REWINDSTACK %d\n"
			       ,G__asm_cp,paran);
#endif
	G__asm_inst[G__asm_cp] = G__REWINDSTACK;
	G__asm_inst[G__asm_cp+1] = paran;
	G__inc_cp_asm(2,0);
      }
#endif /* 1880 */
      G__fundamental_conversion_operator(var->type[ig15]
					 ,var->p_tagtable[ig15]
					 ,var->p_typetable[ig15]
					 ,var->reftype[ig15]
					 ,var->constvar[ig15]
					 ,&result,ttt);
#ifndef G__OLDIMPLEMENTATION1880
      if(G__asm_noverflow && paran) {
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: REWINDSTACK %d\n"
			       ,G__asm_cp,paran);
#endif
	G__asm_inst[G__asm_cp] = G__REWINDSTACK;
	G__asm_inst[G__asm_cp+1] = -paran;
	G__inc_cp_asm(2,0);
      }
#endif /* 1880 */
    }
#endif /* ON672 */
      
#ifdef G__ASM
    /*******************************************************
     * bytecode generation for G__letvariable()
     *******************************************************/
    G__ASSERT(0==G__decl || 1==G__decl);
    if(G__asm_noverflow
#ifndef G__OLDIMPLEMENTATION1349
       && 0==G__decl_obj
#endif
       ) {
      if( 
#ifndef G__OLDIMPLEMENTATION1073
	 (0==G__decl || !G__asm_wholefunction) &&
#endif
#ifndef G__OLDIMPLEMENTATION1661
	  ('v'!=G__var_type || 'u'!=var->type[ig15])
#else
	 1
#endif
	  ) { /* ??? */
#ifndef G__OLDIMPLEMENTATION1671
	if(result.type) {
	  G__asm_gen_stvar(G__struct_offset,ig15,paran,var,item
			   ,store_struct_offset,G__var_type
#ifndef G__OLDIMPLEMENTATION2089
                           , &result
#endif
                           );
	}
	else if('u'==G__var_type) {
	  G__ASSERT(0==G__decl || 1==G__decl);
	  if(G__decl) {
	    if(G__reftype) {
	      G__redecl(var,ig15);
#ifndef G__OLDIMPLEMENTATION1720
	      if(G__no_exec_compile) G__abortbytecode();
#endif
	    }
	    else G__class_2nd_decl_i(var,ig15);
	  }
	  else if(G__cppconstruct) {
	    G__class_2nd_decl_c(var,ig15);
	  }
	}
#else
	G__asm_gen_stvar(G__struct_offset,ig15,paran,var,item
			 ,store_struct_offset,G__var_type
#ifndef G__OLDIMPLEMENTATION2089
			 , &result
#endif
                         );
#endif
      }
    }
    else if('u'==G__var_type&&G__AUTO==var->statictype[ig15]&&
	    (G__decl||G__cppconstruct)) {
#ifndef G__OLDIMPLEMENTATION1671
#ifdef G__ASM
      if(G__asm_noverflow) {
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: LD_VAR  %s index=%d paran=%d\n"
				    ,G__asm_cp,var->varnamebuf[ig15],ig15,0);
#endif
	G__asm_inst[G__asm_cp]=G__LD_VAR;
	G__asm_inst[G__asm_cp+1]=ig15;
	G__asm_inst[G__asm_cp+2]=0;
	G__asm_inst[G__asm_cp+3]='p';
	G__asm_inst[G__asm_cp+4]=(long)var;
	G__inc_cp_asm(5,0);
	
	G__asm_inst[G__asm_cp] = G__PUSHSTROS;
	G__asm_inst[G__asm_cp+1] = G__SETSTROS;
	G__inc_cp_asm(2,0);
#ifdef G__ASM_DBG
	if(G__asm_dbg) {
	  G__fprinterr(G__serr,"%3x: PUSHSTROS\n",G__asm_cp-2);
	  G__fprinterr(G__serr,"%3x: SETSTROS\n",G__asm_cp-1);
	}
#endif
      }
#endif
#endif
      G__class_2nd_decl(var,ig15);
      result.obj.i=var->p[ig15];
      result.type='u';
      result.tagnum=var->p_tagtable[ig15];
      result.typenum=var->p_typetable[ig15];
      result.ref=var->p[ig15];
#ifndef G__OLDIMPLEMENTATION797
      G__var_type = 'p';
#endif
#ifndef G__OLDIMPLEMENTATION1802
      if(vv!=varname) free((void*)varname);
#endif
      return(result);
    }
    
  exec_asm_letvar:
#endif

    /*******************************************************
     * static class/struct member
     *******************************************************/
    if(G__struct_offset && G__LOCALSTATIC==var->statictype[ig15])
      G__struct_offset=0;
    
      
    /*******************************************************
     * Assign G__null to existing variable is ingored.
     * This is in most cases duplicate declaration.
     *******************************************************/
    if(result.type=='\0') { 
#ifndef G__OLDIMPLEMENTATION1398
      if(G__asm_noverflow&&'u'==G__var_type&&G__AUTO==var->statictype[ig15]&&
	 (G__decl||G__cppconstruct)) {
	int store_asm_noverflow = G__asm_noverflow;
	G__asm_noverflow = 0;
	G__class_2nd_decl(var,ig15);
	G__asm_noverflow = store_asm_noverflow;
	result.obj.i=var->p[ig15];
	result.type='u';
	result.tagnum=var->p_tagtable[ig15];
	result.typenum=var->p_typetable[ig15];
	result.ref=var->p[ig15];
      }
#endif
      G__var_type = 'p';
#ifndef G__OLDIMPLEMENTATION798
      if(G__reftype && G__PVOID!=G__globalvarpointer) {
	var->p[ig15] = G__globalvarpointer;
      }
#endif
#ifndef G__OLDIMPLEMENTATION1802
      if(vv!=varname) free((void*)varname);
#endif
      return(result);
    }
    
      /* ON199 */
      /********************************************************
      * In case of duplicate declaration, make it normal assignment
      ********************************************************/
      switch(G__var_type) {
      case 'p': /* normal assignment */
      case 'v': /* *pointer assignment */
      case 'P': /* assignment to pointer, illegal */
	break;
      case 'u': /* special case, initialization of static member. So, class
		 * object may have problem with block scope. */
	break;
      default: /* duplicated declaration handled this way because cint
		* does not support block scope */
	G__var_type='p';
	break;
      }
      
      /* check const variable */
      if((var->constvar[ig15])&&(G__funcheader==0)&&
	 (tolower(var->type[ig15])!='p')) {
#ifdef G__OLDIMPLEMENTATION1119
	if(var->constvar[ig15]&G__LOCKVAR) {
	  G__lockedvariable(var->varnamebuf[ig15]);
	  G__var_type='p';
#ifndef G__OLDIMPLEMENTATION1802
	  if(vv!=varname) free((void*)varname);
#endif
	  return(result);
	}
#endif
#ifndef G__OLDIMPLEMENTATION1364
	if(((0==G__prerun && !G__decl) 
	    || G__COMPILEDGLOBAL==var->statictype[ig15]) && 
	   (islower(var->type[ig15])||
	    ('p'==G__var_type&&(var->constvar[ig15]&G__PCONSTVAR))||
	    ('v'==G__var_type&&(var->constvar[ig15]&G__CONSTVAR)))) {
	  G__changeconsterror(var->varnamebuf[ig15],"ignored const");
	  G__var_type='p';
#ifndef G__OLDIMPLEMENTATION1802
	  if(vv!=varname) free((void*)varname);
#endif
	  return(result);
	}
#else
	if((0==G__prerun || G__COMPILEDGLOBAL==var->statictype[ig15]) && 
	   !G__decl &&
	   (islower(var->type[ig15])||
	    ('p'==G__var_type&&(var->constvar[ig15]&G__PCONSTVAR))||
	    ('v'==G__var_type&&(var->constvar[ig15]&G__CONSTVAR)))) {
	  G__changeconsterror(var->varnamebuf[ig15],"ignored const");
	  G__var_type='p';
#ifndef G__OLDIMPLEMENTATION1802
	  if(vv!=varname) free((void*)varname);
#endif
	  return(result);
	}
#endif
      }
#ifndef G__OLDIMPLEMENTATION1394
      if(-1!=var->p_typetable[ig15] && 
	 G__newtype.isconst[var->p_typetable[ig15]]) {
	int constvar = G__newtype.isconst[var->p_typetable[ig15]];
	/* int ttype = G__newtype.type[var->p_typetable[ig15]]; */
	if(((0==G__prerun && !G__decl) 
	    || G__COMPILEDGLOBAL==var->statictype[ig15]) && 
	   (islower(var->type[ig15])||
	    ('p'==G__var_type&&(constvar&G__PCONSTVAR))||
	    ('v'==G__var_type&&(constvar&G__CONSTVAR)))) {
	  G__changeconsterror(var->varnamebuf[ig15],"ignored const");
	  G__var_type='p';
#ifndef G__OLDIMPLEMENTATION1802
	  if(vv!=varname) free((void*)varname);
#endif
	  return(result);
	}
      }
#endif
      
      /*************************************************
       * Variable found, set done flags
       *************************************************/
      done++;
      
      /*************************************************
       * type array[A][B][C][D]
       *
       *  ary = B*C*D which is stored into
       * var->varlabel[var_identity][0]
       *************************************************/
      ary=var->varlabel[ig15][0];
      
      /*************************************************
       *  array[i][j][k][l]
       *
       *  p_inc = B*C*D*i + C*D*j + D*k + l
       *  pp_inc = 
       *************************************************/
      p_inc=0;
      for(ig25=0;ig25<paran&&ig25<var->paran[ig15];ig25++) {
	p_inc += ary*G__int(para[ig25]);
	ary /= var->varlabel[ig15][ig25+2];
      }
      pp_inc=0;
      ary=var->varlabel[ig15][ig25+3];
      while(ig25<paran&&var->varlabel[ig15][ig25+4]) {
	pp_inc += ary*G__int(para[ig25]);
	ary /= var->varlabel[ig15][ig25+4];
	++ig25;
      }

#ifdef G__ASM
      if(G__no_exec_compile && ('u'!=tolower(var->type[ig15])||ig25<paran)) {
	result.obj.d = 0;
	result.obj.i = 1;
	result.tagnum = var->p_tagtable[ig15];
	result.typenum = var->p_typetable[ig15];
	if(isupper(var->type[ig15])) {
	  switch(G__var_type) {
	  case 'v':
	    result.type=tolower(var->type[ig15]);
	    break;
	  case 'P':
	    result.type=var->type[ig15];
	    break;
	  default:
	    if(var->paran[ig15]<paran) result.type=tolower(var->type[ig15]);
	    else result.type=var->type[ig15];
	    break;
	  }
	}
	else {
	  switch(G__var_type) {
	  case 'p':
	    if(var->paran[ig15]<=paran) result.type=var->type[ig15];
	    else result.type=toupper(var->type[ig15]);
#ifndef G__OLDIMPLEMENTATION1498
	    if('u'==result.type && -1!=result.tagnum &&
	       'e'!=G__struct.type[result.tagnum]) {
	      result.ref = 1;
	      G__tryindexopr(&result,para,paran,ig25);
	      para[0]=result;
	      para[0]=G__letVvalue(&para[0],expression);
#ifdef G__OLDIMPLEMENTATION1768 /* side effect, t599.cxx,t601.cxx */
	      if('u'==result.type && -1!=result.tagnum &&
		 'e'!=G__struct.type[result.tagnum]) {
		G__classassign(0,result.tagnum,expression);
	      }
#endif
	    }
#endif
	    break;
	  case 'P':
	    result.type=toupper(var->type[ig15]);
	    break;
	  default:
	    G__reference_error(item);
	    break;
	  }
	}
	G__var_type='p';
#ifndef G__OLDIMPLEMENTATION1802
	if(vv!=varname) free((void*)varname);
#endif
	return(result);
      }
#endif
      
      /*************************************************
       *  check p_inc doesn't violate segmentation
       *
       *  0 <= p_inc < A*B*C*D = var->varlabel[iden][1]
       *************************************************/
      if(
#ifndef G__OLDIMPLEMENTATION1502
	 0==G__no_exec_compile &&
#endif
	 (p_inc<0||p_inc>var->varlabel[ig15][1]||
	 (ig25<paran&&tolower(var->type[ig15])!='u')) 
	 && var->reftype[ig15]==G__PARANORMAL) {
	G__arrayindexerror(ig15,var,item,p_inc);
#ifndef G__OLDIMPLEMENTATION1802
	if(vv!=varname) free((void*)varname);
#endif
	return(expression);
      }

#ifdef G__SECURITY
      if(0==G__no_exec_compile&&'v'==G__var_type&&isupper(var->type[ig15])&&
#ifndef G__OLDIMPLEMENTATION475
	 G__PARANORMAL==var->reftype[ig15]&&
	 0==var->varlabel[ig15][1]&&   /* 2011 ??? */
#endif
	 0==(*(long*)(G__struct_offset+var->p[ig15]))) {
	G__assign_error(item,&result);
#ifndef G__OLDIMPLEMENTATION1802
	if(vv!=varname) free((void*)varname);
#endif
	return(G__null);
      }
#endif
#ifndef G__OLDIMPLEMENTATION575
      if(G__security&G__SECURE_POINTER_TYPE && !G__definemacro && 
	 isupper(var->type[ig15]) && 'p'==G__var_type && 0==paran && 
#if !defined(G__OLDIMPLEMENTATION2191)
	 '1'!=var->type[ig15] &&
#else
	 'Q'!=var->type[ig15] &&
#endif
	 (('Y'!=var->type[ig15] && 'Y'!=result.type && result.obj.i)||
	  G__security&G__SECURE_CAST2P) ) {
	if(var->type[ig15]!=result.type ||
	   ('U'==result.type && G__security&G__SECURE_CAST2P &&
#ifdef G__VIRTUALBASE
	    -1==G__ispublicbase(var->p_tagtable[ig15],result.tagnum
				,G__STATICRESOLUTION2))) {
#else
	    -1==G__ispublicbase(var->p_tagtable[ig15],result.tagnum))) {
#endif
#ifndef G__OLDIMPLEMENTATION1802
	  G__CHECK(G__SECURE_POINTER_TYPE,0!=result.obj.i,{if(vv!=varname)free((void*)varname);return(G__null);});
#else
	  G__CHECK(G__SECURE_POINTER_TYPE,0!=result.obj.i,return(G__null));
#endif
	}
      }
#else
      if(G__security&G__SECURE_CAST2P && !G__definemacro && 
	 isupper(var->type[ig15])) {
	if(var->type[ig15]!=result.type ||
	   ('U'==result.type &&
#ifdef G__VIRTUALBASE
	    -1==G__ispublicbase(var->p_tagtable[ig15],result.tagnum
				,G__STATICRESOLUTION2))) {
#else
	    -1==G__ispublicbase(var->p_tagtable[ig15],result.tagnum))) {
#endif
#ifndef G__OLDIMPLEMENTATION1802
	  G__CHECK(G__SECURE_CAST2P,1,{if(vv!=varname)free((void*)varname);return(G__null);});
#else
	  G__CHECK(G__SECURE_CAST2P,1,return(G__null));
#endif
	}
      }
#endif
#ifndef G__OLDIMPLEMENTATION1802
      G__CHECK(G__SECURE_POINTER_AS_ARRAY 
	       ,(var->paran[ig15]<paran&&isupper(var->type[ig15]))
	       ,{if(vv!=varname)free((void*)varname);return(G__null);});
      G__CHECK(G__SECURE_POINTER_ASSIGN
	       ,var->paran[ig15]>paran||isupper(var->type[ig15])
	       ,{if(vv!=varname)free((void*)varname);return(G__null);});
#else
      G__CHECK(G__SECURE_POINTER_AS_ARRAY 
	       ,(var->paran[ig15]<paran&&isupper(var->type[ig15]))
	       ,return(G__null));
      G__CHECK(G__SECURE_POINTER_ASSIGN
	       ,var->paran[ig15]>paran||isupper(var->type[ig15])
	       ,return(G__null));
#endif
#ifdef G__SECURITY
      if(G__security&G__SECURE_GARBAGECOLLECTION &&
#ifndef G__OLDIMPLEMENTATION545
	 (!G__no_exec_compile) &&
#endif
	 isupper(var->type[ig15]) && 'v'!=G__var_type && 'P'!=G__var_type 
#ifndef G__OLDIMPLEMENTATION575
	 && ((0==paran&&0==var->varlabel[ig15][1]) ||
	     (1==paran&&1==var->varlabel[ig15][2]&&0==var->varlabel[ig15][3]))
#endif
	 ) {
	address = G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC;	 
	if(address && *(long*)address) {
	  G__del_refcount((void*)(*(long*)address),(void**)address);
	}
	if(isupper(result.type) && result.obj.i && address) {
	  G__add_refcount((void*)result.obj.i,(void**)address);
	}
      }
#endif
      
#ifndef G__OLDIMPLEMENTATION448
      /********************************************************
      * assign bit-field value
      ********************************************************/
      if(var->bitfield[ig15] && 'p'==G__var_type) {
	int mask,finalval,original;
	address=G__struct_offset+var->p[ig15];
	original = *(int*)address;
	mask = ((1<<var->bitfield[ig15])-1);
	mask = mask<<var->varlabel[ig15][G__MAXVARDIM-1];
	finalval= (original&(~mask))
	  + ((result.obj.i<<var->varlabel[ig15][G__MAXVARDIM-1])&mask);
	(*(int*)address) = finalval;
#ifndef G__OLDIMPLEMENTATION1802
	if(vv!=varname) free((void*)varname);
#endif
	return(result);
      }
#endif

      /**************************************************/
      
      switch(var->type[ig15]) {
	
#ifndef G__OLDIMPLEMENTATION2189
      case 'n': /* G__int64 */
	G__ASSIGN_VAR(G__LONGLONGALLOC,G__int64,G__Longlong,result.obj.ll)
	break;
      case 'm': /* G__uint64 */
	G__ASSIGN_VAR(G__LONGLONGALLOC,G__uint64
		      ,G__ULonglong,result.obj.ull)
	break;
      case 'q': /* long double */
	G__ASSIGN_VAR(G__LONGDOUBLEALLOC,long double
		      ,G__Longdouble,result.obj.ld)
	break;
#endif

#ifndef G__OLDIMPLEMENTATION1604
      case 'g': /* bool */
	switch(result.type) {
	case 'd':
	case 'f':
	  result.obj.d = result.obj.d?1:0;
	  break;
	default:
	  result.obj.i = result.obj.i?1:0;
	  break;
	}
#ifdef G__BOOL4BYTE
	G__ASSIGN_VAR(G__INTALLOC,int,G__int,result.obj.i)
#else
	G__ASSIGN_VAR(G__CHARALLOC,unsigned char,G__int,result.obj.i)
#endif
#endif
      case 'i': /* int */
	G__ASSIGN_VAR(G__INTALLOC,int,G__int,result.obj.i)
	  
      case 'd': /* double */
	G__ASSIGN_VAR(G__DOUBLEALLOC,double,G__double,result.obj.d)
	    
      case 'c': /* char */
#ifndef G__OLDIMPLEMENTATION1647
	if(G__decl && INT_MAX==var->varlabel[ig15][1] && paran &&
	   paran==var->paran[ig15] && 'p'==G__var_type && 
	   0==G__struct_offset && 'C'==result.type && result.obj.i) {
	  var->p[ig15] = result.obj.i;
#ifndef G__OLDIMPLEMENTATION1802
	  if(vv!=varname) free((void*)varname);
#endif
	  return(result);
	}
	else {
	  G__ASSIGN_VAR(G__CHARALLOC,char,G__int,result.obj.i)
	}
#else
	G__ASSIGN_VAR(G__CHARALLOC,char,G__int,result.obj.i)
#endif
	      
      case 'b': /* unsigned char */
	G__ASSIGN_VAR(G__CHARALLOC,unsigned char ,G__int,result.obj.i)

      case 's': /* short int */
	G__ASSIGN_VAR(G__SHORTALLOC,short,G__int,result.obj.i)

      case 'r': /* unsigned short int */
	G__ASSIGN_VAR(G__SHORTALLOC,unsigned short ,G__int,result.obj.i)

      case 'h': /* unsigned int */
	G__ASSIGN_VAR(G__INTALLOC,unsigned int,G__int,result.obj.i)

      case 'l': /* long int */
	G__ASSIGN_VAR(G__LONGALLOC,long ,G__int,result.obj.i)


      case 'k': /* unsigned long int */
	G__ASSIGN_VAR(G__LONGALLOC,unsigned long ,G__int,result.obj.i)

      case 'f': /* float */
	G__ASSIGN_VAR(G__FLOATALLOC,float,G__double,result.obj.d)


	/***************************************
	 * G__letvariable(), old variable
	 * file and void pointers are same as
	 * char pointer
	 ***************************************/
      case 'E': /* file pointer */
      case 'Y': /* void pointer */
#ifndef G__OLDIMPLEMENTATION2191
      case '1': /* pointer to function */
#else
      case 'Q': /* pointer to function */
#endif
      case 'C': /* char pointer */
	G__ASSIGN_PVAR(char,G__int,result.obj.i)
	break;

#ifndef G__OLDIMPLEMENTATION2189
      case 'N':
	G__ASSIGN_PVAR(G__int64,G__Longlong,result.obj.ll)
	break;
      case 'M':
	G__ASSIGN_PVAR(G__uint64,G__ULonglong,result.obj.ull)
	break;
#ifndef G__OLDIMPLEMENTATION2191
      case 'Q':
	G__ASSIGN_PVAR(long double,G__Longdouble,result.obj.ld)
	break;
#endif
#endif

#ifndef G__OLDIMPLEMENTATION1604
      case 'G': /* bool */
#endif
      case 'B': /* unsigned char pointer */
	G__ASSIGN_PVAR(unsigned char,G__int,result.obj.i)
	break;

      case 'S': /* short pointer */
	G__ASSIGN_PVAR(short,G__int,result.obj.i)
	break;

      case 'R': /* unsigned short pointer */
	G__ASSIGN_PVAR(unsigned short,G__int,result.obj.i)
	break;

      case 'I': /* int pointer */
	G__ASSIGN_PVAR(int,G__int,result.obj.i)
	break;

      case 'H': /* unsigned int pointer */
	G__ASSIGN_PVAR(unsigned int,G__int,result.obj.i)
	break;

      case 'L': /* long int pointer */
	G__ASSIGN_PVAR(long,G__int,result.obj.i)
	break;

      case 'K': /* unsigned long int pointer */
	G__ASSIGN_PVAR(unsigned long,G__int,result.obj.i)
	break;

      case 'F': /* float pointer */
	G__ASSIGN_PVAR(float,G__double,result.obj.d)
	break;

      case 'D': /* double pointer */
	G__ASSIGN_PVAR(double,G__double,result.obj.d)
	break;

      case 'u': /* struct,union */
	if(ig25<paran) {
	  result.tagnum=var->p_tagtable[ig15];
	  result.typenum=var->p_typetable[ig15];
	  result.ref=(long)(G__struct_offset+(var->p[ig15])  
			    +p_inc*G__struct.size[var->p_tagtable[ig15]]);
	  G__letint(&result,'u',(result.ref));
	  G__tryindexopr(&result,para,paran,ig25);
	  para[0]=G__letVvalue(&result,expression);
#ifndef G__OLDIMPLEMENTATION1802
	  if(vv!=varname) free((void*)varname);
#endif
	  return(para[0]);
	}
	else {
	  /* 1068 Don't know how to implement */
	  G__letstruct(&result,p_inc,var,ig15,item,paran,G__struct_offset);
	}
	break;
	
#ifdef G__ROOT
#if !defined(G__OLDIMPLEMENTATION481)
      case 'Z': /* struct,union */
	G__reference_error(item);
	break;
#elif !defined(G__OLDIMPLEMENTATION467)
      case 'Z': /* struct,union */
#endif
#endif
      case 'U': /* struct,union */
	if(ig25<paran
	   && ((
#ifdef G__OLDIMPLEMENTATION1430
		1==paran-ig25 &&
#endif
		G__PARANORMAL==var->reftype[ig15]) ||
	       paran-ig25==var->reftype[ig15])
	   ) {
	  result.tagnum=var->p_tagtable[ig15];
	  result.typenum=var->p_typetable[ig15];
	  result.ref = 0;
	  address = G__struct_offset + var->p[ig15]+p_inc*G__LONGALLOC;
	  result.ref=((*(long *)(address))+pp_inc*G__struct.size[var->p_tagtable[ig15]]);
	  G__letint(&result,'u',result.ref);
	  G__tryindexopr(&result,para,paran,ig25);
	  para[0]=G__letVvalue(&result,expression);
#ifndef G__OLDIMPLEMENTATION1802
	  if(vv!=varname) free((void*)varname);
#endif
	  return(para[0]);
	}
	else {
	  G__letstructp(result ,G__struct_offset ,ig15 ,p_inc ,var ,paran
			,item ,para ,pp_inc);
	}
	break;

     case 'a': /* pointer to member function */
	G__letpointer2memfunc(var,paran,ig15,item,p_inc,&result
			      ,G__struct_offset);
	break;

#ifndef G__OLDIMPLEMENTATION1939
      case 'T': /* macro char* */
	if((G__globalcomp==G__NOLINK)&&(G__prerun==0)&&
	   (G__double(result)!=G__double(G__getitem(item)))) {
	  G__changeconsterror(varname ,"enforced macro");
	}
	*(long*)var->p[ig15] = result.obj.i;
	break;
#endif
	
      case 'p': /* macro int */
      case 'P': /* macro double */
	if((G__globalcomp==G__NOLINK)&&(G__prerun==0)&&
	   (G__double(result)!=G__double(G__getitem(item)))) {
	  G__changeconsterror(varname ,"enforced macro");
	}
      default: /* case 'X' automatic variable */
	G__letautomatic(var,ig15,G__struct_offset,p_inc,result);
	break;
      }
    }
    
    /********************************************************
     * If this is a variable declaration and the variable name
     * is not found in the local variable table, stop searching
     * old variable name.
     *  Duplicate declaration is just ignored and the initilize
     * value is stored into the old variable.
     *********************************************************/
  
  if(done==0) { /* new variable allocation */
    /***********************************************************
     * If no old variable, allocate new variable.
     ***********************************************************/
    result = G__allocvariable(result ,para ,varglobal ,varlocal ,paran
			      ,varhash ,item ,varname ,parameter[0][0]);
  }

#ifdef G__OLDIMPLEMENTATION537
/* #ifdef G__ROOT */
  /* THIS PART HAS NOT BEEN PROVEN TO BE SAFE YET */
  G__ASSERT(0==G__decl || 1==G__decl);
  if(G__decl||G__cppconstruct) result.ref = (long)(&var->p[ig15]);
/* #endif G__ROOT */
#endif
  
  G__var_type = 'p';
#ifndef G__OLDIMPLEMENTATION1802
  if(vv!=varname) free((void*)varname);
#endif
  return(result);
}


#ifndef G__OLDIMPLEMENTATION456
/******************************************************************
* G__getpointer2pointer()
*
*
******************************************************************/
static void G__getpointer2pointer(presult,var,ig15,paran)
G__value *presult;
struct G__var_array *var;
int ig15;
int paran;
{
  switch(G__var_type) {
  case 'v':
    switch(var->reftype[ig15]) {
    case G__PARAP2P:
      G__letint(presult,var->type[ig15],*(long*)presult->ref);
      presult->obj.reftype.reftype=G__PARANORMAL;
      break;
    case G__PARAP2P2P:
      G__letint(presult,var->type[ig15],*(long*)presult->ref);
      presult->obj.reftype.reftype=G__PARAP2P;
      break;
    case G__PARANORMAL:
      if(var->paran[ig15]>paran) {
	if(INT_MAX==var->varlabel[ig15][1]) {
	  G__letint(presult,var->type[ig15],presult->ref);
	}
	else {
	  G__letint(presult,var->type[ig15],*(long*)presult->ref);
	}
      }
#ifndef G__OLDIMPLEMENTATION707
      break;
    case G__PARAREFERENCE:
      break;
    default:
      G__letint(presult,var->type[ig15],*(long*)presult->ref);
      presult->obj.reftype.reftype=var->reftype[ig15]-1;
      break;
#else
    default:
      break;
#endif
    }
    break;
  case 'p':
    if(paran<var->paran[ig15]) {
      switch(var->reftype[ig15]) {
      case G__PARANORMAL:
	presult->obj.reftype.reftype = G__PARAP2P;
	break;
      case G__PARAP2P:
      default:
	presult->obj.reftype.reftype = G__PARAP2P2P;
	break;
      }
#ifndef G__OLDIMPLEMENTATION1190
      presult->obj.reftype.reftype += var->paran[ig15]-paran-1;
#endif
    }
    else if(paran==var->paran[ig15]) {
      presult->obj.reftype.reftype = var->reftype[ig15];
    }
    break;
  case 'P':
    /* this part is not precise. Should handle like above 'p' case */
    if(var->paran[ig15]==paran) { /* must be PPTYPE */
      switch(var->reftype[ig15]) {
      case G__PARANORMAL:
	presult->obj.reftype.reftype = G__PARAP2P;
	break;
#ifndef G__OLDIMPLEMENTATION707
      case G__PARAP2P:
	presult->obj.reftype.reftype = G__PARAP2P2P;
	break;
      default:
	presult->obj.reftype.reftype = var->reftype[ig15]+1;
	break;
#else
      case G__PARAP2P:
      default:
	presult->obj.reftype.reftype = G__PARAP2P2P;
	break;
#endif
      }
    }
    break;
  default:
    break;
  }
}
#endif


/******************************************************************
* G__value G__getvariable(item,known2,varglobal,varlocal)
*
*
******************************************************************/
G__value G__getvariable(item,known2,varglobal,varlocal)
char *item;
int *known2;
struct G__var_array *varglobal,*varlocal;
{
  struct G__var_array *var;
  char varname[G__MAXNAME*2];
  char parameter[G__MAXVARDIM][G__ONELINE];
  G__value para[G__MAXVARDIM],result=G__null;
  char result7[G__ONELINE];
  int ig15,paran,ig35,ig25,ary,ig2;
  int lenitem,nest=0;
  int single_quote=0,double_quote=0,paren=0,flag=0;
  int done=0;
  int p_inc;
  int pp_inc;
  long G__struct_offset; /* used to be int */
  char store_var_type;
  /* char ttt[G__LONGLINE]; */
  char *tagname=NULL,*membername=NULL;
  int varhash;
  long address;
  struct G__input_file store_ifile;
  int store_vartype;
  long store_struct_offset;
  int store_tagnum;
#ifndef G__OLDIMPLEMENTATION2146
  int posbracket=0;
  int posparenthesis=0;
#endif

  
#ifdef G__ASM
  /************************************************************
   * If G__asm_exec is set by 'for(', 'while(' or 'do{}while('
   * loop in G__exec_statements(), execute this part and skip
   * parsing string.
   ************************************************************/
  if(G__asm_exec) {
    ig15=G__asm_index;
    paran = G__asm_param->paran;
    
    for(ig35=0;ig35<paran;ig35++) {
      para[ig35] = G__asm_param->para[ig35];
    }
    
    para[paran]=G__null;
    var=varglobal;
    if(varlocal==NULL)
      G__struct_offset =0;
    else
      G__struct_offset=G__store_struct_offset;
    goto G__exec_asm_getvar;
  }
#endif
  
  
  /************************************************************
   * get length of expression.
   ************************************************************/
  lenitem=strlen(item);
  
  
  /************************************************************
   * check '*varname'  or '*(pointer expression)'
   *
   ************************************************************/
  switch(item[0]) {
  case '*': /* value of pointer */
    /**************************************************
     * if '*(pointer expression)' evaluate pointer
     * expression and get data from the address and
     * return. Also *a++, *a--, *++a, *--a
     **************************************************/
    if(item[1]=='(' || '+'==item[1] || '-'==item[1] ||
       '+'==item[lenitem-1] || '-'==item[lenitem-1]) {
      int store_var_type=G__var_type;
      G__var_type='p';
      *known2=1;
      result=G__getexpr(item+1);
      para[0]=G__tovalue(result);
      if('p'!=store_var_type) para[0]=G__toXvalue(para[0],store_var_type);
      return(para[0]);
    }
    /**************************************************
     * if '*varname' 
     *  not sure what to do with G__var_type
     **************************************************/
    if(G__var_type=='p') {
      /* set variable type flag */
      G__var_type = 'v'; 
    }
    else {
      /* set variable type flag */
      G__var_type = toupper(G__var_type); 
    }
    /**************************************************
     * remove '*' from expression. 
     * char *item is modified.
     **************************************************/
    for(ig2=0;ig2<lenitem;ig2++) item[ig2]=item[ig2+1];
    break;
  case '&': /* pointer */
    /**************************************************
     * if '&varname'
     * this case only happens when '&tag.varname'.
     **************************************************/
#ifdef G__DEBUG
    /*
      G__fprinterr(G__serr,"Check error: G__getvariable(%s) G__var_type=%c, should not happen FILE:%s LINE:%d\n"
      ,item,G__var_type,G__ifile.name,G__ifile.line_number);
      */
#endif
    G__var_type = 'P'; /* set variable type flag */
    for(ig2=0;ig2<lenitem;ig2++) item[ig2]=item[ig2+1];
    break;
  case '(':
    /* casting or deeper parenthesis */
    return(G__null);
  }
  
  /************************************************************
   * store G__var_type.  G__var_type is changed when evaluating
   * array index and struct offset. To avoid unexpected change,
   * G__var_type is stored and reset to 'p'. This will be 
   * restored later in this function.
   ************************************************************/
  store_var_type=G__var_type;
  G__var_type='p';
  
  
  /************************************************************
   * sinse expression string might be modifed by removing '*'
   * from char *item, * get length of expression again.
   ************************************************************/
  lenitem=strlen(item);
  
  
  /************************************************************
   * Following while loop checks if the variable is struct or 
   * union member. If unsurrounded '.' or '->' found set flag=1
   * and split tagname and membername.
   *
   *  'tag[].member[]'  or 'tag[]->member[]'
   *        ^                    ^^          set flag=1
   *                                         tagname="tag[]"
   *                                         membername="member[]"
   *
   *  'varname[tag.member]'  or 'varname[func(tag->member,"ab.")]'
   *              ^                              ^          ^
   *   These '.' and '->' doesn't count because they are surrounded 
   *  by parenthesis or quotation.  paren, double_quote and 
   *  single_quote are used to identify if they are surreunded by
   *  (){}[] or ""'' and not set flag=1.
   *
   * C++:
   *  G__getvariable() is called before G__getfunction(). So,
   * C++ member function will be handled in G__getvariable()
   * rather than G__getfunction().
   *
   *  'func().mem'
   *  'mem.func()'
   *  'tag.mem.func()'
   *
   ************************************************************/
  /* strcpy(ttt,item); */
  ig2=0;
  while((ig2<lenitem)&&(flag==0)) {
    
    switch(item[ig2]) {
      
    case '.':
      /*************************************************
       * set flag=1 which means, this is a member of
       * struct of union.
       *************************************************/
      if((paren==0)&&(double_quote==0)&&(single_quote==0)) {
	/**************************************
	 * To get full struct member name path
	 * when not found
	 **************************************/
	strcpy(result7,item);
	result7[ig2++]='\0';
	tagname=result7;
	membername=result7+ig2;
	flag=1;
      }
      break;
      
    case '-':
      /*************************************************
       * set flag=1 which means, this is a member of
       * struct of union.
       *************************************************/
      if((paren==0)&&(double_quote==0)&&(single_quote==0)&&
	 (item[ig2+1]=='>')) {
	/**************************************
	 * To get full struct member name path
	 * when not found
	 **************************************/
	strcpy(result7,item);
	result7[ig2++]='\0';
	result7[ig2++]='\0';
	tagname=result7;
	membername=result7+ig2;
#ifndef G__OLDIMPLEMENTATION1013
	flag=2;
#else
	flag=1;
#endif
      }
      break;
      
    case '\\':
      /*************************************************
       * if backslash, don't check next char
       * This case is for escaping quotation like
       * '"xxx\"xxxx"' , '\''
       *************************************************/
      ig2++;
      break;
      
      /*************************************************
       * set flag for quotation
       *************************************************/
    case '\'':
      if(double_quote==0) {
	single_quote ^= 1;
      }
      break;
    case '\"':
      if(single_quote==0) {
	double_quote ^= 1;
      }
      break;
      
#ifndef G__OLDIMPLEMENTATION2146
    case '[':
      if((single_quote==0)&&(double_quote==0)) {
	if(!paran && !posbracket) posbracket=ig2;
	paren++;
      }
      break;
    case '(':
      if((single_quote==0)&&(double_quote==0)) {
	if(!paran && !posparenthesis) posparenthesis=ig2;
	paren++;
      }
      break;
    case '{':  /* this shouldn't appear */
      if((single_quote==0)&&(double_quote==0)) {
	paren++;
      }
      break;
#else /* 2146 */
    case '[':
      
    case '{':  /* this shouldn't appear */
    case '(':
      if((single_quote==0)&&(double_quote==0)) {
	paren++;
      }
      break;
#endif /* 2146 */
      
      /*************************************************
       * decrement paren for close parenthesis
       *************************************************/
    case '}':  /* this shouldn't appear */
    case ']':
    case ')':
      if((single_quote==0)&&(double_quote==0)) {
	paren--;
      }
      break;
    }
    ig2++;
  }
  
  /************************************************************
   * reset single_quote, double_quote and paren,
   * This part can probably be omitted. paran is not used in
   * following section. single_quote and double_quote are
   * initialized when getting parameters.
   ************************************************************/
  single_quote=0;double_quote=0;paren=0;
  
  
  
  /************************************************************
   * if struct member, do following.
   *
   * C++:
   ************************************************************/
#ifndef G__OLDIMPLEMENTATION1013
  if(flag) {
    result = G__getstructmem(store_var_type
			     ,varname
			     ,membername
			     ,tagname
			     ,known2
			     ,varglobal
			     ,flag
			     );
    return(result);
  }
#else
  if(flag==1) {
    result = G__getstructmem(store_var_type
			     ,varname
			     ,membername
			     ,tagname
			     ,known2
			     ,varglobal
			     );
    return(result);
  }
#endif
  /************************************************************
   * end of struct/union member handling.
   * If variable is struct,union member like 'tag.mem', it should 
   * return value upto this point.
   ************************************************************/
  
  
  /************************************************************
   * varglobal==NULL means, G__getvariable() is called from
   * G__getvariable() or G__letvariable() and G__store_struct_offset 
   * is set by parent G__getvariable().
   *  This is done in different manner in case of loop
   * compilation execution.
   ************************************************************/
  if(varglobal==NULL) {
    G__struct_offset = G__store_struct_offset ;
  }
  else {
    G__struct_offset = 0;
  }
  
  
  /************************************************************
   * Separate variable name scaning char *item upto "[("
   *
   *   'varname[]'
   *           ^
   ************************************************************/
  ig15=0;
  varhash=0;
  while((item[ig15]!='(')&&(item[ig15]!='[')&&(ig15<lenitem)) {
    varname[ig15]=item[ig15];
    varhash+=item[ig15++];
  }
  
  /************************************************************
   * if 'funcname()'
   *            ^
   * OLD IMPLEMENTATION13
   *  return.  This case shouldn't happen because same thing is
   * done when checking struct,union member ship. So, following
   * code is redundant and can be removed.
   * NEW
   *  return if this is a function
   ************************************************************/
  if(item[ig15]=='(') {
    /* if 'func(xxxx)' return */
    return(G__null);
  }

#ifndef G__OLDIMPLEMENTATION2146
  /************************************************************
   * var[x](a,b);
   ************************************************************/
  if(item[ig15]=='[' && posparenthesis) {
    /* G__abortbytecode(); */
    item[posparenthesis] = 0;
    result = G__getvariable(item,known2,varglobal,varlocal);
    if(!known2) return(G__null);
    item[posparenthesis] = '(';
    result = G__pointer2func(&result,(char*)NULL,item+posparenthesis,known2);
    *known2=1;
    return(result);
  }
#endif
  
  /************************************************************
   * set null char to varname. Should be move above.
   ************************************************************/
  varname[ig15++]='\0';
  
  
  /************************************************************
   * if '[expression]' , means no variable name but only array
   * index.  Syntax error. return null.
   ************************************************************/
  if(ig15==1) {
    /* if '[xxxx]' return */
    G__getvariable_error(item);
    *known2=1;
    return(G__null);
  }
  
  /************************************************************
   * scan inside parenthesis '[]' to get array index if any
   *
   ************************************************************/
  paran=0;
  if(ig15<lenitem) {
    /* while((item[ig15]!='!')&&(ig15<lenitem)) { */
    while(ig15<lenitem) {
      ig35 = 0;
      nest=0;
      single_quote=0;
      double_quote=0;
      while(((item[ig15]!=']')||(nest>0)
	     ||(single_quote>0)||(double_quote>0))
	    &&(ig15<lenitem)) {
	switch(item[ig15]) {
	case '"' : /* double quote */
	  if(single_quote==0) {
	    double_quote ^= 1;
	  }
	  break;
	case '\'' : /* single quote */
	  if(double_quote==0) {
	    single_quote ^= 1;
	  }
	  break;
	case '(':
	case '[':
	case '{':
	  if((double_quote==0)&&(single_quote==0)) { 
	    nest++;
	  }
	  break;
	case ')':
	case ']':
	case '}':
	  if((double_quote==0)&&(single_quote==0)) { 
	    nest--;
	  }
	  break;
	}
	parameter[paran][ig35++]=item[ig15++];
      }
      ig15++;
      if((item[ig15]=='[')&&(ig15<lenitem)) ig15++;
      parameter[paran++][ig35]='\0';
      parameter[paran][0]='\0';
    }
  }
  
  /************************************************************
   * evaluate parameter if any
   ************************************************************/
  /* restore base environment */
#ifdef G__ASM
  if(G__asm_noverflow&&paran&&
     (G__store_struct_offset!=G__memberfunc_struct_offset
#ifndef G__OLDIMPLEMENTATION2155
	|| G__do_setmemfuncenv
#endif
      )) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: SETMEMFUNCENV\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp]=G__SETMEMFUNCENV;
    G__inc_cp_asm(1,0);
  }
#endif
  store_struct_offset = G__store_struct_offset;
  store_tagnum = G__tagnum;
  store_vartype = G__var_type;
  G__var_type = 'p';
  G__tagnum = G__memberfunc_tagnum;
  G__store_struct_offset = G__memberfunc_struct_offset;
  /* evaluate parameter */
  for(ig15=0;ig15<paran;ig15++) para[ig15]=G__getexpr(parameter[ig15]);
  /* recover function call environment */
#ifdef G__ASM
  if(G__asm_noverflow&&paran&&
     (G__store_struct_offset!=store_struct_offset
#ifndef G__OLDIMPLEMENTATION2155
      || G__do_setmemfuncenv
#endif
      )) {
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: RECMEMFUNCENV\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp]=G__RECMEMFUNCENV;
    G__inc_cp_asm(1,0);
  }
#endif
  G__store_struct_offset=store_struct_offset;
  G__tagnum=store_tagnum;
  G__var_type = store_vartype;
  
  
  /************************************************************
   * G__var_type was stored in store_var_type right after
   * checking pointer access operators.
   * It is restored at this point. There shouldn't be any
   * recursive call of G__getvariable() or G__getexpr() after
   * this point.
   ************************************************************/
  G__var_type=store_var_type;
  
  
  
  /***********************************************************
   * search old local and global variables.
   *
   ***********************************************************/
  var = G__searchvariable(varname,varhash,varlocal,varglobal,&G__struct_offset
			  ,&store_struct_offset,&ig15,0);

#ifndef G__OLDIMPLEMENTATION986
  if(!var && 
#ifndef G__OLDIMPLEMENTATION1551
     (G__prerun || G__eval_localstatic)
#else
     G__prerun 
#endif
     && G__func_now>=0) {
    char temp[G__ONELINE];
    int itmpx;
#ifndef G__OLDIMPLEMENTATION1012
    if(-1!=G__tagdefining)
      sprintf(temp,"%s\\%x\\%x\\%x",varname,G__func_page,G__func_now
	      ,G__tagdefining);
#else
    if(-1!=G__memberfunc_tagnum) /* questionable */
      sprintf(temp,"%s\\%x\\%x\\%x",varname,G__func_page,G__func_now
	      ,G__memberfunc_tagnum);
#endif
    else
      sprintf(temp,"%s\\%x\\%x" ,varname,G__func_page,G__func_now);
    G__hash(temp,varhash,itmpx);
    var = G__searchvariable(temp,varhash,varlocal,varglobal,&G__struct_offset
			    ,&store_struct_offset,&ig15,0);
#ifndef G__OLDIMPLEMENTATION1119
    if(var) G__struct_offset = 0;
    if(!var && G__getarraydim && !G__IsInMacro()) {
      G__const_noerror=0;
      G__genericerror("Error: Illegal array dimension (Ignore subsequent errors)");
      *known2=1;
      return(G__null);
    }
#endif
  }
#endif
    
    /*************************************************
     * Get value if variable name matchs
     *
     * This is a very very long 'if{}' clause.
     *************************************************/
    if(var) {

#ifndef G__OLDIMPLEMENTATION1259
      result.isconst = var->constvar[ig15];
#ifndef G__OLDIMPLEMENTATION1394
      if(-1!=var->p_typetable[ig15]) {
	result.isconst |= G__newtype.isconst[var->p_typetable[ig15]];
      }
#endif
#endif

#ifndef G__OLDIMPLEMENTATION1119
      if(G__getarraydim && !G__IsInMacro() && 
#ifndef G__OLDIMPLEMENTATION2191
	 'j'!=var->type[ig15]
#else
	 'm'!=var->type[ig15]
#endif
	 &&'p'!=var->type[ig15]&&
	 (0==(G__CONSTVAR&var->constvar[ig15])||
	  (G__DYNCONST&var->constvar[ig15]))) {
        G__const_noerror=0;
	G__genericerror("Error: Non-static-const variable in array dimension");
	G__fprinterr(G__serr," (cint allows this only in interactive command and special form macro which\n");
	G__fprinterr(G__serr,"  is special extension. It is not allowed in source code. Please ignore\n");
	G__fprinterr(G__serr,"  subsequent errors.)\n");
	*known2=1;
	return(G__null);
      }
#endif

/* #define G__OLDIMPLEMENTATION793 */
#ifndef G__OLDIMPLEMENTATION793
      if(var->p[ig15] == 0 && G__struct_offset==0 && 0==G__no_exec_compile) {
	*known2=1;
#ifndef G__OLDIMPLEMENTATION1175
	result = G__null;
#ifndef G__OLDIMPLEMENTATION1737
	result.tagnum = var->p_tagtable[ig15];
	result.typenum = var->p_typetable[ig15];
#endif
        switch(G__var_type) {
	case 'p':
          if(var->paran[ig15]<=paran) {
	    result.type = var->type[ig15];
	    break;
	  }
	case 'P':
	  if(islower(var->type[ig15])) result.type=toupper(var->type[ig15]);
	  else {
	    result.type = var->type[ig15];
	    switch(var->reftype[ig15]) {
	    case G__PARANORMAL:
	      result.obj.reftype.reftype = G__PARAP2P;
	      break;
	    case G__PARAREFERENCE:
	      result.obj.reftype.reftype = var->reftype[ig15];
	      break;
	    default:
	      result.obj.reftype.reftype = var->reftype[ig15]+1;
	      break;
	    }
	  }
	  break;
	}
	return(result);
#else
	return(G__null);
#endif
      }
#endif
      
#ifdef G__ASM
      if(G__asm_noverflow) {
	/************************************
	 * LD_MSTR or LD_VAR instruction
	 ************************************/
	if(G__struct_offset) {
#ifdef G__NEWINHERIT
	  if(G__struct_offset!=store_struct_offset) {
#ifdef G__ASM_DBG
	    if(G__asm_dbg) 
	      G__fprinterr(G__serr,"%3x: ADDSTROS %d\n"
		      ,G__asm_cp,G__struct_offset-store_struct_offset);
#endif
	    G__asm_inst[G__asm_cp]=G__ADDSTROS;
	    G__asm_inst[G__asm_cp+1]=G__struct_offset-store_struct_offset;
	    G__inc_cp_asm(2,0);
	  }
#endif
#ifdef G__ASM_DBG
	  if(G__asm_dbg) 
	    G__fprinterr(G__serr,
		    "%3x: LD_MSTR  %s index=%d paran=%d\n"
		    ,G__asm_cp,item,ig15,paran);
#endif
	  G__asm_inst[G__asm_cp]=G__LD_MSTR;
	  G__asm_inst[G__asm_cp+1]=ig15;
	  G__asm_inst[G__asm_cp+2]=paran;
	  G__asm_inst[G__asm_cp+3]=G__var_type;
	  G__asm_inst[G__asm_cp+4]=(long)var;
	  G__inc_cp_asm(5,0);
#ifdef G__NEWINHERIT
	  if(G__struct_offset!=store_struct_offset) {
#ifdef G__ASM_DBG
	    if(G__asm_dbg) 
	      G__fprinterr(G__serr,"%3x: ADDSTROS %d\n"
		      ,G__asm_cp,-G__struct_offset+store_struct_offset);
#endif
	    G__asm_inst[G__asm_cp]=G__ADDSTROS;
	    G__asm_inst[G__asm_cp+1]= -G__struct_offset+store_struct_offset;
	    G__inc_cp_asm(2,0);
	  }
#endif
	}
	else {
#ifdef G__ASM_DBG
	  if(G__asm_dbg) {
	    if(G__asm_wholefunction && G__ASM_VARLOCAL==store_struct_offset
#ifndef G__OLDIMPLEMENTATION776
	       && G__LOCALSTATIC!=var->statictype[ig15]
#endif
	       ) 
	      G__fprinterr(G__serr,
		      "%3x: LD_LVAR  %s index=%d paran=%d\n"
		      ,G__asm_cp,item,ig15,paran);
	    else 
	      G__fprinterr(G__serr,
		      "%3x: LD_VAR  %s index=%d paran=%d\n"
		      ,G__asm_cp,item,ig15,paran);
	  }
#endif
#ifdef G__ASM_WHOLEFUNC
	  if(G__asm_wholefunction && G__ASM_VARLOCAL==store_struct_offset
#ifndef G__OLDIMPLEMENTATION776
	     && G__LOCALSTATIC!=var->statictype[ig15]
#endif
	     ) 
	    G__asm_inst[G__asm_cp]=G__LD_LVAR;
	  else
	    G__asm_inst[G__asm_cp]=G__LD_VAR;
#else
	  G__asm_inst[G__asm_cp]=G__LD_VAR;
#endif
	  G__asm_inst[G__asm_cp+1]=ig15;
	  G__asm_inst[G__asm_cp+2]=paran;
	  G__asm_inst[G__asm_cp+3]=G__var_type;
	  G__asm_inst[G__asm_cp+4]=(long)var;
	  G__inc_cp_asm(5,0);
	}
      }

#ifndef G__OLDIMPLEMENTATION776
      if(G__no_exec_compile && 
	 (G__CONSTVAR!=var->constvar[ig15] || isupper(var->type[ig15]) ||
	  G__PARAREFERENCE==var->reftype[ig15] || 
	  var->varlabel[ig15][1] || /* 2011 ??? */
	  var->p[ig15]<0x1000) &&
	 'p'!=tolower(var->type[ig15])) {
#else
      if(G__no_exec_compile) {
#endif
#ifdef G__OLDIMPLEMENTATION1121
	if(isupper(var->type[ig15])) {
	  long dmy=0;
	  result.ref = (long)(&dmy);
	  G__getpointer2pointer(&result,var,ig15,paran);
	}
	else {
	  result.obj.reftype.reftype=G__PARANORMAL;
	}
#endif
	*known2=1;
	result.obj.d=0.0;
        switch(var->type[ig15]) {
        case 'd':
        case 'f':
	  break;
#ifndef G__OLDIMPLEMENTATION1001
        case 'T':
	  result.type = 'C';
#endif
	default:
	  result.obj.i=1;
	  break;
	}
	G__returnvartype(&result,var,ig15,paran);
#ifndef G__OLDIMPLEMENTATION1121
	if(isupper(var->type[ig15])) {
	  long dmy=0;
	  result.ref = (long)(&dmy);
	  G__getpointer2pointer(&result,var,ig15,paran);
	}
#endif
	result.tagnum=var->p_tagtable[ig15];
	result.typenum=var->p_typetable[ig15];
	result.ref=G__struct_offset+var->p[ig15];
	G__var_type='p';
	if('u'==tolower(var->type[ig15])) {
#ifndef G__OLDIMPLEMENTATION2145
	  int varparan=var->paran[ig15];
	  if('U'==var->type[ig15]) ++varparan;
	  if(var->reftype[ig15]>G__PARAREFERENCE) {
#ifndef G__OLDIMPLEMENTATION2170
	    varparan += (var->reftype[ig15]%G__PARAREF)-G__PARAP2P+1;
#else
	    varparan += (var->reftype[ig15]%G__PARAREF)-G__PARAP2P;
#endif
	  }
	  for(ig25=0;ig25<paran&&ig25<varparan;ig25++) ;
	  while(ig25<paran&&var->varlabel[ig15][ig25+4]) ++ig25;
	  if(ig25<paran) {
	    G__tryindexopr(&result,para,paran,ig25);
	  }
#else
	  for(ig25=0;ig25<paran&&ig25<var->paran[ig15];ig25++) ;
	  while(ig25<paran&&var->varlabel[ig15][ig25+4]) ++ig25;
	  if(ig25<paran) {
	    G__tryindexopr(&result,para,paran,ig25);
	  }
#endif
	}
	return(result);
      }
      
    G__exec_asm_getvar:
#endif

      /*******************************************************
      * static class/struct member
      *******************************************************/
      if(G__struct_offset && G__LOCALSTATIC==var->statictype[ig15])
	G__struct_offset=0;
      
      done++;
      
      /* Get start pointer of the variable */
      /* ig35= var->varlabel[ig15][0]; */
      
      
      /*************************************************
       * type array[A][B][C][D]
       *
       *  ary = B*C*D which is stored into
       * var->varlabel[var_identity][0]
       *************************************************/
      ary=var->varlabel[ig15][0];
      
      /*************************************************
       *  array[i][j][k][l]
       *
       *  p_inc = B*C*D*i + C*D*j + D*k + l
       *  pp_inc = 
       *************************************************/
      p_inc=0;
      for(ig25=0;ig25<paran&&ig25<var->paran[ig15];ig25++) {
	p_inc += ary*G__int(para[ig25]);
	ary /= var->varlabel[ig15][ig25+2];
      }
      pp_inc=0;
      ary=var->varlabel[ig15][ig25+3];
#ifndef G__OLDIMPLEMENTATION1410
      if(0==ary) ary=1; /* questionable */
#endif
      while(ig25<paran&&var->varlabel[ig15][ig25+4]) {
	pp_inc += ary*G__int(para[ig25]);
	ary /= var->varlabel[ig15][ig25+4];
	++ig25;
      }
      
      /*************************************************
       *  check p_inc doesn't violate segmentation
       *
       *  0 <= p_inc < A*B*C*D = var->varlabel[iden][1]
       *************************************************/
#ifndef G__OLDIMPLEMENTATION805
      if((p_inc<0||p_inc-1>var->varlabel[ig15][1]||
#else
      if((p_inc<0||p_inc>var->varlabel[ig15][1]||
#endif
	 (ig25<paran&&tolower(var->type[ig15])!='u')) &&
         var->reftype[ig15]==G__PARANORMAL){
	G__arrayindexerror(ig15 ,var ,item,p_inc);
	*known2=1;
	return(G__null);
      }
      
      
      /**************************************************/

      /* ON199 */
      
      
      
      /**************************************************
       * return struct and typedef information 
       **************************************************/
      result.tagnum=var->p_tagtable[ig15];
      result.typenum=var->p_typetable[ig15];
      
      result.ref = 0;
      
      
      /**************************************************/
      
      *known2=1;
      
      /**************************************************/
      
#ifdef G__SECURITY
      if(0==G__no_exec_compile&&'v'==G__var_type&&isupper(var->type[ig15])&&
#ifndef G__OLDIMPLEMENTATION475
	 G__PARANORMAL==var->reftype[ig15]&&
	 0==var->varlabel[ig15][1]&&  /* 2011 ??? */
#endif
	 0==(*(long*)(G__struct_offset+var->p[ig15]))) {
	G__reference_error(item);
	return(G__null);
      }
#endif
      G__CHECK(G__SECURE_POINTER_AS_ARRAY 
	       ,(var->paran[ig15]<paran&&isupper(var->type[ig15]))
	       ,return(G__null));
      G__CHECK(G__SECURE_POINTER_REFERENCE
	       ,(isupper(var->type[ig15]&&'E'!=var->type[ig15])||
		 var->paran[ig15]>paran)
	       ,return(G__null));

#ifndef G__OLDIMPLEMENTATION448
      /********************************************************
      * get bit-field value
      ********************************************************/
      if(var->bitfield[ig15] && 'p'==G__var_type) {
	int original,mask,finalval;
	address=G__struct_offset+var->p[ig15];
	original = *(int*)address;
	mask = (1<<var->bitfield[ig15])-1;
	finalval = (original>>var->varlabel[ig15][G__MAXVARDIM-1])&mask;
	G__letint(&result,var->type[ig15],finalval);
	return(result);
      }
#endif

#ifndef G__OLDIMPLEMENTATION1329
      /* This is quite a tricky and unsophisticated way of dealing with
       * typedef char charary[100]; access. NEED IMPROVEMENT sometime */
      if(-1!=var->p_typetable[ig15] && 
	 G__newtype.nindex[var->p_typetable[ig15]] && 
	 'c'==tolower(var->type[ig15])) {
	int typenumx = var->p_typetable[ig15];
	char store_var_type = G__var_type;
	int sizex = G__Lsizeof(G__newtype.name[typenumx]);
	G__var_type = store_var_type;
#ifndef G__OLDIMPLEMENTATION1632
	/* This is still questionable, but should be better than the old
	 * implementation */
	if(var->paran[ig15]>paran) p_inc /= var->varlabel[ig15][0];
#else
	if(p_inc>1) --p_inc; /* questionable */
#endif
	switch(var->type[ig15]) {
	case 'c': /* char */
	  G__GET_VAR(sizex, char ,G__letint,'c','C')
	case 'C': /* char */
	  --paran;
	  G__GET_VAR(sizex, char ,G__letint,'c','C')
	}
      }
#endif

#ifndef G__OLDIMPLEMENTATION1492
      if(1==G__decl && 1==G__getarraydim && 0==G__struct_offset &&
	 var->p[ig15]<100) {
	/* prevent segv in following example. A bit tricky.
	*  void f(const int n) { int a[n]; } */
	G__abortbytecode();
	return(result);
      }
#endif
      
      switch(var->type[ig15]) {
	
      case 'i': /* int */
	G__GET_VAR(G__INTALLOC ,int ,G__letint ,'i' ,'I')
      case 'd': /* double */
	G__GET_VAR(G__DOUBLEALLOC ,double ,G__letdouble ,'d' ,'D')
      case 'c': /* char */
	G__GET_VAR(G__CHARALLOC,char ,G__letint,'c','C')
      case 'b': /* unsigned char */
	G__GET_VAR(G__CHARALLOC,unsigned char ,G__letint,'b','B')
      case 's': /* short int */
	G__GET_VAR(G__SHORTALLOC,short ,G__letint,'s','S')
      case 'r': /* unsigned short int */
	G__GET_VAR(G__SHORTALLOC,unsigned short ,G__letint,'r','R')
      case 'h': /* unsigned int */
	G__GET_VAR(G__INTALLOC,unsigned int ,G__letint,'h','H')
      case 'l': /* long int */
	G__GET_VAR(G__LONGALLOC,long ,G__letint,'l','L')
      case 'k': /* unsigned long int */
	G__GET_VAR(G__LONGALLOC,unsigned long ,G__letint,'k','K')
      case 'f': /* float */
	G__GET_VAR(G__FLOATALLOC,float ,G__letdouble,'f','F')
#ifndef G__OLDIMPLEMENTATION2189
      case 'n':
	G__GET_VAR(G__LONGLONGALLOC ,G__int64,G__letLonglong,'n' ,'N')
      case 'm':
	G__GET_VAR(G__LONGLONGALLOC ,G__uint64,G__letULonglong
		   ,'m' ,'M')
      case 'q':
	G__GET_VAR(G__LONGDOUBLEALLOC ,long double,G__letLongdouble,'q' ,'Q')
#endif
#ifndef G__OLDIMPLEMENTATION1604
      case 'g': /* bool */
#ifdef G__BOOL4BYTE
	G__GET_VAR(G__INTALLOC ,int,G__letbool ,'g' ,'G')
#else
	G__GET_VAR(G__CHARALLOC ,unsigned char ,G__letint ,'g' ,'G')
#endif
#endif

	  /****************************************
	   * G__getvariable()
	   * void pointer is same as char
	   ****************************************/
#ifndef G__OLDIMPLEMENTATION2191
      case '1': /* void */
#else
      case 'Q': /* void */
#endif
      case 'Y': /* void */
      case 'E': /* FILE */
      case 'C': /* char pointer */
	G__GET_PVAR(char,G__letint,long
		    ,tolower(var->type[ig15])
		    ,var->type[ig15])
	break;

#ifndef G__OLDIMPLEMENTATION2189
      case 'N': /* G__int64 */
	G__GET_PVAR(G__int64,G__letLonglong,long
		    ,tolower(var->type[ig15])
		    ,var->type[ig15])
	break;
      case 'M': /* G__uint64 */
	G__GET_PVAR(G__uint64,G__letULonglong,long
		    ,tolower(var->type[ig15])
		    ,var->type[ig15])
	break;
#ifndef G__OLDIMPLEMENTATION2191
      case 'Q': /* long double */
	G__GET_PVAR(long double,G__letLongdouble,long
		    ,tolower(var->type[ig15])
		    ,var->type[ig15])
	break;
#endif
#endif

#ifndef G__OLDIMPLEMENTATION1604
      case 'G': /* bool */
#endif
      case 'B': /* unsigned char pointer */
	G__GET_PVAR(unsigned char,G__letint,long ,'b','B')
	break;

      case 'S': /* short int pointer */
	G__GET_PVAR(short,G__letint,long,'s','S')
	break;

      case 'R': /* unsigned short int pointer */
	G__GET_PVAR(unsigned short,G__letint,long ,'r','R')
	break;

      case 'I': /* int */
	G__GET_PVAR(int,G__letint,long,'i','I')
	break;

      case 'H': /* unsigned int */
	G__GET_PVAR(unsigned int,G__letint,long,'h','H')
	break;

      case 'L': /* long int */
	G__GET_PVAR(long,G__letint,long,'l','L')
	break;

      case 'K': /* unsigned long int */
	G__GET_PVAR(unsigned long,G__letint,long ,'k','K')
	break;

      case 'F': /* float */
	G__GET_PVAR(float,G__letdouble,double,'f','F')
	break;

      case 'D': /* double */
	G__GET_PVAR(double,G__letdouble,double,'d','D')
	break;

      case 'u': /* struct, union */
	G__GET_STRUCTVAR;
	if(ig25<paran) {
	  G__tryindexopr(&result,para,paran,ig25);
	}
	break;

      case 'U': /* struct,uniont pointer */
	G__GET_STRUCTPVAR1(G__struct.size[var->p_tagtable[ig15]]
			  ,G__letint,'u','U')
	G__GET_STRUCTPVAR2(G__struct.size[var->p_tagtable[ig15]]
			  ,G__letint,'u','U')
	if(ig25<paran) {
	  G__tryindexopr(&result,para,paran,ig25);
	}
	break;

#ifndef G__OLDIMPLEMENTATION2191
      case 'j': /* macro */
#else
      case 'm': /* macro */
#endif
#ifndef G__OLDIMPLEMENTATION942
        {
          fpos_t pos;
          struct G__funcmacro_stackelt* store_stack = G__funcmacro_stack;
          G__funcmacro_stack = 0;
          fgetpos (G__ifile.fp, &pos); /* ifile might already be mfp */
          store_ifile = G__ifile;
	  G__ifile.fp=G__mfp;
	  strcpy(G__ifile.name,G__macro);
	  fsetpos(G__ifile.fp,(fpos_t *)var->p[ig15]);
	  G__nobreak=1;
	  result=G__exec_statement();
	  G__nobreak=0;
	  G__ifile=store_ifile;
          fsetpos (G__ifile.fp, &pos);
          G__funcmacro_stack = store_stack;
        }
#else
	store_ifile = G__ifile;
	G__ifile.fp=G__mfp;
	strcpy(G__ifile.name,G__macro);
	fsetpos(G__ifile.fp,(fpos_t *)var->p[ig15]);
	G__nobreak=1;
	result=G__exec_statement();
	G__nobreak=0;
	G__ifile=store_ifile;
#endif
	break;

      case 'a': /* pointer to member function */
	switch(G__var_type) {
	case 'p':
	  if(var->paran[ig15]<=paran) {
	    result.ref = (G__struct_offset+var->p[ig15]+p_inc*G__P2MFALLOC);
	    result.obj.i = result.ref;
	    result.type = 'a';
	    result.tagnum = -1;
	    result.typenum = -1;
	  }
	  else { /* array */
	    G__letint(&result,'A'
		      ,(G__struct_offset+var->p[ig15]+p_inc*G__P2MFALLOC));
	  }
	  break;
	default:
	  G__reference_error(item);
	}
	break;

#ifdef G__ROOT
#if !defined(G__OLDIMPLEMENTATION481)
      case 'Z':
	if(G__GetSpecialObject) {
          store_var_type = G__var_type;
	  result=(*G__GetSpecialObject)(var->varnamebuf[ig15]
					,(void**)var->p[ig15]
					,(void**)(var->p[ig15]+G__LONGALLOC)
					);
         /************************************************************
         * G__var_type was stored in store_var_type just before the
         * call to G__GetSpecialObject which might have recursive
         * calls to G__getvariable() or G__getexpr()
         * It is restored at this point.
         ************************************************************/
          G__var_type=store_var_type;
#ifndef G__FONS81
	  if (0==result.obj.i)
	    *known2 = 0;
	  else
	    var->p_tagtable[ig15] = result.tagnum;
#endif

	  switch(G__var_type) {
	  case 'p':
	    break;
	  case 'v':
	    result.ref = result.obj.i;
	    result.type = tolower(result.type);
	    break;
	  default:
	    G__reference_error(item);
	    break;
	  }
	}
	break;
#elif !defined(G__OLDIMPLEMENTATION467)
      case 'Z':
	G__GET_PVAR(char,G__letint,long ,tolower('U') ,'u')
	  if(result.obj.i) {
	    result=(*G__GetSpecialObject)(var->varnamebuf[ig15]
					  ,(void*)result.obj.i);
	    *(long*)var->p[ig15]=result.obj.i;
	  }
	break;
#endif
#endif

#ifndef G__OLDIMPLEMENTATION904
      case 'T': /* #define xxx "abc" */
	G__GET_PVAR(char,G__letint,long ,tolower(var->type[ig15]) ,'C')
	break;
#endif
	
      default: /* case 'X' automatic variable */
	/* ig35 = var->varpointer[ig35];*/
	
	G__var_type='p';
	
	if(isupper(var->type[ig15])) {
	  G__letdouble(&result,'d',
		       (double)(*(double *)(G__struct_offset+var->p[ig15]+p_inc*G__DOUBLEALLOC)));
	}
	else {
#ifndef G__OLDIMPLEMENTATION1900
	  G__letint(&result,'i',
		    *(long*)(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC));
#else
	  G__letint(&result,'i',
		    *(int*)(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC));
#endif
	}
	break;
      }
    }
  /***********************************************************
   * end of variable name search 'while()' loop.
   *   searched all old local and global variables.
   ***********************************************************/
  
  
  if(done==0) { /* undefined variable */

    /*******************************************
     * if variable name not found, then search
     * for 'this' keywords
     *******************************************/
    if( (*known2=G__getthis(&result,varname,item)) ) {
#ifndef G__OLDIMPLEMENTATION679
      if(0<paran) G__genericerror("Error: syntax error");
#endif
      return(result);
    }
    /*******************************************
     * if variable name not found, then search
     * for function name. The identifier might
     * be pointer to function.
     *******************************************/
    G__var_type='p';
    /*
     *  Maybe type is 'Q' instead of 'C' 
     * but type 'Q'(pointer to function) is not implemented.
     */
    G__search_func(varname,&result);
    if(result.obj.i==0)
      return(G__null);
    else {
#ifndef G__OLDIMPLEMENTATION1876
      *known2=2;
#else
      *known2=1;
#endif
#ifndef G__OLDIMPLEMENTATION549
#ifdef G__ASM
      if(G__asm_noverflow) {
#ifdef G__ASM_DBG
        if(G__asm_dbg) G__fprinterr(G__serr,"%3x: LD '%c' from %x\n"
			       ,G__asm_cp,G__int(result)
			       ,G__asm_dt);
#endif
        G__asm_inst[G__asm_cp]=G__LD;
        G__asm_inst[G__asm_cp+1]=G__asm_dt;
        G__asm_stack[G__asm_dt]=result;
        G__inc_cp_asm(2,1);
      }
#endif /* G__ASM */
#endif /* ON549 */
      return(result);
    }
  }
  else {
#ifndef G__OLDIMPLEMENTATION456
    /**************************************************
    * handling pointer to pointer in G__value
    **************************************************/
    if(isupper(var->type[ig15])
#ifndef G__OLDIMPLEMENTATION802
       && 'P'!=var->type[ig15] && 'O'!=var->type[ig15]
#endif
       ) {
      G__getpointer2pointer(&result,var,ig15,paran);
    }
#endif
    /* return values for non-automatic variable */
    G__var_type='p';
    return(result);
  }
  
}

#ifndef G__OLDIMPLEMENTATION1013
/******************************************************************
* G__IsInMacro()
******************************************************************/
int G__IsInMacro()
{
  if(G__nfile>G__ifile.filenum
     || G__dispmsg >= G__DISPROOTSTRICT
     ) return(0);
  else return(1);
}
#endif

/******************************************************************
* G__value G__getstructmem()
*
* Called by
*    G__getvariable()
*
******************************************************************/
G__value G__getstructmem(store_var_type
			 ,varname       /* buffer, no input */
			 ,membername
			 ,tagname
			 ,known2
			 ,varglobal     /* used as top level flag */
#ifndef G__OLDIMPLEMENTATION1013
			 ,objptr
#endif
			 )
int store_var_type;
char *varname;
char *membername;
char *tagname;
int *known2;
struct G__var_array *varglobal;
#ifndef G__OLDIMPLEMENTATION1013
int objptr;  /* 1 : object , 2 : pointer */
#endif
{
  int store_tagnum;
  long store_struct_offset; /* used to be int */
  int flag;
  G__value result;
#ifndef G__OLDIMPLEMENTATION1259
  G__SIGNEDCHAR_T store_isconst;
#endif
#ifndef G__OLDIMPLEMENTATION1681
  char *px;
#endif
#ifndef G__OLDIMPLEMENTATION2155
  int store_do_setmemfuncenv;
#endif
  
  /****************************************************
   * pointer access operators are removed at the
   * beginning of this function. Add it again to membername
   * because child G__getvariable() needs that information.
   ****************************************************/
  if(store_var_type=='P') {
    sprintf(varname,"&%s",membername);
    strcpy(membername,varname);
  }
  else if(store_var_type=='v') {
    sprintf(varname,"*%s",membername);
    strcpy(membername,varname);
  }
  
  /****************************************************
   * store G__tagnum and G__store_struct_offset to
   * local variable.
   * They will be restored later in this clause.
   ****************************************************/
  store_tagnum=G__tagnum;
  store_struct_offset = G__store_struct_offset;
#ifndef G__OLDIMPLEMENTATION1259
  store_isconst = G__isconst;
#endif
  
#ifdef G__ASM
  /****************************************************
   * loop compilation to store 
   * G__store_struct_offset.
   ****************************************************/
  if(G__asm_noverflow) {
    /*************************************
     * PUSHSTROS
     *************************************/
    G__asm_inst[G__asm_cp] = G__PUSHSTROS;
#ifdef G__ASM_DBG
    if(G__asm_dbg)
      G__fprinterr(G__serr,"%3x: PUSHSTROS\n",G__asm_cp);
#endif
    G__inc_cp_asm(1,0);
  }
#endif
  
  flag = 0;
  
#ifndef G__OLDIMPLEMENTATION1065
  if(
#ifndef G__OLDIMPLEMENTATION1681
     ((px=strchr(tagname,'.')) && isalpha(*(px+1)))
#else
     strchr(tagname,'.') 
#endif
     && (strchr(tagname,'+')|| strchr(tagname,'-')||
	 strchr(tagname,'*')|| strchr(tagname,'/')||
	 strchr(tagname,'%')|| strchr(tagname,'&')||
	 strchr(tagname,'|')|| strchr(tagname,'^')||
	 strchr(tagname,'!') )) {
    result = G__getexpr(tagname);
    if(result.type) flag=1;
  }
  if(flag) { 
  } 
  else 
#endif
  /****************************************************
   * Get entry pointer for the struct,union and
   * store it to a global variable G__store_struct_offset.
   * In any cases, tagname is a varaible of struct,union
   * type.
   ****************************************************/
  if(varglobal) {
    /********************************************
     * If this is a top level like
     *   'tag.subtag.mem'
     *    --- ----------
     * tagname membername
     * get it from G__global and G__p_local
     ********************************************/
    result = G__getvariable(tagname ,&flag ,&G__global ,G__p_local);
  }
  else {
    /********************************************
     * If this is not a top level like
     *   'tag.subtag.mem'
     *        ------ ---
     *       tagname membername
     * get it '&tag' which is G__struct.memvar[].
     *        OR
     * member is referenced in member function
     *  'subtag.mem'
     *   ------ ---
     ********************************************/
    G__incsetup_memvar(G__tagnum);
    result = G__getvariable(tagname ,&flag
			    ,(struct G__var_array*)NULL
			    ,G__struct.memvar[G__tagnum]);
  }
  
  if(flag==0) {
    /***************************************************
     * object not found as variable, 
     * try function which returns struct.
     * Referencing to freed memory area. So, this
     * implementation is bad.
     ***************************************************/
    if(varglobal) {
      result = G__getfunction(tagname,&flag,G__TRYNORMAL);
    }
    else {
#ifndef G__OLDIMPLEMENTATION627
      /* Strange, I do not recall why I did this. Maybe this wasn't necessary
       * from the beginning. */
      /* G__incsetup_memfunc(G__tagnum); */
#endif
      result = G__getfunction(tagname,&flag,G__CALLMEMFUNC);
    }

#ifndef G__OLDIMPLEMENTATION1065
    if(flag==0 && (strchr(tagname,'+')|| strchr(tagname,'-')||
		   strchr(tagname,'*')|| strchr(tagname,'/')||
		   strchr(tagname,'%')|| strchr(tagname,'&')||
		   strchr(tagname,'|')|| strchr(tagname,'^')||
		   strchr(tagname,'!')
#ifndef G__OLDIMPLEMENTATION1351
		   || strstr(tagname,"new ")
#endif
		   )) {
      result = G__getexpr(tagname);
      if(result.type) flag=1;
    }
#endif
    
    /*************************************************
     * if no function like that then return. error 
     * message will be displayed by G__getitem().
     **************************************************/
    if(flag==0) {
#define G__OLDIMPLEMENTATION965
#ifndef G__OLDIMPLEMENTATION965
      G__warnundefined(tagname);
#endif
      return(G__null);
    }
#ifndef G__OLDIMPLEMENTATION407
    else if(G__no_exec_compile&&0==result.obj.i) {
      result.obj.i=G__PVOID;
    }
#endif
  }
  
  G__store_struct_offset = result.obj.i;
  G__tagnum = result.tagnum;
#ifndef G__OLDIMPLEMENTATION1259
  G__isconst = result.isconst;
#endif
  
#ifndef G__OLDIMPLEMENTATION925
  if(G__tagnum<0 
#ifndef G__OLDIMPLEMENTATION1212
     || (isupper(result.type)&&result.obj.reftype.reftype>=G__PARAP2P)
#else
     || (isupper(result.type)&&result.obj.reftype.reftype)
#endif
     ) {
#ifndef G__OLDIMPLEMENTATION1080
    if('~'!=membername[0]) {
#ifndef G__OLDIMPLEMENTATION1103
      if(0==G__const_noerror) {
#endif
	G__fprinterr(G__serr,
		"Error: non class,struct,union object %s used with . or ->"
		,tagname);
	G__genericerror((char*)NULL);
#ifndef G__OLDIMPLEMENTATION1103
      }
#endif
    }
    /* else { ignore destructor to fundamemtal types and pointer } */
#endif
    *known2=1;
#ifndef G__OLDIMPLEMENTATION1259
    G__tagnum=store_tagnum;
    G__store_struct_offset = store_struct_offset;
    G__isconst = store_isconst;
#endif
    return(G__null);
  }
#else
  if(G__tagnum<0) {
    *known2=1;
#ifndef G__OLDIMPLEMENTATION1103
    if(0==G__const_noerror) {
#endif
      G__fprinterr(G__serr,"Error: non class,struct,union object %s used with . or ->"
	      ,tagname);
      G__genericerror((char*)NULL);
#ifndef G__OLDIMPLEMENTATION1103
    }
#endif
#ifndef G__OLDIMPLEMENTATION1259
    G__tagnum=store_tagnum;
    G__store_struct_offset = store_struct_offset;
    G__isconst = store_isconst;
#endif
    return(G__null);
  }
#endif
  else if(0==G__store_struct_offset
#ifndef G__OLDIMPLEMENTATION1151
	  && G__ASM_FUNC_NOP==G__asm_wholefunction
#endif
	  ) {
    *known2=1;
#ifndef G__OLDIMPLEMENTATION1151
    if(0==G__const_noerror) {
      G__fprinterr(G__serr,"Error: illegal pointer to class object %s 0x%lx %d "
	      ,tagname,G__store_struct_offset,G__tagnum);
    }
#else
    G__fprinterr(G__serr,"Error: illegal pointer to class object %s 0x%lx %d "
	    ,tagname,G__store_struct_offset,G__tagnum);
#endif
    G__genericerror((char*)NULL);
    if(G__interactive) {
      G__fprinterr(G__serr,"!!!Input return value by 'retuurn [val]'\n");
#ifndef G__OLDIMPLEMENTATION630
      G__interactive_undefined=1;
      G__pause();
      G__interactive_undefined=0;
#else
      G__pause();
#endif
#ifndef G__OLDIMPLEMENTATION1259
      G__tagnum=store_tagnum;
      G__store_struct_offset = store_struct_offset;
      G__isconst = store_isconst;
#endif
      return(G__interactivereturnvalue);
    }
    else {
#ifndef G__OLDIMPLEMENTATION1259
      G__tagnum=store_tagnum;
      G__store_struct_offset = store_struct_offset;
      G__isconst = store_isconst;
#endif
      return(G__null);
    }
  }

#ifdef G__ASM
  /****************************************************
   * Set struct offset for inner loop compilation.
   ****************************************************/
  if(G__asm_noverflow) {
    /*************************************
     * SETSTROS
     *************************************/
#ifdef G__ASM_DBG
    if(G__asm_dbg)
      G__fprinterr(G__serr,"%3x: SETSTROS\n",G__asm_cp);
#endif
    G__asm_inst[G__asm_cp] = G__SETSTROS;
    G__inc_cp_asm(1,0);
  }
#endif

#ifndef G__OLDIMPLEMENTATION1170
  if('u'==result.type&&2==objptr&&-1!=result.tagnum&&
     strncmp(G__struct.name[result.tagnum],"auto_ptr<",9)==0) {
    int knownx=0;
    char comm[20];
    strcpy(comm,"operator->()");
    result = G__getfunction(comm,&knownx,G__TRYMEMFUNC);
    if(knownx) {
      G__tagnum = result.tagnum;
      G__store_struct_offset = result.obj.i;
#ifdef G__ASM
      if(G__asm_noverflow) {
        /*************************************
         * SETSTROS
         *************************************/
        G__asm_inst[G__asm_cp] = G__SETSTROS;
#ifdef G__ASM_DBG
        if(G__asm_dbg)
          G__fprinterr(G__serr,"%3x: SETSTROS\n",G__asm_cp);
#endif
        G__inc_cp_asm(1,0);
      }
#endif
    }
  }
#endif

#ifndef G__OLDIMPLEMENTATION1013
  /****************************************************
   * check if . or -> matches
   ****************************************************/
#ifndef G__OLDIMPLEMENTATION1265
  if(islower(result.type)&&2==objptr) {
    char bufB[30] = "operator->()";
    int flagB=0;
    int store_tagnumB = G__tagnum;
    long store_struct_offsetB = G__store_struct_offset;
    G__tagnum = result.tagnum;
    G__store_struct_offset = result.obj.i;
    result = G__getfunction(bufB,&flagB,G__TRYMEMFUNC);
    if(flagB) {
      G__tagnum = result.tagnum;
      G__store_struct_offset = result.obj.i;
#ifdef G__ASM
      if(G__asm_noverflow) {
	/*************************************
	 * SETSTROS
	 *************************************/
	G__asm_inst[G__asm_cp] = G__SETSTROS;
#ifdef G__ASM_DBG
	if(G__asm_dbg)
	  G__fprinterr(G__serr,"%3x: SETSTROS\n",G__asm_cp);
#endif
        G__inc_cp_asm(1,0);
      }
#endif
    }
    else {
      G__tagnum = store_tagnumB;
      G__store_struct_offset = store_struct_offsetB;
      if(
	 /* #ifdef G__ROOT */
	 G__dispmsg >= G__DISPROOTSTRICT ||
	 /* #endif */
	 G__ifile.filenum<=G__gettempfilenum()) {
	if(G__dispmsg>=G__DISPWARN) {
	  G__fprinterr(G__serr,"Warning: wrong member access operator '->'");
	  G__printlinenum();
	}
      }
    }
  }
  if(isupper(result.type)&&1==objptr) {
    if(
       /* #ifdef G__ROOT */
       G__dispmsg >= G__DISPROOTSTRICT ||
       /* #endif */
       G__ifile.filenum<=G__gettempfilenum()) {
      if(G__dispmsg>=G__DISPWARN) {
	G__fprinterr(G__serr,"Warning: wrong member access operator '.'");
	G__printlinenum();
      }
    }
  }
#else /* 1265 */
  if(
     /* #ifdef G__ROOT */
     G__dispmsg >= G__DISPROOTSTRICT ||
     /* #endif */
     G__ifile.filenum<=G__gettempfilenum() &&
     ((isupper(result.type)&&1==objptr)||(islower(result.type)&&2==objptr))) {
    if(G__dispmsg>=G__DISPWARN) {
      G__fprinterr(G__serr,"Warning: wrong member access operator '.' or '->'");
      G__printlinenum();
    }
  }
#endif /* 1265 */
#endif /* 1013 */

#ifndef G__OLDIMPLEMENTATION1151
  if(2==objptr && G__initval_eval) G__dynconst = G__DYNCONST;
#endif
  
  /*******************************************************************
   * end of getting struct offset
   *  G__tagnum and G__store_struct_offset should be
   * ready by now.
   ********************************************************************/
  
  
  /****************************************************
   * get variable value.
   *  If membername includes another hierchy of struct
   * ,union member, G__getvariable() will be recursively
   * called from following G__getvariable().
   *
   ****************************************************/
#ifndef G__OLDIMPLEMENTATION2155
  store_do_setmemfuncenv = G__do_setmemfuncenv;
  G__do_setmemfuncenv = 1;
#endif

  G__incsetup_memvar(G__tagnum);
  result=G__getvariable(membername,known2
			,(struct G__var_array*)NULL
			,G__struct.memvar[G__tagnum]);
  
  /****************************************************
   * C++:
   *  if *known2==0                  'tag.func()'
   *                                  --- ------
   *                              tagname membername
   *                 or              'tag.func().func2()'
   * should call G__getfunction() for interpreted member
   * function.
   ****************************************************/
  if(*known2==0) {
#ifndef G__OLDIMPLEMENTATION627
    /* G__incsetup_memfunc(G__tagnum); */
#endif
    if('&'==membername[0]) {
      G__var_typeB='P';
      result=G__getfunction(membername+1,known2,G__CALLMEMFUNC);
      G__var_typeB='p';
    }
    else if('*'==membername[0]) {
      G__var_typeB='v';
      result=G__getfunction(membername+1,known2,G__CALLMEMFUNC);
      G__var_typeB='p';
    }
    else {
      result=G__getfunction(membername,known2,G__CALLMEMFUNC);
#ifndef G__OLDIMPLEMENTATION695
      result = G__toXvalue(result,store_var_type);
#endif
    }
  }
  
#ifndef G__OLDIMPLEMENTATION2155
  G__do_setmemfuncenv = store_do_setmemfuncenv;
#endif
  
  /****************************************************
   * restore G__tagnum and G__store_struct_offset 
   * because evaluation is finished.
   ****************************************************/
  G__tagnum=store_tagnum;
  G__store_struct_offset = store_struct_offset;
#ifndef G__OLDIMPLEMENTATION1259
  G__isconst = store_isconst;
#endif
  
#ifdef G__ASM
  /****************************************************
   * pop struct offset for inner loop compilation.
   ****************************************************/
  if(G__asm_noverflow) {
    /*************************************
     * POPSTROS
     *************************************/
    G__asm_inst[G__asm_cp] = G__POPSTROS;
#ifdef G__ASM_DBG
    if(G__asm_dbg)
      G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp);
#endif
    G__inc_cp_asm(1,0);
  }
#endif
  
  /****************************************************
   * return result
   ****************************************************/
  return(result);
}
  
/******************************************************************
* G__value G__letstructmem()
*
* Called by
*   G__letvariable()
*
******************************************************************/
G__value G__letstructmem(store_var_type
			 ,varname
			 ,membername
			 ,tagname
			 ,varglobal
			 ,expression
#ifndef G__OLDIMPLEMENTATION1013
			 ,objptr
#endif
			 )
int store_var_type;
char *varname;
char *membername;
char *tagname;
struct G__var_array *varglobal;
G__value expression;
#ifndef G__OLDIMPLEMENTATION1013
int objptr;  /* 1 : object , 2 : pointer */
#endif
{
  int store_tagnum;
  long store_struct_offset; /* used to be long */
  int flag;
  G__value result;
#ifndef G__OLDIMPLEMENTATION1259
  G__SIGNEDCHAR_T store_isconst;
#endif
#ifndef G__OLDIMPLEMENTATION2155
  int store_do_setmemfuncenv;
#endif
  
  /* add pointer operater if necessary */
  if(store_var_type=='P') {
    sprintf(varname,"&%s",membername);
    strcpy(membername,varname);
  }
  if(store_var_type=='v') {
    sprintf(varname,"*%s",membername);
    strcpy(membername,varname);
  }
  
  store_tagnum=G__tagnum;
  store_struct_offset = G__store_struct_offset;
#ifndef G__OLDIMPLEMENTATION1259
  store_isconst = G__isconst;
#endif
  
#ifdef G__ASM
  if(G__asm_noverflow) {
    /*************************************
     * PUSHSTROS
     *************************************/
    G__asm_inst[G__asm_cp] = G__PUSHSTROS;
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: PUSHSTROS\n",G__asm_cp);
#endif
    G__inc_cp_asm(1,0);
  }
#endif
  
  flag = 0;
  
  /****************************************************
   * Get entry pointer for the struct,union and
   * store it to a global variable G__store_struct_offset.
   * In any cases, tagname is a varaible of struct,union
   * type.
   ****************************************************/
  if(varglobal) {
    /********************************************
     * If this is a top level like
     *   'tag.subtag.mem'
     *    --- ----------
     * tagname membername
     * get it from G__global and G__p_local
     ********************************************/
    result = G__getvariable(tagname ,&flag ,&G__global ,G__p_local);
  }
  else {
    /********************************************
     * If this is not a top level like
     *   'tag.subtag.mem'
     *        ------ ---
     *       tagname membername
     * get it '&tag' which is G__struct.memvar[].
     *        OR
     * member is referenced in member function
     *  'subtag.mem'
     *   ------ ---
     ********************************************/
    G__incsetup_memvar(G__tagnum);
    result = G__getvariable(tagname ,&flag ,(struct G__var_array*)NULL
			    ,G__struct.memvar[G__tagnum]);
  }
  G__store_struct_offset = result.obj.i;
  G__tagnum = result.tagnum;
#ifndef G__OLDIMPLEMENTATION1259
  G__isconst = result.isconst;
#endif
  
#ifndef G__OLDIMPLEMENTATION965
  if(flag==0) {
    G__warnundefined(tagname);
#ifndef G__OLDIMPLEMENTATION1259
    G__tagnum=store_tagnum;
    G__store_struct_offset = store_struct_offset;
    G__isconst = store_isconst;
#endif
    return(G__null);
  }
#endif
  
  if(G__tagnum<0) {
#ifndef G__OLDIMPLEMENTATION1259
    G__tagnum=store_tagnum;
    G__store_struct_offset = store_struct_offset;
    G__isconst = store_isconst;
#endif
    return(G__null);
  }
  else if(0==G__store_struct_offset) {
#ifndef G__OLDIMPLEMENTATION1151
    if(0==G__const_noerror) {
      G__fprinterr(G__serr,"Error: illegal pointer to class object %s 0x%lx %d "
	      ,tagname,G__store_struct_offset,G__tagnum);
    }
#else
    G__fprinterr(G__serr,"Error: illegal pointer to class object %s 0x%lx %d "
	    ,tagname,G__store_struct_offset,G__tagnum);
#endif
    G__genericerror((char*)NULL);
#ifndef G__OLDIMPLEMENTATION1259
    G__tagnum=store_tagnum;
    G__store_struct_offset = store_struct_offset;
    G__isconst = store_isconst;
#endif
    return(expression);
  }
    
  /*************************************
   * object not found, return 
   * G__getitem() will display error
   * message
   *************************************/
  if(flag==0) {
#ifndef G__OLDIMPLEMENTATION1259
    G__tagnum=store_tagnum;
    G__store_struct_offset = store_struct_offset;
    G__isconst = store_isconst;
#endif
    return(G__null);
  }

#ifdef G__ASM
  if(G__asm_noverflow) {
    /*************************************
     * SETSTROS
     *************************************/
    G__asm_inst[G__asm_cp] = G__SETSTROS;
#ifdef G__ASM_DBG
    if(G__asm_dbg)
      G__fprinterr(G__serr,"%3x: SETSTROS\n",G__asm_cp);
#endif
    G__inc_cp_asm(1,0);
  }
#endif

#ifndef G__OLDIMPLEMENTATION1170
  if('u'==result.type&&2==objptr&&-1!=result.tagnum&&
     strncmp(G__struct.name[result.tagnum],"auto_ptr<",9)==0) {
    int knownx=0;
    char comm[20];
    strcpy(comm,"operator->()");
    result = G__getfunction(comm,&knownx,G__TRYMEMFUNC);
    if(knownx) {
      G__tagnum = result.tagnum;
      G__store_struct_offset = result.obj.i;
#ifdef G__ASM
      if(G__asm_noverflow) {
        /*************************************
         * SETSTROS
         *************************************/
        G__asm_inst[G__asm_cp] = G__SETSTROS;
#ifdef G__ASM_DBG
        if(G__asm_dbg)
          G__fprinterr(G__serr,"%3x: SETSTROS\n",G__asm_cp);
#endif
        G__inc_cp_asm(1,0);
      }
#endif
    }
  }
#endif
  
#ifndef G__OLDIMPLEMENTATION1013
  /****************************************************
   * check if . or -> matches
   ****************************************************/
#ifndef G__OLDIMPLEMENTATION1265
  if(islower(result.type)&&2==objptr) {
    char bufB[30] = "operator->()";
    int flagB=0;
    int store_tagnumB = G__tagnum;
    long store_struct_offsetB = G__store_struct_offset;
    G__tagnum = result.tagnum;
    G__store_struct_offset = result.obj.i;
    result = G__getfunction(bufB,&flagB,G__TRYMEMFUNC);
    if(flagB) {
      G__tagnum = result.tagnum;
      G__store_struct_offset = result.obj.i;
#ifdef G__ASM
      if(G__asm_noverflow) {
	/*************************************
	 * SETSTROS
	 *************************************/
	G__asm_inst[G__asm_cp] = G__SETSTROS;
#ifdef G__ASM_DBG
	if(G__asm_dbg)
	  G__fprinterr(G__serr,"%3x: SETSTROS\n",G__asm_cp);
#endif
        G__inc_cp_asm(1,0);
      }
#endif
    }
    else {
      G__tagnum = store_tagnumB;
      G__store_struct_offset = store_struct_offsetB;
      if(
	 /* #ifdef G__ROOT */
	 G__dispmsg >= G__DISPROOTSTRICT ||
	 /* #endif */
	 G__ifile.filenum<=G__gettempfilenum()) {
	if(G__dispmsg>=G__DISPWARN) {
	  G__fprinterr(G__serr,"Warning: wrong member access operator '->'");
	  G__printlinenum();
	}
      }
    }
  }
  if(isupper(result.type)&&1==objptr) {
    if(
       /* #ifdef G__ROOT */
       G__dispmsg >= G__DISPROOTSTRICT ||
       /* #endif */
       G__ifile.filenum<=G__gettempfilenum()) {
      if(G__dispmsg>=G__DISPWARN) {
	G__fprinterr(G__serr,"Warning: wrong member access operator '.'");
	G__printlinenum();
      }
    }
  }
#else
  if(
     /* #ifdef G__ROOT */
     G__dispmsg >= G__DISPROOTSTRICT ||
     /* #endif */
     (G__ifile.filenum<=G__gettempfilenum() &&
     ((isupper(result.type)&&1==objptr)||(islower(result.type)&&2==objptr)))){
    if(G__dispmsg>=G__DISPWARN) {
      G__fprinterr(G__serr,"Warning: wrong member access operator '.' or '->'");
      G__printlinenum();
    }
  }
#endif
#endif
  
  
  /****************************************************
   * assign variable value.
   *  If membername includes another hierchy of struct
   * ,union member, G__letvariable() will be recursively
   * called from following G__letvariable().
   *
   ****************************************************/
#ifndef G__OLDIMPLEMENTATION2155
  store_do_setmemfuncenv = G__do_setmemfuncenv;
  G__do_setmemfuncenv = 1;
#endif

  G__incsetup_memvar(G__tagnum);
  result=G__letvariable(membername,expression
			,(struct G__var_array*)NULL
			,G__struct.memvar[G__tagnum]);
  
#ifndef G__OLDIMPLEMENTATION2155
  G__do_setmemfuncenv = store_do_setmemfuncenv;
#endif

  /****************************************************
   * restore G__tagnum and G__store_struct_offset 
   * because evaluation is finished.
   ****************************************************/
  G__tagnum=store_tagnum;
  G__store_struct_offset = store_struct_offset;
#ifndef G__OLDIMPLEMENTATION1259
  G__isconst = store_isconst;
#endif
  
#ifdef G__ASM
  if(G__asm_noverflow) {
    /*************************************
     * POPSTROS
     *************************************/
    G__asm_inst[G__asm_cp] = G__POPSTROS;
#ifdef G__ASM_DBG
    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp);
#endif
    G__inc_cp_asm(1,0);
  }
#endif
  
  /****************************************************
   * return result
   ****************************************************/
  return(result);
}

/******************************************************************
* G__letstruct()
*
* Called by
*    G__letvariable()
*
*  MUST CORRESPOND TO G__ASSIGN_VAR
*
* Note:
*  G__letstruct and G__classassign in struct.c have special handling
* of operator=(). When interpretation, overloaded assignment operator
* is recognized in G__letstruct and G__classassign functions. They
* set appropreate environment (G__store_struct_offset, G__tagnum) 
* and try to call operator=(). It may not be required to search for
* non member operator=() function, so, some part of these functions
* could be omitted.
*  
******************************************************************/
void G__letstruct(result
	     ,p_inc
	     ,var
	     ,ig15
	     ,item
	     ,paran
	     ,G__struct_offset
	     )
G__value *result;
int p_inc;
struct G__var_array *var;
int ig15;
char *item;
int paran;
long G__struct_offset; /* used to be int */
{
  char ttt[G__ONELINE];
  char result7[G__ONELINE];
  int ig2;
  long store_struct_offset; /* used to be int */
  int largestep=0;
  int store_tagnum;
  G__value para;
  int store_prerun=0,store_debug=0,store_step=0;
  long store_asm_inst=0;
  long addr;

  if(G__asm_exec) {
#ifndef G__OLDIMPLEMENTATION1340
    void *p1 = (void*)(G__struct_offset+var->p[ig15]+p_inc*G__struct.size[var->p_tagtable[ig15]]);
    void *p2 = (void*)result->obj.i ;
    size_t size = (size_t)G__struct.size[var->p_tagtable[ig15]];
    memcpy(p1,p2,size);
#else
    memcpy((void *)(G__struct_offset+var->p[ig15]+p_inc*G__struct.size[var->p_tagtable[ig15]])
	   ,(void *)(G__int(*result))
	   ,(size_t)G__struct.size[var->p_tagtable[ig15]]);
#endif
    return;
  }
  
  switch(G__var_type) {
  case 'p': /* return value */
    if(var->paran[ig15]<=paran) {
      /* value , struct,union */
      
      /* store flags */
      store_prerun = G__prerun;
      G__prerun = 0;
      if(store_prerun) {
	store_debug = G__debug;
	store_step = G__step;
	G__debug = G__debugtrace;
	G__step = G__steptrace;
	G__setdebugcond();
      }
      else {
	if(G__breaksignal) {
	  G__break=0;
	  G__setdebugcond();
	  if(G__pause()==3) {
	    if(G__return==G__RETURN_NON) {
	      G__step=0;
	      G__setdebugcond();
	      largestep=1;
	    }
	  }
	  if(G__return>G__RETURN_NORMAL) 
	    return;
	}
      }
      
#ifndef G__OLDIMPLEMENTATION738
      if(-1!=result->tagnum&&('u'==result->type||'i'==result->type)) {
#else
      if(result->type=='u') {
#endif
	if(result->obj.i)
	  sprintf(ttt,"(%s)(%ld)"
		  ,G__fulltagname(result->tagnum,1),result->obj.i);
	else
	  sprintf(ttt,"(%s)%ld"
		  ,G__fulltagname(result->tagnum,1),result->obj.i);
      }
      else {
	G__valuemonitor(*result,ttt);
      }
      
      G__ASSERT(0==G__decl || 1==G__decl);
      if(G__decl) {
	/**************************************
	 * copy constructor
	 **************************************/
	sprintf(result7,"%s(%s)",G__struct.name[var->p_tagtable[ig15]],ttt);
	
	store_tagnum = G__tagnum;
	G__tagnum = var->p_tagtable[ig15];
	store_struct_offset = G__store_struct_offset;
	G__store_struct_offset=(G__struct_offset+var->p[ig15]+p_inc*G__struct.size[var->p_tagtable[ig15]]);
	
	if(G__dispsource) {
	  G__fprinterr(G__serr,
		  "\n!!!Calling constructor 0x%lx.%s for declaration"
		  ,G__store_struct_offset ,result7);
	}
#ifdef G__SECURITY
	G__castcheckoff=1;
#endif
	
	ig2=0;
	G__decl=0;
#ifndef G__OLDIMPLEMENTATION1073
	G__oprovld=1;
#endif
#ifndef G__OLDIMPLEMENTATION1373
	{ /* rather big change, risk5, 2000/8/26 */
	  int store_cp = G__asm_cp;
	  int store_dt = G__asm_dt;
	  G__getfunction(result7,&ig2 ,G__TRYCONSTRUCTOR);
	  if(ig2 && G__asm_noverflow) {
	    int x;
	    G__asm_dt = store_dt;
	    if(G__LD_FUNC==G__asm_inst[G__asm_cp-5]) {
	      for(x=0;x<5;x++) 
		G__asm_inst[store_cp+x] = G__asm_inst[G__asm_cp-5+x];
	      G__asm_cp = store_cp + 5;
	    }
	    else if(G__LD_IFUNC==G__asm_inst[G__asm_cp-8]) {
	      for(x=0;x<8;x++) 
		G__asm_inst[store_cp+x] = G__asm_inst[G__asm_cp-8+x];
	      G__asm_cp = store_cp + 8;
	    }
	  }
#ifndef G__OLDIMPLEMENTATION1641
	  else if(!ig2 && 'U'==result->type) {
	    G__fprinterr(G__serr,"Error: Constructor %s not found",result7);
	    G__genericerror((char*)NULL);
	  }
#endif
	}
#else
	G__getfunction(result7,&ig2 ,G__TRYCONSTRUCTOR);
#endif
#ifndef G__OLDIMPLEMENTATION1073
	G__oprovld=0;
	if(G__asm_wholefunction && 0==ig2) {
	  G__asm_gen_stvar(G__struct_offset,ig15,paran,var,item
			   ,G__ASM_VARLOCAL,G__var_type
#ifndef G__OLDIMPLEMENTATION2089
                           ,result
#endif
                           );
	}
#endif
	G__decl=1;
	
	G__store_struct_offset = store_struct_offset;
	G__tagnum = store_tagnum;
      }
      else {
	
#ifdef G__ASM
	if(G__asm_noverflow) {
	  store_asm_inst=G__asm_inst[G__asm_cp-5];
	  if(G__ST_VAR==store_asm_inst) 
	      G__asm_inst[G__asm_cp-5] = G__LD_VAR;
#ifndef G__OLDIMPLEMENTATION2060
	  else if(G__ST_LVAR==store_asm_inst) 
	      G__asm_inst[G__asm_cp-5] = G__LD_LVAR;
#endif
	  else 
	      G__asm_inst[G__asm_cp-5] = G__LD_MSTR;
	  G__asm_inst[G__asm_cp] = G__PUSHSTROS;
	  G__asm_inst[G__asm_cp+1] = G__SETSTROS;
	  G__inc_cp_asm(2,0);
#ifdef G__ASM_DBG
	  if(G__asm_dbg) {
	    G__fprinterr(G__serr,"ST_VAR or ST_MSTR replaced with LD_VAR or LD_MSTR(2)\n");
	    G__fprinterr(G__serr,"%3x: PUSHSTROS\n",G__asm_cp-2);
	    G__fprinterr(G__serr,"%3x: SETSTROS\n",G__asm_cp-1);
	  }
#endif
	}
	G__oprovld = 1;
#endif
	/**************************************
	 * operator=() overloading
	 **************************************/
	/* searching for member function */
	sprintf(result7,"operator=(%s)" ,ttt);
	
	store_tagnum = G__tagnum;
	G__tagnum = var->p_tagtable[ig15];
	store_struct_offset = G__store_struct_offset;
	G__store_struct_offset=(G__struct_offset+var->p[ig15]+p_inc*G__struct.size[var->p_tagtable[ig15]]);
	
	ig2=0;
	para=G__getfunction(result7,&ig2 ,G__TRYMEMFUNC);

#ifndef G__OLDIMPLEMENTATION656
	if(0==ig2 && G__tagnum!=result->tagnum) {
	  /**************************************
	   * copy constructor
	   **************************************/
	  long store_globalvarpointer;
	  sprintf(result7,"%s(%s)",G__struct.name[G__tagnum],ttt);
	  if(G__CPPLINK==G__struct.iscpplink[G__tagnum]) {
	    G__abortbytecode();
	    store_globalvarpointer = G__globalvarpointer;
	    G__globalvarpointer = G__store_struct_offset;
	    G__getfunction(result7,&ig2 ,G__TRYCONSTRUCTOR);
	    G__globalvarpointer = store_globalvarpointer;
	  }
	  else {
	    G__getfunction(result7,&ig2 ,G__TRYCONSTRUCTOR);
	  }
	}
#endif
	
	G__store_struct_offset = store_struct_offset;
	G__tagnum = store_tagnum;
	
	/* searching for global function */
	if(ig2==0) {
#ifdef G__ASM
	  if(G__asm_noverflow) {
	    G__inc_cp_asm(-2,0); 
#ifdef G__ASM_DBG
	    if(G__asm_dbg) {
	      G__fprinterr(G__serr,"PUSHSTROS,SETSTROS cancelled");
	      G__printlinenum();
	    }
#endif
	  }
#endif
	  addr=(G__struct_offset+var->p[ig15]
		+p_inc*G__struct.size[var->p_tagtable[ig15]]);
#ifndef G__OLDIMPLEMENTATION979
	  if(addr<0)
	    sprintf(result7 ,"operator=((%s)(%ld),%s)"
		    ,G__fulltagname(var->p_tagtable[ig15],1),addr ,ttt);
	  else
	    sprintf(result7 ,"operator=((%s)%ld,%s)"
		    ,G__fulltagname(var->p_tagtable[ig15],1),addr ,ttt);
#else
	  if(addr<0)
	    sprintf(result7 ,"operator=((%s)(%ld),%s)"
		    ,G__struct.name[var->p_tagtable[ig15]] ,addr ,ttt);
	  else
	    sprintf(result7 ,"operator=((%s)%ld,%s)"
		    ,G__struct.name[var->p_tagtable[ig15]] ,addr ,ttt);
#endif
	  para=G__getfunction(result7,&ig2 ,G__TRYNORMAL);
	}
#ifdef G__ASM
	else {
	  if(G__asm_noverflow) {
#ifdef G__ASM_DBG
	    if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp);
#endif
	    G__asm_inst[G__asm_cp] = G__POPSTROS;
	    G__inc_cp_asm(1,0); 
	  }
	}
	G__oprovld = 0;
#endif
      }
      
      /* restore flags */
      if(store_prerun) {
	G__debug = store_debug;
	G__step = store_step;
	G__setdebugcond();
      }
      else {
	if(largestep) {
	  G__step=1;
	  G__setdebugcond();
	  largestep=0;
	}
      }
      G__prerun = store_prerun;
      
      if(ig2) { /* in case overloaded = or constructor is found */
	*result = para;
      }
      else { /* in case no overloaded = or constructor, memberwise copy */

#ifndef G__OLDIMPLEMENTATION674
        /* try conversion operator for class object */
        if('u'==result->type && -1!=result->tagnum) {
          int tagnum = var->p_tagtable[ig15];
          if(G__class_conversion_operator(tagnum,result,ttt)) {
            long pdest
                =G__struct_offset+var->p[ig15]+p_inc*G__struct.size[tagnum];
            G__classassign(pdest,tagnum,*result);
	    return;
          }
        }
#endif /* ON674 */

#ifdef G__ASM
	if(G__asm_noverflow
#ifndef G__OLDIMPLEMENTATION1073
	   && store_asm_inst
#endif
	   ) {
#ifdef G__ASM_DBG
	  if(G__asm_dbg) G__fprinterr(G__serr,"ST_VAR or ST_MSTR recovered no_exec_compile=%d\n",G__no_exec_compile);
#endif
	  G__asm_inst[G__asm_cp-5]=store_asm_inst;
	}
#ifndef G__OLDIMPLEMENTATION1751
	if(G__no_exec_compile || (G__globalcomp && G__int(*result)==0) ) { 
	  /* with -c-1 or -c-2 option */
	  return;
        }
#else
	if(G__no_exec_compile) return;
#endif
#endif /* of G__ASM */

	if(result->tagnum==var->p_tagtable[ig15]) {
	  memcpy((void *)(G__struct_offset+var->p[ig15]+p_inc*G__struct.size[var->p_tagtable[ig15]])
		 ,(void *)(G__int(*result))
		 ,(size_t)G__struct.size[var->p_tagtable[ig15]]);
	}
#ifndef G__OLDIMPLEMENTATION668
	else if(-1!=(addr=G__ispublicbase(var->p_tagtable[ig15]
					 ,result->tagnum,0))) {
          int tagnum=var->p_tagtable[ig15];
          long pdest=G__struct_offset+var->p[ig15]+p_inc*G__struct.size[tagnum];
	  memcpy((void *)(pdest) ,(void *)(G__int(*result)+addr)
		 ,(size_t)G__struct.size[tagnum]);
	  if(-1!=G__struct.virtual_offset[tagnum]) 
	    *(long*)(pdest+G__struct.virtual_offset[tagnum]) = tagnum;
	}
#endif
	else {
	  G__fprinterr(G__serr,"Error: Assignment to %s type incompatible " ,item);
	  G__genericerror((char*)NULL);
	}
      }
      break;
    } /* end of if(var->varlabel[ig15][paran+1]==0) */
    else if(G__funcheader && paran==0 && isupper(result->type)) {
      /* K&R style 'type a[]' initialization */
      if(var->p[ig15]!=G__PINVALID&&G__COMPILEDGLOBAL!=var->statictype[ig15])
	free((void*)var->p[ig15]);
      var->p[ig15] = result->obj.i; 
      var->statictype[ig15]=G__COMPILEDGLOBAL;
      break;
    }
  default :
    if('u'==G__var_type) {
      G__letint(result,'u',G__struct_offset+var->p[ig15]);
      result->tagnum = var->p_tagtable[ig15];
      result->typenum = var->p_typetable[ig15];
      break;
    }
#ifndef G__OLDIMPLEMENTATION751
    if('v'==G__var_type) {
      char refopr[G__MAXNAME];
      long store_struct_offsetX = G__store_struct_offset;
      int store_tagnumX = G__tagnum;
      int done=0;
#ifndef G__OLDIMPLEMENTATION1662
      int store_var_type = G__var_type;
      G__var_type='p';
#endif
      G__store_struct_offset 
	= (long)(G__struct_offset+(var->p[ig15])
		 +p_inc*G__struct.size[var->p_tagtable[ig15]]);
      G__tagnum = var->p_tagtable[ig15];
#ifndef G__OLDIMPLEMENTATION1661
#ifdef G__ASM
      if(G__asm_noverflow) {
	if(G__struct_offset) {
	  G__asm_inst[G__asm_cp]=G__LD_MSTR;
	}
	else {
	  G__asm_inst[G__asm_cp]=G__LD_VAR;
	}
#ifdef G__ASM_DBG
	if(G__asm_dbg) 
	  G__fprinterr(G__serr,"%3x: LD_VAR  %s index=%d paran=%d\n"
		       ,G__asm_cp,var->varnamebuf[ig15],ig15,0);
#endif
	G__asm_inst[G__asm_cp+1]=ig15;
	G__asm_inst[G__asm_cp+2]=paran;
	G__asm_inst[G__asm_cp+3]='p';
	G__asm_inst[G__asm_cp+4]=(long)var;
	G__inc_cp_asm(5,0);
#ifdef G__ASM_DBG
	if(G__asm_dbg) {
	  G__fprinterr(G__serr,"%3x: PUSHSTROS\n",G__asm_cp-2);
	  G__fprinterr(G__serr,"%3x: SETSTROS\n",G__asm_cp-1);
	}
#endif
	G__asm_inst[G__asm_cp] = G__PUSHSTROS;
	G__asm_inst[G__asm_cp+1] = G__SETSTROS;
	G__inc_cp_asm(2,0);
      }
#endif
#endif /* 1661 */
      strcpy(refopr,"operator*()");
      para=G__getfunction(refopr,&done,G__TRYMEMFUNC);
      G__tagnum = store_tagnumX;
      G__store_struct_offset = store_struct_offsetX;
#ifndef G__OLDIMPLEMENTATION1662
      G__var_type=store_var_type;
#endif
#ifndef G__OLDIMPLEMENTATION1661
#ifdef G__ASM
      if(G__asm_noverflow) {
#ifdef G__ASM_DBG
	if(G__asm_dbg) {
	  G__fprinterr(G__serr,"%3x: POPSTROS\n",G__asm_cp-2);
	}
#endif
	G__asm_inst[G__asm_cp] = G__POPSTROS;
	G__inc_cp_asm(1,0);
      }
#endif
#endif /* 1661 */
      if(0==done) {
	G__assign_error(item,result);
      }
      G__letVvalue(&para,*result);
    }
    else
      G__assign_error(item,result);
#else
    G__assign_error(item,result);
#endif
    break;
  }
}

/******************************************************************
* G__letstructp()
*
* Called by
*   G__letvariable()
*
*  MUST CORRESPOND TO G__ASSIGN_PVAR
******************************************************************/
void G__letstructp(result
		   ,G__struct_offset
		   ,ig15
		   ,p_inc
		   ,var
		   ,paran
		   ,item
		   ,para
		   ,pp_inc
		   )
G__value result;
long G__struct_offset; /* used to be int */
int ig15;
int p_inc;
struct G__var_array *var;
int paran;
char *item;
G__value *para;
int pp_inc;
{
  long address;
  /* int ig25; */
  int baseoffset;

#ifndef G__OLDIMPLEMENTATION1383
  if(INT_MAX==var->varlabel[ig15][1] && 'v'==G__var_type &&
     0==p_inc && 0== paran) {
    /* Trick  f(A **x) { *x=0; } as f(AA **x) { x[0]=0; } */
    G__var_type = 'p';
    paran = 1;
  }
#endif
  
  switch(G__var_type) {
  case 'v': 
    switch(var->reftype[ig15]) {
    case G__PARANORMAL: 
      if(G__no_exec_compile) 
	G__classassign(G__PVOID ,var->p_tagtable[ig15] ,result);
      else 
	G__classassign((*(long*)(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC))
		     ,var->p_tagtable[ig15] ,result);
      break;
  case G__PARAP2P:
      if(var->paran[ig15]<paran) {
	if(G__no_exec_compile) {
	  G__classassign(G__PVOID ,var->p_tagtable[ig15] ,result);
	}
	else {
	  address = G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC;
	  G__classassign((*(((long*)(*(long *)address))+pp_inc))
			 ,var->p_tagtable[ig15] ,result);
	}
      }
      else {
#ifndef G__OLDIMPLEMENTATION940
	if(0==G__no_exec_compile)
	  *(long *)(*(long *)(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC))
	    = G__int(result);
#else
	*(long *)(*(long *)(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC))
	  = G__int(result);
#endif
      }
      break;
    }
    break;
  case 'p': /* var = expr; assign to pointer variable */
    if(var->paran[ig15]<=paran) {
      if(var->paran[ig15]<paran) {
#ifndef G__OLDIMPLEMENTATION638
	address = G__struct_offset+var->p[ig15] +p_inc*G__LONGALLOC;
	if(G__PARANORMAL==var->reftype[ig15]) {
	  if(G__no_exec_compile) address=G__PVOID;
	  else {
	    address
	      =(*(long*)address)+pp_inc*G__struct.size[var->p_tagtable[ig15]];
	  }
	  /* to be fixed */
	  G__classassign(address,var->p_tagtable[ig15],result);
	}
	else if(var->paran[ig15]==paran-1) {
	  if(0==G__no_exec_compile)
	    *(((long*)(*(long *)address))+pp_inc) = G__int(result);
	}
	else if(var->paran[ig15]==paran-2) {
	  if(0==G__no_exec_compile) {
	    address=(long)((long*)(*(long *)(address))+para[0].obj.i);
	    if(G__PARAP2P==var->reftype[ig15]) {
	      address=(long)((*((long*)(address)))
			     +para[1].obj.i
			     *G__struct.size[var->p_tagtable[ig15]]);
	      G__classassign(address,var->p_tagtable[ig15],result);
	    }
	    else if(G__PARAP2P<var->reftype[ig15]) {
	      address=(long)((long*)(*(long *)(address))+para[1].obj.i);
	      *(long *)address = G__int(result);
	    }
	  }
	}
	else if(var->paran[ig15]==paran-3) {
	  if(0==G__no_exec_compile) {
	    address=(long)((long*)(*(long *)(address))+para[0].obj.i);
	    address=(long)((long*)(*(long *)(address))+para[1].obj.i);
	    if(G__PARAP2P2P==var->reftype[ig15]) {
	      address=(long)((*((long*)(address)))
			     +para[2].obj.i
			     *G__struct.size[var->p_tagtable[ig15]]);
	      G__classassign(address,var->p_tagtable[ig15],result);
	    }
	    else if(G__PARAP2P2P<var->reftype[ig15]) {
	      address=(long)((long*)(*(long *)(address))+para[2].obj.i);
	      *(long *)address = G__int(result);
	    }
	  }
	}
	else {
	  if(0==G__no_exec_compile)
	    G__classassign(
		 ((*(((long*)(*(long *)address))+para[0].obj.i))+para[1].obj.i)
		 ,var->p_tagtable[ig15],result);
	}
	  
#else
	/* class X *a[10]; a[1][2]=b; */
	/* class X *a;     a[1]=b;    */
	address = G__struct_offset+var->p[ig15]
	  +p_inc*G__LONGALLOC;
	if(G__no_exec_compile)
	  address=G__PVOID;
	else
	  address
	    =(*(long*)address)+pp_inc*G__struct.size[var->p_tagtable[ig15]];
	/* to be fixed */
	G__classassign(address,var->p_tagtable[ig15],result);
#endif
      }
      else {
	/* check if tagnum matches. 
	 * If unmatch, check for class inheritance.
	 * If derived class pointer is assigned to 
	 * base class pointer, add offset and assign.
	 */
	if(G__no_exec_compile) {
#ifndef G__OLDIMPLEMENTATION2081
	  /* Base class casting at this position does not make sense.
           * because ST_VAR is already generated in G__asm_gen_stvar */
#endif
          return;
	}
#ifndef G__OLDIMPLEMENTATION1006
	if('U'!=result.type && 'Y'!= result.type && 0!=result.obj.i
	   && ('u'!=result.type||result.obj.i==G__p_tempbuf->obj.ref)) {
	  G__assign_error(item,&result);
	  return;
	}
#endif
	if(var->p_tagtable[ig15] == result.tagnum 
	   || 0==result.obj.i || 'Y'==result.type) { /* checked */
	  *(long *)(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC)
	    = G__int(result);
	}
#ifdef G__VIRTUALBASE
	else if(-1 != (baseoffset=G__ispublicbase(var->p_tagtable[ig15]
						  ,result.tagnum
						  ,result.obj.i))) {
#else
	else if(-1 != (baseoffset=G__ispublicbase(var->p_tagtable[ig15]
						  ,result.tagnum))) {
#endif
	  *(long *)(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC)
	    = G__int(result)+baseoffset;
	}
	else {
	  G__assign_error(item,&result);
	}
      }
    }
    else { 
      /* K&R pointer to pointer 'type **a,type *a[]' 
       * initialization */
#ifndef G__OLDIMPLEMENTATION457
      if(var->p[ig15]!=G__PINVALID&&G__COMPILEDGLOBAL!=var->statictype[ig15])
	free((void*)var->p[ig15]);
#else
      if(var->p[ig15]!=G__PINVALID) free((void*)var->p[ig15]);
#endif
      var->p[ig15] = result.obj.i;
      var->statictype[ig15]=G__COMPILEDGLOBAL;
    }
    break;
  default:
    G__assign_error(item,&result);
    break;
  }
}



/******************************************************************
* G__returnvartype()
*
* 1998 may fix only one case. Other cases may need to be fided as well.
******************************************************************/
void G__returnvartype(presult,var,ig15,paran)
G__value *presult;
struct G__var_array *var;
int ig15;
int paran;
{
  presult->type=var->type[ig15];
  if(isupper(presult->type)) presult->obj.reftype.reftype = var->reftype[ig15];

  switch(presult->type) {
  case 'p':
  case 'x':
    presult->type='i';
    return;
  case 'P':
  case 'X':
    presult->type='d';
    return;
#ifndef G__OLDIMPLEMENTATION2191
  case 'j': /* questionable */
#else
  case 'm':
#endif
    G__abortbytecode();
    presult->type='i';
    return;
  }

  if(islower(var->type[ig15])) {
    switch(G__var_type) {
    case 'p':
      if(var->paran[ig15]<=paran) {
	presult->type=var->type[ig15];
      }
      else {
	presult->type=toupper(var->type[ig15]);
      }
      break;
    case 'P':
      presult->type=toupper(var->type[ig15]);
      break;
    default: /* 'v' */
      presult->type=var->type[ig15];
      break;
    }
  }
  else {
    switch(G__var_type) {
    case 'v':
      presult->type=tolower(var->type[ig15]);
      break;
    case 'P':
      presult->type=toupper(var->type[ig15]);
      break;
    default: /* 'p' */
      if(var->paran[ig15]==paran) {
	presult->type=var->type[ig15];
      }
      else if(var->paran[ig15]<paran) {
#ifndef G__OLDIMPLEMENTATION1998
	int pointlevel;
	int reftype = var->reftype[ig15];
	if(!reftype) reftype=1;
	pointlevel = reftype - paran;
	switch(pointlevel) {
	case 0:
	  presult->type=tolower(var->type[ig15]);
	  presult->obj.reftype.reftype = G__PARANORMAL;
	  break;
	case 1:
	  presult->type=toupper(var->type[ig15]);
	  presult->obj.reftype.reftype = G__PARANORMAL;
	  break;
	default:
	  presult->type=toupper(var->type[ig15]);
	  presult->obj.reftype.reftype = pointlevel;
	  break;
	}
#else
	switch(var->reftype[ig15]) {
        case G__PARANORMAL:
	  presult->type=tolower(var->type[ig15]);
	  break;
	case G__PARAP2P:
	  presult->obj.reftype.reftype = G__PARANORMAL;
	  break;
	case G__PARAP2P2P:
	  presult->obj.reftype.reftype = G__PARAP2P;
	  break;
#ifndef G__OLDIMPLEMENTATION707
	default:
	  presult->obj.reftype.reftype = var->reftype[ig15]-1;
	  break;
#endif
	}
#endif
      }
      else {
	presult->type=toupper(var->type[ig15]);
      }
      break;
    }
  }
  return;
}


/******************************************************************
* G__value G__allocvariable()
*
* Called by
*   G__letvariable()
*
******************************************************************/
G__value G__allocvariable( /* expression, */
			  result
			  ,para
			  ,varglobal
			  ,varlocal
			  ,paran
			  ,varhash
			  ,item
			  ,varname
			  ,parameter00
			  )
/* G__value expression; */
G__value result;
G__value para[];
struct G__var_array *varglobal,*varlocal;
int paran;
int varhash;
char *item;
char *varname;
int parameter00;
{
  struct G__var_array *var;
  char ttt[G__ONELINE];
  int ig25;
  int ary;
  int ig15;
  int p_inc;
  int i;
  int autoobjectflag=0;
  int store_tagnum=0;
  int store_typenum=0;
  int store_globalvarpointer=0;
  int store_var_type=0;
#ifndef G__OLDIMPLEMENTATION448
  int bitlocation=0;
#endif
  
  /***************************************************
   * New variable 
   *
   * global or local
   ***************************************************/
  if(G__p_local!=NULL) { /* equal to G__prerun==0 */
    var=varlocal;
  }
  else { /* equal to G__prerun==1 */
    var=varglobal;
  }
  if(!var) {
    G__fprinterr(G__serr,"Error: Illegal assignment to %s",item);
    G__genericerror((char*)NULL);
    return(result);
  }
  if(G__def_struct_member && G__PARAREFERENCE==G__reftype) {
#ifndef G__OLDIMPLEMENTATION1088
    if(G__NOLINK==G__globalcomp
#ifndef G__OLDIMPLEMENTATION1241
       && -1!=G__def_tagnum && 'n'!=G__struct.type[G__def_tagnum]
#endif
       ) {
#endif
      G__genericerror("Limitation: Reference member not supported. Please use pointer");
      return(result);
#ifndef G__OLDIMPLEMENTATION1088
    }
    else if(G__access==G__PUBLIC
#ifndef G__OLDIMPLEMENTATION1241
	    && -1!=G__def_tagnum && 'n'!=G__struct.type[G__def_tagnum]
#endif
	    ) {
      G__fprinterr(G__serr,"Limitation: Reference member not accessible from the interpreter");
      G__printlinenum();
      G__access=G__PRIVATE;
    }
#endif
  }
  if('U'==G__var_type&&'U'==result.type&&
     -1==G__ispublicbase(G__tagnum,result.tagnum,G__STATICRESOLUTION2)
#define G__OLDIMPLEMENTATION1213
#ifdef G__OLDIMPLEMENTATION1213
     && -1==G__ispublicbase(result.tagnum,G__tagnum,G__STATICRESOLUTION2)
#endif
     ) {
    G__fprinterr(G__serr,"Error: Illegal initialization of pointer, wrong type %s"
	    ,G__type2string(result.type ,result.tagnum ,result.typenum
			    ,result.obj.reftype.reftype,0));
    G__genericerror((char*)NULL);
    /* G__genericerror("Error: Illegal initialization of pointer"); */
    return(result);
  }

  while(var->next) var=var->next;

#ifndef G__OLDIMPLEMENTATION1089
  if(0==G__definemacro&&G__NOLINK==G__globalcomp&&'p'==G__var_type
#ifndef G__OLDIMPLEMENTATION1097
     && G__automaticvar
#endif
     ) {
    if(-1==result.tagnum) {
      G__var_type=result.type;
#ifndef G__PHILIPPE21
      if(0==G__const_noerror) {
        /* to follow the example of other places .. Should printlinenum
           be replaced by G__genericerror ? */
#endif
	if(G__dispmsg>=G__DISPWARN) {
	  G__fprinterr(G__serr,
	  "Warning: Automatic variable %s is allocated",item);
	  G__printlinenum();
	}
#ifndef G__PHILIPPE21
      }
#endif
    }
    else {
#else
  if(0==G__definemacro&&G__NOLINK==G__globalcomp&&'p'==G__var_type&&
     -1!=result.tagnum) {
#endif
#ifndef G__OLDIMPLEMENTATION1372
#ifndef G__OLDIMPLEMENTATION1470
    autoobjectflag=1;
#endif
    if(G__IsInMacro()) {
      /* undeclared variable assignment of class/struct will create 
       * a global object of pointer or reference */
#ifdef G__OLDIMPLEMENTATION1470
      autoobjectflag=1;
#endif
      if(G__p_local) {
	var=varglobal;
	while(var->next) var=var->next;
      }
    }
#else
    /* undeclared variable assignment of class/struct will create 
     * a global object of pointer or reference */
    autoobjectflag=1;
    if(G__p_local) {
      var=varglobal;
      while(var->next) var=var->next;
    }
#endif
    store_var_type = G__var_type;
    store_tagnum = G__tagnum;
    store_typenum = G__typenum;
    store_globalvarpointer = G__globalvarpointer;
    G__var_type = result.type;
    G__tagnum = result.tagnum;
    G__typenum = -1;
    if(isupper(result.type)) {
      /* a = new T(init);  a is a pointer */
#ifndef G__ROOT
      G__fprinterr(G__serr,"Warning: Automatic variable %s* %s is allocated"
		   ,G__fulltagname(result.tagnum,1),item);
      G__printlinenum();
#endif
      G__reftype = G__PARANORMAL;
    }
    else {
      if(result.tagnum==G__p_tempbuf->obj.tagnum &&
	 G__templevel==G__p_tempbuf->level) {
	/* a = T(init); a is an object */
	G__globalvarpointer = result.obj.i;
#ifndef G__ROOT
	G__fprinterr(G__serr,"Warning: Automatic variable %s %s is allocated"
		     ,G__fulltagname(result.tagnum,1),item);
	G__printlinenum();
#endif
	G__reftype = G__PARANORMAL;
	G__p_tempbuf->obj.obj.i=0;
	G__pop_tempobject();
      }
      else {
	/* T b; 
	 * a = b; a is a reference type */
	G__reftype = G__PARAREFERENCE;
	if(G__ASM_FUNC_NOP==G__asm_wholefunction) 
	  G__fprinterr(G__serr,
		  "Error: Illegal Assignment to an undeclared symbol %s"
		  ,item);
	G__genericerror((char*)NULL);
	G__tagnum = store_tagnum;
	G__typenum = store_typenum;
	G__globalvarpointer = store_globalvarpointer;
	G__var_type = store_var_type;
	return(result);
      }
    }
  }
#ifndef G__OLDIMPLEMENTATION1089
  }
#endif

#ifndef G__OLDIMPLEMENTATION448
  /***************************************************
   * get bitfield location
   * this part must be done before new *var allocation
   ***************************************************/
  if(G__bitfield) {
    ig15 = var->allvar;
    if(0==ig15 || 0==var->bitfield[ig15-1]) {
      /* the first element in the bit-field */
      bitlocation = 0;
    }
    else {
      bitlocation=var->varlabel[ig15-1][G__MAXVARDIM-1]
	+var->bitfield[ig15-1];
      if((int)(G__INTALLOC*8)<bitlocation+G__bitfield) {
	bitlocation = 0;
      }
    }
    if(-1==G__bitfield) {
      /* unsigned int a : 4;
       * unsigned int   : 0; <- in case of this, new allocation unit for b
       * unsigned int b : 3;              */
      G__bitfield = G__INTALLOC*8-bitlocation;
    }
  }
#endif

  /***************************************************
   * New variable 
   *
   * Assign index or allocate new variable array
   ***************************************************/
  
  if(var->allvar<G__MEMDEPTH) {
    /****************************************
     * If no overflow just assign index
     ****************************************/
    ig15 = var->allvar;
  }
  else {
    /****************************************
     * If overflow allocate another vararray
     ****************************************/
    var->next = 
      (struct G__var_array *)malloc(sizeof(struct G__var_array)) ;
#ifdef G__OLDIMPLEMENTATION1776_YET
    memset(var->next,0,sizeof(struct G__var_array));
#endif

#ifndef G__OLDIMPLEMENTATION2038
    var->next->enclosing_scope = (struct G__var_array*)NULL;
    var->next->inner_scope = (struct G__var_array**)NULL;
#endif
    
    /***************************************
     * assign local variable var to new array
     * This will be confusing, be careful
     ***************************************/
#ifndef G__FONS80
    var->next->tagnum = var->tagnum;
#endif
    var = var->next;
    
    /***************************************
     * Initialize the new varaible array
     ***************************************/
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
    ig15=0;
  }
  
  
  /***************************************************
   * Class of the variable
   *
   *  auto, file scope static, function scope static
   ***************************************************/
  
  var->statictype[var->allvar] = G__AUTO; /* auto */

#ifndef G__OLDIMPLEMENTATION1552
  if(2==G__decl_obj) { /* this is set in decl.c G__initstructary */
    var->statictype[var->allvar] = G__AUTOARYDISCRETEOBJ; /* auto */
  }
#endif


#ifndef G__OLDIMPLEMENTATION612
  /***************************************************
  * if namespace, set G__static_alloc, it should be
  * reset in G__define_var()
  ***************************************************/
  if(G__def_struct_member && -1!=G__tagdefining && 
     'n'==G__struct.type[G__tagdefining]
#ifndef G__OLDIMPLEMENTATION689
     && 'p'!=tolower(G__var_type) /* macro is global */
#endif
     ) {
#ifndef G__OLDIMPLEMENTATION1284
      if(G__PVOID!=G__globalvarpointer
#ifndef G__OLDIMPLEMENTATION1845
	 && !G__cppconstruct
#endif
	 ) 
	var->statictype[var->allvar] = G__COMPILEDGLOBAL;
      else 
	var->statictype[var->allvar] = G__LOCALSTATIC;
#else
      G__static_alloc=G__LOCALSTATIC;
#endif
  }
#endif
  
  if(G__static_alloc) {
    if(G__p_local!=NULL) { /* equal to G__prerun==0 */
      G__varname_now=varname;
      /**************************************
       * Function scope static variable 
       * No real malloc(), get pointer from
       * global variable array which is suffixed
       * as varname\funcname.
       * Also, static class/struct member
       ***************************************/
      var->statictype[var->allvar] = G__LOCALSTATIC;
    }
    
    else { /* equal to G__prerun==1 */
      
      if(G__func_now != -1) {
	/***************************************
	 * Function scope static variable
	 * Variable allocated to global variable
	 * array named varname\funcname. The
	 * variable can be exclusively accessed
	 * with in a specific function.
	 ****************************************/
#ifdef G__NEWINHERIT
	if(-1!=G__p_ifunc->tagnum) /* questionable */
	  sprintf(ttt,"%s\\%x\\%x\\%x" ,varname,G__func_page,G__func_now
		  ,G__p_ifunc->tagnum);
	else
	  sprintf(ttt,"%s\\%x\\%x" ,varname,G__func_page,G__func_now);
#else
	if(-1!=G__p_ifunc->basetagnum[G__func_now]) /* questionable */
	  sprintf(ttt,"%s\\%x\\%x\\%x" ,varname,G__func_page,G__func_now
		  ,G__p_ifunc->basetagnum[G__func_now]);
	else
	  sprintf(ttt,"%s\\%x\\%x" ,varname,G__func_page,G__func_now);
#endif
#ifndef G__OLDIMPLEMENTATION2156
	if(G__cintv6) {
	  if(0==G__const_noerror||result.isconst&G__STATICCONST) 
	    strcpy(varname,ttt);
	  else sprintf(varname,"_%s",ttt);
	}
	else
	  strcpy(varname,ttt);
#else
	strcpy(varname,ttt);
#endif
	/* BUG FIX, 25Feb94, ig15 was used */
	G__hash(ttt,varhash,ig25)
	var->statictype[var->allvar] = G__LOCALSTATICBODY;
      }
#ifndef G__OLDIMPLEMENTATION1091
      else if(G__nfile<G__ifile.filenum) {
	if(G__dispmsg>=G__DISPWARN) {
	  G__fprinterr(G__serr,"Warning: 'static' ignored in '{ }' style macro");
	  G__printlinenum();
	}
	G__static_alloc=0;
      }
#endif
      else {
	/****************************************
	 * File scope static variable
	 * file index is stored as statictype[].
	 * The variable can be exclusively accessed
	 * with in an indexed file.
	 *****************************************/
	var->statictype[var->allvar] = G__ifile.filenum;
      }
    }
  }
  
  /***************************************************
   * member access control
   *
   ***************************************************/
  if(G__def_struct_member
#ifndef G__OLDIMPLEMENTATION1885
     || (G__enumdef /* && G__dispmsg>=G__DISPROOTSTRICT */)
#endif
     ) var->access[var->allvar] = G__access;
  else                     var->access[var->allvar] = G__PUBLIC;

#ifndef G__NEWINHERIT
  var->isinherit[var->allvar] = 0;
#endif

#ifdef G__OLDIMPLEMENTATION448
  var->bitfield[var->allvar] = G__bitfield;
  G__bitfield=0;
#endif

#ifndef G__OLDIMPLEMENTATION1700
  if(-1==var->tagnum)
    var->globalcomp[var->allvar] = G__default_link?G__globalcomp:G__NOLINK;
  else
    var->globalcomp[var->allvar] = G__globalcomp;
#else
  var->globalcomp[var->allvar] = G__globalcomp;
#endif
  
  /***************************************************
   * store maximum array parameters into var->varlabel[][]
   * 
   * typedef xxx type[a][b];
   * type array[A][B][C][D]
   *
   *   var->varlabel[var_identity][1]=a;
   *   var->varlabel[var_identity][2]=b;
   *   var->varlabel[var_identity][3]=A;
   *   var->varlabel[var_identity][4]=B;
   *   var->varlabel[var_identity][5]=C;
   *   var->varlabel[var_identity][6]=D;
   ***************************************************/
  if(G__typedefnindex) {
    parameter00='1';
  }
  for(ig25=0;ig25<G__typedefnindex;ig25++) {
    var->varlabel[var->allvar][ig25+1]=G__typedefindex[ig25];
  }
  for(i=0;i<paran;i++) {
    var->varlabel[var->allvar][++ig25]=G__int(para[i]);
  }
  paran=ig25;
  
  /***************************************************
   * fill zero to rest of the var->varlabel[][]
   * 
   *   var->varlabel[var_identity][5]=0;
   *   var->varlabel[var_identity][6]=0;
   *                 .
   *   var->varlabel[var_identity][G__MAXVARDIM-1]=0;
   ***************************************************/
  while(ig25<G__MAXVARDIM-1) var->varlabel[var->allvar][++ig25]=0;
  
  /**************************************************
   * if we have an array like
   *
   *  type array[A][B][C];
   *
   * ary = B*C;       Assume A=B=C=2;
   *
   * p_inc=    0          1          2          3          4
   * -----------------------------------------------------------
   *  array[0][0][0], [0][0][1], [0][1][0], [0][1][1], [1][0][0]
   *                                                      ^
   * if array[x][0][0] is accessed                    (B*C)*x
   *
   * this number ary should be stored into 
   *  var->varlabel[var_identity][0]
   *************************************************/
  ig25=2;
  ary=1;
  while(var->varlabel[ig15][ig25]!=0) ary *= var->varlabel[ig15][ig25++] ;
  var->varlabel[ig15][0] = ary;
  
  /* Get p_inc */
  /************************************************************
   *  type array[A][B][C];
   *
   * assign var->varlabel[var_identity][1] to 
   * maximum address increment which is A*B*C = A*ary
   ************************************************************/
  p_inc= ary*var->varlabel[ig15][1];
  if(p_inc==0) p_inc=1; /* this is special for this case */
  var->varlabel[ig15][1] = p_inc-1;

  /************************************************************
   * pointer to array reimplementation
   ************************************************************/
  var->varlabel[ig15][ig25] = 1;
  var->paran[ig15]=paran;
  ig25=paran+4;
  
  if(isupper(G__var_type)) {
    i=0;
    ary=1;
    while(G__p2arylabel[i]) {
      ary *= G__p2arylabel[i];
      var->varlabel[ig15][ig25+i] = G__p2arylabel[i];
      ++i;
    }
    var->varlabel[ig15][ig25+i] = 1;
    var->varlabel[ig15][ig25-1] = ary;
  }
  G__p2arylabel[0]=0;
  
  /************************************************************
   * Finally
   * type array[A][B][C][D]
   *   var->varlabel[var_identity][0]=B*C*D; or 1;
   *   var->varlabel[var_identity][1]=A*B*C*D-1;
   *   var->varlabel[var_identity][2]=B;
   *   var->varlabel[var_identity][3]=C;
   *   var->varlabel[var_identity][4]=D;
   *   var->varlabel[var_identity][5]=1;
   *
   * if type (*pary[A][B][C][D])[x][y][z]
   *   var->varlabel[var_identity][6]=x*y*z or 1;
   *   var->varlabel[var_identity][7]=x;
   *   var->varlabel[var_identity][8]=y;
   *   var->varlabel[var_identity][9]=z;
   *   var->varlabel[var_identity][10]=1;
   *   var->varlabel[var_identity][11]=0;
   *
   ***********************************************************/
  
  
  if(paran>0) {

#ifndef G__OLDIMPLEMENTATION2093
    if(paran>=G__MAXVARDIM) {
      G__fprinterr(G__serr
                  ,"Limitation: Cint can handle only upto %d dimention array"
                  ,G__MAXVARDIM-1);
      G__genericerror((char*)NULL);
      return(result);
    }
#endif

    /************************************************************
     * In case of 'func(type a[][B][C][D],type b[A][B][C])' 
     * and ANSI style function parameter and runtime
     * When prerun, result.type is null.
     *    formal argument of function runtime ANSI
     ***********************************************************/
    if(G__funcheader && '\0'!=result.type) {
      G__ASSERT(G__globalvarpointer == G__PVOID);
      G__globalvarpointer = result.obj.i;
      result.type = '\0';
#ifndef G__OLDIPMLEMENTATION877
      if(G__asm_wholefunction) {
	G__abortbytecode();
	G__genericerror((char*)NULL);
      }
#endif
    }
    /************************************************************
     * In case of 'type a[][B][C][D];'
     *    formal argument of function at prerun ANSI
     *    formal argument of function at prerun(globalcomp) K&R
     *    formal argument of function runtime ANSI
     *    formal argument of function runtime K&R
     *    global compiled variable assignment
     ***********************************************************/
    if(parameter00=='\0') {
#ifndef G__OLDIPMLEMENTATION877
      if(G__asm_wholefunction && (0==G__funcheader||1==paran)
#ifndef G__OLDIPMLEMENTATION1617
	 && 0==G__static_alloc
#endif
	 ) {
      /* Tried, but does not work */
      /* if(1==paran && G__asm_wholefunction && G__funcheader) {  */
#else
      if(1==paran && G__asm_wholefunction) {
#endif
	var->paran[ig15] = paran = 0;
	var->varlabel[ig15][0] = 1;
	var->varlabel[ig15][1] = 0;
	if(isupper(G__var_type)) {
	  switch(G__reftype) {
	  case G__PARANORMAL: G__reftype = G__PARAP2P; break;
	  case G__PARAP2P: G__reftype = G__PARAP2P2P; break;
#ifndef G__OLDIMPLEMENTATION707
	  default: ++G__reftype; break;
#endif
	  }
	}
	else G__var_type = toupper(G__var_type);
      } 
      else {
	if(G__PVOID==G__globalvarpointer) {
	  if('\0'!=result.type) {
	    G__globalvarpointer = result.obj.i;
	    /* bug fix char str[]="abcd"; */
	    result.type='\0';
	  }
	  else {
	    /* formal argument prerun or auto sized 
	     * array. Prevent allocating huge memory */
	    /* bug fix in local static automatic array */
	    if(0==G__static_alloc||1==G__prerun) G__globalvarpointer=G__PINVALID;
	  }
	}
	var->varlabel[ig15][1] = INT_MAX;
      }
    }
  } /* end of paran>0 */
  
  /************************************************************
   * If G__globalvarpointer is set, set statictype as COMPILEGLOBAL
   * so that this memory area won't be freed when destruction.
   ***********************************************************/
  /* REMIND! Not sure about testing LOCALSTATIC here. This relates to
   * precompiling static member variables. Static members are classifed
   * as G__LOCALSTATIC (above this function) in both interpreted and
   * precompiled case. I'm not 100% sure that there isn't any other
   * case falling into LOCALSTATIC upto this point but need to change
   * to COMPILEDGLOBAL.
   */
  if(G__globalvarpointer!=G__PVOID && 0==G__cppconstruct &&
     G__LOCALSTATIC!=var->statictype[var->allvar]) {
    var->statictype[var->allvar] = G__COMPILEDGLOBAL;
  }

#ifdef G__FONS_COMMENT
  /*****************************************************************
   * store comment information specific to CERN's ROOT
   *****************************************************************/
  if(G__setcomment) {
#ifndef G__OLDIMPLEMENTATION469
    var->comment[var->allvar].p.com = G__setcomment;
    var->comment[var->allvar].filenum = -2;
#else
    var->comment[var->allvar].p.pos = (fpos_t)0;
    var->comment[var->allvar].p.com = G__setcomment;
    var->comment[var->allvar].filenum = -1;
#endif
  }
  else {
    var->comment[var->allvar].p.com = (char*)NULL;
#ifdef G__OLDIMPLEMENTATION469
    var->comment[var->allvar].p.pos = (fpos_t)0;
#endif
    var->comment[var->allvar].filenum = -1;
  }
#endif

#ifdef G__VARIABLEFPOS
  var->filenum[var->allvar] = G__ifile.filenum;
  var->linenum[var->allvar] = G__ifile.line_number;
#endif

#ifndef G__OLDIMPLEMENTATION1940
 {
   char* pp;
   pp = strchr(varname,'-');
   if(!pp) pp = strchr(varname,'+');
   if(pp) {
     G__fprinterr(G__serr,"Error: Variable name has bad character '%s'"
	,varname);
     G__genericerror((char*)NULL);
   }
 }
#endif
  
  /*****************************************************************
   * set variable information to the table
   *****************************************************************/
#ifndef G__OLDIMPLEMENTATION1543
  G__savestring(&var->varnamebuf[var->allvar],varname);
#else /* 1543 */
#ifndef G__OLDIMPLEMENTATION853
  if(strlen(varname)>G__MAXNAME-1) {
    strncpy(var->varnamebuf[var->allvar],varname,G__MAXNAME-1);
    var->varnamebuf[var->allvar][G__MAXNAME-1]=0;
    G__fprinterr(G__serr,"Limitation: Variable name length overflow strlen(%s)>%d"
	    ,varname,G__MAXNAME-1);
    G__genericerror((char*)NULL);
  }
  else
    strcpy(var->varnamebuf[var->allvar],varname);
#else
  strcpy(var->varnamebuf[var->allvar],varname);
#endif
#endif /* 1543 */
  var->hash[var->allvar]=varhash;

  ig15 = var->allvar;
  var->type[ig15]=G__var_type;

  /* if var->p_tagtable[ig15]==-1, 
     it is not a struct,union */
  /* var->p_tagtable[ig15] = -1; */
  
  /* store tag identity */
  var->p_tagtable[ig15] = G__tagnum;
  /* store typedef identity */
  var->p_typetable[ig15] = G__typenum;

#if !defined(G__OLDIMPLEMENTATION2191)
  if('1'!=G__var_type) var->reftype[var->allvar] = G__reftype;
  else var->reftype[var->allvar] = G__PARANORMAL;
#elif !defined(G__OLDIMPLEMENTATION1266)
  if('Q'!=G__var_type) var->reftype[var->allvar] = G__reftype;
  else var->reftype[var->allvar] = G__PARANORMAL;
#else
  var->reftype[var->allvar] = G__reftype;
#endif
  
  /* set const flag if it is a constant */
#ifndef G__OLDIMPLEMENTATION1119
  G__constvar |= G__dynconst;
  G__dynconst=0;
#endif
  var->constvar[ig15] = G__constvar ;
#ifndef G__OLDIMPLEMENTATION2156
  if(G__cintv6 && (result.isconst&G__STATICCONST) & var->constvar[ig15]) { 
    var->constvar[ig15] |= result.isconst&G__STATICCONST;
  }
#endif
  
  /* allocate variable and pointer */
  var->allvar++;
  var->varlabel[var->allvar][0] =var->varlabel[var->allvar-1][0]+1;

  if('p'==G__var_type && 
     (G__static_alloc || G__constvar || G__prerun || G__def_struct_member)&&
     !G__macro_defining) { /* in case of enumerater */
    G__var_type='i';
  }

#ifndef G__OLDIMPLEMENTATION672
  if('u'!=tolower(var->type[ig15])&&'u'==result.type&&-1!=result.tagnum) {
#ifndef G__OLDIMPLEMENTATION1821
    int store_decl = G__decl;
    G__decl=0;
#endif
    G__fundamental_conversion_operator(var->type[ig15]
				       ,var->p_tagtable[ig15]
				       ,var->p_typetable[ig15]
				       ,var->reftype[ig15]
				       ,var->constvar[ig15]
				       ,&result,ttt);
#ifndef G__OLDIMPLEMENTATION1821
    G__decl=store_decl;
#endif
  }
#endif /* ON672 */

  /*****************************************************************
   * bytecode generation for G__allocvariable()
   *****************************************************************/
#ifdef G__ASM
#ifdef G__ASM_WHOLEFUNC
  /* following line is temprary, limitation for having class object as local
   * variable */
  if(G__asm_wholefunction) {
    if(
       /* G__funcheader case must be implemented, the following line is 
	* deleted */
       ('u'==G__var_type && G__PARAREFERENCE!=G__reftype 
#ifndef G__OLDIMPLEMENTATION1073
	&& G__funcheader
#endif
	) || 
       (1==paran && '\0'==parameter00)) {
#ifndef G__OLDIMPLEMENTATION1164
      if(0==G__xrefflag) {
#endif
	G__abortbytecode();
	G__asm_wholefunc_default_cp=0;
	G__no_exec=1;
#ifndef G__OLDIMPLEMENTATION1497
	G__return=G__RETURN_IMMEDIATE;
#else
	G__return=G__RETURN_NORMAL;
#endif
#ifdef G__ASM_DBG
	if(G__asm_dbg) {
	  G__fprinterr(G__serr,"bytecode compile aborted by automatic class object. Use pointer to class obj + new");
	  G__printlinenum();
	}
#endif
	return(result);
#ifndef G__OLDIMPLEMENTATION1164
      }
#endif
    }
  }

  if(G__asm_wholefunc_default_cp) {
    /* default param eval masked for bytecode func compilation, recovered */
    G__asm_noverflow=1;
  }

  /* may need refinement */
  if(G__asm_noverflow && G__asm_wholefunction && 1==p_inc) {
    if(G__funcheader) {
      /* function argument */
      if(G__PARAREFERENCE==G__reftype) {
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: INIT_REF\n",G__asm_cp);
#endif
	G__asm_inst[G__asm_cp] = G__INIT_REF;
	G__asm_inst[G__asm_cp+1]=ig15;
	G__asm_inst[G__asm_cp+2]=paran;
	G__asm_inst[G__asm_cp+3]=G__var_type;
	G__asm_inst[G__asm_cp+4]=(long)var;
	G__inc_cp_asm(5,0);
      }
      else {
	G__asm_gen_stvar(0,ig15,paran,var,item,G__ASM_VARLOCAL,'p'
#ifndef G__OLDIMPLEMENTATION2089
	                , &result
#endif
                        );
      }
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: POP\n",G__asm_cp);
#endif
      G__asm_inst[G__asm_cp] = G__POP;
      G__inc_cp_asm(1,0);
    }
    else if(result.type && !G__static_alloc) {
      /* normal object */
      G__asm_gen_stvar(0,ig15,paran,var,item,G__ASM_VARLOCAL,'p'
#ifndef G__OLDIMPLEMENTATION2089
	              , &result
#endif
                      );
    }
#ifndef G__OLDIMPLEMENTATION1073 /* 1073 is disabled */
    else if('u'==G__var_type && G__PARAREFERENCE!=G__reftype &&
	    -1!=G__tagnum&&'e'!=G__struct.type[G__tagnum]) {
      if(G__CPPLINK==G__struct.iscpplink[G__tagnum]) { /* precompiled class */
	/* Move LD_FUNC instruction */
	int ix;
	G__inc_cp_asm(-5,0);
	for(ix=4;ix>=0;ix--) 
	  G__asm_inst[G__asm_cp+ix+3] = G__asm_inst[G__asm_cp+ix];
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: CTOR_SETGVP %s index=%d paran=%d\n"
			       ,G__asm_cp,item,ig15,paran);
#endif
	G__asm_inst[G__asm_cp]=G__CTOR_SETGVP;
	G__asm_inst[G__asm_cp+1]=ig15;
	G__asm_inst[G__asm_cp+2]=(long)var;
	G__inc_cp_asm(3,0);
	G__inc_cp_asm(5,0); /* increment for moved LD_FUNC instruction */
      }
      else { /* Interpreted class */
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: LD_VAR  %s index=%d paran=%d\n"
			       ,G__asm_cp,item,ig15,paran);
#endif
	G__asm_inst[G__asm_cp]=G__LD_LVAR;
	G__asm_inst[G__asm_cp+1]=ig15;
	G__asm_inst[G__asm_cp+2]=paran;
	G__asm_inst[G__asm_cp+3]='p';
	G__asm_inst[G__asm_cp+4]=(long)var;
	G__inc_cp_asm(5,0);
#ifdef G__ASM_DBG
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: PUSHSTROS\n",G__asm_cp);
	if(G__asm_dbg) G__fprinterr(G__serr,"%3x: SETSTROS\n",G__asm_cp+1);
#endif
	G__asm_inst[G__asm_cp]=G__PUSHSTROS;
	G__asm_inst[G__asm_cp+1]=G__SETSTROS;
	G__inc_cp_asm(2,0);
      }
    }
#endif /* ON1073 */
  }
#endif /* G__ASM_WHOLEFUNC */

  if(G__asm_noverflow && !G__funcheader
#ifdef G__ASM_WHOLEFUNC
     && G__ASM_FUNC_NOP==G__asm_wholefunction
#endif /* G__ASM_WHOLEFUNC */
     ) {
    if(result.type) {
      G__asm_gen_stvar(0,ig15,paran,var,item,0,'p'
#ifndef G__OLDIMPLEMENTATION2089
                      , &result
#endif
                      );
    }
    else if('u'==G__var_type) {
      G__ASSERT(0==G__decl || 1==G__decl);
      if(G__decl) {
	if(G__reftype) {
	  G__redecl(var,ig15);
#ifndef G__OLDIMPLEMENTATION1720
	  if(G__no_exec_compile) G__abortbytecode();
#endif
	}
	else G__class_2nd_decl_i(var,ig15);
      }
      else if(G__cppconstruct) {
	G__class_2nd_decl_c(var,ig15);
      }
    }
  }
#endif /* G__ASM */

  /*****************************************************************
   * security check
   *****************************************************************/
  G__CHECK(G__SECURE_POINTER_INSTANTIATE,isupper(G__var_type)&&'E'!=G__var_type
	   ,return(result));
#ifndef G__OLDIMPLEMENTATION575
  G__CHECK(G__SECURE_POINTER_TYPE
	   ,isupper(G__var_type)&&result.obj.i&&G__var_type!=result.type
	   && !G__funcheader && 
	   (('Y'!=G__var_type&&result.obj.i)||G__security&G__SECURE_CAST2P)
	   ,return(result));
#else
  G__CHECK(G__SECURE_CAST2P
	   ,isupper(G__var_type)&&result.obj.i&&G__var_type!=result.type
	   && !G__funcheader 
	   ,return(result));
#endif
  G__CHECK(G__SECURE_FILE_POINTER,'E'==G__var_type,return(result));
  /* G__CHECK(G__SECURE_ARRAY,p_inc>1,return(result)); */


  /*****************************************************************
   * bit field allocation
   *****************************************************************/
#ifndef G__OLDIMPLEMENTATION448
  var->bitfield[ig15] = G__bitfield;
  if(G__bitfield) {
    G__bitfield=0;
    var->varlabel[ig15][G__MAXVARDIM-1] = bitlocation;
    if(0==bitlocation) {
      var->p[ig15] = G__malloc(1,G__INTALLOC,item);
    }
    else {
      var->p[ig15] = G__malloc(1,0,item) - G__INTALLOC;
    }
    return(result);
  }
#endif

  
  /*****************************************************************
   * actual variable allocation
   *****************************************************************/
  switch(G__var_type) {
    
    /*********************************************************
     * THIS PART MUST CORRESPOND TO G__ALLOC_VAR_REF
     *********************************************************/
  case 'u': /* struct, union */
    if(G__struct.isabstract[G__tagnum]&&0==G__ansiheader&&0==G__funcheader
       &&G__PARANORMAL==G__reftype
#ifndef G__OLDIMPLEMENTATION1556
       &&(G__CPPLINK!=G__globalcomp||G__tagdefining!=G__tagnum)
#endif
       ) {
      G__fprinterr(G__serr,"Error: abstract class object '%s %s' declared",G__struct.name[G__tagnum],item);
      G__genericerror((char*)NULL);
#ifndef G__OLDIMPLEMENTATION2221
      G__display_purevirtualfunc(G__tagnum);
#endif
      var->hash[ig15]=0;
    }
    /* type var; normal variable */
    var->p[ig15] = G__malloc((p_inc),G__struct.size[G__tagnum], item);
    /*******************************************
     * bug fix  7 Jan 1992
     * old: 'if(G__ansiheader!=0)'
     * When pre-RUN with ANSI style header, 
     * G__int(result)==NULL and 
     * memcpy tried to copy data from address 0.
     * This caused error on BSD.
     * When G__CPLUSPLUS or G__IFUNCPARA are not 
     * specified, there are no problems.
     *******************************************/
    if(G__ansiheader!=0&&result.type!='\0'&& G__globalvarpointer==G__PVOID
#ifndef G__OLDIMPLEMENTATION1624
       &&(0==G__static_alloc||-1==G__func_now)
#endif
       )
      memcpy((void *)var->p[ig15] ,(void *)(G__int(result))
	     ,(size_t)G__struct.size[var->p_tagtable[ig15]]);
    result.obj.i = var->p[ig15];
    break;

  case 'U': /* struct, union pointer */
    if(p_inc>1 && result.type!='\0') { /* char *argv[]; */
      var->p[ig15] = G__int(result);
    }
    else {
      var->p[ig15] = G__malloc((p_inc),G__LONGALLOC,item);
      if((G__def_struct_member==0&&G__ASM_FUNC_NOP==G__asm_wholefunction)&&
	 ((!G__static_alloc)||(G__prerun))&&
	 (G__globalvarpointer==G__PVOID||result.type!='\0')) {
	int baseoffset;
	if(-1!=(baseoffset=G__ispublicbase(var->p_tagtable[ig15]
					   ,result.tagnum
					   ,result.obj.i))) {
	  *((long *)var->p[ig15])=G__int(result)+baseoffset;
	}
	else {
	  *((long *)var->p[ig15])=G__int(result);
	}
      }
    }
    /* ensure returning 0 for not running constructor */
#ifndef G__OLDIMPLEMENTATION1194
    if(!autoobjectflag) result.obj.i = 0;
#else
    result.obj.i = 0;
#endif
    break;
    
    
    /***************************************
     * G__letvariable(), new variable
     * file and void pointers are same as
     * char pointer
     ***************************************/
#ifdef G__ROOT
#if !defined(G__OLDIMPLEMENTATION481)
  case 'Z': /* ROOT special object */
    var->p[ig15]=(long)malloc(G__LONGALLOC*2);
    *(long*)var->p[ig15]=0;
    *(long*)(var->p[ig15]+G__LONGALLOC)=0;
    break;
#elif !defined(G__OLDIMPLEMENTATION467)
  case 'Z': /* ROOT special object */
#endif
#endif
#ifndef G__OLDIMPLEMENTATION2191
  case '1': /* void */
#else
  case 'Q': /* void */
#endif
  case 'Y': /* void pointer */
  case 'E': /* FILE pointer */
    
  case 'c': /* char */
  case 'C': /* char pointer */
    G__ALLOC_VAR_REF(G__CHARALLOC,char,G__int)
      
      /*******************************************************
       * initialization of string
       *******************************************************/
      if(result.type=='C'&& G__var_type=='c') {
#ifndef G__OLDIMPLEMENTATION1886
	if(G__asm_wholefunction != G__ASM_FUNC_COMPILE) {
	  if((int)strlen((char *)result.obj.i)>(int)var->varlabel[ig15][1]) {
	    strncpy((char *)var->p[ig15] ,(char *)result.obj.i
		    ,(size_t)var->varlabel[ig15][1]+1);
	  }
	  else {
	    strcpy((char *)var->p[ig15] ,(char *)result.obj.i);
	  }
	}
	else {
#ifdef G__ASM_DBG
	  if(G__asm_dbg) {
	    G__fprinterr(G__serr,"%3x: LD_VAR  %s index=%d paran=%d\n"
			 ,G__asm_cp,var->varnamebuf[ig15],ig15,0);
	  }
#endif
	  G__asm_inst[G__asm_cp]=G__LD_LVAR;
	  G__asm_inst[G__asm_cp+1]=ig15;
	  G__asm_inst[G__asm_cp+2]=0;
	  G__asm_inst[G__asm_cp+3]='P';
	  G__asm_inst[G__asm_cp+4]=(long)var;
	  G__inc_cp_asm(5,0);
	  G__asm_inst[G__asm_cp]=G__SWAP;
	  G__inc_cp_asm(1,0);
	  G__asm_inst[G__asm_cp]=G__LD_FUNC;
	  G__asm_inst[G__asm_cp+1] = (long)("strcpy");
	  G__asm_inst[G__asm_cp+2] = 677;
	  G__asm_inst[G__asm_cp+3]= 2;
	  G__asm_inst[G__asm_cp+4]=(long)G__compiled_func;
	  G__inc_cp_asm(5,0);
	}
#else
	if(strlen((char *)result.obj.i)>var->varlabel[ig15][1]) {
	  strncpy((char *)var->p[ig15] ,(char *)result.obj.i
		  ,(size_t)var->varlabel[ig15][1]+1);
	}
	else {
	  strcpy((char *)var->p[ig15] ,(char *)result.obj.i);
	}
#endif
      }
    break;

#ifndef G__OLDIMPLEMENTATION2189
  case 'n':
  case 'N':
    G__ALLOC_VAR_REF(G__LONGLONGALLOC,G__int64,G__Longlong)
    break;
  case 'm':
  case 'M':
    G__ALLOC_VAR_REF(G__LONGLONGALLOC,G__int64,G__ULonglong)
    break;
#ifndef G__OLDIMPLEMENTATION2191
  case 'q':
  case 'Q':
    G__ALLOC_VAR_REF(G__LONGDOUBLEALLOC,long double,G__Longdouble)
    break;
#endif
#endif
    
#ifndef G__OLDIMPLEMENTATION1604
  case 'g': /* bool */
    result.obj.i = result.obj.i?1:0;
#ifdef G__BOOL4BYTE
    G__ALLOC_VAR_REF(G__INTALLOC,int,G__int)
    break;
#endif
  case 'G': /* bool */
#endif
  case 'b': /* unsigned char */
  case 'B': /* unsigned char pointer */
    G__ALLOC_VAR_REF(G__CHARALLOC,unsigned char,G__int)
    break;
    
  case 's': /* short int */
  case 'S': /* short int pointer */
    G__ALLOC_VAR_REF(G__SHORTALLOC,short,G__int)
    break;

  case 'r': /* unsigned short int */
  case 'R': /* unsigned short int pointer */
    G__ALLOC_VAR_REF(G__SHORTALLOC,unsigned short,G__int)
    break;

  case 'i': /* int */
  case 'I': /* int pointer */
    G__ALLOC_VAR_REF(G__INTALLOC,int,G__int)
    break;

  case 'h': /* unsigned int */
  case 'H': /* unsigned int pointer */
    G__ALLOC_VAR_REF(G__INTALLOC,unsigned int,G__int)
    break;

  case 'l': /* long int */
  case 'L': /* long int pointer */
    G__ALLOC_VAR_REF(G__LONGALLOC,long,G__int)
    break;

  case 'k': /* unsigned long int */
  case 'K': /* unsigned long int pointer */
    G__ALLOC_VAR_REF(G__LONGALLOC,unsigned long,G__int)
    break;

  case 'f': /* float */
  case 'F': /* float pointer */
    G__ALLOC_VAR_REF(G__FLOATALLOC,float,G__double)
    break;
    
  case 'd': /* double */
  case 'D': /* double pointer */
    G__ALLOC_VAR_REF(G__DOUBLEALLOC,double,G__double)
    break;
    
    /****************************************************
     * FILE and void are ignored
     ****************************************************/
  case 'e': /* file */
    G__genericerror("Limitation: FILE type variable can not be declared unless type FILE is explicitly defined");
    var->hash[ig15] = 0;
    break;
    
  case 'y': /* void */
    G__genericerror("Error: void type variable can not be declared");
    var->hash[ig15] = 0;
    break;

#ifndef G__OLDIMPLEMENTATION2191
  case 'j': /* macro file position */
#else
  case 'm': /* macro file position */
#endif
    var->p[ig15] = G__malloc(1,sizeof(fpos_t),item);
    *(fpos_t *)var->p[ig15] = *(fpos_t *)result.obj.i;
    break;

  case 'a': /* pointer to member function */
    var->p[ig15] = G__malloc(p_inc,G__P2MFALLOC,item);
    if((G__def_struct_member==0&&G__ASM_FUNC_NOP==G__asm_wholefunction)&&
       ((!G__static_alloc)||(G__prerun))&& result.obj.i &&
       (G__globalvarpointer==G__PVOID||result.type!='\0')) {
#ifdef G__PTR2MEMFUNC
      if('C'==result.type) 
	*(long*)var->p[ig15] = result.obj.i;
      else
	memcpy((void*)var->p[ig15], (void*)result.obj.i,G__P2MFALLOC);
#else
      memcpy((void*)var->p[ig15], (void*)result.obj.i,G__P2MFALLOC);
#endif
    }
    break;
    
#ifndef G__OLDIMPLEMENTATION1347
#ifndef G__OLDIMPLEMENTATION2191
    /* case '1':  */ /* function, ???Questionable??? */
#else
  case 'q': /* function, ???Questionable??? */
#endif
    var->p[ig15] = G__malloc(p_inc,sizeof(long),item);
    break;
#endif
    
    /****************************************************
     * Automatic variable and macro
     *   p : macro int
     *   P : macro double
     *   o : auto int
     *   O : auto double
     ****************************************************/
  default: /* case 'p' macro or 'o' automatic variable */
    
    /*************************************************
     * if not macro definition, print out warning
     *************************************************/
    if((G__definemacro==0)&&(G__globalcomp==G__NOLINK)) {
#ifndef G__OLDIMPLEMENTATION566
      if(-1!=var->tagnum) {
	if(G__dispmsg>=G__DISPWARN) {
	  G__fprinterr(G__serr,"Warning: Undeclared data member %s",item);
	  G__genericerror((char*)NULL);
	}
	return(result);
      }
#endif
#ifndef G__OLDIMPLEMENTATION1089
      /* Following code will never be used */
#ifndef G__PHILIPPE21
      if(0==G__const_noerror) {
        /* the next comment seems obsolete. the code is sometimes used! */
#endif
      G__fprinterr(G__serr,"Error: Undeclared variable %s",item);
      G__genericerror((char*)NULL);
#ifndef G__PHILIPPE21
      }
#endif
#else
      if(G__dispmsg>=G__DISPWARN) {
	G__fprinterr(G__serr,"Warning: Undeclared symbol %s. Automatic variable allocated",item);
	G__printlinenum();
      }
#endif
      var->type[ig15]='o';
    }
    
    /*************************************************
     * re-allocate array index information
     *************************************************/
    for(ig25=0;ig25<paran;ig25++) {
      var->varlabel[ig15][ig25+1]=G__int(para[ig25])+1;
    }
    
    ary=1;
    for(ig25=2;ig25<paran+1;ig25++) {
      ary *= var->varlabel[ig15][ig25++] ;
    }
    var->varlabel[ig15][0] = ary;
    
    p_inc= ary*var->varlabel[ig15][1];
    if(p_inc==0) p_inc=1; /* this is special for this case */
    var->varlabel[ig15][1] = p_inc-1;
    
    
    /*************************************************
     * allocate double macro or not
     *************************************************/
    if(G__isdouble(result)) {
      /*  'P' macro double, 'O' auto double */
      var->type[ig15]=toupper(var->type[ig15]);
      var->p[ig15] = G__malloc((p_inc),G__DOUBLEALLOC, item);
      if((
#ifdef G__OLDIMPLEMENTATION689
	  G__def_struct_member==0&&
#endif
	  G__ASM_FUNC_NOP==G__asm_wholefunction)&& 
	 ((!G__static_alloc)||(G__prerun))&&
	 (G__globalvarpointer==G__PVOID|| result.type!='\0'))
	*((double *)var->p[ig15]+(p_inc-1)) = G__double(result);
    }
    /*************************************************
     * allocate int macro or not
     *************************************************/
    else {
      /*  'p' macro int, 'o' auto int */

#ifndef G__OLDIMPLEMENTATION904
      if('C'==result.type) var->type[ig15]='T';
#endif
      var->p[ig15] = G__malloc((p_inc+1),G__LONGALLOC,item);
      if((
#ifdef G__OLDIMPLEMENTATION689
	  G__def_struct_member==0&&
#endif
	  G__ASM_FUNC_NOP==G__asm_wholefunction)&& 
	 ((!G__static_alloc)||(G__prerun))&&
	 (G__globalvarpointer==G__PVOID|| result.type!='\0'))
	*((long *)var->p[ig15]+(p_inc-1)) = G__int(result);
    }
    break;
  }

  /* Check for un-assigned internal pointer */
  G__CHECK(G__SECURE_POINTER_INIT
	   ,!G__def_struct_member&&isupper(G__var_type)&&
	   G__ASM_FUNC_NOP==G__asm_wholefunction&&
	   var->p[ig15]&&0==(*(long*)var->p[ig15])
	   ,*(long*)var->p[ig15]=0);

#ifdef G__SECURITY
      if(G__security&G__SECURE_GARBAGECOLLECTION && !G__def_struct_member &&
#ifndef G__OLDIMPLEMENTATION545
	 (!G__no_exec_compile) &&
#endif
	 isupper(G__var_type) && var->p[ig15] && (*((long *)var->p[ig15]))) {
	G__add_refcount((void*)(*(long*)var->p[ig15]),(void**)var->p[ig15]);
      }
#endif

  if(autoobjectflag) {
    var->statictype[ig15] = G__AUTO;
    G__tagnum = store_tagnum;
    G__typenum = store_typenum;
    G__globalvarpointer = store_globalvarpointer;
    G__var_type = store_var_type;
  }

#ifndef G__OLDIMPLEMENTATION1985
  if(-1!=var->tagnum && G__prerun && 
#ifndef G__OLDIMPLEMENTATION1999
     G__access==G__PUBLIC &&
#endif
     strcmp(varname,"G__virtualinfo")==0) {
    G__struct.virtual_offset[var->tagnum] = var->p[ig15];
  }
#endif


  return(result);
}



/******************************************************************
* G__getvarentry()
*
* Used in 
*   debug.c    G__lock_variable   : Unimportant
*   debug.c    G__unlock_variable : Unimportant
*   decl.c     G__initary         : no data member 
*   decl.c     G__initstruct      : no data member
* * func.c     G__getfunction  : To get pointer to function,
* * sizeof.c   G__Lsizeof  : Only this one has to deal with the access rule
*   loadfile.c G__include_file   : #include MACRO expansion, no data member
*   newlink.c  G__specify_link   : Only for global scope
*
******************************************************************/
struct G__var_array *G__getvarentry(varname,varhash,pi,varglobal,varlocal)
char *varname;
int varhash;
int *pi;
struct G__var_array *varglobal,*varlocal;
{
  struct G__var_array *var=NULL;
  int ilg,ig15;
  /* long G__struct_offset; */
  int in_memfunc=0;
#ifdef G__NEWINHERIT
  int basen;
  int isbase;
  int accesslimit;
  int memfunc_or_friend=0;
  struct G__inheritance *baseclass=NULL;
#endif
  
  /***********************************************************
   * search old local and global variables.
   *
   ***********************************************************/
  ilg=G__LOCAL; /* start from local variable */
  
  while(ilg<=G__GLOBAL+1) {
    
    /***********************************************
     * switch local and global for getvariable
     ************************************************/
    switch(ilg) {
    case G__LOCAL:
      in_memfunc=0;
      /******************************************
       * Beginning , local or global entry
       ******************************************/
      if(varlocal&&(!G__def_struct_member)) {
	var=varlocal;
	if(varglobal) {
	  if(G__exec_memberfunc) {
	    ilg=G__MEMBER;
	  }
	  else {
	    ilg=G__GLOBAL;
	  }
	}
	else {
	  ilg=G__NOTHING;
	}
      }
      else {
	var=varglobal;
	ilg=G__NOTHING;
      }
      break;
      
    case G__MEMBER:
      in_memfunc=1;
#ifdef G__OLDIMPLEMENTATION589_YET /* not activated due to bug */
      G__ASSERT(0<=G__memberfunc_tagnum);
      G__incsetup_memvar(G__memberfunc_tagnum);
      var = G__struct.memvar[G__memberfunc_tagnum] ;
#else
#ifndef G__OLDIMPLEMENTATION2223
      if(-1!=G__tagnum) {
	G__incsetup_memvar(G__tagnum);
	var = G__struct.memvar[G__tagnum] ;
      }
      else {
	var = (struct G__var_array*)NULL;
      }
#else
      G__incsetup_memvar(G__tagnum);
      var = G__struct.memvar[G__tagnum] ;
#endif
#endif
      ilg = G__GLOBAL;
      break;
      
    case G__GLOBAL:
      /******************************************
       * global entry
       ******************************************/
      in_memfunc=0;
      /* G__struct_offset = 0; */
      var=varglobal;
      ilg=G__NOTHING;
      break;
    }
    
    
    /*************************************************
     * Searching for variable name 
     *
     *************************************************/
#ifdef G__NEWINHERIT
    /* If searching for class member, check access rule */
    if(in_memfunc ||(struct G__var_array*)NULL==varglobal) {
      isbase=1;
      basen=0;
#ifdef G__OLDIMPLEMENTATION589_YET /* not activated due to bug */
      if(in_memfunc) baseclass=G__struct.baseclass[G__memberfunc_tagnum];
      else           baseclass=G__struct.baseclass[G__tagnum];
#else
      baseclass = G__struct.baseclass[G__tagnum];
#endif
      if(G__exec_memberfunc || G__isfriend(G__tagnum)) {
	accesslimit = G__PUBLIC_PROTECTED_PRIVATE ;
	memfunc_or_friend = 1;
      }
      else {
	accesslimit = G__PUBLIC;
	memfunc_or_friend = 0;
      }
    }
    else {
#ifndef G__OLDIMPLEMENTATION2098
      if(G__decl) accesslimit = G__PUBLIC_PROTECTED_PRIVATE ;
      else        accesslimit = G__PUBLIC;
#else
      accesslimit = G__PUBLIC;
#endif
      isbase=0;
      basen=0;
    }
    /* search for variable name and access rule match */
    do {
    next_base:
      while(var) {
	ig15=0;
	while(ig15<var->allvar) {
	  if(varhash==var->hash[ig15] && 
	     strcmp(varname,var->varnamebuf[ig15])==0 &&
	     (var->statictype[ig15]<0||
#ifndef G__OLDIMPLEMENTATION1135
	      G__filescopeaccess(G__ifile.filenum,var->statictype[ig15])
#else
	      G__ifile.filenum==var->statictype[ig15]
#endif
	      )&&
	     (var->access[ig15]&accesslimit)) {
	    *pi=ig15;
	    return(var);
	  }
	  ++ig15;
	}
	var=var->next;
      }
      /* next base class if searching for class member */
      if(isbase) {
	while(baseclass && basen<baseclass->basen) {
	  if(memfunc_or_friend) {
	    if((baseclass->baseaccess[basen]&G__PUBLIC_PROTECTED) ||
	       baseclass->property[basen]&G__ISDIRECTINHERIT) {
	      accesslimit = G__PUBLIC_PROTECTED;
	      G__incsetup_memvar(baseclass->basetagnum[basen]);
	      var = G__struct.memvar[baseclass->basetagnum[basen]];
	      ++basen;
	      goto next_base;
	    }
	  }
	  else {
	    if(baseclass->baseaccess[basen]&G__PUBLIC) {
	      accesslimit = G__PUBLIC;
	      G__incsetup_memvar(baseclass->basetagnum[basen]);
	      var = G__struct.memvar[baseclass->basetagnum[basen]];
	      ++basen;
	      goto next_base;
	    }
	  }
	  ++basen;
	}
	isbase=0;
      }
    } while(isbase);
#else
    while(var) {
      ig15=0;
      while(ig15<var->allvar) {
	if(varhash==var->hash[ig15] && 
	   strcmp(varname,var->varnamebuf[ig15])==0 &&
	   (var->statictype[ig15]<0||
#ifndef G__OLDIMPLEMENTATION1135
	    G__filescopeaccess(G__ifile.filenum,var->statictype[ig15])
#else
	    G__ifile.filenum==var->statictype[ig15]
#endif
	    )&&
	   (G__PUBLIC==var->access[ig15] || in_memfunc ||
	    G__isfriend(G__tagnum))) {
	  *pi=ig15;
	  return(var);
	}
	++ig15;
      }
      var=var->next;
    }
#endif
  }
  /***********************************************************
   * end of variable name search 'while()' loop.
   *   searched all old local and global variables.
   ***********************************************************/
  
  return((struct G__var_array *)NULL);
}


/**************************************************************************
* G__getthis()
*
**************************************************************************/
int G__getthis(result7,varname,item)
G__value *result7;
char *varname,*item;
{
  if(G__exec_memberfunc && strcmp(varname,"this")==0) {
    if(0==G__store_struct_offset) {
      G__genericerror("Error: Can't use 'this' pointer in static member func");
      return(0);
    }
#ifdef G__ASM
    if(G__asm_noverflow) {
#ifdef G__ASM_DBG
      if(G__asm_dbg) G__fprinterr(G__serr,"%3x: LD_THIS %c\n",G__asm_cp,G__var_type);
#endif
      G__asm_inst[G__asm_cp] = G__LD_THIS;
      G__asm_inst[G__asm_cp+1] = G__var_type;
      G__inc_cp_asm(2,0);
    }
#endif
    switch(G__var_type) {
    case 'v':
      G__letint(result7,'u',G__store_struct_offset);
      result7->ref = G__store_struct_offset;
      break;
    case 'P':
      G__reference_error(item);
      break;
    case 'p':
    default:
      G__letint(result7,'U',G__store_struct_offset);
      break;
    }
    G__var_type = 'p'; /* pointer to struct,class */
    result7->typenum = G__typenum;
    result7->tagnum = G__tagnum;
#ifndef G__OLDIMPLEMENTATION1131
    result7->ref = 0;
#endif
#ifndef G__OLDIMPLEMENTATION2185
    result7->isconst = 0;
#endif
    return(1);
  }
  return(0);
}

/**************************************************************************
* G__letpointer2memfunc()
*
**************************************************************************/
void G__letpointer2memfunc(var,paran,ig15,item,p_inc,presult,G__struct_offset)
struct G__var_array *var;
int paran;
int ig15;
char *item;
int p_inc;
G__value *presult;
long G__struct_offset;
{
  switch(G__var_type) {
  case 'p': /* var = expr; assign to value */
    if(var->paran[ig15]<=paran) { /*assign to type element*/
#ifdef G__PTR2MEMFUNC
      if('C'==presult->type) 
	*(long*)(G__struct_offset+var->p[ig15]+p_inc*G__P2MFALLOC)
	  = presult->obj.i;
      else
	memcpy((void*)(G__struct_offset+var->p[ig15]+p_inc*G__P2MFALLOC)
	       ,(void*)presult->obj.i,G__P2MFALLOC);
#else
      memcpy((void*)(G__struct_offset+var->p[ig15]+p_inc*G__P2MFALLOC)
	     ,(void*)presult->obj.i,G__P2MFALLOC);
#endif
      break;
    }
  default:
    G__assign_error(item,presult);
    break;
  }
}

/**************************************************************************
* G__letautomatic()
*
**************************************************************************/
void G__letautomatic(var,ig15,G__struct_offset,p_inc,result)
struct G__var_array *var;
int ig15;
long G__struct_offset;
int p_inc;
G__value result;
{
  if(isupper(var->type[ig15])) {
    *(double *)(G__struct_offset+var->p[ig15]+p_inc*G__DOUBLEALLOC) 
      = G__double(result);
  }
  else {
    *(long *)(G__struct_offset+var->p[ig15]+p_inc*G__LONGALLOC) 
      = G__int(result);
  }
}

#ifdef G__FRIEND
/**************************************************************************
* G__isfriend()
*
**************************************************************************/
int G__isfriend(tagnum)
int tagnum;
{
  struct G__friendtag *friendtag;
  if(G__exec_memberfunc) {
    if(G__memberfunc_tagnum==tagnum) return(1);
#ifndef G__OLDIMPLEMENTATION1458
    if (G__memberfunc_tagnum < 0) return 0;
#endif
    friendtag = G__struct.friendtag[G__memberfunc_tagnum];
    while(friendtag) {
      if(friendtag->tagnum==tagnum) return(1);
      friendtag=friendtag->next;
    }
  }
  if(-1!=G__func_now && G__p_local && G__p_local->ifunc) {
    friendtag = G__p_local->ifunc->friendtag[G__p_local->ifn];
    while(friendtag) {
      if(friendtag->tagnum==tagnum) return(1);
      friendtag=friendtag->next;
    }
  }
  return(0);
}

/**************************************************************************
* G__parse_friend
*
*  friend class A;
*  friend type func(param);
*  friend type operator<<(param);
*  friend A<T,U> operator<<(param);
*  friend const A<T,U> operator<<(param);
* 
**************************************************************************/
int G__parse_friend(piout,pspaceflag,mparen)
int *piout,*pspaceflag;
int mparen;
{
#ifdef G__FRIEND
  int friendtagnum,envtagnum;
  struct G__friendtag *friendtag;
  int tagtype=0;
#else
  static int state=1;
#endif
  int store_tagnum,store_def_tagnum,store_def_struct_member;
  int store_tagdefining,store_access;
  fpos_t pos;
  int line_number;
#ifndef G__OLDIMPLEMENTATION1826
  char classname[G__LONGLINE];
#else
  char classname[G__ONELINE];
#endif
  int c;
#ifndef G__OLDIMPLEMENTATION458
  int def_tagnum,tagdefining;
#endif

#ifndef G__FRIEND
  if(G__NOLINK==G__store_globalcomp&&G__NOLINK==G__globalcomp) {
    if(state) {
      G__genericerror("Limitation: friend privilege not supported");
      state=0;
    }
  }
#endif

  fgetpos(G__ifile.fp,&pos);
  line_number = G__ifile.line_number;
  c = G__fgetname_template(classname,";");
  if(isspace(c)) {
    if(strcmp(classname,"class")==0) {
      c=G__fgetname_template(classname,";");
      tagtype='c';
    }
    else if(strcmp(classname,"struct")==0) {
      c=G__fgetname_template(classname,";");
      tagtype='s';
    }
    else {
#ifndef G__OLDIMPLEMENTATION1489
      if(strcmp(classname,"const")==0 || strcmp(classname,"volatile")==0 ||
	 strcmp(classname,"register")==0) {
	c = G__fgetname_template(classname,";");
      }
#else
      c=G__fignorestream(";,(");
#endif
      switch(c) {
      case ';':
      case ',':
	tagtype='c';
	break;
      case '(':
	break;
      }
    }
  }
  else if(';'==c) {
    tagtype='c';
  }

#ifdef G__FRIEND
  envtagnum = G__get_envtagnum();
  if(-1==envtagnum) {
    G__genericerror("Error: friend keyword appears outside class definition");
  }
#endif
  
  
#ifdef G__FRIEND
  store_tagnum = G__tagnum;
  store_def_tagnum = G__def_tagnum;
  store_def_struct_member = G__def_struct_member;
  store_tagdefining = G__tagdefining;
  store_access = G__access;
  
  G__friendtagnum=envtagnum;

  if(-1!=G__tagnum) G__tagnum = G__struct.parent_tagnum[G__tagnum];
  if(-1!=G__def_tagnum) G__def_tagnum = G__struct.parent_tagnum[G__def_tagnum];
  if(-1!=G__tagdefining)G__tagdefining=G__struct.parent_tagnum[G__tagdefining];
  if(-1!=G__tagdefining||-1!=G__def_tagnum) G__def_struct_member=1;
  else                                      G__def_struct_member=0;
  G__access = G__PUBLIC;
  G__var_type='p';

  if(tagtype) {
    while(classname[0]) {
#ifndef G__OLDIMPLEMENTATION458
      def_tagnum=G__def_tagnum;
#ifdef G__OLDIMPLEMENTATION1107
      tagdefining=G__tagdefining; /* ??? not good ??? */
#endif
      G__def_tagnum=store_def_tagnum;
      G__tagdefining=store_tagdefining;
#ifndef G__OLDIMPLEMENTATION1107
      tagdefining=G__tagdefining; /* ??? good ??? */
#endif
      friendtagnum=G__defined_tagname(classname,2);
      G__def_tagnum=def_tagnum;
      G__tagdefining=tagdefining;
      if(-1==friendtagnum) friendtagnum=G__search_tagname(classname,tagtype);
#else
      friendtagnum=G__search_tagname(classname,tagtype);
#endif
      /* friend class ... ; */
      if(-1!=envtagnum) {
	friendtag = G__struct.friendtag[friendtagnum];
	if(friendtag) {
	  while(friendtag->next) friendtag=friendtag->next;
	  friendtag->next
	    =(struct G__friendtag*)malloc(sizeof(struct G__friendtag));
	  friendtag->next->next=(struct G__friendtag*)NULL;
	  friendtag->next->tagnum=envtagnum;
	}
	else {
	  G__struct.friendtag[friendtagnum]
	    =(struct G__friendtag*)malloc(sizeof(struct G__friendtag));
	  friendtag = G__struct.friendtag[friendtagnum];
	  friendtag->next=(struct G__friendtag*)NULL;
	  friendtag->tagnum=envtagnum;
	}
      }
      if(';'!=c) c = G__fgetstream(classname,";,");
      else       classname[0]='\0';
    }
#else
  if(-1!=G__defined_tagname(classname,1)) {
    if(';'!=c) c = G__fignorestream(";");
#endif
  }
  else {
    /* friend type f() {  } ; */
    fsetpos(G__ifile.fp,&pos);
    G__ifile.line_number = line_number;

#ifndef G__FRIEND
    store_tagnum = G__tagnum;
    store_def_tagnum = G__def_tagnum;
    store_def_struct_member = G__def_struct_member;
    store_tagdefining = G__tagdefining;
    store_access = G__access;
    G__tagnum = -1;
    G__def_tagnum = -1;
    G__tagdefining = -1;
    G__def_struct_member = 0;
    G__access = G__PUBLIC;
    G__var_type='p';
#ifndef G__OLDIMPLEMENTATION1297
     /* friend function belongs to the inner-most namespace 
      * not the parent class! */
    while((G__def_tagnum!=-1)&&(G__struct.type[G__def_tagnum]!='n')) {
      G__def_tagnum = G__struct.parent_tagnum[G__def_tagnum];
    }
#endif
#endif
    
#ifndef G__OLDIMPLEMENTATION1541
     /* friend function belongs to the inner-most namespace 
      * not the parent class! In fact, this fix is not perfect, because
      * a friend function can also be a member function. This fix works
      * better only because there is no strict checking for non-member
      * function. */
    if(G__NOLINK!=G__globalcomp && -1!=G__def_tagnum && 	
       'n'!=G__struct.type[G__def_tagnum]) {
      if(G__dispmsg>=G__DISPWARN) {
	G__fprinterr(G__serr,"Warning: This friend declaration may cause creation of wrong stub function in dictionary. Use '#pragma link off function ...;' to avoid it.");
	G__printlinenum();
      }
    }
    while((G__def_tagnum!=-1)&&(G__struct.type[G__def_tagnum]!='n')) {
      G__def_tagnum = G__struct.parent_tagnum[G__def_tagnum];
      G__tagdefining = G__def_tagnum;
      G__tagnum = G__def_tagnum;
    }
#endif
    G__exec_statement();
    
#ifndef G__FRIEND
    G__tagnum = store_tagnum;
    G__def_tagnum = store_def_tagnum;
    G__def_struct_member = store_def_struct_member;
    G__tagdefining = store_tagdefining;
    G__access = store_access;
#endif
  }
    
#ifdef G__FRIEND
  G__tagnum = store_tagnum;
  G__def_tagnum = store_def_tagnum;
  G__def_struct_member = store_def_struct_member;
  G__tagdefining = store_tagdefining;
  G__access = store_access;
  G__friendtagnum = -1;
#endif
    
  *pspaceflag = -1;
  *piout=0;
  return(!mparen);
}
#endif

#ifndef G__OLDIMPLEMENTATION488
/**************************************************************************
* G__deletevariable()
*
* delete variable from global varaible table. return 1 if successful.
*
**************************************************************************/
int G__deletevariable(varname)
char *varname;
{
  long struct_offset=0;
  long store_struct_offset=0;
  int ig15;
  int varhash;
  int isdecl=0;
  struct G__var_array *var;
  int cpplink = G__NOLINK;

  G__hash(varname,varhash,ig15);

  var = G__searchvariable(varname,varhash
			  ,(struct G__var_array*)NULL,&G__global
			  ,&struct_offset,&store_struct_offset,&ig15,isdecl);

  if(var) {
    int i;
    int done;
    int store_tagnum;
    char temp[G__ONELINE];
    switch(var->type[ig15]) {
    case 'u':
      store_struct_offset = G__store_struct_offset;
      store_tagnum = G__tagnum;
      G__store_struct_offset = var->p[ig15];
      G__tagnum = var->p_tagtable[ig15];
      sprintf(temp,"~%s()",var->varnamebuf[ig15]);
      /********************************************************
       * destruction of array 
       ********************************************************/
      if(G__CPPLINK==G__struct.iscpplink[G__tagnum]) {
	G__store_struct_offset = var->p[ig15];
	if((i=var->varlabel[ig15][1])>0
#ifndef G__OLDIMPLEMENTATION2011
	   || var->paran[ig15]
#endif
	   ) G__cpp_aryconstruct=i+1;
	G__getfunction(temp,&done,G__TRYDESTRUCTOR); 
	G__cpp_aryconstruct=0;
	cpplink = G__CPPLINK;
      }
      else {
	int size=G__struct.size[G__tagnum];
	for(i=var->varlabel[ig15][1];i>=0;--i) {
	  G__store_struct_offset = var->p[ig15]+size*i;
	  if(G__dispsource) {
	    G__fprinterr(G__serr,"\n0x%lx.%s",G__store_struct_offset,temp);
	  }
	  done=0;
	  G__getfunction(temp,&done,G__TRYDESTRUCTOR); 
	  if(0==done) break;
	}
	G__tagnum=store_tagnum;
	G__store_struct_offset = store_struct_offset;
      }
      break;
    default:
#ifdef G__SECURITY
      if(G__security&G__SECURE_GARBAGECOLLECTION && 
#ifndef G__OLDIMPLEMENTATION545
	 (!G__no_exec_compile) &&
#endif
	 isupper(var->type[ig15]) && var->p[ig15]) {
	long address;
	i=var->varlabel[ig15][1]+1;
	do {
	  --i;
	  address = var->p[ig15] + G__LONGALLOC*i;
	  if(*((long*)address)) {
	    G__del_refcount((void*)(*((long*)address)),(void**)address);
	  }
	} while(i);
      }
#endif
      break;
    }
    if(G__NOLINK==cpplink && var->p[ig15]) free((void*)var->p[ig15]);
    var->p[ig15] = 0;
    var->varnamebuf[ig15][0] = '\0';
    var->hash[ig15]=0;
    return(1);
  }
  else {
    return(0);
  }
}
#endif

#ifndef G__OLDIMPLEMENTATION488
/**************************************************************************
* G__deleteglobal()
*
* delete variable from global varaible table. return 1 if successful.
*
**************************************************************************/
int G__deleteglobal(pin) 
void* pin;
{
  long p=(long)pin;
  struct G__var_array *var;
  int ig15;

#ifndef G__OLDIMPLEMENTATION1035
  G__LockCriticalSection();
#endif

  var = &G__global;

  while(var) {
    for(ig15=0;ig15<var->allvar;ig15++) {
      if(p==var->p[ig15]) { 
        var->p[ig15] = 0;
        var->varnamebuf[ig15][0] = '\0';
        var->hash[ig15]=0;
        /* return(1); */
      }
      if(isupper(var->type[ig15])&&var->p[ig15]&&p==(*(long*)var->p[ig15])) {
        if(G__AUTO==var->globalcomp[ig15]) free((void*)var->p[ig15]);
        var->p[ig15] = 0;
        var->varnamebuf[ig15][0] = '\0';
        var->hash[ig15]=0;
        /* return(1); */
      }
    }
    var=var->next;
  }
#ifndef G__OLDIMPLEMENTATION1035
  G__UnlockCriticalSection();
#endif
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
