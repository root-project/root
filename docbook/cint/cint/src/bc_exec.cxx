/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file bc_exec.cxx
 ************************************************************************
 * Description:
 *  bytecode executor, execution subsystem
 ************************************************************************
 * Copyright(c) 2004~2004  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "bc_exec.h"
#include "bc_eh.h"
#include "bc_autoobj.h"
#include "bc_debug.h"

/***********************************************************************
* G__bc_exec_virtualbase_bytecode()
*  This function is set as LD_FUNC 4th oprand (*func)()
***********************************************************************/
extern "C" int G__bc_exec_virtualbase_bytecode(G__value *result7
			,char *funcname        // objtagnum
			,struct G__param *libp
			,int hash              // vtblindex,basetagnum
			) {

  // TODO, below is a dead copy of virtual_bytecode which may not work

  long vtagnum = (long)funcname; // tagnum of given pointer
  int vtblindex = hash&0xffff; // virtual function table index
  int vbasetagnum = hash/0x10000; // vbasetagnum
  int voffset=G__struct.virtual_offset[vtagnum]; // offset for true tagnum info
  int tagnum = *(long*)(G__store_struct_offset+voffset); // tagnum of object
          //            A
          //  *           B       << origin of virtual function
          //  *         C C C     << offset=B.Offset()
          //          D
          //  *       E E E E E   << offset=C.Offset()+vtbloffset
          //  G__store_struct_offset -= (Evtbl.offset-Cvtbl.offset)
  G__Vtable *vtblptr = (G__Vtable*)G__struct.vtable[vtagnum];
  G__Vtabledata *vtbldataptr = vtblptr->resolve(vtblindex,vbasetagnum);
  int offsetptr = vtbldataptr->GetOffset();

  G__Vtable *vtbl = (G__Vtable*)G__struct.vtable[tagnum];
  G__Vtabledata *vtbldata = vtbl->resolve(vtblindex,vbasetagnum);
  int offset = vtbldata->GetOffset();

  struct G__ifunc_table_internal *ifunc = G__get_ifunc_internal(vtbldata->GetIfunc());
  int ifn = vtbldata->GetIfn();

  if(G__BYTECODE_NOTYET==ifunc->pentry[ifn]->bytecodestatus) {
    if(G__BYTECODE_FAILURE==G__bc_compile_function(ifunc,ifn)) return(0);
  }

  G__store_struct_offset -= (offset - offsetptr); // TODO, need review 
  int result=G__exec_bytecode(result7,(char*)ifunc->pentry[ifn]->bytecode,libp,hash);
  G__store_struct_offset += (offset - offsetptr); // TODO, need review

  result = -(offset-offsetptr);

  return(result);
}

/***********************************************************************
* G__bc_exec_virtual_bytecode()
*  This function is set as LD_FUNC 4th oprand (*func)()
*
*  G__ifunc_table <>--* func  <>-- vtblindex,basetag  ...(used by compiler)
*  LD_FUNC(bytecode) <>--- vtagnum,(vtblindex,basetag)
*                           |            |
*                           v            v
*  G__tagtable <>-----* class <>--1 vtbl[ ] <>--* vfunc <>-- ifunc,ifn,offset
*                             <>--1 vtblos[ ]
*
*     vfunc = vtbl[vtblindex+vtblos[basetag]];
*                            
*
*  From object
*   *(p+voffset)    tagnum
*
* Multiple inheritance:
*    class A     f1  f2  f3                    vtbloffset A,0
*    class B                 f4  f5  f6  f7    vtbloffset B,0
*    class C     f1  f2  f3  f4  f5  f6  f7    vtbloffset A,0 B,3
*      f5 vtblindex = 1 for B, 4 for C
*
***********************************************************************/
extern "C" int G__bc_exec_virtual_bytecode(G__value *result7
			,char *funcname        // vtagnum
			,struct G__param *libp
			,int hash              // vtblindex,basetagnum
			) {

  long vtagnum = (long)funcname; // tagnum of given pointer
  int vtblindex = hash&0xffff; // virtual function table index
  int vbasetagnum = hash/0x10000; // vbasetagnum
  int voffset=G__struct.virtual_offset[vtagnum]; // offset for true tagnum info
  int tagnum = *(long*)(G__store_struct_offset+voffset); // tagnum of object
          //            A
          //  *           B       << origin of virtual function
          //  *         C C C     << offset=B.Offset()
          //          D
          //  *       E E E E E   << offset=C.Offset()+vtbloffset
          //  G__store_struct_offset -= (Evtbl.offset-Cvtbl.offset)
  G__Vtable *vtblptr = (G__Vtable*)G__struct.vtable[vtagnum];
  G__Vtabledata *vtbldataptr = vtblptr->resolve(vtblindex,vbasetagnum);
  int offsetptr = vtbldataptr->GetOffset();

  G__Vtable *vtbl = (G__Vtable*)G__struct.vtable[tagnum];
  G__Vtabledata *vtbldata = vtbl->resolve(vtblindex,vbasetagnum);
  int offset = vtbldata->GetOffset();

  struct G__ifunc_table_internal *ifunc = G__get_ifunc_internal(vtbldata->GetIfunc());
  int ifn = vtbldata->GetIfn();

  if(G__BYTECODE_NOTYET==ifunc->pentry[ifn]->bytecodestatus) {
    if(G__BYTECODE_FAILURE==G__bc_compile_function(ifunc,ifn)) return(0);
  }

  G__store_struct_offset -= (offset - offsetptr); // TODO, need review 
  int result=G__exec_bytecode(result7,(char*)ifunc->pentry[ifn]->bytecode,libp,hash);
  G__store_struct_offset += (offset - offsetptr); // TODO, need review

  result = -(offset-offsetptr);

  return(result);
}

/***********************************************************************
* G__bc_exec_normal_bytecode()
*  This function is set as LD_FUNC 4th oprand (*func)()
***********************************************************************/
extern "C" int G__bc_exec_normal_bytecode(G__value *result7
			,char *funcname        // ifunc
			,struct G__param *libp
			,int hash              // ifn
			) {
  struct G__ifunc_table_internal* ifunc = (struct G__ifunc_table_internal*)funcname;
  int ifn = hash;

  if(G__BYTECODE_NOTYET==ifunc->pentry[ifn]->bytecodestatus) {
    if(G__BYTECODE_FAILURE==G__bc_compile_function(ifunc,ifn)) return(0);
  }

  return(G__exec_bytecode(result7,(char*)ifunc->pentry[ifn]->bytecode,libp,hash));
}

/***********************************************************************
* G__bc_exec_ctor_bytecode()
*  This function is set as LD_FUNC 4th oprand (*func)()
***********************************************************************/
extern "C" int G__bc_exec_ctor_bytecode(G__value *result7
			,char *funcname        // ifunc
			,struct G__param *libp
			,int hash              // ifn
			) {
  struct G__ifunc_table_internal* ifunc = (struct G__ifunc_table_internal*)funcname;
  int ifn = hash;

  if(G__BYTECODE_NOTYET==ifunc->pentry[ifn]->bytecodestatus) {
    if(G__BYTECODE_FAILURE==G__bc_compile_function(ifunc,ifn)) return(0);
  }

  int result
    = G__exec_bytecode(result7,(char*)ifunc->pentry[ifn]->bytecode,libp,hash);
  result7->obj.i = (long)G__store_struct_offset;
  result7->ref = (long)G__store_struct_offset;
  result7->type = 'u';
  result7->tagnum = ifunc->tagnum;
  return(result);
}


//////////////////////////////////////////////////////////////////////////

/***********************************************************************
* G__bc_exec_ctorary_bytecode()
*  This function is set as LD_FUNC 4th oprand (*func)()
* Review, This function could also be used for operator= ary
***********************************************************************/
extern "C" int G__bc_exec_ctorary_bytecode(G__value *result7
			,char *funcname        // ifunc
			,struct G__param *libp 
			,int hash              // ifn
			) {
  struct G__ifunc_table_internal* ifunc = (struct G__ifunc_table_internal*)funcname;
  int ifn = hash; 
  int tagnum = ifunc->tagnum;
  int size = G__struct.size[tagnum];
  int result = 0;
  // todo, This solution relies on global variable G__cpp_aryconstruct
  int n = G__cpp_aryconstruct?G__cpp_aryconstruct:1;
  G__cpp_aryconstruct=0;

  if(G__BYTECODE_NOTYET==ifunc->pentry[ifn]->bytecodestatus) {
    if(G__BYTECODE_FAILURE==G__bc_compile_function(ifunc,ifn)) return(0);
  }

  long store_struct_offset=G__store_struct_offset;
  for(int i=0;i<n;i++) {
    result
      =G__exec_bytecode(result7,(char*)ifunc->pentry[ifn]->bytecode,libp,hash);
    G__store_struct_offset += size;
    if(1==libp->paran&&'U'==libp->para[0].type&&tagnum==libp->para[0].tagnum) {
      if(libp->para[0].obj.i) {
	if(libp->para[0].ref==libp->para[0].obj.i) libp->para[0].ref += size;
	libp->para[0].obj.i += size;
      }
    }
  }
  G__store_struct_offset=store_struct_offset;

  return(result);
}

//////////////////////////////////////////////////////////////////////////

/***********************************************************************
* G__bc_exec_dtorary_bytecode()
*  This function is set as LD_FUNC 4th oprand (*func)()
* CAUTION:
*  This is used only for dtor array because if the object is an array,
*  it is always referenced as object and not by a pointer.  Because dtor
*  can be virtual, dtor referenced by pointer has to use G__bc_virtual_bytecode
***********************************************************************/
extern "C" int G__bc_exec_dtorary_bytecode(G__value *result7
			,char *funcname        // ifunc
			,struct G__param *libp
			,int hash              // ifn
			) {
  struct G__ifunc_table_internal* ifunc = (struct G__ifunc_table_internal*)funcname;
  int ifn = hash; 
  int tagnum = ifunc->tagnum;
  int size = G__struct.size[tagnum];
  int result = 0;
  // todo, This solution relies on global variable G__cpp_aryconstruct
  int n = G__cpp_aryconstruct?G__cpp_aryconstruct:1;
  G__cpp_aryconstruct=0;

  if(G__BYTECODE_NOTYET==ifunc->pentry[ifn]->bytecodestatus) {
    if(G__BYTECODE_FAILURE==G__bc_compile_function(ifunc,ifn)) return(0);
  }

  long store_struct_offset=G__store_struct_offset;
  G__store_struct_offset += (n-1)*size;
  for(int i=0;i<n;i++) {
    result =
      G__exec_bytecode(result7,(char*)ifunc->pentry[ifn]->bytecode,libp,hash);
    G__store_struct_offset -= size;
  }
  G__store_struct_offset=store_struct_offset;

  return(result);
}

//////////////////////////////////////////////////////////////////////////
/***********************************************************************
* G__bc_exec_try_bytecode()
***********************************************************************/
extern "C" int G__bc_exec_try_bytecode(int start,
				       int stack,
				       G__value *presult,
				       long localmem) {
  G__bc_store_bytecode_env env;
  env.save();
  G__catchexception = 0; // don't use try.catch in G__ExceptionWrapper()
                         // this is restored in env.restore()
#if ENABLE_CPP_EXCEPTIONS
  try {
#endif //ENABLE_CPP_EXCEPTIONS
    G__exec_asm(start,stack,presult,localmem);
    env.restore();
    return(G__TRY_NORMAL);
#if ENABLE_CPP_EXCEPTIONS
  }
  catch(G__bc_exception& /* x */) {
    env.restore();
    return(G__TRY_INTERPRETED_EXCEPTION);
  }
  //catch(G__exception& x) { 
  //  This is not needed. G__exception is for interpreter
  //}
  catch(std::exception& /* x */) {
    env.restore();
    return(G__TRY_COMPILED_EXCEPTION);
  }
  // catch(G__bc_compile_error& x) {
  //   Never catch compile error exception here. It has to be caught in
  //   bytecode compiler.
  // }
  catch(...) {
    env.restore();
    return(G__TRY_UNCAUGHT); 
  }

  return(G__TRY_UNCAUGHT); // never happens
#endif //ENABLE_CPP_EXCEPTIONS
}

//////////////////////////////////////////////////////////////////////////
/***********************************************************************
* G__bc_exec_throw_bytecode()
***********************************************************************/
extern "C" int G__bc_exec_throw_bytecode(G__value* pval) {
#if ENABLE_CPP_EXCEPTIONS
   // coverity[exception_thrown]: we don't care.
  throw G__bc_exception(*pval);
#else //ENABLE_CPP_EXCEPTIONS
	G__fprinterr(G__serr, "G__bc_exe_throw_bytecode has no effect with exceptions disabled! %s, %d\n", __FILE__, __LINE__);
#endif //ENABLE_CPP_EXCEPTIONS
  return 0;
}

//////////////////////////////////////////////////////////////////////////
/***********************************************************************
* G__bc_exec_typematch_bytecode()
***********************************************************************/
extern "C" int G__bc_exec_typematch_bytecode(G__value* catchtype,G__value* excptobj) {
  if(catchtype->type == excptobj->type) {
    switch(catchtype->type) {
    case 'u':
    case 'U':
      if(catchtype->tagnum==excptobj->tagnum) return(1);
      if(-1!=G__ispublicbase(catchtype->tagnum,excptobj->tagnum,0)) return(1);
      return(0);
      break;
    default:
      return(1);
    }
  }
  return(0);
}

//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////
class G__exec_bytecode_autoobj {
  int m_scopelevel;
  void* m_localmem;
  int m_isheap;
 public:
  void Set(int scopelevel,void* localmem,int isheap) {
    m_scopelevel = scopelevel;
    m_localmem = localmem;
    m_isheap = isheap;
  }
  ~G__exec_bytecode_autoobj();
};
//////////////////////////////////////////////////////////////////////////
G__exec_bytecode_autoobj::~G__exec_bytecode_autoobj() {
  G__scopelevel = m_scopelevel;
  G__delete_autoobjectstack(m_scopelevel+1);
  if(m_isheap) free((void*)m_localmem);
}

//////////////////////////////////////////////////////////////////////////

/**************************************************************************
* G__exec_bytecode()
*
*
**************************************************************************/
extern "C" int G__exec_bytecode(G__value *result7,G__CONST char *funcname,struct G__param *libp,int /*hash*/)
{
  int i;
  struct G__bytecodefunc *bytecode;
  G__value asm_stack_g[G__MAXSTACK]; /* data stack */
  long *store_asm_inst;
  G__value *store_asm_stack;
  char *store_asm_name;
  int store_asm_name_p;
  struct G__param *store_asm_param;
  int store_asm_exec;
  int store_asm_noverflow;
  int store_asm_cp;
  int store_asm_dt;
  int store_asm_index; /* maybe unneccessary */
  int store_tagnum;
  long localmem;
  int store_exec_memberfunc;
  long store_memberfunc_struct_offset;
  int store_memberfunc_tagnum;
#define G__LOCALBUFSIZE 32
  char localbuf[G__LOCALBUFSIZE];
  struct G__var_array *var;
  void *ptmpbuf;
  if (G__asm_dbg) {
    G__fprinterr(G__serr, "G__exec_bytecode: starting bytecode execution ...\n");
  }

  G__bc_funccallstack_obj.setlinenum(G__ifile.line_number);

  /* use funcname as bytecode struct */
  bytecode = (struct G__bytecodefunc*)funcname;
  var = bytecode->var;

  ptmpbuf=G__allocheapobjectstack(G__get_ifunc_ref(bytecode->ifunc),bytecode->ifn,++G__scopelevel);
#ifdef G__ASM_DBG
  if(G__asm_dbg) {
    fprintf(G__serr,"tmpobj=%p scope%d\n",ptmpbuf,G__scopelevel);
  }
#endif

#ifdef G__ASM_DBG
  if(G__asm_dbg||(G__dispsource
                  &&0==G__stepover
                 )) {
    if(bytecode->ifunc->tagnum>=0) 
      G__fprinterr(G__serr,"Running bytecode function %s::%s inst=%lx->%lx stack=%lx->%lx stros=%lx %d\n"
		   ,G__struct.name[bytecode->ifunc->tagnum]
		   ,bytecode->ifunc->funcname[bytecode->ifn]
		   ,(long)G__asm_inst,(long)bytecode->pinst
		   ,(long)G__asm_stack,(long)asm_stack_g
		   ,G__store_struct_offset,G__tagnum);
    else
      G__fprinterr(G__serr,"Running bytecode function %s inst=%lx->%lx stack=%lx->%lx\n"
	      ,bytecode->ifunc->funcname[bytecode->ifn]
	      ,(long)G__asm_inst,(long)bytecode->pinst
	      ,(long)G__asm_stack,(long)asm_stack_g);
  }
#endif


  /* Push loop compilation environment */
  store_asm_inst = G__asm_inst;
  store_asm_stack = G__asm_stack;
  store_asm_name = G__asm_name;
  store_asm_name_p = G__asm_name_p;
  store_asm_param  = G__asm_param ;
  store_asm_exec  = G__asm_exec ;
  store_asm_noverflow  = G__asm_noverflow ;
  store_asm_cp  = G__asm_cp ;
  store_asm_dt  = G__asm_dt ;
  store_asm_index  = G__asm_index ;
  store_tagnum = G__tagnum;
  store_exec_memberfunc = G__exec_memberfunc;
  store_memberfunc_struct_offset=G__memberfunc_struct_offset;
  store_memberfunc_tagnum=G__memberfunc_tagnum;

  /* set new bytecode environment */
  G__asm_inst = bytecode->pinst;
  G__asm_stack = asm_stack_g;
  G__asm_name = bytecode->asm_name;
  G__asm_name_p = 0;
  G__tagnum = bytecode->var->tagnum;
  G__asm_noverflow = 0; /* bug fix */
  if(bytecode->ifunc->tagnum>=0) G__exec_memberfunc = 1;
  else                           G__exec_memberfunc = 0;
  G__memberfunc_struct_offset=G__store_struct_offset;
  G__memberfunc_tagnum=G__tagnum;

  /* copy constant buffer */
  {
    int nx = bytecode->stacksize;
    int ny = G__MAXSTACK-nx;
    for(i=0;i<nx;i++) asm_stack_g[ny+i] = bytecode->pstack[i];
  }

  /* copy arguments to stack in reverse order */
  int idx = 0;
  for (i = 0; i < libp->paran; ++i) {
    int j = libp->paran - i - 1;
    G__asm_stack[j] = libp->para[i];
    if (var && (
      !G__asm_stack[j].ref ||
      ((var->reftype[idx] == G__PARAREFERENCE) && (var->type[idx] != libp->para[i].type))
    )) {
	switch (var->type[idx]) {
	case 'f':
	  G__asm_stack[j].ref=(long)G__Floatref(&libp->para[i]);
	  break;
	case 'd':
	  G__asm_stack[j].ref=(long)G__Doubleref(&libp->para[i]);
	  break;
	case 'c':
	  G__asm_stack[j].ref=(long)G__Charref(&libp->para[i]);
	  break;
	case 's':
	  G__asm_stack[j].ref=(long)G__Shortref(&libp->para[i]);
	  break;
	case 'i':
	  G__asm_stack[j].ref=(long)G__Intref(&libp->para[i]);
	  break;
	case 'l':
	  G__asm_stack[j].ref=(long)G__Longref(&libp->para[i]);
	  break;
	case 'b':
	  G__asm_stack[j].ref=(long)G__UCharref(&libp->para[i]);
	  break;
	case 'r':
	  G__asm_stack[j].ref=(long)G__UShortref(&libp->para[i]);
	  break;
	case 'h':
	  G__asm_stack[j].ref=(long)G__UIntref(&libp->para[i]);
	  break;
	case 'k':
	  G__asm_stack[j].ref=(long)G__ULongref(&libp->para[i]);
	  break;
	case 'u':
	  G__asm_stack[j].ref=libp->para[i].obj.i;
	  break;
	case 'g':
	  G__asm_stack[j].ref=(long)G__UCharref(&libp->para[i]);
	  break;
	case 'n':
	  G__asm_stack[j].ref=(long)G__Longlongref(&libp->para[i]);
	  break;
	case 'm':
	  G__asm_stack[j].ref=(long)G__ULonglongref(&libp->para[i]);
	  break;
	case 'q':
	  G__asm_stack[j].ref=(long)G__Longdoubleref(&libp->para[i]);
	  break;
	default:
	  G__asm_stack[j].ref=(long)(&libp->para[i].obj.i);
	  break;
	}
    }
    ++idx;
    if (var) {
       if (idx >= var->allvar) {
          var = var->next;
          idx = 0;
       }
    }
  }

  G__exec_bytecode_autoobj autoobj;

  /* allocate local memory */
  if(bytecode->varsize>G__LOCALBUFSIZE) {
    localmem = (long)malloc(bytecode->varsize);
    autoobj.Set(G__scopelevel-1,(void*)localmem,1);
  }
  else {
    localmem=(long)localbuf;
    autoobj.Set(G__scopelevel-1,(void*)localmem,0);
  }

#ifdef G__DUMPFILE 
  if(G__dumpfile!=NULL) {
    int ipara;
    G__FastAllocString resultx(G__ONELINE);
    for(ipara=0;ipara<G__dumpspace;ipara++) fprintf(G__dumpfile," ");
    fprintf(G__dumpfile,"%s(",bytecode->ifunc->funcname[bytecode->ifn]);
    for(ipara=1;ipara<= libp->paran;ipara++) {
      if(ipara!=1) fprintf(G__dumpfile,",");
      G__valuemonitor(libp->para[ipara-1],resultx);
      fprintf(G__dumpfile,"%s",resultx());
    }
    fprintf(G__dumpfile,");/*%s %d (bc)*/\n" ,G__ifile.name,G__ifile.line_number);
    G__dumpspace += 3;
    
  }
#endif

  /* run bytecode function */
  
  G__bc_funccallstack_obj.push(bytecode,localmem,G__store_struct_offset,G__ifile.line_number,libp);
  ++bytecode->ifunc->busy[bytecode->ifn];
  G__exec_asm(/*start*/0,/*stack*/libp->paran,result7,localmem);
  --bytecode->ifunc->busy[bytecode->ifn];
  G__bc_funccallstack_obj.pop();

#ifndef G__OLDIMPLEMENTATION1259
  result7->isconst = bytecode->ifunc->isconst[bytecode->ifn];
#endif
  if (
    ( // direct ptr into whole-func compiled local var block
      (result7->ref >= (long) localmem) &&
      (result7->ref < (long)(localmem + bytecode->varsize))
    ) ||
    (result7->ref < 1000000L) // offset into whole-func local var blk
  ) {
    // Returned value has a ptr into the local vars, kill the
    // ptr, we are about to (conceptually anyway) destroy the
    // local var stack.
    result7->ref = 0;
  }
#ifdef G__ASM_DBG
  if(G__asm_dbg) {
    G__FastAllocString temp(G__ONELINE);
    G__fprinterr(G__serr,"returns %s\n",G__valuemonitor(*result7,temp));
  }
#endif
  if(G__RETURN_IMMEDIATE>=G__return) G__return=G__RETURN_NON;


#ifdef G__ASM_DBG
  if(G__asm_dbg||(G__dispsource
                  &&0==G__stepover
                  )) {
    if(bytecode->ifunc->tagnum>=0) 
      G__fprinterr(G__serr,"Exit bytecode function %s::%s restore inst=%lx stack=%lx\n"
	      ,G__struct.name[bytecode->ifunc->tagnum]
	      ,bytecode->ifunc->funcname[bytecode->ifn]
	      ,(long)store_asm_inst,(long)store_asm_stack);
    else
      G__fprinterr(G__serr,"Exit bytecode function %s restore inst=%lx stack=%lx\n"
	      ,bytecode->ifunc->funcname[bytecode->ifn]
	      ,(long)store_asm_inst,(long)store_asm_stack);
  }
#endif
#ifdef G__DUMPFILE
  if(G__dumpfile!=NULL) {
    int ipara;
    G__FastAllocString resultx(G__ONELINE);
    G__dumpspace -= 3;
    for(ipara=0;ipara<G__dumpspace;ipara++) fprintf(G__dumpfile," ");
    G__valuemonitor(*result7,resultx);
    fprintf(G__dumpfile ,"/* return(bc) %s()=%s*/\n" 
	    ,bytecode->ifunc->funcname[bytecode->ifn],resultx());
  }
#endif

  /* restore bytecode environment */
  G__asm_inst = store_asm_inst;
  G__asm_stack = store_asm_stack;
  G__asm_name = store_asm_name;
  G__asm_name_p = store_asm_name_p;
  G__asm_param  = store_asm_param ;
  G__asm_exec  = store_asm_exec ;
  G__asm_noverflow  = store_asm_noverflow ;
  G__asm_cp  = store_asm_cp ;
  G__asm_dt  = store_asm_dt ;
  G__asm_index  = store_asm_index ;
  G__tagnum = store_tagnum;
  G__exec_memberfunc = store_exec_memberfunc;
  G__memberfunc_struct_offset=store_memberfunc_struct_offset;
  G__memberfunc_tagnum=store_memberfunc_tagnum;

  if(ptmpbuf) G__copyheapobjectstack(ptmpbuf,result7,G__get_ifunc_ref(bytecode->ifunc),bytecode->ifn);
 
  if (G__asm_dbg) {
    G__fprinterr(G__serr, "G__exec_bytecode: end bytecode execution ...\n");
  }
  return(0);
}

/***********************************************************************
* G__bc_throw_compile_error()
***********************************************************************/
int G__bc_throw_compile_error() {
#if ENABLE_CPP_EXCEPTIONS
  throw G__bc_compile_error();
#else //ENABLE_CPP_EXCEPTIONS
  G__fprinterr(G__serr, "G__bc_throw_compile_error has no effect with exceptions disabled. %s %d\n", __FILE__, __LINE__);
#endif //ENABLE_CPP_EXCEPTIONS
  return 0;
}

/***********************************************************************
* G__bc_throw_runtime_error()
***********************************************************************/
int G__bc_throw_runtime_error() {
#if ENABLE_CPP_EXCEPTIONS
  throw G__bc_runtime_error();
#else //ENABLE_CPP_EXCEPTIONS
	G__fprinterr(G__serr, "G__bc_throw_runtime_error has no effect with exceptions disabled. %s %d\n", __FILE__, __LINE__);
#endif //ENABLE_CPP_EXCEPTIONS
  return 0;
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

extern "C" {
  extern int G__asm_step;
  double G__doubleM(G__value *buf);
  typedef void (*G__p2f_tovalue) G__P((G__value*));
  void G__asm_toXvalue(G__value* result);
}

#include "bc_exec_asm.h"



