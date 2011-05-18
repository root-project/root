/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file bc_autoobj.cxx
 ************************************************************************
 * Description:
 *  stack buffer for automatic object, execution subsystem
 ************************************************************************
 * Copyright(c) 2004~2004  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "bc_autoobj.h"
#include "bc_type.h"
#include "Api.h"

/*************************************************************************
 * G__get_autoobjectstack
 *************************************************************************/
//G__autoobjectstack G__autoobjectstack_obj;
G__autoobjectstack& G__get_autoobjectstack() {
  static G__autoobjectstack G__autoobjectstack_obj;
  return G__autoobjectstack_obj;
}

/*************************************************************************
 * class G__autoobject
 *************************************************************************/
G__autoobject::~G__autoobject() {
  int sizex = G__struct.size[m_tagnum];
  //a. In case of array, call each dtor, not using delete[], -> later review this
  //b. In case of array in heap, free only the youngest pointer address at the end
  for(int i=m_num-1;i>=0;--i) {
    G__calldtor((void*)((long)m_p+sizex*i),m_tagnum,i?0:m_isheap);
  }
}
/////////////////////////////////////////////////////////////////////////
void G__autoobject::disp(void) const {
  fprintf(G__serr,"(%p,tagnum%d,num%d,scope%d,heap%d)"
	  ,m_p,m_tagnum,m_num,m_scopelevel,m_isheap);
}

/*************************************************************************
 * class G__autoobjectstack
 *************************************************************************/
/////////////////////////////////////////////////////////////////////////
void G__autoobjectstack::disp(int scopelevel) const {
   fprintf(G__serr,"autostack=%d scope=%d ",(int)m_ctnr.size(),scopelevel);
  list<G__autoobject*>::const_iterator i;
  for(i=m_ctnr.begin();i!=m_ctnr.end();++i) {
    (*i)->disp();
  }
  fprintf(G__serr,"\n");
}

/*************************************************************************
 *************************************************************************
 * C wrappers
 *************************************************************************
 *************************************************************************/

/*************************************************************************
 * G__push_autoobjectstack
 *************************************************************************/
extern "C" void* G__push_heapobjectstack(int tagnum,int num,int scopelevel) {
  return G__get_autoobjectstack().push(tagnum,num,scopelevel);
}

/*************************************************************************
 * G__push_autoobjectstack
 *************************************************************************/
extern "C" void G__push_autoobjectstack(void *p,int tagnum,int num
					,int scopelevel,int isheap) {
  G__get_autoobjectstack().push(p,tagnum,num,scopelevel,isheap);
}


/*************************************************************************
 * G__push_autoobjectstack
 *************************************************************************/
extern "C" void G__delete_autoobjectstack(int scopelevel) {
  //G__get_autoobjectstack().disp(scopelevel); // DEBUG
  G__get_autoobjectstack().Autodelete(scopelevel);
}


/*************************************************************************
 * G__allocheapobjectstack
 *  This function is called by G__exec_bytecode for returning class object
 *************************************************************************/
extern "C" void* G__allocheapobjectstack(struct G__ifunc_table *ifuncref,int ifn,int scopelevel)
{
  // CAUTION: This operation is inefficient. Doing type checking at run-time.

  G__ifunc_table_internal *ifunc = G__get_ifunc_internal(ifuncref);
  int tagnum = ifunc->p_tagtable[ifn];
  void *p;
  // get function return value
  G__value dmy;
  dmy.type = ifunc->type[ifn];
  dmy.tagnum = ifunc->p_tagtable[ifn];
  dmy.typenum = ifunc->p_typetable[ifn];
  dmy.obj.reftype.reftype = ifunc->reftype[ifn];
  dmy.isconst = ifunc->isconst[ifn];
  G__TypeReader cls;
  cls.Init(dmy);
  cls.setreftype(ifunc->reftype[ifn]);
  cls.setisconst(ifunc->isconst[ifn]);
  cls.setstatic(ifunc->staticalloc[ifn]);
  if(!cls.IsValid() || 
     (cls.Property()&(G__BIT_ISPOINTER|G__BIT_ISREFERENCE|G__BIT_ISSTATIC)) ||
     0==(cls.Property()&(G__BIT_ISCLASS|G__BIT_ISSTRUCT|G__BIT_ISUNION)) ||
     (-1!=dmy.tagnum&&strcmp(G__struct.name[dmy.tagnum],ifunc->funcname[ifn])==0)) {
    // This function does not return class object
    p = (void*)NULL;
  }
  else {
    // reserve heap object if this function returns class object. 
    // At the end of G__exec_bytecoode, copy ctor is called to copy the result.
    // It is done in G__copyheapobjectstack()
    p = G__push_heapobjectstack(tagnum,1,scopelevel);
#ifdef G__ASM_DBG
    if(G__asm_dbg) {
      G__fprinterr(G__serr,"temp object %lx %s reserved for %s\n"
		   ,p,cls.Name(),ifunc->funcname[ifn]);
    }
#endif
  }
  return(p);
}

/*************************************************************************
 * G__copyheapobjectstack
 *  This function is called by G__exec_bytecode for returning class object
 *************************************************************************/
extern "C" void G__copyheapobjectstack(void* p,G__value *result
				,struct G__ifunc_table *ifunc,int ifn) {
  if(!p) return;

  // CAUTION: This operation is inefficient. Doing type checking at run-time.

  int tagnum = G__get_ifunc_internal(ifunc)->p_tagtable[ifn];
  G__ClassInfo cls(tagnum);
  G__MethodInfo m;
  int funcmatch;

  // search copy constructor  first
  m = cls.GetCopyConstructor();
  funcmatch = G__TRYCONSTRUCTOR;

  if(!m.IsValid()) {
    // if copy ctor is not found, then try  default ctor + operator=
    // search default constructor for initialization
    m = cls.GetDefaultConstructor();
    funcmatch = G__TRYCONSTRUCTOR;
    if(m.IsValid()) {
      G__value dmyresult;
      struct G__ifunc_table *ifunc2=(struct G__ifunc_table*)m.Handle();
      int ifn2=m.Index();
      struct G__param* para = new G__param();
      para->paran = 0;
      para->para[0] = G__null;
      G__callfunc0(&dmyresult,ifunc2,ifn2,para,p,funcmatch);
      delete para;
    }

    // search assignment operator
    m = cls.GetAssignOperator();
    funcmatch = G__CALLMEMFUNC;
  }

  if(m.IsValid()) {
    // in case,  copy ctor or operator=  is found
    struct G__ifunc_table *ifunc2=(struct G__ifunc_table*)m.Handle();
    int ifn2=m.Index();
    G__value dmyresult;
    struct G__param* para = new G__param();
    para->paran = 1;
    para->para[0] = *result;
#ifdef G__ASM_DBG
    if(G__asm_dbg) {
      G__fprinterr(G__serr,"temp object copy ctor %lx <= %lx %s for %s\n"
		   ,p,result->obj.i,cls.Name(),G__get_ifunc_internal(ifunc2)->funcname[ifn2]);
    }
#endif
    G__callfunc0(&dmyresult,ifunc2,ifn,para,p,funcmatch);
    result->obj.i = (long)p;
    result->ref = result->obj.i;
    delete para;
  }
  else {
    // if there is no copy ctor nor operator=
    // memberwise copy
#ifdef G__ASM_DBG
    if(G__asm_dbg) { 
      G__fprinterr(G__serr,"temp object memcpy %lx <= %lx %s for %s\n"
		   ,p,result->obj.i,cls.Name(),G__get_ifunc_internal(ifunc)->funcname[ifn]);
    }
#endif
    memcpy(p,(void*)result->obj.i,G__struct.size[tagnum]);
    result->obj.i = (long)p;
    result->ref = result->obj.i;
  }
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
