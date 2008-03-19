#if 0
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

using namespace Cint::Bytecode;
using namespace Cint::Internal;

/*************************************************************************
 * G__get_autoobjectstack
 *************************************************************************/
//G__autoobjectstack G__autoobjectstack_obj;
Cint::Bytecode::G__autoobjectstack& Cint::Bytecode::G__get_autoobjectstack() {
  static G__autoobjectstack G__autoobjectstack_obj;
  return G__autoobjectstack_obj;
}

/*************************************************************************
 * class G__autoobject
 *************************************************************************/
Cint::Bytecode::G__autoobject::~G__autoobject() {
   Type type(m_tagnum);
   int sizex = type.SizeOf();
  //a. In case of array, call each dtor, not using delete[], -> later review this
  //b. In case of array in heap, free only the youngest pointer address at the end
  for(int i=m_num-1;i>=0;--i) {
    G__calldtor((void*)((long)m_p+sizex*i),m_tagnum,i?0:m_isheap);
  }
}
/////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__autoobject::disp(void) const {
  fprintf(G__serr,"(%p,tagnum%d,num%d,scope%d,heap%d)"
          ,m_p,G__get_tagnum(m_tagnum),m_num,m_scopelevel,m_isheap);
}

/*************************************************************************
 * class G__autoobjectstack
 *************************************************************************/
/////////////////////////////////////////////////////////////////////////
void Cint::Bytecode::G__autoobjectstack::disp(int scopelevel) const {
   fprintf(G__serr,"autostack=%d scope=%d ",(int)m_ctnr.size(),scopelevel);
   std::list<G__autoobject*>::const_iterator i;
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
void* Cint::Bytecode::G__push_heapobjectstack(const Reflex::Scope& tagnum,int num,int scopelevel) {
  return G__get_autoobjectstack().push(tagnum,num,scopelevel);
}

/*************************************************************************
 * G__push_autoobjectstack
 *************************************************************************/
void Cint::Bytecode::G__push_autoobjectstack(void *p,const Reflex::Scope& tagnum,int num
                                        ,int scopelevel,int isheap) {
  G__get_autoobjectstack().push(p,tagnum,num,scopelevel,isheap);
}


/*************************************************************************
 * G__push_autoobjectstack
 *************************************************************************/
void Cint::Bytecode::G__delete_autoobjectstack(int scopelevel) {
  //G__get_autoobjectstack().disp(scopelevel); // DEBUG
  G__get_autoobjectstack().Autodelete(scopelevel);
}


/*************************************************************************
 * G__allocheapobjectstack
 *  This function is called by G__exec_bytecode for returning class object
 *************************************************************************/
void* Cint::Bytecode::G__allocheapobjectstack(const Reflex::Member& func,int scopelevel)
{
  // CAUTION: This operation is inefficient. Doing type checking at run-time.

  void *p;
  // get function return value
  G__value dmy;
  G__value_typenum(dmy) = func.TypeOf().ReturnType();
  G__TypeReader cls;
  cls.Init(dmy);
  cls.setreftype(G__get_reftype(func.TypeOf()));
#pragma message (FIXME("G__TypeReader needs to use func return const vs. func const without setting int isconst"))
  //cls.setisconst(G__get_isconst(func.TypeOf()));
  cls.setstatic(func.IsStatic());
  if(!cls.IsValid() || 
     (cls.Property()&(G__BIT_ISPOINTER|G__BIT_ISREFERENCE|G__BIT_ISSTATIC)) ||
     0==(cls.Property()&(G__BIT_ISCLASS|G__BIT_ISSTRUCT|G__BIT_ISUNION)) ||
     func.IsConstructor()) {
    // This function does not return class object
    p = (void*)NULL;
  }
  else {
    // reserve heap object if this function returns class object. 
    // At the end of G__exec_bytecoode, copy ctor is called to copy the result.
    // It is done in G__copyheapobjectstack()
    p = G__push_heapobjectstack(func.DeclaringScope(),1,scopelevel);
#ifdef G__ASM_DBG
    if(G__asm_dbg) {
      G__fprinterr(G__serr,"temp object %lx %s reserved for %s\n"
                   ,p,cls.Name(),func.Name().c_str());
    }
#endif
  }
  return(p);
}

/*************************************************************************
 * G__copyheapobjectstack
 *  This function is called by G__exec_bytecode for returning class object
 *************************************************************************/
void Cint::Bytecode::G__copyheapobjectstack(void* p,G__value *result
                                ,const Reflex::Member& func) {
  if(!p) return;

  // CAUTION: This operation is inefficient. Doing type checking at run-time.

  G__ClassInfo cls(G__get_tagnum(func.TypeOf().ReturnType().RawType()));
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
      struct G__param para;
      para.paran = 0;
      para.para[0] = G__null;
      G__callfunc0(&dmyresult,G__Dict::GetDict().GetFunction(m.Handle()),&para,p,funcmatch);
    }

    // search assignment operator
    m = cls.GetAssignOperator();
    funcmatch = G__CALLMEMFUNC;
  }

  if(m.IsValid()) {
    // in case,  copy ctor or operator=  is found
    G__value dmyresult;
    struct G__param para;
    para.paran = 1;
    para.para[0] = *result;
#ifdef G__ASM_DBG
    if(G__asm_dbg) {
      G__fprinterr(G__serr,"temp object copy ctor %lx <= %lx %s for %s\n"
                   ,p,result->obj.i,cls.Name(),func.Name().c_str());
    }
#endif
    G__callfunc0(&dmyresult,G__Dict::GetDict().GetFunction(m.Handle()),&para,p,funcmatch);
    result->obj.i = (long)p;
    result->ref = result->obj.i;
  }
  else {
    // if there is no copy ctor nor operator=
    // memberwise copy
#ifdef G__ASM_DBG
    if(G__asm_dbg) { 
      G__fprinterr(G__serr,"temp object memcpy %lx <= %lx %s for %s\n"
                   ,p,result->obj.i,cls.Name(),func.Name().c_str());
    }
#endif
    memcpy(p,(void*)result->obj.i,func.TypeOf().ReturnType().RawType().SizeOf());
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
#endif // 0
