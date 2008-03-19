/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file bc_autoobj.h
 ************************************************************************
 * Description:
 *  stack buffer for automatic object, execution subsystem
 ************************************************************************
 * Copyright(c) 2004~2004  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifndef AUTOOBJ_H
#define AUTOOBJ_H

#include "G__ci.h"
#include "common.h"

#include <list>

//
// Note: Functions in this source file are used in run-time environment
//       and not in compile time.
//

namespace Cint {
   namespace Bytecode {

/*************************************************************************
 * class G__autoobject
 *************************************************************************/
class G__autoobject {
 public:
    G__autoobject(void *p,const Reflex::Scope& tagnum,int num,int scopelevel,int isheap) 
    : m_p(p) , m_tagnum(tagnum) , m_num(num), m_scopelevel(scopelevel) 
    , m_isheap(isheap) {}
  int Scopelevel() const { return(m_scopelevel); }
  ~G__autoobject() ;
  void disp(void) const;
 private:
  void *m_p;
  Reflex::Scope m_tagnum;
  int m_num;
  int m_scopelevel;
  int m_isheap;
};

/*************************************************************************
 * class G__autoobjectstack
 *************************************************************************/
class G__autoobjectstack {
 private:
  std::list<G__autoobject*> m_ctnr;
  int m_busy;

 private:
  int Scopelevel() const { 
    if(!m_ctnr.empty()) return(m_ctnr.back()->Scopelevel()); 
    else return(-1);
  }
  void pop(void) { 
    G__autoobject* p = m_ctnr.back();
    delete p;
    m_ctnr.pop_back(); 
  }

 public: 
  G__autoobjectstack() { m_busy=0; }
  void* push(const Reflex::Scope& tagnum,int num,int scopelevel) {
     if(!tagnum || tagnum.IsTopScope()||num==0||0>=static_cast<const Reflex::Type&>(tagnum).SizeOf()) return((void*)NULL);
     void *p = malloc(static_cast<const Reflex::Type&>(tagnum).SizeOf()*num);
      push(p,tagnum,num,scopelevel,1);
      return(p);
  }
  void push(void *p,const Reflex::Scope& tagnum,int num,int scopelevel,int isheap) {
      m_ctnr.push_back(new G__autoobject(p,tagnum,num,scopelevel,isheap));
  }
  void Autodelete(int scopelevel) {
#ifdef G__ASM_DBG
     if(Cint::Internal::G__asm_dbg) disp(scopelevel);
#endif
    if(m_busy) return;
    while(m_ctnr.size() && scopelevel<Scopelevel()) {
      m_busy=1;
      pop();
      m_busy=0;
    }
  }
  void disp(int scopelevel = ::Cint::Internal::G__scopelevel) const;
};

// static object
G__autoobjectstack& G__get_autoobjectstack();

/*************************************************************************
 * C wrappers
 *************************************************************************/

/*************************************************************************
 * G__push_autoobjectstack
 *************************************************************************/
void* G__push_heapobjectstack(const Reflex::Scope& tagnum,int num,int scopelevel) ;

/*************************************************************************
 * G__push_autoobjectstack
 *************************************************************************/
void G__push_autoobjectstack(void *p,const Reflex::Scope&,int num
                                        ,int scopelevel,int isheap) ;


/*************************************************************************
 * G__push_autoobjectstack
 *************************************************************************/
void G__delete_autoobjectstack(int scopelevel) ;

/*************************************************************************
 * G__allocheapobjectstack
 *************************************************************************/
void* G__allocheapobjectstack(const Reflex::Member& func,int scopelevel);

/*************************************************************************
 * G__copyheapobjectstack
 *************************************************************************/
void G__copyheapobjectstack(void* p,G__value* result, const Reflex::Member& func);

   } // namespace Bytecode
} // namespace Cint

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
