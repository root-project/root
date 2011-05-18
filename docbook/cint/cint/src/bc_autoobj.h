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

#if !defined(__sun) && (!defined(_MSC_VER) || _MSC_VER > 1200) && !(defined(__xlC__) || defined(__xlc__))
//extern "C" {
#ifdef __CINT__
#include "../G__ci.h"
#else
#include "common.h"
#endif
//}
#else
#include "G__ci.h"
#include "common.h"
#endif

#include <list>
using namespace std;

//
// Note: Functions in this source file are used in run-time environment
//       and not in compile time.
//

/*************************************************************************
 * class G__autoobject
 *************************************************************************/
class G__autoobject {
 public:
  G__autoobject(void *p,int tagnum,int num,int scopelevel,int isheap) 
    : m_p(p) , m_tagnum(tagnum) , m_num(num), m_scopelevel(scopelevel) 
    , m_isheap(isheap) {}
  int Scopelevel() const { return(m_scopelevel); }
  ~G__autoobject() ;
  void disp(void) const;
 private:
  void *m_p;
  int m_tagnum;
  int m_num;
  int m_scopelevel;
  int m_isheap;
};

/*************************************************************************
 * class G__autoobjectstack
 *************************************************************************/
class G__autoobjectstack {
 private:
  list<G__autoobject*> m_ctnr;
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
  void* push(int tagnum,int num,int scopelevel) {
      if(-1==tagnum||num==0||0>=G__struct.size[tagnum]) return((void*)NULL);
      void *p = malloc(G__struct.size[tagnum]*num);
      push(p,tagnum,num,scopelevel,1);
      return(p);
  }
  void push(void *p,int tagnum,int num,int scopelevel,int isheap) {
      m_ctnr.push_back(new G__autoobject(p,tagnum,num,scopelevel,isheap));
  }
  void Autodelete(int scopelevel) {
#ifdef G__ASM_DBG
  if(G__asm_dbg) disp(scopelevel);
#endif
    if(m_busy) return;
    while(m_ctnr.size() && scopelevel<Scopelevel()) {
      m_busy=1;
      pop();
      m_busy=0;
    }
  }
  void disp(int scopelevel=G__scopelevel) const;
};

// static object
G__autoobjectstack& G__get_autoobjectstack();

/*************************************************************************
 * C wrappers
 *************************************************************************/

/*************************************************************************
 * G__push_autoobjectstack
 *************************************************************************/
extern "C" void* G__push_heapobjectstack(int tagnum,int num,int scopelevel) ;

/*************************************************************************
 * G__push_autoobjectstack
 *************************************************************************/
extern "C" void G__push_autoobjectstack(void *p,int tagnum,int num
					,int scopelevel,int isheap) ;


/*************************************************************************
 * G__push_autoobjectstack
 *************************************************************************/
extern "C" void G__delete_autoobjectstack(int scopelevel) ;

/*************************************************************************
 * G__allocheapobjectstack
 *************************************************************************/
extern "C" void* G__allocheapobjectstack(struct G__ifunc_table *ifunc,int ifn,int scopelevel);

/*************************************************************************
 * G__copyheapobjectstack
 *************************************************************************/
extern "C" void G__copyheapobjectstack(void* p,G__value* result,struct G__ifunc_table *ifunc,int ifn);

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
