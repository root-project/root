/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Header file bc_debug.h
 ************************************************************************
 * Description:
 *  debugging features
 ************************************************************************
 * Copyright(c) 2004~2004  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifndef BC_DEBUG_H
#define BC_DEBUG_H

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

//#include "bc_inst.h"
//#include "bc_type.h"
//#include "bc_reader.h"

#include <deque>
//#include <string>
using namespace std;


////////////////////////////////////////////////////////////////////

/***********************************************************************
 * G__bc_funccall
 ***********************************************************************/
class G__bc_funccall {
 public:
  // ctor and dtor
  G__bc_funccall() 
    : m_bytecode(0), m_localmem(0), m_struct_offset(0), m_line_number(0)
     , m_libp(0) { } 

  G__bc_funccall(struct G__bytecodefunc* bc
                 ,long localmem,long struct_offset,int line_number
		 ,struct G__param *libp) 
    : m_bytecode(bc), m_localmem(localmem), m_struct_offset(struct_offset),
    m_line_number(line_number), m_libp(libp) { } 

  G__bc_funccall(const G__bc_funccall& x)
    : m_bytecode(x.m_bytecode), m_localmem(x.m_localmem)
    , m_struct_offset(x.m_struct_offset), m_line_number(x.m_line_number) 
    , m_libp(x.m_libp) { } 

  ~G__bc_funccall() { }

  void setlinenum(int line) { m_line_number = line; }

  // query functions
  struct G__input_file getifile() const; 
  int setstackenv(struct G__view* pview) const; 
  int disp(FILE* fout) const;

 private:
  struct G__bytecodefunc *m_bytecode; // 0 if not in function
  long m_localmem;                    // 0 if not in function
  long m_struct_offset;               // 0 if static function
  int m_line_number;                  
  struct G__param *m_libp;
};

/***********************************************************************
 * G__bc_funccallstack
 ***********************************************************************/
class G__bc_funccallstack {
 public:
  // ctor and dtor
  G__bc_funccallstack();
  ~G__bc_funccallstack();

  // push , pop,  used in G__exec_bytecode
  void push(struct G__bytecodefunc* bc
            ,long localmem,long struct_offset,int line_number
	    ,struct G__param* libp) {
    m_funccallstack.push_front(G__bc_funccall(bc,localmem,struct_offset
					     ,line_number,libp));
  }
  void pop() { if(m_funccallstack.size()) m_funccallstack.pop_front(); }

  void setlinenum(int line) 
    {if(m_funccallstack.size()) m_funccallstack[0].setlinenum(line);}

  // query functions, used in debug interface. 
  G__bc_funccall& getStackPosition(int i=0);
  int setstackenv(int i,struct G__view* pview) ;
  struct G__input_file getifile(int i) ;  
  int disp(FILE* fout) const;
  
 private:
  G__bc_funccall        m_staticenv;
  deque<G__bc_funccall> m_funccallstack;
};
/////////////////////////////////////////////////////////////////////

/***********************************************************************
 * static objects
 ***********************************************************************/
extern G__bc_funccallstack G__bc_funccallstack_obj;

/***********************************************************************
 * C function wrappers
 ***********************************************************************/
extern "C" int G__bc_setdebugview(int i,struct G__view* pview) ;
extern "C" int G__bc_showstack(FILE* fout);
extern "C" void G__bc_setlinenum(int line);

/////////////////////////////////////////////////////////////////////

#endif
