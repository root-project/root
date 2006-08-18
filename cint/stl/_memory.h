/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
#ifndef _MEMORY_H
#define _MEMORY_H

/**********************************************************************
* auto_ptr
**********************************************************************/
template <class X> class auto_ptr {
private:
  X* ptr;
  mutable bool owns;
  //template<class Y> struct auto_ptr_ref { };
public:
  typedef X element_type;
  explicit auto_ptr(X* p = 0) : ptr(p), owns(p?true:false) {}
  auto_ptr(auto_ptr& a) {owns=a.owns; ptr=a.ptr; a.owns=0;}
  
  // this implementation may not be correct
  template <class T> auto_ptr(auto_ptr<T>& a) {owns=a.owns; ptr=a.release();}
  
#if 0
  // this does not exist in standard.
  template <class T> auto_ptr(T* a) {
    ptr=a;
    owns=true;
  }
#endif
  
  auto_ptr& operator=(auto_ptr& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.ptr;
      a.owns = 0;
    }
    return(*this);
  }
  
  // this implementation may not be correct
  template <class T> auto_ptr& operator=(auto_ptr<T>& a) {
    if (a.ptr != ptr) {
      if (owns) delete ptr;
      owns = a.owns;
      ptr = a.release();
    }
    return(*this);
  }
  
  ~auto_ptr() { if(owns) delete ptr; }
  
  X& operator*() const { return *ptr; }
  X* operator->() const { return ptr; }
  X* get() const { return ptr; }
  X* release() { owns=false; return ptr; }
  void reset(X* p=0) {
    if(p!=ptr || !owns) { 
      if(owns) delete ptr;  
      ptr = p; 
      owns = p?true:false;
    }
  }

  // auto_ptr conversions
  //auto_ptr(auto_ptr_ref<X>& x) { }
  //template<class Y> operator auto_ptr_ref<Y>() { return auto_ptr_ref<Y>(); }
  //template<class Y> operator auto_ptr<Y>() { return auto_ptr<T>(); }
};

#if defined(__CINT__) && !defined(__MAKECINT__)
template<class T,class X>
auto_ptr<T>& operator=(auto_ptr<T>& a,auto_ptr<X>& b) {
  a.reset(b.release());
  return(a);
}
#endif

#endif // _MEMORY_H
