/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * I/O manipulator header for iomanip.h
 ************************************************************************
 * Description:
 *  CINT IOMANIP header file
 ************************************************************************
 * Author                  Masaharu Goto 
 * Copyright(c) 1995~1999  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifndef G__IOMANIP_H
#define G__IOMANIP_H

#include <iostream.h>

#ifdef G__IOMANIP_TEMPLATE

/*********************************************************************
* template implementation of iomanip , not used now
*********************************************************************/

////////////////////////////////////////////////////////////////////////
template<class T,class E> class G__CINT_IOMANIP {
  public:
   E i;
   G__CINT_IOMANIP(int in) { i=in; }
};

////////////////////////////////////////////////////////////////////////
template<class T,class E> ostream& operator<<(ostream& ostr,G__CINT_IOMANIP<T,E>& i) {
  ostr.T(i.i);
  return(ostr);
}

////////////////////////////////////////////////////////////////////////
template<class T,class E> istream& operator<<(istream& istr,G__CINT_IOMANIP<T,E>& i) {
  istr.T(i.i);
  return(istr);
}

////////////////////////////////////////////////////////////////////////
typedef G__CINT_IOMANIP<width,int> setw;
typedef G__CINT_IOMANIP<fill,int> setfill;
typedef G__CINT_IOMANIP<setf,int> setioflags;
typedef G__CINT_IOMANIP<unsetf,int> resetioflags;
typedef G__CINT_IOMANIP<precision,int> setprecision;

////////////////////////////////////////////////////////////////////////


#else // G__IOMANIP_TEMPLATE

/*********************************************************************
* flat implementation of iomanip
*********************************************************************/

////////////////////////////////////////////////////////////////////////
class setw { 
 public: 
  int i; 
  setw(int in) {i=in;}
} ;
ostream& operator<<(ostream& ostr,setw& i) {
 ostr.width(i.i);
 return(ostr);
}

////////////////////////////////////////////////////////////////////////
class setfill { 
 public: 
  int i; 
  setfill(int in) {i=in;}
} ;
ostream& operator<<(ostream& ostr,setfill& i) {
 ostr.fill(i.i);
 return(ostr);
}

////////////////////////////////////////////////////////////////////////
class setiosflags { 
 public: 
  int i; 
  setiosflags(int in) {i=in;}
} ;
ostream& operator<<(ostream& ostr,setiosflags& i) {
 ostr.setf(i.i);
 return(ostr);
}

////////////////////////////////////////////////////////////////////////
class resetiosflags { 
 public: 
  int i; 
  resetiosflags(int in) {i=in;}
} ;
ostream& operator<<(ostream& ostr,resetiosflags& i) {
 ostr.unsetf(i.i);
 return(ostr);
}

////////////////////////////////////////////////////////////////////////
class setprecision { 
 public: 
  int i; 
  setprecision(int in) {i=in;}
} ;
ostream& operator<<(ostream& ostr,setprecision& i) {
 ostr.precision(i.i);
 return(ostr);
}
#endif // G__IOMANIP_TEMPLATE

////////////////////////////////////////////////////////////////////////
class setbase { 
 public: 
  int i; 
  setbase(int in) {i=in;}
} ;
ostream& operator<<(ostream& ostr,setbase& i) {
#pragma ifndef G__TMPLTIOS
 if(8==i.i)       ostr.flags(ios::oct);
 else if(10==i.i) ostr.flags(ios::dec);
 else if(16==i.i) ostr.flags(ios::hex);
#pragma else
 if(8==i.i)       ostr.flags(ios_base::oct);
 else if(10==i.i) ostr.flags(ios_base::dec);
 else if(16==i.i) ostr.flags(ios_base::hex);
#pragma endif
 return(ostr);
}
istream& operator>>(istream& istr,setbase& i) {
#pragma ifndef G__TMPLTIOS
 if(8==i.i)       istr.flags(ios::oct);
 else if(10==i.i) istr.flags(ios::dec);
 else if(16==i.i) istr.flags(ios::hex);
#pragma else
 if(8==i.i)       istr.flags(ios_base::oct);
 else if(10==i.i) istr.flags(ios_base::dec);
 else if(16==i.i) istr.flags(ios_base::hex);
#pragma endif
 return(istr);
}

////////////////////////////////////////////////////////////////////////
#ifdef G__OLDIMPLEMENTATION843
// avoid loop compilation abort, workaround
setw setw(int in) { setw a(in); return a; }
setfill setfill(int in) { setfill a(in); return a; }
setiosflags setiosflags(int in) { setiosflags a(in); return a; }
resetiosflags resetiosflags(int in) { resetiosflags a(in); return a; }
setprecision setprecision(int in) { setprecision a(in); return a; }
setbase setbase(int in) { setbase a(in); return a; }
#endif

////////////////////////////////////////////////////////////////////////

#endif // G__IOMANIP_H

