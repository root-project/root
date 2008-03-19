/* /% C++ %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file main/G__setup.c
 ************************************************************************
 * Description:
 *  Archived precompiled library initialization routine
 ************************************************************************
 * Copyright(c) 1995~1999  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#ifdef G__CXXLINK_ON
void G__cpp_setup();
#endif
#ifdef G__CLINK_ON
void G__c_setup();
#endif

int G__globalsetup() {
#ifdef G__CXXLINK_ON
  G__cpp_setup();
#endif
#ifdef G__CLINK_ON
  G__c_setup();
#endif
  return(0);
}

