/* /% C %/ */
/***********************************************************************
 * cint (C/C++ interpreter)
 ************************************************************************
 * Source file end.c
 ************************************************************************
 * Description:
 *  Cleanup function
 ************************************************************************
 * Copyright(c) 1995~2001  Masaharu Goto 
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/

#include "common.h"

#ifdef G__SHAREDLIB
extern std::list<G__DLLINIT>* G__initpermanentsl;
#endif //G__SHAREDLIB

extern "C" {

// Static functions.
// -- None.

// External functions.
int G__call_atexit();
int G__interpretexit();

// Functions in the C interface.
void G__exit(int rtn);

//______________________________________________________________________________
//
//  Static functions.
//

//______________________________________________________________________________
//
//  External functions.
//

//______________________________________________________________________________
int G__call_atexit()
{
   // -- Execute atexit function.
   // Note: atexit is reset before calling the function to avoid recursive atexit call.
   G__FastAllocString temp(G__ONELINE);
   if (G__breaksignal) {
      G__fprinterr(G__serr, "!!! atexit() call\n");
   }
   G__ASSERT(G__atexit);
   temp = G__atexit;
   temp += "()";
   G__atexit = 0;
   G__getexpr(temp);
   return 0;
}

//______________________________________________________________________________
int G__interpretexit()
{
   if (G__atexit) {
      G__call_atexit();
   }
   G__scratch_all();
   // FIXME: Do we need to G__Lock/UnlockCriticalSection here?
#ifdef G__SHAREDLIB
   delete G__initpermanentsl;
   G__initpermanentsl = 0;
#endif //G__SHAREDLIB
   if (G__breaksignal) {
      G__fprinterr(G__serr, "\nEND OF EXECUTION\n");
   }
   return 0;
}

//______________________________________________________________________________
//
//  Functions in the C interface.
//

//______________________________________________________________________________
void G__exit(int rtn)
{
   G__scratch_all();
   fflush(G__sout);
   fflush(G__serr);
   exit(rtn);
}

} // extern "C"

/*
 * Local Variables:
 * c-tab-always-indent:nil
 * c-indent-level:3
 * c-continued-statement-offset:3
 * c-brace-offset:-3
 * c-brace-imaginary-offset:0
 * c-argdecl-indent:0
 * c-label-offset:-3
 * compile-command:"make -k"
 * End:
 */
