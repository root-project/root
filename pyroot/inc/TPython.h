// @(#)root/pyroot:$Name:  $:$Id: TPython.h,v 1.5 2004/07/17 06:32:15 brun Exp $
// Author: Wim Lavrijsen   April 2004

#ifndef ROOT_TPython
#define ROOT_TPython

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
// TPython                                                                  //
//                                                                          //
// Access to the python interpreter.                                        //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////


// Bindings
#include "TPyReturn.h"

// ROOT
#ifndef ROOT_TObject
#include "TObject.h"
#endif


class TPython {

private:
   static Bool_t Initialize();

public:
// execute a python statement (e.g. "import ROOT" )
   static void Exec( const char* cmd );

// evaluate a python expression (e.g. "1+1")
   static const TPyReturn& Eval( const char *expr );

// bind a ROOT object with, at the python side, the name "label"
   static Bool_t Bind( TObject* obj, const char* label );

// enter an interactive python session (exit with ^D)
   static void Prompt();

   ClassDef(TPython,0)   //Access to the python interpreter
};

#endif
