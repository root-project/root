// @(#)root/pyroot:$Name$:$Id$
// Author: Wim Lavrijsen   April 2004 

#ifndef ROOT_TPython
#define ROOT_TPython

// ROOT
class TObject;


class Python {
public:
// execute a python statement (eg. "import ROOT" )
   static void exec( char* cmd );

// evaluate a python expression (eg. "1+1")
   static TObject* eval( char* expr );

// bind a ROOT object with at the python side with name "label"
   static bool bind( TObject*, char* label );

// enter an interactive python session (exit with ^D)
   static void prompt();

private:
   static bool initialize_();
};

#endif
