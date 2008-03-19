/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
/*********************************************************************
* parseplot.c
*
*  PERL/AWK like text processing script example.
*  
*  Read "data" file, parse it and get X/Y value from it.
* Then plot an XY graph.
*
*********************************************************************/

// Cint defines following utility classes
#include <regexp.h>     // RegExp
#include <readfile.h>   // ReadFile
#include <array.h>      // array class and XYgraph object

#define BUFSIZE 2048

main()
{
  // Create regular expression patterns
  RegExp X=RegExp("x[ \t]*=");
  RegExp Y=RegExp("y[ \t]*=");
  RegExp End=RegExp("end");

  // Instantiate array buffer of BUFSIZE
  array x=array(0,0,BUFSIZE),y;

  // Read the file
  ReadFile f=ReadFile("data");  // Open file "data" for reading 
  if(!f.isvalid()) exit();
  f.setendofline("#");          // #... is handled as comment
  int ix=0,iy=0;
  while(f.read()) {             // Parse lines
    if(X==f.argv[0])         x[ix++]=atof(f.argv[3]);
    else if(Y==f.argv[0])    y[iy++]=atof(f.argv[3]);
    else if(f.argv[0]==End)  break;
  }

  // Resize the array buffer
  if(ix<=iy) int truesize=ix;
  else       int truesize=iy;
  x.resize(truesize);
  y.resize(truesize);

  // Plot XY graph with some calculation (you need 'xgraph')
  plot << x << y << y*y/5 << endl;
}
