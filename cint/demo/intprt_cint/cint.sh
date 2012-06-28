/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (root-cint@cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
echo 'CINT:: This is very slow. CINT interpreter interprets itself.'
echo 
cint.exe -I$CINTSYSDIR -I$CINTSYSDIR/src +P testmain.c $*

