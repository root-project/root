/* -*- C++ -*- */
/*************************************************************************
 * Copyright(c) 1995~2005  Masaharu Goto (cint@pcroot.cern.ch)
 *
 * For the licensing terms see the file COPYING
 *
 ************************************************************************/
//$Id: rflx_gendict.cxx,v 1.1.1.2 2007/09/10 15:57:08 axel Exp $

#include "rflx_gendict.h"
#include "rflx_gensrc.h"
//#include "Reflex/Reflex.h"

#include "G__ci.h"
#include "common.h"
#include "global.h"

#include <iostream>

void rflx_gendict(const char *linkfilename, const char *sourcefile)
{
   rflx_gensrc gensrc(linkfilename, sourcefile);
   gensrc.gen_file();
}
