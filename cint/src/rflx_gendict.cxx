//$Id: rflx_gendict.cxx,v 1.1 2005/11/16 14:58:14 roiser Exp $

#include "rflx_gendict.h"
#include "rflx_gensrc.h"
//#include "Reflex/Reflex.h"

#include "G__ci.h"
#include "global.h"

#include <iostream>

void rflx_gendict(const char * linkfilename,
                  const char * sourcefile) {
  rflx_gensrc gensrc(linkfilename, sourcefile);
  gensrc.gen_file();
}
