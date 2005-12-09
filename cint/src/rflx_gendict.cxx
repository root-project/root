//$Id: rflx_gendict.cxx,v 1.2 2005/12/08 15:45:24 axel Exp $

#include "rflx_gendict.h"
#include "rflx_gensrc.h"
//#include "Reflex/Reflex.h"

#include "G__ci.h"
#include "global.h"

#include <iostream>

void rflx_gendict(const char *linkfilename, const char *sourcefile)
{
   rflx_gensrc gensrc(linkfilename, sourcefile);
   gensrc.gen_file();
}
