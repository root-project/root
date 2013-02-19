#!/bin/sh

# create first one large pch
./build/unix/makeonepch.sh
err=$?
if [ $err -ne 0 ]; then
   echo "Failed to make one pch"
   exit $err
fi

# create one large allLinkDef.h
rm -f include/allLinkDef.h

# create allLinkDef.h including all LinkDefs
find */inc */*/inc -name \*LinkDef\*.h | \
  grep -v -e '^./test/' -e '^./roottest/' -e '/RooFitCore_LinkDef.h$' | \
  sed -e 's|^|#include "|' -e 's|$|"|' > alldefs.h

mv alldefs.h include/allLinkDef.h

# generate one large pcm
rm -f core/base/src/allDict.* lib/allDict_rdict.pcm
core/utils/src/rootcling_tmp -1 -f core/base/src/allDict.cxx -c -Iinclude allHeaders.h include/allLinkDef.h

exit $?
