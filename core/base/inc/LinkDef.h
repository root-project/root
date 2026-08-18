#include "LinkDef1.h"
#include "LinkDef2.h"
#include "LinkDef3.h"
#include "../../cont/inc/LinkDef.h"
#include "../../meta/inc/LinkDef.h"

#if defined(SYSTEM_TYPE_winnt)
#include "../../os/winnt/inc/LinkDef.h"
#elif defined(SYSTEM_TYPE_macosx)
#if defined(R__HAS_COCOA)
#include "../../os/macosx/inc/LinkDef.h"
#endif
#include "../../os/unix/inc/LinkDef.h"
#elif defined(SYSTEM_TYPE_unix)
#include "../../os/unix/inc/LinkDef.h"
#else
# error "Unsupported system type."
#endif
