#include "base/inc/LinkDef1.h"
#include "base/inc/LinkDef2.h"
#include "base/inc/LinkDef3.h"
#include "cont/inc/LinkDef.h"
#include "meta/inc/LinkDef.h"

#if defined(SYSTEM_TYPE_winnt)
#include "winnt/inc/LinkDef.h"
#elif defined(SYSTEM_TYPE_macosx)
#include "macosx/inc/LinkDef.h"
#include "unix/inc/LinkDef.h"
#elif defined(SYSTEM_TYPE_unix)
#include "unix/inc/LinkDef.h"
#else
# error "Unsupported system type."
#endif
