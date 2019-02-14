#include "core/base/inc/LinkDef1.h"
#include "core/base/inc/LinkDef2.h"
#include "core/base/inc/LinkDef3.h"
#include "core/cont/inc/LinkDef.h"
#include "core/meta/inc/LinkDef.h"

#if defined(SYSTEM_TYPE_winnt)
#include "core/winnt/inc/LinkDef.h"
#elif defined(SYSTEM_TYPE_macosx)
#include "core/macosx/inc/LinkDef.h"
#include "core/unix/inc/LinkDef.h"
#elif defined(SYSTEM_TYPE_unix)
#include "core/unix/inc/LinkDef.h"
#else
# error "Unsupported system type."
#endif
