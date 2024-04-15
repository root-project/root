#include "LinkDef1.h"
#include "LinkDef2.h"
#include "LinkDef3.h"
#include "../../cont/inc/LinkDef.h"
#include "../../meta/inc/LinkDef.h"

#if defined(SYSTEM_TYPE_winnt)
#include "../../winnt/inc/LinkDef.h"
#elif defined(SYSTEM_TYPE_macosx)
#if defined(R__HAS_COCOA)
#include "../../macosx/inc/LinkDef.h"
#endif
#include "../../unix/inc/LinkDef.h"
#elif defined(SYSTEM_TYPE_unix)
#include "../../unix/inc/LinkDef.h"
#else
# error "Unsupported system type."
#endif
