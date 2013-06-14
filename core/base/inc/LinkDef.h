#include "core/base/inc/LinkDef1.h"
#include "core/base/inc/LinkDef2.h"
#include "core/base/inc/LinkDef3.h"
#include "core/clib/inc/LinkDef.h"
#include "core/cont/inc/LinkDef.h"
#include "core/meta/inc/LinkDef.h"
#include "core/metautils/inc/LinkDef.h"
#include "core/textinput/inc/LinkDef.h"
#include "core/zip/inc/LinkDef.h"

#if defined(SYSTEM_TYPE_winnt)
#include "core/winnt/inc/LinkDef.h"
#elif defined(SYSTEM_TYPE_macosx)
#include "core/macosx/inc/LinkDef.h"
#include "core/unix/inc/LinkDef.h"
#else
#include "core/unix/inc/LinkDef.h"
#endif
