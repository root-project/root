// @(#)root/win32:$Name$:$Id$
// Author: Valery Fine   27/03/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TError
#include "TError.h"
#endif


#ifndef ROOT_TVirtualMenuItem
#include "TWin32MenuItem.h"
#endif.


class TCustomMenuItem : public TWin32MenuItem {

private:
    void SetItemStatus(EMenuModified state){ ; }

protected:
    void ExecuteEventCB(TWin32Canvas *c){ ; }

public:
    const char *ClassName(){ return "TCustomMenuItem"; }
    void Checked(){Warning("Checked","Can't check/uncheck a separator") ; }

    void Disable(){Warning("Disable","Can't Enable/Disable a separator") ; }
    void Enable(){Warning("Enable","Can't Enable/Disable a separator") ; }
    void ExecuteEvent(TWin32Canvas *c){ ; }
    TWin32Menu *GetPopUpItem(){ return 0;}

    void Grayed(){Warning("Grayed","Can't Gray a separator") ; };
    void UnChecked(){Warning("UnChecked","Can't check/uncheck a separator") ; }

};
