// @(#)root/qt:$Name:  $:$Id: TQtTermInputHandler.h,v 1.3 2005/02/24 22:20:20 fine Exp $
// Author: Valeri Fine   02/03/2005
/****************************************************************************
** $Id: TQtTermInputHandler.h,v 1.3 2005/02/24 22:20:20 fine Exp $
**
** Copyright (C) 2004 by Valeri Fine.  All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.
*****************************************************************************/

#ifndef ROOT_TQtTermInputHandler
#define ROOT_TQtTermInputHandler

#include "TSysEvtHandler.h"
#ifndef __CINT__
#include <qobject.h>
#endif  

//
// TQtTermInputHandler is to process the stdin ROOT events from within Qt event loop
// It would be nice to derive it from the TTermInputHandler class.
// Unfortunately this class is defined locally within TRint.cxx file scope
//

//___________________________________________________________________
class  TQtTermInputHandler  : 
#ifndef __CINT__
  public QObject,
#endif  
  public TFileHandler {
#ifndef __CINT__
 Q_OBJECT
#endif  
public:
   TQtTermInputHandler (Int_t fd=0);
   Bool_t Notify();
   Bool_t ReadNotify() { return Notify(); }
public slots:
   void Activate(int fd);
};

#endif
