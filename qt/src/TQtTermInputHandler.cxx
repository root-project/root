// @(#)root/qt:$Name:  $:$Id: TQtTermInputHandler.cxx,v 1.2 2004/07/28 00:12:41 rdm Exp $
// Author: Valeri Fine   25/01/2005

/****************************************************************************
** $Id: TQtTermInputHandler.cxx,v 1.1 2005/02/04 19:47:27 fine Exp $
**
** Copyright (C) 2004 by Valeri Fine.  All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.
*****************************************************************************/


#include "TQtTermInputHandler.h"
#include "TApplication.h"
#include <qsocketnotifier.h> 

// TQtTermInputHandler is process the stdin ROOT event from within Qt event loop
// It would be nice to derive it from the TTermInputHandler class.
// Unfortunately this class is defined locally within TRint.cxx file scope

//______________________________________________________________________________
TQtTermInputHandler::TQtTermInputHandler (Int_t fd) : TFileHandler(fd, 1) 
{ 
   QSocketNotifier *sn;
   sn = new QSocketNotifier(fd, QSocketNotifier::Read, this,"QtTermInputHandler");
   QObject::connect( sn, SIGNAL(activated(int)),this, SLOT(Activate (int)) );
}
//______________________________________________________________________________
Bool_t TQtTermInputHandler::Notify()
{
   return gApplication->HandleTermInput();
}
//______________________________________________________________________________
void TQtTermInputHandler::Activate(int /*fd*/){
  //  Qt slot to activate ROOT TFileHandler from the Qt event loop
  //QSocketNotifier *sn = sender();
  //if (sn->Type() == QSocketNotifier::Read)
#if ROOT_VERSION_CODE > ROOT_VERSION(4,00,8)
     SetReadReady();
#else     
     fprintf(stderr,"Tour ROTO version is too OLD !!!\n"
     "You have to update your ROOT version to use to access ROOT prompt"
     "from the Qt event loop. Sorry !!!\n");
#endif   
  //else if (sn->Type() == QSocketNotifier::Write)
  //   SetWriteReady();
}
