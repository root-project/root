/// \file rootwebpage.cpp
/// \ingroup CanvasPainter ROOT7
/// \author Sergey Linev <S.Linev@gsi.de>
/// \date 2017-06-29
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "rootwebpage.h"

#include <stdio.h>

void RootWebPage::javaScriptConsoleMessage(JavaScriptConsoleMessageLevel, const QString &message, int lineNumber,
                                           const QString &sourceID)
{
   QByteArray ba = message.toLatin1();
   QByteArray src = sourceID.toLatin1();

   printf("CONSOLE %s:%d: %s\n", src.data(), lineNumber, ba.data());
}
