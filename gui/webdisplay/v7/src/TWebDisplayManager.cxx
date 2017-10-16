/// \file TWebDisplayManager.cxx
/// \ingroup WebGui ROOT7
/// \author Sergey Linev <s.linev@gsi.de>
/// \date 2017-10-16
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TWebDisplayManager.hxx"

#include "THttpServer.h"

const std::shared_ptr<ROOT::Experimental::TWebDisplayManager> &ROOT::Experimental::TWebDisplayManager::Get()
{
   static std::shared_ptr<ROOT::Experimental::TWebDisplayManager> sManager;
   return sManager;
}

//static std::shared_ptr<ROOT::Experimental::TWebDisplayManager> ROOT::Experimental::TWebDisplayManager::Create()
//{
//   return std::make_shared<ROOT::Experimental::TWebDisplayManager>();
//}
