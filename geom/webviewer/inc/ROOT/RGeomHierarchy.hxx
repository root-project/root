// Author: Sergey Linev, 3.03.2023

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RGeomHierarchy
#define ROOT7_RGeomHierarchy

#include <ROOT/RWebDisplayArgs.hxx>
#include <ROOT/RGeomData.hxx>

#include <memory>

class TGeoManager;
class TGeoVolume;

namespace ROOT {

class RWebWindow;

class RGeomHierarchy {

protected:

   RGeomDescription &fDesc;                  ///<! geometry description, shared with external
   std::shared_ptr<RWebWindow> fWebWindow;   ///<! web window to show geometry

   void WebWindowCallback(unsigned connid, const std::string &arg);

   void ProcessSignal(const std::string &kind);

public:

   RGeomHierarchy(RGeomDescription &desc, bool use_server_threads = false);
   virtual ~RGeomHierarchy();

   void Show(const RWebDisplayArgs &args = "");

   void Update();

   void BrowseTo(const std::string &itemname);

   RGeomDescription &Description() { return fDesc; }

   void ClearOnClose(const std::shared_ptr<void> &handle);
};

} // namespace ROOT

#endif
