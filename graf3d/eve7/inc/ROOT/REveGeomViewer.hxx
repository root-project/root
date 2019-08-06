// @(#)root/eve7:$Id$
// Author: Sergey Linev, 13.12.2018

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_REveGeomViewer
#define ROOT7_REveGeomViewer

#include <ROOT/RWebDisplayArgs.hxx>
#include <ROOT/REveGeomData.hxx>

#include <memory>

class TGeoManager;

namespace ROOT {
namespace Experimental {

class RWebWindow;
class REveManager;

class REveGeomViewer {

   friend class REveManager;

protected:

   TGeoManager *fGeoManager{nullptr};        ///<! geometry to show
   REveGeomDescription fDesc;                ///<! geometry description, send to the client as first message

   std::shared_ptr<RWebWindow> fWebWindow;   ///<! web window to show geometry

   void WebWindowCallback(unsigned connid, const std::string &arg);

   std::vector<int> GetStackFromJson(const std::string &json, bool node_ids = false);

   void SendGeometry(unsigned connid);

public:

   REveGeomViewer(TGeoManager *mgr = nullptr);
   virtual ~REveGeomViewer();

   void SetGeometry(TGeoManager *mgr);

   void SelectVolume(const std::string &volname);

   /** Configures maximal number of visible nodes and faces */
   void SetLimits(int nnodes = 5000, int nfaces = 100000)
   {
      fDesc.SetMaxVisNodes(nnodes);
      fDesc.SetMaxVisFaces(nfaces);
   }

   /** Configures default draw option for geometry */
   void SetDrawOptions(const std::string &opt) { fDesc.SetDrawOptions(opt); }

   void Show(const RWebDisplayArgs &args = "", bool always_start_new_browser = false);

   void Update();

};

}}

#endif
