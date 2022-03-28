sap.ui.define([], function() {

   "use strict";

   class GlViewer {

      constructor(viewer_class)  {
         this.viewer_class = viewer_class;

         // console.log(this.get_name() + " - constructor");
      }

      init(controller)
      {
         // console.log(this.get_name() + ".init()");

         if (this.controller) throw new Error(this.get_name() + "already initialized.");

         this.controller = controller;
      }

      cleanup()
      {
         // console.log(this.get_name() + ".cleanup()");
         delete this.controller;
      }

      //==============================================================================

      get_name() { return "EVE.GlViewer" + this.viewer_class; }
      get_view() { return this.controller.getView(); }
      get_width() { return this.controller.getView().$().width(); }
      get_height() { return this.controller.getView().$().height(); }

      //==============================================================================

      make_object(/* name */) { return null; }
      get_top_scene()  { return null; }
      get_manager()  { return this.controller.mgr; }

   } // class GlViewer

   return GlViewer;

});
