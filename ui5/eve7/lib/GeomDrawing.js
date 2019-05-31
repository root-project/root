sap.ui.define(['sap/ui/core/Control',
               "sap/ui/core/ResizeHandler"
],function(CoreControl, ResizeHandler) {

   "use strict";

   var GeomDrawing = CoreControl.extend("rootui5.eve7.lib.GeomDrawing", {

      metadata : {
         properties : {           // setter and getter are created behind the scenes, incl. data binding and type validation
            "color" : {type: "sap.ui.core.CSSColor", defaultValue: "#fff"} // you can give a default value and more
         }
      },

      onBeforeRendering: function() {
         // remove Canvas and painter from DOM
         if (this.geom_painter) {
            this.geom_painter.clear_3d_canvas();
            this.geom_painter.AccessTopPainter(false);
         }
      },

      // the part creating the HTML:
      renderer : function(oRm, oControl) { // static function, so use the given "oControl" instance instead of "this" in the renderer function
         oRm.write("<div");
         oRm.writeControlData(oControl);  // writes the Control ID and enables event handling - important!
         // oRm.addStyle("background-color", oControl.getColor());  // write the color property; UI5 has validated it to be a valid CSS color
         oRm.addStyle("width", "100%");
         oRm.addStyle("height", "100%");
         oRm.addStyle("overflow", "hidden");
         oRm.writeStyles();
         oRm.writeClasses();              // this call writes the above class plus enables support for Square.addStyleClass(...)
         oRm.write(">");
         oRm.write("</div>"); // no text content to render; close the tag
      },

      onAfterRendering: function() {
         ResizeHandler.register(this, this.onResize.bind(this));

         if (this.geom_painter) {
            // this should be moved to GeomPainter itself !!!

            this.geom_painter.SetDivId(this.getDomRef(), 5);

            var size = this.geom_painter.size_for_3d();

            this.geom_painter.add_3d_canvas(size, this.geom_painter._renderer.domElement);

            // set top painter only when first child exists
            this.geom_painter.AccessTopPainter(true);

            this.geom_painter.Render3D();
         }
      },

      setGeomPainter: function(painter) {
         this.geom_painter = painter;
      },

      onResize: function() {
         if (this.resize_tmout) clearTimeout(this.resize_tmout);
         this.resize_tmout = setTimeout(this.onResizeTimeout.bind(this), 100); // minimal latency
      },

      onResizeTimeout: function() {
         delete this.resize_tmout;
         if (this.geom_painter)
            this.geom_painter.CheckResize();
      }
   });

   return GeomDrawing;

});

