sap.ui.define([
   'sap/ui/core/Control',
   "sap/ui/core/ResizeHandler"
], function (Control, ResizeHandler) {

   "use strict";

   return Control.extend("eve.GeomDraw", { // call the new Control type "my.ColorBox" and let it inherit from sap.ui.core.Control

      // the control API:
      metadata : {
         properties : {           // setter and getter are created behind the scenes, incl. data binding and type validation
            "color" : {type: "sap.ui.core.CSSColor", defaultValue: "#fff"} // you can give a default value and more
         }
      },

      // the part creating the HTML:
      renderer : function(oRm, oControl) { // static function, so use the given "oControl" instance instead of "this" in the renderer function
         oRm.write("<div"); 
         oRm.writeControlData(oControl);  // writes the Control ID and enables event handling - important!
         oRm.addStyle("background-color", oControl.getColor());  // write the color property; UI5 has validated it to be a valid CSS color
         oRm.addStyle("width", "100%");  // write the color property; UI5 has validated it to be a valid CSS color
         oRm.addStyle("height", "100%");  // write the color property; UI5 has validated it to be a valid CSS color
         oRm.addStyle("overflow", "hidden");  // write the color property; UI5 has validated it to be a valid CSS color
         oRm.writeStyles();
         oRm.addClass("myColorBox");      // add a CSS class for styles common to all control instances
         oRm.writeClasses();              // this call writes the above class plus enables support for Square.addStyleClass(...)
         oRm.write(">"); 
         oRm.write("</div>"); // no text content to render; close the tag
      },

      // an event handler:
      onclick : function(evt) {   // is called when the Control's area is clicked - no further event registration required
      },
      
      onAfterRendering: function() {
         ResizeHandler.register(this, this.onResize.bind(this));
      },
      
      onResize: function() {
         if (this.resize_tmout) clearTimeout(this.resize_tmout);
         this.resize_tmout = setTimeout(this.onResizeTimeout.bind(this), 100); // minimal latency
      },
      
      onResizeTimeout: function() {
         delete this.resize_tmout;
         console.log('Resize GEOM drawing'); 
      },

      handleChange: function (oEvent) {
         var newColor = oEvent.getParameter("colorString");
         this.setColor(newColor);
         // TODO: fire a "change" event, in case the application needs to react explicitly when the color has changed
         // but when the color is bound via data binding, it will be updated also without this event
      }
   });

});
