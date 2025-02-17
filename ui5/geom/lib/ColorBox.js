sap.ui.define([
   'sap/ui/core/Control'
], function(CoreControl) {

   "use strict";

   return CoreControl.extend("rootui5.geom.lib.ColorBox", { // call the new Control type "my.ColorBox" and let it inherit from sap.ui.core.Control

      // the control API:
      metadata : {
         properties : {           // setter and getter are created behind the scenes, incl. data binding and type validation
            "color" : { type: "sap.ui.core.CSSColor", defaultValue: "#fff"} // you can give a default value and more
         }
      },

      // the part creating the HTML:
      renderer(oRm, oControl) { // static function, so use the given "oControl" instance instead of "this" in the renderer function
         // if (!oControl.getVisible()) return;

         if (!oControl.getColor() || (oControl.getColor() == '#fff')) return;

         oRm.write("<div");
         oRm.writeControlData(oControl);  // writes the Control ID and enables event handling - important!
         oRm.addStyle("background-color", oControl.getColor());  // write the color property; UI5 has validated it to be a valid CSS color
         oRm.writeStyles();
         oRm.addClass("geomColorBox");      // add a CSS class for styles common to all control instances
         oRm.writeClasses();              // this call writes the above class plus enables support for Square.addStyleClass(...)
         oRm.write(">");
         oRm.write("</div>"); // no text content to render; close the tag
      }

/*
      // an event handler:
      onclick(evt) {   // is called when the Control's area is clicked - no further event registration required
         sap.ui.require([
            'sap/ui/unified/ColorPickerPopover'
         ], function (ColorPickerPopover) {
            if (!this.oColorPickerPopover) {
               this.oColorPickerPopover = new ColorPickerPopover({
                  change: this.handleChange.bind(this)
               });
            }
            this.oColorPickerPopover.setColorString(this.getColor());
            this.oColorPickerPopover.openBy(this);
         }.bind(this));
      },

      handleChange(oEvent) {
         let newColor = oEvent.getParameter("colorString");
         this.setColor(newColor);
         // TODO: fire a "change" event, in case the application needs to react explicitly when the color has changed
         // but when the color is bound via data binding, it will be updated also without this event
      }
*/

   });

});
