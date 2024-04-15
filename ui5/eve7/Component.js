sap.ui.define([
   'sap/ui/core/UIComponent',
   'sap/ui/core/mvc/XMLView',
   'sap/m/routing/Router' /* Router is only needed for packaging */
], function(UIComponent) {
   "use strict";

   return UIComponent.extend("rootui5.eve7.Component", {
      metadata : {
         manifest: "json"
      },
      init: function() {
         UIComponent.prototype.init.apply(this, arguments);
         this.getRouter().initialize();
      }
   });
});
