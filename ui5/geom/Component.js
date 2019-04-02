sap.ui.define([
   'sap/ui/core/UIComponent', 'sap/ui/core/mvc/XMLView', "rootui5/eve7/model/BrowserModel"
], function(UIComponent, XMLView, BrowserModel) {
   "use strict";

   var Component = UIComponent.extend("rootui5.eve7.ComponentGeom", {
      metadata : {
         manifest: "json"
      },
      init: function() {
         UIComponent.prototype.init.apply(this, arguments);
         // this.getRouter().initialize();
      }
   });

   return Component;

});
