sap.ui.define([
   'sap/ui/core/UIComponent', 'sap/ui/core/mvc/XMLView', "rootui5/geom/model/BrowserModel"
], function(UIComponent, XMLView, BrowserModel) {
   "use strict";

   var Component = UIComponent.extend("rootui5.geom.Component", {
      metadata : {
         manifest: "json"
      },
      init: function() {
         UIComponent.prototype.init.apply(this, arguments);

         console.log('Creating browser model!!!');
         this.setModel(new BrowserModel());
         // this.getRouter().initialize();
      }
   });

   return Component;

});
