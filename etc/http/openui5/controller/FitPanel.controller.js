sap.ui.define([
   'sap/ui/core/mvc/Controller',
   'sap/ui/model/json/JSONModel'
], function (Controller, JSONModel) {
   "use strict";

   return Controller.extend("sap.ui.jsroot.controller.FitPanel", {

      onInit : function() {
         var model = new JSONModel({ SelectedClass: "none" });
         this.getView().setModel(model);
      },

      onExit : function() {
      },

      handleFitPress : function() {
      },

      handleClosePress : function() {
         var main = sap.ui.getCore().byId("TopCanvasId");
         if (main) main.getController().showLeftArea("");
      }

   });

});
