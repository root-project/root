sap.ui.define([
   'sap/ui/core/mvc/Controller',
   'sap/ui/model/json/JSONModel'
], function (Controller, JSONModel) {
   "use strict";

   return Controller.extend("rootui5.eve7.controller.Ged", {

      onInit : function() {
         console.log('init GED editor');
      },

      onExit : function() {
         console.log('exit GED editor');
      },

      closeGedEditor: function() {
         console.log('close GED editor');
         this.getView().getViewData().summaryCtrl.closeGedEditor();
      }

   });

});
