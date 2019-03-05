sap.ui.define([
   'rootui5/panel/Controller',
   'sap/ui/model/json/JSONModel'
], function (panelController, JSONModel) {
   
   "use strict";

   return panelController.extend("rootui5.fitpanel.controller.FitPanel", {

      // function called from GuiPanelController
      onPanelInit : function() {
         var id = this.getView().getId();
         console.log("Initialization FitPanel id = " + id);
         // such data will be produced on server from TFitPanelModel
         var model = new JSONModel({
            fDataNames:[ { fId:"1", fName: "----" } ],
            fSelectDataId: "0",
            fModelNames: [ { fId:"1", fName: "----" } ],
            fSelectModelId: "0"
         });
         this.getView().setModel(model);
      },

      // function called from GuiPanelController
      onPanelExit : function() {
      },

      OnWebsocketMsg: function(handle, msg) {
         if (msg.indexOf("MODEL:")==0) {
            var json = msg.substr(6);
            var data = JSROOT.parse(json);

            if (data) {
               this.getView().setModel(new JSONModel(data));
               console.log('FitPanel set new model');
            }

         } else {
            console.log('FitPanel Get message ' + msg);
         }
      },

      handleFitPress : function() {
         console.log('Press fit');
         // To now with very simple logic
         // One can bind some parameters direct to the model and use values from model
         var v1 = this.getView().byId("FitData"),
             v2 = this.getView().byId("FitModel");

         if (this.websocket && v1 && v2)
            this.websocket.Send('DOFIT:"' + v1.getValue() + '","' + v2.getValue() + '"');
      }

   });

});
