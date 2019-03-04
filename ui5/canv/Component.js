sap.ui.define([
   "sap/ui/core/UIComponent"
], function (UIComponent, JSONModel, ResourceModel) {
   "use strict";
   return UIComponent.extend("sap.ui.jsroot.Component", {
       metadata : {
         rootView: "sap.ui.jsroot.view.Canvas",
         dependencies : {
            libs : [ "sap.m" ]
         },
         config : {
            sample : {
               stretch: true,
               files : [ "Canvas.view.xml", "Canvas.controller.js" ]
            }
         }
       },
       init : function () {
         // call the init function of the parent
         UIComponent.prototype.init.apply(this, arguments);
      }
   });
});