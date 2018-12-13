sap.ui.define([
   'sap/ui/core/mvc/Controller',
   'sap/ui/model/json/JSONModel',
   "sap/ui/core/ResizeHandler"
], function (Controller, JSONModel, ResizeHandler) {

   "use strict";

   return Controller.extend("eve.GeomDraw", {

      onInit : function() {
         var id = this.getView().getId();
         console.log("eve.GeomDraw.onInit id = ", id);

         var data = this.getView().getViewData();
      },

      onAfterRendering: function() {
         console.log("GeomDraw: on after rendering");
         ResizeHandler.register(this.getView(), this.onResize.bind(this));
      },

      onResize: function(event) {
         // use timeout
         // console.log("resize painter")
         if (this.resize_tmout) clearTimeout(this.resize_tmout);
         this.resize_tmout = setTimeout(this.onResizeTimeout.bind(this), 100); // minimal latency
      },

      onResizeTimeout: function() {
         delete this.resize_tmout;
         
         // TODO: should be specified somehow in XML file
         this.getView().$().css("overflow", "hidden").css("width", "100%").css("height", "100%");
         
         console.log('GeomDraw: COMPLETE RESIZE HANDLING');
         
         //if (this.geo_painter)
         //   this.geo_painter.CheckResize();
      }

   });

});
