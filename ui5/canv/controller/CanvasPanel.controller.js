sap.ui.define([
   'sap/ui/core/mvc/Controller',
   'sap/ui/core/ResizeHandler'
], function (Controller, ResizeHandler) {
   "use strict";

   return Controller.extend("sap.ui.jsroot.controller.CanvasPanel", {

      onBeforeRendering: function() {
      },

      setPainter: function(painter) {
         this.canvas_painter = painter;
      },

      getPainter: function() {
         return this.canvas_painter;
      },

      onAfterRendering: function() {
         if (this.canvas_painter && this.canvas_painter._window_handle) {
            this.canvas_painter.SetDivId(this.getView().getDomRef(), -1);
            this.canvas_painter.UseWebsocket(this.canvas_painter._window_handle);
            delete this.canvas_painter._window_handle;
         }
      },

      onResize: function(event) {
         // use timeout
         if (this.resize_tmout) clearTimeout(this.resize_tmout);
         this.resize_tmout = setTimeout(this.onResizeTimeout.bind(this), 300); // minimal latency
      },

      drawCanvas : function(can, opt, call_back) {
         if (this.canvas_painter) {
            this.canvas_painter.Cleanup();
            delete this.canvas_painter;
         }

         if (!this.getView().getDomRef()) return JSROOT.CallBack(call_back, null);

         var oController = this;
         JSROOT.draw(this.getView().getDomRef(), can, opt, function(painter) {
            oController.canvas_painter = painter;
            JSROOT.CallBack(call_back, painter);
         });
      },

      onResizeTimeout: function() {
         delete this.resize_tmout;
         if (this.canvas_painter)
            this.canvas_painter.CheckCanvasResize();
      },

      onInit: function() {
         // this.canvas_painter = JSROOT.openui5_canvas_painter;
         // delete JSROOT.openui5_canvas_painter;

         console.log("INIT CANVAS PANEL");

/*
         console.log(sap.ui.getCore().byId("TopCanvasId").getViewData());

         var oModel = sap.ui.getCore().getModel(this.getView().getId());
         if (oModel) {
            var oData = oModel.getData();

            if (oData.canvas_painter) {
               this.canvas_painter = oData.canvas_painter;
               delete oData.canvas_painter;
            }
         }*/

         ResizeHandler.register(this.getView(), this.onResize.bind(this));
      },

      onExit: function() {
         if (this.canvas_painter) {
            this.canvas_painter.Cleanup();
            delete this.canvas_painter;
         }
      }
   });

});
