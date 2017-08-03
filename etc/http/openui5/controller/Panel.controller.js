sap.ui.define([
   'sap/ui/core/mvc/Controller',
   'sap/ui/core/ResizeHandler',

], function (Controller, ResizeHandler) {
   "use strict";

   console.log('READ Panel.controller.js');

   var res = Controller.extend("sap.ui.jsroot.controller.Panel", {

       // preferDOM: true,

       onAfterRendering: function() {
         //if (sap.HTML.prototype.onAfterRendering) {
         //   sapHTML.prototype.onAfterRendering.apply(this, arguments);
         //}
         var view = this.getView();
         var dom = view.getDomRef();

         if (this.canvas_painter && this.canvas_painter._configured_socket_kind) {
            this.canvas_painter.SetDivId(dom, -1);
            this.canvas_painter.OpenWebsocket(this.canvas_painter._configured_socket_kind);
            delete this.canvas_painter._configured_socket_kind;
         }

       },

       onBeforeRendering: function() {
       },

       onResize : function(event) {
          // use timeout
          if (this.resize_tmout) clearTimeout(this.resize_tmout);
          this.resize_tmout = setTimeout(this.onResizeTimeout.bind(this), 300); // minimal latency
       },

       onResizeTimeout : function() {
          delete this.resize_tmout;
          if (this.canvas_painter)
             this.canvas_painter.CheckCanvasResize();
       },

       onInit : function() {
          this.canvas_painter = JSROOT.openui5_canvas_painter;
          delete JSROOT.openui5_canvas_painter;
          console.log('INIT JSROOT PANEL DONE', typeof this.canvas_painter);
          ResizeHandler.register(this.getView(), this.onResize.bind(this));
      }
   });


   return res;

});
