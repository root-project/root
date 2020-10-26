sap.ui.define([
   'sap/ui/core/mvc/Controller',
   'sap/ui/core/ResizeHandler'
], function (Controller, ResizeHandler) {
   "use strict";

   return Controller.extend("rootui5.canv.controller.CanvasPanel", {

      preserveCanvasContent: function () {
         // workaround, openui5 does not preserve
         let dom = this.getView().getDomRef();
         if (this.canvas_painter && dom && dom.children.length && !this._mainChild) {
            this._mainChild = dom.children[0];
            dom.removeChild(this._mainChild);
         }
      },

      onBeforeRendering: function() {
         this.preserveCanvasContent();
      },

      setPainter: function(painter) {
         this.canvas_painter = painter;
      },

      getPainter: function() {
         return this.canvas_painter;
      },

      onAfterRendering: function() {
         let dom = this.getView().getDomRef(), check_resize = false;

         if (dom && this._mainChild) {
            dom.appendChild(this._mainChild)
            delete this._mainChild;
            check_resize = true;
         }

         if (dom && !dom.children.length) {
             let d = document.createElement("div");
             d.style = "position:relative;left:0;right:0;top:0;bottom:0;height:100%;width:100%";
             dom.appendChild(d);
          }

         if (this.canvas_painter) {
            this.canvas_painter.SetDivId(dom.lastChild, -1);
            if (check_resize) this.canvas_painter.CheckCanvasResize();
         }

         if (this.canvas_painter && this.canvas_painter._window_handle) {
            this.canvas_painter.UseWebsocket(this.canvas_painter._window_handle, this.canvas_painter._window_handle_href);
            delete this.canvas_painter._window_handle;
         }
      },

      onResize: function(/* event */) {
         // use timeout
         if (this.resize_tmout) clearTimeout(this.resize_tmout);
         this.resize_tmout = setTimeout(this.onResizeTimeout.bind(this), 300); // minimal latency
      },

      drawCanvas: function(can, opt, call_back) {
         if (this.canvas_painter) {
            this.canvas_painter.Cleanup();
            delete this.canvas_painter;
         }

         let dom = this.getView().getDomRef();

         if (!dom)
            return JSROOT.CallBack(call_back, null);

         if (dom.children.length == 0) {
             let d = document.createElement("div");
             d.style= "position:relative;left:0;right:0;top:0;bottom:0;height:100%;width:100%";
             dom.appendChild(d);
          }

         JSROOT.draw(dom.lastChild, can, opt).then(painter => {
            this.canvas_painter = painter;
            JSROOT.CallBack(call_back, painter);
         });
      },

      onResizeTimeout: function() {
         delete this.resize_tmout;
         if (this.canvas_painter)
            this.canvas_painter.CheckCanvasResize();
      },

      onInit: function() {
         console.log("INIT CANVAS PANEL");

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
