sap.ui.define([
   'sap/ui/core/mvc/Controller',
   'sap/ui/core/ResizeHandler'
], function (Controller, ResizeHandler) {
   "use strict";

   return Controller.extend('rootui5.canv.controller.Panel', {

      onInit() {
         this.rendering_perfromed = false;
      },

      onExit() {
         this.cleanupPainter();
      },

      cleanupPainter() {
         this.object_painter?.cleanup();
         delete this.object_painter;
      },

      preservePainterContent() {
         // workaround, openui5 does not preserve DOM elements when calling onBeforeRendering
         let dom = this.getView().getDomRef();
         if (this.object_painter && dom?.children.length && !this._mainChild) {
            this._mainChild = dom.children[0];
            dom.removeChild(this._mainChild);
         }
      },

      restorePainterContent() {
         // workaround, openui5 does not preserve DOM elements when do rendering
         let dom = this.getView().getDomRef();
         if (this.object_painter && dom && this._mainChild) {
            dom.appendChild(this._mainChild)
            delete this._mainChild;
         }
      },

      onBeforeRendering() {
         this.preservePainterContent();
         this.rendering_perfromed = false;
      },

      onAfterRendering() {
         this.restorePainterContent();
         ResizeHandler.register(this.getView(), () => this.onResize());
         this.rendering_perfromed = true;
         let arr = this.renderFuncs;
         delete this.renderFuncs;
         arr?.forEach(func => func(this.getView().getDomRef()));
      },

      setObjectPainter(painter) {
         this.object_painter = painter;
      },

      getRenderPromise() {
         if (this.rendering_perfromed)
            return Promise.resolve(this.getView().getDomRef());

         return new Promise(resolveFunc => {
            if (!this.renderFuncs) this.renderFuncs = [];
            this.renderFuncs.push(resolveFunc);
         });
      },

      onResize() {
         // use timeout
         if (this.resize_tmout) clearTimeout(this.resize_tmout);
         this.resize_tmout = setTimeout(() => this.onResizeTimeout(), 100); // minimal latency
      },

      onResizeTimeout() {
         delete this.resize_tmout;
         this.object_painter?.checkResize();
      }

   });

});
