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
         this.object_painter?.cleanup();
         delete this.object_painter;
      },

      onBeforeRendering() {
         this.object_painter?.cleanup();
         delete this.object_painter;
         this.rendering_perfromed = false;
      },

      onAfterRendering() {
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
