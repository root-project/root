sap.ui.define([
   'sap/ui/core/mvc/Controller',
   'sap/ui/core/ResizeHandler'
], function (Controller,
             ResizeHandler) {
   "use strict";

   return Controller.extend('rootui5.canv.controller.CanvasPanel', {

      preserveCanvasContent() {
         // workaround, openui5 does not preserve DOM elements when calling onBeforeRendering
         let dom = this.getView().getDomRef();
         if (this.canvas_painter && dom?.children.length && !this._mainChild) {
            this._mainChild = dom.lastChild;
            dom.removeChild(this._mainChild);
         }
      },

      onBeforeRendering() {
         this.preserveCanvasContent();
         this._has_after_rendering = false;
      },

      setPainter(painter) {
         this.canvas_painter = painter;
      },

      getPainter() {
         return this.canvas_painter;
      },

      onAfterRendering() {
         // workaround for openui5 problem - called before actual dimension of HTML element is assigned
         // using timeout and resize event to handle it correctly
         this._has_after_rendering = true;
         this.invokeResizeTimeout(10);
      },

      hasValidSize() {
         return (this.getView().$().width() > 0) && (this.getView().$().height() > 0);
      },

      invokeResizeTimeout(tmout) {
        if (this.resize_tmout) {
            clearTimeout(this.resize_tmout);
            delete this.resize_tmout;
        }

        if (this.hasValidSize() && this._has_after_rendering)
           this.onResizeTimeout();
        else
           this.resize_tmout = setTimeout(this.onResizeTimeout.bind(this), tmout);
      },

      /** @summary Set fixed size for canvas container */
      setFixedSize(w, h, on) {
         let dom = this.getView().getDomRef();
         if (!dom?.lastChild) return false;

         if (!this.isFixedSize && !on) return false;

         this.isFixedSize = w && h && on;

         if (this.isFixedSize)
            dom.lastChild.style = `position:relative;left:0px;top:0px;width:${w}px;height:${h}px`;
         else
            dom.lastChild.style = 'position:relative;inset:0px;height:100%;width:100%';

         return this.isFixedSize;
      },

      /** @summary Handle openui5 resize glitch
        * @desc onAfterRendering method does not provide valid dimension of the HTML element
        * One should wait either resize event or timeout and check if valid size is there
        * Only then normal rendering can be started
        * Method also used to check resize events  */
      onResizeTimeout() {
         delete this.resize_tmout;

         if (!this.hasValidSize())
             return this.invokeResizeTimeout(5000); // very rare check if something changed

         let check_resize = true;

         if (this._has_after_rendering) {

            this._has_after_rendering = false;
            check_resize = false;

            let dom = this.getView().getDomRef();

            if (dom && this._mainChild) {
               dom.appendChild(this._mainChild)
               delete this._mainChild;
               check_resize = true;
            }

            if (dom && !dom.children.length) {
               let d = document.createElement('div');
               d.style = 'position:relative;inset:0px;height:100%;width:100%';
               dom.appendChild(d);
               this.isFixedSize = false;
            }

            if (this.canvas_painter) {
               this.canvas_painter.setDom(dom.lastChild);
               this.canvas_painter.setPadName('');
            }

            if (this.canvas_painter && this.canvas_painter._window_handle) {
               this.canvas_painter.useWebsocket(this.canvas_painter._window_handle);
               delete this.canvas_painter._window_handle;
            }
         }

         if (this.canvas_painter && check_resize)
            this.canvas_painter.checkCanvasResize();
      },

      onInit() {
         ResizeHandler.register(this.getView(), this.invokeResizeTimeout.bind(this, 200));
      },

      onExit() {
         if (this.canvas_painter) {
            this.canvas_painter.cleanup();
            delete this.canvas_painter;
         }
      }
   });

});
