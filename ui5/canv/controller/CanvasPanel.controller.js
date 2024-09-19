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
            this.canvas_painter._ignore_resize = true; // ignore possible resize events
         }
      },

      rememberAreaSize() {
         this._prev_w = this.getView().$().width();
         this._prev_h = this.getView().$().height();
         this.invokeResizeTimeout(100); // we expect some changes, at the same time ignore resize
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
         // issue longer resize timeout
         // normall ui5 resize event with short timeout should follow very fast
         // only then one can check size of element and perform rendering
         this._has_after_rendering = true;
         this.invokeResizeTimeout(300);
      },

      hasValidSize() {
         return (this.getView().$().width() > 0) && (this.getView().$().height() > 0);
      },

      invokeResizeTimeout(tmout) {
         if (this.resize_tmout) {
            clearTimeout(this.resize_tmout);
            delete this.resize_tmout;
         }

         if (this.canvas_painter)
            this.canvas_painter._ignore_resize = true; // ignore possible resize events

         this.resize_tmout = setTimeout(() => this.onResizeTimeout(), tmout);
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

         // do nothing, for the moment rendering not yet finished
         if (this._has_after_rendering === false)
            return;

         if (!this.hasValidSize())
            return this.invokeResizeTimeout(5000); // very rare check if something changed

         if (this.canvas_painter)
            this.canvas_painter._ignore_resize = false;

         let check_resize = true;

         if (this._has_after_rendering) {

            delete this._has_after_rendering;
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

            if (this.canvas_painter?._window_handle) {
               this.canvas_painter.useWebsocket(this.canvas_painter._window_handle);
               delete this.canvas_painter._window_handle;
            }
         }

         if (check_resize) {
            if (this._prev_w && this._prev_h) {
               const dw = this._prev_w - this.getView().$().width(),
                     dh = this._prev_h - this.getView().$().height();
               delete this._prev_w;
               delete this._prev_h;
               if (this.resizeBrowser(dw, dh))
                  return this.invokeResizeTimeout(100);
            }

            this.canvas_painter?.checkCanvasResize();
         }
      },

      resizeBrowser(delta_w, delta_h) {
         if (!this.canvas_painter?.resizeBrowser ||
              this.canvas_painter?.embed_canvas ||
              typeof window !== 'object')
              return;
         const wbig = window.outerWidth,
               hbig = window.outerHeight;

         if (wbig && hbig && (delta_w || delta_h)) {
            this.canvas_painter.resizeBrowser(Math.max(100, wbig + delta_w), Math.max(100, hbig + delta_h));
            return true;
         }
      },

      onInit() {
         ResizeHandler.register(this.getView(), () => this.invokeResizeTimeout(10));
      },

      onExit() {
         if (this.canvas_painter) {
            this.canvas_painter.cleanup();
            delete this.canvas_painter;
         }
      }
   });

});
