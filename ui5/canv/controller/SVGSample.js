sap.ui.define([
   'sap/ui/core/Control',
   'sap/ui/core/ResizeHandler'
], function (Control, ResizeHandler) {
   "use strict";

   return Control.extend('rootui5.canv.controller.SVGSample', {
      metadata: {
         properties: {
            svgsample : { type: 'object', group: 'Misc', defaultValue: null }
         },
         defaultAggregation: null
      },

      init() {
         this.attachModelContextChange({}, this.modelChanged, this);

         this.resize_id = ResizeHandler.register(this, this.onResize.bind(this));
      },

      exit() {
      },

      onAfterRendering() {
         this._setSVG();
      },

      renderer(oRm, oControl){
         //first up, render a div for the ShadowBox
         oRm.write('<div');

         //next, render the control information, this handles your sId (you must do this for your control to be properly tracked by ui5).
         oRm.writeControlData(oControl);

         oRm.addClass('sapUiSizeCompact');
         oRm.addClass('sapMSlt');

         oRm.writeClasses();

         oRm.addStyle('width','50%');
         // oRm.addStyle('height','100%');

         oRm.writeStyles();

         oRm.write('>');

         //next, iterate over the content aggregation, and call the renderer for each control
         //$(oControl.getContent()).each(function(){
         //    oRm.renderControl(this);
         //});

         //and obviously, close off our div
         oRm.write('</div>')
     },

     _setSVG: function() {
        let dom = this.$();
        if (!dom) return;

        let w = dom.innerWidth(), h = dom.innerHeight();
        if (!w || !h) return;
        dom.empty();

        let svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', w);
        svg.setAttribute('height', h);
        svg.setAttribute('viewBox', `0 0 ${w} ${h}`);

        dom.get(0).appendChild(svg);

        let attr = this.getProperty('svgsample');
        if (attr && (typeof attr == 'object') && (typeof attr.createSample == 'function')) {
           attr.createSample(svg, w, h, true);
        } else {
           let txt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
           svg.appendChild(txt);
           txt.innerHTML = 'none';
        }
     },

     onResize() {
        this._setSVG();
     },

     modelPropertyChanged() {
        this._setSVG();
     },

     modelChanged() {
        if (this._lastModel !== this.getModel()) {
           this._lastModel = this.getModel();
           this.getModel().attachPropertyChange({}, this.modelPropertyChanged, this);
        }
     }

   });

});
