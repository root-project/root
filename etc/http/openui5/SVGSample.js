sap.ui.define([
   'sap/ui/core/Control',
   'sap/ui/core/ResizeHandler'
], function (Control, ResizeHandler) {
   "use strict";

   var SVGSample = Control.extend("sap.ui.jsroot.SVGSample", {
      metadata: {
         properties: {
            svgsample : {type : "object", group : "Misc", defaultValue : null}
         },
         defaultAggregation: null
      },

      init: function() {

         // console.log('init');

         // svg images are always loaded without @2
         this.addEventDelegate({
            onAfterRendering: function() { this._setSVG(); }
         }, this);

         this.attachModelContextChange({}, this.modelChanged, this);

         this.resize_id = ResizeHandler.register(this, this.onResize.bind(this));
      },

      destroy: function() {
         console.log('destroy SVG');
      },

      renderer: function(oRm,oControl){
         //first up, render a div for the ShadowBox
         oRm.write("<div");

         //next, render the control information, this handles your sId (you must do this for your control to be properly tracked by ui5).
         oRm.writeControlData(oControl);


         oRm.addClass("sapUiSizeCompact");
         oRm.addClass("sapMSlt");

         oRm.writeClasses();

         oRm.addStyle("width","50%");
         // oRm.addStyle("height","100%");

         oRm.writeStyles();

         oRm.write(">");

         //next, iterate over the content aggregation, and call the renderer for each control
         //$(oControl.getContent()).each(function(){
         //    oRm.renderControl(this);
         //});

         //and obviously, close off our div
         oRm.write("</div>")
     },
   });

   SVGSample.prototype._setSVG = function() {
      var dom = this.$();
      if (!dom) return;

      var w = dom.innerWidth(), h = dom.innerHeight();
      dom.empty();

      var svg = d3.select(dom.get(0)).append("svg").attr("width", w).attr("height",h).attr("viewBox","0 0 " + w + " " + h);

      var attr = this.getProperty("svgsample");
      if (attr && (typeof attr == "object") && (typeof attr.CreateSample == "function"))
         attr.CreateSample(svg,w,h);
      else
         svg.append("text").text("none");
   }

   SVGSample.prototype.onResize = function() {
      this._setSVG();
   }

   SVGSample.prototype.modelChanged = function() {
      if (this._lastModel !== this.getModel()) {
         this._lastModel = this.getModel();
         this.getModel().attachPropertyChange({}, this.modelPropertyChanged, this);
      }
   }

   SVGSample.prototype.modelPropertyChanged = function() {
      this._setSVG();
   }

   return SVGSample;

});
