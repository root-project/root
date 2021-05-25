sap.ui.define([
   'sap/m/Button',
   'sap/m/ButtonRenderer',
   "sap/m/ColorPalettePopover"
], function (Button, ButtonRenderer, ColorPalettePopover) {

   "use strict";

   var ColorButton = Button.extend("rootui5.fitpanel.controller.ColorButton", {
      metadata: {
         properties: {
            color : { type : "string", group : "Misc", defaultValue : null }
         }
      },

      // official ROOT colors 1 .. 15, more is not supported by ColorPalettePopover
      rootColors: ['#000000','#ff0000','#00ff00','#0000ff','#ffff00','#ff00ff','#00ffff','#59d354','#5954d8',
                   '#fefefe', '#c0b6ac','#4c4c4c','#666666','#7f7f7f', '#999999'],
                   // ,, '#b2b2b2','#cccccc','#e5e5e5','#f2f2f2','#ccc6aa','#ccc6aa','#c1bfa8','#bab5a3','#b2a596','#b7a39b','#ad998c','#9b8e82','#876656','#afcec6'];

      renderer: ButtonRenderer.render,

      getHtmlColor: function() {
         var val = this.getColor();
         if (!val) return null;

         var ival = parseInt(val);
         if (!isNaN(ival) && (ival>=0) && (ival<16))
            return (ival==0) ? "#ffffff" : this.rootColors[ival];

         if (typeof val !== 'string')
            return null;

         return val;
      },

      onAfterRendering: function() {
         this.$().children().css('background-color', this.getHtmlColor());
      },

      pickColor: function(event) {
         this.setColor(event.getParameters().value);
      },

      firePress: function(event) {
          var oCPPop = new ColorPalettePopover( {
             defaultColor: this.getHtmlColor() || '#ff0000',
             colors: this.rootColors,
             colorSelect: this.pickColor.bind(this)
          });

          oCPPop.openBy(this);
      }

   });

   return ColorButton;
});
