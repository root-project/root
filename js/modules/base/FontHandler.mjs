
const root_fonts = ['Arial', 'iTimes New Roman',
      'bTimes New Roman', 'biTimes New Roman', 'Arial',
      'oArial', 'bArial', 'boArial', 'Courier New',
      'oCourier New', 'bCourier New', 'boCourier New',
      'Symbol', 'Times New Roman', 'Wingdings', 'iSymbol',
      'Verdana', 'iVerdana', 'bVerdana', 'biVerdana'];


// taken from symbols.html, counted only for letters and digits
const root_fonts_aver_width = [0.5778,0.5314,
      0.5809, 0.5540, 0.5778,
      0.5783,0.6034,0.6030,0.6003,
      0.6004,0.6003,0.6005,
      0.5521,0.5521,0.5664,0.5314,
      0.5664,0.5495,0.5748,0.5578];

/**
 * @summary Helper class for font handling
 *
 */

class FontHandler {
   /** @summary constructor */
   constructor(fontIndex, size, scale, name, style, weight) {
      this.name = "Arial";
      this.style = null;
      this.weight = null;

      if (scale && (size < 1)) {
         size *= scale;
         this.scaled = true;
      }

      this.size = Math.round(size || 11);
      this.scale = scale;

      if (fontIndex !== null) {

         let indx = Math.floor(fontIndex / 10),
             fontName = root_fonts[indx] || "Arial";

         while (fontName.length > 0) {
            if (fontName[0] === 'b')
               this.weight = "bold";
            else if (fontName[0] === 'i')
               this.style = "italic";
            else if (fontName[0] === 'o')
               this.style = "oblique";
            else
               break;
            fontName = fontName.slice(1);
         }

         this.name = fontName;
         this.aver_width = root_fonts_aver_width[indx] || 0.55;
      } else {
         this.name = name;
         this.style = style || null;
         this.weight = weight || null;
         this.aver_width = this.weight ? 0.58 : 0.55;
      }

      if ((this.name == 'Symbol') || (this.name == 'Wingdings')) {
         this.isSymbol = this.name;
         this.name = "Times New Roman";
      } else {
         this.isSymbol = "";
      }

      this.func = this.setFont.bind(this);
   }

   /** @summary Assigns font-related attributes */
   setFont(selection, arg) {
      selection.attr("font-family", this.name);
      if (arg != 'without-size')
         selection.attr("font-size", this.size)
                  .attr("xml:space", "preserve");
      if (this.weight)
         selection.attr("font-weight", this.weight);
      if (this.style)
         selection.attr("font-style", this.style);
   }

   /** @summary Set font size (optional) */
   setSize(size) { this.size = Math.round(size); }

   /** @summary Set text color (optional) */
   setColor(color) { this.color = color; }

   /** @summary Set text align (optional) */
   setAlign(align) { this.align = align; }

   /** @summary Set text angle (optional) */
   setAngle(angle) { this.angle = angle; }

   /** @summary Allign angle to step raster, add optional offset */
   roundAngle(step, offset) {
      this.angle = parseInt(this.angle || 0);
      if (!Number.isInteger(this.angle)) this.angle = 0;
      this.angle = Math.round(this.angle/step) * step + (offset || 0);
      if (this.angle < 0)
         this.angle += 360;
      else if (this.angle >= 360)
         this.angle -= 360;
   }

   /** @summary Clears all font-related attributes */
   clearFont(selection) {
      selection.attr("font-family", null)
               .attr("font-size", null)
               .attr("xml:space", null)
               .attr("font-weight", null)
               .attr("font-style", null);
   }

   /** @summary Returns true in case of monospace font
     * @private */
   isMonospace() {
      let n = this.name.toLowerCase();
      return (n.indexOf("courier") == 0) || (n == "monospace") || (n == "monaco");
   }

   /** @summary Return full font declaration which can be set as font property like "12pt Arial bold"
     * @private */
   getFontHtml() {
      let res = Math.round(this.size) + "pt " + this.name;
      if (this.weight) res += " " + this.weight;
      if (this.style) res += " " + this.style;
      return res;
   }

} // class FontHandler

export { FontHandler };
