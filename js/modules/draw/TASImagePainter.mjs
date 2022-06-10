import { create, isNodeJs } from '../core.mjs';
import { toHex } from '../base/colors.mjs';
import { ObjectPainter } from '../base/ObjectPainter.mjs';
import { TPavePainter } from '../hist/TPavePainter.mjs';
import { ensureTCanvas } from '../gpad/TCanvasPainter.mjs';


let node_canvas, btoa_func = globalThis?.btoa;

///_begin_exclude_in_qt5web_
if(isNodeJs()) { node_canvas = await import('canvas').then(h => h.default); btoa_func = await import("btoa").then(h => h.default); } /// cutNodeJs
///_end_exclude_in_qt5web_


/**
 * @summary Painter for TASImage object.
 *
 * @private
 */

class TASImagePainter extends ObjectPainter {

   /** @summary Decode options string  */
   decodeOptions(opt) {
      this.options = { Zscale: false };

      if (opt && (opt.indexOf("z") >= 0)) this.options.Zscale = true;
   }

   /** @summary Create RGBA buffers */
   createRGBA(nlevels) {
      let obj = this.getObject();

      if (!obj || !obj.fPalette) return null;

      let rgba = new Array((nlevels+1) * 4), indx = 1, pal = obj.fPalette; // precaclucated colors

      for(let lvl = 0; lvl <= nlevels; ++lvl) {
         let l = 1.*lvl/nlevels;
         while ((pal.fPoints[indx] < l) && (indx < pal.fPoints.length-1)) indx++;

         let r1 = (pal.fPoints[indx] - l) / (pal.fPoints[indx] - pal.fPoints[indx-1]),
             r2 = (l - pal.fPoints[indx-1]) / (pal.fPoints[indx] - pal.fPoints[indx-1]);

         rgba[lvl*4]   = Math.min(255, Math.round((pal.fColorRed[indx-1] * r1 + pal.fColorRed[indx] * r2) / 256));
         rgba[lvl*4+1] = Math.min(255, Math.round((pal.fColorGreen[indx-1] * r1 + pal.fColorGreen[indx] * r2) / 256));
         rgba[lvl*4+2] = Math.min(255, Math.round((pal.fColorBlue[indx-1] * r1 + pal.fColorBlue[indx] * r2) / 256));
         rgba[lvl*4+3] = Math.min(255, Math.round((pal.fColorAlpha[indx-1] * r1 + pal.fColorAlpha[indx] * r2) / 256));
      }

      return rgba;
   }

   /** @summary Draw image */
   drawImage() {
      let obj = this.getObject(),
          is_buf = false,
          fp = this.getFramePainter(),
          rect = fp ? fp.getFrameRect() : this.getPadPainter().getPadRect();

      this.wheel_zoomy = true;

      if (obj._blob) {
         // try to process blob data due to custom streamer
         if ((obj._blob.length == 15) && !obj._blob[0]) {
            obj.fImageQuality = obj._blob[1];
            obj.fImageCompression = obj._blob[2];
            obj.fConstRatio = obj._blob[3];
            obj.fPalette = {
                _typename: "TImagePalette",
                fUniqueID: obj._blob[4],
                fBits: obj._blob[5],
                fNumPoints: obj._blob[6],
                fPoints: obj._blob[7],
                fColorRed: obj._blob[8],
                fColorGreen: obj._blob[9],
                fColorBlue: obj._blob[10],
                fColorAlpha: obj._blob[11]
            };

            obj.fWidth = obj._blob[12];
            obj.fHeight = obj._blob[13];
            obj.fImgBuf = obj._blob[14];

            if ((obj.fWidth * obj.fHeight != obj.fImgBuf.length) ||
                  (obj.fPalette.fNumPoints != obj.fPalette.fPoints.length)) {
               console.error('TASImage _blob decoding error', obj.fWidth * obj.fHeight, '!=', obj.fImgBuf.length, obj.fPalette.fNumPoints, "!=", obj.fPalette.fPoints.length);
               delete obj.fImgBuf;
               delete obj.fPalette;
            }

         } else if ((obj._blob.length == 3) && obj._blob[0]) {
            obj.fPngBuf = obj._blob[2];
            if (!obj.fPngBuf || (obj.fPngBuf.length != obj._blob[1])) {
               console.error('TASImage with png buffer _blob error', obj._blob[1], '!=', (obj.fPngBuf ? obj.fPngBuf.length : -1));
               delete obj.fPngBuf;
            }
         } else {
            console.error('TASImage _blob len', obj._blob.length, 'not recognized');
         }

         delete obj._blob;
      }

      let url, constRatio = true;

      if (obj.fImgBuf && obj.fPalette) {

         is_buf = true;

         let nlevels = 1000;
         this.rgba = this.createRGBA(nlevels); // precaclucated colors

         let min = obj.fImgBuf[0], max = obj.fImgBuf[0];
         for (let k = 1; k < obj.fImgBuf.length; ++k) {
            let v = obj.fImgBuf[k];
            min = Math.min(v, min);
            max = Math.max(v, max);
         }

         // does not work properly in Node.js, causes "Maximum call stack size exceeded" error
         // min = Math.min.apply(null, obj.fImgBuf),
         // max = Math.max.apply(null, obj.fImgBuf);

         // create countor like in hist painter to allow palette drawing
         this.fContour = {
            arr: new Array(200),
            rgba: this.rgba,
            getLevels: function() { return this.arr; },
            getPaletteColor: function(pal, zval) {
               if (!this.arr || !this.rgba) return "white";
               let indx = Math.round((zval - this.arr[0]) / (this.arr[this.arr.length-1] - this.arr[0]) * (this.rgba.length-4)/4) * 4;
               return "#" + toHex(this.rgba[indx],1) + toHex(this.rgba[indx+1],1) + toHex(this.rgba[indx+2],1) + toHex(this.rgba[indx+3],1);
            }
         };
         for (let k = 0; k < 200; k++)
            this.fContour.arr[k] = min + (max-min)/(200-1)*k;

         if (min >= max) max = min + 1;

         let xmin = 0, xmax = obj.fWidth, ymin = 0, ymax = obj.fHeight; // dimension in pixels

         if (fp && (fp.zoom_xmin != fp.zoom_xmax)) {
            xmin = Math.round(fp.zoom_xmin * obj.fWidth);
            xmax = Math.round(fp.zoom_xmax * obj.fWidth);
         }

         if (fp && (fp.zoom_ymin != fp.zoom_ymax)) {
            ymin = Math.round(fp.zoom_ymin * obj.fHeight);
            ymax = Math.round(fp.zoom_ymax * obj.fHeight);
         }

         let canvas;

         if (isNodeJs()) {
            canvas = node_canvas.createCanvas(xmax - xmin, ymax - ymin);
         } else {
            canvas = document.createElement('canvas');
            canvas.width = xmax - xmin;
            canvas.height = ymax - ymin;
         }

         if (!canvas)
            return null;

         let context = canvas.getContext('2d'),
             imageData = context.getImageData(0, 0, canvas.width, canvas.height),
             arr = imageData.data;

         for(let i = ymin; i < ymax; ++i) {
            let dst = (ymax - i - 1) * (xmax - xmin) * 4,
                row = i * obj.fWidth;
            for(let j = xmin; j < xmax; ++j) {
               let iii = Math.round((obj.fImgBuf[row + j] - min) / (max - min) * nlevels) * 4;
               // copy rgba value for specified point
               arr[dst++] = this.rgba[iii++];
               arr[dst++] = this.rgba[iii++];
               arr[dst++] = this.rgba[iii++];
               arr[dst++] = this.rgba[iii++];
            }
         }

         context.putImageData(imageData, 0, 0);

         url = canvas.toDataURL(); // create data url to insert into image

         constRatio = obj.fConstRatio;

      } else if (obj.fPngBuf) {
         let pngbuf = "";

         if (typeof obj.fPngBuf == "string") {
            pngbuf = obj.fPngBuf;
         } else {
            for (let k = 0; k < obj.fPngBuf.length; ++k)
               pngbuf += String.fromCharCode(obj.fPngBuf[k] < 0 ? 256 + obj.fPngBuf[k] : obj.fPngBuf[k]);
         }

         url = "data:image/png;base64," + btoa_func(pngbuf);
      }

      if (url)
         this.createG(fp ? true : false)
             .append("image")
             .attr("href", url)
             .attr("width", rect.width)
             .attr("height", rect.height)
             .attr("preserveAspectRatio", constRatio ? null : "none");

      if (!url || !this.isMainPainter() || !is_buf || !fp)
         return this;

      return this.drawColorPalette(this.options.Zscale, true).then(() => {
         fp.setAxesRanges(create("TAxis"), 0, 1, create("TAxis"), 0, 1, null, 0, 0);
         fp.createXY({ ndim: 2, check_pad_range: false });
         return fp.addInteractivity();
      });
   }

   /** @summary Checks if it makes sense to zoom inside specified axis range */
   canZoomInside(axis,min,max) {
      let obj = this.getObject();

      if (!obj || !obj.fImgBuf)
         return false;

      if ((axis == "x") && ((max - min) * obj.fWidth > 3)) return true;

      if ((axis == "y") && ((max - min) * obj.fHeight > 3)) return true;

      return false;
   }

   /** @summary Draw color palette
     * @private */
   drawColorPalette(enabled, can_move) {

      if (!this.isMainPainter())
         return Promise.resolve(null);

      if (!this.draw_palette) {
         let pal = create('TPave');

         Object.assign(pal, { _typename: "TPaletteAxis", fName: "TPave", fH: null, fAxis: create('TGaxis'),
                               fX1NDC: 0.91, fX2NDC: 0.95, fY1NDC: 0.1, fY2NDC: 0.9, fInit: 1 } );

         pal.fAxis.fChopt = "+";

         this.draw_palette = pal;
         this.fPalette = true; // to emulate behaviour of hist painter
      }

      let pal_painter = this.getPadPainter().findPainterFor(this.draw_palette);

      if (!enabled) {
         if (pal_painter) {
            pal_painter.Enabled = false;
            pal_painter.removeG(); // completely remove drawing without need to redraw complete pad
         }
         return Promise.resolve(null);
      }

      let frame_painter = this.getFramePainter();

      // keep palette width
      if (can_move && frame_painter) {
         let pal = this.draw_palette;
         pal.fX2NDC = frame_painter.fX2NDC + 0.01 + (pal.fX2NDC - pal.fX1NDC);
         pal.fX1NDC = frame_painter.fX2NDC + 0.01;
         pal.fY1NDC = frame_painter.fY1NDC;
         pal.fY2NDC = frame_painter.fY2NDC;
      }

      if (pal_painter) {
         pal_painter.Enabled = true;
         return pal_painter.drawPave("");
      }


      let prev_name = this.selectCurrentPad(this.getPadName());

      return TPavePainter.draw(this.getDom(), this.draw_palette).then(p => {

         pal_painter = p;

         this.selectCurrentPad(prev_name);
         // mark painter as secondary - not in list of TCanvas primitives
         pal_painter.$secondary = true;

         // make dummy redraw, palette will be updated only from histogram painter
         pal_painter.redraw = function() {};
      });
   }

   /** @summary Toggle colz draw option
     * @private */
   toggleColz() {
      let obj = this.getObject(),
          can_toggle = obj && obj.fPalette;

      if (can_toggle) {
         this.options.Zscale = !this.options.Zscale;
         this.drawColorPalette(this.options.Zscale, true);
      }
   }

   /** @summary Redraw image */
   redraw(reason) {
      let img = this.draw_g ? this.draw_g.select("image") : null,
          fp = this.getFramePainter();

      if (img && !img.empty() && (reason !== "zoom") && fp) {
         img.attr("width", fp.getFrameWidth()).attr("height", fp.getFrameHeight());
      } else {
         return this.drawImage();
      }
   }

   /** @summary Process click on TASImage-defined buttons */
   clickButton(funcname) {
      if (!this.isMainPainter()) return false;

      switch(funcname) {
         case "ToggleColorZ": this.toggleColz(); break;
         default: return false;
      }

      return true;
   }

   /** @summary Fill pad toolbar for TASImage */
   fillToolbar() {
      let pp = this.getPadPainter(), obj = this.getObject();
      if (pp && obj && obj.fPalette) {
         pp.addPadButton("th2colorz", "Toggle color palette", "ToggleColorZ");
         pp.showPadButtons();
      }
   }

   /** @summary Draw TASImage object */
   static draw(dom, obj, opt) {
      let painter = new TASImagePainter(dom, obj, opt);
      painter.decodeOptions(opt);
      return ensureTCanvas(painter, false)
                 .then(() => painter.drawImage())
                 .then(() => {
                     painter.fillToolbar();
                     return painter;
                 });
   }

} // class TASImagePainter

export { TASImagePainter };
