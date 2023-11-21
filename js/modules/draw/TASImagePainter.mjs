import { create, settings, isNodeJs, isStr, btoa_func, clTAxis, clTPaletteAxis, clTImagePalette, getDocument } from '../core.mjs';
import { toHex } from '../base/colors.mjs';
import { assignContextMenu } from '../gui/menu.mjs';
import { DrawOptions } from '../base/BasePainter.mjs';
import { ObjectPainter } from '../base/ObjectPainter.mjs';
import { TPavePainter } from '../hist/TPavePainter.mjs';
import { ensureTCanvas } from '../gpad/TCanvasPainter.mjs';

/**
 * @summary Painter for TASImage object.
 *
 * @private
 */

class TASImagePainter extends ObjectPainter {

   /** @summary Decode options string  */
   decodeOptions(opt) {
      const d = new DrawOptions(opt);

      this.options = { Zscale: false };

      const obj = this.getObject();

      if (d.check('CONST')) {
         this.options.constRatio = true;
         if (obj) obj.fConstRatio = true;
         console.log('use const');
      }
      if (d.check('Z')) this.options.Zscale = true;
   }

   /** @summary Create RGBA buffers */
   createRGBA(nlevels) {
      const obj = this.getObject(),
            pal = obj?.fPalette;
      if (!pal) return null;

      const rgba = new Array((nlevels+1) * 4).fill(0); // precaclucated colors

      for (let lvl = 0, indx = 1; lvl <= nlevels; ++lvl) {
         const l = lvl/nlevels;
         while ((pal.fPoints[indx] < l) && (indx < pal.fPoints.length-1)) indx++;

         const r1 = (pal.fPoints[indx] - l) / (pal.fPoints[indx] - pal.fPoints[indx-1]),
               r2 = (l - pal.fPoints[indx-1]) / (pal.fPoints[indx] - pal.fPoints[indx-1]);

         rgba[lvl*4] = Math.min(255, Math.round((pal.fColorRed[indx-1] * r1 + pal.fColorRed[indx] * r2) / 256));
         rgba[lvl*4+1] = Math.min(255, Math.round((pal.fColorGreen[indx-1] * r1 + pal.fColorGreen[indx] * r2) / 256));
         rgba[lvl*4+2] = Math.min(255, Math.round((pal.fColorBlue[indx-1] * r1 + pal.fColorBlue[indx] * r2) / 256));
         rgba[lvl*4+3] = Math.min(255, Math.round((pal.fColorAlpha[indx-1] * r1 + pal.fColorAlpha[indx] * r2) / 256));
      }

      return rgba;
   }

   /** @summary Create url using image buffer
     * @private */
   async makeUrlFromImageBuf(obj, fp) {
      const nlevels = 1000;
      this.rgba = this.createRGBA(nlevels); // precaclucated colors

      let min = obj.fImgBuf[0], max = obj.fImgBuf[0];
      for (let k = 1; k < obj.fImgBuf.length; ++k) {
         const v = obj.fImgBuf[k];
         min = Math.min(v, min);
         max = Math.max(v, max);
      }

      // does not work properly in Node.js, causes 'Maximum call stack size exceeded' error
      // min = Math.min.apply(null, obj.fImgBuf),
      // max = Math.max.apply(null, obj.fImgBuf);

      // create countor like in hist painter to allow palette drawing
      this.fContour = {
         arr: new Array(200),
         rgba: this.rgba,
         getLevels() { return this.arr; },
         getPaletteColor(pal, zval) {
            if (!this.arr || !this.rgba) return 'white';
            const indx = Math.round((zval - this.arr[0]) / (this.arr[this.arr.length-1] - this.arr[0]) * (this.rgba.length-4)/4) * 4;
            return '#' + toHex(this.rgba[indx], 1) + toHex(this.rgba[indx+1], 1) + toHex(this.rgba[indx+2], 1) + toHex(this.rgba[indx+3], 1);
         }
      };
      for (let k = 0; k < 200; k++)
         this.fContour.arr[k] = min + (max-min)/(200-1)*k;

      if (min >= max) max = min + 1;

      const z = this.getImageZoomRange(fp, obj.fConstRatio, obj.fWidth, obj.fHeight),
            pr = isNodeJs()
                 ? import('canvas').then(h => h.default.createCanvas(z.xmax - z.xmin, z.ymax - z.ymin))
                 : new Promise(resolveFunc => {
                    const c = document.createElement('canvas');
                    c.width = z.xmax - z.xmin;
                    c.height = z.ymax - z.ymin;
                    resolveFunc(c);
                 });

      return pr.then(canvas => {
         const context = canvas.getContext('2d'),
               imageData = context.getImageData(0, 0, canvas.width, canvas.height),
               arr = imageData.data;

         for (let i = z.ymin; i < z.ymax; ++i) {
            let dst = (z.ymax - i - 1) * (z.xmax - z.xmin) * 4;
            const row = i * obj.fWidth;
            for (let j = z.xmin; j < z.xmax; ++j) {
               let iii = Math.round((obj.fImgBuf[row + j] - min) / (max - min) * nlevels) * 4;
               // copy rgba value for specified point
               arr[dst++] = this.rgba[iii++];
               arr[dst++] = this.rgba[iii++];
               arr[dst++] = this.rgba[iii++];
               arr[dst++] = this.rgba[iii++];
            }
         }

         context.putImageData(imageData, 0, 0);

         return { url: canvas.toDataURL(), constRatio: obj.fConstRatio, can_zoom: true };
      });
   }

   getImageZoomRange(fp, constRatio, width, height) {
      const res = { xmin: 0, xmax: width, ymin: 0, ymax: height };
      if (!fp) return res;

      let offx = 0, offy = 0, sizex = width, sizey = height;

      if (constRatio) {
         const image_ratio = height/width,
               frame_ratio = fp.getFrameHeight() / fp.getFrameWidth();

         if (image_ratio > frame_ratio) {
            const w2 = height / frame_ratio;
            offx = Math.round((w2 - width)/2);
            sizex = Math.round(w2);
         } else {
            const h2 = frame_ratio * width;
            offy = Math.round((h2 - height)/2);
            sizey = Math.round(h2);
         }
      }

      if (fp.zoom_xmin !== fp.zoom_xmax) {
         res.xmin = Math.min(width, Math.max(0, Math.round(fp.zoom_xmin * sizex) - offx));
         res.xmax = Math.min(width, Math.max(0, Math.round(fp.zoom_xmax * sizex) - offx));
      }
      if (fp.zoom_ymin !== fp.zoom_ymax) {
         res.ymin = Math.min(height, Math.max(0, Math.round(fp.zoom_ymin * sizey) - offy));
         res.ymax = Math.min(height, Math.max(0, Math.round(fp.zoom_ymax * sizey) - offy));
      }
      return res;
   }

   /** @summary Produce data url from png buffer */
   async makeUrlFromPngBuf(obj, fp) {
      const buf = obj.fPngBuf;
      let pngbuf = '';

      if (isStr(buf))
         pngbuf = buf;
      else {
         for (let k = 0; k < buf.length; ++k)
            pngbuf += String.fromCharCode(buf[k] < 0 ? 256 + buf[k] : buf[k]);
      }

      const res = { url: 'data:image/png;base64,' + btoa_func(pngbuf), constRatio: obj.fConstRatio, can_zoom: fp && !isNodeJs() },
            doc = getDocument();

      if (!res.can_zoom || ((fp?.zoom_xmin === fp?.zoom_xmax) && (fp?.zoom_ymin === fp?.zoom_ymax)))
         return res;

      return new Promise(resolveFunc => {
         const image = doc.createElement('img');

         image.onload = () => {
            const canvas = doc.createElement('canvas');
            canvas.width = image.width;
            canvas.height = image.height;

            const context = canvas.getContext('2d');
            context.drawImage(image, 0, 0);

            const arr = context.getImageData(0, 0, image.width, image.height).data,
                  z = this.getImageZoomRange(fp, res.constRatio, image.width, image.height),
                  canvas2 = doc.createElement('canvas');
            canvas2.width = z.xmax - z.xmin;
            canvas2.height = z.ymax - z.ymin;

            const context2 = canvas2.getContext('2d'),
                  imageData2 = context2.getImageData(0, 0, canvas2.width, canvas2.height),
                  arr2 = imageData2.data;

            for (let i = z.ymin; i < z.ymax; ++i) {
                let dst = (z.ymax - i - 1) * (z.xmax - z.xmin) * 4,
                    src = ((image.height - i - 1) * image.width + z.xmin) * 4;
                for (let j = z.xmin; j < z.xmax; ++j) {
                   // copy rgba value for specified point
                   arr2[dst++] = arr[src++];
                   arr2[dst++] = arr[src++];
                   arr2[dst++] = arr[src++];
                   arr2[dst++] = arr[src++];
                }
            }

            context2.putImageData(imageData2, 0, 0);

            res.url = canvas2.toDataURL();

            resolveFunc(res);
         };

         image.onerror = () => resolveFunc(res);

         image.src = res.url;
      });
   }

   /** @summary Draw image */
   async drawImage() {
      const obj = this.getObject(),
            fp = this.getFramePainter(),
            rect = fp?.getFrameRect() ?? this.getPadPainter().getPadRect();

      this.wheel_zoomy = true;

      if (obj._blob) {
         // try to process blob data due to custom streamer
         if ((obj._blob.length === 15) && !obj._blob[0]) {
            obj.fImageQuality = obj._blob[1];
            obj.fImageCompression = obj._blob[2];
            obj.fConstRatio = obj._blob[3];
            obj.fPalette = {
                _typename: clTImagePalette,
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

            if ((obj.fWidth * obj.fHeight !== obj.fImgBuf.length) ||
                  (obj.fPalette.fNumPoints !== obj.fPalette.fPoints.length)) {
               console.error(`TASImage _blob decoding error ${obj.fWidth * obj.fHeight} != ${obj.fImgBuf.length} ${obj.fPalette.fNumPoints} != ${obj.fPalette.fPoints.length}`);
               delete obj.fImgBuf;
               delete obj.fPalette;
            }
         } else if ((obj._blob.length === 3) && obj._blob[0]) {
            obj.fPngBuf = obj._blob[2];
            if (obj.fPngBuf?.length !== obj._blob[1]) {
               console.error(`TASImage with png buffer _blob error ${obj._blob[1]} != ${obj.fPngBuf?.length}`);
               delete obj.fPngBuf;
            }
         } else
            console.error(`TASImage _blob len ${obj._blob.length} not recognized`);

         delete obj._blob;
      }

      let promise;

      if (obj.fImgBuf && obj.fPalette)
         promise = this.makeUrlFromImageBuf(obj, fp);
      else if (obj.fPngBuf)
         promise = this.makeUrlFromPngBuf(obj, fp);
      else
         promise = Promise.resolve(null);

      return promise.then(res => {
         if (!res?.url)
            return this;

         const img = this.createG(!!fp)
             .append('image')
             .attr('href', res.url)
             .attr('width', rect.width)
             .attr('height', rect.height)
             .attr('preserveAspectRatio', res.constRatio ? null : 'none');

         if (!this.isBatchMode()) {
            if (settings.MoveResize || settings.ContextMenu)
               img.style('pointer-events', 'visibleFill');

            if (res.can_zoom)
               img.style('cursor', 'pointer');
         }

         assignContextMenu(this);

         if (!fp || !res.can_zoom)
            return this;

         return this.drawColorPalette(this.options.Zscale, true).then(() => {
            fp.setAxesRanges(create(clTAxis), 0, 1, create(clTAxis), 0, 1, null, 0, 0);
            fp.createXY({ ndim: 2, check_pad_range: false });
            return fp.addInteractivity();
         });
      });
   }

   /** @summary Fill TASImage context */
   fillContextMenuItems(menu) {
      const obj = this.getObject();
      if (obj) {
         menu.addchk(obj.fConstRatio, 'Const ratio', flag => {
            obj.fConstRatio = flag;
            this.interactiveRedraw('pad', `exec:SetConstRatio(${flag})`);
         }, 'Change const ratio flag of image');
      }
      if (obj?.fPalette) {
         menu.addchk(this.options.Zscale, 'Color palette', flag => {
            this.options.Zscale = flag;
            this.drawColorPalette(flag, true);
         }, 'Toggle color palette');
      }
   }

   /** @summary Checks if it makes sense to zoom inside specified axis range */
   canZoomInside(axis, min, max) {
      const obj = this.getObject();

      if (!obj)
         return false;

      if (((axis === 'x') || (axis === 'y')) && (max - min > 0.01)) return true;

      return false;
   }

   /** @summary Draw color palette
     * @private */
   async drawColorPalette(enabled, can_move) {
      if (!this.isMainPainter())
         return null;

      if (!this.draw_palette) {
         const pal = create(clTPaletteAxis);
         Object.assign(pal, { fX1NDC: 0.91, fX2NDC: 0.95, fY1NDC: 0.1, fY2NDC: 0.9, fInit: 1 });
         pal.fAxis.fChopt = '+';
         this.draw_palette = pal;
         this._color_palette = true; // to emulate behaviour of hist painter
      }

      let pal_painter = this.getPadPainter().findPainterFor(this.draw_palette);

      if (!enabled) {
         if (pal_painter) {
            pal_painter.Enabled = false;
            pal_painter.removeG(); // completely remove drawing without need to redraw complete pad
         }
         return null;
      }

      const fp = this.getFramePainter();

      // keep palette width
      if (can_move && fp) {
         const pal = this.draw_palette;
         pal.fX2NDC = fp.fX2NDC + 0.01 + (pal.fX2NDC - pal.fX1NDC);
         pal.fX1NDC = fp.fX2NDC + 0.01;
         pal.fY1NDC = fp.fY1NDC;
         pal.fY2NDC = fp.fY2NDC;
      }

      if (pal_painter) {
         pal_painter.Enabled = true;
         return pal_painter.drawPave('');
      }

      const prev_name = this.selectCurrentPad(this.getPadName());

      return TPavePainter.draw(this.getDom(), this.draw_palette).then(p => {
         pal_painter = p;

         this.selectCurrentPad(prev_name);
         // mark painter as secondary - not in list of TCanvas primitives
         pal_painter.setSecondary(this);

         // make dummy redraw, palette will be updated only from histogram painter
         pal_painter.redraw = function() {};
      });
   }

   /** @summary Toggle colz draw option
     * @private */
   toggleColz() {
      if (this.getObject()?.fPalette) {
         this.options.Zscale = !this.options.Zscale;
         return this.drawColorPalette(this.options.Zscale, true);
      }
   }

   /** @summary Redraw image */
   redraw() {
      return this.drawImage();
   }

   /** @summary Process click on TASImage-defined buttons
     * @desc may return promise or simply false */
   clickButton(funcname) {
      if (this.isMainPainter() && funcname === 'ToggleColorZ')
         return this.toggleColz();

      return false;
   }

   /** @summary Fill pad toolbar for TASImage */
   fillToolbar() {
      const pp = this.getPadPainter();
      if (pp && this.getObject()?.fPalette) {
         pp.addPadButton('th2colorz', 'Toggle color palette', 'ToggleColorZ');
         pp.showPadButtons();
      }
   }

   /** @summary Draw TASImage object */
   static async draw(dom, obj, opt) {
      const painter = new TASImagePainter(dom, obj, opt);
      painter.setAsMainPainter();
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
