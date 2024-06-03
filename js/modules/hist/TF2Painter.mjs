import { createHistogram, setHistogramTitle, kNoStats, settings, gStyle, clTF2, clTH2F, isStr, isFunc } from '../core.mjs';
import { TH2Painter } from '../hist/TH2Painter.mjs';
import { proivdeEvalPar } from '../base/func.mjs';
import { produceTAxisLogScale, scanTF1Options } from '../hist/TF1Painter.mjs';
import { ObjectPainter, getElementMainPainter } from '../base/ObjectPainter.mjs';
import { DrawOptions, floatToString } from '../base/BasePainter.mjs';
import { THistPainter } from '../hist2d/THistPainter.mjs';


/**
  * @summary Painter for TF2 object
  *
  * @private
  */

class TF2Painter extends TH2Painter {

   /** @summary Returns drawn object name */
   getObjectName() { return this.$func?.fName ?? 'func'; }

   /** @summary Returns drawn object class name */
   getClassName() { return this.$func?._typename ?? clTF2; }

   /** @summary Returns true while function is drawn */
   isTF1() { return true; }

   /** @summary Returns primary function which was then drawn as histogram */
   getPrimaryObject() { return this.$func; }

   /** @summary Update histogram */
   updateObject(obj /*, opt */) {
      if (!obj || (this.getClassName() !== obj._typename)) return false;
      delete obj.evalPar;
      const histo = this.getHisto();

      if (this.webcanv_hist) {
         const h0 = this.getPadPainter()?.findInPrimitives('Func', clTH2F);
         if (h0) this.updateAxes(histo, h0, this.getFramePainter());
      }

      this.$func = obj;
      this.createTF2Histogram(obj, histo);
      this.scanContent();
      return true;
   }

   /** @summary Redraw TF2
     * @private */
   redraw(reason) {
      if (!this._use_saved_points && (reason === 'logx' || reason === 'logy' || reason === 'zoom')) {
         this.createTF2Histogram(this.$func, this.getHisto());
         this.scanContent();
      }

      return super.redraw(reason);
   }

   /** @summary Create histogram for TF2 drawing
     * @private */
   createTF2Histogram(func, hist) {
      let nsave = func.fSave.length - 6;
      if ((nsave > 0) && (nsave !== (func.fSave[nsave+4]+1) * (func.fSave[nsave+5]+1)))
         nsave = 0;

      this._use_saved_points = (nsave > 0) && (settings.PreferSavedPoints || (this.use_saved > 1));

      const fp = this.getFramePainter(),
            pad = this.getPadPainter()?.getRootPad(true),
            logx = pad?.fLogx, logy = pad?.fLogy,
            gr = fp?.getGrFuncs(this.second_x, this.second_y);
      let xmin = func.fXmin, xmax = func.fXmax,
          ymin = func.fYmin, ymax = func.fYmax,
          npx = Math.max(func.fNpx, 20),
          npy = Math.max(func.fNpy, 20);

      if (gr?.zoom_xmin !== gr?.zoom_xmax) {
         const dx = (xmax - xmin) / npx;
         if ((xmin < gr.zoom_xmin) && (gr.zoom_xmin < xmax))
            xmin = Math.max(xmin, gr.zoom_xmin - dx);
         if ((xmin < gr.zoom_xmax) && (gr.zoom_xmax < xmax))
            xmax = Math.min(xmax, gr.zoom_xmax + dx);
      }

      if (gr?.zoom_ymin !== gr?.zoom_ymax) {
         const dy = (ymax - ymin) / npy;
         if ((ymin < gr.zoom_ymin) && (gr.zoom_ymin < ymax))
            ymin = Math.max(ymin, gr.zoom_ymin - dy);
         if ((ymin < gr.zoom_ymax) && (gr.zoom_ymax < ymax))
            ymax = Math.min(ymax, gr.zoom_ymax + dy);
      }

      const ensureBins = (nx, ny) => {
         if (hist.fNcells !== (nx + 2) * (ny + 2)) {
            hist.fNcells = (nx + 2) * (ny + 2);
            hist.fArray = new Float32Array(hist.fNcells);
         }
         hist.fArray.fill(0);
         hist.fXaxis.fNbins = nx;
         hist.fXaxis.fXbins = [];
         hist.fYaxis.fNbins = ny;
         hist.fYaxis.fXbins = [];
      };

      delete this._fail_eval;

      if (!this._use_saved_points) {
         let iserror = false;

         if (!func.evalPar && !proivdeEvalPar(func))
            iserror = true;

         ensureBins(npx, npy);
         hist.fXaxis.fXmin = xmin;
         hist.fXaxis.fXmax = xmax;
         hist.fYaxis.fXmin = ymin;
         hist.fYaxis.fXmax = ymax;

         if (logx)
            produceTAxisLogScale(hist.fXaxis, npx, xmin, xmax);
         if (logy)
            produceTAxisLogScale(hist.fYaxis, npy, ymin, ymax);

         for (let j = 0; (j < npy) && !iserror; ++j) {
            for (let i = 0; (i < npx) && !iserror; ++i) {
               const x = hist.fXaxis.GetBinCenter(i+1),
                     y = hist.fYaxis.GetBinCenter(j+1);
               let z = 0;

               try {
                  z = func.evalPar(x, y);
               } catch {
                  iserror = true;
               }

               if (!iserror)
                  hist.setBinContent(hist.getBin(i + 1, j + 1), Number.isFinite(z) ? z : 0);
            }
         }

         if (iserror)
            this._fail_eval = true;

         if (iserror && (nsave > 6))
            this._use_saved_points = true;
      }

      if (this._use_saved_points) {
         npx = Math.round(func.fSave[nsave+4]);
         npy = Math.round(func.fSave[nsave+5]);
         const xmin = func.fSave[nsave], xmax = func.fSave[nsave+1],
               ymin = func.fSave[nsave+2], ymax = func.fSave[nsave+3],
               dx = (xmax - xmin) / npx,
               dy = (ymax - ymin) / npy;
          function getSave(x, y) {
            if (x < xmin || x > xmax) return 0;
            if (dx <= 0) return 0;
            if (y < ymin || y > ymax) return 0;
            if (dy <= 0) return 0;
            const ibin = Math.min(npx-1, Math.floor((x-xmin)/dx)),
                  jbin = Math.min(npy-1, Math.floor((y-ymin)/dy)),
                  xlow = xmin + ibin*dx,
                  ylow = ymin + jbin*dy,
                  t = (x-xlow)/dx,
                  u = (y-ylow)/dy,
                  k1 = jbin*(npx+1) + ibin,
                  k2 = jbin*(npx+1) + ibin +1,
                  k3 = (jbin+1)*(npx+1) + ibin +1,
                  k4 = (jbin+1)*(npx+1) + ibin;
            return (1-t)*(1-u)*func.fSave[k1] +t*(1-u)*func.fSave[k2] +t*u*func.fSave[k3] + (1-t)*u*func.fSave[k4];
         }

         ensureBins(func.fNpx, func.fNpy);
         hist.fXaxis.fXmin = func.fXmin;
         hist.fXaxis.fXmax = func.fXmax;
         hist.fYaxis.fXmin = func.fYmin;
         hist.fYaxis.fXmax = func.fYmax;

         for (let j = 0; j < func.fNpy; ++j) {
            const y = hist.fYaxis.GetBinCenter(j + 1);
            for (let i = 0; i < func.fNpx; ++i) {
               const x = hist.fXaxis.GetBinCenter(i + 1),
                     z = getSave(x, y);
               hist.setBinContent(hist.getBin(i+1, j+1), Number.isFinite(z) ? z : 0);
            }
         }
      }

      hist.fName = 'Func';
      setHistogramTitle(hist, func.fTitle);
      hist.fMinimum = func.fMinimum;
      hist.fMaximum = func.fMaximum;
      // fHistogram->SetContour(fContour.fN, levels);
      hist.fLineColor = func.fLineColor;
      hist.fLineStyle = func.fLineStyle;
      hist.fLineWidth = func.fLineWidth;
      hist.fFillColor = func.fFillColor;
      hist.fFillStyle = func.fFillStyle;
      hist.fMarkerColor = func.fMarkerColor;
      hist.fMarkerStyle = func.fMarkerStyle;
      hist.fMarkerSize = func.fMarkerSize;
      hist.fBits |= kNoStats;

      return hist;
   }

   /** @summary Extract function ranges */
   extractAxesProperties(ndim) {
      super.extractAxesProperties(ndim);

      const func = this.$func, nsave = func?.fSave.length ?? 0;

      if (nsave > 6 && this._use_saved_points) {
         this.xmin = Math.min(this.xmin, func.fSave[nsave-6]);
         this.xmax = Math.max(this.xmax, func.fSave[nsave-5]);
         this.ymin = Math.min(this.ymin, func.fSave[nsave-4]);
         this.ymax = Math.max(this.ymax, func.fSave[nsave-3]);
      }
      if (func) {
         this.xmin = Math.min(this.xmin, func.fXmin);
         this.xmax = Math.max(this.xmax, func.fXmax);
         this.ymin = Math.min(this.ymin, func.fYmin);
         this.ymax = Math.max(this.ymax, func.fYmax);
      }
   }

   /** @summary retrurn tooltips for TF2 */
   getTF2Tooltips(pnt) {
      const lines = [this.getObjectHint()],
            funcs = this.getFramePainter()?.getGrFuncs(this.options.second_x, this.options.second_y);

      if (!funcs || !isFunc(this.$func?.evalPar)) {
         lines.push('grx = ' + pnt.x, 'gry = ' + pnt.y);
         return lines;
      }

      const x = funcs.revertAxis('x', pnt.x),
            y = funcs.revertAxis('y', pnt.y);
      let z = 0, iserror = false;

       try {
          z = this.$func.evalPar(x, y);
       } catch {
          iserror = true;
       }

      lines.push('x = ' + funcs.axisAsText('x', x),
                 'y = ' + funcs.axisAsText('y', y),
                 'value = ' + (iserror ? '<fail>' : floatToString(z, gStyle.fStatFormat)));
      return lines;
   }

   /** @summary process tooltip event for TF2 object */
   processTooltipEvent(pnt) {
      if (this._use_saved_points)
         return super.processTooltipEvent(pnt);

      let ttrect = this.draw_g?.selectChild('.tooltip_bin');

      if (!this.draw_g || !pnt) {
         ttrect?.remove();
         return null;
      }

      const res = { name: this.$func?.fName, title: this.$func?.fTitle,
                  x: pnt.x, y: pnt.y,
                  color1: this.lineatt?.color ?? 'green',
                  color2: this.fillatt?.getFillColorAlt('blue') ?? 'blue',
                  lines: this.getTF2Tooltips(pnt), exact: true, menu: true };

      if (pnt.disabled)
         ttrect.remove();
      else {
         if (ttrect.empty()) {
            ttrect = this.draw_g.append('svg:circle')
                                .attr('class', 'tooltip_bin')
                                .style('pointer-events', 'none')
                                .style('fill', 'none')
                                .attr('r', (this.lineatt?.width ?? 1) + 4);
         }

         ttrect.attr('cx', pnt.x)
               .attr('cy', pnt.y)
               .call(this.lineatt?.func);
      }

      return res;
   }

   /** @summary fill information for TWebCanvas
    * @desc Used to inform webcanvas when evaluation failed
     * @private */
   fillWebObjectOptions(opt) {
      opt.fcust = this._fail_eval && !this.use_saved ? 'func_fail' : '';
   }

   /** @summary draw TF2 object */
   static async draw(dom, tf2, opt) {
      const web = scanTF1Options(opt);
      opt = web.opt;
      delete web.opt;

      const d = new DrawOptions(opt);
      if (d.empty())
         opt = 'cont3';
      else if (d.opt === 'SAME')
         opt = 'cont2 same';

      // workaround for old waves.C
      const o2 = isStr(opt) ? opt.toUpperCase() : '';
      if (o2 === 'SAMECOLORZ' || o2 === 'SAMECOLOR' || o2 === 'SAMECOLZ')
         opt = 'samecol';

      if ((opt.indexOf('same') === 0) || (opt.indexOf('SAME') === 0)) {
         if (!getElementMainPainter(dom))
            opt = 'A_ADJUST_FRAME_' + opt.slice(4);
      }

      let hist;

      if (web.webcanv_hist) {
         const dummy = new ObjectPainter(dom);
         hist = dummy.getPadPainter()?.findInPrimitives('Func', clTH2F);
      }

      if (!hist) {
         hist = createHistogram(clTH2F, 20, 20);
         hist.fBits |= kNoStats;
      }

      const painter = new TF2Painter(dom, hist);

      painter.$func = tf2;
      Object.assign(painter, web);
      painter.createTF2Histogram(tf2, hist);
      return THistPainter._drawHist(painter, opt);
   }

} // class TF2Painter

export { TF2Painter };
