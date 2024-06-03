import { settings, gStyle, isStr, isFunc, clTH1D, createHistogram, setHistogramTitle, clTF1, kNoStats } from '../core.mjs';
import { floatToString } from '../base/BasePainter.mjs';
import { getElementMainPainter, ObjectPainter } from '../base/ObjectPainter.mjs';
import { THistPainter } from '../hist2d/THistPainter.mjs';
import { TH1Painter } from '../hist2d/TH1Painter.mjs';
import { proivdeEvalPar, _getTF1Save } from '../base/func.mjs';


/** @summary Create log scale for axis bins
  * @private */
function produceTAxisLogScale(axis, num, min, max) {
   let lmin, lmax;

   if (max > 0) {
      lmax = Math.log(max);
      lmin = min > 0 ? Math.log(min) : lmax - 5;
   } else {
      lmax = -10;
      lmin = -15;
   }

   axis.fNbins = num;
   axis.fXbins = new Array(num + 1);
   for (let i = 0; i <= num; ++i)
      axis.fXbins[i] = Math.exp(lmin + i / num * (lmax - lmin));
   axis.fXmin = Math.exp(lmin);
   axis.fXmax = Math.exp(lmax);
}

function scanTF1Options(opt) {
   if (!isStr(opt)) opt = '';
   let p = opt.indexOf(';webcanv_hist'), webcanv_hist = false, use_saved = 0;
   if (p >= 0) {
      webcanv_hist = true;
      opt = opt.slice(0, p);
   }
   p = opt.indexOf(';force_saved');
   if (p >= 0) {
      use_saved = 2;
      opt = opt.slice(0, p);
   }
   p = opt.indexOf(';prefer_saved');
   if (p >= 0) {
      use_saved = 1;
      opt = opt.slice(0, p);
   }
   return { opt, webcanv_hist, use_saved };
}


/**
  * @summary Painter for TF1 object
  *
  * @private
  */

class TF1Painter extends TH1Painter {

   /** @summary Returns drawn object name */
   getObjectName() { return this.$func?.fName ?? 'func'; }

   /** @summary Returns drawn object class name */
   getClassName() { return this.$func?._typename ?? clTF1; }

   /** @summary Returns true while function is drawn */
   isTF1() { return true; }

   /** @summary Returns primary function which was then drawn as histogram */
   getPrimaryObject() { return this.$func; }

   /** @summary Update function */
   updateObject(obj /*, opt */) {
      if (!obj || (this.getClassName() !== obj._typename)) return false;
      delete obj.evalPar;
      const histo = this.getHisto();

      if (this.webcanv_hist) {
         const h0 = this.getPadPainter()?.findInPrimitives('Func', clTH1D);
         if (h0) this.updateAxes(histo, h0, this.getFramePainter());
      }

      this.$func = obj;
      this.createTF1Histogram(obj, histo);
      this.scanContent();
      return true;
   }

   /** @summary Redraw TF1
     * @private */
   redraw(reason) {
      if (!this._use_saved_points && (reason === 'logx' || reason === 'zoom')) {
         this.createTF1Histogram(this.$func, this.getHisto());
         this.scanContent();
      }

      return super.redraw(reason);
   }

   /** @summary Create histogram for TF1 drawing
     * @private */
   createTF1Histogram(tf1, hist) {
      const fp = this.getFramePainter(),
            pad = this.getPadPainter()?.getRootPad(true),
            logx = pad?.fLogx,
            gr = fp?.getGrFuncs(this.second_x, this.second_y);
      let xmin = tf1.fXmin, xmax = tf1.fXmax, np = Math.max(tf1.fNpx, 100);

      if (gr?.zoom_xmin !== gr?.zoom_xmax) {
         const dx = (xmax - xmin) / np;
         if ((xmin < gr.zoom_xmin) && (gr.zoom_xmin < xmax))
            xmin = Math.max(xmin, gr.zoom_xmin - dx);
         if ((xmin < gr.zoom_xmax) && (gr.zoom_xmax < xmax))
            xmax = Math.min(xmax, gr.zoom_xmax + dx);
      }

      this._use_saved_points = (tf1.fSave.length > 3) && (settings.PreferSavedPoints || (this.use_saved > 1));

      const ensureBins = num => {
         if (hist.fNcells !== num + 2) {
            hist.fNcells = num + 2;
            hist.fArray = new Float32Array(hist.fNcells);
         }
         hist.fArray.fill(0);
         hist.fXaxis.fNbins = num;
         hist.fXaxis.fXbins = [];
      };

      delete this._fail_eval;

      // this._use_saved_points = true;

      if (!this._use_saved_points) {
         let iserror = false;

         if (!tf1.evalPar) {
            try {
               if (!proivdeEvalPar(tf1))
                  iserror = true;
            } catch {
               iserror = true;
            }
         }

         ensureBins(np);

         if (logx)
            produceTAxisLogScale(hist.fXaxis, np, xmin, xmax);
          else {
            hist.fXaxis.fXmin = xmin;
            hist.fXaxis.fXmax = xmax;
         }

         for (let n = 0; (n < np) && !iserror; n++) {
            const x = hist.fXaxis.GetBinCenter(n + 1);
            let y = 0;
            try {
               y = tf1.evalPar(x);
            } catch (err) {
               iserror = true;
            }

            if (!iserror)
               hist.setBinContent(n + 1, Number.isFinite(y) ? y : 0);
         }

         if (iserror)
            this._fail_eval = true;

         if (iserror && (tf1.fSave.length > 3))
            this._use_saved_points = true;
      }

      // in the case there were points have saved and we cannot calculate function
      // if we don't have the user's function
      if (this._use_saved_points) {
         np = tf1.fSave.length - 3;
         let custom_xaxis = null;
         xmin = tf1.fSave[np + 1];
         xmax = tf1.fSave[np + 2];

         if (xmin === xmax) {
            // xmin = tf1.fSave[np];
            const mp = this.getMainPainter();
            if (isFunc(mp?.getHisto))
               custom_xaxis = mp?.getHisto()?.fXaxis;
         }

         if (custom_xaxis) {
            ensureBins(hist.fXaxis.fNbins);
            Object.assign(hist.fXaxis, custom_xaxis);
            // TODO: find first bin

            for (let n = 0; n < np; ++n) {
               const y = tf1.fSave[n];
               hist.setBinContent(n + 1, Number.isFinite(y) ? y : 0);
            }
         } else {
            ensureBins(tf1.fNpx);
            hist.fXaxis.fXmin = tf1.fXmin;
            hist.fXaxis.fXmax = tf1.fXmax;

            for (let n = 0; n < tf1.fNpx; ++n) {
               const y = _getTF1Save(tf1, hist.fXaxis.GetBinCenter(n + 1));
               hist.setBinContent(n + 1, Number.isFinite(y) ? y : 0);
            }
         }
      }

      hist.fName = 'Func';
      setHistogramTitle(hist, tf1.fTitle);
      hist.fMinimum = tf1.fMinimum;
      hist.fMaximum = tf1.fMaximum;
      hist.fLineColor = tf1.fLineColor;
      hist.fLineStyle = tf1.fLineStyle;
      hist.fLineWidth = tf1.fLineWidth;
      hist.fFillColor = tf1.fFillColor;
      hist.fFillStyle = tf1.fFillStyle;
      hist.fMarkerColor = tf1.fMarkerColor;
      hist.fMarkerStyle = tf1.fMarkerStyle;
      hist.fMarkerSize = tf1.fMarkerSize;
      hist.fBits |= kNoStats;
   }

   /** @summary Extract function ranges */
   extractAxesProperties(ndim) {
      super.extractAxesProperties(ndim);

      const func = this.$func, nsave = func?.fSave.length ?? 0;

      if (nsave > 3 && this._use_saved_points) {
         this.xmin = Math.min(this.xmin, func.fSave[nsave - 2]);
         this.xmax = Math.max(this.xmax, func.fSave[nsave - 1]);
      }
      if (func) {
         this.xmin = Math.min(this.xmin, func.fXmin);
         this.xmax = Math.max(this.xmax, func.fXmax);
      }
   }

   /** @summary Checks if it makes sense to zoom inside specified axis range */
   canZoomInside(axis, min, max) {
      const nsave = this.$func?.fSave.length ?? 0;
      if ((nsave > 3) && this._use_saved_points && (axis === 'x')) {
         // in the case where the points have been saved, useful for example
         // if we don't have the user's function
         const nb_points = nsave - 2,
             xmin = this.$func.fSave[nsave - 2],
             xmax = this.$func.fSave[nsave - 1];

         return Math.abs(xmax - xmin) / nb_points < Math.abs(max - min);
      }

      // if function calculated, one always could zoom inside
      return (axis === 'x') || (axis === 'y');
   }

      /** @summary retrurn tooltips for TF2 */
   getTF1Tooltips(pnt) {
      delete this.$tmp_tooltip;
      const lines = [this.getObjectHint()],
            funcs = this.getFramePainter()?.getGrFuncs(this.options.second_x, this.options.second_y);

      if (!funcs || !isFunc(this.$func?.evalPar)) {
         lines.push('grx = ' + pnt.x, 'gry = ' + pnt.y);
         return lines;
      }

      const x = funcs.revertAxis('x', pnt.x);
      let y = 0, gry = 0, iserror = false;

       try {
          y = this.$func.evalPar(x);
          gry = Math.round(funcs.gry(y));
       } catch {
          iserror = true;
       }

      lines.push('x = ' + funcs.axisAsText('x', x),
                 'value = ' + (iserror ? '<fail>' : floatToString(y, gStyle.fStatFormat)));

      if (!iserror)
         this.$tmp_tooltip = { y, gry };
      return lines;
   }

   /** @summary process tooltip event for TF1 object */
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
                    lines: this.getTF1Tooltips(pnt), exact: true, menu: true };

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
               .attr('cy', this.$tmp_tooltip.gry ?? pnt.y)
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

   /** @summary draw TF1 object */
   static async draw(dom, tf1, opt) {
      const web = scanTF1Options(opt);
      opt = web.opt;
      delete web.opt;
      let hist;

      if (web.webcanv_hist) {
         const dummy = new ObjectPainter(dom);
         hist = dummy.getPadPainter()?.findInPrimitives('Func', clTH1D);
      }

      if (!hist) {
         hist = createHistogram(clTH1D, 100);
         hist.fBits |= kNoStats;
      }

      if (!opt && getElementMainPainter(dom))
         opt = 'same';

      const painter = new TF1Painter(dom, hist);

      painter.$func = tf1;
      Object.assign(painter, web);

      painter.createTF1Histogram(tf1, hist);

      return THistPainter._drawHist(painter, opt);
   }

} // class TF1Painter

export { TF1Painter, produceTAxisLogScale, scanTF1Options };
