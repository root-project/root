import { createHistogram, kNoStats, settings, gStyle, clTF2, clTH2F, isStr, isFunc } from '../core.mjs';
import { TH2Painter } from '../hist/TH2Painter.mjs';
import { proivdeEvalPar, produceTAxisLogScale } from '../hist/TF1Painter.mjs';
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

   /** @summary Update histogram */
   updateObject(obj /*, opt*/) {
      if (!obj || (this.getClassName() != obj._typename)) return false;
      delete obj.evalPar;
      let histo = this.getHisto();

      if (this.webcanv_hist) {
         let h0 = this.getPadPainter()?.findInPrimitives('Func', clTH2F);
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
      if (!this._use_saved_points && (reason == 'logx' || reason == 'logy' || reason == 'zoom')) {
         this.createTF2Histogram(this.$func, this.getHisto());
         this.scanContent();
      }

      return super.redraw(reason);
   }

   /** @summary Create histogram for TF2 drawing
     * @private */
   createTF2Histogram(func, hist = undefined) {
      let nsave = func.fSave.length;
      if ((nsave > 6) && (nsave !== (func.fSave[nsave-2]+1)*(func.fSave[nsave-1]+1) + 6))
         nsave = 0;

      this._use_saved_points = (nsave > 6) && (settings.PreferSavedPoints || this.force_saved);

      let fp = this.getFramePainter(),
          pad = this.getPadPainter()?.getRootPad(true),
          logx = pad?.fLogx, logy = pad?.fLogy,
          xmin = func.fXmin, xmax = func.fXmax,
          ymin = func.fYmin, ymax = func.fYmax,
          gr = fp?.getGrFuncs(this.second_x, this.second_y);

     if (gr?.zoom_xmin !== gr?.zoom_xmax) {
         xmin = Math.min(xmin, gr.zoom_xmin);
         xmax = Math.max(xmax, gr.zoom_xmax);
      }

     if (gr?.zoom_ymin !== gr?.zoom_ymax) {
         ymin = Math.min(ymin, gr.zoom_ymin);
         ymax = Math.max(ymax, gr.zoom_ymax);
      }

      const ensureBins = (nx, ny) => {
         if (hist.fNcells !== (nx + 2) * (ny + 2)) {
            hist.fNcells = (nx + 2) * (ny + 2);
            hist.fArray = new Float32Array(hist.fNcells);
            hist.fArray.fill(0);
         }
         hist.fXaxis.fNbins = nx;
         hist.fXaxis.fXbins = [];
         hist.fYaxis.fNbins = ny;
         hist.fYaxis.fXbins = [];
      };

      delete this._fail_eval;

      if (!this._use_saved_points) {
         let npx = Math.max(func.fNpx, 20),
             npy = Math.max(func.fNpy, 20),
             iserror = false;

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

         for (let j = 0; (j < npy) && !iserror; ++j)
            for (let i = 0; (i < npx) && !iserror; ++i) {

               let x = hist.fXaxis.GetBinCenter(i+1),
                   y = hist.fYaxis.GetBinCenter(j+1),
                   z = 0;

               try {
                  z = func.evalPar(x, y);
               } catch {
                  iserror = true;
               }

               if (!iserror)
                  hist.setBinContent(hist.getBin(i + 1, j + 1), Number.isFinite(z) ? z : 0);
            }

         if (iserror)
            this._fail_eval = true;

         if (iserror && (nsave > 6))
            this._use_saved_points = true;
      }

      if (this._use_saved_points) {
         let npx = Math.round(func.fSave[nsave-2]),
             npy = Math.round(func.fSave[nsave-1]),
             dx = (func.fSave[nsave-5] - func.fSave[nsave-6]) / (npx - 1),
             dy = (func.fSave[nsave-3] - func.fSave[nsave-4]) / (npy - 1);

         ensureBins(npx+1, npy+1);
         hist.fXaxis.fXmin = func.fSave[nsave-6] - dx/2;
         hist.fXaxis.fXmax = func.fSave[nsave-5] + dx/2;
         hist.fYaxis.fXmin = func.fSave[nsave-4] - dy/2;
         hist.fYaxis.fXmax = func.fSave[nsave-3] + dy/2;

         for (let k = 0, j = 0; j <= npy; ++j)
            for (let i = 0; i <= npx; ++i) {
               let z = func.fSave[k++];
               hist.setBinContent(hist.getBin(i+1,j+1), Number.isFinite(z) ? z : 0);
            }
      }

      hist.fName = 'Func';
      hist.fTitle = func.fTitle;
      hist.fMinimum = func.fMinimum;
      hist.fMaximum = func.fMaximum;
      //fHistogram->SetContour(fContour.fN, levels);
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

   extractAxesProperties(ndim) {
      super.extractAxesProperties(ndim);

      let func = this.$func, nsave = func?.fSave.length ?? 0;

      if (nsave > 6 && this._use_saved_points) {
         let npx = Math.round(func.fSave[nsave-2]),
             npy = Math.round(func.fSave[nsave-1]),
             dx = (func.fSave[nsave-5] - func.fSave[nsave-6]) / (npx - 1),
             dy = (func.fSave[nsave-3] - func.fSave[nsave-4]) / (npy - 1);

         this.xmin = Math.min(this.xmin, func.fSave[nsave-6] - dx/2);
         this.xmax = Math.max(this.xmax, func.fSave[nsave-5] + dx/2);
         this.ymin = Math.min(this.ymin, func.fSave[nsave-4] - dy/2);
         this.ymax = Math.max(this.ymax, func.fSave[nsave-3] + dy/2);

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

      let lines = [ this.getObjectHint() ],
          funcs = this.getFramePainter()?.getGrFuncs(this.options.second_x, this.options.second_y);

      if (!funcs || !isFunc(this.$func?.evalPar)) {
         lines.push('grx = ' + pnt.x, 'gry = ' + pnt.y);
         return lines;
      }

      let x = funcs.revertAxis('x', pnt.x),
          y = funcs.revertAxis('y', pnt.y),
          z = 0, iserror = false;

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

      let ttrect = this.draw_g?.select('.tooltip_bin');

      if (!this.draw_g || !pnt) {
         ttrect?.remove();
         return null;
      }

      let res = { name: this.$func?.fName, title: this.$func?.fTitle,
                  x: pnt.x, y: pnt.y,
                  color1: this.lineatt?.color ?? 'green',
                  color2: this.fillatt?.getFillColorAlt('blue') ?? 'blue',
                  lines: this.getTF2Tooltips(pnt), exact: true, menu: true };

      if (ttrect.empty())
         ttrect = this.draw_g.append('svg:circle')
                             .attr('class', 'tooltip_bin')
                             .style('pointer-events', 'none')
                             .style('fill', 'none')
                             .attr('r', (this.lineatt?.width ?? 1) + 4);

      ttrect.attr('cx', pnt.x)
            .attr('cy', pnt.y)
            .call(this.lineatt?.func)

      return res;
   }

   /** @summary fill information for TWebCanvas
     * @private */
   fillWebObjectOptions(opt) {
      // mark that saved points are used or evaluation failed
      opt.fcust = this._fail_eval ? 'func_fail' : '';
   }

   /** @summary draw TF2 object */
   static async draw(dom, tf2, opt) {
      if (!isStr(opt)) opt = '';
      let p = opt.indexOf(';webcanv_hist'), webcanv_hist = false, force_saved = false;
      if (p >= 0) {
         webcanv_hist = true;
         opt = opt.slice(0, p);
      }
      p = opt.indexOf(';force_saved');
      if (p >= 0) {
         force_saved = true;
         opt = opt.slice(0, p);
      }


      let d = new DrawOptions(opt);
      if (d.empty())
         opt = 'cont3';
      else if (d.opt === 'SAME')
         opt = 'cont2 same';
      else
         opt = d.opt;

      // workaround for old waves.C
      if (opt == 'SAMECOLORZ' || opt == 'SAMECOLOR' || opt == 'SAMECOLZ')
         opt = 'SAMECOL';

      if (opt.indexOf('SAME') == 0)
         if (!getElementMainPainter(dom))
            opt = 'A_ADJUST_FRAME_' + opt.slice(4);

      let hist;

      if (webcanv_hist) {
         let dummy = new ObjectPainter(dom);

         hist = dummy.getPadPainter()?.findInPrimitives('Func', clTH2F);
      }

      if (!hist) hist = createHistogram(clTH2F, 20, 20);

      let painter = new TF2Painter(dom, hist);

      painter.$func = tf2;
      painter.webcanv_hist = webcanv_hist;
      painter.force_saved = force_saved;
      painter.createTF2Histogram(tf2, hist);

      return THistPainter._drawHist(painter, opt);
   }

} // class TF2Painter

export { TF2Painter };
