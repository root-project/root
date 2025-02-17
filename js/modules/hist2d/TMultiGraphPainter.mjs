import { create, createHistogram, clTH1I, clTH2I, clTObjString, clTHashList, kNoZoom, kNoStats, BIT } from '../core.mjs';
import { DrawOptions } from '../base/BasePainter.mjs';
import { ObjectPainter } from '../base/ObjectPainter.mjs';
import { FunctionsHandler } from './THistPainter.mjs';
import { TH1Painter, PadDrawOptions } from './TH1Painter.mjs';
import { TGraphPainter } from './TGraphPainter.mjs';
import { ensureTCanvas } from '../gpad/TCanvasPainter.mjs';


const kResetHisto = BIT(17);

/**
 * @summary Painter for TMultiGraph object.
 *
 * @private
 */

class TMultiGraphPainter extends ObjectPainter {

   /** @summary Create painter
     * @param {object|string} dom - DOM element for drawing or element id
     * @param {object} obj - TMultiGraph object to draw */
   constructor(dom, mgraph) {
      super(dom, mgraph);
      this.firstpainter = null;
      this.painters = []; // keep painters to be able update objects
   }

   /** @summary Cleanup TMultiGraph painter */
   cleanup() {
      this.painters = [];
      super.cleanup();
   }

   /** @summary Update TMultiGraph object */
   updateObject(obj) {
      if (!this.matchObjectType(obj))
         return false;

      const mgraph = this.getObject(),
            graphs = obj.fGraphs,
            pp = this.getPadPainter();

      mgraph.fTitle = obj.fTitle;

      let isany = false;
      if (this.firstpainter) {
         const histo = this.scanGraphsRange(graphs, obj.fHistogram, pp?.getRootPad(true), true);
         if (this.firstpainter.updateObject(histo))
            isany = true;
      }

      const ngr = Math.min(graphs.arr.length, this.painters.length);

      // TODO: handle changing number of graphs
      for (let i = 0; i < ngr; ++i) {
         if (this.painters[i].updateObject(graphs.arr[i], (graphs.opt[i] || this._restopt) + this._auto))
            isany = true;
      }

      this._funcHandler = new FunctionsHandler(this, pp, obj.fFunctions);

      return isany;
   }

   /** @summary Redraw TMultiGraph
     * @desc may redraw histogram which was used to draw axes
     * @return {Promise} for ready */
    async redraw(reason) {
       const promise = this.firstpainter?.redraw(reason) ?? Promise.resolve(true),
             redrawNext = async indx => {
                if (indx >= this.painters.length)
                   return this;
                return this.painters[indx].redraw(reason).then(() => redrawNext(indx + 1));
             };

       return promise.then(() => redrawNext(0)).then(() => {
          const res = this._funcHandler?.drawNext(0) ?? this;
          delete this._funcHandler;
          return res;
       });
    }

   /** @summary Scan graphs range
     * @return {object} histogram for axes drawing */
   scanGraphsRange(graphs, histo, pad, reset_histo) {
      const mgraph = this.getObject(),
            rw = { xmin: 0, xmax: 0, ymin: 0, ymax: 0, first: true },
            test = (v1, v2) => { return Math.abs(v2-v1) < 1e-6; };
      let maximum, minimum, logx = false, logy = false,
          src_hist, dummy_histo = false;

      if (pad) {
         logx = pad.fLogx;
         logy = pad.fLogv ?? pad.fLogy;
      }

      // ignore existing histogram in 3d case
      if (this._3d && histo && !histo.fXaxis.fLabels)
         histo = null;

      if (!histo)
         src_hist = graphs.arr[0]?.fHistogram;
      else {
         dummy_histo = test(histo.fMinimum, -0.05) && test(histo.fMaximum, 1.05) &&
                       test(histo.fXaxis.fXmin, -0.05) && test(histo.fXaxis.fXmax, 1.05);
         src_hist = histo;
      }

      graphs.arr.forEach(gr => {
         if (gr.fNpoints === 0)
            return;
         if (gr.TestBit(kResetHisto))
            reset_histo = true;
         if (rw.first) {
            rw.xmin = rw.xmax = gr.fX[0];
            rw.ymin = rw.ymax = gr.fY[0];
            rw.first = false;
         }
         for (let i = 0; i < gr.fNpoints; ++i) {
            rw.xmin = Math.min(rw.xmin, gr.fX[i]);
            rw.xmax = Math.max(rw.xmax, gr.fX[i]);
            rw.ymin = Math.min(rw.ymin, gr.fY[i]);
            rw.ymax = Math.max(rw.ymax, gr.fY[i]);
         }
      });

      if (rw.xmin === rw.xmax)
         rw.xmax += 1;
      if (rw.ymin === rw.ymax)
         rw.ymax += 1;
      const dx = 0.05 * (rw.xmax - rw.xmin),
            dy = 0.05 * (rw.ymax - rw.ymin);

      let uxmin = rw.xmin - dx,
          uxmax = rw.xmax + dx;
      if (logy) {
         if (rw.ymin <= 0)
            rw.ymin = 0.001 * rw.ymax;
         minimum = rw.ymin / (1 + 0.5 * Math.log10(rw.ymax / rw.ymin));
         maximum = rw.ymax * (1 + 0.2 * Math.log10(rw.ymax / rw.ymin));
      } else {
         minimum = rw.ymin - dy;
         maximum = rw.ymax + dy;
      }
      if (minimum < 0 && rw.ymin >= 0)
         minimum = 0;
      if (maximum > 0 && rw.ymax <= 0)
         maximum = 0;

      const glob_minimum = minimum, glob_maximum = maximum;

      if (uxmin < 0 && rw.xmin >= 0)
         uxmin = logx ? 0.9 * rw.xmin : 0;
      if (uxmax > 0 && rw.xmax <= 0)
         uxmax = logx? 1.1 * rw.xmax : 0;

      if (mgraph.fMinimum !== kNoZoom)
         rw.ymin = minimum = mgraph.fMinimum;
      if (mgraph.fMaximum !== kNoZoom)
         rw.ymax = maximum = mgraph.fMaximum;

      if (minimum < 0 && rw.ymin >= 0 && logy)
         minimum = 0.9 * rw.ymin;
      if (maximum > 0 && rw.ymax <= 0 && logy)
         maximum = 1.1 * rw.ymax;
      if (minimum <= 0 && logy)
         minimum = 0.001 * maximum;
      if (!logy && minimum > 0 && minimum < 0.05*maximum)
         minimum = 0;
      if (uxmin <= 0 && logx)
         uxmin = (uxmax > 1000) ? 1 : 0.001 * uxmax;

      // Create a temporary histogram to draw the axis (if necessary)
      if (!histo || reset_histo || dummy_histo) {
         let xaxis, yaxis;
         if (this._3d) {
            histo = createHistogram(clTH2I, graphs.arr.length, 10);
            xaxis = histo.fXaxis;
            xaxis.fXmin = 0;
            xaxis.fXmax = graphs.arr.length;
            xaxis.fLabels = create(clTHashList);
            for (let i = 0; i < graphs.arr.length; i++) {
               const lbl = create(clTObjString);
               lbl.fString = graphs.arr[i].fTitle || `gr${i}`;
               lbl.fUniqueID = graphs.arr.length - i; // graphs drawn in reverse order
               xaxis.fLabels.Add(lbl, '');
            }
            xaxis = histo.fYaxis;
            yaxis = histo.fZaxis;
         } else {
            histo = createHistogram(clTH1I, 10);
            xaxis = histo.fXaxis;
            yaxis = histo.fYaxis;
         }

         if (src_hist) {
            xaxis.fTimeDisplay = src_hist.fXaxis.fTimeDisplay;
            xaxis.fTimeFormat = src_hist.fXaxis.fTimeFormat;
            xaxis.fTitle = src_hist.fXaxis.fTitle;
            yaxis.fTitle = src_hist.fYaxis.fTitle;
         }

         histo.fTitle = mgraph.fTitle;
         if (histo.fTitle.indexOf(';') >= 0) {
            const t = histo.fTitle.split(';');
            histo.fTitle = t[0];
            if (t[1]) xaxis.fTitle = t[1];
            if (t[2]) yaxis.fTitle = t[2];
         }
         xaxis.fXmin = uxmin;
         xaxis.fXmax = uxmax;
      }

      const axis = this._3d ? histo.fZaxis : histo.fYaxis;
      axis.fXmin = Math.min(minimum, glob_minimum);
      axis.fXmax = Math.max(maximum, glob_maximum);
      if (histo.fMinimum === kNoZoom)
         histo.fMinimum = minimum;
      if (histo.fMaximum === kNoZoom)
         histo.fMaximum = maximum;
      histo.fBits |= kNoStats;

      return histo;
   }

   /** @summary draw special histogram for axis
     * @return {Promise} when ready */
   async drawAxisHist(histo, hopt) {
      return TH1Painter.draw(this.getDrawDom(), histo, hopt);
   }

   /** @summary Draw graph  */
   async drawGraph(dom, gr, opt /* , pos3d */) {
      return TGraphPainter.draw(dom, gr, opt);
   }

   /** @summary method draws next graph  */
   async drawNextGraph(indx, pad_painter) {
      const graphs = this.getObject().fGraphs;

      // at the end of graphs drawing draw functions (if any)
      if (indx >= graphs.arr.length)
         return this;

      const gr = graphs.arr[indx],
            draw_opt = (graphs.opt[indx] || this._restopt) + this._auto,
            pos3d = graphs.arr.length - indx,
            subid = `graphs_${indx}`;

      // handling of 'pads' draw option
      if (pad_painter) {
         const subpad_painter = pad_painter.getSubPadPainter(indx+1);
         if (!subpad_painter)
            return this;

         subpad_painter.cleanPrimitives(true);

         return this.drawGraph(subpad_painter, gr, draw_opt, pos3d).then(subp => {
            if (subp) {
               subp.setSecondaryId(this, subid);
               this.painters.push(subp);
            }
            return this.drawNextGraph(indx+1, pad_painter);
         });
      }

      // used in automatic colors numbering
      if (this._auto)
         gr.$num_graphs = graphs.arr.length;

      return this.drawGraph(this.getPadPainter(), gr, draw_opt, pos3d).then(subp => {
         if (subp) {
            subp.setSecondaryId(this, subid);
            this.painters.push(subp);
         }

         return this.drawNextGraph(indx+1);
      });
   }

   /** @summary Fill TMultiGraph context menu */
   fillContextMenuItems(menu) {
      menu.addRedrawMenu(this);
   }

   /** @summary Redraw TMultiGraph object using provided option
     * @private */
   async redrawWith(opt, skip_cleanup) {
      if (!skip_cleanup) {
         this.firstpainter = null;
         this.painters = [];
         const pp = this.getPadPainter();
         pp?.removePrimitive(this, true);
         if (this._pads)
            pp?.divide(0, 0);
      }

      const d = new DrawOptions(opt),
            mgraph = this.getObject();

      this._3d = d.check('3D');
      this._auto = ''; // extra options for auto colors
      this._pads = d.check('PADS');
      ['PFC', 'PLC', 'PMC'].forEach(f => { if (d.check(f)) this._auto += ' ' + f; });

      let hopt = '', pad_painter = null;
      if (d.check('FB') && this._3d) hopt += 'FB'; // will be directly combined with LEGO
      PadDrawOptions.forEach(name => { if (d.check(name)) hopt += ';' + name; });

      this._restopt = d.remain();

      let promise = Promise.resolve(true);
      if (this._pads) {
         promise = ensureTCanvas(this, false).then(() => {
            pad_painter = this.getPadPainter();
            return pad_painter.divide(mgraph.fGraphs.arr.length, 0, true);
         });
      } else if (d.check('A') || !this.getMainPainter()) {
         const histo = this.scanGraphsRange(mgraph.fGraphs, mgraph.fHistogram, this.getPadPainter()?.getRootPad(true));

         promise = this.drawAxisHist(histo, hopt).then(ap => {
            ap.setSecondaryId(this, 'hist'); // mark that axis painter generated from mg
            this.firstpainter = ap;
         });
      }

      return promise.then(() => {
         this.addToPadPrimitives();
         return this.drawNextGraph(0, pad_painter);
      }).then(() => {
         if (this._pads)
            return this;
         const handler = new FunctionsHandler(this, this.getPadPainter(), this.getObject().fFunctions, true);
         return handler.drawNext(0); // returns painter
      });
   }

   /** @summary Draw TMultiGraph object */
   static async draw(dom, mgraph, opt) {
      const painter = new TMultiGraphPainter(dom, mgraph, opt);
      return painter.redrawWith(opt, true);
   }

} // class TMultiGraphPainter

export { TMultiGraphPainter };
