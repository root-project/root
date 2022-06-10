import { settings, isBatchMode, gStyle } from '../core.mjs';
import { floatToString } from '../base/BasePainter.mjs';
import { RObjectPainter } from '../base/RObjectPainter.mjs';
import { ensureRCanvas } from '../gpad/RCanvasPainter.mjs';
import { addDragHandler } from '../gpad/TFramePainter.mjs';


const ECorner = { kTopLeft: 1, kTopRight: 2, kBottomLeft: 3, kBottomRight: 4 };

/**
 * @summary Painter for RPave class
 *
 * @private
 */

class RPavePainter extends RObjectPainter {

   /** @summary Draw pave content
     * @desc assigned depending on pave class */
   drawContent() { return Promise.resolve(this); }

   /** @summary Draw pave */
   drawPave() {

      let rect = this.getPadPainter().getPadRect(),
          fp = this.getFramePainter();

      this.onFrame = fp && this.v7EvalAttr("onFrame", true);
      this.corner = this.v7EvalAttr("corner", ECorner.kTopRight);

      let visible      = this.v7EvalAttr("visible", true),
          offsetx      = this.v7EvalLength("offsetX", rect.width, 0.02),
          offsety      = this.v7EvalLength("offsetY", rect.height, 0.02),
          pave_width   = this.v7EvalLength("width", rect.width, 0.3),
          pave_height  = this.v7EvalLength("height", rect.height, 0.3);

      this.createG();

      this.draw_g.classed("most_upper_primitives", true); // this primitive will remain on top of list

      if (!visible)
         return Promise.resolve(this);

      this.createv7AttLine("border_");

      this.createv7AttFill();

      let pave_x = 0, pave_y = 0,
          fr = this.onFrame ? fp.getFrameRect() : rect;
      switch (this.corner) {
         case ECorner.kTopLeft:
            pave_x = fr.x + offsetx;
            pave_y = fr.y + offsety;
            break;
         case ECorner.kBottomLeft:
            pave_x = fr.x + offsetx;
            pave_y = fr.y + fr.height - offsety - pave_height;
            break;
         case ECorner.kBottomRight:
            pave_x = fr.x + fr.width - offsetx - pave_width;
            pave_y = fr.y + fr.height - offsety - pave_height;
            break;
         case ECorner.kTopRight:
         default:
            pave_x = fr.x + fr.width - offsetx - pave_width;
            pave_y = fr.y + offsety;
      }

      this.draw_g.attr("transform", `translate(${pave_x},${pave_y})`);

      this.draw_g.append("svg:rect")
                 .attr("x", 0)
                 .attr("width", pave_width)
                 .attr("y", 0)
                 .attr("height", pave_height)
                 .call(this.lineatt.func)
                 .call(this.fillatt.func);

      this.pave_width = pave_width;
      this.pave_height = pave_height;

      // here should be fill and draw of text

      return this.drawContent().then(() => {

         if (isBatchMode()) return this;

         // TODO: provide pave context menu as in v6
         if (settings.ContextMenu && this.paveContextMenu)
            this.draw_g.on("contextmenu", evnt => this.paveContextMenu(evnt));

         addDragHandler(this, { x: pave_x, y: pave_y, width: pave_width, height: pave_height,
                                minwidth: 20, minheight: 20, redraw: d => this.sizeChanged(d) });

         return this;
      });
   }

   /** @summary Process interactive moving of the stats box */
   sizeChanged(drag) {
      this.pave_width = drag.width;
      this.pave_height = drag.height;

      let pave_x = drag.x,
          pave_y = drag.y,
          rect = this.getPadPainter().getPadRect(),
          fr = this.onFrame ? this.getFramePainter().getFrameRect() : rect,
          offsetx = 0, offsety = 0, changes = {};

      switch (this.corner) {
         case ECorner.kTopLeft:
            offsetx = pave_x - fr.x;
            offsety = pave_y - fr.y;
            break;
         case ECorner.kBottomLeft:
            offsetx = pave_x - fr.x;
            offsety = fr.y + fr.height - pave_y - this.pave_height;
            break;
         case ECorner.kBottomRight:
            offsetx = fr.x + fr.width - pave_x - this.pave_width;
            offsety = fr.y + fr.height - pave_y - this.pave_height;
            break;
         case ECorner.kTopRight:
         default:
            offsetx = fr.x + fr.width - pave_x - this.pave_width;
            offsety = pave_y - fr.y;
      }

      this.v7AttrChange(changes, "offsetX", offsetx / rect.width);
      this.v7AttrChange(changes, "offsetY", offsety / rect.height);
      this.v7AttrChange(changes, "width", this.pave_width / rect.width);
      this.v7AttrChange(changes, "height", this.pave_height / rect.height);
      this.v7SendAttrChanges(changes, false); // do not invoke canvas update on the server

      this.draw_g.select("rect")
                 .attr("width", this.pave_width)
                 .attr("height", this.pave_height);

      this.drawContent();
   }

   /** @summary Redraw RPave object */
   redraw(/*reason*/) {
      return this.drawPave();
   }

   /** @summary draw RPave object */
   static draw(dom, pave, opt) {
      let painter = new RPavePainter(dom, pave, opt, "pave");
      return ensureRCanvas(painter, false).then(() => painter.drawPave());
   }
}


/**
 * @summary Painter for RLegend class
 *
 * @private
 */

class RLegendPainter extends RPavePainter {

   /** @summary draw RLegend content */
   drawContent() {
      let legend     = this.getObject(),
          textFont   = this.v7EvalFont("text", { size: 12, color: "black", align: 22 }),
          width      = this.pave_width,
          height     = this.pave_height,
          nlines     = legend.fEntries.length,
          pp         = this.getPadPainter();

      if (legend.fTitle) nlines++;

      if (!nlines || !pp) return Promise.resolve(this);

      let stepy = height / nlines, posy = 0, margin_x = 0.02 * width;

      textFont.setSize(height/(nlines * 1.2));
      this.startTextDrawing(textFont, 'font' );

      if (legend.fTitle) {
         this.drawText({ latex: 1, width: width - 2*margin_x, height: stepy, x: margin_x, y: posy, text: legend.fTitle });
         posy += stepy;
      }

      for (let i = 0; i < legend.fEntries.length; ++i) {
         let objp = null, entry = legend.fEntries[i], w4 = Math.round(width/4);

         this.drawText({ latex: 1, width: 0.75*width - 3*margin_x, height: stepy, x: 2*margin_x + w4, y: posy, text: entry.fLabel });

         if (entry.fDrawableId != "custom") {
            objp = pp.findSnap(entry.fDrawableId, true);
         } else if (entry.fDrawable.fIO) {
            objp = new RObjectPainter(this.getDom(), entry.fDrawable.fIO);
            if (entry.fLine) objp.createv7AttLine();
            if (entry.fFill) objp.createv7AttFill();
            if (entry.fMarker) objp.createv7AttMarker();
         }

         if (objp && entry.fFill && objp.fillatt)
            this.draw_g
              .append("svg:path")
              .attr("d", `M${Math.round(margin_x)},${Math.round(posy + stepy*0.1)}h${w4}v${Math.round(stepy*0.8)}h${-w4}z`)
              .call(objp.fillatt.func);

         if (objp && entry.fLine && objp.lineatt)
            this.draw_g
              .append("svg:path")
              .attr("d", `M${Math.round(margin_x)},${Math.round(posy + stepy/2)}h${w4}`)
              .call(objp.lineatt.func);

         if (objp && entry.fError && objp.lineatt)
            this.draw_g
              .append("svg:path")
              .attr("d", `M${Math.round(margin_x + width/8)},${Math.round(posy + stepy*0.2)}v${Math.round(stepy*0.6)}`)
              .call(objp.lineatt.func);

         if (objp && entry.fMarker && objp.markeratt)
            this.draw_g.append("svg:path")
                .attr("d", objp.markeratt.create(margin_x + width/8, posy + stepy/2))
                .call(objp.markeratt.func);

         posy += stepy;
      }

      return this.finishTextDrawing();
   }

   /** @summary draw RLegend object */
   static draw(dom, legend, opt) {
      let painter = new RLegendPainter(dom, legend, opt, "legend");
      return ensureRCanvas(painter, false).then(() => painter.drawPave());
   }

} // class RLegendPainter


/**
 * @summary Painter for RPaveText class
 *
 * @private
 */

class RPaveTextPainter extends RPavePainter {

   /** @summary draw RPaveText content */
   drawContent() {
      let pavetext  = this.getObject(),
          textFont  = this.v7EvalFont("text", { size: 12, color: "black", align: 22 }),
          width     = this.pave_width,
          height    = this.pave_height,
          nlines    = pavetext.fText.length;

      if (!nlines) return;

      let stepy = height / nlines, posy = 0, margin_x = 0.02 * width;

      textFont.setSize(height/(nlines * 1.2))

      this.startTextDrawing(textFont, 'font');

      for (let i = 0; i < pavetext.fText.length; ++i) {
         let line = pavetext.fText[i];

         this.drawText({ latex: 1, width: width - 2*margin_x, height: stepy, x: margin_x, y: posy, text: line });
         posy += stepy;
      }

      return this.finishTextDrawing(undefined, true);
   }

   /** @summary draw RPaveText object */
   static draw(dom, pave, opt) {
      let painter = new RPaveTextPainter(dom, pave, opt, "pavetext");
      return ensureRCanvas(painter, false).then(() => painter.drawPave());
   }

} // class RPaveTextPainter

/**
 * @summary Painter for RHistStats class
 *
 * @private
 */

class RHistStatsPainter extends RPavePainter {

   /** @summary clear entries from stat box */
   clearStat() {
      this.stats_lines = [];
   }

   /** @summary add text entry to stat box */
   addText(line) {
      this.stats_lines.push(line);
   }

   /** @summary update statistic from the server */
   updateStatistic(reply) {
      this.stats_lines = reply.lines;
      this.drawStatistic(this.stats_lines);
   }

   /** @summary fill statistic */
   fillStatistic() {
      let pp = this.getPadPainter();
      if (pp && pp._fast_drawing) return false;

      let obj = this.getObject();
      if (obj.fLines !== undefined) {
         this.stats_lines = obj.fLines;
         delete obj.fLines;
         return true;
      }

      if (this.v7OfflineMode()) {
         let main = this.getMainPainter();
         if (!main || (typeof main.fillStatistic !== 'function')) return false;
         // we take statistic from main painter
         return main.fillStatistic(this, gStyle.fOptStat, gStyle.fOptFit);
      }

      // show lines which are exists, maybe server request will be recieved later
      return (this.stats_lines !== undefined);
   }

   /** @summary format float value as string
     * @private */
   format(value, fmt) {
      if (!fmt) fmt = "stat";

      switch(fmt) {
         case "stat" : fmt = gStyle.fStatFormat; break;
         case "fit": fmt = gStyle.fFitFormat; break;
         case "entries": if ((Math.abs(value) < 1e9) && (Math.round(value) == value)) return value.toFixed(0); fmt = "14.7g"; break;
         case "last": fmt = this.lastformat; break;
      }

      let res = floatToString(value, fmt || "6.4g", true);

      this.lastformat = res[1];

      return res[0];
   }

   /** @summary Draw content */
   drawContent() {
      if (this.fillStatistic())
         return this.drawStatistic(this.stats_lines);

      return Promise.resolve(this);
   }

   /** @summary Change mask */
   changeMask(nbit) {
      let obj = this.getObject(), mask = (1<<nbit);
      if (obj.fShowMask & mask)
         obj.fShowMask = obj.fShowMask & ~mask;
      else
         obj.fShowMask = obj.fShowMask | mask;

      if (this.fillStatistic())
         this.drawStatistic(this.stats_lines);
   }

   /** @summary Context menu */
   statsContextMenu(evnt) {
      evnt.preventDefault();
      evnt.stopPropagation(); // disable main context menu

      createMenu(evnt, this).then(menu => {
         let obj = this.getObject(),
             action = this.changeMask.bind(this);

         menu.add("header: StatBox");

         for (let n=0;n<obj.fEntries.length; ++n)
            menu.addchk((obj.fShowMask & (1<<n)), obj.fEntries[n], n, action);

         return this.fillObjectExecMenu(menu);
     }).then(menu => menu.show());
   }

   /** @summary Draw statistic */
   drawStatistic(lines) {

      let textFont = this.v7EvalFont("stats_text", { size: 12, color: "black", align: 22 }),
          first_stat = 0, num_cols = 0, maxlen = 0,
          width = this.pave_width,
          height = this.pave_height;

      if (!lines) return Promise.resolve(this);

      let nlines = lines.length;
      // adjust font size
      for (let j = 0; j < nlines; ++j) {
         let line = lines[j];
         if (j > 0) maxlen = Math.max(maxlen, line.length);
         if ((j == 0) || (line.indexOf('|') < 0)) continue;
         if (first_stat === 0) first_stat = j;
         let parts = line.split("|");
         if (parts.length > num_cols)
            num_cols = parts.length;
      }

      // for characters like 'p' or 'y' several more pixels required to stay in the box when drawn in last line
      let stepy = height / nlines, has_head = false, margin_x = 0.02 * width;

      let text_g = this.draw_g.select(".statlines");
      if (text_g.empty())
         text_g = this.draw_g.append("svg:g").attr("class", "statlines");
      else
         text_g.selectAll("*").remove();

      textFont.setSize(height/(nlines * 1.2));
      this.startTextDrawing(textFont, 'font' , text_g);

      if (nlines == 1) {
         this.drawText({ width: width, height: height, text: lines[0], latex: 1, draw_g: text_g });
      } else
      for (let j = 0; j < nlines; ++j) {
         let posy = j*stepy;

         if (first_stat && (j >= first_stat)) {
            let parts = lines[j].split("|");
            for (let n = 0; n < parts.length; ++n)
               this.drawText({ align: "middle", x: width * n / num_cols, y: posy, latex: 0,
                               width: width/num_cols, height: stepy, text: parts[n], draw_g: text_g });
         } else if (lines[j].indexOf('=') < 0) {
            if (j == 0) {
               has_head = true;
               let max_hlen = Math.max(maxlen, Math.round((width-2*margin_x)/stepy/0.65));
               if (lines[j].length > max_hlen + 5)
                  lines[j] = lines[j].slice(0,max_hlen+2) + "...";
            }
            this.drawText({ align: (j == 0) ? "middle" : "start", x: margin_x, y: posy,
                            width: width - 2*margin_x, height: stepy, text: lines[j], draw_g: text_g });
         } else {
            let parts = lines[j].split("="), args = [];

            for (let n = 0; n < 2; ++n) {
               let arg = {
                  align: (n == 0) ? "start" : "end", x: margin_x, y: posy,
                  width: width-2*margin_x, height: stepy, text: parts[n], draw_g: text_g,
                  _expected_width: width-2*margin_x, _args: args,
                  post_process: function(painter) {
                    if (this._args[0].ready && this._args[1].ready)
                       painter.scaleTextDrawing(1.05*(this._args[0].result_width && this._args[1].result_width)/this.__expected_width, this.draw_g);
                  }
               };
               args.push(arg);
            }

            for (let n = 0; n < 2; ++n)
               this.drawText(args[n]);
         }
      }

      let lpath = "";

      if (has_head)
         lpath += "M0," + Math.round(stepy) + "h" + width;

      if ((first_stat > 0) && (num_cols > 1)) {
         for (let nrow = first_stat; nrow < nlines; ++nrow)
            lpath += "M0," + Math.round(nrow * stepy) + "h" + width;
         for (let ncol = 0; ncol < num_cols - 1; ++ncol)
            lpath += "M" + Math.round(width / num_cols * (ncol + 1)) + "," + Math.round(first_stat * stepy) + "V" + height;
      }

      if (lpath) this.draw_g.append("svg:path").attr("d",lpath) /*.call(this.lineatt.func)*/;

      return this.finishTextDrawing(text_g);
   }

   /** @summary Redraw stats box */
   redraw(reason) {
      if (reason && (typeof reason == "string") && (reason.indexOf("zoom") == 0) && this.v7NormalMode()) {
         let req = {
            _typename: "ROOT::Experimental::RHistStatBoxBase::RRequest",
            mask: this.getObject().fShowMask // lines to show in stat box
         };

         this.v7SubmitRequest("stat", req, reply => this.updateStatistic(reply));
      }

      return this.drawPave();
   }

   /** @summary draw RHistStats object */
   static draw(dom, stats, opt) {
      let painter = new RHistStatsPainter(dom, stats, opt, stats);
      return ensureRCanvas(painter, false).then(() => painter.drawPave());
   }

} // class RHistStatsPainter


export { RPavePainter, RLegendPainter, RPaveTextPainter, RHistStatsPainter };
