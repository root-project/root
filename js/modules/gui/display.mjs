/// different display kinds and browser layout

import { select as d3_select, drag as d3_drag } from '../d3.mjs';

import { browser, internals, toJSON } from '../core.mjs';

import { compressSVG, BasePainter } from '../base/BasePainter.mjs';

import { getElementCanvPainter, selectActivePad, cleanup, resize, ObjectPainter } from '../base/ObjectPainter.mjs';

import { createMenu } from './menu.mjs';

import { detectRightButton, injectStyle } from './utils.mjs';


/**
 * @summary Base class to manage multiple document interface for drawings
 *
 * @private
 */

class MDIDisplay extends BasePainter {
   /** @summary constructor */
   constructor(frameid) {
      super();
      this.frameid = frameid;
      if (frameid != "$batch$") {
         this.setDom(frameid);
         this.selectDom().property('mdi', this);
      }
      this.cleanupFrame = cleanup; // use standard cleanup function by default
      this.active_frame_title = ""; // keep title of active frame
   }

   /** @summary Assign func which called for each newly created frame */
   setInitFrame(func) {
      this.initFrame = func;
      this.forEachFrame(frame => func(frame));
   }

   /** @summary method called before new frame is created */
   beforeCreateFrame(title) { this.active_frame_title = title; }

   /** @summary method called after new frame is created
     * @private */
   afterCreateFrame(frame) {
      if (typeof this.initFrame == 'function')
         this.initFrame(frame);
      return frame;
   }

   /** @summary method dedicated to iterate over existing panels
     * @param {function} userfunc is called with arguments (frame)
     * @param {boolean} only_visible let select only visible frames */
   forEachFrame(userfunc, only_visible) {
      console.warn(`forEachFrame not implemented in MDIDisplay ${typeof userfunc} ${only_visible}`);
   }

   /** @summary method dedicated to iterate over existing panles
     * @param {function} userfunc is called with arguments (painter, frame)
     * @param {boolean} only_visible let select only visible frames */
   forEachPainter(userfunc, only_visible) {
      this.forEachFrame(frame => {
         new ObjectPainter(frame).forEachPainter(painter => userfunc(painter, frame));
      }, only_visible);
   }

   /** @summary Returns total number of drawings */
   numDraw() {
      let cnt = 0;
      this.forEachFrame(() => ++cnt);
      return cnt;
   }

   /** @summary Serach for the frame using item name */
   findFrame(searchtitle, force) {
      let found_frame = null;

      this.forEachFrame(frame => {
         if (d3_select(frame).attr('frame_title') == searchtitle)
            found_frame = frame;
      });

      if (!found_frame && force)
         found_frame = this.createFrame(searchtitle);

      return found_frame;
   }

   /** @summary Activate frame */
   activateFrame(frame) { this.active_frame_title = d3_select(frame).attr('frame_title'); }

   /** @summary Return active frame */
   getActiveFrame() { return this.findFrame(this.active_frame_title); }

   /** @summary perform resize for each frame
     * @protected */
   checkMDIResize(only_frame_id, size) {

      let resized_frame = null;

      this.forEachPainter((painter, frame) => {

         if (only_frame_id && (d3_select(frame).attr('id') != only_frame_id)) return;

         if ((painter.getItemName()!==null) && (typeof painter.checkResize == 'function')) {
            // do not call resize for many painters on the same frame
            if (resized_frame === frame) return;
            painter.checkResize(size);
            resized_frame = frame;
         }
      });
   }

   /** @summary Cleanup all drawings */
   cleanup() {
      this.active_frame_title = "";

      this.forEachFrame(this.cleanupFrame);

      this.selectDom().html("").property('mdi', null);
   }

} // class MDIDisplay


/**
 * @summary Custom MDI display
 *
 * @desc All HTML frames should be created before and add via {@link CustomDisplay.addFrame} calls
 * @private
 */

class CustomDisplay extends MDIDisplay {
   constructor() {
      super("dummy");
      this.frames = {}; // array of configured frames
   }

   addFrame(divid, itemname) {
      if (!(divid in this.frames)) this.frames[divid] = "";

      this.frames[divid] += (itemname + ";");
   }

   forEachFrame(userfunc) {
      let ks = Object.keys(this.frames);
      for (let k = 0; k < ks.length; ++k) {
         let node = d3_select("#"+ks[k]);
         if (!node.empty())
            userfunc(node.node());
      }
   }

   createFrame(title) {
      this.beforeCreateFrame(title);

      let ks = Object.keys(this.frames);
      for (let k = 0; k < ks.length; ++k) {
         let items = this.frames[ks[k]];
         if (items.indexOf(title+";") >= 0)
            return d3_select("#"+ks[k]).node();
      }
      return null;
   }

   cleanup() {
      super.cleanup();
      this.forEachFrame(frame => d3_select(frame).html(""));
   }

} // class CustomDisplay

/**
 * @summary Generic grid MDI display
 *
 * @private
 */

class GridDisplay extends MDIDisplay {

 /** @summary Create GridDisplay instance
   * @param {string} frameid - where grid display is created
   * @param {string} kind - kind of grid
   * @desc  following kinds are supported
   *    - vertical or horizontal - only first letter matters, defines basic orientation
   *    - 'x' in the name disable interactive separators
   *    - v4 or h4 - 4 equal elements in specified direction
   *    - v231 -  created 3 vertical elements, first divided on 2, second on 3 and third on 1 part
   *    - v23_52 - create two vertical elements with 2 and 3 subitems, size ratio 5:2
   *    - gridNxM - normal grid layout without interactive separators
   *    - gridiNxM - grid layout with interactive separators
   *    - simple - no layout, full frame used for object drawings */
   constructor(frameid, kind, kind2) {

      super(frameid);

      this.framecnt = 0;
      this.getcnt = 0;
      this.groups = [];
      this.vertical = kind && (kind[0] == 'v');
      this.use_separarators = !kind || (kind.indexOf("x")<0);
      this.simple_layout = false;

      let dom = this.selectDom();
      dom.style('overflow','hidden');

      if (kind === "simple") {
         this.simple_layout = true;
         this.use_separarators = false;
         this.framecnt = 1;
         return;
      }

      let num = 2, arr = undefined, sizes = undefined;

      if ((kind.indexOf("grid") == 0) || kind2) {
         if (kind2) kind = kind + "x" + kind2;
               else kind = kind.slice(4).trim();
         this.use_separarators = false;
         if (kind[0] === "i") {
            this.use_separarators = true;
            kind = kind.slice(1);
         }

         let separ = kind.indexOf("x"), sizex, sizey;

         if (separ > 0) {
            sizey = parseInt(kind.slice(separ + 1));
            sizex = parseInt(kind.slice(0, separ));
         } else {
            sizex = sizey = parseInt(kind);
         }

         if (!Number.isInteger(sizex)) sizex = 3;
         if (!Number.isInteger(sizey)) sizey = 3;

         if (sizey > 1) {
            this.vertical = true;
            num = sizey;
            if (sizex > 1)
               arr = new Array(num).fill(sizex);
         } else if (sizex > 1) {
            this.vertical = false;
            num = sizex;
         } else {
            this.simple_layout = true;
            this.use_separarators = false;
            this.framecnt = 1;
            return;
         }
         kind = "";
      }

      if (kind && kind.indexOf("_") > 0) {
         let arg = parseInt(kind.slice(kind.indexOf("_")+1), 10);
         if (Number.isInteger(arg) && (arg > 10)) {
            kind = kind.slice(0, kind.indexOf("_"));
            sizes = [];
            while (arg>0) {
               sizes.unshift(Math.max(arg % 10, 1));
               arg = Math.round((arg-sizes[0])/10);
               if (sizes[0]===0) sizes[0]=1;
            }
         }
      }

      kind = kind ? parseInt(kind.replace( /^\D+/g, ''), 10) : 0;
      if (Number.isInteger(kind) && (kind > 1)) {
         if (kind < 10) {
            num = kind;
         } else {
            arr = [];
            while (kind>0) {
               arr.unshift(kind % 10);
               kind = Math.round((kind-arr[0])/10);
               if (arr[0]==0) arr[0]=1;
            }
            num = arr.length;
         }
      }

      if (sizes && (sizes.length!==num)) sizes = undefined;

      if (!this.simple_layout) {
         injectStyle(`
.jsroot_vline:after {
    content:"";
    position: absolute;
    top: 0;
    bottom: 0;
    left: 50%;
    border-left: 1px dotted #ff0000;
}
.jsroot_hline:after {
    content:"";
    position: absolute;
    left: 0;
    right: 0;
    top: 50%;
    border-top: 1px dotted #ff0000;
}
.jsroot_separator {
   pointer-events: all;
   border: 0;
   margin: 0;
   padding: 0;
}
`, dom.node());
         this.createGroup(this, dom, num, arr, sizes);
      }
   }

   /** @summary Create frames group
     * @private */
   createGroup(handle, main, num, childs, sizes) {

      if (!sizes) sizes = new Array(num);
      let sum1 = 0, sum2 = 0;
      for (let n = 0; n < num; ++n)
         sum1 += (sizes[n] || 1);
      for (let n = 0; n < num; ++n) {
         sizes[n] = Math.round(100 * (sizes[n] || 1) / sum1);
         sum2 += sizes[n];
         if (n==num-1) sizes[n] += (100-sum2); // make 100%
      }

      for (let cnt = 0; cnt < num; ++cnt) {
         let group = { id: cnt, drawid: -1, position: 0, size: sizes[cnt] };
         if (cnt > 0) group.position = handle.groups[cnt-1].position + handle.groups[cnt-1].size;
         group.position0 = group.position;

         if (!childs || !childs[cnt] || childs[cnt]<2) group.drawid = this.framecnt++;

         handle.groups.push(group);

         let elem = main.append("div").attr('groupid', group.id);

         if (handle.vertical)
            elem.style('float', 'bottom').style('height',group.size+'%').style('width','100%');
         else
            elem.style('float', 'left').style('width',group.size+'%').style('height','100%');

         if (group.drawid>=0) {
            elem.classed('jsroot_newgrid', true);
            if (typeof this.frameid === 'string')
               elem.attr('id', this.frameid + "_" + group.drawid);
         } else {
            elem.style('display','flex').style('flex-direction', handle.vertical ? "row" : "column");
         }

         if (childs && (childs[cnt]>1)) {
            group.vertical = !handle.vertical;
            group.groups = [];
            elem.style('overflow','hidden');
            this.createGroup(group, elem, childs[cnt]);
         }
      }

      if (this.use_separarators && this.createSeparator)
         for (let cnt = 1; cnt < num; ++cnt)
            this.createSeparator(handle, main, handle.groups[cnt]);
   }

   /** @summary Handle interactive sepearator movement
     * @private */
   handleSeparator(elem, action) {
      let separ = d3_select(elem),
          parent = elem.parentNode,
          handle = separ.property('handle'),
          id = separ.property('separator_id'),
          group = handle.groups[id];

       const findGroup = grid => {
         let chld = parent.firstChild;
         while (chld) {
            if (chld.getAttribute("groupid") == grid)
               return d3_select(chld);
            chld = chld.nextSibling;
         }
         // should never happen, but keep it here like
         return d3_select(parent).select(`[groupid='${grid}']`);
       }, setGroupSize = grid => {
          let name = handle.vertical ? 'height' : 'width',
              size = handle.groups[grid].size+'%';
          findGroup(grid).style(name, size)
                         .selectAll(".jsroot_separator").style(name, size);
       }, resizeGroup = grid => {
          let sel = findGroup(grid);
          if (!sel.classed('jsroot_newgrid')) sel = sel.select(".jsroot_newgrid");
          sel.each(function() { resize(this); });
       };

      if (action == "start") {
         group.startpos = group.position;
         group.acc_drag = 0;
         return;
      }

      if (action == "end") {
         if (Math.abs(group.startpos - group.position) >= 0.5) {
            resizeGroup(id-1);
            resizeGroup(id);
          }
          return;
      }

      let pos;
      if (action == "restore") {
          pos = group.position0;
      } else if (handle.vertical) {
          group.acc_drag += action.dy;
          pos = group.startpos + ((group.acc_drag + 2) / parent.clientHeight) * 100;
      } else {
          group.acc_drag += action.dx;
          pos = group.startpos + ((group.acc_drag + 2) / parent.clientWidth) * 100;
      }

      let diff = group.position - pos;

      if (Math.abs(diff) < 0.3) return; // if no significant change, do nothing

      // do not change if size too small
      if (Math.min(handle.groups[id-1].size-diff, group.size+diff) < 3) return;

      handle.groups[id-1].size -= diff;
      group.size += diff;
      group.position = pos;

      separ.style(handle.vertical ? 'top' : 'left', `calc(${pos}% - 2px)`);

      setGroupSize(id-1);
      setGroupSize(id);

      if (action == "restore") {
          resizeGroup(id-1);
          resizeGroup(id);
      }

   }

   /** @summary Create group separator
     * @private */
   createSeparator(handle, main, group) {
      let separ = main.append("div");

      separ.classed('jsroot_separator', true)
           .classed(handle.vertical ? 'jsroot_hline' : 'jsroot_vline', true)
           .property('handle', handle)
           .property('separator_id', group.id)
           .style('position', 'absolute')
           .style(handle.vertical ? 'top' : 'left', `calc(${group.position}% - 2px)`)
           .style(handle.vertical ? 'width' : 'height', (handle.size || 100)+"%")
           .style(handle.vertical ? 'height' : 'width', '5px')
           .style('cursor', handle.vertical ? "ns-resize" : "ew-resize");

      let pthis = this, drag_move =
        d3_drag().on("start", function() { pthis.handleSeparator(this, "start"); })
                 .on("drag", function(evnt) { pthis.handleSeparator(this, evnt); })
                 .on("end", function() { pthis.handleSeparator(this, "end"); });

      separ.call(drag_move).on("dblclick", function() { pthis.handleSeparator(this, "restore"); });

      // need to get touches events handling in drag
      if (browser.touches && !main.on("touchmove"))
         main.on("touchmove", function() { });
   }


   /** @summary Call function for each frame */
   forEachFrame(userfunc) {
      if (this.simple_layout)
         userfunc(this.getGridFrame());
      else
         this.selectDom().selectAll('.jsroot_newgrid').each(function() {
            userfunc(this);
         });
   }

   /** @summary Returns active frame */
   getActiveFrame() {
      if (this.simple_layout)
         return this.getGridFrame();

      let found = super.getActiveFrame();
      if (found) return found;

      this.forEachFrame(frame => { if (!found) found = frame; });

      return found;
   }

   /** @summary Returns number of frames in grid layout */
   numGridFrames() { return this.framecnt; }

   /** @summary Return grid frame by its id */
   getGridFrame(id) {
      if (this.simple_layout)
         return this.selectDom('origin').node();
      let res = null;
      this.selectDom().selectAll('.jsroot_newgrid').each(function() {
         if (id-- === 0) res = this;
      });
      return res;
   }

   /** @summary Create new frame */
   createFrame(title) {
      this.beforeCreateFrame(title);

      let frame = null, maxloop = this.framecnt || 2;

      while (!frame && maxloop--) {
         frame = this.getGridFrame(this.getcnt);
         if (!this.simple_layout && this.framecnt)
            this.getcnt = (this.getcnt+1) % this.framecnt;

         if (d3_select(frame).classed("jsroot_fixed_frame")) frame = null;
      }

      if (frame) {
         this.cleanupFrame(frame);
         d3_select(frame).attr('frame_title', title);
      }

      return this.afterCreateFrame(frame);
   }

} // class GridDisplay

// ================================================

/**
 * @summary Generic flexible MDI display
 *
 * @private
 */

class FlexibleDisplay extends MDIDisplay {

   constructor(frameid) {
      super(frameid);
      this.cnt = 0; // use to count newly created frames
      this.selectDom().on('contextmenu', evnt => this.showContextMenu(evnt))
                      .style('overflow', 'auto');
   }

   /** @summary Cleanup all drawings */
   cleanup() {
      this.selectDom().style('overflow', null)
                      .on('contextmenu', null);
      this.cnt = 0;
      super.cleanup();
   }

   /** @summary call function for each frame */
   forEachFrame(userfunc,  only_visible) {
      if (typeof userfunc != 'function') return;

      let mdi = this, top = this.selectDom().select('.jsroot_flex_top');

      top.selectAll(".jsroot_flex_draw").each(function() {
         // check if only visible specified
         if (only_visible && (mdi.getFrameState(this) == "min")) return;

         userfunc(this);
      });
   }

   /** @summary return active frame */
   getActiveFrame() {
      let found = super.getActiveFrame();
      if (found && d3_select(found.parentNode).property("state") != "min") return found;

      found = null;
      this.forEachFrame(frame => { found = frame; }, true);
      return found;
   }

   /** @summary actiavte frame */
   activateFrame(frame) {
      if ((frame === 'first') || (frame === 'last')) {
         let res = null;
         this.forEachFrame(f => { if (frame == 'last' || !res) res = f; }, true);
         frame = res;
      }
      if (!frame) return;
      if (frame.getAttribute("class") != "jsroot_flex_draw") return;

      if (this.getActiveFrame() === frame) return;

      super.activateFrame(frame);

      let main = frame.parentNode;
      main.parentNode.append(main);

      if (this.getFrameState(frame) != "min") {
         selectActivePad({ pp: getElementCanvPainter(frame), active: true });
         resize(frame);
      }
   }

   /** @summary get frame state */
   getFrameState(frame) {
      let main = d3_select(frame.parentNode);
      return main.property("state");
   }

   /** @summary returns frame rect */
   getFrameRect(frame) {
      if (this.getFrameState(frame) == "max") {
         let top = this.selectDom().select('.jsroot_flex_top');
         return { x: 0, y: 0, w: top.node().clientWidth, h: top.node().clientHeight };
      }

      let main = d3_select(frame.parentNode), left = main.style('left'), top = main.style('top');

      return { x: parseInt(left.slice(0, left.length-2)), y: parseInt(top.slice(0, top.length-2)),
               w: main.node().clientWidth, h: main.node().clientHeight };
   }

   /** @summary change frame state */
   changeFrameState(frame, newstate,no_redraw) {
      let main = d3_select(frame.parentNode),
          state = main.property("state"),
          top = this.selectDom().select('.jsroot_flex_top');

      if (state == newstate)
         return false;

      if (state == "normal")
          main.property('original_style', main.attr('style'));

      // clear any previous settings
      top.style('overflow', null);

      switch (newstate) {
         case "min":
            main.style("height","auto").style("width", "auto");
            main.select(".jsroot_flex_draw").style("display","none");
            break;
         case "max":
            main.style("height","100%").style("width", "100%").style('left','').style('top','');
            main.select(".jsroot_flex_draw").style("display", null);
            top.style('overflow', 'hidden');
            break;
         default:
            main.select(".jsroot_flex_draw").style("display", null);
            main.attr("style", main.property("original_style"));
      }

      main.select(".jsroot_flex_header").selectAll("button").each(function(d) {
         let btn = d3_select(this);
         if (((d.t == "minimize") && (newstate == "min")) ||
             ((d.t == "maximize") && (newstate == "max")))
               btn.html("&#x259E;").attr("title", "restore");
         else
            btn.html(d.n).attr("title", d.t);
      });

      main.property("state", newstate);
      main.select(".jsroot_flex_resize").style("display", (newstate == "normal") ? null : "none");

      // adjust position of new minified rect
      if (newstate == "min") {
         const rect = this.getFrameRect(frame),
               top = this.selectDom().select('.jsroot_flex_top'),
               ww = top.node().clientWidth,
               hh = top.node().clientHeight,
               arr = [], step = 4,
               crossX = (r1,r2) => ((r1.x <= r2.x) && (r1.x + r1.w >= r2.x)) || ((r2.x <= r1.x) && (r2.x + r2.w >= r1.x)),
               crossY = (r1,r2) => ((r1.y <= r2.y) && (r1.y + r1.h >= r2.y)) || ((r2.y <= r1.y) && (r2.y + r2.h >= r1.y));

         this.forEachFrame(f => { if ((f!==frame) && (this.getFrameState(f) == "min")) arr.push(this.getFrameRect(f)); });

         rect.y = hh;
         do {
            rect.x = step;
            rect.y -= rect.h + step;
            let maxx = step, iscrossed = false;
            arr.forEach(r => {
               if (crossY(r,rect)) {
                  maxx = Math.max(maxx, r.x + r.w + step);
                  if (crossX(r,rect)) iscrossed = true;
               }
            });
            if (iscrossed) rect.x = maxx;
         } while ((rect.x + rect.w > ww - step) && (rect.y > 0));
         if (rect.y < 0) { rect.x = step; rect.y = hh - rect.h - step; }

         main.style("left", rect.x + "px").style("top", rect.y + "px");
      } else if (!no_redraw) {
         resize(frame);
      }

      return true;
   }

   /** @summary handle button click
     * @private */
   _clickButton(btn) {
      let kind = d3_select(btn).datum(),
          main = d3_select(btn.parentNode.parentNode),
          frame = main.select(".jsroot_flex_draw").node();

      if (kind.t == "close") {
         this.cleanupFrame(frame);
         main.remove();
         this.activateFrame('last'); // set active as last non-minfied window
         return;
      }

      let state = main.property("state"), newstate;
      if (kind.t == "maximize")
         newstate = (state == "max") ? "normal" : "max";
      else
         newstate = (state == "min") ? "normal" : "min";

      if (this.changeFrameState(frame, newstate))
         this.activateFrame(newstate != "min" ? frame : 'last');
   }

   /** @summary create new frame */
   createFrame(title) {

      this.beforeCreateFrame(title);

      let mdi = this,
          dom = this.selectDom(),
          top = dom.select(".jsroot_flex_top");

      if (this.cnt == 0) injectStyle(`
.jsroot_flex_top {
   overflow: auto;
   position: relative;
   height: 100%;
   width: 100%;
}

.jsroot_flex_btn {
   float: right;
   padding: 0;
   width: 1.4em;
   text-align: center;
   font-size: 10px;
   margin-top: 2px;
   margin-right: 4px;
}

.jsroot_flex_header {
   height: 23px;
   overflow: hidden;
   background-color: lightblue;
}

.jsroot_flex_header p {
   margin: 1px;
   float: left;
   font-size: 14px;
   padding-left: 5px;
}

.jsroot_flex_draw {
   overflow: hidden;
   width: 100%;
   height: calc(100% - 24px);
}

.jsroot_flex_frame {
   border: 1px solid black;
   box-shadow: 1px 1px 2px 2px #aaa;
   background: white;
}

.jsroot_flex_resize {
   position: absolute;
   right: 2px;
   bottom: 2px;
   overflow: hidden;
   cursor: nwse-resize;
}

.jsroot_flex_resizable_helper {
   border: 2px dotted #00F;
}`, dom.node());

      if (top.empty())
         top = dom.append("div").classed("jsroot_flex_top", true);

      let w = top.node().clientWidth,
          h = top.node().clientHeight,
          main = top.append('div');


      main.html(`<div class="jsroot_flex_header"><p>${title}</p></div>
                 <div id="${this.frameid}_cont${this.cnt}" class="jsroot_flex_draw"></div>
                 <div class="jsroot_flex_resize">&#x25FF;</div>`);

      main.attr("class", "jsroot_flex_frame")
         .style("position", "absolute")
         .style('left', Math.round(w * (this.cnt % 5)/10) + "px")
         .style('top', Math.round(h * (this.cnt % 5)/10) + "px")
         .style('width', Math.round(w * 0.58) + "px")
         .style('height', Math.round(h * 0.58) + "px")
         .property("state", "normal")
         .select(".jsroot_flex_header")
         .on("click", function() { mdi.activateFrame(d3_select(this.parentNode).select(".jsroot_flex_draw").node()); })
         .selectAll("button")
         .data([{ n: '&#x2715;', t: "close" }, { n: '&#x2594;', t: "maximize" }, { n: '&#x2581;', t: "minimize" }])
         .enter()
         .append("button")
         .attr("type", "button")
         .attr("class", "jsroot_flex_btn")
         .attr("title", d => d.t)
         .html(d => d.n)
         .on("click", function() { mdi._clickButton(this); });

      let moving_frame = null, moving_div = null, doing_move = false,
          drag_object = d3_drag().subject(Object), current = [];
      drag_object.on("start", function(evnt) {
         if (evnt.sourceEvent.target.type == "button")
            return mdi._clickButton(evnt.sourceEvent.target);

         if (detectRightButton(evnt.sourceEvent)) return;

         let main = d3_select(this.parentNode);
         if(!main.classed("jsroot_flex_frame") || (main.property("state") == "max")) return;

         doing_move = !d3_select(this).classed("jsroot_flex_resize");
         if (!doing_move && (main.property("state") == "min")) return;

         mdi.activateFrame(main.select(".jsroot_flex_draw").node());

         moving_div = top.append('div').classed("jsroot_flex_resizable_helper", true);

         moving_div.attr("style", main.attr("style"));

         if (main.property("state") == "min")
            moving_div.style("width", main.node().clientWidth + "px")
                      .style("height", main.node().clientHeight + "px");

         evnt.sourceEvent.preventDefault();
         evnt.sourceEvent.stopPropagation();

         moving_frame = main;
         current = [];

      }).on("drag", function(evnt) {
         if (!moving_div) return;
         evnt.sourceEvent.preventDefault();
         evnt.sourceEvent.stopPropagation();
         let changeProp = (i,name,dd) => {
            if (i >= current.length) {
               let v = moving_div.style(name);
               current[i] = parseInt(v.slice(0,v.length-2));
            }
            current[i] += dd;
            moving_div.style(name, Math.max(0, current[i])+"px");
         };
         if (doing_move) {
            changeProp(0, "left", evnt.dx);
            changeProp(1, "top", evnt.dy);
         } else {
            changeProp(0, "width", evnt.dx);
            changeProp(1, "height", evnt.dy);
         }
      }).on("end", function(evnt) {
         if (!moving_div) return;
         evnt.sourceEvent.preventDefault();
         evnt.sourceEvent.stopPropagation();
         if (doing_move) {
            moving_frame.style("left", moving_div.style("left"));
            moving_frame.style("top", moving_div.style("top"));
         } else {
            moving_frame.style("width", moving_div.style("width"));
            moving_frame.style("height", moving_div.style("height"));
         }
         moving_div.remove();
         moving_div = null;
         if (!doing_move)
            resize(moving_frame.select(".jsroot_flex_draw").node());
      });

      main.select(".jsroot_flex_header").call(drag_object);
      main.select(".jsroot_flex_resize").call(drag_object);

      let draw_frame = main.select('.jsroot_flex_draw')
                           .attr('frame_title', title)
                           .property('frame_cnt', this.cnt++)
                           .node();

      return this.afterCreateFrame(draw_frame);
   }

   /** @summary minimize all frames */
   minimizeAll() {
      this.forEachFrame(frame => this.changeFrameState(frame, "min"));
   }

   /** @summary close all frames */
   closeAllFrames() {
      let arr = [];
      this.forEachFrame(frame => arr.push(frame));
      arr.forEach(frame => {
         this.cleanupFrame(frame);
         d3_select(frame.parentNode).remove();
      });
   }

   /** @summary cascade frames */
   sortFrames(kind) {
      let arr = [];
      this.forEachFrame(frame => {
         let state = this.getFrameState(frame);
         if (state=="min") return;
         if (state == "max") this.changeFrameState(frame, "normal", true);
         arr.push(frame);
      });

      if (arr.length == 0) return;

      let top = this.selectDom(),
          w = top.node().clientWidth,
          h = top.node().clientHeight,
          dx = Math.min(40, Math.round(w*0.4/arr.length)),
          dy = Math.min(40, Math.round(h*0.4/arr.length)),
          nx = Math.ceil(Math.sqrt(arr.length)), ny = nx;

      // calculate number of divisions for "tile" sorting
      if ((nx > 1) && (nx*(nx-1) >= arr.length))
        if (w > h) ny--; else nx--;

      arr.forEach((frame,i) => {
         let main = d3_select(frame.parentNode);
         if (kind == "cascade")
            main.style('left', (i*dx) + "px")
                .style('top', (i*dy) + "px")
                .style('width', Math.round(w * 0.58) + "px")
                .style('height', Math.round(h * 0.58) + "px");
         else
            main.style('left', Math.round(w/nx*(i%nx)) + "px")
                .style('top', Math.round(h/ny*((i-i%nx)/nx)) + "px")
                .style('width', Math.round(w/nx - 4) + "px")
                .style('height', Math.round(h/ny - 4) + "px");
         resize(frame);
      });
   }

   /** @summary context menu */
   showContextMenu(evnt) {
      // handle context menu only for MDI area
      if ((evnt.target.getAttribute("class") != "jsroot_flex_top") || (this.numDraw() == 0)) return;

      evnt.preventDefault();

      let arr = [];
      this.forEachFrame(f => arr.push(f));
      let active = this.getActiveFrame();
      arr.sort((f1,f2) => { return  d3_select(f1).property('frame_cnt') < d3_select(f2).property('frame_cnt') ? -1 : 1; });

      createMenu(evnt, this).then(menu => {
         menu.add("header:Flex");
         menu.add("Cascade", () => this.sortFrames("cascade"));
         menu.add("Tile", () => this.sortFrames("tile"));
         menu.add("Minimize all", () => this.minimizeAll());
         menu.add("Close all", () => this.closeAllFrames());
         menu.add("separator");

         arr.forEach((f,i) => menu.addchk((f===active), ((this.getFrameState(f) == "min") ? "[min] " : "") + d3_select(f).attr("frame_title"), i,
                      arg => {
                        let frame = arr[arg];
                        if (this.getFrameState(frame) == "min")
                           this.changeFrameState(frame, "normal");
                        this.activateFrame(frame);
                      }));

         menu.show();
      });
   }

} // class FlexibleDisplay


/**
 * @summary Batch MDI display
 *
 * @desc Can be used together with hierarchy painter in node.js
 * @private
 */

class BatchDisplay extends MDIDisplay {

   constructor(width, height, jsdom_body) {
      super("$batch$");
      this.frames = []; // array of configured frames
      this.width = width || 1200;
      this.height = height || 800;
      this.jsdom_body = jsdom_body || d3_select('body'); // d3 body handle
   }

   forEachFrame(userfunc) {
      this.frames.forEach(userfunc)
   }

   createFrame(title) {
      this.beforeCreateFrame(title);

      let frame =
         this.jsdom_body.append('div')
             .style("visible", "hidden")
             .attr("width", this.width).attr("height", this.height)
             .style("width", this.width + "px").style("height", this.height + "px")
             .attr("id","jsroot_batch_" + this.frames.length)
             .attr("frame_title", title);

      if (this.frames.length == 0)
         internals.svg_3ds = undefined;

      this.frames.push(frame.node());

      return this.afterCreateFrame(frame.node());
   }

   /** @summary Returns number of created frames */
   numFrames() { return this.frames.length; }

   /** @summary returns JSON representation if any
     * @desc Now works only for inspector, can be called once */
   makeJSON(id, spacing) {
      let frame = this.frames[id];
      if (!frame) return;
      let obj = d3_select(frame).property('_json_object_');
      if (obj) {
         d3_select(frame).property('_json_object_', null);
         return toJSON(obj, spacing);
      }
   }

   /** @summary Create SVG for specified frame id */
   makeSVG(id) {
      let frame = this.frames[id];
      if (!frame) return;
      let main = d3_select(frame);
      let has_workarounds = internals.svg_3ds && internals.processSvgWorkarounds;
      main.select('svg')
          .attr("xmlns", "http://www.w3.org/2000/svg")
          .attr("xmlns:xlink", "http://www.w3.org/1999/xlink")
          .attr("width", this.width)
          .attr("height", this.height)
          .attr("title", null).attr("style", null).attr("class", null).attr("x", null).attr("y", null);

      let svg = main.html();
      if (has_workarounds)
         svg = internals.processSvgWorkarounds(svg, id != this.frames.length-1);

      svg = compressSVG(svg);

      main.remove();
      return svg;
   }

} // class BatchDisplay


/**
  * @summary Special browser layout
  *
  * @desc Contains three different areas for browser (left), status line (bottom) and central drawing
  * Main application is normal browser, but also used in other applications like ROOT6 canvas
  * @private
  */

class BrowserLayout {

   /** @summary Constructor */
   constructor(id, hpainter, objpainter) {
      this.gui_div = id;
      this.hpainter = hpainter; // painter for brwoser area (if any)
      this.objpainter = objpainter; // painter for object area (if any)
      this.browser_kind = null; // should be 'float' or 'fix'
   }

   /** @summary Selects main element */
   main() {
      return d3_select("#" + this.gui_div);
   }

   /** @summary Returns drawing divid */
   drawing_divid() {
      return this.gui_div + "_drawing";
   }

   /** @summary Check resize action */
   checkResize() {
      if (this.hpainter && (typeof this.hpainter.checkResize == 'function'))
         this.hpainter.checkResize();
      else if (this.objpainter && (typeof this.objpainter.checkResize == 'function')) {
         this.objpainter.checkResize(true);
      }
   }

   /** @summary method used to create basic elements
     * @desc should be called only once */
   create(with_browser) {
      let main = this.main();

      main.append("div").attr("id", this.drawing_divid())
                        .classed("jsroot_draw_area", true)
                        .style('position',"absolute").style('left',0).style('top',0).style('bottom',0).style('right',0);

      if (with_browser) main.append("div").classed("jsroot_browser", true);

      injectStyle(`
.jsroot_browser {
   pointer-events: none;
   position: absolute;
   left: 0;
   top: 0;
   bottom: 0;
   right:0;
   margin: 0;
   border: 0;
   overflow: hidden;
}
.jsroot_draw_area {
   background-color: #E6E6FA;
   overflow: hidden;
   margin: 0;
   border: 0;
}
.jsroot_browser_title {
   font-family: Verdana;
   font-size: 20px;
}
.jsroot_browser_btns {
   pointer-events: all;
   opacity: 0;
   display:flex;
   flex-direction: column;
}
.jsroot_browser_btns:hover {
   opacity: 0.3;
}
.jsroot_browser_area {
   background-color: #E6E6FA;
   font-size: 12px;
   font-family: Verdana;
   pointer-events: all;
   box-sizing: initial;
}
.jsroot_browser_area p {
   margin-top: 5px;
   margin-bottom: 5px;
   white-space: nowrap;
}
.jsroot_browser_hierarchy {
   flex: 1;
   margin-top: 2px;
}
.jsroot_status_area {
   background-color: #E6E6FA;
   overflow: hidden;
   font-size: 12px;
   font-family: Verdana;
   pointer-events: all;
}
.jsroot_float_browser {
   border: solid 3px white;
}
.jsroot_browser_resize {
   position: absolute;
   right: 3px;
   bottom: 3px;
   margin-bottom: 0px;
   margin-right: 0px;
   opacity: 0.5;
   cursor: se-resize;
}
.jsroot_separator {
   pointer-events: all;
   border: 0;
   margin: 0;
   padding: 0;
}
.jsroot_h_separator {
   cursor: ns-resize;
   background-color: azure;
}
.jsroot_v_separator {
   cursor: ew-resize;
   background-color: azure;
}
.jsroot_status_label {
   margin: 3px;
   margin-left: 5px;
   font-size: 14px;
   vertical-align: middle;
   white-space: nowrap;
}
`, main.node());
   }

   /** @summary Create buttons in the layout */
   createBrowserBtns() {
      let br = this.main().select(".jsroot_browser");
      if (br.empty()) return;
      let btns = br.append("div").classed("jsroot_browser_btns", true).classed("jsroot", true);
      btns.style('position',"absolute").style("left","7px").style("top","7px");
      if (browser.touches) btns.style('opacity','0.2'); // on touch devices should be always visible
      return btns;
   }

   /** @summary Remove browser buttons */
   removeBrowserBtns() {
      this.main().select(".jsroot_browser").select(".jsroot_browser_btns").remove();
   }

   /** @summary Set browser content */
   setBrowserContent(guiCode) {
      let main = d3_select("#" + this.gui_div + " .jsroot_browser");
      if (main.empty()) return;

      main.insert('div', ".jsroot_browser_btns").classed('jsroot_browser_area', true)
          .style('position',"absolute").style('left',0).style('top',0).style('bottom',0).style('width','250px')
          .style('overflow', 'hidden')
          .style('padding-left','5px')
          .style('display','flex').style('flex-direction', 'column')   /* use the flex model */
          .html("<p class='jsroot_browser_title'>title</p><div class='jsroot_browser_resize' style='display:none'>&#9727</div>" + guiCode);
   }

   /** @summary Check if there is browser content */
   hasContent() {
      let main = d3_select("#" + this.gui_div + " .jsroot_browser");
      if (main.empty()) return false;
      return !main.select(".jsroot_browser_area").empty();
   }

   /** @summary Delete content */
   deleteContent() {
      let main = d3_select("#" + this.gui_div + " .jsroot_browser");
      if (main.empty()) return;

      this.createStatusLine(0, "delete");

      this.toggleBrowserVisisbility(true);

      main.selectAll("*").remove();
      delete this.browser_visible;
      delete this.browser_kind;

      this.checkResize();
   }

   /** @summary Returns true when status line exists */
   hasStatus() {
      let main = d3_select("#"+this.gui_div+" .jsroot_browser");
      if (main.empty()) return false;

      let id = this.gui_div + "_status",
          line = d3_select("#"+id);

      return !line.empty();
   }

   /** @summary Set browser title text
     * @desc Title also used for dragging of the float browser */
   setBrowserTitle(title) {
      let main = d3_select("#" + this.gui_div + " .jsroot_browser");
      if (!main.empty())
         main.select(".jsroot_browser_title").text(title).style('cursor',this.browser_kind == 'flex' ? "move" : null);
   }

   /** @summary Toggle browser kind
     * @desc used together with browser buttons */
   toggleKind(browser_kind) {
      if (this.browser_visible!=='changing') {
         if (browser_kind === this.browser_kind) this.toggleBrowserVisisbility();
                                            else this.toggleBrowserKind(browser_kind);
      }
   }

   /** @summary Creates status line */
   createStatusLine(height, mode) {

      let main = d3_select("#"+this.gui_div+" .jsroot_browser");
      if (main.empty())
         return Promise.resolve('');

      let id = this.gui_div + "_status",
          line = d3_select("#"+id),
          is_visible = !line.empty();

      if (mode==="toggle") { mode = !is_visible; } else
      if (mode==="delete") { mode = false; height = 0; delete this.status_layout; } else
      if (mode===undefined) { mode = true; this.status_layout = "app"; }

      if (is_visible) {
         if (mode === true)
            return Promise.resolve(id);

         let hsepar = main.select(".jsroot_h_separator");

         hsepar.remove();
         line.remove();

         if (this.status_layout !== "app")
            delete this.status_layout;

         if (this.status_handler && (internals.showStatus === this.status_handler)) {
            delete internals.showStatus;
            delete this.status_handler;
         }

         this.adjustSeparators(null, 0, true);
         return Promise.resolve("");
      }

      if (mode === false)
         return Promise.resolve("");

      let left_pos = d3_select("#" + this.gui_div + "_drawing").style('left');

      main.insert("div",".jsroot_browser_area")
          .attr("id",id)
          .classed("jsroot_status_area", true)
          .style('position',"absolute").style('left',left_pos).style('height',"20px").style('bottom',0).style('right',0)
          .style('margin',0).style('border',0);

      let hsepar = main.insert("div",".jsroot_browser_area")
                       .classed("jsroot_separator", true).classed("jsroot_h_separator", true)
                       .style('position','absolute').style('left',left_pos).style('right',0).style('bottom','20px').style('height','5px');

      let drag_move = d3_drag().on("start", () => {
          this._hsepar_move = this._hsepar_position;
          hsepar.style('background-color', 'grey');
      }).on("drag", evnt => {
          this._hsepar_move -= evnt.dy; // hsepar is position from bottom
          this.adjustSeparators(null, Math.max(5, Math.round(this._hsepar_move)));
      }).on("end", () => {
          delete this._hsepar_move;
          hsepar.style('background-color', null);
          this.checkResize();
      });

      hsepar.call(drag_move);

      // need to get touches events handling in drag
      if (browser.touches && !main.on("touchmove"))
         main.on("touchmove", function() { });

      if (!height || (typeof height === 'string')) height = this.last_hsepar_height || 20;

      this.adjustSeparators(null, height, true);

      if (this.status_layout == "app")
         return Promise.resolve(id);

      this.status_layout = new GridDisplay(id, 'horizx4_1213');

      let frame_titles = ['object name','object title','mouse coordinates','object info'];
      for (let k = 0; k < 4; ++k)
         d3_select(this.status_layout.getGridFrame(k))
           .attr('title', frame_titles[k]).style('overflow','hidden')
           .append("label").attr("class","jsroot_status_label");

      internals.showStatus = this.status_handler = this.showStatus.bind(this);

      return Promise.resolve(id);
   }

   /** @summary Adjust separator positions */
   adjustSeparators(vsepar, hsepar, redraw, first_time) {

      if (!this.gui_div) return;

      let main = d3_select("#" + this.gui_div + " .jsroot_browser"), w = 5;

      if ((hsepar===null) && first_time && !main.select(".jsroot_h_separator").empty()) {
         // if separator set for the first time, check if status line present
         hsepar = main.select(".jsroot_h_separator").style('bottom');
         if ((typeof hsepar=='string') && (hsepar.length > 2) && (hsepar.indexOf('px') == hsepar.length-2))
            hsepar = hsepar.slice(0,hsepar.length-2);
         else
            hsepar = null;
      }

      if (hsepar!==null) {
         hsepar = parseInt(hsepar);
         let elem = main.select(".jsroot_h_separator"), hlimit = 0;

         if (!elem.empty()) {
            if (hsepar < 5) hsepar = 5;

            let maxh = main.node().clientHeight - w;
            if (maxh > 0) {
               if (hsepar < 0) hsepar += maxh;
               if (hsepar > maxh) hsepar = maxh;
            }

            this.last_hsepar_height = hsepar;
            elem.style('bottom', hsepar+'px').style('height', w+'px');
            d3_select("#" + this.gui_div + "_status").style('height', hsepar+'px');
            hlimit = (hsepar+w) + 'px';
         }

         this._hsepar_position = hsepar;

         d3_select("#" + this.gui_div + "_drawing").style('bottom',hlimit);
      }

      if (vsepar!==null) {
         vsepar = parseInt(vsepar);
         if (vsepar < 50) vsepar = 50;
         this._vsepar_position = vsepar;
         main.select(".jsroot_browser_area").style('width',(vsepar-5)+'px');
         d3_select("#" + this.gui_div + "_drawing").style('left',(vsepar+w)+'px');
         main.select(".jsroot_h_separator").style('left', (vsepar+w)+'px');
         d3_select("#" + this.gui_div + "_status").style('left',(vsepar+w)+'px');
         main.select(".jsroot_v_separator").style('left',vsepar+'px').style('width',w+"px");
      }

      if (redraw) this.checkResize();
   }

   /** @summary Show status information inside special fields of browser layout */
   showStatus(/*name, title, info, coordinates*/) {
      if (!this.status_layout) return;

      let maxh = 0;
      for (let n = 0; n < 4; ++n) {
         let lbl = this.status_layout.getGridFrame(n).querySelector('label');
         maxh = Math.max(maxh, lbl.clientHeight);
         lbl.innerHTML = arguments[n] || "";
      }

      if (!this.status_layout.first_check) {
         this.status_layout.first_check = true;
         if ((maxh > 5) && ((maxh > this.last_hsepar_height) || (maxh < this.last_hsepar_height+5)))
            this.adjustSeparators(null, maxh, true);
      }
   }

   /** @summary Toggle browser visibility */
   toggleBrowserVisisbility(fast_close) {
      if (!this.gui_div || (typeof this.browser_visible==='string')) return;

      let main = d3_select("#" + this.gui_div + " .jsroot_browser"),
          area = main.select('.jsroot_browser_area');

      if (area.empty()) return;

      let vsepar = main.select(".jsroot_v_separator"),
          drawing = d3_select("#" + this.gui_div + "_drawing"),
          tgt = area.property('last_left'),
          tgt_separ = area.property('last_vsepar'),
          tgt_drawing = area.property('last_drawing');

      if (!this.browser_visible) {
         if (fast_close) return;
         area.property('last_left', null).property('last_vsepar',null).property('last_drawing', null);
      } else {
         area.property('last_left', area.style('left'));
         if (!vsepar.empty()) {
            area.property('last_vsepar', vsepar.style('left'));
            area.property('last_drawing', drawing.style('left'));
         }

         tgt = (-area.node().clientWidth - 10) + "px";
         let mainw = main.node().clientWidth;

         if (vsepar.empty() && (area.node().offsetLeft > mainw/2))
            tgt = (mainw+10) + "px";

         tgt_separ = "-10px";
         tgt_drawing = "0px";
      }

      let visible_at_the_end  = !this.browser_visible, _duration = fast_close ? 0 : 700;

      this.browser_visible = 'changing';

      area.transition().style('left', tgt).duration(_duration).on("end", () => {
         if (fast_close) return;
         this.browser_visible = visible_at_the_end;
         if (visible_at_the_end) this.setButtonsPosition();
      });

      if (!visible_at_the_end)
         main.select(".jsroot_browser_btns").transition().style('left', '7px').style('top', '7px').duration(_duration);

      if (!vsepar.empty()) {
         vsepar.transition().style('left', tgt_separ).duration(_duration);
         drawing.transition().style('left', tgt_drawing).duration(_duration).on("end", this.checkResize.bind(this));
      }

      if (this.status_layout && (this.browser_kind == 'fix')) {
         main.select(".jsroot_h_separator").transition().style('left', tgt_drawing).duration(_duration);
         main.select(".jsroot_status_area").transition().style('left', tgt_drawing).duration(_duration);
      }
   }

   /** @summary Adjust browser size */
   adjustBrowserSize(onlycheckmax) {
      if (!this.gui_div || (this.browser_kind !== "float")) return;

      let main = d3_select("#" + this.gui_div + " .jsroot_browser");
      if (main.empty()) return;

      let area = main.select(".jsroot_browser_area"),
          cont = main.select(".jsroot_browser_hierarchy"),
          chld = d3_select(cont.node().firstChild);

      if (onlycheckmax) {
         if (area.node().parentNode.clientHeight - 10 < area.node().clientHeight)
            area.style('bottom', '0px').style('top','0px');
         return;
      }

      if (chld.empty()) return;
      let h1 = cont.node().clientHeight,
          h2 = chld.node().clientHeight;

      if ((h2!==undefined) && (h2 < h1*0.7)) area.style('bottom', '');
   }

   /** @summary Set buttons position */
   setButtonsPosition() {
      if (!this.gui_div) return;

      let main = d3_select("#"+this.gui_div+" .jsroot_browser"),
          btns = main.select(".jsroot_browser_btns"),
          top = 7, left = 7;

      if (btns.empty()) return;

      if (this.browser_visible) {
         let area = main.select(".jsroot_browser_area");

         top = area.node().offsetTop + 7;

         left = area.node().offsetLeft - main.node().offsetLeft + area.node().clientWidth - 27;
      }

      btns.style('left', left+'px').style('top', top+'px');
   }

   /** @summary Toggle browser kind */
   toggleBrowserKind(kind) {

      if (!this.gui_div)
         return Promise.resolve(null);

      if (!kind) {
         if (!this.browser_kind)
            return Promise.resolve(null);
         kind = (this.browser_kind === "float") ? "fix" : "float";
      }

      let main = d3_select("#"+this.gui_div+" .jsroot_browser"),
          area = main.select(".jsroot_browser_area");

      if (this.browser_kind === "float") {
          area.style('bottom', '0px')
              .style('top', '0px')
              .style('width','').style('height','')
              .classed('jsroot_float_browser', false);

           //jarea.resizable("destroy")
           //     .draggable("destroy");
      } else if (this.browser_kind === "fix") {
         main.select(".jsroot_v_separator").remove();
         area.style('left', '0px');
         d3_select("#"+this.gui_div+"_drawing").style('left','0px'); // reset size
         main.select(".jsroot_h_separator").style('left','0px');
         d3_select("#"+this.gui_div+"_status").style('left','0px'); // reset left
         this.checkResize();
      }

      this.browser_kind = kind;
      this.browser_visible = true;

      main.select(".jsroot_browser_resize").style("display", (kind === "float") ? null : "none");
      main.select(".jsroot_browser_title").style("cursor", (kind === "float") ? "move" : null);

      if (kind === "float") {
         area.style('bottom', '40px').classed('jsroot_float_browser', true);
        let drag_move = d3_drag().on("start", () => {
           let sl = area.style('left'), st = area.style('top');
           this._float_left = parseInt(sl.slice(0,sl.length-2));
           this._float_top = parseInt(st.slice(0,st.length-2));
           this._max_left = main.node().clientWidth - area.node().offsetWidth - 1;
           this._max_top = main.node().clientHeight - area.node().offsetHeight - 1;

        }).filter(evnt => {
            return main.select(".jsroot_browser_title").node() === evnt.target;
        }).on("drag", evnt => {
           this._float_left += evnt.dx;
           this._float_top += evnt.dy;

           area.style('left', Math.min(Math.max(0, this._float_left), this._max_left) + "px")
               .style('top', Math.min(Math.max(0, this._float_top), this._max_top) + "px");

           this.setButtonsPosition();
        });

        let drag_resize = d3_drag().on("start", () => {
           let sw = area.style('width');
           this._float_width = parseInt(sw.slice(0,sw.length-2));
           this._float_height = area.node().clientHeight;
           this._max_width = main.node().clientWidth - area.node().offsetLeft - 1;
           this._max_height = main.node().clientHeight - area.node().offsetTop - 1;

        }).on("drag", evnt => {
           this._float_width += evnt.dx;
           this._float_height += evnt.dy;

           area.style('width', Math.min(Math.max(100, this._float_width), this._max_width) + "px")
               .style('height', Math.min(Math.max(100, this._float_height), this._max_height) + "px");

           this.setButtonsPosition();
        });

        main.call(drag_move);
        main.select(".jsroot_browser_resize").call(drag_resize);

        this.adjustBrowserSize();

     } else {

        area.style('left', 0).style('top', 0).style('bottom', 0).style('height', null);

        let vsepar =
           main.append('div')
               .classed("jsroot_separator", true).classed('jsroot_v_separator', true)
               .style('position', 'absolute').style('top',0).style('bottom',0);

        let drag_move = d3_drag().on("start", () => {
            this._vsepar_move = this._vsepar_position;
            vsepar.style('background-color', 'grey');
        }).on("drag", evnt => {
            this._vsepar_move += evnt.dx;
            this.setButtonsPosition();
            this.adjustSeparators(Math.round(this._vsepar_move), null);
        }).on("end", () => {
            delete this._vsepar_move;
            vsepar.style('background-color', null);
            this.checkResize();
        });

        vsepar.call(drag_move);

        // need to get touches events handling in drag
        if (browser.touches && !main.on("touchmove"))
           main.on("touchmove", function() { });

        this.adjustSeparators(250, null, true, true);
     }

      this.setButtonsPosition();

      return Promise.resolve(this);
   }

} // class BrowserLayout

export { MDIDisplay, CustomDisplay, BatchDisplay, GridDisplay, FlexibleDisplay, BrowserLayout };
