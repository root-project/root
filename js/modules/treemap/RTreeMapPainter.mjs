import {ObjectPainter, ensureTCanvas} from 'jsroot';
import RTreeMapTooltip from "./RTreeMapTooltip.mjs";

function computeFnv(str)
{
   const FNV_offset = 14695981039346656037n;
   const FNV_prime = 1099511628211n;
   let h = FNV_offset;
   for (let i = 0; i < str.length; ++i) {
      const octet = BigInt(str.charCodeAt(i) & 0xFF);
      h = h ^ octet;
      h = h * FNV_prime;
   }
   return h;
}

class RTreeMapPainter extends ObjectPainter {
   static CONSTANTS = {
      STROKE_WIDTH : 0.15,
      STROKE_COLOR: "black",

      COLOR_HOVER_BOOST: 10,

      DATA_UNITS: [ "B", "KB", "MB", "GB", "TB", "PB", "EB" ],

      MAIN_TREEMAP: {LEFT: 0.025, BOTTOM: 0.05, RIGHT: 0.825, TOP: 0.9},

      LEGEND: {
         START_Y: 0.835,
         ITEM_HEIGHT: 0.05,
         BOX_WIDTH: 0.05,
         TEXT_OFFSET_X: 0.01,
         TEXT_OFFSET_Y: 0.01,
         TEXT_LINE_SPACING: 0.015,
         MAX_ITEMS: 10
      },

      TEXT: {SIZE_VW: 0.6, MIN_RECT_WIDTH: 0.025, MIN_RECT_HEIGHT: 0.05, PADDING: 10, LEAF_OFFSET_Y: 0.015},

      INDENT: 0.005,
      LEGEND_INDENT_MULTIPLIER: 4
   };

   constructor(dom, obj, opt)
   {
      super(dom, obj, opt);
      this.tooltip = new RTreeMapTooltip(this);
      this.rootIndex = 0;
      this.parentIndices = [];
      const rframe = this.getPadPainter().getLayerSvg('primitives_layer').select('g.root_frame path');
      if(rframe !== null) rframe.attr("visibility", "hidden");
   }


   cleanup() {
      this.getPadPainter().getLayerSvg('primitives_layer').select('g.root_frame path').attr("visibility", "visible");
      this.tooltip.cleanup();
      super.cleanup();
   }

   appendRect(begin, end, color, strokeColor = RTreeMapPainter.CONSTANTS.STROKE_COLOR,
              strokeWidth = RTreeMapPainter.CONSTANTS.STROKE_WIDTH, node = null)
   {
      const rect = this.getG()
                      .append('rect')
                      .attr('x', this.axisToSvg('x', begin.x, this.isndc))
                      .attr('y', this.axisToSvg('y', begin.y, this.isndc))
                      .attr('width', `${Math.abs(end.x - begin.x) * 100}%`)
                      .attr('height', `${Math.abs(end.y - begin.y) * 100}%`)
                      .attr('fill', color)
                      .attr('stroke', strokeColor)
                      .attr('stroke-width', strokeWidth)
                      .attr("pointer-events", "fill");

      if (node) {
         rect.datum(node);
         this.attachPointerEventsTreeMap(rect, node);
      }
      return rect;
   }

   appendText(content, pos, size, color, anchor = "start")
   {
      return this.getG()
         .append("text")
         .attr("x", this.axisToSvg("x", pos.x, this.isndc))
         .attr("y", this.axisToSvg("y", pos.y, this.isndc))
         .attr("font-size", `${size}vw`)
         .attr("fill", color)
         .attr("text-anchor", anchor)
         .attr("pointer-events", "none")
         .text(content)
   }

   getRgbList(rgbStr) { return rgbStr.slice(4, -1).split(",").map((x) => parseInt(x)); }

   toRgbStr(rgbList) { return `rgb(${rgbList.join()})`; }

   attachPointerEventsTreeMap(element, node)
   {
      const original_color = element.attr("fill");
      const hovered_color =
         this.toRgbStr(this.getRgbList(original_color)
                          .map((color) => Math.min(color + RTreeMapPainter.CONSTANTS.COLOR_HOVER_BOOST, 255)));
      let currentMouseX = 0;
      let currentMouseY = 0;
      const mouseEnter = () => {
         element.attr('fill', hovered_color);
         this.tooltip.content = this.tooltip.generateTooltipContent(node);
         this.tooltip.x = currentMouseX;
         this.tooltip.y = currentMouseY;
      };
      const mouseLeave = () => {
         element.attr('fill', original_color);
         this.tooltip.hideTooltip();
      };
      const mouseMove = (event) => {
         this.tooltip.x = event.pageX;
         this.tooltip.y = event.pageY;
         this.tooltip.showTooltip();
      };
      const click = () => {
         const obj = this.getObject();
         const nodeIndex = obj.fNodes.findIndex((elem) => elem === node);
         if(nodeIndex === this.rootIndex) {
            this.rootIndex = this.parentIndices[nodeIndex];
         }
         else {
            let parentIndex = nodeIndex;
            while (this.parentIndices[parentIndex] !== this.rootIndex) {
               parentIndex = this.parentIndices[parentIndex];
            }
            this.rootIndex = parentIndex;
            if(obj.fNodes[parentIndex].fNChildren === 0) this.rootIndex = this.parentIndices[nodeIndex];
         }
         this.redraw();
      };
      this.attachPointerEvents(element, {
         'mouseenter' : mouseEnter,
         'mouseleave' : mouseLeave,
         'mousemove' : mouseMove,
         'click' : click
      });
   }

   attachPointerEventsLegend(element, type)
   {
      const rects = this.getG().selectAll("rect");
      const mouseEnter =
         () => { rects.filter((node) => node !== undefined && node.fType !== type).attr('opacity', '0.5'); };
      const mouseLeave = () => { rects.attr("opacity", "1"); };
      this.attachPointerEvents(
         element, {'mouseenter' : mouseEnter, 'mouseleave' : mouseLeave, 'mousemove' : () => {}, 'click' : () => {}});
   }

   attachPointerEvents(element, events)
   {
      for (const [key, value] of Object.entries(events)) {
         element.on(key, value)
      }
   }

   computeColor(n)
   {
      const hash = Number(computeFnv(String(n)) & 0xFFFFFFFFn)
      const r = (hash >> 16) & 0xFF;
      const g = (hash >> 8) & 0xFF;
      const b = hash & 0xFF;
      return this.toRgbStr([ r, g, b ]);
   }

   getDataStr(bytes)
   {
      const units = RTreeMapPainter.CONSTANTS.DATA_UNITS;
      const order = Math.floor(Math.log10(bytes) / 3);
      const finalSize = bytes / Math.pow(1000, order);
      return `${finalSize.toFixed(2)}${units[order]}`;
   }

   computeWorstRatio(row, width, height, totalSize, horizontalRows)
   {
      if (row.length === 0)
         return 0;

      const sumRow = row.reduce((sum, child) => sum + child.fSize, 0);
      if (sumRow === 0)
         return 0;

      let worstRatio = 0;
      for (const child of row) {
         const ratio = horizontalRows ? (child.fSize * width * totalSize) / (sumRow * sumRow * height)
                                      : (child.fSize * height * totalSize) / (sumRow * sumRow * width);
         const aspectRatio = Math.max(ratio, 1 / ratio);
         if (aspectRatio > worstRatio)
            worstRatio = aspectRatio;
      }
      return worstRatio;
   }

   squarifyChildren(children, rect, horizontalRows, totalSize)
   {
      const width = rect.topRight.x - rect.bottomLeft.x;
      const height = rect.topRight.y - rect.bottomLeft.y;
      let remaining = [...children ].sort((a, b) => b.fSize - a.fSize);
      const result = [];
      let remainingBegin = {...rect.bottomLeft};

      while (remaining.length > 0) {
         const row = [];
         let currentWorstRatio = Infinity;
         const remainingWidth = rect.topRight.x - remainingBegin.x;
         const remainingHeight = rect.topRight.y - remainingBegin.y;

         if (remainingWidth <= 0 || remainingHeight <= 0)
            break;

         while (remaining.length > 0) {
            row.push(remaining.shift());
            const newWorstRatio =
               this.computeWorstRatio(row, remainingWidth, remainingHeight, totalSize, horizontalRows);
            if (newWorstRatio > currentWorstRatio) {
               remaining.unshift(row.pop());
               break;
            }
            currentWorstRatio = newWorstRatio;
         }

         const sumRow = row.reduce((sum, child) => sum + child.fSize, 0);
         if (sumRow === 0)
            continue;

         const dimension = horizontalRows ? (sumRow / totalSize * height) : (sumRow / totalSize * width);
         let position = 0;

         for (const child of row) {
            const childDimension = child.fSize / sumRow * (horizontalRows ? width : height);
            const childBegin = horizontalRows ? {x : remainingBegin.x + position, y : remainingBegin.y}
                                              : {x : remainingBegin.x, y : remainingBegin.y + position};
            const childEnd = horizontalRows
                                ? {x : remainingBegin.x + position + childDimension, y : remainingBegin.y + dimension}
                                : {x : remainingBegin.x + dimension, y : remainingBegin.y + position + childDimension};

            result.push({node : child, rect : {bottomLeft : childBegin, topRight : childEnd}});
            position += childDimension;
         }

         if (horizontalRows)
            remainingBegin.y += dimension;
         else
            remainingBegin.x += dimension;
      }
      return result;
   }

   drawLegend()
   {
      const obj = this.getObject();
      const diskMap = {};

      let stack = [ this.rootIndex ];
      while (stack.length > 0) {
         const node = obj.fNodes[stack.pop()];
         if (node.fNChildren === 0)
            diskMap[node.fType] = (diskMap[node.fType] || 0) + node.fSize;
         stack = stack.concat(Array.from({length : node.fNChildren}, (_, a) => a + node.fChildrenIdx));
      }

      const diskEntries = Object.entries(diskMap)
                             .sort((a, b) => b[1] - a[1])
                             .slice(0, RTreeMapPainter.CONSTANTS.LEGEND.MAX_ITEMS)
                             .filter(([, size ]) => size > 0)

      const legend = RTreeMapPainter.CONSTANTS.LEGEND;

      diskEntries.forEach(([ typeName, size ], index) => {
         const posY = legend.START_Y - index * legend.ITEM_HEIGHT;
         const posX = legend.START_Y + legend.ITEM_HEIGHT + legend.TEXT_OFFSET_X;
         const textSize = RTreeMapPainter.CONSTANTS.TEXT.SIZE_VW;

         const rect = this.appendRect({x : legend.START_Y, y : posY},
                                      {x : legend.START_Y + legend.ITEM_HEIGHT, y : posY - legend.ITEM_HEIGHT},
                                      this.computeColor(typeName));
         this.attachPointerEventsLegend(rect, typeName);

         const diskOccupPercent = `${(size / obj.fNodes[this.rootIndex].fSize * 100).toFixed(2)}%`;
         const diskOccup = `(${this.getDataStr(size)} / ${this.getDataStr(obj.fNodes[this.rootIndex].fSize)})`;

         [typeName, diskOccup, diskOccupPercent].forEach(
            (content, i) =>
               this.appendText(content, {x : posX, y : posY - legend.TEXT_OFFSET_Y - legend.TEXT_LINE_SPACING * (i)},
                               textSize, "black"));
      });
   }

   trimText(textElement, rect)
   {
      const nodeElem = textElement.node();
      let textContent = nodeElem.textContent;
      const availablePx = Math.abs(this.axisToSvg('x', rect.topRight.x, this.isndc) -
                                   this.axisToSvg('x', rect.bottomLeft.x, this.isndc)) -
                          RTreeMapPainter.CONSTANTS.TEXT.PADDING;

      while (nodeElem.getComputedTextLength() > availablePx && textContent.length > 0) {
         textContent = textContent.slice(0, -1);
         nodeElem.textContent = textContent + '…';
      }
      return textContent;
   }

   drawTreeMap(node, rect, depth = 0)
   {
      const isLeaf = node.fNChildren === 0;
      const color = isLeaf ? this.computeColor(node.fType) : `rgb(100,100,100)`;
      this.appendRect({x : rect.bottomLeft.x, y : rect.topRight.y}, {x : rect.topRight.x, y : rect.bottomLeft.y}, color,
                      RTreeMapPainter.CONSTANTS.STROKE_COLOR, RTreeMapPainter.CONSTANTS.STROKE_WIDTH, node);

      const rectWidth = rect.topRight.x - rect.bottomLeft.x;
      const rectHeight = rect.topRight.y - rect.bottomLeft.y;
      const labelBase = `${node.fName} (${this.getDataStr(node.fSize)})`;

      const textConstants = RTreeMapPainter.CONSTANTS.TEXT;
      const textSize = (rectWidth <= textConstants.MIN_RECT_WIDTH || rectHeight <= textConstants.MIN_RECT_HEIGHT)
                          ? 0
                          : textConstants.SIZE_VW;

      if (textSize > 0) {
         const textElement = this.appendText(labelBase, {
            x : rect.bottomLeft.x + (isLeaf ? rectWidth / 2 : RTreeMapPainter.CONSTANTS.INDENT),
            y : isLeaf ? (rect.bottomLeft.y + rect.topRight.y) / 2 : (rect.topRight.y - textConstants.LEAF_OFFSET_Y)
         },
                                             textSize, "white", isLeaf ? "middle" : "start");
         textElement.textContent = this.trimText(textElement, rect);
      }

      if (!isLeaf && node.fNChildren > 0) {
         const obj = this.getObject();
         const children = obj.fNodes.slice(node.fChildrenIdx, node.fChildrenIdx + node.fNChildren);
         const totalSize = children.reduce((sum, child) => sum + child.fSize, 0);

         if (totalSize > 0) {
            const indent = RTreeMapPainter.CONSTANTS.INDENT;
            const innerRect = {
               bottomLeft : {x : rect.bottomLeft.x + indent, y : rect.bottomLeft.y + indent},
               topRight : {
                  x : rect.topRight.x - indent,
                  y : rect.topRight.y - indent * RTreeMapPainter.CONSTANTS.LEGEND_INDENT_MULTIPLIER
               }
            };

            const width = innerRect.topRight.x - innerRect.bottomLeft.x;
            const height = innerRect.topRight.y - innerRect.bottomLeft.y;
            const horizontalRows = width > height;

            const rects = this.squarifyChildren(children, innerRect, horizontalRows, totalSize);
            rects.forEach(
               ({node : childNode, rect : childRect}) => { this.drawTreeMap(childNode, childRect, depth + 1); });
         }
      }
   }

   createParentIndices()
   {
      const obj = this.getObject();
      this.parentIndices = new Array(obj.fNodes.length).fill(0);
      obj.fNodes.forEach((node, index) => {
         for (let i = node.fChildrenIdx; i < node.fChildrenIdx + node.fNChildren; i++) {
            this.parentIndices[i] = index;
         }
      })
   }

   getDirectory()
   {
      const obj = this.getObject();
      let result = "";
      let currentIndex = this.rootIndex;
      while(currentIndex !== 0) {
         result = obj.fNodes[currentIndex].fName + "/" + result;
         currentIndex = this.parentIndices[currentIndex];
      }
      return result;
   }

   redraw()
   {
      const obj = this.getObject();
      this.createG();
      this.isndc = true;

      if (obj.fNodes && obj.fNodes.length > 0) {
         this.createParentIndices()
         const mainArea = RTreeMapPainter.CONSTANTS.MAIN_TREEMAP;
         this.drawTreeMap(
            obj.fNodes[this.rootIndex],
            {bottomLeft : {x : mainArea.LEFT, y : mainArea.BOTTOM}, topRight : {x : mainArea.RIGHT, y : mainArea.TOP}});
         this.drawLegend();
         this.appendText(this.getDirectory(), {x : RTreeMapPainter.CONSTANTS.MAIN_TREEMAP.LEFT, y : RTreeMapPainter.CONSTANTS.MAIN_TREEMAP.TOP+0.01},
             RTreeMapPainter.CONSTANTS.TEXT.SIZE_VW, "black");
      }
      return this;
   }

   static async draw(dom, obj, opt)
   {
      const painter = new RTreeMapPainter(dom, obj, opt);
      return ensureTCanvas(painter, false).then(() => painter.redraw());
   }
}
export {RTreeMapPainter};
