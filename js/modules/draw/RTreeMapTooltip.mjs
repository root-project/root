class RTreeMapTooltip {

   static CONSTANTS = { DELAY: 0, OFFSET_X: 10, OFFSET_Y: -10, PADDING: 8, BORDER_RADIUS: 4 };

   constructor(painter)
   {
      this.painter = painter;
      this.tooltip = null;
      this.content = '';
      this.x = 0;
      this.y = 0;
   }

   cleanup() {
      if (this.tooltip !== null) {
         document.body.removeChild(this.tooltip);
         this.tooltip = null;
      }
   }

   createTooltip()
   {
      if (this.tooltip)
         return;

      this.tooltip = document.createElement('div');
      this.tooltip.style.cssText = `
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: ${RTreeMapTooltip.CONSTANTS.PADDING}px;
            border-radius: ${RTreeMapTooltip.CONSTANTS.BORDER_RADIUS}px;
            font-size: 12px;
            pointer-events: none;
            z-index: 10000;
            opacity: 0;
            transition: opacity 0.2s;
            max-width: 200px;
            word-wrap: break-word;
        `;
      document.body.appendChild(this.tooltip);
   }

   showTooltip()
   {
      if (!this.tooltip)
         this.createTooltip();

      this.tooltip.innerHTML = this.content;
      this.tooltip.style.left = (this.x + RTreeMapTooltip.CONSTANTS.OFFSET_X) + 'px';
      this.tooltip.style.top = (this.y + RTreeMapTooltip.CONSTANTS.OFFSET_Y) + 'px';
      this.tooltip.style.opacity = '1';
   }

   hideTooltip()
   {
      if (this.tooltip)
         this.tooltip.style.opacity = '0';
   }

   generateTooltipContent(node)
   {
      const isLeaf = node.fNChildren === 0;
      let content = (node.fName.length > 0) ? `<strong>${node.fName}</strong><br>` : '';

      content += `<i>${(isLeaf ? 'Column' : 'Field')}</i><br>`;
      content += `Size: ${this.painter.getDataStr(node.fSize)}<br>`;

      if (isLeaf && node.fType !== undefined)
         content += `Type: ${node.fType}<br>`;

      if (!isLeaf)
         content += `Children: ${node.fNChildren}<br>`;

      const obj = this.painter.getObject();
      if (obj.fNodes && obj.fNodes.length > 0) {
         const totalSize = obj.fNodes[0].fSize,
               percentage = ((node.fSize / totalSize) * 100).toFixed(2);
         content += `Disk Usage: ${percentage}%`;
      }

      return content;
   }

}

export { RTreeMapTooltip };
