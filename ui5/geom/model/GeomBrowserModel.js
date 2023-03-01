sap.ui.define([
    'rootui5/browser/model/BrowserModel'
], function(BrowserModel) {

   'use strict';

   return BrowserModel.extend('rootui5.geom.model.GeomBrowserModel', {

      constructor: function() {
         BrowserModel.apply(this);
      },

      buildTreeNode(nodes, cache, indx, expand_lvl) {
         let tnode = cache[indx];
         if (tnode) return tnode;

         let node = nodes[indx];

         cache[indx] = tnode = { name: node.name, id: indx, color: node.color, material: node.material, node_visible: node.vis != 0 };

         if (expand_lvl > 0)
            tnode.expanded = true;

         if (node.chlds?.length) {
            tnode.childs = [];
            tnode.nchilds = node.chlds.length;
            for (let k = 0; k < tnode.nchilds; ++k)
               tnode.childs.push(this.buildTreeNode(nodes, cache, node.chlds[k], expand_lvl-1));
         } else {
            tnode.end_node = true; // TODO: no need for such flag ??
         }

         return tnode;
      },

      buildTree(nodes, expand_lvl) {
         if (expand_lvl === undefined)
            expand_lvl = 1;
         return this.buildTreeNode(nodes, [], 0, expand_lvl);
      }

   });

});
