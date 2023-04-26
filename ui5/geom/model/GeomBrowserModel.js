sap.ui.define([
    'rootui5/browser/model/BrowserModel'
], function(BrowserModel) {

   'use strict';

   /** @summary Create node copying item attributes from RGeoItem or RGeomNodeBase */

   function createTreeNode(id, item, node) {
      if (node === item)
         return node;
      if (!node)
         node = { id, name: item.name };
      node.color = item.color;
      node.material = item.material;
      node.visible = item.vis > 0;
      return node;
   };

   return BrowserModel.extend('rootui5.geom.model.GeomBrowserModel', {

      constructor: function() {
         BrowserModel.apply(this);
      },

      /** @summary Build tree model based on the nodes */
      buildTree(nodes, expand_lvl) {
         if (expand_lvl === undefined)
            expand_lvl = 1;

         let cache = [];
         this.logicalNodes = cache;

         function buildTreeNode(id, expand_lvl) {
            let tnode = cache[id];
            if (tnode) return tnode;

            let node = nodes[id];
            cache[id] = tnode = createTreeNode(id, node);

            if (expand_lvl > 0)
               tnode.expanded = true;

            if (node.chlds?.length) {
               tnode.childs = [];
               tnode.nchilds = node.chlds.length;
               for (let k = 0; k < tnode.nchilds; ++k)
                  tnode.childs.push(buildTreeNode(node.chlds[k], expand_lvl-1));
            } else {
               tnode.end_node = true; // TODO: no need for such flag ??
            }

            return tnode;
         };

         return buildTreeNode(0, expand_lvl);
      },

      /** @summary Provide logical node for the id.
        * @desc Either use existing one from the full model or create one based on specified item description  */
      provideLogicalNode(item) {
         let id = item.id;

         if (!this.logicalNodes)
            this.logicalNodes = [];
         let node = this.logicalNodes[id];

         if (node)
            return createTreeNode(id, item, node);

         node = this.logicalNodes[id] = createTreeNode(id, item);

         return node;
      },

      /** @summary Append additional attributes to node description in the table */
      addNodeAttributes(node, item) {
         // here attributes common for TGeoNode/TGeoVolume collected, single object as on server
         node._node = this.provideLogicalNode(item);

         if (item.pvis !== undefined) {
            // this is RGeoItem with physical node settings
            node.pvisible = item.pvis != 0;
            node.top = item.top;
         } else {
            // this is full model
            node.pvisible = this.hController?.getPhysVisibilityEntry(node.path)?.visible ?? node._node.visible;
            node.top = this.hController?.getPhysTopNode(node.path) ?? false;
         }
      }

   });

});
