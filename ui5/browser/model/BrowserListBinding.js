sap.ui.define([
   "sap/ui/model/json/JSONListBinding"
], function(JSONListBinding) {
   "use strict";

   let hRootListBinding = JSONListBinding.extend("rootui5.browser.model.BrowserListBinding", {

      // called by the TreeTable to know the amount of entries
      getLength() {
         return this.getModel().getLength();
      },

      // function is called by the TreeTable when requesting the data to display
      getNodes(iStartIndex, iLength, iThreshold) {
         let args = { begin: iStartIndex, end: iStartIndex + iLength, threshold: iThreshold },
            nodes = this.getModel().buildFlatNodes(args),
            aNodes = [];

         for (let i = args.begin; i < args.end; i++)
            aNodes.push(nodes && nodes[i] ? nodes[i] : null);

         return aNodes;
      },

      getContextByIndex(iIndex) {
         return this.getModel().getContext(this.getPath() + "/" + iIndex);
      },

      // required by openui5 from versions ~1.100.0
      getNodeByIndex(indx) {
         return this.getModel().getNodeByIndex(indx);
      },

      findNode() {
      },

      nodeHasChildren(oNode) {
         return oNode.type === "folder";
      },

      isExpanded(iIndex) {
         let elem = this.getModel().getElementByIndex(iIndex);

         return elem?.expanded ?? false;
      },

      expand(iIndex) {
         if (this.getModel().toggleNode(iIndex, true))
            this.checkUpdate(true);
      },

      collapse(iIndex) {
         if (this.getModel().toggleNode(iIndex, false))
            this.checkUpdate(true);
      },

      collapseToLevel(lvl) {
      },

      expandToLevel(lvl) {
      },

      // called by the TreeTable when a node is expanded/collapsed
      toggleIndex(iIndex) {
      },

      getSelectedIndex() {
      },

      isIndexSelectable() {
      },

      // dummy for compatibility with newest 1.70.0 version
      attachSelectionChanged() {
      }

   });

   return hRootListBinding;

});
