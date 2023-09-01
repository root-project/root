sap.ui.define([
    "sap/ui/model/json/JSONListBinding"
], function(JSONListBinding) {
    "use strict";

    let hRootListBinding = JSONListBinding.extend("rootui5.browser.model.BrowserListBinding", {

        // called by the TreeTable to know the amount of entries
        getLength: function() {
           return this.getModel().getLength();
        },

        // function is called by the TreeTable when requesting the data to display
        getNodes: function(iStartIndex, iLength, iThreshold) {

           let args = { begin: iStartIndex, end: iStartIndex + iLength, threshold: iThreshold },
               nodes = this.getModel().buildFlatNodes(args),
               aNodes = [];

           for (let i = args.begin; i < args.end; i++)
              aNodes.push(nodes && nodes[i] ? nodes[i] : null);

           return aNodes;
        },

        getContextByIndex: function(iIndex) {
           return this.getModel().getContext(this.getPath() + "/" + iIndex);
        },

        findNode: function() {
        },

        nodeHasChildren: function(oNode) {
           return oNode.type === "folder";
        },

        isExpanded: function(iIndex) {
            let elem = this.getModel().getElementByIndex(iIndex);

            return elem ? !!elem.expanded : false;
        },

        expand: function(iIndex) {
           if (this.getModel().toggleNode(iIndex, true))
              this.checkUpdate(true);
        },

        collapse: function(iIndex) {
           if (this.getModel().toggleNode(iIndex, false))
              this.checkUpdate(true);
        },

        collapseToLevel: function(lvl) {
           // console.log('root.model.hListBinding#collapseToLevel', lvl);
        },

        expandToLevel: function(lvl) {
           // console.log('root.model.hListBinding#expandToLevel', lvl);
        },

        // called by the TreeTable when a node is expanded/collapsed
        toggleIndex: function(iIndex) {
            // was used before
        },

        getSelectedIndex: function() {
        },

        isIndexSelectable: function() {
        },

        attachSelectionChanged: function() {
           // dummy for compatibility with newest 1.70.0 version
        }

    });

    return hRootListBinding;

});
