sap.ui.define([
    "sap/base/Log",
    "sap/ui/model/json/JSONListBinding"
], function(Log, JSONListBinding) {
    "use strict";

    var hRootListBinding = JSONListBinding.extend("rootui5.eve7.model.BrowserListBinding", {

        // called by the TreeTable to know the amount of entries
        getLength: function() {
           // Log.warning("root.model.hListBinding#getLength()");
           return this.getModel().getLength();
        },

        // function is called by the TreeTable when requesting the data to display
        getNodes: function(iStartIndex, iLength, iThreshold) {

           var args = { begin: iStartIndex, end: iStartIndex + iLength, threshold: iThreshold },
               nodes = this.getModel().buildFlatNodes(args), aNodes = [];

           for (var i = args.begin; i < args.end; i++)
              aNodes.push(nodes[i] || null);

           console.log("root.model.hListBinding#getNodes(" + iStartIndex + ", " + iLength + ", " + iThreshold + ") res = " + aNodes.length);

           return aNodes;
        },

        getContextByIndex: function(iIndex) {
            Log.warning("root.model.hListBinding#getContextByIndex(" + iIndex + ")");
            return this.getModel().getContext(this.getPath() + "/" + iIndex);
        },

        findNode: function() {
            Log.warning("root.model.hListBinding#findNode()");
        },

        nodeHasChildren: function(oNode) {
           // Log.warning("root.model.hListBinding#nodeHasChildren(" + oNode.type + ")");
            return oNode.type === "folder";
        },

        isExpanded: function(iIndex) {
            var elem = this.getModel().getElementByIndex(iIndex);
            var res = elem ? !!elem.expanded : false;

            Log.warning("root.model.hListBinding#isExpanded(" + iIndex + ") res = " + res + "  iselem = " + (elem ? elem._name : "---"));

            return res;
        },

        expand: function(iIndex) {
            Log.warning("root.model.hListBinding#expand(" + iIndex + ")");
        },

        collapse: function(iIndex) {
            Log.warning("root.model.hListBinding#collapse(" + iIndex + ")");
        },

        collapseToLevel: function(lvl) {
           console.log('root.model.hListBinding#collapseToLevel', lvl);
        },

        expandToLevel: function(lvl) {
           console.log('root.model.hListBinding#expandToLevel', lvl);
        },

        // called by the TreeTable when a node is expanded/collapsed
        toggleIndex: function(iIndex) {
            Log.warning("root.model.hListBinding#toggleIndex(" + iIndex + ")");
            if (this.getModel().toggleNode(iIndex))
               this.checkUpdate(true);

            // QUESTION: why one should call checkUpdate?, should it be done automatically always?
        },

        getSelectedIndex: function() {
            Log.warning("root.model.hListBinding#getSelectedIndex(" + JSON.stringify(arguments) + ")");
        },

        isIndexSelectable: function() {
            Log.warning("root.model.hListBinding#isIndexSelectable(" + JSON.stringify(arguments) + ")");
        }

    });

    return hRootListBinding;

});