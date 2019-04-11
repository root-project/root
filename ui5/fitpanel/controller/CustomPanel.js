jQuery.sap.declare("localapp.controller.CustomPanel");

(function () {
	"use strict";

	/** @lends sap.m.sample.P13nDialogWithCustomPanel.CustomPanel */
	sap.m.P13nPanel.extend("localapp.controller.CustomPanel", {
		constructor: function (sId, mSettings) {
			sap.m.P13nPanel.apply(this, arguments);
		},
		metadata: {
			library: "sap.m",
			aggregations: {

				/**
				 * Control embedded into CustomPanel
				 */
				content: {
					type: "sap.m.RadioButtonGroup",
					multiple: false,
					singularName: "content"
				}
			}
		},
		renderer: function (oRm, oControl) {
			if (!oControl.getVisible()) {
				return;
			}
			oRm.renderControl(oControl.getContent());
		},
	});
})();