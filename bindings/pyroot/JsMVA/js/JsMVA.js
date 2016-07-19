/**
 * Created by Attila Bagoly <battila93@gmail.com> on 5/14/16.
 */

(function(factory){

    var JSROOT_source_dir = "https://root.cern.ch/js/notebook/scripts/";

    var url = "";
    if (requirejs.s.contexts.hasOwnProperty("_")) {
        url = requirejs.s.contexts._.config.paths["JsMVA"].replace("JsMVA.min","");
    }
    if ((console!==undefined) && (typeof console.log == 'function')) {
        if (url!=""){
            console.log("JsMVA source dir:" + url.substring(0, url.length-1));
        } else {
            console.log("JsMVA source dir can't be resolved, requireJS doesn't have context '_', this will be a problem!");
        }
    }

    require.config({
        paths: {
            'JsRootCore': JSROOT_source_dir+'JSRootCore',
            'nn': url+'NeuralNetwork.min',
            'dtree': url+'DecisionTree.min',
            'IChart': url+'IChart'
        }
    });

    define(['JsRootCore'], function(jsroot){
        return factory({}, jsroot);
    });

}(function(JsMVA, JSROOT){

    JsMVA.drawTH2 = function(divid, dat_json){
        var obj = JSROOT.parse(dat_json);
        JSROOT.draw(divid, obj, "colz;PAL50;text");
    };

    JsMVA.draw = function(divid, dat_json){
        var obj = JSROOT.parse(dat_json);
        JSROOT.draw(divid, obj);
    };

    JsMVA.drawNeuralNetwork = function(divid, dat_json){
        var obj = JSON.parse(dat_json);
        require(['nn'], function(nn){
            nn.draw(divid, obj);
        });
    };

    JsMVA.drawDecisionTree = function(divid, dat_json){
        require(['dtree'], function(dtree){
            var obj = JSON.parse(dat_json);
            dtree.draw(divid, obj);
        });
    };

    JsMVA.drawTrainingTestingErrors = function(divid, dat_json){
        var obj = JSROOT.parse(dat_json);
        JSROOT.draw(divid, obj, "ALP");
    };

    JsMVA.updateTrainingTestingErrors = function(divid, dat_json){
        var obj = JSROOT.parse(dat_json);
        JSROOT.redraw(divid, obj, "ALP");
    };

    return JsMVA;
}));
