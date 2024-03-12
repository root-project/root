import { isStr, clTF2, clTF3 } from '../core.mjs';
import * as jsroot_math from './math.mjs';

/** @summary Assign `evalPar` function for TF1 object
  * @private */

function proivdeEvalPar(obj, check_save) {
   obj.$math = jsroot_math;

   let _func = obj.fTitle, isformula = false, pprefix = '[';
   if (_func === 'gaus') _func = 'gaus(0)';
   if (isStr(obj.fFormula?.fFormula)) {
     if (obj.fFormula.fFormula.indexOf('[](double*x,double*p)') === 0) {
        isformula = true; pprefix = 'p[';
        _func = obj.fFormula.fFormula.slice(21);
     } else {
        _func = obj.fFormula.fFormula;
        pprefix = '[p';
     }

     if (obj.fFormula.fClingParameters && obj.fFormula.fParams) {
        obj.fFormula.fParams.forEach(pair => {
           const regex = new RegExp(`(\\[${pair.first}\\])`, 'g'),
               parvalue = obj.fFormula.fClingParameters[pair.second];
           _func = _func.replace(regex, (parvalue < 0) ? `(${parvalue})` : parvalue);
        });
      }
   }

   if (!_func)
      return !check_save || (obj.fSave?.length > 2);

   obj.formulas?.forEach(entry => {
      _func = _func.replaceAll(entry.fName, entry.fTitle);
   });

   _func = _func.replace(/\b(TMath::SinH)\b/g, 'Math.sinh')
                .replace(/\b(TMath::CosH)\b/g, 'Math.cosh')
                .replace(/\b(TMath::TanH)\b/g, 'Math.tanh')
                .replace(/\b(TMath::ASinH)\b/g, 'Math.asinh')
                .replace(/\b(TMath::ACosH)\b/g, 'Math.acosh')
                .replace(/\b(TMath::ATanH)\b/g, 'Math.atanh')
                .replace(/\b(TMath::ASin)\b/g, 'Math.asin')
                .replace(/\b(TMath::ACos)\b/g, 'Math.acos')
                .replace(/\b(TMath::Atan)\b/g, 'Math.atan')
                .replace(/\b(TMath::ATan2)\b/g, 'Math.atan2')
                .replace(/\b(sin|SIN|TMath::Sin)\b/g, 'Math.sin')
                .replace(/\b(cos|COS|TMath::Cos)\b/g, 'Math.cos')
                .replace(/\b(tan|TAN|TMath::Tan)\b/g, 'Math.tan')
                .replace(/\b(exp|EXP|TMath::Exp)\b/g, 'Math.exp')
                .replace(/\b(log|LOG|TMath::Log)\b/g, 'Math.log')
                .replace(/\b(log10|LOG10|TMath::Log10)\b/g, 'Math.log10')
                .replace(/\b(pow|POW|TMath::Power)\b/g, 'Math.pow')
                .replace(/\b(pi|PI)\b/g, 'Math.PI')
                .replace(/\b(abs|ABS|TMath::Abs)\b/g, 'Math.abs')
                .replace(/\bsqrt\(/g, 'Math.sqrt(')
                .replace(/\bxygaus\(/g, 'this.$math.gausxy(this, x, y, ')
                .replace(/\bgaus\(/g, 'this.$math.gaus(this, x, ')
                .replace(/\bgausn\(/g, 'this.$math.gausn(this, x, ')
                .replace(/\bexpo\(/g, 'this.$math.expo(this, x, ')
                .replace(/\blandau\(/g, 'this.$math.landau(this, x, ')
                .replace(/\blandaun\(/g, 'this.$math.landaun(this, x, ')
                .replace(/\b(TMath::|ROOT::Math::)/g, 'this.$math.');

   if (_func.match(/^pol[0-9]$/) && (parseInt(_func[3]) === obj.fNpar - 1)) {
      _func = '[0]';
      for (let k = 1; k < obj.fNpar; ++k)
         _func += ` + [${k}] * `+ ((k === 1) ? 'x' : `Math.pow(x,${k})`);
   }

   if (_func.match(/^chebyshev[0-9]$/) && (parseInt(_func[9]) === obj.fNpar - 1)) {
      _func = `this.$math.ChebyshevN(${obj.fNpar-1}, x, `;
      for (let k = 0; k < obj.fNpar; ++k)
         _func += (k === 0 ? '[' : ', ') + `[${k}]`;
      _func += '])';
   }

   for (let i = 0; i < obj.fNpar; ++i)
      _func = _func.replaceAll(pprefix + i + ']', `(${obj.GetParValue(i)})`);

   for (let n = 2; n < 10; ++n)
      _func = _func.replaceAll(`x^${n}`, `Math.pow(x,${n})`);

   if (isformula) {
      _func = _func.replace(/x\[0\]/g, 'x');
      if (obj._typename === clTF3) {
         _func = _func.replace(/x\[1\]/g, 'y');
         _func = _func.replace(/x\[2\]/g, 'z');
         obj.evalPar = new Function('x', 'y', 'z', _func).bind(obj);
      } else if (obj._typename === clTF2) {
         _func = _func.replace(/x\[1\]/g, 'y');
         obj.evalPar = new Function('x', 'y', _func).bind(obj);
      } else
         obj.evalPar = new Function('x', _func).bind(obj);
   } else if (obj._typename === clTF3)
      obj.evalPar = new Function('x', 'y', 'z', 'return ' + _func).bind(obj);
   else if (obj._typename === clTF2)
      obj.evalPar = new Function('x', 'y', 'return ' + _func).bind(obj);
   else
      obj.evalPar = new Function('x', 'return ' + _func).bind(obj);

   return true;
}


/** @summary Get interpolation in saved buffer
  * @desc Several checks must be done before function can be used
  * @private */
function _getTF1Save(func, x) {
   const np = func.fSave.length - 3,
         xmin = func.fSave[np + 1],
        xmax = func.fSave[np + 2],
        dx = (xmax - xmin) / np;
    if (x < xmin)
       return func.fSave[0];
    if (x > xmax)
       return func.fSave[np];

    const bin = Math.min(np - 1, Math.floor((x - xmin) / dx));
    let xlow = xmin + bin * dx,
        xup = xlow + dx,
        ylow = func.fSave[bin],
        yup = func.fSave[bin + 1];

    if (!Number.isFinite(ylow) && (bin < np - 1)) {
       xlow += dx; xup += dx;
       ylow = yup; yup = func.fSave[bin + 2];
    } else if (!Number.isFinite(yup) && (bin > 0)) {
       xup -= dx; xlow -= dx;
       yup = ylow; ylow = func.fSave[bin - 1];
    }

    return ((xup * ylow - xlow * yup) + x * (yup - ylow)) / dx;
}

/** @summary Provide TF1 value
  * @desc First try evaluate, if not possible - check saved buffer
  * @private */
function getTF1Value(func, x, skip_eval = undefined) {
   let y = 0, iserr = false;
   if (!func)
      return 0;

   if (!skip_eval && !func.evalPar) {
      try {
         if (!proivdeEvalPar(func))
            iserr = true;
      } catch {
         iserr = true;
      }
   }

   if (func.evalPar && !iserr) {
      try {
         y = func.evalPar(x);
         return y;
      } catch {
         y = 0;
      }
   }

   const np = func.fSave.length - 3;
   if ((np < 2) || (func.fSave[np + 1] === func.fSave[np + 2])) return 0;
   return _getTF1Save(func, x);
}

export { proivdeEvalPar, getTF1Value, _getTF1Save };
