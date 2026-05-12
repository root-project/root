\defgroup Python Python Interface
\brief Python bindings and utilities for ROOT.

# Python Interface

ROOT is a C++ framework that exposes its full API to Python through dynamic bindings generated at runtime via [cppyy](https://cppyy.readthedocs.io/). Every ROOT class is available  in Python automatically without manual wrapping.
Additional pythonizations are layered on top of some classes to make the experience feel natively Pythonic.

## Installation

## Installation

\htmlonly
<div class="install-tabs">
  <div class="tab-buttons">
    <button class="tab-btn active" onclick="switchTab(this, 'pip')">pip</button>
    <button class="tab-btn" onclick="switchTab(this, 'conda')">conda</button>
  </div>
  <div id="pip" class="tab-panel active">
    <pre><code>pip install root</code></pre>
  </div>
  <div id="conda" class="tab-panel" style="display:none;">
    <pre><code>conda install -c conda-forge root</code></pre>
  </div>
</div>

<style>
.install-tabs {
  border: 1px solid #ccc;
  border-radius: 6px;
  overflow: hidden;
  max-width: 420px;
  font-family: monospace;
}
.tab-buttons {
  display: flex;
  background: #f0f0f0;
  border-bottom: 1px solid #ccc;
}
.tab-btn {
  padding: 6px 18px;
  border: none;
  background: none;
  cursor: pointer;
  font-size: 13px;
  border-bottom: 2px solid transparent;
}
.tab-btn.active {
  border-bottom: 2px solid #1a73e8;
  font-weight: 600;
  color: #1a73e8;
}
.tab-panel pre {
  margin: 0;
  padding: 12px 16px;
  background: #fff;
}
</style>

<script>
function switchTab(btn, id) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.style.display = 'none');
  btn.classList.add('active');
  document.getElementById(id).style.display = 'block';
}
</script>
\endhtmlonly


**With pip:**
~~~{.sh}
pip install root
~~~

**With conda:**
~~~{.sh}
conda install -c conda-forge root
~~~

## Quickstart

~~~{.sh}
import ROOT

df = ROOT.RDataFrame("tree", "file.root")
df.Histo1D("px").Draw()
~~~

## Topics

| Module | Description |
|---|---|
| @ref Py_RDataFrame "RDataFrame" | Analyse ROOT data with a high-level Python API |
| @ref Py_IO "I/O" | Reading and writing ROOT files, TTree and RNTuple from Python |
| @ref Py_UHI "UHI - Unified Histogram Interface" | Using ROOT histograms in Python |
| @ref Py_ML "ML / RDataLoader" | Feed ROOT data directly into models for training |
| @ref Py_Interop "Interoperability" | Using ROOT alongside other Python packages |
| @ref Pythonizations "Pythonizations" | Advanced - how ROOT adapts C++ classes for idiomatic Python use |
