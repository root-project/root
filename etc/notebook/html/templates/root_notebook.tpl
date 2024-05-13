{% extends "full.tpl" %}

{% block body %}
<body>
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">

      <div id="root_banner">
        <a href="https://root.cern" title="ROOT Data Analysis Framework">
          <img src="https://root.cern/assets/images/splash_optimized.jpg" alt="ROOT Notebook"/>
        </a>
      </div>

      {% include "basic.tpl" %}

    </div>
  </div>
</body>
{%- endblock body %}
