{% extends "full.tpl" %}

{% block body %}
<body>
  <div tabindex="-1" id="notebook" class="border-box-sizing">
    <div class="container" id="notebook-container">

      <div id="root_banner">
        <a href="https://root.cern.ch" title="ROOT Data Analysis Framework">
          <img src="https://root.cern.ch/drupal/sites/default/files/images/root6-banner.jpg" alt="ROOTaaS Notebook"/>
        </a>
      </div>
      <div id="notebook_beta_message">
        <h2>Notebook Beta</h2>
      </div>

      {% include "basic.tpl" %}

    </div>
  </div>
</body>
{%- endblock body %}
