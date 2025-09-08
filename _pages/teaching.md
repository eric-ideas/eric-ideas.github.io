---
layout: page
title: teaching
permalink: /teaching/
description: My teaching materials.
nav: true
nav_order: 3   # adjust to control position in navbar
horizontal: false
---

<!--
This page lists items from the /_teaching collection.
Make sure _config.yml includes:

collections:
  teaching:
    output: true
    permalink: /teaching/:name/

And put Markdown files in /_teaching/ with front matter like:

---
title: "Intermediate Microeconomics"
importance: 5
image: /assets/img/microeconomics.png
url: /teaching/intermediate-microeconomics/
---

-->

<div class="projects">

{% assign sorted_teaching = site.teaching | sort: "importance" %}
{% if page.horizontal %}
  <div class="container">
    <div class="row row-cols-1 row-cols-md-2">
      {% for project in sorted_teaching %}
        {% include projects_horizontal.liquid %}
      {% endfor %}
    </div>
  </div>
{% else %}
  <div class="row row-cols-1 row-cols-md-3">
    {% for project in sorted_teaching %}
      {% include projects.liquid %}
    {% endfor %}
  </div>
{% endif %}

</div>
