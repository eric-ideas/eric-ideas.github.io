---
layout: page
title: Gallery
permalink: /gallery/
description: A growing collection of my gallery works.
nav: true
nav_order: 4   # adjust to control position in navbar
display_categories: [maths, economics, stats]  # optional filtering
horizontal: false
---

<!--
This page lists items from the /_gallery collection.
Make sure _config.yml includes:

collections:
  gallery:
    output: true
    permalink: /gallery/:name/

And put Markdown files in /_gallery/ with front matter like:

---
title: "Spectral Theorem Poster"
category: maths
importance: 5
image: /assets/img/spectral.png
url: /gallery/spectral-theorem/
---

-->

<div class="projects">

{% if site.enable_project_categories and page.display_categories %}
  {%- comment -%} Categorized gallery view {%- endcomment -%}
  {% for category in page.display_categories %}
    {% assign categorized_items = site.gallery | where_exp: "p", "p.categories contains category or p.category == category" %}
    {% assign sorted_gallery = categorized_items | sort: "importance" %}
    {% if sorted_gallery.size > 0 %}
      <h2 id="{{ category }}" class="category">{{ category | capitalize }}</h2>
      {% if page.horizontal %}
        <div class="container">
          <div class="row row-cols-1 row-cols-md-2">
            {% for project in sorted_gallery %}
              {% include projects_horizontal.liquid project=project %}
            {% endfor %}
          </div>
        </div>
      {% else %}
        <div class="row row-cols-1 row-cols-md-3">
          {% for project in sorted_gallery %}
            {% include projects.liquid project=project %}
          {% endfor %}
        </div>
      {% endif %}
    {% endif %}
  {% endfor %}

{% else %}
  {%- comment -%} Un-categorized gallery view {%- endcomment -%}
  {% assign sorted_gallery = site.gallery | sort: "importance" %}
  {% if page.horizontal %}
    <div class="container">
      <div class="row row-cols-1 row-cols-md-2">
        {% for project in sorted_gallery %}
          {% include projects_horizontal.liquid project=project %}
        {% endfor %}
      </div>
    </div>
  {% else %}
    <div class="row row-cols-1 row-cols-md-3">
      {% for project in sorted_gallery %}
        {% include projects.liquid project=project %}
      {% endfor %}
    </div>
  {% endif %}
{% endif %}
</div>
