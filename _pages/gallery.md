---
layout: page
title: gallery
permalink: /gallery/
description: A growing collection of your cool projects.
nav: true
nav_order: 3
display_categories: [maths, economics, stats]  # categories to show (exact matches)
horizontal: false                               # set true to use horizontal cards
---

<!--
This page lists items from the /_gallery collection.

Requirements:
1) _config.yml
   enable_project_categories: true
   collections:
     gallery:
       output: true
       permalink: /gallery/:name/

2) Items in /_gallery/ with front matter like:
   ---
   title: "My Project"
   category: maths            # or: categories: [economics, stats]
   importance: 10             # lower/greater numbers sort earlier depending on reverse
   image: /assets/img/cover.png
   url: /gallery/my-project/  # optional if you generate detail pages
   ---

3) Includes expected:
   _includes/projects.liquid
   _includes/projects_horizontal.liquid
-->

<div class="projects">

{% if site.enable_project_categories and page.display_categories %}
  {%- comment -%} Categorized view {%- endcomment -%}
  {% for category in page.display_categories %}
    {% assign categorized_items = site.gallery | where_exp: "p", "p.categories contains category or p.category == category" %}
    {% assign sorted_projects = categorized_items | sort: "importance" %}
    {% if sorted_projects.size > 0 %}
      <h2 id="{{ category }}" class="category">{{ category | capitalize }}</h2>

      {% if page.horizontal %}
        <div class="container">
          <div class="row row-cols-1 row-cols-md-2">
            {% for project in sorted_projects %}
              {% include projects_horizontal.liquid %}
            {% endfor %}
          </div>
        </div>
      {% else %}
        <div class="row row-cols-1 row-cols-md-3">
          {% for project in sorted_projects %}
            {% include projects.liquid %}
          {% endfor %}
        </div>
      {% endif %}
    {% endif %}
  {% endfor %}

{% else %}

  {%- comment -%} Un-categorized view: show everything {%- endcomment -%}
  {% assign sorted_projects = site.gallery | sort: "importance" %}

  {% if page.horizontal %}
    <div class="container">
      <div class="row row-cols-1 row-cols-md-2">
        {% for project in sorted_projects %}
          {% include projects_horizontal.liquid %}
        {% endfor %}
      </div>
    </div>
  {% else %}
    <div class="row row-cols-1 row-cols-md-3">
      {% for project in sorted_projects %}
        {% include projects.liquid %}
      {% endfor %}
    </div>
  {% endif %}

{% endif %}
</div>
