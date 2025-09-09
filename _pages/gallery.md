---
layout: page
title: gallery
permalink: /gallery/
description: A growing collection of my notes and vibe-coded visualization.
nav: true
nav_order: 4
display_categories: [notes, maths, economics, stats]
horizontal: false
---

<!-- pages/gallery.md -->
<div class="projects">
{% if site.enable_project_categories and page.display_categories %}
  {% comment %} Display categorized gallery {% endcomment %}
  {% for category in page.display_categories %}
    {% assign categorized_gallery = site.gallery | where: "category", category %}
    {% if categorized_gallery.size > 0 %}
      <a id="{{ category }}" href=".#{{ category }}">
        <h2 class="category">{{ category }}</h2>
      </a>
      {% assign sorted_gallery = categorized_gallery | sort: "importance" %}
      {% comment %} Generate cards for each gallery item {% endcomment %}
      {% if page.horizontal %}
      <div class="container">
        <div class="row row-cols-1 row-cols-md-2">
          {% for project in sorted_gallery %}
            {% include projects_horizontal.liquid %}
          {% endfor %}
        </div>
      </div>
      {% else %}
      <div class="row row-cols-1 row-cols-md-3">
        {% for project in sorted_gallery %}
          {% include projects.liquid %}
        {% endfor %}
      </div>
      {% endif %}
    {% endif %}
  {% endfor %}

{% else %}

  {% comment %} Display gallery without categories {% endcomment %}
  {% assign sorted_gallery = site.gallery | sort: "importance" %}
  {% if sorted_gallery.size > 0 %}
    {% if page.horizontal %}
    <div class="container">
      <div class="row row-cols-1 row-cols-md-2">
        {% for project in sorted_gallery %}
          {% include projects_horizontal.liquid %}
        {% endfor %}
      </div>
    </div>
    {% else %}
    <div class="row row-cols-1 row-cols-md-3">
      {% for project in sorted_gallery %}
        {% include projects.liquid %}
      {% endfor %}
    </div>
    {% endif %}
  {% endif %}

{% endif %}
</div>
