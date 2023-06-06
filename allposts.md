---
layout: page
title: Все статьи
permalink: /allposts/
---

<div class="container mx-auto px-2 py-1">
    {% for post in site.posts %}
        {% unless post.tags contains "additional" %}
            {% include allposts_block.html %}
        {% endunless %}
    {% endfor %}
    <h3><a class="no-underline text-accent" href="{{ site.baseurl }}{% link additional.md %}">Еще статьи...</a></h3>
</div>
{% include tagcloud.html %}
{% include link-to-my-knowledge-simple.html %}
