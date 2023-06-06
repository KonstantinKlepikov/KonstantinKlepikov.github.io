---
layout: page
permalink: /additional/
---

<div class="container mx-auto px-2 py-1">
    {% for post in site.posts %}
        {% if post.tags contains "additional" %}
            {% include allposts_block.html %}
        {% endif %}
    {% endfor %}
</div>
{% include tagcloud.html %}
{% include link-to-my-knowledge-simple.html %}