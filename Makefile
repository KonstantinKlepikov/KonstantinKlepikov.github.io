# target: all - Default target. Does nothing.
all:
	echo "Hello, this is make for my knowledgebase"
	echo "Try 'make help' and search available options"

# target: help - List of options
help:
	egrep "^# target:" [Mm]akefile

# terget: serve - build site for production
serve:
	python3 tag_generator.py
	JEKYLL_ENV=development bundle exec jekyll serve

# target: build - build site for deployment
build:
	JEKYLL_ENV=production bundle exec jekyll build

# target: ruby - set ruby env
ruby:
	rvm use 2.6.0
