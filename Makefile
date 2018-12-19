VIRTUAL_ENV=venv
ENVIRONMENT_URL=https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip
EXAMPLE_NOTEBOOK=https://raw.githubusercontent.com/udacity/deep-reinforcement-learning/master/p3_collab-compet/Tennis.ipynb

${VIRTUAL_ENV}:
	virtualenv ${VIRTUAL_ENV} -p python3.6
	(source ${VIRTUAL_ENV}/bin/activate; pip install -r requirements.txt;)

.PHONY: virtualenv
virtualenv: ${VIRTUAL_ENV}

.PHONY: freeze
freeze:
	(source ${VIRTUAL_ENV}/bin/activate; pip freeze > requirements.txt; )

.PHONY: clean
clean:
	rm -rf ${VIRTUAL_ENV}
	rm -rf files/

files/Tennis.app.zip:
	mkdir -p files/
	wget -O $@ ${ENVIRONMENT_URL}

files/Tennis.app: files/Tennis.app.zip
	unzip -d files $@.zip

files/Tennis.ipynb:
	mkdir -p files/
	wget -O $@ ${EXAMPLE_NOTEBOOK}

.PHONY: setup
setup: files/Tennis.ipynb files/Tennis.app virtualenv
