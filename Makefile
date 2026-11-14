.PHONY: setup qdrant load bench smoke run serve docker-smoke test reset push-data deploy llm-check

PYTHON ?= .venv/bin/python

setup:
	test -f .env || cp .env.example .env
	python3.11 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"

qdrant:
	docker compose up -d qdrant

load:
	$(PYTHON) -m detector load

bench: load
	$(PYTHON) -m detector embed --benchmark 1000

smoke: qdrant
	N_POSTS=2000 $(PYTHON) -m detector all
	$(PYTHON) -m detector serve

run: qdrant
	$(PYTHON) -m detector all --posts 10000
	$(PYTHON) -m detector serve

serve:
	$(PYTHON) -m detector serve

docker-smoke:
	test -f .env || cp .env.example .env
	docker compose up -d qdrant
	docker compose run --rm -e N_POSTS=2000 app python -m detector all
	docker compose up -d --force-recreate app

test:
	$(PYTHON) -m pytest

reset:
	@read -r -p "This deletes data/ and qdrant_storage/. Type 'yes' to continue: " answer; \
	if [ "$$answer" = "yes" ]; then \
		docker compose stop qdrant; \
		rm -rf data qdrant_storage; \
		echo "Pipeline data removed."; \
	else \
		echo "Cancelled."; \
	fi

push-data:
	test -n "$(VPS)" || (echo "Usage: make push-data VPS=root@<ip>"; exit 1)
	docker compose stop qdrant
	tar czf /tmp/junction-data.tgz data qdrant_storage
	scp /tmp/junction-data.tgz $(VPS):/tmp/junction-data.tgz
	ssh $(VPS) 'cd app && docker compose down && tar xzf /tmp/junction-data.tgz && rm /tmp/junction-data.tgz && docker compose --profile prod up -d'

deploy:
	git pull
	docker compose --profile prod up -d --build

llm-check:
	$(PYTHON) -m detector.llm_check
