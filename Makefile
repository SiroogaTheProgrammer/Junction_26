.PHONY: setup qdrant load bench smoke run serve test reset cleanup vps-clean deploy llm-check

PYTHON ?= .venv/bin/python

setup:
	test -f .env || cp .env.example .env
	python3.11 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/pip install -e ".[dev]"

qdrant:
	docker compose --profile offline up -d qdrant

load:
	$(PYTHON) -m detector load

bench: load
	$(PYTHON) -m detector embed --benchmark 1000

# Legacy offline pipeline: storage cleanup would delete its results, so it is disabled here.
smoke: qdrant
	N_POSTS=2000 $(PYTHON) -m detector all
	STORAGE_CLEANUP=false $(PYTHON) -m detector serve

run: qdrant
	$(PYTHON) -m detector all --posts 10000
	STORAGE_CLEANUP=false $(PYTHON) -m detector serve

serve:
	$(PYTHON) -m detector serve

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

cleanup:
	$(PYTHON) -m detector cleanup --dry-run
	@read -r -p "Delete the files listed above? Type 'yes' to continue: " answer; \
	if [ "$$answer" = "yes" ]; then $(PYTHON) -m detector cleanup; else echo "Cancelled."; fi

# On the VPS: remove old image layers, build cache and the unused Qdrant service.
vps-clean:
	docker compose rm -sf qdrant
	docker image prune -f
	docker builder prune -f

deploy:
	git pull
	docker compose --profile prod up -d --build
	docker image prune -f

llm-check:
	$(PYTHON) -m detector.llm_check
