.PHONY: up down logs shell migrate createsuperuser ingest-year test lint format mypy install-hooks

up:
	docker-compose up -d --build

down:
	docker-compose down --volumes

logs:
	docker-compose logs -f

shell:
	docker-compose exec web python manage.py shell

migrate:
	docker-compose exec web python manage.py migrate

createsuperuser:
	docker-compose exec web python manage.py createsuperuser

ingest-year:
	docker-compose exec web python manage.py ingest_year_ohlcv EURUSD M15

test:
	docker-compose exec web pytest

lint:
	docker-compose exec web flake8 .
	docker-compose exec web isort . --check-only
	docker-compose exec web black . --check
	docker-compose exec web mypy .

format:
	docker-compose exec web isort .
	docker-compose exec web black .

install-hooks:
	pip install pre-commit
	pre-commit install


.PHONY: scan-patterns test-patterns verify-patterns

scan-patterns:
	@echo "Scanning patterns for EURUSD H1 for the last 30 days..."
	docker-compose exec web python manage.py shell -c "from apps.analytics.tasks import scan_for_candidate_patterns; from django.utils import timezone; import datetime; start = timezone.now() - datetime.timedelta(days=30); end = timezone.now(); scan_for_candidate_patterns.delay('EURUSD', 'H1', start, end)"

verify-patterns:
	@echo "Verifying recent patterns for EURUSD H1..."
	docker-compose exec web python manage.py shell -c "from apps.analytics.tasks import verify_pattern_candidates_task; result={'symbol': 'EURUSD', 'timeframe': 'H1'}; verify_pattern_candidates_task.delay(result)"

test-patterns:
	docker-compose exec web pytest apps/analytics/patterns/