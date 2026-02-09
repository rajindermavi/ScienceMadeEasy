.PHONY: up down restart logs status wipe

up:
	docker compose up -d

down:
	docker compose down

restart:
	docker compose down
	docker compose up -d

logs:
	docker compose logs -f qdrant

status:
	docker compose ps
