run:
	poetry run python bert_tutorial/main.py
link:
	ln -s `poetry env info -p` .venv
up:
	docker-compose up -d
	docker exec -it bert_tutorial bash
