PY ?= python3

.PHONY: todo explorar rapido graficar entregable test

todo: explorar graficar entregable test

explorar:
	$(PY) scripts/explorar.py

rapido:
	$(PY) scripts/explorar.py --rapido

graficar:
	$(PY) scripts/graficar.py

entregable:
	$(PY) scripts/construir_entregable.py

test:
	$(PY) -m pytest -q
