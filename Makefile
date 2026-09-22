.PHONY: install test eval eval-visible eval-original cli demo clean

install:  ## no external dependencies required; placeholder for symmetry
	@echo "No dependencies to install (Python 3.11+ standard library only)."

test:  ## run offline unit tests
	python3 -m unittest discover -s tests -v

eval:  ## run the full evaluation suite (visible + original cases)
	python3 -m evaluation.runner

eval-visible:
	python3 -m evaluation.runner --file visible

eval-original:
	python3 -m evaluation.runner --file original

cli:  ## interactive chat
	python3 -m app.cli

demo:  ## run the scripted demo (also used for GIF recording)
	python3 scripts/demo.py

clean:
	rm -rf .cache traces eval-results.json
