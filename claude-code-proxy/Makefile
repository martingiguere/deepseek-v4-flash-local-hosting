.PHONY: build test test-ollama test-bifrost run-ollama run-bifrost install clean

NAME=ccp
INSTALL_DIR=$(HOME)/bin

build:
	go build -ldflags="-s -w" -o $(NAME) .

test-ollama: build
	CCP_BACKEND=ollama python3 test_ccp.py

test-bifrost: build
	CCP_BACKEND=bifrost python3 test_ccp.py

test: test-ollama test-bifrost

run-ollama: build
	CCP_BACKEND=ollama CCP_OLLAMA_API_KEY="$$CCP_OLLAMA_API_KEY" ./$(NAME)

run-bifrost: build
	CCP_BACKEND=bifrost CCP_BIFROST_API_KEY="$$CCP_BIFROST_API_KEY" ./$(NAME)

install: build
	mkdir -p $(INSTALL_DIR)
	cp $(NAME) $(INSTALL_DIR)/$(NAME)

clean:
	rm -f $(NAME)