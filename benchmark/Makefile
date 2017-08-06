SRC_DIR = src
INC_DIR = include
BUILD_DIR = build
LIB_DIR = lib
BIN_DIR = bin

CC = c++ -std=c++11 -O2
CFLAGS = -fPIC -I$(TF_INC) `pkg-config opencv --cflags` -I$(INC_DIR)
LDFLAGS = `pkg-config opencv --libs` -L$(TF_LIB) -ltensorflow -lglut -lGLEW -lGL -lgflags

SRC = main.cc renderer.cc utils.cc processor.cc
HEADER = timer.h renderer.h utils.h processor.h
SRCS = $(addprefix $(SRC_DIR)/, $(SRC))
HEADERS = $(addprefix $(INC_DIR)/, $(HEADER))

all: $(BIN_DIR)/benchmark

# Main exectutable
$(BIN_DIR)/benchmark: $(BIN_DIR) $(SRCS) $(HEADERS)
	$(CC) -o $@ $(SRCS) $(CFLAGS) $(LDFLAGS) 

$(BUILD_DIR):
	mkdir -p $@

$(LIB_DIR):
	mkdir -p $@

$(BIN_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR) $(BIN_DIR)
