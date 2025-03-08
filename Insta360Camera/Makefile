# Compiler and linker settings
# Detect architecture
ARCH := $(shell uname -m)

# Set library path based on architecture
ifeq ($(ARCH), x86_64)
    LIBS = -Llib/ -lCameraSDK
	LOC = -rpath=lib/
else ifeq ($(ARCH), aarch64)
    LIBS = -Llib_jetson/ -lCameraSDK
	LOC = -rpath=lib_jetson/
else 
    $(error Unsupported architecture: $(ARCH))
endif


CXX = g++
CXXFLAGS = -std=c++17 -Iinclude `pkg-config --cflags opencv4` # Include directory for header files
LDFLAGS =  -Wl,$(LOC) -lavcodec -lavutil -lswscale $(LIBS) `pkg-config --libs opencv4` # Path to the library
# LDFLAGS =  -lavcodec -lavutil -lswscale -Llib/ -lCameraSDK  `pkg-config --libs opencv4` # Path to the library

# Directories
SRCDIR = src
OBJDIR = obj
BINDIR = bin

# Files
# SRC = src/stream.cpp # All source files in src/
SRC = src/shared_memtest.cpp # All source files in src/
# SRC = src/main.cpp # All source files in src/

# SRC = src/simple_cam.cpp # All source files in src/

# SRC = $(wildcard $(SRCDIR)/*.cpp)  # All source files in src/

OBJ = $(SRC:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)  # Object files in obj/
TARGET = $(BINDIR)/main  # Output executable

# Default target (build the executable)
all: $(TARGET)

# Build the executable

$(TARGET) : $(OBJ)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJ) $(LDFLAGS)

# $(TARGET): $(OBJ)
# 	$(CXX) $(OBJ) $(LDFLAGS) -o $@

# Rule to compile source files to object files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# # Rule to compile source files to object files
# $(OBJDIR)/%.o: $(SRCDIR)/%.cpp
# 	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up the generated files
clean:
	rm -rf $(OBJDIR)/*.o $(TARGET)

# Create the necessary directories if they don't exist
$(OBJDIR) $(BINDIR):
	mkdir -p $@

.PHONY: all clean
