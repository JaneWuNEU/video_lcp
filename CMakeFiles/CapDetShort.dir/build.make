# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ramon/Desktop/Thesis

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ramon/Desktop/Thesis

# Include any dependencies generated for this target.
include CMakeFiles/CapDetShort.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/CapDetShort.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/CapDetShort.dir/flags.make

CMakeFiles/CapDetShort.dir/CapDetShort.cpp.o: CMakeFiles/CapDetShort.dir/flags.make
CMakeFiles/CapDetShort.dir/CapDetShort.cpp.o: CapDetShort.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ramon/Desktop/Thesis/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/CapDetShort.dir/CapDetShort.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/CapDetShort.dir/CapDetShort.cpp.o -c /home/ramon/Desktop/Thesis/CapDetShort.cpp

CMakeFiles/CapDetShort.dir/CapDetShort.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/CapDetShort.dir/CapDetShort.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ramon/Desktop/Thesis/CapDetShort.cpp > CMakeFiles/CapDetShort.dir/CapDetShort.cpp.i

CMakeFiles/CapDetShort.dir/CapDetShort.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/CapDetShort.dir/CapDetShort.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ramon/Desktop/Thesis/CapDetShort.cpp -o CMakeFiles/CapDetShort.dir/CapDetShort.cpp.s

CMakeFiles/CapDetShort.dir/CapDetShort.cpp.o.requires:

.PHONY : CMakeFiles/CapDetShort.dir/CapDetShort.cpp.o.requires

CMakeFiles/CapDetShort.dir/CapDetShort.cpp.o.provides: CMakeFiles/CapDetShort.dir/CapDetShort.cpp.o.requires
	$(MAKE) -f CMakeFiles/CapDetShort.dir/build.make CMakeFiles/CapDetShort.dir/CapDetShort.cpp.o.provides.build
.PHONY : CMakeFiles/CapDetShort.dir/CapDetShort.cpp.o.provides

CMakeFiles/CapDetShort.dir/CapDetShort.cpp.o.provides.build: CMakeFiles/CapDetShort.dir/CapDetShort.cpp.o


# Object files for target CapDetShort
CapDetShort_OBJECTS = \
"CMakeFiles/CapDetShort.dir/CapDetShort.cpp.o"

# External object files for target CapDetShort
CapDetShort_EXTERNAL_OBJECTS =

CapDetShort: CMakeFiles/CapDetShort.dir/CapDetShort.cpp.o
CapDetShort: CMakeFiles/CapDetShort.dir/build.make
CapDetShort: /usr/local/lib/libopencv_dnn.so.4.1.1
CapDetShort: /usr/local/lib/libopencv_gapi.so.4.1.1
CapDetShort: /usr/local/lib/libopencv_highgui.so.4.1.1
CapDetShort: /usr/local/lib/libopencv_ml.so.4.1.1
CapDetShort: /usr/local/lib/libopencv_objdetect.so.4.1.1
CapDetShort: /usr/local/lib/libopencv_photo.so.4.1.1
CapDetShort: /usr/local/lib/libopencv_stitching.so.4.1.1
CapDetShort: /usr/local/lib/libopencv_video.so.4.1.1
CapDetShort: /usr/local/lib/libopencv_videoio.so.4.1.1
CapDetShort: /usr/local/lib/libopencv_imgcodecs.so.4.1.1
CapDetShort: /usr/local/lib/libopencv_calib3d.so.4.1.1
CapDetShort: /usr/local/lib/libopencv_features2d.so.4.1.1
CapDetShort: /usr/local/lib/libopencv_flann.so.4.1.1
CapDetShort: /usr/local/lib/libopencv_imgproc.so.4.1.1
CapDetShort: /usr/local/lib/libopencv_core.so.4.1.1
CapDetShort: CMakeFiles/CapDetShort.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ramon/Desktop/Thesis/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable CapDetShort"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/CapDetShort.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/CapDetShort.dir/build: CapDetShort

.PHONY : CMakeFiles/CapDetShort.dir/build

CMakeFiles/CapDetShort.dir/requires: CMakeFiles/CapDetShort.dir/CapDetShort.cpp.o.requires

.PHONY : CMakeFiles/CapDetShort.dir/requires

CMakeFiles/CapDetShort.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/CapDetShort.dir/cmake_clean.cmake
.PHONY : CMakeFiles/CapDetShort.dir/clean

CMakeFiles/CapDetShort.dir/depend:
	cd /home/ramon/Desktop/Thesis && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ramon/Desktop/Thesis /home/ramon/Desktop/Thesis /home/ramon/Desktop/Thesis /home/ramon/Desktop/Thesis /home/ramon/Desktop/Thesis/CMakeFiles/CapDetShort.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/CapDetShort.dir/depend

