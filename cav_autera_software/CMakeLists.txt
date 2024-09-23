cmake_minimum_required(VERSION 3.10)
project(autera_software)

# == project/user settings == #
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(VCPKG_DIR $ENV{VCPKG_DIR})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_DOCS OFF)
set(BUILD_TESTS OFF)
# ================== #

# Configure vcpkg location
include(${VCPKG_DIR}/scripts/buildsystems/vcpkg.cmake)
set(CMAKE_TOOLCHAIN_FILE ${VCPKG_DIR}/scripts/buildsystems/vcpkg.cmake)

# Dependencies
find_package(Eigen3 CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

# glob to find all files that need to be included
file(GLOB_RECURSE sources
    src/**.cpp
    src/**.cc
)

# Target definitions
add_library(${PROJECT_NAME} STATIC ${sources})
if (MSVC)
    target_compile_options(${PROJECT_NAME} PUBLIC /O3)
else()
    target_compile_options(${PROJECT_NAME} PUBLIC -O3)
    #target_compile_options(${PROJECT_NAME} PUBLIC -O0 -g) # uncomment for debugging purposes
endif()

# Include directories.
target_include_directories(${PROJECT_NAME} PUBLIC
    include
)

# Linking phase
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${YAML_CPP_LIBRARIES}
)

# == GOOGLETEST BUILD == #
if(BUILD_TESTS) 
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)
    SET(GCC_COVERAGE_COMPILE_FLAGS "-fprofile-arcs -ftest-coverage -fno-inline")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}") 
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")
    enable_testing()
    find_package(GTest CONFIG REQUIRED)
    file(GLOB_RECURSE tests test/*.cpp)
    add_executable(test_${PROJECT_NAME} ${tests})
    target_compile_options(test_${PROJECT_NAME} PUBLIC)
    target_include_directories(test_${PROJECT_NAME} PUBLIC 
        include
        src 
        test)
    target_link_libraries(test_${PROJECT_NAME} PRIVATE  
        ${PROJECT_NAME} # Linking the controls library.
        GTest::gtest
        GTest::gtest_main
    )
    include(GoogleTest)
    gtest_discover_tests(test_${PROJECT_NAME})
endif()
# ====================== #

# == DOCUMENTATION GENERATION (DOXYGEN -> SPHINX) ==
if(BUILD_DOCS)
    # Doxygen
    find_package(Doxygen REQUIRED)

    set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/documentation/autodoc/doxygen)
    set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)

    file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR}) #Doxygen won't create this for us
    add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
                    DEPENDS ${sources}
                    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/documentation/autodoc/doxygen
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    MAIN_DEPENDENCY ${CMAKE_CURRENT_SOURCE_DIR}/documentation/autodoc/doxygen
                    COMMENT "Generating docs")

    add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})

    #Look for an executable called sphinx-build
    find_program(SPHINX_EXECUTABLE
                NAMES sphinx-build
                DOC "Path to sphinx-build executable")

    include(FindPackageHandleStandardArgs)

    #Handle standard arguments to find_package like REQUIRED and QUIET
    find_package_handle_standard_args(Sphinx
                                    "Failed to find sphinx-build executable"
                                    SPHINX_EXECUTABLE)

    set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/documentation/autodoc)
    set(SPHINX_BUILD ${CMAKE_CURRENT_SOURCE_DIR}/documentation/autodoc/sphinx)

    add_custom_target(Sphinx ALL
                    COMMAND ${SPHINX_EXECUTABLE} -b html
                    # Tell Breathe where to find the Doxygen output
                    -Dbreathe_projects.${PROJECT_NAME}=${DOXYGEN_OUTPUT_DIR}/xml
                    ${SPHINX_SOURCE} ${SPHINX_BUILD}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    COMMENT "Generating documentation with Sphinx")
endif()
# ================== #
