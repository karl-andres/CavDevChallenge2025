cmake_minimum_required(VERSION 3.10)
project(besee)

# == project/user settings == #
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY $<1:${CMAKE_SOURCE_DIR}/../bin>)
set(VCPKG_DIR $ENV{VCPKG_DIR})
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_DOCS OFF)
set(BUILD_TESTS OFF)
# required for Windows due to conflict between C++17 and Windows header
add_compile_definitions(_HAS_STD_BYTE=0)
# ================== #

# configure vcpkg location
include(${VCPKG_DIR}/scripts/buildsystems/vcpkg.cmake)

# dependencies
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)

# build cav controller code as static library
add_subdirectory(../cav_autera_software autera_software)

# glob to find all files that need to be included (exclude main.cpp for test building)
file(GLOB_RECURSE sources 
    src/*.cpp
)
list(FILTER sources EXCLUDE REGEX "(.*)/src/main\.cpp")

set(includes
    include
    ../cav_autera_software/include
    ../bin
)

set(linking_dependencies
    autera_software
    plantmodel
    Eigen3::Eigen 
    ${YAML_CPP_LIBRARIES}
)

# target definitions
add_executable(${CMAKE_PROJECT_NAME} ${sources} src/main.cpp)
if (MSVC)
    target_compile_options(${CMAKE_PROJECT_NAME} PUBLIC /O3)
else()
    target_compile_options(${CMAKE_PROJECT_NAME} PUBLIC -O3)
    #target_compile_options(${CMAKE_PROJECT_NAME} PUBLIC -O0 -g) # uncomment for debugging purposes
endif()

include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${includes})

# linking phase
add_dependencies(${CMAKE_PROJECT_NAME} autera_software)
target_link_directories(${CMAKE_PROJECT_NAME} PUBLIC ${includes})
target_link_libraries(${CMAKE_PROJECT_NAME} PUBLIC ${linking_dependencies})

# == GOOGLETEST BUILD == #
if(BUILD_TESTS) 
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    set(CMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE PRE_TEST)
    SET(GCC_COVERAGE_COMPILE_FLAGS "-fprofile-arcs -ftest-coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GCC_COVERAGE_COMPILE_FLAGS}") 
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${GCC_COVERAGE_LINK_FLAGS}")
    enable_testing()
    find_package(GTest CONFIG REQUIRED)
    file(GLOB_RECURSE tests test/*.cpp)
    add_executable(test_${CMAKE_PROJECT_NAME} ${sources} ${tests})
    target_compile_options(test_${CMAKE_PROJECT_NAME} PUBLIC -O0 -g)
    include_directories(test_${CMAKE_PROJECT_NAME} PUBLIC ${includes})
    target_link_libraries(test_${CMAKE_PROJECT_NAME} ${linking_dependencies} GTest::gtest_main GTest::gtest)
    include(GoogleTest)
    gtest_discover_tests(test_${CMAKE_PROJECT_NAME})
endif()
# ====================== #

# == DOCUMENTATION GENERATION (DOXYGEN -> SPHINX) ==
if(BUILD_DOCS)
    # Doxygen
    find_package(Doxygen REQUIRED)

    set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen)
    set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)

    file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR}) #Doxygen won't create this for us
    add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
                    DEPENDS ${sources}
                    COMMAND ${DOXYGEN_EXECUTABLE} docs/Doxyfile
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    MAIN_DEPENDENCY docs/Doxyfile
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

    set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/docs)
    set(SPHINX_BUILD ${CMAKE_CURRENT_SOURCE_DIR}/docs/sphinx)

    add_custom_target(Sphinx ALL
                    COMMAND ${SPHINX_EXECUTABLE} -b html
                    # Tell Breathe where to find the Doxygen output
                    -Dbreathe_projects.${CMAKE_PROJECT_NAME}=${DOXYGEN_OUTPUT_DIR}/xml
                    ${SPHINX_SOURCE} ${SPHINX_BUILD}
                    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                    COMMENT "Generating documentation with Sphinx")
endif()
# ================== #
