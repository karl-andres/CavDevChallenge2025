# Description
Bird's Eye Simulator Environment and Engine is a simulator written in C++ to to create scenarios and test cruise control and lane centering algorithms.
For any questions or feedback, please contact khzyms@mcmaster.ca.

## Folder structure

    ├── bin             # Output of compiling BESEE is stored here along with dlls
    ├── CAS             # Files storing calibration parameters
    ├── config          # .yaml files defining possible cars, road shapes, and scenarios
        |── drive_cycle # CSV files containing different drive cycles
    ├── cpp				      # C++ source code
        ├── build       
        ├── include     # .h header files
        ├── src         # .cpp source files
    ├── loggers         # CSVs are outputted here 
    ├── python          # Python utilities for visualization and testing of csv output files

## Installation

1. Install vcpkg, a cross platform package manager we use to manage dependencies.
  - https://learn.microsoft.com/en-us/vcpkg/get_started/get-started?pivots=shell-cmd
2. Install GTest, Eigen3, and yaml-cpp using vcpkg.
  - `vcpkg install gtest eigen3 yaml-cpp easyloggingpp`
3. Install cmake.
  - On Windows, download the installer: https://cmake.org/download/
  - On Linux, open terminal and type: `sudo apt update && sudo apt install cmake build-essential`
4. Add your vcpkg directory to your environment variables so cmake can find your packages during the build process.
  - On Windows, search for Environment Variables and add a new one under System Variables. Call it VCPKG_DIR and set it to your install location.
  - On Linux, run `gedit ~/.bashrc` and add a line `export VCPKG_DIR=[YOUR_LOCATION]`. More information: https://askubuntu.com/questions/58814/how-do-i-add-environment-variables#58828
5. Install Python dependencies using the pip package manger.
  - Run `pip install -r requirements.txt`.

## Building and Running the Project

1. Navigate to the cpp folder (`cd cpp`) and build with CMake.
  - Clear the build folder. `rm -rf build` This is needed when you make changes to your CMakeLists.txt file since CMake caches variables.
  - Create the build folder. `mkdir -p build`
  - Go to the build folder. `cd build`
  - Run CMake. `cmake ..`
  - Build the cmake output. `cmake --build . --config Release -j` We build in release mode to improve performance. -j uses all CPU cores.
  - To enable debug symbols, remove `--config Release` and change the target_compile_options in CMakeLists.txt to use the O0 optimization flag instead of O3.
2. Navigate to the bin folder (`cd ../../bin`) and run the besee executable.
  - On Windows `besee.exe`
  - On Linux `./besee`
  - BESEE runs an open straight road scenario for 100 seconds by default and saves it to loggers_{SCENARIO_NAME} in the loggers folder. You can modify this by passing command line arguments.
  - For example: `besee straight_road_lead_vehicle_ftp 50 newName`
  - For a list of all possible options, check the scenarios.yaml file in the config folder.
3. Navigate to the python folder (`cd ../python`) and run the visualizer `python viz/viz_besee.py {PATH_TO_CSV}`.
  - To see graphs analyzing the ego vehicle's driving, run `python viz/vloggers.py {PATH_TO_CSV}`.
  
## Running Pytest for requirement validation

1. Navigate to python test folder `cd python/test`
2. Run `pytest -s` to run all besee scenarios and run all test functions beginning in `test`.