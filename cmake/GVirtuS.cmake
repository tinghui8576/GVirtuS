include_directories(include)

if(NOT ${PROJECT_NAME} STREQUAL "gvirtus-common")
    include_directories(${GVIRTUS_HOME}/include)
    link_directories(${GVIRTUS_HOME}/lib)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGVIRTUS_HOME=\\\"${GVIRTUS_HOME}\\\"")
set(CMAKE_SKIP_RPATH ON)

# Include the support to external projects
include(ExternalProject)

# Set the external install location
set(EXTERNAL_INSTALL_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/external)

# log4cplus
find_path(LOG4CPLUS_INCLUDE_DIR log4cplus/logger.h)
find_library(LOG4CPLUS_LIBRARY log4cplus)

if(NOT LOG4CPLUS_INCLUDE_DIR OR NOT LOG4CPLUS_LIBRARY)
    message(STATUS "log4cplus not found. Downloading and installing it.")
    include(ExternalProject)

    set(LOG4CPLUS_INSTALL_DIR ${EXTERNAL_INSTALL_LOCATION})
    set(LOG4CPLUS_INCLUDE_DIR ${LOG4CPLUS_INSTALL_DIR}/include)
    set(LOG4CPLUS_LIBRARY ${LOG4CPLUS_INSTALL_DIR}/lib/liblog4cplus.so)

    ExternalProject_Add(log4cplus
        URL https://downloads.sourceforge.net/log4cplus/log4cplus-stable/2.1.2/log4cplus-2.1.2.tar.gz
        TIMEOUT 360
        PREFIX ${CMAKE_CURRENT_BINARY_DIR}/log4cplus-prefix
        BUILD_IN_SOURCE 1
        CONFIGURE_COMMAND ./configure --prefix=${LOG4CPLUS_INSTALL_DIR} CFLAGS=-fPIC CPPFLAGS=-I${LOG4CPLUS_INCLUDE_DIR} LDFLAGS=-L${LOG4CPLUS_INSTALL_DIR}/lib
        BUILD_COMMAND make
        INSTALL_COMMAND make install
    )
    set(LOG4CPLUS_EXTERNALLY_DOWNLOADED TRUE)
else()
    message(STATUS "log4cplus found in ${LOG4CPLUS_INCLUDE_DIR}")
    set(LOG4CPLUS_EXTERNALLY_DOWNLOADED FALSE)
endif()

# Export variables
set(LIBLOG4CPLUS ${LOG4CPLUS_LIBRARY})
include_directories(${LOG4CPLUS_INCLUDE_DIR})


find_package(Threads REQUIRED)

if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/gvirtus)
    message(STATUS "Installing ${CMAKE_CURRENT_SOURCE_DIR}/include/gvirtus to ${GVIRTUS_HOME}/include")
    install(DIRECTORY include/gvirtus DESTINATION ${GVIRTUS_HOME}/include)
endif()

macro(gvirtus_install_target target_name)
    install(TARGETS ${target_name}
            LIBRARY DESTINATION ${GVIRTUS_HOME}/lib
            ARCHIVE DESTINATION ${GVIRTUS_HOME}/lib
            RUNTIME DESTINATION ${GVIRTUS_HOME}/bin
            INCLUDES DESTINATION ${GVIRTUS_HOME}/include)
endmacro()

function(gvirtus_add_backend)
    if(ARGC LESS 3)
        message(FATAL_ERROR "Usage: gvirtus_add_backend(wrapped_library version source1 ...)")
    endif()
    list(GET ARGV 0 wrapped_library)
    list(REMOVE_AT ARGV 0)
    list(GET ARGV 0 version)
    list(REMOVE_AT ARGV 0)
    if(NOT "${PROJECT_NAME}" STREQUAL "gvirtus-plugin-${wrapped_library}")
        message(FATAL_ERROR "This project should be named gvirtus-plugin-${wrapped_library}")
    endif()
    add_library(${PROJECT_NAME} SHARED
            ${ARGV})
    target_link_libraries(${PROJECT_NAME} ${LIBLOG4CPLUS} gvirtus-common gvirtus-communicators)
    install(TARGETS ${PROJECT_NAME}
            LIBRARY DESTINATION ${GVIRTUS_HOME}/lib)
endfunction()

function(gvirtus_add_frontend)
    if(ARGC LESS 3)
        message(FATAL_ERROR "Usage: gvirtus_add_frontend(wrapped_library version source1 ...)")
    endif()
    list(GET ARGV 0 wrapped_library)
    list(REMOVE_AT ARGV 0)
    list(GET ARGV 0 version)
    list(REMOVE_AT ARGV 0)
    add_library(${wrapped_library} SHARED
            ${ARGV})
    target_link_libraries(${wrapped_library} ${LIBLOG4CPLUS} gvirtus-common gvirtus-communicators gvirtus-frontend)
    set_target_properties(${wrapped_library} PROPERTIES VERSION ${version})
    string(REGEX REPLACE "\\..*" "" soversion ${version})
    set_target_properties(${wrapped_library} PROPERTIES SOVERSION ${soversion})
    string(TOUPPER "LIB${wrapped_library}_${version}" script)
    set(script "${script} {\n local:\n*;\n};")
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/lib${wrapped_library}.map "${script}")
    install(TARGETS ${wrapped_library}
            LIBRARY DESTINATION ${GVIRTUS_HOME}/lib/frontend)
endfunction()
