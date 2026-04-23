#include <Python.h>
#include <wchar.h>

int main(int argc, char* argv[]) {
    PyStatus status;
    PyConfig config;
    PyConfig_InitPythonConfig(&config);

    status = PyConfig_SetString(&config, &config.program_name,
        L"/home/dylenthomas/LiveASRonRPi-4/tests/python_embed_test/build/venv/bin/python3");
    
    if (PyStatus_Exception(status)) {
        PyConfig_Clear(&config);
        return -1;
    }

    config.module_search_paths_set = 1;
    status = PyWideStringList_Append(&config.module_search_paths, 
        L"/home/dylenthomas/LiveASRonRPi-4/tests/python_embed_test/build/venv/lib");
    
    if (PyStatus_Exception(status)) {
        PyConfig_Clear(&config);
        return -1;
    }

    status = Py_InitializeFromConfig(&config);
    if (PyStatus_Exception(status)) {
        PyConfig_Clear(&config);
        Py_ExitStatusException(status);
        return -1;
    }

    PyConfig_Clear(&config);

    PyRun_SimpleString(
        "import sys\n"
        "print('Python sys.path:', sys.path)\n"
        "print('Python version:', sys.version)\n"
    );

    printf("Initialized Python!\n");

    PyObject* pModule = PyImport_Import(PyUnicode_DecodeFSDefault("pythonTest"));
    if (pModule == NULL) {
        if (PyErr_Occurred()) { PyErr_Print(); }
        fprintf(stderr, "ERROR: Failed to import module.\n");
        return -1;
    }
}