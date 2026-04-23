#include <Python.h>
#include <wchar.h>

int main(int argc, char* argv[]) {
    Py_Initialize();

    PyObject *sys_path = PySys_GetObject("path");
    PyObject *test_path_str = PyUnicode_FromString("/home/dylenthomas/LiveASRonRPi-4/tests/python_embed_test");
    PyObject *site_packages_path_str = PyUnicode_FromString("/home/dylenthomas/LiveASRonRPi-4/tests/python_embed_test/build/venv/lib/python3.14/site-packages");
    PyList_Append(sys_path, test_path_str);
    PyList_Append(sys_path, site_packages_path_str);

    PyRun_SimpleString("import sys; print('sys.path:', sys.path)");

    PyObject* pModule = PyImport_Import(PyUnicode_DecodeFSDefault("pythonTest"));
    if (pModule == NULL) {
        if (PyErr_Occurred()) { PyErr_Print(); }
        fprintf(stderr, "ERROR: Failed to import module.\n");
        return -1;
    }

    PyObject *pFunc = PyObject_GetAttrString(pModule, "main");
    PyObject *pResult = PyObject_CallNoArgs(pFunc);

    Py_DECREF(pModule);
    Py_DECREF(test_path_str);
    Py_DECREF(site_packages_path_str);
    Py_DECREF(pFunc);
    Py_DECREF(pResult);
}