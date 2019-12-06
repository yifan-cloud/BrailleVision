#include <Python.h>
#include <numpy/arrayobject.h>

/* returns a C array copy of the given numpy array,
 * and sets the given shape array to the numpy array dimensions
 */
double* py_array_to_c(PyArrayObject *in_array, int *shape)
{
    NpyIter *in_iter;
    NpyIter_IterNextFunc *in_iternext;

    /*  create the iterator */
    in_iter = NpyIter_New(in_array, NPY_ITER_READONLY, NPY_KEEPORDER,
                             NPY_NO_CASTING, NULL);
    if (in_iter == NULL)
        goto fail;
    
    in_iternext = NpyIter_GetIterNext(in_iter, NULL);
    if (in_iternext == NULL) {
        NpyIter_Deallocate(in_iter);
        goto fail;
    }

    double ** in_dataptr = (double **) NpyIter_GetDataPtrArray(in_iter);

    npy_intp *in_shape = PyArray_SHAPE(in_array);
    shape[0] = in_shape[0];
    shape[1] = in_shape[1];
    shape[2] = in_shape[2];
    int len = shape[0] * shape[1] * shape[2];
    double* out_array = (double*) malloc(sizeof(double) * len);

    /*  iterate over the array */
    int idx = 0;
    do {
        out_array[idx] = **in_dataptr;
        idx++;
    } while(in_iternext(in_iter));

    NpyIter_Deallocate(in_iter);

    return out_array;

    /*  in case bad things happen */
    fail:
        Py_XDECREF(out_array);
        return NULL;
}

int main(int argc, char *argv[])
{
    PyObject *pName, *pModule, *pFunc;
    PyObject *pArgs, *pValue, *pShape;
    int i;
    //pShape is for array shape

    Py_Initialize();
    pName = PyString_FromString("realsense");
    if (pName == NULL)
        return 1;

    pModule = PyImport_Import(pName); //TODO: not sure how to know if the top-level code executes
    Py_DECREF(pName); //DECREF = decrement ref count

    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, "bin_ndarray");
        /* pFunc is a new reference */

        if (pFunc && PyCallable_Check(pFunc)) {
            //constructing python function args
            //TODO: need ndarray, tuple, and potentially operation
            /*
            not sure how to get an ndarray in C, but I suppose we could get it from Python to start with
            and just keep the reference
            use PyTuple_New to construct the shape
            string can probably be converted more easily
            */
           pShape = PyTuple_New(3);
           long val = 4;
           PyTuple_SetItem(pShape, 0, PyLong_FromLong(val)); //need to turn C ints into python ints. 4, 4, 3
           val = 3;
           PyTuple_SetItem(pShape, 1, PyLong_FromLong(val));
           PyTuple_SetItem(pShape, 2, PyLong_FromLong(val));

            pArgs = PyTuple_New(2); //assuming 2 if not passing in operation
            // for (i = 0; i < argc - 3; ++i) {
            //     pValue = PyInt_FromLong(atoi(argv[i + 3]));
            //     if (!pValue) {
            //         Py_DECREF(pArgs);
            //         Py_DECREF(pModule);
            //         fprintf(stderr, "Cannot convert argument\n");
            //         return 1;
            //     }
            //     /* pValue reference stolen here: */
            //     PyTuple_SetItem(pArgs, i, pValue);
            // }

            PyTuple_SetItem(pArgs, 1, pShape);
            Py_DECREF(pShape);

            //call function and get return val
            pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);
            if (pValue != NULL) {
                printf("Result of call: %ld\n", PyInt_AsLong(pValue));
                Py_DECREF(pValue);
            }
            else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                fprintf(stderr,"Call failed\n");
                return 1;
            }
        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \"bin_ndarray\"\n");
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"realsense\"\n");
        return 1;
    }
    Py_Finalize();
    return 0;
}

