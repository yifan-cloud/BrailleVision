"""
reference: https://scipy-lectures.org/advanced/interfacing_with_c/interfacing_with_c.html
"""

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

    int shape[3];

    Py_Initialize();
    pName = PyString_FromString("image_retrieval");
    if (pName == NULL)
        return 1;

    //TODO: not sure how to know if the top-level code executes
    pModule = PyImport_Import(pName);
    Py_DECREF(pName); //DECREF = decrement ref count

    if (pModule != NULL) {
        pFunc = PyObject_GetAttrString(pModule, "getBinnedDepthImg");
        /* pFunc is a new reference */

        if (pFunc && PyCallable_Check(pFunc)) {
            //constructing python function args
            pArgs = PyTuple_New(0); /* pArgs is a new reference */

            //call function and get return val
            pValue = PyObject_CallObject(pFunc, pArgs); /* pValue is a new reference */
            Py_DECREF(pArgs);

            //if function call failed
            if (pValue == NULL) {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                fprintf(stderr,"Call failed\n");
                return 1;
            }
            
            //TODO: not sure this should be double when the bin makes the np array ints
            double* depth_array = py_array_to_c(pValue, shape);
            Py_DECREF(pValue); //now done w/ return value
        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \"getBinnedDepthImg\"\n");
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \"image_retrieval\"\n");
        return 1;
    }
    Py_Finalize();
    return 0;
}

