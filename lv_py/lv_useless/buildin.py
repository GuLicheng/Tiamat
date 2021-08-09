#谨慎使用
import inspect 
import functools
import ctypes

class PyObject(ctypes.Structure):
    class PyType(ctypes.Structure):
        pass
    ssize = ctypes.c_int64 if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_int32
    _fields_ = [
        ('ob_refcnt', ssize),
        ('ob_type', ctypes.POINTER(PyType)),
    ]
    def incref(self):
        self.ob_refcnt += 1
    def decref(self):
        self.ob_refcnt -= 1


def sign(klass,value):
    class SlotsProxy(PyObject):
        _fields_ = [('dict', ctypes.POINTER(PyObject))]
    name = klass.__name__
    target = klass.__dict__
    proxy_dict = SlotsProxy.from_address(id(target))
    namespace = {}
    ctypes.pythonapi.PyDict_SetItem(
        ctypes.py_object(namespace),
        ctypes.py_object(name),
        proxy_dict.dict,
    )
    namespace[name]["__sign__"] = value

# 修改 built-in  str
sign(str,"某某人专用")
"1234567".__sign__