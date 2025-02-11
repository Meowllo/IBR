import ctypes
import os

# Define C types for the structures
class DataPoint(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int),
                ("values", ctypes.POINTER(ctypes.c_double))]

class Data(ctypes.Structure):
    _fields_ = [("size", ctypes.c_int),
                ("points", ctypes.POINTER(DataPoint))]

# Load the shared library
lib = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)), "multi_bisect.so"))

# Define argument and return types for the C++ function
lib.multi_bisect_v2.restype = ctypes.POINTER(Data)
lib.multi_bisect_v2.argtypes = [
    ctypes.POINTER(Data),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_int,
    ctypes.c_bool
]

lib.test.restype = ctypes.POINTER(Data)
lib.test.argtypes = [ctypes.POINTER(Data)]

# lib.free_topk_result.argtypes = [ctypes.POINTER(TopKResult)]

# Helper function to create a Data structure from Python lists
def create_data(data_list):
    points = []
    for point in data_list:
        values = (ctypes.c_double * len(point))(*point)
        points.append(DataPoint(len(point), values))
    points_array = (DataPoint * len(points))(*points)
    return Data(len(points), points_array)

# Helper function to convert TopKResult to Python list
def topkresult_to_list(topk_result: ctypes.POINTER):
    result = []
    for i in range(topk_result.contents.size):
        point = topk_result.contents.points[i]
        result.append([point.values[j] for j in range(point.size)])
    return result


def multi_bisect_cpp(data, n, k, constraints, lb=0, ub=1, epsilon=0.001, max_loop=4, verbose=False):
    data = create_data(data)
    constraints = (ctypes.c_double * len(constraints))(*constraints)
    result_ptr = lib.multi_bisect_v2(data, n, k, constraints, len(constraints), lb, ub, epsilon, max_loop, verbose)
    result = topkresult_to_list(result_ptr)
    return result
