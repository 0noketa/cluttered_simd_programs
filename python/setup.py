import platform
from setuptools import Extension, setup
from Cython.Build import cythonize

env_ = {}
if platform.architecture()[0] == "64bit":
    env_["PTR_BITS"] = 64
else:
    env_["PTR_BITS"] = 32

m =  platform.machine()
if m == "":
    m = "_"
env_["__i386__"] = int(m in ["i386", "i686", "x86"])
env_["__AMD64__"] = int(m in ["x86_64", "AMD64"])
env_["__MMX__"] = env_["__i386__"]
env_["__SSE2__"] = env_["__AMD64__"]
env_["__AVX2__"] = 0
env_["__arm__"] = int(m.startswith("arm"))
env_["__aarch64__"] = int(m == "aarch64")
env_["__NEON__"] = 0


setup(ext_modules=cythonize(
    module_list=[
        Extension("simd_lib", ["simd_lib.pyx", "../src/sort.c", "../src/search.c"])
    ],
    compile_time_env=env_))
