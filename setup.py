from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Compiler import Options

Options.annotate = True

ext_modules = [Extension(
  'corr_lda',
  ['corr_lda.pyx'],
  language="c++",
  extra_compile_args=["-std=c++11"],
  extra_link_args=["-std=c++11"]
  )]

setup (
  name = 'Corr_LDA',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)
