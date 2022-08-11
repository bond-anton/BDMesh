import sys
from codecs import open
from os import path, remove
import re

from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext

from Cython.Build import cythonize


here = path.abspath(path.dirname(__file__))
package_name = 'BDMesh'
version_file = path.join(here, package_name, '_version.py')
with open(version_file, 'rt') as f:
    version_file_line = f.read()
version_re = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(version_re, version_file_line, re.M)
if mo:
    version_string = mo.group(1)
else:
    raise RuntimeError('Unable to find version string in %s.' % (version_file,))

readme_file = path.join(here, 'README.md')
with open(readme_file, encoding='utf-8') as f:
    long_description = f.read()

extensions = [
    Extension(
        'BDMesh._helpers',
        ['BDMesh/_helpers.pyx'],
        depends=['BDMesh/_helpers.pxd'],
    ),
    Extension(
        'BDMesh.Mesh1D',
        ['BDMesh/Mesh1D.pyx'],
        depends=['BDMesh/Mesh1D.pxd'],
    ),
    Extension(
        'BDMesh.Mesh1DUniform',
        ['BDMesh/Mesh1DUniform.pyx'],
        depends=['BDMesh/Mesh1DUniform.pxd'],
    ),
    Extension(
        'BDMesh.TreeMesh1D',
        ['BDMesh/TreeMesh1D.pyx'],
        depends=['BDMesh/TreeMesh1D.pxd'],
    ),
    Extension(
        'BDMesh.TreeMesh1DUniform',
        ['BDMesh/TreeMesh1DUniform.pyx'],
        depends=['BDMesh/TreeMesh1DUniform.pxd'],
    ),
]

c_opt = {'msvc': [],
         'mingw32': [],
         'unix': []}
l_opt = {'mingw32': [],
         'unix': []}


# check whether compiler supports a flag
def has_flag(compiler, flag_name):
    import tempfile
    from distutils.errors import CompileError
    with tempfile.NamedTemporaryFile('w', suffix='.cpp') as file:
        file.write('int main (int argc, char **argv) { return 0; }')
        try:
            res = compiler.compile([file.name], extra_postargs=[flag_name])
            for item in res:
                remove(item)
        except CompileError:
            return False
    return True


# filter flags, returns list of accepted flags
def flag_filter(compiler, flags):
    result = []
    for flag in flags:
        if has_flag(compiler, flag):
            result.append(flag)
    return result


class CustomBuildExt(build_ext):
    def build_extensions(self):
        c = self.compiler.compiler_type
        print('Compiler:', c)
        opts = flag_filter(self.compiler, c_opt.get(c, []))
        linker_opts = flag_filter(self.compiler, l_opt.get(c, []))
        for e in self.extensions:
            e.extra_compile_args = opts
            e.extra_link_args = linker_opts
        build_ext.build_extensions(self)


setup(
    name=package_name,
    version=version_string,

    description='BD Mesh generator',
    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://github.com/bond-anton/BDMesh',

    author='Anton Bondarenko',
    author_email='bond.anton@gmail.com',

    license='Apache Software License',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],

    keywords='Mesh',

    packages=find_packages(exclude=['demo', 'tests', 'docs', 'contrib', 'build', 'dist', 'venv']),
    ext_modules=cythonize(extensions, compiler_directives={'language_level': sys.version_info[0]}),
    package_data={'BDMesh': ['Mesh1D.pxd', 'Mesh1DUniform.pxd',
                             'TreeMesh1D.pxd', 'TreeMesh1DUniform.pxd']},

    tests_require=['numpy'],

    cmdclass={'build_ext': CustomBuildExt},
)
