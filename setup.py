from setuptools import setup, find_packages
import codecs
import os.path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

setup(
    name = 'javiface',
    version = get_version("javiface/__init__.py"),
    description = 'Efficient and accurate image-based head pose estimation',
    long_description = "".join(open("README.md", "r").readlines()),
    long_description_content_type = "text/markdown",
    url = 'https://github.com/thohemp/6DRepNet',
    author = 'Javier Javier Daza',
    author_email = 'javierjdaza@gmail.com',
    license = 'MIT',
    packages = find_packages(),
    install_requires = open('requirements.txt').readlines(),
    classifiers = [
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',        
    ],
)
