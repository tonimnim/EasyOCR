"""
The OCR engine behind the Tonits eFootball screenshot reader (tonits_vision): EasyOCR's CRAFT detector
and generation-2 recognizers, trimmed to English.
"""
from io import open
from setuptools import setup

with open('requirements.txt', encoding="utf-8-sig") as f:
    requirements = f.readlines()

def readme():
    with open('README.md', encoding="utf-8-sig") as f:
        README = f.read()
    return README

setup(
    name='easyocr',
    packages=['easyocr'],
    include_package_data=True,
    version='1.7.2',
    install_requires=requirements,
    entry_points={"console_scripts": ["easyocr= easyocr.cli:main"]},
    license='Apache License 2.0',
    description='EasyOCR engine trimmed for the Tonits eFootball screenshot reader',
    long_description=readme(),
    long_description_content_type="text/markdown",
    author='Rakpong Kittinaradorn (EasyOCR)',
    author_email='r.kittinaradorn@gmail.com',
    maintainer='tonimnim',
    url='https://github.com/tonimnim/EasyOCR',
    download_url='https://github.com/tonimnim/EasyOCR.git',
    keywords=['ocr', 'efootball', 'screenshot', 'tonits'],
    classifiers=[
        'Development Status :: 5 - Production/Stable'
    ],

)
