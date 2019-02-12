#!/bin/bash
pip install -r requirements.txt
git clone https://github.com/allenai/allennlp.git
cd allennlp
INSTALL_TEST_REQUIREMENTS=true scripts/install_requirements.sh
pip install --editable .
cd ..
python -c "import nltk; nltk.download('punkt')"
git clone https://www.github.com/nvidia/apex.git
cd apex
python setup.py install
cd ..
