## Adding MNIST to .travis.yml  


For this homework, your are required to download the ```MNIST dataset``` in order to pass all the tests locally.

However, since you should not push the ```MNIST dataset``` to your git repo, the tests on ```travis``` will never pass.

Here is a small tutorial on how to add ```MNIST dataset``` to your ```travis``` to get the green checkmark for your repo:)


You can do it by simply adding the following commands in your ```.travis.yml``` file after the line ```pip install -r requirements.txt```:

```
- cd data
- wget -c http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
- wget -c http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
- wget -c http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
- wget -c http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
- gzip -d train-images-idx3-ubyte.gz
- gzip -d train-labels-idx1-ubyte.gz
- gzip -d t10k-images-idx3-ubyte.gz
- gzip -d t10k-labels-idx1-ubyte.gz
- cd ..
```

These commands enter your data directory, download the MNIST dataset, and unzip the files. It might take longer than usual for it to build since downloading MNIST could be slow.

</br>
</br>

For your convenience, here is what my entire ```.travis.yml``` looks like for this homework:
```
# what language the build will be configured for
language: python

# specify what versions of python will be used
python:
    - 3.8

# what branches should be evaluated
branches:
    only:
        - master

# list of commands to run to setup the environment
install:
      - sudo apt-get update
      # We do this conditionally because it saves us some downloading if the
      # version is the same.
      - if [[ "$TRAVIS_PYTHON_VERSION" == "2.7" ]]; then
          wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh;
        else
          wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
        fi
      - bash miniconda.sh -b -p $HOME/miniconda
      - source "$HOME/miniconda/etc/profile.d/conda.sh"
      - hash -r
      - conda config --set always_yes yes --set changeps1 no
      - conda update -q conda
      # Useful for debugging any issues with conda
      - conda info -a

      # Enlist your dependencies
      - conda create -q -n test-environment python=$TRAVIS_PYTHON_VERSION 
      - conda activate test-environment
      - pip install -r requirements.txt
      - cd data
      - wget -c http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
      - wget -c http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
      - wget -c http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
      - wget -c http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
      - gzip -d train-images-idx3-ubyte.gz
      - gzip -d train-labels-idx1-ubyte.gz
      - gzip -d t10k-images-idx3-ubyte.gz
      - gzip -d t10k-labels-idx1-ubyte.gz
      - cd ..

# the actual commands to run
script:
    - python -m pytest -s
```

