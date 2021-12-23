# Experiments postprocessing data


## New System Setup

### Get a recent-ish python

Either use `pyenv`:

```
pyenv install 3.8.11
pyenv shell 3.8.11
```

or `source load-env.sh` to get a recent-ish python

### Install Poetry

Use a python from pyenv to install `poetry`


configure poetry to use a local venv
```
poetry config virtualenvs.in-project true
```

set the local python version
```
pyenv local 3.8.11
```

### Use Poetry to run

Install the project dependencies in a venv
```
poetry install
```

Run the script
```
poetry run python postprocess.py
```


## Other notes

This folder was initialized with
```
poetry init
```

MacOS: I had an SSL error, which seems to be related to python not using the macos SSL certificates when installed with pyenv
eventually fixed it with

```
python
> print(ssl.get_default_verify_paths())
DefaultVerifyPaths(cafile='/usr/local/etc/openssl@1.1/cert.pem', capath='/usr/local/etc/openssl@1.1/certs', openssl_cafile_env='SSL_CERT_FILE', openssl_cafile='/usr/local/etc/openssl@1.1/cert.pem', openssl_capath_env='SSL_CERT_DIR', openssl_capath='/usr/local/etc/openssl@1.1/certs')
```

then `export REQUESTS_CA_BUNDLE=/usr/local/etc/openssl@1.1/cert.pem` before using poetry to install anything

then
```
poetry shell
```

to enter the shell and run the script